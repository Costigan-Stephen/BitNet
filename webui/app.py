from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union
from urllib import error as urlerror
from urllib import request as urlrequest

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import RagStore


ROOT_DIR = Path(__file__).resolve().parents[1]
WEBUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBUI_DIR / "static"
DATA_DIR = WEBUI_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TRAIN_DATASETS_DIR = DATA_DIR / "train_datasets"
RAG_DB_PATH = DATA_DIR / "rag_index.json"
LOGGER = logging.getLogger("bitnet.webui")
RAG_UPLOAD_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
    ".docx",
}
RAG_REFUSAL_PATTERNS = [
    "i can't directly",
    "i cannot directly",
    "unable to directly access",
    "can't access files",
    "cannot access files",
    "can't directly access",
    "cannot directly access",
    "can't view",
    "cannot view",
    "can't analyze files",
    "cannot analyze files",
    "can't read files",
    "cannot read files",
]
RAG_EXTRACTIVE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "said",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def _creation_flags() -> int:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return int(subprocess.CREATE_NO_WINDOW)
    return 0


class ProcessState:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.started_at: Optional[float] = None
        self.command: list[str] = []
        self.logs: deque[str] = deque(maxlen=1200)
        self.last_error: Optional[str] = None
        self.lock = threading.RLock()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line.rstrip())

    def tail(self, limit: int = 200) -> list[str]:
        with self.lock:
            if limit <= 0:
                return []
            return list(self.logs)[-limit:]


class LlamaServerManager:
    def __init__(self) -> None:
        self.state = ProcessState()
        self.host = "127.0.0.1"
        self.port = 8080
        self.model_path: Optional[str] = None
        self.extra: dict[str, Any] = {}

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _resolve_binary(self) -> Path:
        env_override = os.getenv("BITNET_LLAMA_SERVER_BIN")
        if env_override:
            candidate = Path(env_override).expanduser().resolve()
            if candidate.exists():
                return candidate

        candidates = [
            ROOT_DIR / "build" / "bin" / "llama-server.exe",
            ROOT_DIR / "build" / "bin" / "Release" / "llama-server.exe",
            ROOT_DIR / "build" / "bin" / "llama-server",
            ROOT_DIR / "build" / "bin" / "Release" / "llama-server",
            ROOT_DIR / "3rdparty" / "llama.cpp" / "build" / "bin" / "llama-server.exe",
            ROOT_DIR / "3rdparty" / "llama.cpp" / "build" / "bin" / "llama-server",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        from_path = shutil.which("llama-server")
        if from_path:
            return Path(from_path).resolve()
        from_path_exe = shutil.which("llama-server.exe")
        if from_path_exe:
            return Path(from_path_exe).resolve()

        raise FileNotFoundError(
            "Could not find llama-server binary. Build BitNet first (setup_env.py) "
            "or set BITNET_LLAMA_SERVER_BIN."
        )

    def _drain_pipe(self, pipe: Any, stream_name: str) -> None:
        try:
            for raw_line in iter(pipe.readline, ""):
                if not raw_line:
                    break
                self.state.append_log(f"[{stream_name}] {raw_line.rstrip()}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _wait_for_health(self, timeout_sec: float = 45.0) -> None:
        deadline = time.time() + timeout_sec
        health_url = f"{self.base_url}/health"
        while time.time() < deadline:
            if self.state.process is not None and self.state.process.poll() is not None:
                raise RuntimeError("llama-server exited before becoming healthy.")
            try:
                with urlrequest.urlopen(health_url, timeout=1.0) as response:
                    if response.status == 200:
                        return
            except (urlerror.URLError, TimeoutError):
                pass
            time.sleep(0.35)
        raise TimeoutError(f"Timed out waiting for {health_url}")

    def start(
        self,
        model_path: str,
        host: str,
        port: int,
        ctx_size: int,
        threads: int,
        n_predict: int,
        lora_paths: list[str],
        extra_args: list[str],
    ) -> dict[str, Any]:
        with self.state.lock:
            if self.state.is_running():
                raise RuntimeError("llama-server is already running.")

            binary = self._resolve_binary()
            model = Path(model_path).expanduser().resolve()
            if not model.exists():
                raise FileNotFoundError(f"Model not found: {model}")

            command = [
                str(binary),
                "-m",
                str(model),
                "-c",
                str(ctx_size),
                "-t",
                str(threads),
                "-n",
                str(n_predict),
                "--host",
                host,
                "--port",
                str(port),
                "-ngl",
                "0",
                "-cb",
            ]

            for lora in lora_paths:
                lora_path = Path(lora).expanduser().resolve()
                if not lora_path.exists():
                    raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")
                command.extend(["--lora", str(lora_path)])

            command.extend([arg for arg in extra_args if arg.strip()])

            self.host = host
            self.port = port
            self.model_path = str(model)
            self.extra = {
                "ctx_size": ctx_size,
                "threads": threads,
                "n_predict": n_predict,
                "lora_paths": lora_paths,
                "extra_args": extra_args,
            }
            self.state.command = command
            self.state.logs.clear()
            self.state.last_error = None

            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_creation_flags(),
            )
            self.state.process = process
            self.state.started_at = time.time()

            stdout_thread = threading.Thread(
                target=self._drain_pipe, args=(process.stdout, "stdout"), daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._drain_pipe, args=(process.stderr, "stderr"), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()

        try:
            self._wait_for_health()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.stop()
            raise
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self.state.lock:
            process = self.state.process
            if process is None:
                return self.status()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            self.state.process = None
            self.state.started_at = None
            return self.status()

    def status(self) -> dict[str, Any]:
        process = self.state.process
        running = process is not None and process.poll() is None
        return {
            "running": running,
            "pid": process.pid if running else None,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "model_path": self.model_path,
            "started_at": self.state.started_at,
            "last_error": self.state.last_error,
            "command": self.state.command,
            "extra": self.extra,
        }


class TrainingManager:
    def __init__(self) -> None:
        self.state = ProcessState()
        self.job_config: dict[str, Any] = {}
        self.exit_code: Optional[int] = None
        self.status_value = "idle"

    def _drain_pipe(self, pipe: Any, stream_name: str) -> None:
        try:
            for raw_line in iter(pipe.readline, ""):
                if not raw_line:
                    break
                self.state.append_log(f"[{stream_name}] {raw_line.rstrip()}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def start(self, command: list[str], job_config: dict[str, Any]) -> dict[str, Any]:
        with self.state.lock:
            if self.state.is_running():
                raise RuntimeError("A training job is already running.")

            self.state.command = command
            self.job_config = job_config
            self.state.logs.clear()
            self.state.last_error = None
            self.exit_code = None
            self.status_value = "running"

            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_creation_flags(),
            )
            self.state.process = process
            self.state.started_at = time.time()

            threading.Thread(target=self._drain_pipe, args=(process.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=self._drain_pipe, args=(process.stderr, "stderr"), daemon=True).start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self.state.lock:
            process = self.state.process
            if process is None:
                return self.status()
            if process.poll() is None:
                self.status_value = "stopping"
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            self.exit_code = process.returncode
            self.status_value = "stopped"
            self.state.process = None
            return self.status()

    def status(self) -> dict[str, Any]:
        with self.state.lock:
            process = self.state.process
            if process is not None and process.poll() is not None:
                self.exit_code = process.returncode
                self.status_value = "completed" if process.returncode == 0 else "failed"
                self.state.process = None

            running = self.state.is_running()
            return {
                "running": running,
                "status": "running" if running else self.status_value,
                "pid": process.pid if running and process else None,
                "started_at": self.state.started_at,
                "command": self.state.command,
                "exit_code": self.exit_code,
                "last_error": self.state.last_error,
                "job_config": self.job_config,
            }


class ChatMessage(BaseModel):
    role: str
    content: str


class ServerStartRequest(BaseModel):
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 2048
    threads: int = max(2, (os.cpu_count() or 4) // 2)
    n_predict: int = 4096
    lora_paths: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system_prompt: str = ""
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 512
    repeat_penalty: float = 1.05
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: list[str] = Field(default_factory=list)
    rag_enabled: bool = False
    rag_top_k: int = 4


class RagIndexRequest(BaseModel):
    paths: list[str]
    chunk_size: int = 220
    chunk_overlap: int = 40
    reset: bool = False


class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 4


class TrainStartRequest(BaseModel):
    command: Optional[list[str]] = None
    base_model: Optional[str] = None
    dataset_path: Optional[str] = None
    output_dir: str = "training-output/adapter"
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    lora_rank: int = 16
    lora_alpha: int = 32


class LoraApplyRequest(BaseModel):
    adapters: list[dict[str, Any]]


app = FastAPI(title="BitNet Web UI", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

llama_server = LlamaServerManager()
rag_store = RagStore(RAG_DB_PATH)
training = TrainingManager()


def _ensure_server_running() -> None:
    if not llama_server.status()["running"]:
        raise HTTPException(status_code=503, detail="llama-server is not running.")


def _pick_default_model() -> Optional[Path]:
    env_model = os.getenv("BITNET_DEFAULT_MODEL", "").strip()
    if env_model:
        candidate = Path(env_model).expanduser().resolve()
        if candidate.exists():
            return candidate

    preferred = ROOT_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
    if preferred.exists():
        return preferred

    models_root = ROOT_DIR / "models"
    if not models_root.exists():
        return None

    for candidate in sorted(models_root.rglob("*.gguf")):
        if candidate.is_file():
            return candidate
    return None


def _sanitize_upload_name(filename: str) -> str:
    raw_name = Path(filename or "").name.strip()
    sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw_name).strip("._")
    if not sanitized:
        sanitized = f"upload_{int(time.time() * 1000)}.txt"
    if len(sanitized) > 180:
        stem = Path(sanitized).stem[:160]
        suffix = Path(sanitized).suffix[:20]
        sanitized = f"{stem}{suffix}"
    return sanitized


def _unique_upload_path(filename: str) -> Path:
    return _unique_path_in_dir(UPLOADS_DIR, filename)


def _unique_path_in_dir(base_dir: Path, filename: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / filename
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


async def _upstream_request(
    method: str,
    path: str,
    body: Optional[Union[dict[str, Any], list[Any]]] = None,
) -> Any:
    _ensure_server_running()
    url = f"{llama_server.base_url}{path}"
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.request(method, url, json=body)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"raw": response.text}


async def _upstream_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["stream"] = False
    try:
        response = await _upstream_request("POST", "/v1/chat/completions", request_payload)
        if isinstance(response, dict):
            return response
    except HTTPException:
        pass

    # Some local llama-server builds are unstable on non-stream responses.
    # Fall back to streaming and reconstruct a completion payload.
    _ensure_server_running()
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    url = f"{llama_server.base_url}/v1/chat/completions"
    full_text = ""
    usage: Optional[dict[str, Any]] = None
    finish_reason = "stop"

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            async with client.stream("POST", url, json=stream_payload) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="replace")
                    raise HTTPException(status_code=response.status_code, detail=detail)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload_piece = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (payload_piece.get("choices") or [{}])[0]
                    delta = (choice.get("delta") or {}).get("content", "")
                    if delta:
                        full_text += str(delta)
                    if choice.get("finish_reason"):
                        finish_reason = str(choice.get("finish_reason"))
                    if payload_piece.get("usage") is not None:
                        usage = payload_piece.get("usage")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    rebuilt: dict[str, Any] = {
        "id": "bitnet-upstream-stream-collected",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        rebuilt["usage"] = usage
    return rebuilt


def _build_chat_payload(req: ChatRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []
    retrieval_chunks: list[dict[str, Any]] = []

    query = req.messages[-1].content if req.messages else ""
    should_use_rag = bool(req.rag_enabled)
    if not should_use_rag and query:
        if _query_mentions_document(query):
            should_use_rag = rag_store.status().get("total_chunks", 0) > 0

    if should_use_rag and query:
        _ensure_rag_index_ready()
        retrieval_chunks = rag_store.search(query, top_k=max(1, req.rag_top_k))

    system_parts: list[str] = []
    if req.system_prompt.strip():
        system_parts.append(req.system_prompt.strip())

    if retrieval_chunks:
        context_lines = [
            "You are given retrieved snippets from local indexed files for this request.",
            "Do not claim you cannot access files or documents when snippets are provided here.",
            "Answer using the snippets when relevant and cite source file paths in square brackets.",
            "If the snippets do not contain the requested detail, clearly state what is missing.",
        ]
        for idx, chunk in enumerate(retrieval_chunks, start=1):
            context_lines.append(f"--- BEGIN RETRIEVED CHUNK {idx} | SOURCE: {chunk['source']} ---")
            context_lines.append(_compact_retrieval_text(str(chunk.get("text", ""))))
            context_lines.append(f"--- END RETRIEVED CHUNK {idx} ---")
        system_parts.append("\n".join(context_lines))

    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for msg in req.messages:
        messages.append({"role": msg.role, "content": msg.content})

    payload: dict[str, Any] = {
        "model": "bitnet-local",
        "messages": messages,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "repeat_penalty": req.repeat_penalty,
        "frequency_penalty": req.frequency_penalty,
        "presence_penalty": req.presence_penalty,
        "stream": False,
    }
    if req.stop:
        payload["stop"] = req.stop
    if req.seed is not None:
        payload["seed"] = req.seed
    return payload, retrieval_chunks


def _extract_assistant_text(response_payload: dict[str, Any]) -> str:
    try:
        return str(response_payload["choices"][0]["message"]["content"] or "")
    except Exception:
        return ""


def _query_mentions_document(query: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    if not normalized:
        return False
    keywords = ("attach", "file", "document", "doc", "pdf", "scan", "uploaded", "rag", "index")
    return any(word in normalized for word in keywords)


def _compact_retrieval_text(text: str, max_chars: int = 480) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _ensure_rag_index_ready() -> None:
    status = rag_store.status()
    if status.get("total_chunks", 0) > 0:
        return
    if not UPLOADS_DIR.exists():
        return
    try:
        rag_store.index_paths([str(UPLOADS_DIR)], chunk_size=220, chunk_overlap=40, reset=False)
    except Exception as exc:
        LOGGER.warning("RAG auto-index skipped due to error: %s", exc)


def _looks_like_rag_refusal(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return False
    if any(pattern in normalized for pattern in RAG_REFUSAL_PATTERNS):
        return True

    refusal_regexes = [
        r"\b(?:i\s*(?:am|'m)?\s*sorry[, ]*)?(?:but\s*)?(?:i\s+)?(?:can't|cannot|can not|unable to|do not|don't)\b.{0,90}\b(access|open|view|read|retrieve|analy[sz]e)\b.{0,90}\b(file|files|document|documents|attachment|attached|image|paper|scan)\b",
        r"\b(file|files|document|documents|attachment|attached|image|paper|scan)\b.{0,90}\b(?:can't|cannot|can not|unable to|do not|don't)\b.{0,90}\b(access|open|view|read|retrieve|analy[sz]e)\b",
        r"\bif you (?:can|could|would)?\s*provide\b.{0,120}\b(text|quote|content|details|main points)\b",
    ]
    return any(re.search(pattern, normalized) is not None for pattern in refusal_regexes)


def _build_rag_retry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    retry_payload = dict(payload)
    retry_payload["messages"] = list(payload.get("messages", []))
    retry_instruction = {
        "role": "system",
        "content": (
            "CRITICAL: Retrieved chunks from local files are included in this conversation. "
            "Do not say you cannot access files/documents. "
            "Answer using the retrieved chunks and cite sources in square brackets."
        ),
    }
    retry_payload["messages"].insert(0, retry_instruction)
    return retry_payload


def _build_forced_rag_fallback_response(retrieval_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not retrieval_chunks:
        content = "No retrieved chunks were available to ground the answer."
    else:
        lines = ["I found relevant content in the indexed files:"]
        for idx, chunk in enumerate(retrieval_chunks[:3], start=1):
            source = chunk.get("source", "unknown-source")
            excerpt = str(chunk.get("text", "")).strip().replace("\r", " ").replace("\n", " ")
            excerpt = re.sub(r"\s+", " ", excerpt)[:500]
            lines.append(f"{idx}. [{source}] {excerpt}")
        lines.append("Ask a more specific question and I can extract exact points from these passages.")
        content = "\n".join(lines)

    return {
        "id": "bitnet-rag-fallback",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _extract_latest_user_query(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _tokenize_for_extractive(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _split_sentences(text: str) -> list[str]:
    rough = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [item.strip() for item in rough if item and item.strip()]
    return sentences


def _build_extractive_rag_response(query: str, retrieval_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    query_tokens = [tok for tok in _tokenize_for_extractive(query) if tok not in RAG_EXTRACTIVE_STOPWORDS]
    ranked: list[tuple[int, str, str]] = []

    for chunk in retrieval_chunks[:5]:
        source = str(chunk.get("source", "unknown-source"))
        text = str(chunk.get("text", ""))
        for sentence in _split_sentences(text):
            sentence_tokens = set(_tokenize_for_extractive(sentence))
            if not sentence_tokens:
                continue
            score = sum(1 for tok in query_tokens if tok in sentence_tokens)
            ranked.append((score, sentence, source))

    if ranked:
        ranked.sort(key=lambda row: row[0], reverse=True)
        top = ranked[:3]
        lines = ["From the indexed document(s):"]
        for _, sentence, source in top:
            lines.append(f"- {sentence} [{source}]")
        content = "\n".join(lines)
    else:
        content = _build_forced_rag_fallback_response(retrieval_chunks)["choices"][0]["message"]["content"]

    return {
        "id": "bitnet-rag-extractive",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def _chat_with_rag_retry(
    payload: dict[str, Any],
    retrieval_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    query = _extract_latest_user_query(payload)
    if retrieval_chunks and _query_mentions_document(query):
        return _build_extractive_rag_response(query, retrieval_chunks)

    response = await _upstream_chat_completion(payload)
    content = _extract_assistant_text(response)
    if retrieval_chunks and _looks_like_rag_refusal(content):
        retry_payload = _build_rag_retry_payload(payload)
        response = await _upstream_chat_completion(retry_payload)
        content = _extract_assistant_text(response)
        if _looks_like_rag_refusal(content):
            response = _build_extractive_rag_response(query, retrieval_chunks)
    return response


def _default_train_command(req: TrainStartRequest) -> list[str]:
    if not req.base_model or not req.dataset_path:
        raise HTTPException(
            status_code=400,
            detail="base_model and dataset_path are required when command is not provided.",
        )
    dataset = Path(req.dataset_path).expanduser().resolve()
    if not dataset.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset}")

    output_dir = Path(req.output_dir).expanduser().resolve()
    return [
        sys.executable,
        str(WEBUI_DIR / "train_adapter.py"),
        "--base-model",
        req.base_model,
        "--dataset-path",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(req.epochs),
        "--batch-size",
        str(req.batch_size),
        "--grad-accum-steps",
        str(req.grad_accum_steps),
        "--learning-rate",
        str(req.learning_rate),
        "--max-seq-len",
        str(req.max_seq_len),
        "--lora-rank",
        str(req.lora_rank),
        "--lora-alpha",
        str(req.lora_alpha),
    ]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> JSONResponse:
    llama_status = llama_server.status()
    llama_healthy = False
    if llama_status["running"]:
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.get(f"{llama_server.base_url}/health")
                llama_healthy = response.status_code == 200
        except httpx.HTTPError:
            llama_healthy = False
    return JSONResponse(
        {
            "webui": "ok",
            "llama_running": llama_status["running"],
            "llama_healthy": llama_healthy,
            "base_url": llama_server.base_url,
        }
    )


@app.get("/api/server/status")
async def server_status() -> JSONResponse:
    return JSONResponse(llama_server.status())


@app.get("/api/server/logs")
async def server_logs(limit: int = 200) -> JSONResponse:
    return JSONResponse({"logs": llama_server.state.tail(limit)})


@app.post("/api/server/start")
async def server_start(req: ServerStartRequest) -> JSONResponse:
    try:
        status = await asyncio.to_thread(
            llama_server.start,
            req.model_path,
            req.host,
            req.port,
            req.ctx_size,
            req.threads,
            req.n_predict,
            req.lora_paths,
            req.extra_args,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(status)


@app.post("/api/server/stop")
async def server_stop() -> JSONResponse:
    status = await asyncio.to_thread(llama_server.stop)
    return JSONResponse(status)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    payload, retrieval_chunks = _build_chat_payload(req)
    response = await _chat_with_rag_retry(payload, retrieval_chunks)
    if retrieval_chunks and _looks_like_rag_refusal(_extract_assistant_text(response)):
        response = _build_forced_rag_fallback_response(retrieval_chunks)
    return JSONResponse({"retrieval": retrieval_chunks, "response": response})


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    payload, retrieval_chunks = _build_chat_payload(req)
    _ensure_server_running()

    async def event_stream() -> Any:
        if retrieval_chunks:
            data = json.dumps({"retrieval": retrieval_chunks}, ensure_ascii=True)
            yield f"event: retrieval\ndata: {data}\n\n"

        # For RAG requests we run a guarded non-stream call so we can auto-retry
        # refusal-style answers and return grounded content reliably.
        if retrieval_chunks:
            try:
                response = await _chat_with_rag_retry(payload, retrieval_chunks)
                content = _extract_assistant_text(response)
                if _looks_like_rag_refusal(content):
                    response = _build_forced_rag_fallback_response(retrieval_chunks)
                    content = _extract_assistant_text(response)
                chunk_payload = {
                    "id": "bitnet-rag-stream",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                }
                usage = response.get("usage")
                if usage is not None:
                    chunk_payload["usage"] = usage
                yield f"data: {json.dumps(chunk_payload, ensure_ascii=True)}\n\n"
                yield "data: [DONE]\n\n"
            except HTTPException as exc:
                error_data = json.dumps({"error": str(exc.detail or exc)}, ensure_ascii=True)
                yield f"event: error\ndata: {error_data}\n\n"
            return

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        url = f"{llama_server.base_url}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, json=stream_payload) as response:
                    if response.status_code >= 400:
                        details = await response.aread()
                        error_data = json.dumps({"error": details.decode("utf-8", errors="replace")}, ensure_ascii=True)
                        yield f"event: error\ndata: {error_data}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if line:
                            yield f"{line}\n"
                        else:
                            yield "\n"
            except httpx.HTTPError as exc:
                error_data = json.dumps({"error": str(exc)}, ensure_ascii=True)
                yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/rag/status")
async def rag_status() -> JSONResponse:
    return JSONResponse(rag_store.status())


@app.post("/api/rag/index")
async def rag_index(req: RagIndexRequest) -> JSONResponse:
    if not req.paths:
        raise HTTPException(status_code=400, detail="No paths provided.")
    result = await asyncio.to_thread(
        rag_store.index_paths,
        req.paths,
        req.chunk_size,
        req.chunk_overlap,
        req.reset,
    )
    return JSONResponse(result)


@app.post("/api/rag/upload")
async def rag_upload(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(220),
    chunk_overlap: int = Form(40),
    reset: bool = Form(False),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    saved_paths: list[str] = []
    skipped_files: list[dict[str, str]] = []

    for upload in files:
        original_name = upload.filename or ""
        suffix = Path(original_name).suffix.lower()
        if suffix not in RAG_UPLOAD_EXTENSIONS:
            skipped_files.append(
                {
                    "file": original_name or "unnamed",
                    "reason": f"unsupported extension '{suffix or '(none)'}'",
                }
            )
            await upload.close()
            continue
        target_path = _unique_upload_path(_sanitize_upload_name(original_name))
        try:
            data = await upload.read()
            if not data:
                skipped_files.append({"file": original_name or target_path.name, "reason": "empty file"})
                continue
            target_path.write_bytes(data)
            saved_paths.append(str(target_path))
        except Exception as exc:
            skipped_files.append({"file": original_name or target_path.name, "reason": str(exc)})
        finally:
            await upload.close()

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No valid files uploaded.",
                "skipped_files": skipped_files,
            },
        )

    index_result = await asyncio.to_thread(
        rag_store.index_paths,
        saved_paths,
        chunk_size,
        chunk_overlap,
        reset,
    )
    return JSONResponse(
        {
            "saved_files": saved_paths,
            "saved_count": len(saved_paths),
            "skipped_files": skipped_files,
            "indexed": index_result,
        }
    )


@app.post("/api/rag/search")
async def rag_search(req: RagSearchRequest) -> JSONResponse:
    result = await asyncio.to_thread(rag_store.search, req.query, req.top_k)
    return JSONResponse({"results": result})


@app.post("/api/rag/reset")
async def rag_reset() -> JSONResponse:
    result = await asyncio.to_thread(rag_store.clear)
    return JSONResponse(result)


@app.get("/api/train/status")
async def train_status() -> JSONResponse:
    return JSONResponse(training.status())


@app.get("/api/train/logs")
async def train_logs(limit: int = 200) -> JSONResponse:
    return JSONResponse({"logs": training.state.tail(limit)})


@app.get("/api/train/template")
async def train_template() -> JSONResponse:
    template = {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "dataset_path": str((ROOT_DIR / "data" / "train.jsonl").resolve()),
        "output_dir": str((ROOT_DIR / "training-output" / "adapter").resolve()),
    }
    return JSONResponse(
        {
            "default_command_example": [
                sys.executable,
                str((WEBUI_DIR / "train_adapter.py").resolve()),
                "--base-model",
                template["base_model"],
                "--dataset-path",
                template["dataset_path"],
                "--output-dir",
                template["output_dir"],
            ],
            "notes": [
                "This is LoRA fine-tuning for Hugging Face models.",
                "Use train_adapter.py output with llama.cpp conversion scripts if you need GGUF LoRA adapters.",
            ],
        }
    )


@app.post("/api/train/upload-dataset")
async def train_upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    filename = _sanitize_upload_name(file.filename or "dataset.jsonl")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".json", ".jsonl", ".csv"}:
        raise HTTPException(status_code=400, detail="Unsupported dataset format. Use JSON, JSONL, or CSV.")

    target_path = _unique_path_in_dir(TRAIN_DATASETS_DIR, filename)
    try:
        payload = await file.read()
    finally:
        await file.close()

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    target_path.write_bytes(payload)
    return JSONResponse(
        {
            "dataset_path": str(target_path),
            "filename": target_path.name,
            "size_bytes": len(payload),
        }
    )


@app.post("/api/train/raw-dataset")
async def train_raw_dataset(
    raw_text: str = Form(...),
    format: str = Form("lines"),
    filename: str = Form("raw_dataset"),
) -> JSONResponse:
    text = raw_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Raw dataset text is empty.")

    mode = (format or "lines").strip().lower()
    output_lines: list[str] = []

    if mode == "lines":
        for line in raw_text.splitlines():
            row = line.strip()
            if not row:
                continue
            output_lines.append(json.dumps({"text": row}, ensure_ascii=False))
    elif mode == "jsonl":
        for idx, line in enumerate(raw_text.splitlines(), start=1):
            row = line.strip()
            if not row:
                continue
            try:
                parsed = json.loads(row)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSONL on line {idx}: {exc.msg}") from exc
            if not isinstance(parsed, dict):
                raise HTTPException(status_code=400, detail=f"JSONL line {idx} must be a JSON object.")
            output_lines.append(json.dumps(parsed, ensure_ascii=False))
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'lines' or 'jsonl'.")

    if not output_lines:
        raise HTTPException(status_code=400, detail="No valid training rows found in raw text.")

    safe_name = _sanitize_upload_name(filename)
    stem = Path(safe_name).stem or "raw_dataset"
    target_path = _unique_path_in_dir(TRAIN_DATASETS_DIR, f"{stem}.jsonl")
    target_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return JSONResponse(
        {
            "dataset_path": str(target_path),
            "filename": target_path.name,
            "format": mode,
            "rows": len(output_lines),
        }
    )


@app.post("/api/train/start")
async def train_start(req: TrainStartRequest) -> JSONResponse:
    command = req.command if req.command else _default_train_command(req)
    try:
        status = await asyncio.to_thread(training.start, command, req.dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(status)


@app.post("/api/train/stop")
async def train_stop() -> JSONResponse:
    status = await asyncio.to_thread(training.stop)
    return JSONResponse(status)


@app.get("/api/lora/list")
async def lora_list() -> JSONResponse:
    response = await _upstream_request("GET", "/lora-adapters")
    return JSONResponse(response)


@app.post("/api/lora/apply")
async def lora_apply(req: LoraApplyRequest) -> JSONResponse:
    response = await _upstream_request("POST", "/lora-adapters", req.adapters)
    return JSONResponse(response)


@app.on_event("startup")
async def startup() -> None:
    auto_start_value = os.getenv("BITNET_WEBUI_AUTO_START", "1").strip().lower()
    auto_start = auto_start_value not in {"0", "false", "no", "off"}
    if not auto_start:
        LOGGER.info("BITNET_WEBUI_AUTO_START disabled; not auto-starting llama-server.")
        return

    if llama_server.status()["running"]:
        return

    model = _pick_default_model()
    if model is None:
        message = (
            "Auto-start skipped: no GGUF model found. Set BITNET_DEFAULT_MODEL or place a model in ./models."
        )
        llama_server.state.last_error = message
        LOGGER.warning(message)
        return

    host = os.getenv("BITNET_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.getenv("BITNET_SERVER_PORT", "8080"))
        ctx_size = int(os.getenv("BITNET_SERVER_CTX", "2048"))
        threads = int(os.getenv("BITNET_SERVER_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
        n_predict = int(os.getenv("BITNET_SERVER_N_PREDICT", "4096"))
    except ValueError as exc:
        llama_server.state.last_error = f"Invalid BITNET_SERVER_* env value: {exc}"
        LOGGER.warning(llama_server.state.last_error)
        return

    lora_raw = os.getenv("BITNET_SERVER_LORAS", "").strip()
    lora_paths = [item.strip() for item in lora_raw.replace(";", ",").split(",") if item.strip()]

    extra_args_raw = os.getenv("BITNET_SERVER_EXTRA_ARGS", "").strip()
    extra_args = shlex.split(extra_args_raw) if extra_args_raw else []

    try:
        await asyncio.to_thread(
            llama_server.start,
            str(model),
            host,
            port,
            ctx_size,
            threads,
            n_predict,
            lora_paths,
            extra_args,
        )
        LOGGER.info("Auto-started llama-server on %s:%s with model %s", host, port, model)
    except Exception as exc:
        message = f"Auto-start failed: {exc}"
        llama_server.state.last_error = message
        LOGGER.warning(message)


@app.on_event("shutdown")
def shutdown() -> None:
    if llama_server.status()["running"]:
        llama_server.stop()
    if training.status()["running"]:
        training.stop()

    if os.name != "nt":
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
