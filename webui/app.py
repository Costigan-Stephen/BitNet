from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import RagStore


ROOT_DIR = Path(__file__).resolve().parents[1]
WEBUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBUI_DIR / "static"
DATA_DIR = WEBUI_DIR / "data"
RAG_DB_PATH = DATA_DIR / "rag_index.json"


def _creation_flags() -> int:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return int(subprocess.CREATE_NO_WINDOW)
    return 0


class ProcessState:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.started_at: float | None = None
        self.command: list[str] = []
        self.logs: deque[str] = deque(maxlen=1200)
        self.last_error: str | None = None
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
        self.model_path: str | None = None
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
        self.exit_code: int | None = None
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
    seed: int | None = None
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
    command: list[str] | None = None
    base_model: str | None = None
    dataset_path: str | None = None
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


async def _upstream_request(method: str, path: str, body: dict[str, Any] | list[Any] | None = None) -> Any:
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


def _build_chat_payload(req: ChatRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []
    retrieval_chunks: list[dict[str, Any]] = []

    if req.rag_enabled and req.messages:
        query = req.messages[-1].content
        retrieval_chunks = rag_store.search(query, top_k=max(1, req.rag_top_k))

    system_parts: list[str] = []
    if req.system_prompt.strip():
        system_parts.append(req.system_prompt.strip())

    if retrieval_chunks:
        context_lines = [
            "Use the retrieved context below when it is relevant. Cite source file paths in your answer when possible."
        ]
        for idx, chunk in enumerate(retrieval_chunks, start=1):
            context_lines.append(f"[{idx}] {chunk['source']}")
            context_lines.append(chunk["text"])
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
    response = await _upstream_request("POST", "/v1/chat/completions", payload)
    return JSONResponse({"retrieval": retrieval_chunks, "response": response})


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    payload, retrieval_chunks = _build_chat_payload(req)
    payload["stream"] = True
    _ensure_server_running()

    async def event_stream() -> Any:
        if retrieval_chunks:
            data = json.dumps({"retrieval": retrieval_chunks}, ensure_ascii=True)
            yield f"event: retrieval\ndata: {data}\n\n"

        url = f"{llama_server.base_url}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
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


@app.on_event("shutdown")
def shutdown() -> None:
    if llama_server.status()["running"]:
        llama_server.stop()
    if training.status()["running"]:
        training.stop()

    if os.name != "nt":
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

