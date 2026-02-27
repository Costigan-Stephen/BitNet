from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional, Union
from urllib import error as urlerror
from urllib import request as urlrequest

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import RagStore


ROOT_DIR = Path(__file__).resolve().parents[1]
WEBUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBUI_DIR / "static"
ICONS_DIR = WEBUI_DIR / "icons"
FONTS_DIR = WEBUI_DIR / "fonts"
DATA_DIR = WEBUI_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TRAIN_DATASETS_DIR = DATA_DIR / "train_datasets"
RAG_DB_PATH = DATA_DIR / "rag_index.json"
MODEL_PROFILES_PATH = DATA_DIR / "model_profiles.json"
RAG_PROFILES_PATH = DATA_DIR / "rag_profiles.json"
DEFAULT_TRAIN_BASE_MODEL = "tiiuae/Falcon-E-1B-Base"
DEFAULT_TRAIN_MODEL_REVISION = "prequantized"
PROFILE_UPLOADS_ROOT = UPLOADS_DIR / "profiles"
PROFILE_CODE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
    ".doc",
    ".docx",
    ".pdf",
}
TRAIN_STRUCTURED_DATASET_EXTENSIONS = {".json", ".jsonl", ".csv"}
TRAIN_DOCUMENT_UPLOAD_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".doc",
    ".docx",
    ".pdf",
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


def _normalize_health_host(host: str) -> str:
    value = str(host or "").strip()
    lowered = value.lower()
    if lowered in {"0.0.0.0", "::", "[::]", "::0", "*", "+"}:
        return "127.0.0.1"
    return value or "127.0.0.1"


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
        self.attached_external = False

    @property
    def base_url(self) -> str:
        safe_host = _normalize_health_host(self.host)
        return f"http://{safe_host}:{self.port}"

    def _endpoint_healthy(self, host: str, port: int, timeout_sec: float = 0.9) -> bool:
        safe_host = _normalize_health_host(host)
        health_url = f"http://{safe_host}:{int(port)}/health"
        try:
            with urlrequest.urlopen(health_url, timeout=timeout_sec) as response:
                return response.status == 200
        except Exception:
            return False

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
        health_host = _normalize_health_host(self.host)
        health_url = f"http://{health_host}:{self.port}/health"
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

            clean_host = str(host).strip() or "127.0.0.1"
            clean_port = int(port)

            # Reuse an already-running endpoint instead of spawning duplicates.
            if self._endpoint_healthy(clean_host, clean_port):
                self.host = clean_host
                self.port = clean_port
                if model_path and str(model_path).strip():
                    self.model_path = str(Path(model_path).expanduser().resolve())
                self.extra = {
                    "ctx_size": ctx_size,
                    "threads": threads,
                    "n_predict": n_predict,
                    "lora_paths": lora_paths,
                    "extra_args": extra_args,
                    "reuse_existing": True,
                }
                self.state.process = None
                self.state.started_at = None
                self.state.command = []
                self.state.last_error = None
                self.attached_external = True
                self.state.append_log(
                    f"[manager] Reusing existing llama-server at {self.host}:{self.port} (no new process started)."
                )
                return self.status()

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
                clean_host,
                "--port",
                str(clean_port),
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

            self.host = clean_host
            self.port = clean_port
            self.model_path = str(model)
            self.extra = {
                "ctx_size": ctx_size,
                "threads": threads,
                "n_predict": n_predict,
                "lora_paths": lora_paths,
                "extra_args": extra_args,
                "reuse_existing": False,
            }
            self.state.command = command
            self.state.logs.clear()
            self.state.last_error = None
            self.attached_external = False

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
                if self._endpoint_healthy(self.host, self.port, timeout_sec=0.35):
                    self.attached_external = True
                    self.state.last_error = (
                        f"llama-server at {self.host}:{self.port} is running but unmanaged by this Web UI process."
                    )
                else:
                    self.attached_external = False
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
            self.attached_external = False
            return self.status()

    def status(self) -> dict[str, Any]:
        process = self.state.process
        managed_running = process is not None and process.poll() is None
        external_running = False
        if not managed_running:
            external_running = self._endpoint_healthy(self.host, self.port, timeout_sec=0.35)
        running = managed_running or external_running
        return {
            "running": running,
            "managed": managed_running,
            "external": external_running,
            "attached_external": self.attached_external,
            "pid": process.pid if managed_running else None,
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

    def _derive_last_error_from_logs(self) -> Optional[str]:
        tail = self.state.tail(240)
        if not tail:
            return None

        full_text = "\n".join(tail).lower()
        if "no module named 'triton'" in full_text or 'no module named "triton"' in full_text:
            if os.name == "nt":
                return (
                    "Training failed: missing Triton runtime. Install `python -m pip install triton-windows` "
                    "and restart the Web UI server."
                )
            return "Training failed: missing Triton runtime. Install `python -m pip install triton`."
        if "no module named 'onebitllms.model'" in full_text or 'no module named "onebitllms.model"' in full_text:
            return (
                "Training failed: incompatible onebitllms package layout. "
                "Update dependencies with `python -m pip install -r webui/requirements-train.txt`."
            )
        if "gatedrepoerror" in full_text or "you are trying to access a gated repo" in full_text:
            return (
                "Training failed: Hugging Face gated model access denied. "
                "Use an open model ID or run 'huggingface-cli login' with access."
            )
        if "401 client error" in full_text and "huggingface.co" in full_text:
            return (
                "Training failed: Hugging Face authentication required for the selected model."
            )

        for line in reversed(tail):
            row = str(line).strip()
            if row.startswith("[stderr]"):
                detail = row[len("[stderr]") :].strip()
                if detail:
                    return detail
        return None

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
                if process.returncode != 0 and not self.state.last_error:
                    derived = self._derive_last_error_from_logs()
                    if derived:
                        self.state.last_error = derived
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


class ModelProfileStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._active_profile_id = "base"
        self._profiles: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        try:
            payload = json.loads(self._db_path.read_text(encoding="utf-8"))
        except Exception:
            return

        active_profile_id = str(payload.get("active_profile_id", "base"))
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list):
            profiles = []

        cleaned_profiles: list[dict[str, Any]] = []
        for item in profiles:
            if not isinstance(item, dict):
                continue
            profile_id = str(item.get("id", "")).strip()
            if not profile_id:
                continue
            model_path = str(item.get("model_path", "")).strip()
            cleaned_profiles.append(
                {
                    "id": profile_id,
                    "name": str(item.get("name", profile_id)).strip() or profile_id,
                    "model_path": model_path,
                    "is_base": bool(item.get("is_base", profile_id == "base")),
                    "created_at": float(item.get("created_at", time.time())),
                    "updated_at": float(item.get("updated_at", time.time())),
                    "source": str(item.get("source", "manual")),
                    "rag_profile_id": str(item.get("rag_profile_id", "")).strip() or None,
                }
            )

        self._profiles = cleaned_profiles
        self._active_profile_id = active_profile_id

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_profile_id": self._active_profile_id,
            "profiles": self._profiles,
            "updated_at": time.time(),
        }
        self._db_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _profile_exists(self, profile_id: str) -> bool:
        return any(item.get("id") == profile_id for item in self._profiles)

    def _find_profile(self, profile_id: str) -> Optional[dict[str, Any]]:
        for profile in self._profiles:
            if profile.get("id") == profile_id:
                return profile
        return None

    def _serialize_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        model_path = str(profile.get("model_path", "")).strip()
        model = Path(model_path).expanduser().resolve() if model_path else None
        exists = bool(model and model.exists())
        return {
            "id": str(profile.get("id")),
            "name": str(profile.get("name", "")),
            "model_path": model_path,
            "is_base": bool(profile.get("is_base", False)),
            "created_at": float(profile.get("created_at", 0.0)),
            "updated_at": float(profile.get("updated_at", 0.0)),
            "source": str(profile.get("source", "manual")),
            "rag_profile_id": str(profile.get("rag_profile_id", "")).strip() or None,
            "exists": exists,
        }

    def _resolve_base_model_path_locked(self) -> Optional[Path]:
        base = self._find_profile("base")
        if not base:
            return None
        model_path = str(base.get("model_path", "")).strip()
        if not model_path:
            return None
        candidate = Path(model_path).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _sorted_profiles(self) -> list[dict[str, Any]]:
        return sorted(
            self._profiles,
            key=lambda item: (
                0 if bool(item.get("is_base", False)) else 1,
                str(item.get("name", "")).lower(),
            ),
        )

    def initialize_base(self, model_path: Optional[str]) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            base = self._find_profile("base")
            resolved_model = ""
            if model_path:
                candidate = Path(model_path).expanduser().resolve()
                resolved_model = str(candidate)

            if base is None:
                base = {
                    "id": "base",
                    "name": "Base BitNet",
                    "model_path": resolved_model,
                    "is_base": True,
                    "created_at": now,
                    "updated_at": now,
                    "source": "system",
                }
                self._profiles.append(base)
            else:
                base["is_base"] = True
                if resolved_model:
                    base["model_path"] = resolved_model
                if "rag_profile_id" not in base:
                    base["rag_profile_id"] = None
                base["updated_at"] = now

            if not self._profile_exists(self._active_profile_id):
                self._active_profile_id = "base"
            self._save()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            profiles = [self._serialize_profile(item) for item in self._sorted_profiles()]
            active = next((item for item in profiles if item["id"] == self._active_profile_id), None)
            return {
                "active_profile_id": self._active_profile_id,
                "active_profile": active,
                "profiles": profiles,
            }

    def register_profile(
        self,
        name: str,
        model_path: str,
        set_active: bool = False,
        initialize_from_base: bool = True,
    ) -> dict[str, Any]:
        raw_path = str(model_path or "").strip()
        if not raw_path:
            raise HTTPException(status_code=400, detail="model_path is required.")

        candidate = Path(raw_path).expanduser().resolve()
        if candidate.suffix.lower() != ".gguf":
            raise HTTPException(status_code=400, detail="Only .gguf model paths are supported in model profiles.")

        clean_name = str(name or "").strip() or candidate.stem
        slug = re.sub(r"[^a-z0-9]+", "-", clean_name.lower()).strip("-") or "model"

        with self._lock:
            path_exists = candidate.exists()
            copied_from_base = False
            if path_exists and not candidate.is_file():
                raise HTTPException(status_code=400, detail=f"Model path is not a file: {candidate}")

            if not path_exists and initialize_from_base:
                base_model_path = self._resolve_base_model_path_locked()
                if base_model_path is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Base model file is unavailable; cannot initialize a copy.",
                    )
                if str(base_model_path).lower() == str(candidate).lower():
                    path_exists = True
                else:
                    try:
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(base_model_path, candidate)
                        path_exists = candidate.exists() and candidate.is_file()
                        copied_from_base = path_exists
                    except Exception as exc:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to duplicate base model to {candidate}: {exc}",
                        ) from exc

            source_value = "manual_cloned" if copied_from_base else ("manual" if path_exists else "manual_pending")
            existing = next(
                (item for item in self._profiles if str(item.get("model_path", "")).lower() == str(candidate).lower()),
                None,
            )
            now = time.time()
            if existing:
                if not bool(existing.get("is_base", False)):
                    existing["name"] = clean_name
                    existing["updated_at"] = now
                    existing["source"] = source_value
                    if "rag_profile_id" not in existing:
                        existing["rag_profile_id"] = None
                if set_active and path_exists:
                    self._active_profile_id = str(existing["id"])
                self._save()
                return self.status()

            profile_id = f"{slug}-{int(now * 1000)}"
            profile = {
                "id": profile_id,
                "name": clean_name,
                "model_path": str(candidate),
                "is_base": False,
                "created_at": now,
                "updated_at": now,
                "source": source_value,
                "rag_profile_id": None,
            }
            self._profiles.append(profile)
            if set_active and path_exists:
                self._active_profile_id = profile_id
            self._save()
            return self.status()

    def activate_profile(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Unknown profile id: {profile_id}")
            model_path = str(profile.get("model_path", "")).strip()
            if not model_path:
                raise HTTPException(status_code=400, detail="Profile has no model path configured.")
            candidate = Path(model_path).expanduser().resolve()
            if not candidate.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Profile model file does not exist yet: {candidate}",
                )
            self._active_profile_id = profile_id
            self._save()
            return self.status()

    def remove_profile(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Unknown profile id: {profile_id}")
            if bool(profile.get("is_base", False)):
                raise HTTPException(status_code=400, detail="Base profile cannot be removed.")
            self._profiles = [item for item in self._profiles if item.get("id") != profile_id]
            if self._active_profile_id == profile_id:
                self._active_profile_id = "base"
            self._save()
            return self.status()

    def reset_to_base(self) -> dict[str, Any]:
        with self._lock:
            if not self._profile_exists("base"):
                raise HTTPException(status_code=400, detail="Base profile is not configured.")
            self._active_profile_id = "base"
            self._save()
            return self.status()

    def link_rag_profile(self, profile_id: str, rag_profile_id: Optional[str]) -> dict[str, Any]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Unknown profile id: {profile_id}")
            linked = str(rag_profile_id or "").strip() or None
            profile["rag_profile_id"] = linked
            profile["updated_at"] = time.time()
            self._save()
            return self.status()

    def clear_rag_profile_links(self, rag_profile_id: str) -> dict[str, Any]:
        target = str(rag_profile_id or "").strip()
        if not target:
            return self.status()
        with self._lock:
            changed = False
            for profile in self._profiles:
                linked = str(profile.get("rag_profile_id", "")).strip()
                if linked and linked == target:
                    profile["rag_profile_id"] = None
                    profile["updated_at"] = time.time()
                    changed = True
            if changed:
                self._save()
            return self.status()

    def get_active_model_path(self) -> Optional[str]:
        with self._lock:
            profile = self._find_profile(self._active_profile_id)
            if not profile:
                return None
            model_path = str(profile.get("model_path", "")).strip()
            return model_path or None

    def get_active_rag_profile_id(self) -> Optional[str]:
        with self._lock:
            profile = self._find_profile(self._active_profile_id)
            if not profile:
                return None
            linked = str(profile.get("rag_profile_id", "")).strip()
            return linked or None


class RagProfileStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._active_profile_id: Optional[str] = None
        self._profiles: list[dict[str, Any]] = []
        self._load()

    def _normalize_sources(self, paths: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for raw in paths:
            value = str(raw or "").strip()
            if not value:
                continue
            normalized = self._normalize_path(value)
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
        return merged

    @staticmethod
    def _normalize_path(raw_path: str) -> str:
        try:
            return str(Path(raw_path).expanduser().resolve())
        except Exception:
            return str(Path(raw_path))

    @staticmethod
    def _normalize_code(raw_code: str) -> str:
        code = str(raw_code or "").strip().upper()
        if len(code) != 4:
            return ""
        if any(ch not in PROFILE_CODE_ALPHABET for ch in code):
            return ""
        return code

    def _generate_profile_code(self, existing_codes: set[str]) -> str:
        for _ in range(2048):
            candidate = "".join(secrets.choice(PROFILE_CODE_ALPHABET) for _ in range(4))
            if candidate not in existing_codes:
                return candidate
        raise RuntimeError("Unable to generate a unique 4-character profile code.")

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        try:
            payload = json.loads(self._db_path.read_text(encoding="utf-8"))
        except Exception:
            return

        active_profile_id = str(payload.get("active_profile_id", "")).strip() or None
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list):
            profiles = []

        cleaned: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        changed = False
        for item in profiles:
            if not isinstance(item, dict):
                continue
            profile_id = str(item.get("id", "")).strip()
            if not profile_id:
                continue
            source_paths_raw = item.get("source_paths", [])
            source_paths = source_paths_raw if isinstance(source_paths_raw, list) else []
            code = self._normalize_code(str(item.get("code", "")).strip())
            if not code or code in seen_codes:
                code = self._generate_profile_code(seen_codes)
                changed = True
            seen_codes.add(code)
            cleaned.append(
                {
                    "id": profile_id,
                    "name": str(item.get("name", profile_id)).strip() or profile_id,
                    "description": str(item.get("description", "")).strip(),
                    "code": code,
                    "source_paths": self._normalize_sources([str(value) for value in source_paths]),
                    "created_at": float(item.get("created_at", time.time())),
                    "updated_at": float(item.get("updated_at", time.time())),
                }
            )

        self._profiles = cleaned
        self._active_profile_id = active_profile_id
        if self._active_profile_id and not self._find_profile(self._active_profile_id):
            self._active_profile_id = None
            changed = True
        if self._active_profile_id is None and self._profiles:
            self._active_profile_id = str(self._profiles[0].get("id", "")).strip() or None
            changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_profile_id": self._active_profile_id,
            "profiles": self._profiles,
            "updated_at": time.time(),
        }
        self._db_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _find_profile(self, profile_id: Optional[str]) -> Optional[dict[str, Any]]:
        target = str(profile_id or "").strip()
        if not target:
            return None
        for profile in self._profiles:
            if str(profile.get("id", "")) == target:
                return profile
        return None

    def has_profile(self, profile_id: str) -> bool:
        with self._lock:
            return self._find_profile(profile_id) is not None

    def _serialize_profile(self, profile: dict[str, Any], source_counts: Optional[dict[str, int]]) -> dict[str, Any]:
        source_paths = [str(value) for value in profile.get("source_paths", []) if str(value).strip()]
        normalized_counts: dict[str, int] = {}
        if source_counts:
            for source, count in source_counts.items():
                normalized_counts[self._normalize_path(source).lower()] = int(count)

        indexed_sources = 0
        indexed_chunks = 0
        for source in source_paths:
            count = int(normalized_counts.get(self._normalize_path(source).lower(), 0))
            if count > 0:
                indexed_sources += 1
                indexed_chunks += count

        code = self._normalize_code(str(profile.get("code", "")).strip())
        if not code:
            code = "UNKN"
        upload_dir = PROFILE_UPLOADS_ROOT / code
        return {
            "id": str(profile.get("id", "")),
            "name": str(profile.get("name", "")),
            "description": str(profile.get("description", "")),
            "code": code,
            "upload_dir": str(upload_dir.resolve()),
            "source_paths": source_paths,
            "source_count": len(source_paths),
            "indexed_sources": indexed_sources,
            "indexed_chunks": indexed_chunks,
            "created_at": float(profile.get("created_at", 0.0)),
            "updated_at": float(profile.get("updated_at", 0.0)),
        }

    def status(self, source_counts: Optional[dict[str, int]] = None) -> dict[str, Any]:
        with self._lock:
            rows = [self._serialize_profile(item, source_counts) for item in self._profiles]
            rows.sort(key=lambda item: str(item.get("name", "")).lower())
            active = next((item for item in rows if item.get("id") == self._active_profile_id), None)
            return {
                "active_profile_id": self._active_profile_id,
                "active_profile": active,
                "profiles": rows,
            }

    def create_profile(
        self,
        name: str,
        description: str = "",
        source_paths: Optional[list[str]] = None,
        set_active: bool = True,
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="name is required.")
        now = time.time()
        slug = re.sub(r"[^a-z0-9]+", "-", clean_name.lower()).strip("-") or "rag-profile"
        with self._lock:
            existing_codes = {self._normalize_code(str(row.get("code", "")).strip()) for row in self._profiles}
            existing_codes.discard("")
            profile_id = f"{slug}-{int(now * 1000)}"
            profile = {
                "id": profile_id,
                "name": clean_name,
                "description": str(description or "").strip(),
                "code": self._generate_profile_code(existing_codes),
                "source_paths": self._normalize_sources(source_paths or []),
                "created_at": now,
                "updated_at": now,
            }
            self._profiles.append(profile)
            if set_active:
                self._active_profile_id = profile_id
            self._save()
            return self.status()

    def update_profile(
        self,
        profile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        source_paths: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Unknown RAG profile id: {profile_id}")
            if name is not None:
                clean_name = str(name).strip()
                if not clean_name:
                    raise HTTPException(status_code=400, detail="name cannot be empty.")
                profile["name"] = clean_name
            if description is not None:
                profile["description"] = str(description).strip()
            if source_paths is not None:
                profile["source_paths"] = self._normalize_sources(source_paths)
            profile["updated_at"] = time.time()
            self._save()
            return self.status()

    def set_sources(self, profile_id: str, paths: list[str], mode: str = "add") -> dict[str, Any]:
        clean_mode = str(mode or "add").strip().lower()
        if clean_mode not in {"add", "remove", "replace"}:
            raise HTTPException(status_code=400, detail="mode must be one of: add, remove, replace")
        normalized_paths = self._normalize_sources(paths)
        with self._lock:
            profile = self._find_profile(profile_id)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Unknown RAG profile id: {profile_id}")
            existing = self._normalize_sources([str(value) for value in profile.get("source_paths", [])])
            if clean_mode == "replace":
                updated = normalized_paths
            elif clean_mode == "remove":
                remove_keys = {item.lower() for item in normalized_paths}
                updated = [item for item in existing if item.lower() not in remove_keys]
            else:
                updated = self._normalize_sources(existing + normalized_paths)
            profile["source_paths"] = updated
            profile["updated_at"] = time.time()
            self._save()
            return self.status()

    def activate_profile(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Unknown RAG profile id: {profile_id}")
            self._active_profile_id = str(profile["id"])
            self._save()
            return self.status()

    def clear_active_profile(self) -> dict[str, Any]:
        with self._lock:
            if self._active_profile_id is not None:
                self._active_profile_id = None
                self._save()
            return self.status()

    def remove_profile(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            target = self._find_profile(profile_id)
            if target is None:
                raise HTTPException(status_code=404, detail=f"Unknown RAG profile id: {profile_id}")
            self._profiles = [item for item in self._profiles if str(item.get("id", "")) != str(profile_id)]
            if self._active_profile_id == profile_id:
                self._active_profile_id = str(self._profiles[0].get("id", "")).strip() if self._profiles else None
            self._save()
            return self.status()

    def get_profile_sources(self, profile_id: Optional[str]) -> list[str]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if not profile:
                return []
            return [str(value) for value in profile.get("source_paths", []) if str(value).strip()]

    def get_profile(self, profile_id: Optional[str]) -> Optional[dict[str, Any]]:
        with self._lock:
            profile = self._find_profile(profile_id)
            if not profile:
                return None
            source_paths = [str(value) for value in profile.get("source_paths", []) if str(value).strip()]
            code = self._normalize_code(str(profile.get("code", "")).strip())
            if not code:
                code = "UNKN"
            return {
                "id": str(profile.get("id", "")),
                "name": str(profile.get("name", "")),
                "description": str(profile.get("description", "")),
                "code": code,
                "upload_dir": str((PROFILE_UPLOADS_ROOT / code).resolve()),
                "source_paths": source_paths,
                "source_count": len(source_paths),
            }

    def get_active_profile(self) -> Optional[dict[str, Any]]:
        with self._lock:
            active_id = self._active_profile_id
        return self.get_profile(active_id)

    def get_active_profile_id(self) -> Optional[str]:
        with self._lock:
            return self._active_profile_id

    def get_active_source_paths(self) -> list[str]:
        with self._lock:
            if not self._active_profile_id:
                return []
            profile = self._find_profile(self._active_profile_id)
            if not profile:
                return []
            return [str(value) for value in profile.get("source_paths", []) if str(value).strip()]

    def remove_paths_from_all_profiles(self, paths: list[str]) -> dict[str, Any]:
        normalized_targets = {self._normalize_path(path).lower() for path in self._normalize_sources(paths)}
        if not normalized_targets:
            return self.status()

        with self._lock:
            changed = False
            now = time.time()
            for profile in self._profiles:
                existing = [str(value) for value in profile.get("source_paths", []) if str(value).strip()]
                updated = [item for item in existing if self._normalize_path(item).lower() not in normalized_targets]
                if len(updated) != len(existing):
                    profile["source_paths"] = updated
                    profile["updated_at"] = now
                    changed = True
            if changed:
                self._save()
            return self.status()

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
    rag_profile_id: Optional[str] = None


class RagIndexRequest(BaseModel):
    paths: list[str]
    chunk_size: int = 220
    chunk_overlap: int = 40
    reset: bool = False


class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 4
    profile_id: Optional[str] = None


class TrainStartRequest(BaseModel):
    command: Optional[list[str]] = None
    base_model: Optional[str] = None
    model_revision: str = DEFAULT_TRAIN_MODEL_REVISION
    tokenizer_model: Optional[str] = None
    dataset_path: Optional[str] = None
    output_dir: str = "training-output/bitnet"
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    quantize_output: bool = False
    auto_descope_rag_sources: bool = True


class LoraApplyRequest(BaseModel):
    adapters: list[dict[str, Any]]


class ModelProfileRegisterRequest(BaseModel):
    name: str
    model_path: Optional[str] = None
    model_dir: Optional[str] = None
    initialize_from_base: bool = True
    set_active: bool = False


class ModelProfileActivateRequest(BaseModel):
    profile_id: str


class ModelProfileRemoveRequest(BaseModel):
    profile_id: str


class ModelProfileLinkRagRequest(BaseModel):
    profile_id: str
    rag_profile_id: Optional[str] = None


class RagProfileCreateRequest(BaseModel):
    name: str
    description: str = ""
    source_paths: list[str] = Field(default_factory=list)
    set_active: bool = True


class RagProfileUpdateRequest(BaseModel):
    profile_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    source_paths: Optional[list[str]] = None


class RagProfileActivateRequest(BaseModel):
    profile_id: str


class RagProfileRemoveRequest(BaseModel):
    profile_id: str


class RagProfileSourcesRequest(BaseModel):
    profile_id: str
    paths: list[str]
    mode: str = "add"


class RagExportTrainRequest(BaseModel):
    paths: list[str]
    dataset_name: str = "rag_dataset"
    chunk_size: int = 220
    chunk_overlap: int = 40
    append: bool = False


class DirPickRequest(BaseModel):
    start_path: Optional[str] = None


class RagFilesRemoveRequest(BaseModel):
    paths: list[str]
    delete_files: bool = True


app = FastAPI(title="BitNet Web UI", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/icons", StaticFiles(directory=str(ICONS_DIR)), name="icons")
app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")

llama_server = LlamaServerManager()
rag_store = RagStore(RAG_DB_PATH)
training = TrainingManager()
model_profiles = ModelProfileStore(MODEL_PROFILES_PATH)
rag_profiles = RagProfileStore(RAG_PROFILES_PATH)
TRAIN_CLEANUP_LOCK = threading.RLock()
train_cleanup_state: dict[str, Any] = {
    "last_run_id": "",
    "result": None,
}


@app.middleware("http")
async def static_no_cache(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/") or path.startswith("/icons/") or path.startswith("/fonts/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _llama_healthcheck(timeout_sec: float = 0.9) -> bool:
    safe_host = _normalize_health_host(llama_server.host)
    safe_port = _safe_int(llama_server.port, 8080, 1)
    health_url = f"http://{safe_host}:{safe_port}/health"
    try:
        with urlrequest.urlopen(health_url, timeout=timeout_sec) as response:
            return response.status == 200
    except Exception:
        return False


def _safe_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed < minimum:
        return default
    return parsed


def _resolve_runtime_model_path() -> Optional[Path]:
    candidates: list[Path] = []
    active_model_path = model_profiles.get_active_model_path()
    if active_model_path:
        try:
            candidates.append(Path(active_model_path).expanduser().resolve())
        except Exception:
            pass
    if llama_server.model_path:
        try:
            candidates.append(Path(llama_server.model_path).expanduser().resolve())
        except Exception:
            pass
    default_model = _pick_default_model()
    if default_model is not None:
        candidates.append(default_model)

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def _auto_start_llama_server(reason: str = "on-demand") -> bool:
    model = _resolve_runtime_model_path()
    if model is None:
        message = "llama-server is not running and no valid model could be resolved for auto-start."
        llama_server.state.last_error = message
        LOGGER.warning(message)
        return False

    status = llama_server.status()
    host = str(status.get("host") or llama_server.host or "127.0.0.1").strip() or "127.0.0.1"
    port = _safe_int(status.get("port"), 8080, 1)
    extra = status.get("extra") if isinstance(status.get("extra"), dict) else {}
    if not extra and isinstance(llama_server.extra, dict):
        extra = llama_server.extra

    ctx_size = _safe_int(extra.get("ctx_size"), 2048, 16)
    threads_default = max(2, (os.cpu_count() or 4) // 2)
    threads = _safe_int(extra.get("threads"), threads_default, 1)
    n_predict = _safe_int(extra.get("n_predict"), 4096, 1)

    lora_paths = [str(item).strip() for item in (extra.get("lora_paths") or []) if str(item).strip()]
    extra_args = [str(item).strip() for item in (extra.get("extra_args") or []) if str(item).strip()]

    if bool(status.get("managed")) and llama_server.state.is_running():
        try:
            llama_server.stop()
        except Exception:
            pass

    try:
        llama_server.start(
            str(model),
            host,
            port,
            ctx_size,
            threads,
            n_predict,
            lora_paths,
            extra_args,
        )
        LOGGER.info("Auto-started llama-server (%s) on %s:%s with model %s", reason, host, port, model)
        return True
    except Exception as exc:
        message = f"Auto-start failed ({reason}): {exc}"
        llama_server.state.last_error = message
        LOGGER.warning(message)
        return False


def _ensure_server_running() -> None:
    status = llama_server.status()
    if status.get("running") and _llama_healthcheck(timeout_sec=0.8):
        return

    # Recover from stopped or stale/unhealthy runtime state.
    if _auto_start_llama_server(reason="chat-request"):
        if _llama_healthcheck(timeout_sec=1.2):
            return

    detail = llama_server.state.last_error or "llama-server is not running."
    raise HTTPException(status_code=503, detail=detail)


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


def _profile_upload_dir(profile_code: str) -> Path:
    clean = str(profile_code or "").strip().upper()
    if len(clean) != 4 or any(ch not in PROFILE_CODE_ALPHABET for ch in clean):
        raise HTTPException(status_code=400, detail=f"Invalid profile code: {profile_code}")
    return PROFILE_UPLOADS_ROOT / clean


def _unique_profile_upload_path(profile_code: str, filename: str) -> Path:
    return _unique_path_in_dir(_profile_upload_dir(profile_code), filename)


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


def _normalize_path_str(raw_path: str) -> str:
    try:
        return str(Path(raw_path).expanduser().resolve())
    except Exception:
        return str(Path(raw_path))


def _match_source_key(raw_path: str, source_keys: list[str]) -> Optional[str]:
    normalized = _normalize_path_str(raw_path).lower()
    for key in source_keys:
        if str(key).lower() == normalized:
            return key
    return None


def _safe_dataset_stem(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name or "").strip()).strip("._")
    return cleaned or "rag_dataset"


def _extract_dataset_source_paths(dataset_path: Path) -> list[str]:
    candidate = dataset_path.expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        return []

    sources: list[str] = []
    seen: set[str] = set()

    def add_source(raw_value: Any) -> None:
        if raw_value is None:
            return
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return
            normalized = _normalize_path_str(text)
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            sources.append(normalized)
            return
        if isinstance(raw_value, (list, tuple, set)):
            for item in raw_value:
                add_source(item)

    suffix = candidate.suffix.lower()
    if suffix == ".jsonl":
        try:
            with candidate.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    row = raw_line.strip()
                    if not row:
                        continue
                    try:
                        payload = json.loads(row)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        add_source(payload.get("source"))
        except Exception:
            return sources
        return sources

    if suffix == ".json":
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return sources

        rows: list[Any]
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                rows = payload["data"]
            elif isinstance(payload.get("train"), list):
                rows = payload["train"]
            elif isinstance(payload.get("records"), list):
                rows = payload["records"]
            else:
                rows = [payload]
        else:
            rows = []

        for row in rows:
            if isinstance(row, dict):
                add_source(row.get("source"))
        return sources

    if suffix == ".csv":
        try:
            with candidate.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    source_value = row.get("source")
                    if source_value is None:
                        for key, value in row.items():
                            if str(key or "").strip().lower() == "source":
                                source_value = value
                                break
                    add_source(source_value)
        except Exception:
            return sources

    return sources


def _descope_rag_sources_from_training_dataset(job_config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "applied": False,
        "reason": "",
        "dataset_path": None,
        "profile_id": None,
        "dataset_source_count": 0,
        "matched_source_count": 0,
        "matched_sources": [],
        "removed_index": None,
    }

    auto_descope = bool(job_config.get("auto_descope_rag_sources", True))
    if not auto_descope:
        result["reason"] = "disabled_by_request"
        return result

    dataset_path_raw = str(job_config.get("dataset_path", "")).strip()
    if not dataset_path_raw:
        result["reason"] = "missing_dataset_path"
        return result

    dataset_path = Path(dataset_path_raw).expanduser().resolve()
    result["dataset_path"] = str(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_file():
        result["reason"] = "dataset_not_found"
        return result

    dataset_sources = _extract_dataset_source_paths(dataset_path)
    result["dataset_source_count"] = len(dataset_sources)
    if not dataset_sources:
        result["reason"] = "dataset_has_no_source_fields"
        return result

    profile_id = str(job_config.get("active_rag_profile_id", "")).strip() or rag_profiles.get_active_profile_id()
    result["profile_id"] = profile_id or None
    if not profile_id:
        result["reason"] = "no_active_rag_profile"
        return result
    if not rag_profiles.has_profile(profile_id):
        result["reason"] = "rag_profile_missing"
        return result

    profile_sources = rag_profiles.get_profile_sources(profile_id)
    profile_map = {_normalize_path_str(path).lower(): _normalize_path_str(path) for path in profile_sources}

    matched_sources: list[str] = []
    matched_seen: set[str] = set()
    for source in dataset_sources:
        key = _normalize_path_str(source).lower()
        match = profile_map.get(key)
        if not match or key in matched_seen:
            continue
        matched_seen.add(key)
        matched_sources.append(match)

    result["matched_source_count"] = len(matched_sources)
    result["matched_sources"] = matched_sources
    if not matched_sources:
        result["reason"] = "no_overlap_with_profile_sources"
        return result

    try:
        removed_index = rag_store.remove_sources(matched_sources)
        rag_profiles.set_sources(profile_id, matched_sources, "remove")
    except Exception as exc:
        result["reason"] = f"cleanup_error:{exc.__class__.__name__}"
        result["error"] = str(exc)
        return result

    result["applied"] = True
    result["reason"] = "completed"
    result["removed_index"] = removed_index
    result["remaining_profile_source_count"] = len(rag_profiles.get_profile_sources(profile_id))
    return result


def _is_within_dir(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _safe_model_filename(profile_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(profile_name or "").lower()).strip("-")
    if not slug:
        slug = "model"
    return f"{slug}.gguf"


def _active_rag_profile_context(required: bool = True) -> dict[str, Any]:
    profile = rag_profiles.get_active_profile()
    if profile:
        return profile
    if required:
        raise HTTPException(
            status_code=400,
            detail="No active profile selected. Select a profile before managing RAG files.",
        )
    return {
        "id": None,
        "name": "",
        "code": "",
        "upload_dir": str(UPLOADS_DIR.resolve()),
        "source_paths": [],
        "source_count": 0,
    }


def _available_roots() -> list[str]:
    if os.name == "nt":
        roots: list[str] = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append(str(drive))
        return roots
    return ["/"]


def _pick_directory_native(start_path: Optional[str] = None) -> Optional[str]:
    initial = ""
    if start_path:
        try:
            initial = str(Path(start_path).expanduser().resolve())
        except Exception:
            initial = str(start_path).strip()

    if os.name == "nt":
        escaped = initial.replace("'", "''")
        script_open = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog=New-Object System.Windows.Forms.OpenFileDialog; "
            "$dialog.Title='Select model directory'; "
            "$dialog.CheckFileExists=$false; "
            "$dialog.CheckPathExists=$true; "
            "$dialog.ValidateNames=$false; "
            "$dialog.DereferenceLinks=$true; "
            "$dialog.FileName='Select Folder'; "
            f"$initial='{escaped}'; "
            "if ($initial -and (Test-Path -LiteralPath $initial)) { "
            "$item=Get-Item -LiteralPath $initial; "
            "if ($item.PSIsContainer) { $dialog.InitialDirectory=$initial } "
            "else { $dialog.InitialDirectory=(Split-Path -LiteralPath $initial -Parent) } }; "
            "$result=$dialog.ShowDialog(); "
            "if ($result -eq [System.Windows.Forms.DialogResult]::OK) { "
            "$picked=$dialog.FileName; "
            "$selected=''; "
            "if ((Test-Path -LiteralPath $picked) -and (Get-Item -LiteralPath $picked).PSIsContainer) { "
            "$selected=$picked "
            "} else { "
            "$selected=Split-Path -LiteralPath $picked -Parent "
            "}; "
            "if ($selected) { "
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
            "Write-Output $selected } }"
        )
        script_folder = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog=New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description='Select model directory'; "
            "$dialog.ShowNewFolderButton=$true; "
            f"$initial='{escaped}'; "
            "if ($initial -and (Test-Path -LiteralPath $initial)) { $dialog.SelectedPath=$initial }; "
            "$result=$dialog.ShowDialog(); "
            "if ($result -eq [System.Windows.Forms.DialogResult]::OK) { "
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
            "Write-Output $dialog.SelectedPath }"
        )

        attempts = [
            ["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command", script_open],
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script_open],
            ["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command", script_folder],
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script_folder],
        ]
        last_error = ""
        for command in attempts:
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=240,
                    check=False,
                )
            except Exception as exc:
                last_error = str(exc)
                continue

            if proc.returncode == 0:
                selected = (proc.stdout or "").strip()
                if not selected:
                    return None
                return str(Path(selected).expanduser().resolve())

            last_error = (proc.stderr or proc.stdout or "").strip()

        raise RuntimeError(last_error or "Windows folder picker failed.")

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"tkinter unavailable: {exc}") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    selected = filedialog.askdirectory(
        initialdir=initial if initial and Path(initial).exists() else None,
        title="Select model directory",
        mustexist=False,
    )
    try:
        root.destroy()
    except Exception:
        pass

    if not selected:
        return None
    return str(Path(selected).expanduser().resolve())


def _collect_rag_file_records(
    source_filter: Optional[list[str]] = None,
    uploads_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    source_counts = rag_store.source_chunk_counts()
    records: dict[str, dict[str, Any]] = {}
    filter_enabled = source_filter is not None
    normalized_filter = {
        _normalize_path_str(path).lower()
        for path in (source_filter or [])
        if str(path or "").strip()
    } if filter_enabled else set()

    uploads_scope = (uploads_root or UPLOADS_DIR).resolve()

    for source_path, chunk_count in source_counts.items():
        normalized_source = _normalize_path_str(source_path).lower()
        if filter_enabled and normalized_source not in normalized_filter:
            continue
        path = Path(source_path)
        exists = path.exists()
        stat = path.stat() if exists else None
        records[source_path] = {
            "id": sha1(source_path.lower().encode("utf-8", errors="ignore")).hexdigest()[:12],
            "name": path.name,
            "path": source_path,
            "status": "indexed" if exists else "missing",
            "chunk_count": int(chunk_count),
            "size_bytes": int(stat.st_size) if stat else 0,
            "modified_at": float(stat.st_mtime) if stat else 0.0,
            "exists": exists,
            "in_uploads": _is_within_dir(path, uploads_scope),
        }

    if uploads_scope.exists():
        for candidate in sorted(uploads_scope.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in RAG_UPLOAD_EXTENSIONS:
                continue
            source_path = str(candidate.resolve())
            if source_path in records:
                continue
            stat = candidate.stat()
            records[source_path] = {
                "id": sha1(source_path.lower().encode("utf-8", errors="ignore")).hexdigest()[:12],
                "name": candidate.name,
                "path": source_path,
                "status": "uploaded",
                "chunk_count": 0,
                "size_bytes": int(stat.st_size),
                "modified_at": float(stat.st_mtime),
                "exists": True,
                "in_uploads": True,
            }

    if filter_enabled:
        for source_path in source_filter or []:
            normalized = _normalize_path_str(source_path)
            if normalized in records:
                continue
            candidate = Path(normalized)
            exists = candidate.exists()
            stat = candidate.stat() if exists else None
            records[normalized] = {
                "id": sha1(normalized.lower().encode("utf-8", errors="ignore")).hexdigest()[:12],
                "name": candidate.name,
                "path": normalized,
                "status": "uploaded" if exists else "missing",
                "chunk_count": 0,
                "size_bytes": int(stat.st_size) if stat else 0,
                "modified_at": float(stat.st_mtime) if stat else 0.0,
                "exists": exists,
                "in_uploads": _is_within_dir(candidate, uploads_scope),
            }

    return sorted(records.values(), key=lambda item: (item["status"] != "indexed", -item["modified_at"], item["name"]))


def _rag_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "indexed": 0,
        "uploaded": 0,
        "missing": 0,
    }
    for item in records:
        status = str(item.get("status", "uploaded"))
        if status in summary:
            summary[status] += 1
    return summary


def _rag_profiles_status_payload() -> dict[str, Any]:
    source_counts = rag_store.source_chunk_counts()
    rag_status = rag_profiles.status(source_counts=source_counts)
    model_status = model_profiles.status()
    linked_map: dict[str, list[dict[str, str]]] = {}
    for profile in model_status.get("profiles", []):
        rag_profile_id = str(profile.get("rag_profile_id", "")).strip()
        if not rag_profile_id:
            continue
        linked_map.setdefault(rag_profile_id, []).append(
            {
                "id": str(profile.get("id", "")),
                "name": str(profile.get("name", "")),
            }
        )

    profiles = []
    for row in rag_status.get("profiles", []):
        profile_id = str(row.get("id", ""))
        linked_models = linked_map.get(profile_id, [])
        item = dict(row)
        item["linked_models"] = linked_models
        item["linked_model_count"] = len(linked_models)
        profiles.append(item)

    active_profile_id = rag_status.get("active_profile_id")
    active_profile = next((item for item in profiles if item.get("id") == active_profile_id), None)
    return {
        "active_profile_id": active_profile_id,
        "active_profile": active_profile,
        "profiles": profiles,
    }


def _resolve_rag_profile_sources(profile_id: Optional[str]) -> tuple[Optional[str], list[str]]:
    requested = str(profile_id or "").strip()
    if requested:
        if not rag_profiles.has_profile(requested):
            raise HTTPException(status_code=404, detail=f"Unknown RAG profile id: {requested}")
        sources = rag_profiles.get_profile_sources(requested)
        return requested, sources

    active_profile_id = rag_profiles.get_active_profile_id()
    active_sources = rag_profiles.get_active_source_paths()
    return active_profile_id, active_sources


def _http_error_detail(exc: Exception, default: str = "Unable to reach llama-server.") -> str:
    detail_text = ""
    if isinstance(exc, HTTPException):
        detail_value = exc.detail
        if isinstance(detail_value, (dict, list)):
            try:
                detail_text = json.dumps(detail_value, ensure_ascii=True)
            except Exception:
                detail_text = str(detail_value)
        elif detail_value is not None:
            detail_text = str(detail_value)
    else:
        detail_text = str(exc)

    compact = re.sub(r"\s+", " ", str(detail_text or "")).strip()
    if compact:
        return compact
    return default


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
            raise HTTPException(status_code=502, detail=_http_error_detail(exc)) from exc
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
            raise HTTPException(status_code=502, detail=_http_error_detail(exc)) from exc

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
    rag_profile_id: Optional[str] = None

    query = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            query = str(msg.content or "")
            break
    # Honor the frontend toggle strictly. If RAG is disabled in the UI, do not
    # auto-enable retrieval based on query wording.
    should_use_rag = bool(req.rag_enabled)

    if should_use_rag and query:
        _ensure_rag_index_ready()
        rag_profile_id, source_filter = _resolve_rag_profile_sources(req.rag_profile_id)
        strict_source_filter = rag_profile_id is not None
        retrieval_chunks = rag_store.search(
            query,
            top_k=max(1, req.rag_top_k),
            source_filter=source_filter if strict_source_filter else None,
        )

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
        if rag_profile_id:
            context_lines.append(f"Active RAG profile: {rag_profile_id}")
        for idx, chunk in enumerate(retrieval_chunks, start=1):
            context_lines.append(f"--- BEGIN RETRIEVED CHUNK {idx} | SOURCE: {chunk['source']} ---")
            context_lines.append(_compact_retrieval_text(str(chunk.get("text", ""))))
            context_lines.append(f"--- END RETRIEVED CHUNK {idx} ---")
        system_parts.append("\n".join(context_lines))

    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for msg in req.messages:
        content = str(msg.content or "")
        if not content.strip():
            continue
        messages.append({"role": msg.role, "content": content})

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
    if req.system_prompt.strip():
        # Keep compatibility with backends that read system_prompt directly
        # even when a system-role message is also provided.
        payload["system_prompt"] = req.system_prompt.strip()
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
    command = [
        sys.executable,
        "-u",
        str(WEBUI_DIR / "train_bitnet.py"),
        "--base-model",
        req.base_model,
        "--model-revision",
        (req.model_revision or "").strip() or "main",
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
    ]
    tokenizer_model = (req.tokenizer_model or "").strip()
    if tokenizer_model:
        command.extend(["--tokenizer-model", tokenizer_model])
    if req.quantize_output:
        command.append("--quantize-output")
    return command


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


@app.get("/api/fs/dirs")
async def fs_dirs(path: Optional[str] = None, limit: int = 250) -> JSONResponse:
    max_items = max(25, min(1200, int(limit)))
    roots = _available_roots()

    if not path:
        root_dirs = [{"name": root, "path": root} for root in roots]
        return JSONResponse(
            {
                "path": "",
                "parent": None,
                "roots": roots,
                "dirs": root_dirs,
            }
        )

    try:
        target = Path(path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    if not target.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {target}")
    if not target.is_dir():
        target = target.parent

    try:
        dirs = []
        for candidate in target.iterdir():
            if not candidate.is_dir():
                continue
            dirs.append({"name": candidate.name, "path": str(candidate.resolve())})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read directory: {exc}") from exc

    dirs.sort(key=lambda item: item["name"].lower())
    if len(dirs) > max_items:
        dirs = dirs[:max_items]

    parent = str(target.parent.resolve()) if target.parent != target else None
    return JSONResponse(
        {
            "path": str(target),
            "parent": parent,
            "roots": roots,
            "dirs": dirs,
        }
    )


@app.post("/api/fs/pick-dir")
async def fs_pick_dir(req: DirPickRequest) -> JSONResponse:
    try:
        selected_path = await asyncio.to_thread(_pick_directory_native, req.start_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Directory picker failed: {exc}") from exc
    return JSONResponse(
        {
            "selected_path": selected_path,
            "cancelled": not bool(selected_path),
        }
    )


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


@app.get("/api/models")
async def models_status() -> JSONResponse:
    return JSONResponse(model_profiles.status())


@app.post("/api/models/register")
async def models_register(req: ModelProfileRegisterRequest) -> JSONResponse:
    clean_name = str(req.name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="name is required.")

    raw_model_path = str(req.model_path or "").strip()
    raw_model_dir = str(req.model_dir or "").strip()
    if not raw_model_path:
        if not raw_model_dir:
            raise HTTPException(status_code=400, detail="model_dir is required when model_path is not provided.")
        try:
            model_dir = Path(raw_model_dir).expanduser().resolve()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid model_dir: {exc}") from exc
        raw_model_path = str(model_dir / _safe_model_filename(clean_name))

    status = await asyncio.to_thread(
        model_profiles.register_profile,
        clean_name,
        raw_model_path,
        req.set_active,
        req.initialize_from_base,
    )
    return JSONResponse(status)


@app.post("/api/models/activate")
async def models_activate(req: ModelProfileActivateRequest) -> JSONResponse:
    status = await asyncio.to_thread(model_profiles.activate_profile, req.profile_id)
    linked_rag_profile_id = str((status.get("active_profile") or {}).get("rag_profile_id", "")).strip()
    if linked_rag_profile_id and rag_profiles.has_profile(linked_rag_profile_id):
        await asyncio.to_thread(rag_profiles.activate_profile, linked_rag_profile_id)
    else:
        await asyncio.to_thread(rag_profiles.clear_active_profile)
    rag_status = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse({"models": status, "rag_profiles": rag_status})


@app.post("/api/models/remove")
async def models_remove(req: ModelProfileRemoveRequest) -> JSONResponse:
    status = await asyncio.to_thread(model_profiles.remove_profile, req.profile_id)
    return JSONResponse(status)


@app.post("/api/models/reset-base")
async def models_reset_base() -> JSONResponse:
    status = await asyncio.to_thread(model_profiles.reset_to_base)
    linked_rag_profile_id = str((status.get("active_profile") or {}).get("rag_profile_id", "")).strip()
    if linked_rag_profile_id and rag_profiles.has_profile(linked_rag_profile_id):
        await asyncio.to_thread(rag_profiles.activate_profile, linked_rag_profile_id)
    else:
        await asyncio.to_thread(rag_profiles.clear_active_profile)
    rag_status = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse({"models": status, "rag_profiles": rag_status})


@app.post("/api/models/link-rag")
async def models_link_rag(req: ModelProfileLinkRagRequest) -> JSONResponse:
    linked_rag_profile_id = str(req.rag_profile_id or "").strip() or None
    if linked_rag_profile_id and not rag_profiles.has_profile(linked_rag_profile_id):
        raise HTTPException(status_code=404, detail=f"Unknown RAG profile id: {linked_rag_profile_id}")

    status = await asyncio.to_thread(model_profiles.link_rag_profile, req.profile_id, linked_rag_profile_id)
    active_profile = status.get("active_profile") or {}
    active_id = str(status.get("active_profile_id", "")).strip()
    if active_id and active_id == str(req.profile_id):
        selected_link = str(active_profile.get("rag_profile_id", "")).strip()
        if selected_link and rag_profiles.has_profile(selected_link):
            await asyncio.to_thread(rag_profiles.activate_profile, selected_link)
        else:
            await asyncio.to_thread(rag_profiles.clear_active_profile)

    rag_status = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse({"models": status, "rag_profiles": rag_status})


@app.get("/api/rag/profiles")
async def rag_profiles_status() -> JSONResponse:
    payload = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(payload)


@app.post("/api/rag/profiles/create")
async def rag_profiles_create(req: RagProfileCreateRequest) -> JSONResponse:
    await asyncio.to_thread(
        rag_profiles.create_profile,
        req.name,
        req.description,
        req.source_paths,
        req.set_active,
    )
    payload = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(payload)


@app.post("/api/rag/profiles/update")
async def rag_profiles_update(req: RagProfileUpdateRequest) -> JSONResponse:
    await asyncio.to_thread(
        rag_profiles.update_profile,
        req.profile_id,
        req.name,
        req.description,
        req.source_paths,
    )
    payload = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(payload)


@app.post("/api/rag/profiles/activate")
async def rag_profiles_activate(req: RagProfileActivateRequest) -> JSONResponse:
    await asyncio.to_thread(rag_profiles.activate_profile, req.profile_id)
    payload = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(payload)


@app.post("/api/rag/profiles/remove")
async def rag_profiles_remove(req: RagProfileRemoveRequest) -> JSONResponse:
    await asyncio.to_thread(rag_profiles.remove_profile, req.profile_id)
    model_status = await asyncio.to_thread(model_profiles.clear_rag_profile_links, req.profile_id)
    rag_status = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse({"rag_profiles": rag_status, "models": model_status})


@app.post("/api/rag/profiles/sources")
async def rag_profiles_sources(req: RagProfileSourcesRequest) -> JSONResponse:
    await asyncio.to_thread(rag_profiles.set_sources, req.profile_id, req.paths, req.mode)
    payload = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(payload)


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
        def _small_text_chunks(text: str, chunk_chars: int = 28) -> list[str]:
            content = str(text or "")
            if not content:
                return []
            return [content[idx : idx + chunk_chars] for idx in range(0, len(content), chunk_chars)]

        async def _emit_text_deltas(
            text: str,
            event_id: str,
            usage: Optional[dict[str, Any]] = None,
        ) -> Any:
            parts = _small_text_chunks(text)
            for part in parts:
                chunk_payload = {
                    "id": event_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": part}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk_payload, ensure_ascii=True)}\n\n"

            finish_payload: dict[str, Any] = {
                "id": event_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if usage is not None:
                finish_payload["usage"] = usage
            yield f"data: {json.dumps(finish_payload, ensure_ascii=True)}\n\n"

        async def _recover_completion_response() -> Optional[dict[str, Any]]:
            # Retry non-stream completion with auto-start between attempts.
            for attempt in range(2):
                try:
                    if not _llama_healthcheck(timeout_sec=0.8):
                        _auto_start_llama_server(reason=f"stream-recovery-{attempt + 1}")
                    _ensure_server_running()
                    recovered = await _upstream_chat_completion(payload)
                    if isinstance(recovered, dict):
                        return recovered
                except Exception as recovery_exc:
                    LOGGER.warning(
                        "Stream recovery failed (attempt %s/2): %s",
                        attempt + 1,
                        _http_error_detail(recovery_exc),
                    )
                    await asyncio.sleep(0.35)
            return None

        if retrieval_chunks:
            data = json.dumps({"retrieval": retrieval_chunks}, ensure_ascii=True)
            yield f"event: retrieval\ndata: {data}\n\n"

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        url = f"{llama_server.base_url}/v1/chat/completions"
        streamed_text = ""
        usage_payload: Optional[dict[str, Any]] = None

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, json=stream_payload) as response:
                    if response.status_code >= 400:
                        details_raw = (await response.aread()).decode("utf-8", errors="replace")
                        details = re.sub(r"\s+", " ", str(details_raw or "")).strip()
                        if not details:
                            details = f"llama-server returned HTTP {response.status_code} with an empty error body."
                        error_data = json.dumps({"error": details}, ensure_ascii=True)
                        yield f"event: error\ndata: {error_data}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break

                        try:
                            parsed_piece = json.loads(data)
                        except json.JSONDecodeError:
                            yield f"data: {data}\n\n"
                            continue

                        choice = (parsed_piece.get("choices") or [{}])[0]
                        delta = (choice.get("delta") or {}).get("content", "")
                        if delta:
                            streamed_text += str(delta)

                        usage_value = parsed_piece.get("usage")
                        if isinstance(usage_value, dict):
                            usage_payload = usage_value

                        yield f"data: {json.dumps(parsed_piece, ensure_ascii=True)}\n\n"

                if retrieval_chunks:
                    if not streamed_text.strip():
                        fallback = _build_extractive_rag_response(_extract_latest_user_query(payload), retrieval_chunks)
                        fallback_content = _extract_assistant_text(fallback).strip()
                        if fallback_content:
                            async for chunk_data in _emit_text_deltas(
                                fallback_content,
                                "bitnet-rag-empty-fallback",
                                usage_payload,
                            ):
                                yield chunk_data
                        yield "data: [DONE]\n\n"
                        return

                    if _looks_like_rag_refusal(streamed_text):
                        fallback = _build_extractive_rag_response(_extract_latest_user_query(payload), retrieval_chunks)
                        fallback_content = _extract_assistant_text(fallback).strip()
                        if fallback_content:
                            correction = f"\n\nUsing indexed context:\n{fallback_content}"
                            async for chunk_data in _emit_text_deltas(correction, "bitnet-rag-correction", usage_payload):
                                yield chunk_data

                yield "data: [DONE]\n\n"
            except httpx.HTTPError as exc:
                LOGGER.info("Primary stream path interrupted: %s", _http_error_detail(exc))

                recovered_stream = False
                recovered_text = ""
                recovered_usage: Optional[dict[str, Any]] = None
                try:
                    if not _llama_healthcheck(timeout_sec=0.8):
                        _auto_start_llama_server(reason="stream-retry")
                    _ensure_server_running()

                    async with httpx.AsyncClient(timeout=None) as retry_client:
                        async with retry_client.stream("POST", url, json=stream_payload) as retry_stream_response:
                            if retry_stream_response.status_code >= 400:
                                retry_details = (await retry_stream_response.aread()).decode("utf-8", errors="replace")
                                LOGGER.info(
                                    "Stream retry HTTP %s: %s",
                                    retry_stream_response.status_code,
                                    re.sub(r"\s+", " ", str(retry_details or "")).strip(),
                                )
                            else:
                                async for line in retry_stream_response.aiter_lines():
                                    if not line or not line.startswith("data:"):
                                        continue
                                    data = line[5:].strip()
                                    if not data:
                                        continue
                                    if data == "[DONE]":
                                        break
                                    try:
                                        parsed_piece = json.loads(data)
                                    except json.JSONDecodeError:
                                        continue

                                    choice = (parsed_piece.get("choices") or [{}])[0]
                                    delta = (choice.get("delta") or {}).get("content", "")
                                    if delta:
                                        recovered_text += str(delta)
                                    usage_value = parsed_piece.get("usage")
                                    if isinstance(usage_value, dict):
                                        recovered_usage = usage_value
                                    yield f"data: {json.dumps(parsed_piece, ensure_ascii=True)}\n\n"
                                    recovered_stream = True
                except Exception as retry_stream_exc:
                    LOGGER.info("Stream retry failed: %s", _http_error_detail(retry_stream_exc))

                if recovered_stream:
                    if retrieval_chunks:
                        if not recovered_text.strip():
                            fallback = _build_extractive_rag_response(_extract_latest_user_query(payload), retrieval_chunks)
                            fallback_content = _extract_assistant_text(fallback).strip()
                            if fallback_content:
                                async for chunk_data in _emit_text_deltas(
                                    fallback_content,
                                    "bitnet-rag-empty-fallback-retry",
                                    recovered_usage,
                                ):
                                    yield chunk_data
                        elif _looks_like_rag_refusal(recovered_text):
                            fallback = _build_extractive_rag_response(_extract_latest_user_query(payload), retrieval_chunks)
                            fallback_content = _extract_assistant_text(fallback).strip()
                            if fallback_content:
                                correction = f"\n\nUsing indexed context:\n{fallback_content}"
                                async for chunk_data in _emit_text_deltas(
                                    correction,
                                    "bitnet-rag-correction-retry",
                                    recovered_usage,
                                ):
                                    yield chunk_data
                    yield "data: [DONE]\n\n"
                    return

                retry_response = await _recover_completion_response()
                if isinstance(retry_response, dict):
                    retry_content = _extract_assistant_text(retry_response)
                    if retrieval_chunks and _looks_like_rag_refusal(retry_content):
                        fallback = _build_extractive_rag_response(_extract_latest_user_query(payload), retrieval_chunks)
                        retry_content = _extract_assistant_text(fallback)
                    if retry_content.strip():
                        usage = retry_response.get("usage") if isinstance(retry_response.get("usage"), dict) else None
                        async for chunk_data in _emit_text_deltas(retry_content, "bitnet-stream-recovered", usage):
                            yield chunk_data
                        yield "data: [DONE]\n\n"
                        return

                if retrieval_chunks:
                    fallback = _build_extractive_rag_response(_extract_latest_user_query(payload), retrieval_chunks)
                    fallback_content = _extract_assistant_text(fallback)
                    if fallback_content.strip():
                        async for chunk_data in _emit_text_deltas(fallback_content, "bitnet-stream-rag-fallback"):
                            yield chunk_data
                        yield "data: [DONE]\n\n"
                        return

                error_data = json.dumps({"error": _http_error_detail(exc)}, ensure_ascii=True)
                yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/rag/status")
async def rag_status() -> JSONResponse:
    global_status = await asyncio.to_thread(rag_store.status)
    scope = await asyncio.to_thread(_active_rag_profile_context, False)
    active_profile_id = str(scope.get("id") or "").strip()
    if not active_profile_id:
        return JSONResponse(
            {
                **global_status,
                "profile_scope": {
                    "active_profile_id": None,
                    "active_profile_name": None,
                    "profile_code": None,
                    "upload_dir": None,
                    "source_count": 0,
                    "indexed_source_count": 0,
                    "indexed_chunks": 0,
                },
            }
        )

    source_paths = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]
    source_counts = await asyncio.to_thread(rag_store.source_chunk_counts)
    normalized_counts = {_normalize_path_str(path).lower(): int(count) for path, count in source_counts.items()}
    indexed_source_count = 0
    indexed_chunks = 0
    for source_path in source_paths:
        count = int(normalized_counts.get(_normalize_path_str(source_path).lower(), 0))
        if count > 0:
            indexed_source_count += 1
            indexed_chunks += count

    return JSONResponse(
        {
            **global_status,
            "profile_scope": {
                "active_profile_id": active_profile_id,
                "active_profile_name": str(scope.get("name", "")),
                "profile_code": str(scope.get("code", "")),
                "upload_dir": str(scope.get("upload_dir", "")),
                "source_count": len(source_paths),
                "indexed_source_count": indexed_source_count,
                "indexed_chunks": indexed_chunks,
            },
        }
    )


@app.get("/api/rag/files")
async def rag_files() -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, False)
    active_profile_id = str(scope.get("id") or "").strip()
    if not active_profile_id:
        return JSONResponse(
            {
                "files": [],
                "summary": _rag_summary([]),
                "profile": {
                    "id": None,
                    "name": None,
                    "code": None,
                    "upload_dir": None,
                    "source_count": 0,
                },
            }
        )

    source_paths = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]
    upload_dir = Path(str(scope.get("upload_dir") or "")).expanduser().resolve()
    records = await asyncio.to_thread(_collect_rag_file_records, source_paths, upload_dir)
    return JSONResponse(
        {
            "files": records,
            "summary": _rag_summary(records),
            "profile": {
                "id": active_profile_id,
                "name": str(scope.get("name", "")),
                "code": str(scope.get("code", "")),
                "upload_dir": str(upload_dir),
                "source_count": len(source_paths),
            },
        }
    )


@app.get("/api/rag/file/details")
async def rag_file_details(path: str, chunk_limit: int = 6) -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, False)
    active_profile_id = str(scope.get("id") or "").strip()
    if active_profile_id:
        active_sources = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]
        normalized_sources = {_normalize_path_str(source).lower() for source in active_sources}
        active_upload_dir = Path(str(scope.get("upload_dir") or "")).expanduser().resolve()
        normalized_candidate = _normalize_path_str(path).lower()
        if normalized_candidate not in normalized_sources:
            try:
                candidate = Path(_normalize_path_str(path)).expanduser().resolve()
            except Exception:
                candidate = Path(path)
            if not _is_within_dir(candidate, active_upload_dir):
                raise HTTPException(status_code=404, detail="File is not in the active profile scope.")

    source_counts = await asyncio.to_thread(rag_store.source_chunk_counts)
    source_keys = list(source_counts.keys())
    source_key = _match_source_key(path, source_keys)
    resolved_path = source_key if source_key else _normalize_path_str(path)
    file_path = Path(resolved_path)

    exists = file_path.exists()
    stat = file_path.stat() if exists else None
    chunk_count = int(source_counts.get(source_key, 0)) if source_key else 0

    if source_key:
        chunks = await asyncio.to_thread(rag_store.chunks_for_source, source_key, max(1, chunk_limit))
    elif exists:
        raw_chunks = await asyncio.to_thread(rag_store.extract_chunks_from_file, resolved_path, 220, 40)
        chunks = [{"preview": piece[:420], "length": len(piece.split())} for piece in raw_chunks[: max(1, chunk_limit)]]
    else:
        chunks = []

    status = "indexed" if chunk_count > 0 else ("uploaded" if exists else "missing")
    return JSONResponse(
        {
            "path": resolved_path,
            "name": file_path.name,
            "exists": exists,
            "status": status,
            "chunk_count": chunk_count,
            "size_bytes": int(stat.st_size) if stat else 0,
            "modified_at": float(stat.st_mtime) if stat else 0.0,
            "preview_chunks": chunks,
        }
    )


@app.post("/api/rag/files/remove")
async def rag_files_remove(req: RagFilesRemoveRequest) -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, True)
    active_profile_id = str(scope.get("id") or "").strip()
    active_sources = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]
    allowed_source_keys = {_normalize_path_str(path).lower() for path in active_sources}
    active_upload_dir = Path(str(scope.get("upload_dir") or "")).expanduser().resolve()

    raw_paths = [str(item).strip() for item in req.paths if str(item).strip()]
    if not raw_paths:
        raise HTTPException(status_code=400, detail="No paths provided.")

    normalized_paths: list[str] = []
    seen: set[str] = set()
    for raw in raw_paths:
        normalized = _normalize_path_str(raw)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_paths.append(normalized)

    scoped_paths: list[str] = []
    skipped_files: list[dict[str, str]] = []
    for normalized in normalized_paths:
        key = normalized.lower()
        candidate = Path(normalized).expanduser().resolve()
        in_active_uploads = _is_within_dir(candidate, active_upload_dir)
        if key in allowed_source_keys or in_active_uploads:
            scoped_paths.append(normalized)
        else:
            skipped_files.append({"path": normalized, "reason": "outside_active_profile_scope"})

    if not scoped_paths:
        raise HTTPException(
            status_code=400,
            detail={"error": "No removable paths found in active profile scope.", "skipped_files": skipped_files},
        )

    removed_index = await asyncio.to_thread(rag_store.remove_sources, scoped_paths)
    await asyncio.to_thread(rag_profiles.set_sources, active_profile_id, scoped_paths, "remove")

    deleted_files: list[str] = []
    if req.delete_files:
        for normalized in scoped_paths:
            candidate = Path(normalized).expanduser().resolve()
            if not _is_within_dir(candidate, active_upload_dir):
                skipped_files.append({"path": str(candidate), "reason": "outside_active_uploads"})
                continue
            if not candidate.exists():
                skipped_files.append({"path": str(candidate), "reason": "file_not_found"})
                continue
            if not candidate.is_file():
                skipped_files.append({"path": str(candidate), "reason": "not_a_file"})
                continue
            try:
                candidate.unlink()
                deleted_files.append(str(candidate))
            except Exception as exc:
                skipped_files.append({"path": str(candidate), "reason": f"delete_error:{exc.__class__.__name__}"})

    updated_scope = await asyncio.to_thread(_active_rag_profile_context, False)
    updated_sources = [str(value) for value in updated_scope.get("source_paths", []) if str(value).strip()]
    records = await asyncio.to_thread(_collect_rag_file_records, updated_sources, active_upload_dir)
    rag_profile_status = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(
        {
            "requested_paths": normalized_paths,
            "scoped_paths": scoped_paths,
            "removed_index": removed_index,
            "deleted_files": deleted_files,
            "skipped_files": skipped_files,
            "summary": _rag_summary(records),
            "remaining_files": records,
            "rag_status": rag_store.status(),
            "rag_profiles": rag_profile_status,
        }
    )


@app.post("/api/rag/export-train")
async def rag_export_train(req: RagExportTrainRequest) -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, True)
    active_profile_id = str(scope.get("id") or "").strip()
    active_profile_code = str(scope.get("code") or "").strip()
    active_sources = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]
    active_source_keys = {_normalize_path_str(path).lower() for path in active_sources}
    active_upload_dir = Path(str(scope.get("upload_dir") or "")).expanduser().resolve()

    if not req.paths:
        raise HTTPException(status_code=400, detail="No RAG file paths provided.")

    source_counts = await asyncio.to_thread(rag_store.source_chunk_counts)
    source_keys = list(source_counts.keys())

    rows: list[str] = []
    written_sources: list[str] = []
    skipped_paths: list[dict[str, str]] = []

    for raw_path in req.paths:
        normalized_raw_path = _normalize_path_str(raw_path)
        normalized_raw_key = normalized_raw_path.lower()
        candidate = Path(normalized_raw_path).expanduser().resolve()
        if normalized_raw_key not in active_source_keys and not _is_within_dir(candidate, active_upload_dir):
            skipped_paths.append({"path": normalized_raw_path, "reason": "outside_active_profile_scope"})
            continue

        source_key = _match_source_key(normalized_raw_path, source_keys)
        source_path = source_key if source_key else _normalize_path_str(raw_path)
        chunks: list[str] = []

        if source_key:
            chunks = await asyncio.to_thread(rag_store.chunk_texts_for_source, source_key)
        else:
            candidate = Path(source_path)
            if not candidate.exists():
                skipped_paths.append({"path": source_path, "reason": "not_found"})
                continue
            chunks = await asyncio.to_thread(
                rag_store.extract_chunks_from_file,
                source_path,
                req.chunk_size,
                req.chunk_overlap,
            )

        if not chunks:
            skipped_paths.append({"path": source_path, "reason": "no_chunks"})
            continue

        for chunk in chunks:
            payload = {"text": chunk, "source": source_path}
            rows.append(json.dumps(payload, ensure_ascii=False))
        written_sources.append(source_path)

    if not rows:
        raise HTTPException(status_code=400, detail={"error": "No training rows could be created.", "skipped": skipped_paths})

    dataset_filename = f"{_safe_dataset_stem(req.dataset_name)}.jsonl"
    if req.append:
        dataset_path = TRAIN_DATASETS_DIR / dataset_filename
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    else:
        dataset_path = _unique_path_in_dir(TRAIN_DATASETS_DIR, dataset_filename)
        dataset_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return JSONResponse(
        {
            "dataset_path": str(dataset_path),
            "rows_written": len(rows),
            "source_count": len(set(written_sources)),
            "sources": sorted(set(written_sources)),
            "skipped_paths": skipped_paths,
            "profile_id": active_profile_id,
            "profile_code": active_profile_code,
        }
    )


@app.post("/api/rag/index")
async def rag_index(req: RagIndexRequest) -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, True)
    active_profile_id = str(scope.get("id") or "").strip()
    active_profile_code = str(scope.get("code") or "").strip()
    active_sources = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]

    if not req.paths:
        raise HTTPException(status_code=400, detail="No paths provided.")

    if req.reset and active_sources:
        await asyncio.to_thread(rag_store.remove_sources, active_sources)
        await asyncio.to_thread(rag_profiles.set_sources, active_profile_id, [], "replace")

    normalized_paths: list[str] = []
    seen: set[str] = set()
    for raw in req.paths:
        normalized = _normalize_path_str(raw)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_paths.append(normalized)

    result = await asyncio.to_thread(
        rag_store.index_paths,
        normalized_paths,
        req.chunk_size,
        req.chunk_overlap,
        False,
    )
    await asyncio.to_thread(rag_profiles.set_sources, active_profile_id, normalized_paths, "add")
    return JSONResponse(
        {
            **result,
            "profile_id": active_profile_id,
            "profile_code": active_profile_code,
            "source_paths_added": normalized_paths,
        }
    )


@app.post("/api/rag/upload")
async def rag_upload(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(220),
    chunk_overlap: int = Form(40),
    reset: bool = Form(False),
) -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, True)
    active_profile_id = str(scope.get("id") or "").strip()
    active_profile_code = str(scope.get("code") or "").strip()
    active_sources = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if reset and active_sources:
        await asyncio.to_thread(rag_store.remove_sources, active_sources)
        await asyncio.to_thread(rag_profiles.set_sources, active_profile_id, [], "replace")

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
        target_path = _unique_profile_upload_path(active_profile_code, _sanitize_upload_name(original_name))
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
        False,
    )
    await asyncio.to_thread(rag_profiles.set_sources, active_profile_id, saved_paths, "add")
    return JSONResponse(
        {
            "saved_files": saved_paths,
            "saved_count": len(saved_paths),
            "skipped_files": skipped_files,
            "indexed": index_result,
            "index_skipped_files": index_result.get("skipped_files", []),
            "profile_id": active_profile_id,
            "profile_code": active_profile_code,
            "upload_dir": str(_profile_upload_dir(active_profile_code).resolve()),
        }
    )


@app.post("/api/rag/search")
async def rag_search(req: RagSearchRequest) -> JSONResponse:
    profile_id, source_filter = _resolve_rag_profile_sources(req.profile_id)
    strict_source_filter = profile_id is not None
    result = await asyncio.to_thread(
        rag_store.search,
        req.query,
        req.top_k,
        source_filter if strict_source_filter else None,
    )
    return JSONResponse({"results": result, "profile_id": profile_id})


@app.post("/api/rag/reset")
async def rag_reset() -> JSONResponse:
    scope = await asyncio.to_thread(_active_rag_profile_context, True)
    active_profile_id = str(scope.get("id") or "").strip()
    active_profile_code = str(scope.get("code") or "").strip()
    active_sources = [str(value) for value in scope.get("source_paths", []) if str(value).strip()]
    active_upload_dir = Path(str(scope.get("upload_dir") or "")).expanduser().resolve()

    removed_index = await asyncio.to_thread(rag_store.remove_sources, active_sources) if active_sources else {
        "requested_sources": [],
        "removed_sources": [],
        "missing_sources": [],
        "removed_chunks": 0,
        "total_chunks": rag_store.status().get("total_chunks", 0),
        "updated_at": rag_store.status().get("updated_at", 0.0),
    }
    await asyncio.to_thread(rag_profiles.set_sources, active_profile_id, [], "replace")

    deleted_files: list[str] = []
    skipped_files: list[dict[str, str]] = []
    if active_upload_dir.exists():
        for candidate in sorted(active_upload_dir.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in RAG_UPLOAD_EXTENSIONS:
                continue
            try:
                candidate.unlink()
                deleted_files.append(str(candidate.resolve()))
            except Exception as exc:
                skipped_files.append({"path": str(candidate), "reason": f"delete_error:{exc.__class__.__name__}"})

    records = await asyncio.to_thread(_collect_rag_file_records, [], active_upload_dir)
    rag_profile_status = await asyncio.to_thread(_rag_profiles_status_payload)
    return JSONResponse(
        {
            "profile_id": active_profile_id,
            "profile_code": active_profile_code,
            "removed_index": removed_index,
            "deleted_files": deleted_files,
            "skipped_files": skipped_files,
            "summary": _rag_summary(records),
            "remaining_files": records,
            "rag_profiles": rag_profile_status,
            "rag_status": await asyncio.to_thread(rag_store.status),
        }
    )


@app.get("/api/train/status")
async def train_status() -> JSONResponse:
    status = await asyncio.to_thread(training.status)
    run_id = str(status.get("started_at") or "")
    current_result: Optional[dict[str, Any]] = None
    with TRAIN_CLEANUP_LOCK:
        tracked_run_id = str(train_cleanup_state.get("last_run_id") or "")
        tracked_result = train_cleanup_state.get("result")
        if tracked_run_id == run_id and isinstance(tracked_result, dict):
            current_result = tracked_result

    status_value = str(status.get("status") or "")
    exit_code = status.get("exit_code")
    completed_success = status_value == "completed" and (exit_code == 0 or str(exit_code) == "0")
    if completed_success and run_id and current_result is None:
        job_config_raw = status.get("job_config")
        job_config = job_config_raw if isinstance(job_config_raw, dict) else {}
        cleanup_result = await asyncio.to_thread(_descope_rag_sources_from_training_dataset, job_config)
        with TRAIN_CLEANUP_LOCK:
            train_cleanup_state["last_run_id"] = run_id
            train_cleanup_state["result"] = cleanup_result
        current_result = cleanup_result

    status["rag_descope"] = current_result
    return JSONResponse(status)


@app.get("/api/train/logs")
async def train_logs(limit: int = 200) -> JSONResponse:
    return JSONResponse({"logs": training.state.tail(limit)})


@app.get("/api/train/template")
async def train_template() -> JSONResponse:
    template = {
        "base_model": DEFAULT_TRAIN_BASE_MODEL,
        "model_revision": DEFAULT_TRAIN_MODEL_REVISION,
        "dataset_path": str((ROOT_DIR / "data" / "train.jsonl").resolve()),
        "output_dir": str((ROOT_DIR / "training-output" / "bitnet").resolve()),
    }
    return JSONResponse(
        {
            "default_command_example": [
                sys.executable,
                str((WEBUI_DIR / "train_bitnet.py").resolve()),
                "--base-model",
                template["base_model"],
                "--model-revision",
                template["model_revision"],
                "--dataset-path",
                template["dataset_path"],
                "--output-dir",
                template["output_dir"],
            ],
            "notes": [
                "This is BitNet-native continued training, not PEFT LoRA adapters.",
                "Use a BitNet-compatible base checkpoint (for example tiiuae/Falcon-E-1B-Base with revision prequantized).",
                "Training is allowed only when a non-base model profile is active to preserve the base fallback.",
                "After training, convert/export artifacts to the runtime format you need (GGUF for llama.cpp/bitnet.cpp serving).",
            ],
        }
    )


@app.post("/api/train/upload-dataset")
async def train_upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    filename = _sanitize_upload_name(file.filename or "dataset.jsonl")
    suffix = Path(filename).suffix.lower()
    allowed = TRAIN_STRUCTURED_DATASET_EXTENSIONS | TRAIN_DOCUMENT_UPLOAD_EXTENSIONS
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported training upload format. "
                "Use JSON/JSONL/CSV datasets or documents (.txt/.md/.rst/.log/.doc/.docx/.pdf)."
            ),
        )

    try:
        payload = await file.read()
    finally:
        await file.close()

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    if suffix in TRAIN_STRUCTURED_DATASET_EXTENSIONS:
        target_path = _unique_path_in_dir(TRAIN_DATASETS_DIR, filename)
        target_path.write_bytes(payload)
        return JSONResponse(
            {
                "dataset_path": str(target_path),
                "filename": target_path.name,
                "size_bytes": len(payload),
                "conversion_mode": "passthrough",
            }
        )

    source_path = _unique_path_in_dir(TRAIN_DATASETS_DIR, filename)
    source_path.write_bytes(payload)
    chunks = await asyncio.to_thread(
        rag_store.extract_chunks_from_file,
        str(source_path),
        220,
        40,
    )
    if not chunks:
        try:
            source_path.unlink(missing_ok=True)
        except Exception:
            pass
        pdf_hint = ""
        if suffix == ".pdf":
            pdf_hint = " Install 'pypdf' for best PDF extraction results."
        raise HTTPException(
            status_code=400,
            detail=(
                f"No training text could be extracted from '{filename}'."
                f"{pdf_hint}"
            ),
        )

    rows = [json.dumps({"text": chunk, "source": str(source_path)}, ensure_ascii=False) for chunk in chunks]
    dataset_stem = _safe_dataset_stem(Path(filename).stem or "dataset")
    target_path = _unique_path_in_dir(TRAIN_DATASETS_DIR, f"{dataset_stem}.jsonl")
    target_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return JSONResponse(
        {
            "dataset_path": str(target_path),
            "filename": target_path.name,
            "size_bytes": len(payload),
            "rows": len(rows),
            "source_file": str(source_path),
            "conversion_mode": "document_to_jsonl",
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
    model_status = await asyncio.to_thread(model_profiles.status)
    active_profile = model_status.get("active_profile") or {}
    if not active_profile or bool(active_profile.get("is_base", False)):
        raise HTTPException(
            status_code=400,
            detail=(
                "Training is blocked while Base BitNet is active. "
                "Activate a custom model profile in the Models tab to keep the base fallback unchanged."
            ),
        )

    command = req.command if req.command else _default_train_command(req)
    job_config = req.dict()
    job_config["active_model_profile_id"] = str(model_status.get("active_profile_id", "")).strip() or None
    job_config["active_model_profile_name"] = str(active_profile.get("name", "")).strip() or None
    job_config["active_model_is_base"] = bool(active_profile.get("is_base", False))
    job_config["active_rag_profile_id"] = str(rag_profiles.get_active_profile_id() or "").strip() or None
    try:
        status = await asyncio.to_thread(training.start, command, job_config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = str(status.get("started_at") or "")
    with TRAIN_CLEANUP_LOCK:
        train_cleanup_state["last_run_id"] = ""
        train_cleanup_state["result"] = None
        if run_id:
            train_cleanup_state["last_run_id"] = run_id
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
    base_model = _pick_default_model()
    await asyncio.to_thread(model_profiles.initialize_base, str(base_model) if base_model else None)

    host = os.getenv("BITNET_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.getenv("BITNET_SERVER_PORT", "8080"))
    except ValueError as exc:
        llama_server.state.last_error = f"Invalid BITNET_SERVER_PORT value: {exc}"
        LOGGER.warning(llama_server.state.last_error)
        return
    llama_server.host = host
    llama_server.port = port

    auto_start_value = os.getenv("BITNET_WEBUI_AUTO_START", "1").strip().lower()
    auto_start = auto_start_value not in {"0", "false", "no", "off"}
    if not auto_start:
        LOGGER.info("BITNET_WEBUI_AUTO_START disabled; not auto-starting llama-server.")
        return

    if llama_server.status()["running"]:
        return

    model = None
    active_model_path = await asyncio.to_thread(model_profiles.get_active_model_path)
    if active_model_path:
        candidate = Path(active_model_path).expanduser().resolve()
        if candidate.exists():
            model = candidate

    if model is None:
        model = base_model
    if model is None:
        message = (
            "Auto-start skipped: no GGUF model found. Set BITNET_DEFAULT_MODEL or place a model in ./models."
        )
        llama_server.state.last_error = message
        LOGGER.warning(message)
        return

    try:
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
