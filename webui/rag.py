from __future__ import annotations

import json
import math
import re
import threading
import time
import zipfile
from hashlib import sha1
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SUPPORTED_EXTENSIONS = {
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


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_blob = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_blob)
    texts = [node.text for node in root.iter() if node.tag.endswith("}t") and node.text]
    return " ".join(texts)


def _read_supported_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = words[start:end]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if end >= len(words):
            break
    return chunks


class RagStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._chunks: list[dict[str, Any]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_chunk_len = 0.0
        self._updated_at = 0.0
        self._load()

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        try:
            payload = json.loads(self._db_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._chunks = payload.get("chunks", [])
        self._doc_freq = payload.get("doc_freq", {})
        self._avg_chunk_len = float(payload.get("avg_chunk_len", 0.0))
        self._updated_at = float(payload.get("updated_at", 0.0))

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._chunks,
            "doc_freq": self._doc_freq,
            "avg_chunk_len": self._avg_chunk_len,
            "updated_at": self._updated_at,
        }
        self._db_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _rebuild_stats(self) -> None:
        doc_freq: dict[str, int] = {}
        total_terms = 0
        for item in self._chunks:
            tokens = item["tokens"]
            total_terms += len(tokens)
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._doc_freq = doc_freq
        self._avg_chunk_len = (total_terms / len(self._chunks)) if self._chunks else 0.0
        self._updated_at = time.time()

    def _iter_supported_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in _SUPPORTED_EXTENSIONS else []
        if not path.is_dir():
            return []
        files: list[Path] = []
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_EXTENSIONS:
                files.append(candidate)
        return files

    def index_paths(
        self,
        paths: list[str],
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        reset: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if reset:
                self._chunks = []
            elif self._chunks:
                # Remove duplicate chunk text accumulated from repeated uploads.
                seen_texts: set[str] = set()
                deduped_chunks: list[dict[str, Any]] = []
                for item in self._chunks:
                    text_key = str(item.get("text", ""))
                    if not text_key or text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    deduped_chunks.append(item)
                self._chunks = deduped_chunks
            added_files = 0
            added_chunks = 0

            existing_keys = {str(item.get("text", "")) for item in self._chunks}
            for raw_path in paths:
                path = Path(raw_path).expanduser().resolve()
                for file_path in self._iter_supported_files(path):
                    try:
                        text = _read_supported_text(file_path)
                    except Exception:
                        continue
                    if not text.strip():
                        continue
                    added_files += 1
                    for i, chunk in enumerate(_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
                        tokens = _tokenize(chunk)
                        if not tokens:
                            continue
                        source = str(file_path)
                        dedupe_key = chunk
                        if dedupe_key in existing_keys:
                            continue
                        chunk_hash = sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()[:12]
                        chunk_id = f"{source}:{i}:{chunk_hash}"
                        existing_keys.add(dedupe_key)
                        self._chunks.append(
                            {
                                "id": chunk_id,
                                "source": source,
                                "text": chunk,
                                "tokens": tokens,
                                "length": len(tokens),
                            }
                        )
                        added_chunks += 1
            self._rebuild_stats()
            self._save()
            return {
                "added_files": added_files,
                "added_chunks": added_chunks,
                "total_chunks": len(self._chunks),
                "updated_at": self._updated_at,
            }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._chunks = []
            self._doc_freq = {}
            self._avg_chunk_len = 0.0
            self._updated_at = time.time()
            self._save()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            sources = {item["source"] for item in self._chunks}
            return {
                "total_chunks": len(self._chunks),
                "total_sources": len(sources),
                "avg_chunk_len": self._avg_chunk_len,
                "updated_at": self._updated_at,
            }

    def _to_result(self, score: float, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": round(score, 4),
            "source": chunk["source"],
            "text": chunk["text"][:1200],
            "id": chunk["id"],
        }

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        with self._lock:
            if not self._chunks:
                return []
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []

            n_docs = len(self._chunks)
            k1 = 1.6
            b = 0.75
            avg_len = self._avg_chunk_len if self._avg_chunk_len > 0 else 1.0

            scores: list[tuple[float, dict[str, Any]]] = []
            for chunk in self._chunks:
                freq: dict[str, int] = {}
                for token in chunk["tokens"]:
                    freq[token] = freq.get(token, 0) + 1

                score = 0.0
                doc_len = max(1, chunk["length"])
                for token in query_tokens:
                    tf = freq.get(token, 0)
                    if tf == 0:
                        continue
                    df = self._doc_freq.get(token, 0)
                    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                    denom = tf + k1 * (1 - b + b * (doc_len / avg_len))
                    score += idf * ((tf * (k1 + 1)) / denom)

                if score > 0:
                    scores.append((score, chunk))

            limit = max(1, top_k)
            if scores:
                scores.sort(key=lambda pair: pair[0], reverse=True)
                return [self._to_result(score, chunk) for score, chunk in scores[:limit]]

            # Fallback for low-overlap queries (e.g., "summarize the document"):
            # return newest chunks so the model still gets document context.
            fallback_chunks = self._chunks[-limit:]
            return [self._to_result(0.0, chunk) for chunk in reversed(fallback_chunks)]
