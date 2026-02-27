from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional
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
    ".doc",
    ".docx",
    ".pdf",
}


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_blob = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_blob)
    texts = [node.text for node in root.iter() if node.tag.endswith("}t") and node.text]
    return " ".join(texts)


def _extract_printable_strings(data: bytes, min_len: int = 24) -> str:
    blob = data.decode("latin1", errors="ignore")
    pattern = rf"[A-Za-z0-9][A-Za-z0-9 \t,.;:'\"!?()\[\]{{}}/\-]{{{max(1, min_len - 1)},}}"
    chunks = re.findall(pattern, blob)
    return "\n".join(chunks)


def _read_doc_text(path: Path) -> str:
    antiword = shutil.which("antiword")
    if antiword:
        try:
            proc = subprocess.run(
                [antiword, str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=30,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
        except Exception:
            pass

    if os.name == "nt":
        try:
            raw = str(path.resolve())
            escaped = raw.replace("'", "''")
            ps_script = (
                "$ErrorActionPreference='Stop'; "
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
                "$word=New-Object -ComObject Word.Application; "
                "$word.Visible=$false; $word.DisplayAlerts=0; "
                f"$doc=$word.Documents.Open('{escaped}',$false,$true); "
                "try { $doc.Content.Text } finally { $doc.Close($false); $word.Quit() }"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=45,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
        except Exception:
            pass

    return _extract_printable_strings(path.read_bytes())


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(text)
        if pages:
            return "\n\n".join(pages)
    except Exception:
        pass

    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        try:
            proc = subprocess.run(
                [pdftotext, "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=45,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
        except Exception:
            pass

    return _extract_printable_strings(path.read_bytes(), min_len=32)


def _read_supported_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx_text(path)
    if suffix == ".doc":
        return _read_doc_text(path)
    if suffix == ".pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize_source_path(path_value: str) -> str:
    try:
        return str(Path(path_value).expanduser().resolve()).lower()
    except Exception:
        return str(Path(path_value)).lower()


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
            scanned_files = 0
            added_files = 0
            added_chunks = 0
            skipped_files: list[dict[str, str]] = []

            existing_keys = {str(item.get("text", "")) for item in self._chunks}
            for raw_path in paths:
                path = Path(raw_path).expanduser().resolve()
                for file_path in self._iter_supported_files(path):
                    scanned_files += 1
                    source = str(file_path)
                    try:
                        text = _read_supported_text(file_path)
                    except Exception as exc:
                        skipped_files.append({"file": source, "reason": f"read_error: {exc.__class__.__name__}"})
                        continue
                    if not text.strip():
                        skipped_files.append({"file": source, "reason": "no_text_extracted"})
                        continue
                    file_added_chunks = 0
                    duplicate_chunks = 0
                    dropped_empty_chunks = 0
                    for i, chunk in enumerate(_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
                        tokens = _tokenize(chunk)
                        if not tokens:
                            dropped_empty_chunks += 1
                            continue
                        dedupe_key = chunk
                        if dedupe_key in existing_keys:
                            duplicate_chunks += 1
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
                        file_added_chunks += 1
                        added_chunks += 1
                    if file_added_chunks > 0:
                        added_files += 1
                    else:
                        if duplicate_chunks > 0:
                            reason = "all_chunks_duplicate"
                        elif dropped_empty_chunks > 0:
                            reason = "no_tokenizable_chunks"
                        else:
                            reason = "no_chunks_created"
                        skipped_files.append({"file": source, "reason": reason})
            self._rebuild_stats()
            self._save()
            return {
                "scanned_files": scanned_files,
                "added_files": added_files,
                "added_chunks": added_chunks,
                "total_chunks": len(self._chunks),
                "updated_at": self._updated_at,
                "skipped_files": skipped_files,
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

    def source_chunk_counts(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for item in self._chunks:
                source = str(item.get("source", ""))
                if not source:
                    continue
                counts[source] = counts.get(source, 0) + 1
            return counts

    def chunks_for_source(self, source: str, limit: int = 6) -> list[dict[str, Any]]:
        with self._lock:
            rows = [item for item in self._chunks if str(item.get("source", "")).lower() == source.lower()]
            rows.sort(key=lambda item: str(item.get("id", "")))
            trimmed = rows[: max(1, limit)]
            return [
                {
                    "id": str(item.get("id", "")),
                    "length": int(item.get("length", 0)),
                    "preview": str(item.get("text", ""))[:420],
                }
                for item in trimmed
            ]

    def chunk_texts_for_source(self, source: str) -> list[str]:
        with self._lock:
            rows = [item for item in self._chunks if str(item.get("source", "")).lower() == source.lower()]
            rows.sort(key=lambda item: str(item.get("id", "")))
            return [str(item.get("text", "")) for item in rows if str(item.get("text", "")).strip()]

    def extract_chunks_from_file(
        self,
        source_path: str,
        chunk_size: int = 220,
        chunk_overlap: int = 40,
    ) -> list[str]:
        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            return []
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return []
        try:
            text = _read_supported_text(path)
        except Exception:
            return []
        if not text.strip():
            return []
        return _chunk_text(text, chunk_size=max(1, int(chunk_size)), chunk_overlap=max(0, int(chunk_overlap)))

    def remove_sources(self, sources: list[str]) -> dict[str, Any]:
        with self._lock:
            requested = [str(item).strip() for item in sources if str(item).strip()]
            if not requested:
                return {
                    "requested_sources": [],
                    "removed_sources": [],
                    "missing_sources": [],
                    "removed_chunks": 0,
                    "total_chunks": len(self._chunks),
                    "updated_at": self._updated_at,
                }

            normalized_targets = {item.lower() for item in requested}
            before_counts: dict[str, int] = {}
            for row in self._chunks:
                source = str(row.get("source", ""))
                if not source:
                    continue
                before_counts[source] = before_counts.get(source, 0) + 1

            kept_chunks: list[dict[str, Any]] = []
            removed_chunks = 0
            removed_sources: set[str] = set()
            for row in self._chunks:
                source = str(row.get("source", ""))
                if source and source.lower() in normalized_targets:
                    removed_chunks += 1
                    removed_sources.add(source)
                    continue
                kept_chunks.append(row)

            self._chunks = kept_chunks
            self._rebuild_stats()
            self._save()

            removed_lower = {item.lower() for item in removed_sources}
            missing_sources = [item for item in requested if item.lower() not in removed_lower]
            return {
                "requested_sources": requested,
                "removed_sources": sorted(removed_sources),
                "missing_sources": missing_sources,
                "removed_chunks": removed_chunks,
                "total_chunks": len(self._chunks),
                "updated_at": self._updated_at,
            }

    def _to_result(self, score: float, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": round(score, 4),
            "source": chunk["source"],
            "text": chunk["text"][:1200],
            "id": chunk["id"],
        }

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if not self._chunks:
                return []
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []

            strict_source_filter = source_filter is not None
            normalized_filter = {
                _normalize_source_path(item)
                for item in (source_filter or [])
                if str(item or "").strip()
            }
            if strict_source_filter and not normalized_filter:
                return []

            if strict_source_filter:
                search_chunks = [
                    chunk
                    for chunk in self._chunks
                    if _normalize_source_path(str(chunk.get("source", ""))) in normalized_filter
                ]
            else:
                search_chunks = list(self._chunks)

            if not search_chunks:
                return []

            local_doc_freq: dict[str, int] = {}
            total_terms = 0
            for chunk in search_chunks:
                tokens = [str(token) for token in chunk.get("tokens", []) if str(token)]
                total_terms += len(tokens)
                for token in set(tokens):
                    local_doc_freq[token] = local_doc_freq.get(token, 0) + 1

            n_docs = len(search_chunks)
            k1 = 1.6
            b = 0.75
            avg_len = (total_terms / n_docs) if n_docs > 0 else 1.0

            scores: list[tuple[float, dict[str, Any]]] = []
            for chunk in search_chunks:
                freq: dict[str, int] = {}
                for token in chunk["tokens"]:
                    freq[token] = freq.get(token, 0) + 1

                score = 0.0
                doc_len = max(1, chunk["length"])
                for token in query_tokens:
                    tf = freq.get(token, 0)
                    if tf == 0:
                        continue
                    df = local_doc_freq.get(token, 0)
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
            fallback_chunks = search_chunks[-limit:]
            return [self._to_result(0.0, chunk) for chunk in reversed(fallback_chunks)]
