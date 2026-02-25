from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.core.config import get_settings
from app.infra.es import ElasticsearchStore
from app.services.embedding import EmbeddingFactory
from app.services.parser import parse_file


class IndexerService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ElasticsearchStore()
        self.embedding = EmbeddingFactory.create()

    def index_file(self, file_path: str) -> dict:
        source = Path(file_path).resolve()
        if not source.exists():
            return {"status": "skipped", "reason": "file_not_found", "file_path": file_path}
        if not source.is_file():
            return {"status": "skipped", "reason": "not_file", "file_path": file_path}
        if not self._is_path_allowed(source):
            return {"status": "skipped", "reason": "path_not_allowed", "file_path": file_path}

        ext = source.suffix.lower()
        if self.settings.allowed_extensions and ext not in self.settings.allowed_extensions:
            return {"status": "skipped", "reason": "extension_not_allowed", "file_path": file_path}

        mtime = source.stat().st_mtime
        if source.stat().st_size > self.settings.max_file_size_mb * 1024 * 1024:
            return {"status": "skipped", "reason": "file_too_large", "file_path": file_path}

        text = parse_file(str(source))
        chunk_iter = self._iter_chunks(text, self.settings.chunk_size, self.settings.chunk_overlap)
        first_chunk = next(chunk_iter, None)
        if not first_chunk:
            self.store.delete_by_file(file_path)
            return {"status": "ok", "chunks": 0, "file_path": file_path}

        self.store.delete_by_file(file_path)
        batch_size = max(1, self.settings.embedding_batch_size)
        indexed_count = 0
        timestamp = datetime.fromtimestamp(mtime).isoformat()
        chunk_batch: list[str] = [first_chunk]
        write_vectors = bool(self.settings.index_with_vectors)

        def flush(batch: list[str], start_index: int) -> int:
            vectors = self.embedding.embed_documents(batch) if write_vectors else [None] * len(batch)
            docs = []
            for offset, (content, vector) in enumerate(zip(batch, vectors)):
                idx = start_index + offset
                payload = {
                    "doc_id": ElasticsearchStore.make_doc_id(str(source), idx, mtime),
                    "chunk_id": idx,
                    "content": content,
                }
                if vector is not None:
                    payload["vector"] = vector
                docs.append(payload)
            self.store.upsert_chunks(
                file_path=str(source),
                file_name=source.name,
                mtime=timestamp,
                docs=docs,
            )
            return len(docs)

        current_start = 0
        for chunk in chunk_iter:
            if len(chunk_batch) >= batch_size:
                indexed_count += flush(chunk_batch, current_start)
                current_start = indexed_count
                chunk_batch = []
            chunk_batch.append(chunk)

        if chunk_batch:
            indexed_count += flush(chunk_batch, current_start)
        return {"status": "ok", "chunks": indexed_count, "file_path": str(source)}

    def delete_file(self, file_path: str) -> dict:
        source = Path(file_path).resolve()
        if not self._is_path_allowed(source):
            return {"status": "skipped", "reason": "path_not_allowed", "file_path": str(source)}
        self.store.delete_by_file(str(source))
        return {"status": "ok", "deleted": str(source)}

    def _is_path_allowed(self, path: Path) -> bool:
        if self.settings.allow_external_paths:
            return True
        root = Path(self.settings.knowledge_root).resolve()
        return path.is_relative_to(root)

    @staticmethod
    def _iter_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")

        content = (text or "").strip()
        if not content:
            return
        step = chunk_size - overlap
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end].strip()
            if chunk:
                yield chunk
            if end >= len(content):
                break
            start += step
