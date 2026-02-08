from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.infra.es import ElasticsearchStore
from app.services.chunker import chunk_text
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
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            self.store.delete_by_file(file_path)
            return {"status": "ok", "chunks": 0, "file_path": file_path}

        vectors = self.embedding.embed_documents(chunks)
        docs = []
        for idx, (content, vector) in enumerate(zip(chunks, vectors)):
            docs.append(
                {
                    "doc_id": ElasticsearchStore.make_doc_id(str(source), idx, mtime),
                    "chunk_id": idx,
                    "content": content,
                    "vector": vector,
                }
            )

        self.store.delete_by_file(file_path)
        self.store.upsert_chunks(
            file_path=str(source),
            file_name=source.name,
            mtime=datetime.fromtimestamp(mtime).isoformat(),
            docs=docs,
        )
        return {"status": "ok", "chunks": len(chunks), "file_path": str(source)}

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
