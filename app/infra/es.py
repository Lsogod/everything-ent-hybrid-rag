from __future__ import annotations

import hashlib
from typing import Any

from elasticsearch import Elasticsearch

from app.core.config import get_settings


class ElasticsearchStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Elasticsearch(
            self.settings.es_url,
            basic_auth=(self.settings.es_username, self.settings.es_password)
            if self.settings.es_username and self.settings.es_password
            else None,
            verify_certs=self.settings.es_verify_certs,
            request_timeout=self.settings.request_timeout_seconds,
        )

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.settings.es_index):
            return

        content_mapping: dict[str, Any] = {"type": "text"}
        text_analyzer = (self.settings.es_text_analyzer or "").strip()
        search_analyzer = (self.settings.es_search_analyzer or "").strip()
        if text_analyzer:
            content_mapping["analyzer"] = text_analyzer
        if search_analyzer:
            content_mapping["search_analyzer"] = search_analyzer

        mappings = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "content": content_mapping,
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.settings.embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "chunk_id": {"type": "integer"},
                    "mtime": {"type": "date"},
                }
            },
        }
        self.client.indices.create(index=self.settings.es_index, body=mappings)

    def delete_by_file(self, file_path: str) -> None:
        if not self.client.indices.exists(index=self.settings.es_index):
            return
        self.client.delete_by_query(
            index=self.settings.es_index,
            body={"query": {"term": {"file_path": file_path}}},
            conflicts="proceed",
            refresh=True,
        )

    def upsert_chunks(self, file_path: str, file_name: str, mtime: str, docs: list[dict[str, Any]]) -> None:
        operations: list[dict[str, Any]] = []
        for item in docs:
            doc_source = {
                "doc_id": item["doc_id"],
                "file_path": file_path,
                "file_name": file_name,
                "content": item["content"],
                "chunk_id": item["chunk_id"],
                "mtime": mtime,
            }
            vector = item.get("vector")
            if vector is not None:
                doc_source["vector"] = vector
            operations.append({"index": {"_index": self.settings.es_index, "_id": item["doc_id"]}})
            operations.append(doc_source)

        if operations:
            self.client.bulk(operations=operations, refresh=True)

    def keyword_search(self, query: str, top_k: int, file_paths: list[str] | None = None) -> list[dict[str, Any]]:
        file_filter = self._build_file_filter(file_paths)
        query_body: dict[str, Any] = {"match": {"content": {"query": query}}}
        if file_filter:
            query_body = {
                "bool": {
                    "must": [{"match": {"content": {"query": query}}}],
                    "filter": [file_filter],
                }
            }
        response = self.client.search(
            index=self.settings.es_index,
            size=top_k,
            query=query_body,
        )
        return self._extract_hits(response)

    def vector_search(self, vector: list[float], top_k: int, file_paths: list[str] | None = None) -> list[dict[str, Any]]:
        file_filter = self._build_file_filter(file_paths)
        knn_body: dict[str, Any] = {
            "field": "vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": max(top_k * 4, self.settings.es_num_candidates),
        }
        if file_filter:
            knn_body["filter"] = file_filter
        response = self.client.search(
            index=self.settings.es_index,
            knn=knn_body,
            size=top_k,
        )
        return self._extract_hits(response)

    def count_by_file(self, file_path: str) -> int:
        if not self.client.indices.exists(index=self.settings.es_index):
            return 0
        response = self.client.count(
            index=self.settings.es_index,
            query={"term": {"file_path": file_path}},
        )
        return int(response.get("count", 0))

    @staticmethod
    def make_doc_id(file_path: str, chunk_id: int, mtime: float) -> str:
        raw = f"{file_path}:{chunk_id}:{mtime}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _extract_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for idx, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = hit.get("_source", {})
            output.append(
                {
                    "_id": hit.get("_id"),
                    "_rank": idx,
                    "_score": hit.get("_score", 0.0),
                    "file_path": source.get("file_path"),
                    "file_name": source.get("file_name"),
                    "content": source.get("content"),
                    "chunk_id": source.get("chunk_id"),
                    "mtime": source.get("mtime"),
                }
            )
        return output

    @staticmethod
    def _build_file_filter(file_paths: list[str] | None) -> dict[str, Any] | None:
        if not file_paths:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in file_paths:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        if not normalized:
            return None
        return {"terms": {"file_path": normalized}}
