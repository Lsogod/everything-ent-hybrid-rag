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

        mappings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "zh_en": {
                            "type": "standard",
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "zh_en"},
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
                "vector": item["vector"],
                "chunk_id": item["chunk_id"],
                "mtime": mtime,
            }
            operations.append({"index": {"_index": self.settings.es_index, "_id": item["doc_id"]}})
            operations.append(doc_source)

        if operations:
            self.client.bulk(operations=operations, refresh=True)

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.settings.es_index,
            size=top_k,
            query={"match": {"content": {"query": query}}},
        )
        return self._extract_hits(response)

    def vector_search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.settings.es_index,
            knn={
                "field": "vector",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": max(top_k * 4, self.settings.es_num_candidates),
            },
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
