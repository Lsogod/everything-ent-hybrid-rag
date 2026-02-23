from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.config import get_settings
from app.infra.es import ElasticsearchStore
from app.services.embedding import EmbeddingFactory
from app.services.search import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ElasticsearchStore()
        self.embedding_client = EmbeddingFactory.create()

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        docs, _ = await self.retrieve_with_trace(query)
        return docs

    async def retrieve_with_trace(self, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        total_started = time.perf_counter()
        embedding_started = time.perf_counter()
        query_vector = await asyncio.to_thread(self.embedding_client.embed_query, query)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        search_started = time.perf_counter()
        keyword_task = asyncio.to_thread(
            self._timed_search,
            self.store.keyword_search,
            query,
            self.settings.retrieval_top_k_keyword,
        )
        vector_task = asyncio.to_thread(
            self._timed_search,
            self.store.vector_search,
            query_vector,
            self.settings.retrieval_top_k_vector,
        )
        (keyword_hits, keyword_ms), (vector_hits, vector_ms) = await asyncio.gather(keyword_task, vector_task)
        parallel_search_ms = (time.perf_counter() - search_started) * 1000

        fusion_started = time.perf_counter()
        merged = reciprocal_rank_fusion(keyword_hits, vector_hits, self.settings.rrf_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        final_hits = merged[: self.settings.retrieval_top_k_final]

        total_ms = (time.perf_counter() - total_started) * 1000
        timing_ms = {
            "embedding": round(embedding_ms, 2),
            "keyword_search": round(keyword_ms, 2),
            "vector_search": round(vector_ms, 2),
            "parallel_search_wall": round(parallel_search_ms, 2),
            "fusion": round(fusion_ms, 2),
            "total": round(total_ms, 2),
        }

        trace = self._build_trace(query, query_vector, keyword_hits, vector_hits, merged, final_hits, timing_ms)
        return final_hits, trace

    def _build_trace(
        self,
        query: str,
        query_vector: list[float],
        keyword_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        merged_hits: list[dict[str, Any]],
        final_hits: list[dict[str, Any]],
        timing_ms: dict[str, float],
    ) -> dict[str, Any]:
        keyword_ids = {item.get("_id") for item in keyword_hits if item.get("_id")}
        vector_ids = {item.get("_id") for item in vector_hits if item.get("_id")}
        union_ids = keyword_ids | vector_ids
        overlap_ids = keyword_ids & vector_ids

        keyword_top_k = max(self.settings.retrieval_top_k_keyword, 1)
        vector_top_k = max(self.settings.retrieval_top_k_vector, 1)
        union_count = max(len(union_ids), 1)

        keyword_map = {item.get("_id"): item for item in keyword_hits if item.get("_id")}
        vector_map = {item.get("_id"): item for item in vector_hits if item.get("_id")}

        fused_detailed = self._build_fusion_details(merged_hits, keyword_map, vector_map)
        dropped = [
            item
            for item in fused_detailed
            if item.get("final_rank", 999999) > self.settings.retrieval_top_k_final
        ][:10]

        return {
            "query": query,
            "timing_ms": timing_ms,
            "embedding": {
                "dimension": len(query_vector),
                "mode": self.settings.embedding_mode,
            },
            "keyword": {
                "top_k": self.settings.retrieval_top_k_keyword,
                "count": len(keyword_hits),
                "recall_proxy": round(len(keyword_hits) / keyword_top_k, 4),
                "hits": [self._serialize_hit(item) for item in keyword_hits],
            },
            "vector": {
                "top_k": self.settings.retrieval_top_k_vector,
                "count": len(vector_hits),
                "recall_proxy": round(len(vector_hits) / vector_top_k, 4),
                "hits": [self._serialize_hit(item) for item in vector_hits],
            },
            "fusion": {
                "algorithm": "RRF",
                "rrf_k": self.settings.rrf_k,
                "top_k": self.settings.retrieval_top_k_final,
                "count": len(final_hits),
                "recall_proxy": round(len(final_hits) / union_count, 4),
                "hits": [self._serialize_hit(item) for item in final_hits],
                "detailed_hits": fused_detailed[: self.settings.retrieval_top_k_final],
                "dropped_candidates": dropped,
            },
            "metrics": {
                "keyword_unique": len(keyword_ids),
                "vector_unique": len(vector_ids),
                "union_unique": len(union_ids),
                "overlap_unique": len(overlap_ids),
                "overlap_rate": round(len(overlap_ids) / union_count, 4),
                "fusion_gain": len(union_ids) - len(overlap_ids),
            },
            "flow": {
                "keyword_candidates": len(keyword_hits),
                "vector_candidates": len(vector_hits),
                "union_candidates": len(union_ids),
                "overlap_candidates": len(overlap_ids),
                "final_selected": len(final_hits),
                "dropped_candidates": max(len(union_ids) - len(final_hits), 0),
            },
        }

    @staticmethod
    def _serialize_hit(item: dict[str, Any]) -> dict[str, Any]:
        raw_content = str(item.get("content") or "")
        normalized = raw_content.replace("\n", " ").strip()
        preview = normalized[:180]
        return {
            "id": item.get("_id"),
            "rank": item.get("_rank"),
            "score": item.get("_score"),
            "rrf_score": item.get("rrf_score"),
            "file_name": item.get("file_name"),
            "file_path": item.get("file_path"),
            "chunk_id": item.get("chunk_id"),
            "preview": preview,
            "content": raw_content,
        }

    @staticmethod
    def _timed_search(search_fn, *args) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        hits = search_fn(*args)
        cost_ms = (time.perf_counter() - started) * 1000
        return hits, cost_ms

    def _build_fusion_details(
        self,
        merged_hits: list[dict[str, Any]],
        keyword_map: dict[str, dict[str, Any]],
        vector_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for final_rank, item in enumerate(merged_hits, start=1):
            doc_id = item.get("_id")
            if not doc_id:
                continue

            keyword_hit = keyword_map.get(doc_id)
            vector_hit = vector_map.get(doc_id)

            keyword_rank = keyword_hit.get("_rank") if keyword_hit else None
            vector_rank = vector_hit.get("_rank") if vector_hit else None
            keyword_score = keyword_hit.get("_score") if keyword_hit else None
            vector_score = vector_hit.get("_score") if vector_hit else None
            keyword_rrf = (1.0 / (self.settings.rrf_k + keyword_rank)) if keyword_rank else 0.0
            vector_rrf = (1.0 / (self.settings.rrf_k + vector_rank)) if vector_rank else 0.0

            source = "hybrid"
            if keyword_hit and not vector_hit:
                source = "keyword_only"
            elif vector_hit and not keyword_hit:
                source = "vector_only"

            detail = self._serialize_hit(item)
            detail.update(
                {
                    "final_rank": final_rank,
                    "source": source,
                    "keyword_rank": keyword_rank,
                    "vector_rank": vector_rank,
                    "keyword_score": keyword_score,
                    "vector_score": vector_score,
                    "keyword_rrf": round(keyword_rrf, 8),
                    "vector_rrf": round(vector_rrf, 8),
                    "rrf_score": round(float(item.get("rrf_score", 0.0)), 8),
                }
            )
            details.append(detail)
        return details
