from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    keyword_hits: list[dict[str, Any]], vector_hits: list[dict[str, Any]], rrf_k: int
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict[str, Any]] = {}

    for hit in keyword_hits:
        doc_id = hit.get("_id")
        if not doc_id:
            continue
        rank = hit.get("_rank", 999999)
        scores[doc_id] += 1.0 / (rrf_k + rank)
        docs[doc_id] = hit

    for hit in vector_hits:
        doc_id = hit.get("_id")
        if not doc_id:
            continue
        rank = hit.get("_rank", 999999)
        scores[doc_id] += 1.0 / (rrf_k + rank)
        docs[doc_id] = hit

    merged = []
    for doc_id, score in scores.items():
        item = dict(docs[doc_id])
        item["rrf_score"] = score
        merged.append(item)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged
