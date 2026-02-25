from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch


@dataclass(frozen=True)
class QueryDoc:
    query_id: str
    text: str


def _load_local_snapshot(source: str, root: str) -> tuple[list[QueryDoc], dict[str, dict[str, float]], Path, str]:
    dataset_dir = Path(root) / f"duretrieval_{source}"
    corpus_path = dataset_dir / "corpus.md"
    queries_path = dataset_dir / "queries.json"
    qrels_path = dataset_dir / "qrels.json"
    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise RuntimeError(
            f"local snapshot missing: {dataset_dir}. "
            "run scripts/materialize_duretrieval_knowledge.py first"
        )

    queries_payload = json.loads(queries_path.read_text(encoding="utf-8"))
    qrels_payload = json.loads(qrels_path.read_text(encoding="utf-8"))
    queries = [
        QueryDoc(query_id=str(item["id"]).strip(), text=str(item["text"]).strip())
        for item in queries_payload.get("items", [])
        if str(item.get("id", "")).strip() and str(item.get("text", "")).strip()
    ]
    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_payload.get("items", []):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        rel_docs = {}
        for pid in row.get("relevant_doc_ids", []):
            rid = str(pid).strip()
            if rid:
                rel_docs[rid] = 1.0
        if rel_docs:
            qrels[qid] = rel_docs
    return queries, qrels, corpus_path, f"local/{source}"


def _parse_ks(raw: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        k = int(item)
        if k <= 0:
            continue
        values.append(k)
    if not values:
        raise ValueError("ks must include at least one positive integer")
    return sorted(set(values))


def _cut_queries(
    queries: list[QueryDoc],
    qrels: dict[str, dict[str, float]],
    max_queries: int | None,
) -> tuple[list[QueryDoc], dict[str, dict[str, float]]]:
    kept = [q for q in queries if q.query_id in qrels]
    if max_queries and max_queries > 0:
        kept = kept[:max_queries]
    kept_qids = {q.query_id for q in kept}
    filtered_qrels = {qid: rel for qid, rel in qrels.items() if qid in kept_qids}
    return kept, filtered_qrels


def _build_doc_spans(text: str) -> list[tuple[int, int, str]]:
    pattern = re.compile(r"^## DOC \d+ \| id=(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    spans: list[tuple[int, int, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        doc_id = match.group(1).strip()
        if doc_id:
            spans.append((start, end, doc_id))
    return spans


def _build_chunk_doc_map(corpus_text: str, chunk_size: int, chunk_overlap: int) -> dict[int, list[str]]:
    content = (corpus_text or "").strip()
    if not content:
        return {}
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    doc_spans = _build_doc_spans(content)
    if not doc_spans:
        return {}

    result: dict[int, list[str]] = {}
    step = chunk_size - chunk_overlap
    start = 0
    chunk_id = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk = content[start:end].strip()
        if chunk:
            hit_doc_ids: list[str] = []
            for doc_start, doc_end, doc_id in doc_spans:
                if max(start, doc_start) < min(end, doc_end):
                    hit_doc_ids.append(doc_id)
            if hit_doc_ids:
                result[chunk_id] = hit_doc_ids
            chunk_id += 1
        if end >= len(content):
            break
        start += step
    return result


def _dcg_at_k(scores: list[float], k: int) -> float:
    total = 0.0
    for idx, rel in enumerate(scores[:k], start=1):
        total += rel / math.log2(idx + 1.0)
    return total


def _compute_metrics(
    ranked: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    ks: list[int],
) -> dict[str, float]:
    if not ranked:
        return {}

    metrics: dict[str, float] = {}
    query_ids = sorted(set(ranked.keys()) & set(qrels.keys()))
    if not query_ids:
        return {}

    for k in ks:
        hit = 0.0
        recall = 0.0
        mrr = 0.0
        map_k = 0.0
        ndcg = 0.0

        for qid in query_ids:
            rel_map = qrels[qid]
            rel_docs = set(rel_map.keys())
            ranked_docs = ranked.get(qid, [])[:k]
            relevant_flags = [1 if doc_id in rel_docs else 0 for doc_id in ranked_docs]

            has_hit = any(relevant_flags)
            hit += 1.0 if has_hit else 0.0

            matched = sum(relevant_flags)
            recall += matched / max(len(rel_docs), 1)

            rr = 0.0
            for rank, flag in enumerate(relevant_flags, start=1):
                if flag:
                    rr = 1.0 / rank
                    break
            mrr += rr

            precision_sum = 0.0
            rel_so_far = 0
            for rank, flag in enumerate(relevant_flags, start=1):
                if not flag:
                    continue
                rel_so_far += 1
                precision_sum += rel_so_far / rank
            denom = min(len(rel_docs), k)
            map_k += precision_sum / max(denom, 1)

            gains = [rel_map.get(doc_id, 0.0) for doc_id in ranked_docs]
            ideal = sorted(rel_map.values(), reverse=True)[:k]
            dcg = _dcg_at_k(gains, k)
            idcg = _dcg_at_k(ideal, k)
            ndcg += dcg / idcg if idcg > 0 else 0.0

        denom = max(len(query_ids), 1)
        metrics[f"hit@{k}"] = round(hit / denom, 6)
        metrics[f"recall@{k}"] = round(recall / denom, 6)
        metrics[f"mrr@{k}"] = round(mrr / denom, 6)
        metrics[f"map@{k}"] = round(map_k / denom, 6)
        metrics[f"ndcg@{k}"] = round(ndcg / denom, 6)
    return metrics


def _retrieve_es_bm25(
    client: Elasticsearch,
    index_name: str,
    corpus_file_path: str,
    queries: list[QueryDoc],
    chunk_to_docs: dict[int, list[str]],
    top_k: int,
) -> dict[str, list[str]]:
    ranked: dict[str, list[str]] = {}
    for query in queries:
        response = client.search(
            index=index_name,
            size=top_k,
            query={
                "bool": {
                    "must": [{"match": {"content": {"query": query.text}}}],
                    "filter": [{"term": {"file_path": corpus_file_path}}],
                }
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        docs: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id")
            if not isinstance(chunk_id, int):
                continue
            for doc_id in chunk_to_docs.get(chunk_id, []):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                docs.append(doc_id)
                if len(docs) >= top_k:
                    break
            if len(docs) >= top_k:
                break
        ranked[query.query_id] = docs
    return ranked


def _pick_metric(metrics: dict[str, float], metric_name: str, k: int) -> float:
    value = metrics.get(f"{metric_name}@{k}")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _build_delta(ik_metrics: dict[str, float], std_metrics: dict[str, float], k: int) -> dict[str, float]:
    keys = ["hit", "recall", "mrr", "map", "ndcg"]
    delta: dict[str, float] = {}
    for key in keys:
        ik = _pick_metric(ik_metrics, key, k)
        std = _pick_metric(std_metrics, key, k)
        delta[f"{key}@{k}"] = round(ik - std, 6)
    return delta


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ES BM25 analyzer A/B benchmark (IK vs standard).")
    parser.add_argument("--dataset-source", choices=["c_mteb", "mteb"], default="c_mteb")
    parser.add_argument("--local-dataset-root", default="data/knowledge/datasets")
    parser.add_argument("--es-url", default="http://elasticsearch:9200")
    parser.add_argument("--es-index-ik", default="ent_kb_chunks_ik")
    parser.add_argument("--es-index-standard", default="ent_kb_chunks_standard")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--output-dir", default="eval/reports")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must satisfy 0 <= overlap < chunk-size")

    ks = _parse_ks(args.ks)
    started = time.perf_counter()

    queries, qrels, corpus_path, dataset_name = _load_local_snapshot(args.dataset_source, args.local_dataset_root)
    queries, qrels = _cut_queries(queries, qrels, args.max_queries)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    chunk_to_docs = _build_chunk_doc_map(corpus_text, args.chunk_size, args.chunk_overlap)
    corpus_runtime_path = str(Path("/data/knowledge/datasets") / f"duretrieval_{args.dataset_source}" / "corpus.md")

    if not queries or not qrels:
        raise RuntimeError("no query/qrels available after filtering")
    if not chunk_to_docs:
        raise RuntimeError("chunk to doc mapping is empty")

    client = Elasticsearch(args.es_url)
    results: dict[str, Any] = {}

    ik_started = time.perf_counter()
    ik_ranked = _retrieve_es_bm25(
        client=client,
        index_name=args.es_index_ik,
        corpus_file_path=corpus_runtime_path,
        queries=queries,
        chunk_to_docs=chunk_to_docs,
        top_k=args.top_k,
    )
    ik_metrics = _compute_metrics(ik_ranked, qrels, ks)
    results["ik"] = {
        "metrics": ik_metrics,
        "cost_seconds": round(time.perf_counter() - ik_started, 3),
        "index": args.es_index_ik,
        "analyzer": "ik_max_word/ik_smart",
    }

    std_started = time.perf_counter()
    std_ranked = _retrieve_es_bm25(
        client=client,
        index_name=args.es_index_standard,
        corpus_file_path=corpus_runtime_path,
        queries=queries,
        chunk_to_docs=chunk_to_docs,
        top_k=args.top_k,
    )
    std_metrics = _compute_metrics(std_ranked, qrels, ks)
    results["standard"] = {
        "metrics": std_metrics,
        "cost_seconds": round(time.perf_counter() - std_started, 3),
        "index": args.es_index_standard,
        "analyzer": "standard/standard",
    }

    report = {
        "dataset": dataset_name,
        "method": "bm25",
        "comparison": "es_analyzer_ab",
        "top_k": args.top_k,
        "ks": ks,
        "counts": {
            "queries": len(queries),
            "qrels_queries": len(qrels),
            "qrels_pairs": int(sum(len(item) for item in qrels.values())),
            "chunk_count": len(chunk_to_docs),
        },
        "settings": {
            "es_url": args.es_url,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "corpus_file_path": corpus_runtime_path,
            "es_index_ik": args.es_index_ik,
            "es_index_standard": args.es_index_standard,
        },
        "results": results,
        "comparison_delta": {
            "ik_minus_standard@5": _build_delta(ik_metrics, std_metrics, 5),
            "ik_minus_standard@10": _build_delta(ik_metrics, std_metrics, 10),
        },
    }
    report["cost_seconds_total"] = round(time.perf_counter() - started, 3)
    report["created_at"] = datetime.utcnow().isoformat() + "Z"

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"duretrieval_baseline_{stamp}_es_ab.json"
    _write_json(output_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report saved to: {output_path}")


if __name__ == "__main__":
    main()
