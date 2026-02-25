from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class QueryDoc:
    query_id: str
    text: str


@dataclass(frozen=True)
class ChunkDoc:
    chunk_id: str
    parent_doc_id: str
    chunk_index: int
    text: str


def _load_hf_datasets(
    source: str,
    cache_dir: str | None = None,
    local_dataset_root: str | None = None,
) -> tuple[list[CorpusDoc], list[QueryDoc], dict[str, dict[str, float]], str]:
    if source in {"local-c-mteb", "local-mteb"}:
        local_source = "c_mteb" if source == "local-c-mteb" else "mteb"
        root = local_dataset_root or "data/knowledge/datasets"
        return _load_local_snapshot(local_source, root)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing dependency: datasets. install with `pip install -r eval/requirements.txt`"
        ) from exc

    if source not in {"auto", "c-mteb", "mteb", "local-c-mteb", "local-mteb"}:
        raise ValueError(f"unsupported dataset source: {source}")

    errors: list[str] = []
    if source in {"auto", "c-mteb"}:
        try:
            corpus_ds = load_dataset("C-MTEB/DuRetrieval", split="corpus", cache_dir=cache_dir)
            queries_ds = load_dataset("C-MTEB/DuRetrieval", split="queries", cache_dir=cache_dir)
            qrels_ds = load_dataset("C-MTEB/DuRetrieval-qrels", split="dev", cache_dir=cache_dir)
            return (
                _parse_corpus(corpus_ds),
                _parse_queries(queries_ds),
                _parse_qrels(qrels_ds),
                "C-MTEB/DuRetrieval",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"C-MTEB source failed: {exc}")
            if source == "c-mteb":
                raise RuntimeError("; ".join(errors)) from exc

    if source in {"auto", "mteb"}:
        try:
            corpus_ds = load_dataset("mteb/DuRetrieval", "corpus", split="dev", cache_dir=cache_dir)
            queries_ds = load_dataset("mteb/DuRetrieval", "queries", split="dev", cache_dir=cache_dir)
            qrels_ds = load_dataset("mteb/DuRetrieval", split="dev", cache_dir=cache_dir)
            return (
                _parse_corpus(corpus_ds),
                _parse_queries(queries_ds),
                _parse_qrels(qrels_ds),
                "mteb/DuRetrieval",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"mteb source failed: {exc}")
            if source == "mteb":
                raise RuntimeError("; ".join(errors)) from exc

    raise RuntimeError("failed to load DuRetrieval dataset: " + "; ".join(errors))


def _load_local_snapshot(
    source: str,
    root: str,
) -> tuple[list[CorpusDoc], list[QueryDoc], dict[str, dict[str, float]], str]:
    dataset_dir = Path(root) / f"duretrieval_{source}"
    corpus_path = dataset_dir / "corpus.md"
    queries_path = dataset_dir / "queries.json"
    qrels_path = dataset_dir / "qrels.json"
    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise RuntimeError(
            f"local snapshot missing: {dataset_dir}. "
            "run scripts/materialize_duretrieval_knowledge.py first"
        )

    corpus = _parse_local_corpus(corpus_path.read_text(encoding="utf-8"))
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
    return corpus, queries, qrels, f"local/{source}"


def _parse_local_corpus(raw: str) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    current_id: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_id, buf
        if not current_id:
            return
        text = "\n".join(buf).strip()
        if text:
            docs.append(CorpusDoc(doc_id=current_id, text=text))
        current_id = None
        buf = []

    for line in raw.splitlines():
        if line.startswith("## DOC ") and "id=" in line:
            flush()
            current_id = line.split("id=", 1)[1].strip()
            buf = []
            continue
        if current_id is not None:
            buf.append(line)
    flush()
    return docs


def _pick_key(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row:
            value = str(row[key]).strip()
            if value:
                return value
    raise KeyError(f"none of keys found: {keys}")


def _parse_corpus(dataset_split) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    for row in dataset_split:
        rid = _pick_key(row, ["id", "_id", "doc_id", "pid", "corpus-id"])
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        merged = f"{title}\n{text}".strip() if title else text
        docs.append(CorpusDoc(doc_id=rid, text=merged))
    return docs


def _parse_queries(dataset_split) -> list[QueryDoc]:
    queries: list[QueryDoc] = []
    for row in dataset_split:
        qid = _pick_key(row, ["id", "_id", "query_id", "qid", "query-id"])
        text = str(row.get("text", "") or "").strip()
        queries.append(QueryDoc(query_id=qid, text=text))
    return queries


def _parse_qrels(dataset_split) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for row in dataset_split:
        qid = _pick_key(row, ["qid", "query-id", "query_id"])
        pid = _pick_key(row, ["pid", "corpus-id", "doc_id"])
        score_raw = row.get("score", 1)
        try:
            score = float(score_raw)
        except Exception:  # noqa: BLE001
            score = 1.0
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[pid] = score
    return qrels


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # Keep chunking behavior aligned with online pipeline when app module is available.
    try:
        from app.services.chunker import chunk_text as app_chunk_text

        return app_chunk_text(text, chunk_size, chunk_overlap)
    except Exception:  # noqa: BLE001
        pass

    # Fallback chunker (same sliding-window logic as app chunker).
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def _build_chunk_corpus(
    corpus: list[CorpusDoc],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[ChunkDoc], dict[str, list[str]], dict[str, str]]:
    chunks: list[ChunkDoc] = []
    chunk_ids_by_doc: dict[str, list[str]] = {}
    chunk_text_by_id: dict[str, str] = {}

    for doc in corpus:
        doc_chunks = _chunk_text(doc.text, chunk_size, chunk_overlap)
        if not doc_chunks and doc.text.strip():
            doc_chunks = [doc.text.strip()]

        ids: list[str] = []
        for idx, piece in enumerate(doc_chunks, start=1):
            cid = f"{doc.doc_id}::chunk_{idx}"
            chunk = ChunkDoc(
                chunk_id=cid,
                parent_doc_id=doc.doc_id,
                chunk_index=idx,
                text=piece,
            )
            chunks.append(chunk)
            chunk_text_by_id[cid] = piece
            ids.append(cid)
        chunk_ids_by_doc[doc.doc_id] = ids

    return chunks, chunk_ids_by_doc, chunk_text_by_id


def _cut_dataset(
    corpus: list[CorpusDoc],
    queries: list[QueryDoc],
    qrels: dict[str, dict[str, float]],
    max_corpus: int | None,
    max_queries: int | None,
) -> tuple[list[CorpusDoc], list[QueryDoc], dict[str, dict[str, float]]]:
    if max_corpus and max_corpus > 0:
        corpus = corpus[:max_corpus]

    corpus_ids = {doc.doc_id for doc in corpus}
    filtered_qrels: dict[str, dict[str, float]] = {}
    for qid, rel_docs in qrels.items():
        kept = {pid: score for pid, score in rel_docs.items() if pid in corpus_ids}
        if kept:
            filtered_qrels[qid] = kept

    kept_queries = [q for q in queries if q.query_id in filtered_qrels]
    if max_queries and max_queries > 0:
        kept_queries = kept_queries[:max_queries]

    kept_qids = {q.query_id for q in kept_queries}
    filtered_qrels = {qid: rel for qid, rel in filtered_qrels.items() if qid in kept_qids}
    return corpus, kept_queries, filtered_qrels


def _tokenize(text: str, tokenizer_name: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if tokenizer_name == "jieba":
        try:
            import jieba
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "missing dependency: jieba. install with `pip install -r eval/requirements.txt`"
            ) from exc
        tokens = [tok.strip() for tok in jieba.lcut(cleaned, cut_all=False) if tok.strip()]
        return tokens
    return [tok for tok in cleaned.split() if tok]


def _pick_best_chunks_for_query(
    query_text: str,
    chunk_ids: list[str],
    chunk_text_by_id: dict[str, str],
    tokenizer_name: str,
    per_doc: int,
) -> list[str]:
    if not chunk_ids:
        return []

    q_tokens = set(_tokenize(query_text, tokenizer_name))
    if not q_tokens:
        return chunk_ids[:per_doc]

    scored: list[tuple[int, int, str]] = []
    has_overlap = False
    for order, cid in enumerate(chunk_ids):
        c_tokens = set(_tokenize(chunk_text_by_id.get(cid, ""), tokenizer_name))
        overlap = len(q_tokens & c_tokens)
        if overlap > 0:
            has_overlap = True
        # sort by overlap desc, then original order asc
        scored.append((overlap, -order, cid))

    if not has_overlap:
        return chunk_ids[:per_doc]

    ranked = sorted(scored, key=lambda item: (-item[0], -item[1]))
    return [cid for _, _, cid in ranked[:per_doc]]


def _project_qrels_to_chunks(
    queries: list[QueryDoc],
    doc_qrels: dict[str, dict[str, float]],
    chunk_ids_by_doc: dict[str, list[str]],
    chunk_text_by_id: dict[str, str],
    tokenizer_name: str,
    strategy: str,
    per_doc: int,
) -> dict[str, dict[str, float]]:
    query_text_by_id = {q.query_id: q.text for q in queries}
    chunk_qrels: dict[str, dict[str, float]] = {}

    for qid, rel_docs in doc_qrels.items():
        query_text = query_text_by_id.get(qid, "")
        picked: dict[str, float] = {}
        for doc_id, rel_score in rel_docs.items():
            chunk_ids = chunk_ids_by_doc.get(doc_id, [])
            if not chunk_ids:
                continue

            if strategy == "all":
                selected = chunk_ids
            else:
                selected = _pick_best_chunks_for_query(
                    query_text=query_text,
                    chunk_ids=chunk_ids,
                    chunk_text_by_id=chunk_text_by_id,
                    tokenizer_name=tokenizer_name,
                    per_doc=per_doc,
                )

            for cid in selected:
                picked[cid] = float(rel_score)
        if picked:
            chunk_qrels[qid] = picked
    return chunk_qrels


def _retrieve_bm25(
    corpus: list[CorpusDoc],
    queries: list[QueryDoc],
    top_k: int,
    tokenizer_name: str,
) -> dict[str, list[str]]:
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing dependency: rank-bm25. install with `pip install -r eval/requirements.txt`"
        ) from exc

    tokenized_corpus = [_tokenize(item.text, tokenizer_name) for item in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    pid_by_index = [item.doc_id for item in corpus]

    results: dict[str, list[str]] = {}
    for query in queries:
        q_tokens = _tokenize(query.text, tokenizer_name)
        if not q_tokens:
            results[query.query_id] = []
            continue

        scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float32)
        if scores.size == 0:
            results[query.query_id] = []
            continue

        k = min(top_k, int(scores.shape[0]))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        results[query.query_id] = [pid_by_index[i] for i in idx]
    return results


def _retrieve_dense(
    corpus: list[CorpusDoc],
    queries: list[QueryDoc],
    top_k: int,
    model_name: str,
    batch_size: int,
    device: str | None,
    max_chars: int | None,
) -> dict[str, list[str]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing dependency: sentence-transformers. install with `pip install -r eval/requirements.txt`"
        ) from exc

    model_kwargs: dict[str, Any] = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(model_name, **model_kwargs)

    corpus_texts = _clip_texts([item.text for item in corpus], max_chars)
    corpus_ids = [item.doc_id for item in corpus]
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    query_texts = _clip_texts([item.text for item in queries], max_chars)
    query_ids = [item.query_id for item in queries]
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    results: dict[str, list[str]] = {}
    q_batch = max(1, min(256, batch_size))
    for start in range(0, len(query_embeddings), q_batch):
        q_emb = query_embeddings[start : start + q_batch]
        scores = np.matmul(q_emb, corpus_embeddings.T)
        k = min(top_k, int(scores.shape[1]))
        top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)

        for row_idx in range(sorted_idx.shape[0]):
            query_id = query_ids[start + row_idx]
            doc_ids = [corpus_ids[i] for i in sorted_idx[row_idx].tolist()]
            results[query_id] = doc_ids
    return results


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _clip_texts(texts: list[str], max_chars: int | None) -> list[str]:
    if not max_chars or max_chars <= 0:
        return texts
    clipped: list[str] = []
    for text in texts:
        item = (text or "").strip()
        if len(item) > max_chars:
            item = item[:max_chars]
        clipped.append(item)
    return clipped


def _encode_openai_compatible_embeddings(
    texts: list[str],
    model_name: str,
    base_url: str,
    api_key: str | None,
    batch_size: int,
    expected_dim: int | None,
    use_dimensions: bool,
    timeout_seconds: float,
) -> np.ndarray:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing dependency: openai. install with `pip install -r eval/requirements.txt`"
        ) from exc

    client = OpenAI(
        api_key=api_key or "EMPTY_API_KEY",
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
        max_retries=2,
    )

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        request_payload: dict[str, Any] = {
            "model": model_name,
            "input": batch,
            "encoding_format": "float",
        }
        if use_dimensions and expected_dim:
            request_payload["dimensions"] = expected_dim

        try:
            response = client.embeddings.create(**request_payload)
        except Exception as exc:  # noqa: BLE001
            if use_dimensions and "dimensions" in request_payload:
                request_payload.pop("dimensions", None)
                response = client.embeddings.create(**request_payload)
            else:
                raise RuntimeError(f"openai-compatible embedding request failed: {exc}") from exc

        items = sorted(response.data, key=lambda item: item.index)
        batch_vectors = [item.embedding for item in items]
        if len(batch_vectors) != len(batch):
            raise RuntimeError("embedding response size mismatch")

        if expected_dim:
            for vector in batch_vectors:
                if len(vector) != expected_dim:
                    raise ValueError(
                        f"embedding dimension mismatch: expected={expected_dim}, actual={len(vector)}"
                    )
        vectors.extend(batch_vectors)

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError("invalid embedding matrix shape")
    return _normalize_rows(matrix)


def _retrieve_dense_openai_compatible(
    corpus: list[CorpusDoc],
    queries: list[QueryDoc],
    top_k: int,
    model_name: str,
    base_url: str,
    api_key: str | None,
    batch_size: int,
    expected_dim: int | None,
    use_dimensions: bool,
    timeout_seconds: float,
    max_chars: int | None,
) -> dict[str, list[str]]:
    corpus_texts = _clip_texts([item.text for item in corpus], max_chars)
    corpus_ids = [item.doc_id for item in corpus]
    corpus_embeddings = _encode_openai_compatible_embeddings(
        texts=corpus_texts,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
        expected_dim=expected_dim,
        use_dimensions=use_dimensions,
        timeout_seconds=timeout_seconds,
    )

    query_texts = _clip_texts([item.text for item in queries], max_chars)
    query_ids = [item.query_id for item in queries]
    query_embeddings = _encode_openai_compatible_embeddings(
        texts=query_texts,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
        expected_dim=expected_dim,
        use_dimensions=use_dimensions,
        timeout_seconds=timeout_seconds,
    )

    results: dict[str, list[str]] = {}
    q_batch = max(1, min(256, batch_size))
    for start in range(0, len(query_embeddings), q_batch):
        q_emb = query_embeddings[start : start + q_batch]
        scores = np.matmul(q_emb, corpus_embeddings.T)
        k = min(top_k, int(scores.shape[1]))
        top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)

        for row_idx in range(sorted_idx.shape[0]):
            query_id = query_ids[start + row_idx]
            doc_ids = [corpus_ids[i] for i in sorted_idx[row_idx].tolist()]
            results[query_id] = doc_ids
    return results


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _resolve_openai_dense_config(args) -> dict[str, Any]:
    model = args.dense_model or os.getenv("EMBEDDING_MODEL_CLOUD") or os.getenv("EMBEDDING_MODEL_LOCAL")
    if not model:
        raise ValueError(
            "dense model is required for openai_compatible backend; set --dense-model "
            "or EMBEDDING_MODEL_CLOUD/EMBEDDING_MODEL_LOCAL"
        )

    base_url = (
        args.dense_base_url
        or os.getenv("EMBEDDING_BASE_URL_CLOUD")
        or os.getenv("EMBEDDING_BASE_URL_LOCAL")
        or os.getenv("OPENAI_BASE_URL")
    )
    if not base_url:
        raise ValueError(
            "dense base_url is required for openai_compatible backend; set --dense-base-url "
            "or EMBEDDING_BASE_URL_CLOUD/EMBEDDING_BASE_URL_LOCAL"
        )

    api_key = (
        args.dense_api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("LOCAL_EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    expected_dim = args.dense_expected_dim
    if expected_dim is None:
        dim_raw = (os.getenv("EMBEDDING_DIM") or "").strip()
        expected_dim = int(dim_raw) if dim_raw else None

    if args.dense_use_dimensions == "true":
        use_dimensions = True
    elif args.dense_use_dimensions == "false":
        use_dimensions = False
    else:
        use_dimensions = _env_bool("EMBEDDING_USE_DIMENSIONS", False)

    timeout_seconds = args.dense_timeout
    if timeout_seconds is None:
        timeout_raw = (os.getenv("REQUEST_TIMEOUT_SECONDS") or "").strip()
        timeout_seconds = float(timeout_raw) if timeout_raw else 30.0

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "expected_dim": expected_dim,
        "use_dimensions": use_dimensions,
        "timeout_seconds": timeout_seconds,
    }


def _fuse_rankings_rrf(
    bm25_ranked: dict[str, list[str]],
    dense_ranked: dict[str, list[str]],
    top_k: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
) -> dict[str, list[str]]:
    fused: dict[str, list[str]] = {}
    query_ids = sorted(set(bm25_ranked.keys()) | set(dense_ranked.keys()))

    for qid in query_ids:
        score_map: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ranked.get(qid, []), start=1):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + bm25_weight / (rrf_k + rank)
        for rank, doc_id in enumerate(dense_ranked.get(qid, []), start=1):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)

        ranked_docs = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
        fused[qid] = [doc_id for doc_id, _ in ranked_docs[:top_k]]
    return fused


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run public DuRetrieval (C-MTEB) retrieval baselines and export metrics."
    )
    parser.add_argument(
        "--dataset-source",
        choices=["auto", "c-mteb", "mteb", "local-c-mteb", "local-mteb"],
        default="auto",
    )
    parser.add_argument("--method", choices=["bm25", "dense", "both"], default="both")
    parser.add_argument(
        "--dense-backend",
        choices=["sentence_transformers", "openai_compatible"],
        default="sentence_transformers",
    )
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--device", default=None, help="Dense model device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dense-base-url", default=None)
    parser.add_argument("--dense-api-key", default=None)
    parser.add_argument("--dense-expected-dim", type=int, default=None)
    parser.add_argument("--dense-use-dimensions", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--dense-timeout", type=float, default=None)
    parser.add_argument("--dense-max-chars", type=int, default=4000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-bm25-weight", type=float, default=1.0)
    parser.add_argument("--rrf-dense-weight", type=float, default=1.0)
    parser.add_argument("--eval-granularity", choices=["doc", "chunk"], default="doc")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--chunk-gold-strategy", choices=["best", "all"], default="best")
    parser.add_argument("--chunk-gold-per-doc", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--tokenizer", choices=["jieba", "whitespace"], default="jieba")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-dataset-root", default="data/knowledge/datasets")
    parser.add_argument("--output-dir", default="eval/reports")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive")
    if args.rrf_bm25_weight <= 0 or args.rrf_dense_weight <= 0:
        raise ValueError("rrf weights must be positive")
    if args.dense_expected_dim is not None and args.dense_expected_dim <= 0:
        raise ValueError("dense-expected-dim must be positive")
    if args.dense_timeout is not None and args.dense_timeout <= 0:
        raise ValueError("dense-timeout must be positive")
    if args.dense_max_chars is not None and args.dense_max_chars <= 0:
        raise ValueError("dense-max-chars must be positive when set")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must satisfy 0 <= overlap < chunk-size")
    if args.chunk_gold_per_doc <= 0:
        raise ValueError("chunk-gold-per-doc must be positive")

    ks = _parse_ks(args.ks)
    started = time.perf_counter()

    corpus, queries, qrels, dataset_name = _load_hf_datasets(
        args.dataset_source,
        args.cache_dir,
        args.local_dataset_root,
    )
    corpus, queries, qrels = _cut_dataset(corpus, queries, qrels, args.max_corpus, args.max_queries)

    if not corpus or not queries or not qrels:
        raise RuntimeError(
            "dataset is empty after filtering; adjust max-corpus/max-queries or dataset source options"
        )

    retrieval_corpus: list[CorpusDoc] = corpus
    effective_qrels = qrels
    chunk_stats: dict[str, Any] | None = None
    if args.eval_granularity == "chunk":
        chunk_corpus, chunk_ids_by_doc, chunk_text_by_id = _build_chunk_corpus(
            corpus=corpus,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        retrieval_corpus = [CorpusDoc(doc_id=item.chunk_id, text=item.text) for item in chunk_corpus]
        effective_qrels = _project_qrels_to_chunks(
            queries=queries,
            doc_qrels=qrels,
            chunk_ids_by_doc=chunk_ids_by_doc,
            chunk_text_by_id=chunk_text_by_id,
            tokenizer_name=args.tokenizer,
            strategy=args.chunk_gold_strategy,
            per_doc=args.chunk_gold_per_doc,
        )
        chunk_stats = {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunk_gold_strategy": args.chunk_gold_strategy,
            "chunk_gold_per_doc": args.chunk_gold_per_doc,
            "corpus_docs": len(corpus),
            "corpus_chunks": len(retrieval_corpus),
            "qrels_pairs_doc_level": int(sum(len(item) for item in qrels.values())),
            "qrels_pairs_chunk_level": int(sum(len(item) for item in effective_qrels.values())),
        }

    if not retrieval_corpus or not effective_qrels:
        raise RuntimeError(
            "retrieval corpus or qrels is empty after projection; "
            "adjust chunk parameters/max-corpus/max-queries"
        )

    dense_config: dict[str, Any] | None = None
    dense_model_resolved: str | None = None
    if args.method in {"dense", "both"}:
        if args.dense_backend == "sentence_transformers":
            dense_model_resolved = args.dense_model or "BAAI/bge-base-zh-v1.5"
        else:
            dense_config = _resolve_openai_dense_config(args)
            dense_model_resolved = dense_config["model"]

    report: dict[str, Any] = {
        "dataset": dataset_name,
        "method": args.method,
        "eval_granularity": args.eval_granularity,
        "dense_backend": args.dense_backend if args.method in {"dense", "both"} else None,
        "dense_model": dense_model_resolved if args.method in {"dense", "both"} else None,
        "fusion_method": "rrf" if args.method == "both" else None,
        "rrf_k": args.rrf_k if args.method == "both" else None,
        "rrf_bm25_weight": args.rrf_bm25_weight if args.method == "both" else None,
        "rrf_dense_weight": args.rrf_dense_weight if args.method == "both" else None,
        "top_k": args.top_k,
        "ks": ks,
        "counts": {
            "corpus": len(retrieval_corpus),
            "queries": len(queries),
            "qrels_queries": len(effective_qrels),
            "qrels_pairs": int(sum(len(item) for item in effective_qrels.values())),
        },
        "chunk_eval": chunk_stats,
        "results": {},
    }
    bm25_ranked: dict[str, list[str]] | None = None
    dense_ranked: dict[str, list[str]] | None = None

    if args.method in {"bm25", "both"}:
        bm25_started = time.perf_counter()
        bm25_ranked = _retrieve_bm25(retrieval_corpus, queries, args.top_k, args.tokenizer)
        bm25_metrics = _compute_metrics(bm25_ranked, effective_qrels, ks)
        report["results"]["bm25"] = {
            "metrics": bm25_metrics,
            "cost_seconds": round(time.perf_counter() - bm25_started, 3),
            "tokenizer": args.tokenizer,
        }

    if args.method in {"dense", "both"}:
        dense_started = time.perf_counter()
        if args.dense_backend == "sentence_transformers":
            dense_ranked = _retrieve_dense(
                corpus=retrieval_corpus,
                queries=queries,
                top_k=args.top_k,
                model_name=dense_model_resolved or "BAAI/bge-base-zh-v1.5",
                batch_size=args.batch_size,
                device=args.device,
                max_chars=args.dense_max_chars,
            )
        else:
            if dense_config is None:
                raise RuntimeError("dense config not resolved for openai_compatible backend")
            dense_ranked = _retrieve_dense_openai_compatible(
                corpus=retrieval_corpus,
                queries=queries,
                top_k=args.top_k,
                model_name=dense_config["model"],
                base_url=dense_config["base_url"],
                api_key=dense_config["api_key"],
                batch_size=args.batch_size,
                expected_dim=dense_config["expected_dim"],
                use_dimensions=dense_config["use_dimensions"],
                timeout_seconds=float(dense_config["timeout_seconds"]),
                max_chars=args.dense_max_chars,
            )
        dense_metrics = _compute_metrics(dense_ranked, effective_qrels, ks)
        report["results"]["dense"] = {
            "metrics": dense_metrics,
            "cost_seconds": round(time.perf_counter() - dense_started, 3),
            "backend": args.dense_backend,
            "model": dense_model_resolved,
            "batch_size": args.batch_size,
            "device": args.device,
            "max_chars": args.dense_max_chars,
        }
        if args.dense_backend == "openai_compatible" and dense_config is not None:
            report["results"]["dense"].update(
                {
                    "base_url": dense_config["base_url"],
                    "expected_dim": dense_config["expected_dim"],
                    "use_dimensions": dense_config["use_dimensions"],
                    "timeout_seconds": dense_config["timeout_seconds"],
                    "has_api_key": bool(dense_config["api_key"]),
                }
            )

    if args.method == "both" and bm25_ranked is not None and dense_ranked is not None:
        fusion_started = time.perf_counter()
        fusion_ranked = _fuse_rankings_rrf(
            bm25_ranked=bm25_ranked,
            dense_ranked=dense_ranked,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            bm25_weight=args.rrf_bm25_weight,
            dense_weight=args.rrf_dense_weight,
        )
        fusion_metrics = _compute_metrics(fusion_ranked, effective_qrels, ks)
        report["results"]["fusion"] = {
            "metrics": fusion_metrics,
            "cost_seconds": round(time.perf_counter() - fusion_started, 3),
            "method": "rrf",
            "rrf_k": args.rrf_k,
            "bm25_weight": args.rrf_bm25_weight,
            "dense_weight": args.rrf_dense_weight,
        }

    report["cost_seconds_total"] = round(time.perf_counter() - started, 3)
    report["created_at"] = datetime.utcnow().isoformat() + "Z"

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"duretrieval_baseline_{stamp}.json"
    _write_json(output_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report saved to: {output_path}")


if __name__ == "__main__":
    main()
