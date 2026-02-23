from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from pathlib import PurePosixPath


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QueryDoc:
    query_id: str
    text: str


def _pick_key(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row:
            value = str(row[key]).strip()
            if value:
                return value
    raise KeyError(f"none of keys found: {keys}")


def _load_source(source: str) -> tuple[list[CorpusDoc], list[QueryDoc], dict[str, dict[str, float]], str]:
    from datasets import load_dataset

    if source == "c_mteb":
        corpus_ds = load_dataset("C-MTEB/DuRetrieval", split="corpus")
        queries_ds = load_dataset("C-MTEB/DuRetrieval", split="queries")
        qrels_ds = load_dataset("C-MTEB/DuRetrieval-qrels", split="dev")
        source_name = "C-MTEB/DuRetrieval"
    elif source == "mteb":
        corpus_ds = load_dataset("mteb/DuRetrieval", "corpus", split="dev")
        queries_ds = load_dataset("mteb/DuRetrieval", "queries", split="dev")
        qrels_ds = load_dataset("mteb/DuRetrieval", split="dev")
        source_name = "mteb/DuRetrieval"
    else:
        raise ValueError(f"unsupported source: {source}")

    corpus: list[CorpusDoc] = []
    for row in corpus_ds:
        doc_id = _pick_key(row, ["id", "_id", "doc_id", "pid", "corpus-id"])
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        corpus.append(CorpusDoc(doc_id=doc_id, title=title, text=text))

    queries: list[QueryDoc] = []
    for row in queries_ds:
        query_id = _pick_key(row, ["id", "_id", "query_id", "qid", "query-id"])
        text = str(row.get("text", "") or "").strip()
        queries.append(QueryDoc(query_id=query_id, text=text))

    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_ds:
        query_id = _pick_key(row, ["qid", "query-id", "query_id"])
        doc_id = _pick_key(row, ["pid", "corpus-id", "doc_id"])
        score_raw = row.get("score", 1)
        try:
            score = float(score_raw)
        except Exception:  # noqa: BLE001
            score = 1.0
        if score <= 0:
            continue
        qrels.setdefault(query_id, {})[doc_id] = score

    return corpus, queries, qrels, source_name


def _pick_snapshot(
    corpus: list[CorpusDoc],
    queries: list[QueryDoc],
    qrels: dict[str, dict[str, float]],
    max_corpus: int,
    max_queries: int,
) -> tuple[list[CorpusDoc], list[QueryDoc], dict[str, dict[str, float]]]:
    sampled_corpus = corpus[:max_corpus]
    corpus_ids = {item.doc_id for item in sampled_corpus}

    sampled_qrels: dict[str, dict[str, float]] = {}
    for qid, rel_docs in qrels.items():
        kept = {pid: score for pid, score in rel_docs.items() if pid in corpus_ids}
        if kept:
            sampled_qrels[qid] = kept

    query_map = {q.query_id: q for q in queries}
    sampled_queries: list[QueryDoc] = []
    for qid in sampled_qrels:
        query = query_map.get(qid)
        if query:
            sampled_queries.append(query)
        if len(sampled_queries) >= max_queries:
            break

    kept_qids = {item.query_id for item in sampled_queries}
    sampled_qrels = {qid: rel for qid, rel in sampled_qrels.items() if qid in kept_qids}
    return sampled_corpus, sampled_queries, sampled_qrels


def _to_markdown(corpus: list[CorpusDoc], source_name: str) -> str:
    lines = [
        "# DuRetrieval Snapshot Corpus",
        "",
        f"- source: {source_name}",
        f"- generated_at: {datetime.utcnow().isoformat()}Z",
        "",
    ]
    for idx, item in enumerate(corpus, start=1):
        lines.append(f"## DOC {idx} | id={item.doc_id}")
        if item.title:
            lines.append(f"title: {item.title}")
        lines.append(item.text)
        lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def materialize_one(
    source: str,
    target_root: Path,
    max_corpus: int,
    max_queries: int,
) -> dict[str, Any]:
    corpus, queries, qrels, source_name = _load_source(source)
    sampled_corpus, sampled_queries, sampled_qrels = _pick_snapshot(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        max_corpus=max_corpus,
        max_queries=max_queries,
    )

    source_dir = target_root / f"duretrieval_{source}"
    source_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = source_dir / "corpus.md"
    queries_file = source_dir / "queries.json"
    qrels_file = source_dir / "qrels.json"
    meta_file = source_dir / "meta.json"

    corpus_file.write_text(_to_markdown(sampled_corpus, source_name), encoding="utf-8")
    _write_json(
        queries_file,
        {
            "source": source_name,
            "items": [{"id": item.query_id, "text": item.text} for item in sampled_queries],
        },
    )
    _write_json(
        qrels_file,
        {
            "source": source_name,
            "items": [{"qid": qid, "relevant_doc_ids": sorted(rel.keys())} for qid, rel in sampled_qrels.items()],
        },
    )

    sample_questions = [item.text for item in sampled_queries[:10]]
    meta = {
        "id": f"duretrieval_{source}",
        "title": f"DuRetrieval ({source_name})",
        "source": source_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "corpus_count": len(sampled_corpus),
        "query_count": len(sampled_queries),
        "qrels_count": int(sum(len(rel) for rel in sampled_qrels.values())),
        "corpus_file_path_local": str(corpus_file),
        "corpus_file_path_runtime": str(
            PurePosixPath("/data/knowledge/datasets") / f"duretrieval_{source}" / "corpus.md"
        ),
        "queries_file_path_local": str(queries_file),
        "sample_questions": sample_questions,
    }
    _write_json(meta_file, meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize DuRetrieval snapshots to local knowledge files.")
    parser.add_argument("--output-root", default="data/knowledge/datasets")
    parser.add_argument("--max-corpus", type=int, default=1200)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--sources", default="c_mteb,mteb", help="comma separated: c_mteb,mteb")
    args = parser.parse_args()

    if args.max_corpus <= 0 or args.max_queries <= 0:
        raise ValueError("max-corpus and max-queries must be positive")

    target_root = Path(args.output_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    source_items = [item.strip() for item in args.sources.split(",") if item.strip()]
    if not source_items:
        raise ValueError("no sources selected")

    manifest_items = []
    for source in source_items:
        manifest_items.append(
            materialize_one(
                source=source,
                target_root=target_root,
                max_corpus=args.max_corpus,
                max_queries=args.max_queries,
            )
        )

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_root": str(target_root),
        "items": manifest_items,
    }
    manifest_path = target_root / "duretrieval_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
