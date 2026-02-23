# DuRetrieval Doc Eval (100 Queries, 5k Corpus, Local Dense)

## Purpose
- Evaluate `bm25 + dense + fusion(rrf)` on doc-level gold labels.
- Use local embedding model (`sentence-transformers` backend), no remote embedding token cost.
- Fast validation setup: `max_queries=100`, `max_corpus=5000`.

## Command (Docker)
```bash
docker compose --profile eval run --rm eval python eval/run_duretrieval_baseline.py \
  --method both \
  --eval-granularity doc \
  --dense-backend sentence_transformers \
  --dense-model BAAI/bge-small-zh-v1.5 \
  --max-corpus 5000 \
  --max-queries 100 \
  --top-k 10 \
  --ks 1,3,5,10 \
  --output-dir eval/reports
```

## Latest generated report (this run)
- `eval/reports/duretrieval_baseline_20260223_140855.json`

## Frontend view
1. Open `DuRetrieval 评测` page.
2. Click `刷新基线报表`.
3. Verify latest report appears in `公共基线报表` and `历史报表`.

## Notes
- This is doc-level evaluation (`eval-granularity=doc`), not chunk-level.
- For stricter/full run, remove `--max-corpus`/`--max-queries`.
