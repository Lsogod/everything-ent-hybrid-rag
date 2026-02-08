# Everything-Ent-Hybrid (Enterprise Knowledge RAG v1.1)

Production-oriented RAG system using FastAPI + Celery + Elasticsearch hybrid retrieval.

## v1.1 Additions

- RBAC permission layer (`kb:index`, `kb:delete`, `qa:ask`, `chat:read`, `rbac:manage`)
- Admin APIs for role/permission/user-role management
- Chat history query APIs (sessions + messages)
- Startup auto-initialize DB tables + default RBAC bootstrap

## Features

- Async API gateway with FastAPI (non-blocking request path)
- React frontend built with Vite (chat/index/admin/session views)
- File indexing pipeline with Redis + Celery workers
- File watcher auto-increment indexing for local/NAS directories
- Hybrid retrieval:
  - Keyword path: BM25 in Elasticsearch
  - Vector path: Qwen embedding (cloud/local strategy)
  - RRF fusion for final ranking
- LLM answer generation with strict context grounding + citation metadata
- SSE streaming response for chat endpoint

## Architecture

- `/Users/mac/Documents/rag/app/main.py`: FastAPI startup and health endpoint
- `/Users/mac/Documents/rag/app/api/router.py`: Index/QA/Admin/Chat APIs
- `/Users/mac/Documents/rag/app/services/indexer.py`: parser -> chunker -> embedding -> ES write pipeline
- `/Users/mac/Documents/rag/app/services/retrieval.py`: BM25 + vector retrieval + RRF
- `/Users/mac/Documents/rag/app/services/rbac.py`: RBAC bootstrap and permission service
- `/Users/mac/Documents/rag/app/services/chat_history.py`: chat persistence and query service
- `/Users/mac/Documents/rag/worker/tasks.py`: async indexing/deletion jobs
- `/Users/mac/Documents/rag/frontend/src/App.jsx`: Vite + React frontend main page

## Quick Start (Docker)

1. Prepare env:

```bash
cp .env.example .env
```

If build is slow, keep the default mirror settings in `.env`:
`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
`NPM_REGISTRY=https://registry.npmmirror.com`

Optional base image mirror:
set `PYTHON_IMAGE` to a regional mirror tag (for example an Aliyun/Tencent mirror that contains `python:3.10-slim`).
Elasticsearch image mirror is configurable via `ES_IMAGE`.
Frontend Node image mirror is configurable via `FRONTEND_NODE_IMAGE`.

2. Start all services:

```bash
docker compose up --build -d
```

3. Check health:

```bash
curl http://localhost:18000/health
```

4. Open frontend:

```bash
open http://localhost:5173
```

5. Check current user permissions (`admin` is auto bootstrapped by default):

```bash
curl http://localhost:18000/api/v1/me/permissions -H "X-User-Id: admin"
```

## Frontend Dev

Run Vite dev server only:

```bash
cd frontend
npm install --registry=https://registry.npmmirror.com
npm run dev
```

By default Vite proxies `/api` and `/health` to `http://localhost:18000`.

## Core APIs

- Index file: `POST /api/v1/index/file` (requires `kb:index`)
- Delete index by file: `DELETE /api/v1/index/file` (requires `kb:delete`)
- Ask question (SSE): `POST /api/v1/qa/ask` (requires `qa:ask`)
- List sessions: `GET /api/v1/chat/sessions` (requires `chat:read`)
- Session messages: `GET /api/v1/chat/sessions/{session_id}/messages` (requires `chat:read`)
- Download sample document: `POST /api/v1/debug/sample-doc` (requires `kb:index`)
- File indexed stats: `GET /api/v1/debug/file-stats?file_path=...` (requires `chat:read`)
- RBAC bootstrap: `POST /api/v1/admin/bootstrap`
- Assign role to user: `POST /api/v1/admin/users/{user_id}/roles/{role_name}` (requires `rbac:manage`)

## Headers

- `X-API-Key`: optional global API auth (when `API_KEY` is configured)
- `X-User-Id`: required when `ACL_ENABLED=true`

## RAG可观测调试

问答请求支持 `debug=true`（默认开启），SSE 会额外返回 `type=trace` 事件，包含：

- `keyword`: BM25 召回命中列表
- `vector`: 向量召回命中列表
- `fusion`: RRF 融合筛选后的 Top-K
- `metrics`: overlap/union 等统计

说明：`recall_proxy` 是工程代理指标（命中数与候选规模的比值），用于在线可视化观测，不等同于离线标注评测中的真实 Recall。

## Cloud/Local Embedding Strategy

- `EMBEDDING_MODE=cloud`: use DashScope endpoint and `DASHSCOPE_API_KEY`
- `EMBEDDING_MODE=local`: use local OpenAI-compatible embedding endpoint (for example vLLM)

## Qwen OpenAI-Compatible Mode

This project now uses OpenAI Python SDK style calls with `base_url` for both chat and embeddings.

- Embedding (Qwen3 official-compatible pattern):
  - `EMBEDDING_MODEL_CLOUD=text-embedding-v4`
  - `EMBEDDING_BASE_URL_CLOUD=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `EMBEDDING_DIM=1024`
  - `EMBEDDING_USE_DIMENSIONS=true`
- LLM:
  - `LLM_PROVIDER=qwen`
  - `LLM_MODEL=qwen-plus`
  - `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`

Model/provider config is backend-only (`.env`), not exposed in frontend UI.

## Notes

- Elasticsearch index auto-creates at API/worker startup.
- DB schema auto-creates at API startup.
- When `LLM_API_KEY` is empty, QA endpoint still works with fallback context-only answer mode.
