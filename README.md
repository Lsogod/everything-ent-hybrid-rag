# Everything-Ent-Hybrid（企业知识库 RAG v1.1）

一个面向生产环境的企业知识库 RAG 系统，采用 FastAPI + Celery + Elasticsearch 混合检索架构。

## v1.1 新增内容

- 新增 RBAC 权限层（`kb:index`、`kb:delete`、`qa:ask`、`chat:read`、`rbac:manage`）
- 新增管理员接口：角色/权限/用户角色管理
- 新增聊天历史查询接口（会话列表 + 消息列表）
- API 启动时自动初始化数据库表与默认 RBAC 数据

## 核心功能

- 基于 FastAPI 的异步 API 网关（请求链路非阻塞）
- 基于 Vite + React 的前端（问答/索引/管理/会话查看）
- Redis + Celery 的异步文档索引流水线
- 文件监听（File Watcher）自动增量索引本地或 NAS 目录
- 混合检索能力：
  - 关键词检索：Elasticsearch BM25
  - 语义检索：Qwen Embedding（云端/本地策略切换）
  - 融合排序：RRF
- 严格基于检索上下文回答，并返回引用信息
- 问答接口支持 SSE 流式输出

## 项目结构

- `/Users/mac/Documents/rag/app/main.py`：FastAPI 启动入口与健康检查
- `/Users/mac/Documents/rag/app/api/router.py`：索引/问答/管理/聊天 API
- `/Users/mac/Documents/rag/app/services/indexer.py`：解析 -> 切分 -> 向量化 -> ES 写入
- `/Users/mac/Documents/rag/app/services/retrieval.py`：BM25 + 向量检索 + RRF 融合
- `/Users/mac/Documents/rag/app/services/rbac.py`：RBAC 初始化与权限服务
- `/Users/mac/Documents/rag/app/services/chat_history.py`：聊天记录持久化与查询
- `/Users/mac/Documents/rag/worker/tasks.py`：异步索引/删除任务
- `/Users/mac/Documents/rag/frontend/src/App.jsx`：前端主页面

## 快速启动（Docker）

1. 准备环境变量：

```bash
cp .env.example .env
```

如果镜像构建较慢，建议保持 `.env` 中默认镜像源配置：

- `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- `NPM_REGISTRY=https://registry.npmmirror.com`

可选基础镜像加速项：

- `PYTHON_IMAGE`：可替换为区域镜像（需包含 `python:3.10-slim`）
- `ES_IMAGE`：Elasticsearch 镜像地址
- `FRONTEND_NODE_IMAGE`：前端 Node 镜像地址

2. 启动全部服务：

```bash
docker compose up --build -d
```

3. 健康检查：

```bash
curl http://localhost:18000/health
```

4. 打开前端：

```bash
open http://localhost:5173
```

5. 查看当前用户权限（默认已自动引导 `admin`）：

```bash
curl http://localhost:18000/api/v1/me/permissions -H "X-User-Id: admin"
```

## 前端开发（本地）

仅启动 Vite 开发服务：

```bash
cd frontend
npm install --registry=https://registry.npmmirror.com
npm run dev
```

默认情况下，Vite 会将 `/api` 与 `/health` 代理到 `http://localhost:18000`。

## 核心 API

- 文件索引：`POST /api/v1/index/file`（需要 `kb:index`）
- 按文件删除索引：`DELETE /api/v1/index/file`（需要 `kb:delete`）
- 问答（SSE）：`POST /api/v1/qa/ask`（需要 `qa:ask`）
- 会话列表：`GET /api/v1/chat/sessions`（需要 `chat:read`）
- 会话消息：`GET /api/v1/chat/sessions/{session_id}/messages`（需要 `chat:read`）
- 下载示例文档：`POST /api/v1/debug/sample-doc`（需要 `kb:index`）
- 文件索引统计：`GET /api/v1/debug/file-stats?file_path=...`（需要 `chat:read`）
- RBAC 初始化：`POST /api/v1/admin/bootstrap`
- 给用户分配角色：`POST /api/v1/admin/users/{user_id}/roles/{role_name}`（需要 `rbac:manage`）

## 请求头说明

- `X-API-Key`：全局 API 鉴权（仅当配置了 `API_KEY` 时生效）
- `X-User-Id`：当 `ACL_ENABLED=true` 时必填

## RAG 可观测调试

问答请求支持 `debug=true`（默认开启）。SSE 会额外返回 `type=trace` 事件，包含：

- `keyword`：BM25 召回结果列表
- `vector`：向量召回结果列表
- `fusion`：RRF 融合后的 Top-K
- `metrics`：overlap/union 等统计指标

说明：`recall_proxy` 是在线工程观测指标（命中数与候选规模比值），用于可视化监控，不等同于离线标注评测中的真实 Recall。

## 云端/本地 Embedding 策略

- `EMBEDDING_MODE=cloud`：使用 DashScope 与 `DASHSCOPE_API_KEY`
- `EMBEDDING_MODE=local`：使用本地兼容 OpenAI 的 Embedding 服务（例如 vLLM）

## Qwen OpenAI 兼容模式

本项目对聊天与向量接口都采用 OpenAI Python SDK 风格调用（`base_url` 方式）。

- Embedding（Qwen3 官方兼容用法）：
  - `EMBEDDING_MODEL_CLOUD=text-embedding-v4`
  - `EMBEDDING_BASE_URL_CLOUD=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `EMBEDDING_DIM=1024`
  - `EMBEDDING_USE_DIMENSIONS=true`
- LLM：
  - `LLM_PROVIDER=qwen`
  - `LLM_MODEL=qwen-plus`
  - `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`

模型与供应商配置仅在后端（`.env`）生效，不在前端暴露。

## 备注

- Elasticsearch 索引会在 API/Worker 启动时自动创建。
- 数据库表会在 API 启动时自动创建。
- 当 `LLM_API_KEY` 为空时，问答接口仍可工作（退化为仅上下文回答模式）。
