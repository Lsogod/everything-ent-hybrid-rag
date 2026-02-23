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
- 评测数据集清单：`GET /api/v1/eval/datasets/manifest`（需要 `chat:read`）
- 评测报表列表：`GET /api/v1/eval/reports?limit=20`（需要 `chat:read`）
- 最新评测报表：`GET /api/v1/eval/reports/latest`（需要 `chat:read`）
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

## DuRetrieval（C-MTEB）公共基线评测

用于离线评估检索能力，输出 `BM25 / Dense / Fusion(RRF)` 三路结果的
`Hit@K / Recall@K / MRR@K / MAP@K / NDCG@K`。

### 先将数据集落盘到项目目录

会在 `data/knowledge/datasets/` 生成：

- `duretrieval_c_mteb/corpus.md`
- `duretrieval_mteb/corpus.md`
- `duretrieval_manifest.json`（前端评测页读取）

```bash
python scripts/materialize_duretrieval_knowledge.py --max-corpus 1200 --max-queries 240
```

1. 安装评测依赖：

```bash
pip install -r eval/requirements.txt
```

2. 跑 BM25 基线（推荐先跑这个）：

```bash
python eval/run_duretrieval_baseline.py --method bm25 --top-k 10 --ks 1,3,5,10
```

3. 跑 Dense 基线（默认 `BAAI/bge-base-zh-v1.5`）：

```bash
python eval/run_duretrieval_baseline.py --method dense --device cuda --batch-size 64 --top-k 10 --ks 1,3,5,10
```

3.1 使用在线 OpenAI-compatible Embedding（读取 `.env`）：

```bash
python eval/run_duretrieval_baseline.py --method dense --dense-backend openai_compatible --top-k 10 --ks 1,3,5,10
```

可选覆盖参数（不走默认 `.env`）：

```bash
python eval/run_duretrieval_baseline.py --method dense --dense-backend openai_compatible --dense-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --dense-model text-embedding-v4 --dense-api-key $DASHSCOPE_API_KEY --dense-expected-dim 1024 --dense-use-dimensions true
```

4. 同时跑两种基线并生成报告（自动包含 `fusion`）：

```bash
python eval/run_duretrieval_baseline.py --method both --top-k 10 --ks 1,3,5,10 --output-dir eval/reports
```

5. 指定 RRF 融合参数（可用于调参）：

```bash
python eval/run_duretrieval_baseline.py --method both --rrf-k 60 --rrf-bm25-weight 1.0 --rrf-dense-weight 1.0 --output-dir eval/reports
```

6. 运行严格 chunk 评测（检索与评估都以 chunk 为单位）：

```bash
python eval/run_duretrieval_baseline.py --method both --eval-granularity chunk --chunk-size 500 --chunk-overlap 50 --chunk-gold-strategy best --chunk-gold-per-doc 1 --top-k 10 --ks 1,3,5,10 --output-dir eval/reports
```

常用参数：

- `--dataset-source auto|c-mteb|mteb`：数据源优先级（默认 `auto`）
- `--max-corpus`、`--max-queries`：快速 smoke test 子集
- `--tokenizer jieba|whitespace`：BM25 分词策略
- `--dense-backend sentence_transformers|openai_compatible`：Dense 召回后端（默认本地模型）
- `--dense-model`：Dense 模型名（`openai_compatible` 模式会优先读 `.env` 中 EMBEDDING_MODEL_*）
- `--dense-base-url`、`--dense-api-key`：OpenAI-compatible embedding 地址和密钥
- `--dense-expected-dim`、`--dense-use-dimensions`：在线 embedding 维度控制
- `--rrf-k`：RRF 融合常数（默认 `60`）
- `--rrf-bm25-weight`、`--rrf-dense-weight`：两路召回的融合权重（默认都为 `1.0`）
- `--eval-granularity doc|chunk`：评测粒度（默认 `doc`）
- `--chunk-size`、`--chunk-overlap`：chunk 切分参数（与在线索引建议保持一致）
- `--chunk-gold-strategy best|all`：chunk 金标准投影策略
- `--chunk-gold-per-doc`：`best` 策略下每个相关 doc 选取的 gold chunk 数

结果文件会输出到 `eval/reports/duretrieval_baseline_*.json`，可直接用于版本前后 A/B 对比。
当 `--method both` 时，报告中会包含：

- `results.bm25`
- `results.dense`
- `results.fusion`（RRF 融合）

当 `--eval-granularity chunk` 时，报告还会包含 `chunk_eval` 字段，记录 chunk 总量、gold 投影参数与 qrels 规模。

### 在 Docker 中运行评测

默认 compose 已新增 `eval` 服务（`profile: eval`），不会影响日常 `api/worker` 启动。

1. 构建评测镜像：

```bash
docker compose --profile eval build eval
```

2. 运行 BM25 基线：

```bash
docker compose --profile eval run --rm eval
```

3. 在 Docker 中落盘 DuRetrieval 数据集：

```bash
docker compose --profile eval run --rm eval python scripts/materialize_duretrieval_knowledge.py --max-corpus 1200 --max-queries 240
```

4. 运行 Dense 基线：

```bash
docker compose --profile eval run --rm eval python eval/run_duretrieval_baseline.py --method dense --device cuda --batch-size 64 --top-k 10 --ks 1,3,5,10
```

5. 运行 BM25 + Dense + Fusion（RRF）完整基线：

```bash
docker compose --profile eval run --rm eval python eval/run_duretrieval_baseline.py --method both --top-k 10 --ks 1,3,5,10 --rrf-k 60 --rrf-bm25-weight 1.0 --rrf-dense-weight 1.0 --output-dir eval/reports
```

### 前端评测页

前端新增了 `DuRetrieval 评测` 页签，直接基于以下本地语料问答：

- `/data/knowledge/datasets/duretrieval_c_mteb/corpus.md`
- `/data/knowledge/datasets/duretrieval_mteb/corpus.md`

页面会调用 `GET /api/v1/eval/datasets/manifest` 同步数据集状态（文件存在、已入库 chunk 数）并可一键跑通索引+问答流程。

同时会调用 `GET /api/v1/eval/reports` 拉取离线基线报表，展示最新一份
`bm25 / dense / fusion` 对比表（`Hit@K / Recall@K / MRR@K / MAP@K / NDCG@K`）与历史报表列表。

## 备注

- Elasticsearch 索引会在 API/Worker 启动时自动创建。
- 数据库表会在 API 启动时自动创建。
- 当 `LLM_API_KEY` 为空时，问答接口仍可工作（退化为仅上下文回答模式）。

## DuRetrieval 数据集数量说明（重要）

在本项目的离线检索评测里，`query 数量` 与 `corpus 数量` 是两个不同维度：

- `queries`：要评测的问题条数（例如先跑 100 条 query）。
- `corpus`：被检索的候选文档总数（例如 100001 篇文档）。
- `qrels_queries`：有标注真值的 query 数量。
- `qrels_pairs`：`query -> relevant_doc_id` 标注对数量。

为什么“只跑 100 query”仍常用全量 corpus：

- 检索评测本质是“从完整候选池里排序”。
- 如果只截取小语料，负样本变少，排序更容易，`Hit@K / Recall@K / MRR@K / MAP@K / NDCG@K` 往往会偏高。
- 为了可比性（与历史版本、与公共基线），建议固定同一份 corpus。

可用以下参数控制规模（用于 smoke test）：

```bash
python scripts/materialize_duretrieval_knowledge.py --max-corpus 1200 --max-queries 240
python eval/run_duretrieval_baseline.py --method both --max-corpus 1200 --max-queries 240 --top-k 10 --ks 1,3,5,10
```

建议在报表中同时记录：`corpus / queries / qrels_queries / qrels_pairs`，避免只看 query 数造成误读。

## 检索评测指标含义

以下指标都基于同一份 qrels（金标准）计算：

- `Hit@K`：Top-K 中是否命中至少 1 个相关文档。
- `Recall@K`：Top-K 覆盖了多少相关文档。
- `MRR@K`：第一个相关文档出现得越靠前，分数越高。
- `MAP@K`：综合考虑多个相关文档在 Top-K 内的整体排序质量。
- `NDCG@K`：考虑位置折损后的排序质量（越靠前权重越高）。
- `cost(s)`：该检索方法在本次评测中的耗时（秒）。

常见解读：

- 看“是否命中”：优先看 `Hit@K`。
- 看“覆盖是否全”：优先看 `Recall@K`。
- 看“排序是否靠前”：优先看 `MRR@K / MAP@K / NDCG@K`。
- 看“效果-成本权衡”：结合 `cost(s)` 一起判断。

当前项目默认是 `doc 级评测`；如切换到 `chunk 级评测`，指标会更严格，数值通常会下降，且需保证切分策略与金标准投影策略一致。
