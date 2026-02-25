import { useEffect, useMemo, useState } from 'react';
import './App.css';

const DEFAULT_USER_ID = import.meta.env.VITE_USER_ID || 'admin';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || '';
const THEME_STORAGE_KEY = 'everything-ent-theme';
const DURETRIEVAL_DOCS = [
  {
    id: 'duretrieval_c_mteb',
    title: 'DuRetrieval (C-MTEB) snapshot',
    filePath: '/data/knowledge/datasets/duretrieval_c_mteb/corpus.md',
    prepare: 'local',
    examples: [
      '国家法定节假日共多少天',
      '功和功率的区别',
      '我国古代第一个有伟大成就的爱国诗人是( )',
    ],
  },
  {
    id: 'duretrieval_mteb',
    title: 'DuRetrieval (mteb) snapshot',
    filePath: '/data/knowledge/datasets/duretrieval_mteb/corpus.md',
    prepare: 'local',
    examples: [
      '如何查看好友申请',
      '怎么屏蔽QQ新闻弹窗',
      '宝鸡装修房子多少钱',
    ],
  },
];

const DEMO_DOC_LIBRARY = [
  {
    id: 'fastapi_en',
    title: 'FastAPI 官方 README（英文）',
    filePath: '/data/knowledge/fastapi_official_readme.md',
    prepare: 'remote_sample',
    examples: [
      'What is FastAPI and what are its key features?',
      'How does FastAPI improve developer productivity?',
    ],
  },
  {
    id: 'fastapi_zh',
    title: 'FastAPI 中文首页',
    filePath: '/data/knowledge/fastapi_zh_index.md',
    prepare: 'local',
    examples: [
      'FastAPI 是什么？',
      'FastAPI 文档里提到了哪些学习入口？',
    ],
  },
  {
    id: 'kubernetes_zh',
    title: 'Kubernetes 中文概览',
    filePath: '/data/knowledge/kubernetes_zh_overview.md',
    prepare: 'local',
    examples: [
      'Kubernetes 是什么？',
      'Kubernetes 文档页主要提供了哪些导航入口？',
    ],
  },
  {
    id: 'ant_design_zh',
    title: 'Ant Design 中文 README',
    filePath: '/data/knowledge/ant_design_zh_readme.md',
    prepare: 'local',
    examples: [
      'Ant Design 的核心定位是什么？',
      'Ant Design 适用于哪些场景？',
    ],
  },
  {
    id: 'vue_zh',
    title: 'Vue 中文介绍',
    filePath: '/data/knowledge/vue_zh_introduction.md',
    prepare: 'local',
    examples: [
      'Vue 是什么？',
      'Vue 文档对新手的学习建议是什么？',
    ],
  },
  {
    id: 'elasticsearch_en',
    title: 'Elasticsearch README（英文）',
    filePath: '/data/knowledge/elasticsearch_readme_en.md',
    prepare: 'local',
    examples: [
      'What is Elasticsearch?',
      'What kind of workloads is Elasticsearch optimized for?',
    ],
  },
  {
    id: 'python_en',
    title: 'Python README（英文）',
    filePath: '/data/knowledge/python_readme_en.txt',
    prepare: 'local',
    examples: [
      'What is Python according to the README?',
      'How does Python describe its programming paradigm?',
    ],
  },
  ...DURETRIEVAL_DOCS,
];

const EVAL_DOC_LIBRARY = DURETRIEVAL_DOCS;

const DEFAULT_DOC = DEMO_DOC_LIBRARY[0];
const DEFAULT_QUESTION = DEFAULT_DOC.examples[0];

function createConversationId() {
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function createSteps(query, filePath, docTitle) {
  return [
    {
      id: 'download',
      title: '1. 准备测试文档',
      status: 'idle',
      input: {
        mode: 'local_or_remote',
        endpoint: '/api/v1/debug/sample-doc（仅远程样例文档）',
        doc: docTitle,
        file_path: filePath,
      },
      output: null,
      error: '',
      durationMs: 0,
      explain: {
        what: 'Ensure the selected test document is ready for indexing and retrieval.',
        choose: 'Remote sample files are downloaded first; local files are used directly.',
        done: 'A valid file_path is returned for subsequent steps.',
      },
    },
    {
      id: 'index',
      title: '2. 提交索引任务',
      status: 'idle',
      input: {
        endpoint: '/api/v1/index/file',
        method: 'POST',
        file_path: filePath,
      },
      output: null,
      error: '',
      durationMs: 0,
      explain: {
        what: 'Submit selected document to async indexing queue.',
        choose: 'Only index the selected file, not the whole knowledge base.',
        done: 'Receive task_id, then wait for worker processing.',
      },
    },
    {
      id: 'poll',
      title: '3. 等待入库完成',
      status: 'idle',
      input: {
        endpoint: '/api/v1/debug/file-stats',
        method: 'GET',
        file_path: filePath,
      },
      output: null,
      error: '',
      durationMs: 0,
      explain: {
        what: 'Poll file stats until indexing is complete before QA.',
        choose: 'Poll every second; stop when chunk_count > 0.',
        done: 'indexed=true and chunk_count > 0.',
      },
    },
    {
      id: 'retrieve',
      title: '4. 检索与召回',
      status: 'idle',
      input: {
        endpoint: '/api/v1/qa/ask',
        method: 'POST(SSE)',
        query,
        debug: true,
      },
      output: null,
      error: '',
      durationMs: 0,
      explain: {
        what: 'Run keyword and vector retrieval in parallel, then fuse with RRF.',
        choose: 'Merge candidates and rank by 1/(k+rank), then pick Top-K.',
        done: 'Get fusion hits, candidate flow stats, and timing details.',
      },
    },
    {
      id: 'answer',
      title: '5. 答案生成',
      status: 'idle',
      input: {
        source: 'SSE token stream',
      },
      output: null,
      error: '',
      durationMs: 0,
      explain: {
        what: 'Build LLM messages from Top-K contexts and question.',
        choose: 'Use only retrieved context; do not add external knowledge.',
        done: 'Receive token stream and final answer with llm_input.',
      },
    },
  ];
}

function resolveUrl(baseUrl, path) {
  const base = (baseUrl || '').trim();
  if (!base) return path;
  return `${base.replace(/\/$/, '')}${path}`;
}

function buildHeaders(hasBody = false) {
  const headers = {};
  if (hasBody) headers['Content-Type'] = 'application/json';
  headers['X-User-Id'] = DEFAULT_USER_ID;
  if (DEFAULT_API_KEY) headers['X-API-Key'] = DEFAULT_API_KEY;
  return headers;
}

async function apiGet(baseUrl, path) {
  const res = await fetch(resolveUrl(baseUrl, path), {
    headers: buildHeaders(false),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GET ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function apiPost(baseUrl, path, body) {
  const res = await fetch(resolveUrl(baseUrl, path), {
    method: 'POST',
    headers: buildHeaders(true),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function apiDelete(baseUrl, path, body) {
  const options = {
    method: 'DELETE',
    headers: buildHeaders(Boolean(body)),
  };
  if (body) {
    options.body = JSON.stringify(body);
  }
  const res = await fetch(resolveUrl(baseUrl, path), options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`DELETE ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

function extractSsePayloads(buffer) {
  const payloads = [];
  let remaining = buffer;

  while (true) {
    const splitAt = remaining.indexOf('\n\n');
    if (splitAt === -1) break;

    const frame = remaining.slice(0, splitAt).trim();
    remaining = remaining.slice(splitAt + 2);

    if (!frame) continue;
    const lines = frame.split(/\r?\n/);
    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const raw = line.replace(/^data:\s?/, '').trim();
      if (raw) payloads.push(raw);
    }
  }

  return { payloads, remaining };
}

function formatNumber(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return value.toFixed(4);
}

function formatMs(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return `${value.toFixed(2)} ms`;
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(1)}%`;
}

function formatDateTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function reportTimestamp(item) {
  const ts = Date.parse(item?.created_at || '');
  return Number.isFinite(ts) ? ts : 0;
}

function sortReportsByCreatedAt(items) {
  return [...(items || [])].sort((a, b) => {
    const diff = reportTimestamp(b) - reportTimestamp(a);
    if (diff !== 0) return diff;
    return String(b?.file_name || '').localeCompare(String(a?.file_name || ''));
  });
}

function parseMetricKey(key) {
  if (typeof key !== 'string') return null;
  const parts = key.split('@');
  if (parts.length !== 2) return null;
  const metric = parts[0]?.trim().toLowerCase();
  const k = Number(parts[1]);
  if (!metric || !Number.isFinite(k) || k <= 0) return null;
  return { metric, k };
}

function collectMetricNames(methodMetricsMap) {
  const seen = new Set();
  Object.values(methodMetricsMap || {}).forEach((metrics) => {
    if (!metrics || typeof metrics !== 'object') return;
    Object.keys(metrics).forEach((key) => {
      const parsed = parseMetricKey(key);
      if (parsed) seen.add(parsed.metric);
    });
  });
  const preferredOrder = ['hit', 'recall', 'mrr', 'map', 'ndcg'];
  const ordered = preferredOrder.filter((metric) => seen.has(metric));
  const extras = Array.from(seen).filter((metric) => !preferredOrder.includes(metric)).sort();
  return [...ordered, ...extras];
}

function collectKs(summary) {
  const fromSummary = Array.isArray(summary?.ks)
    ? summary.ks
        .map((item) => Number(item))
        .filter((item) => Number.isFinite(item) && item > 0)
    : [];
  const fromMetrics = [];
  const methodMetricsMap = summary?.metrics || {};
  Object.values(methodMetricsMap).forEach((metrics) => {
    if (!metrics || typeof metrics !== 'object') return;
    Object.keys(metrics).forEach((key) => {
      const parsed = parseMetricKey(key);
      if (parsed) fromMetrics.push(parsed.k);
    });
  });
  // Always render k=10 so users can quickly verify whether @10 exists for the selected report.
  return Array.from(new Set([...fromSummary, ...fromMetrics, 10])).sort((a, b) => a - b);
}

function getMetricValue(metrics, metric, k) {
  if (!metrics || typeof metrics !== 'object') return null;
  const exact = metrics[`${metric}@${k}`];
  return typeof exact === 'number' ? exact : null;
}

function formatMetricValue(value) {
  return typeof value === 'number' && !Number.isNaN(value) ? value.toFixed(4) : '-';
}

function statusLabel(status) {
  if (status === 'done') return '已完成';
  if (status === 'running') return '执行中';
  if (status === 'error') return '失败';
  return '等待';
}

function StepCard({ step }) {
  const output = step.error
    ? { error: step.error }
    : step.output || { message: '等待执行' };

  return (
    <article className={`step-card step-${step.status}`}>
      <header>
        <h3>{step.title}</h3>
        <div className="step-meta">
          <span>{statusLabel(step.status)}</span>
          <span>{step.durationMs ? `${step.durationMs} ms` : '-'}</span>
        </div>
      </header>
      {!!step.explain && (
        <section className="step-explain">
          <p><strong>做什么：</strong>{step.explain.what}</p>
          <p><strong>如何选择：</strong>{step.explain.choose}</p>
          <p><strong>完成标准：</strong>{step.explain.done}</p>
        </section>
      )}
      <div className="step-io">
        <section>
          <p>输入</p>
          <pre>{JSON.stringify(step.input, null, 2)}</pre>
        </section>
        <section>
          <p>输出</p>
          <pre>{JSON.stringify(output, null, 2)}</pre>
        </section>
      </div>
    </article>
  );
}

function HitList({ title, hits, onOpenChunk }) {
  return (
    <section className="trace-block">
      <h4>{title}</h4>
      {!hits?.length && <p className="muted">无命中</p>}
      {!!hits?.length && (
        <ul>
          {hits.slice(0, 5).map((hit) => (
            <li key={`${title}-${hit.id}-${hit.rank}`}>
              <div>
                <strong>#{hit.rank}</strong>
                <span>{hit.file_name || hit.file_path || hit.id}</span>
              </div>
              <small>score={formatNumber(hit.score)} rrf={formatNumber(hit.rrf_score)}</small>
              <p>{hit.preview}</p>
              {!!hit.content && (
                <button type="button" className="inline-btn" onClick={() => onOpenChunk(title, hit)}>
                  查看详情
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function TimingList({ timing }) {
  if (!timing) return <p className="muted">无耗时数据</p>;
  const rows = [
    ['embedding', timing.embedding],
    ['keyword_search', timing.keyword_search],
    ['vector_search', timing.vector_search],
    ['parallel_search_wall', timing.parallel_search_wall],
    ['fusion', timing.fusion],
    ['total', timing.total],
  ];
  return (
    <ul className="kv-list">
      {rows.map(([k, v]) => (
        <li key={k}>
          <span>{k}</span>
          <strong>{formatMs(v)}</strong>
        </li>
      ))}
    </ul>
  );
}

function FlowList({ flow }) {
  if (!flow) return <p className="muted">无候选流转数据</p>;
  const rows = [
    ['keyword_candidates', flow.keyword_candidates],
    ['vector_candidates', flow.vector_candidates],
    ['union_candidates', flow.union_candidates],
    ['overlap_candidates', flow.overlap_candidates],
    ['final_selected', flow.final_selected],
    ['dropped_candidates', flow.dropped_candidates],
  ];
  return (
    <ul className="kv-list">
      {rows.map(([k, v]) => (
        <li key={k}>
          <span>{k}</span>
          <strong>{v ?? '-'}</strong>
        </li>
      ))}
    </ul>
  );
}

function FusionTable({ rows, onOpenChunk, sourceTitle }) {
  if (!rows?.length) return <p className="muted">无融合明细</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>final</th>
            <th>src</th>
            <th>k_rank</th>
            <th>v_rank</th>
            <th>k_rrf</th>
            <th>v_rrf</th>
            <th>rrf_total</th>
            <th>chunk</th>
            <th>detail</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`fusion-${row.id}-${row.final_rank}`}>
              <td>{row.final_rank}</td>
              <td>{row.source}</td>
              <td>{row.keyword_rank ?? '-'}</td>
              <td>{row.vector_rank ?? '-'}</td>
              <td>{formatNumber(row.keyword_rrf)}</td>
              <td>{formatNumber(row.vector_rrf)}</td>
              <td>{formatNumber(row.rrf_score)}</td>
              <td>{row.chunk_id ?? '-'}</td>
              <td>
                {!!row.content && (
                  <button type="button" className="inline-btn" onClick={() => onOpenChunk(sourceTitle, row)}>
                    查看
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChunkDetailPanel({ chunk, onClose }) {
  useEffect(() => {
    if (!chunk) return undefined;
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [chunk, onClose]);

  if (!chunk) return null;
  return (
    <div className="chunk-modal-backdrop" onClick={onClose}>
      <section className="chunk-modal" onClick={(event) => event.stopPropagation()}>
        <div className="chunk-detail-head">
          <h4>Chunk 详情</h4>
          <button type="button" className="inline-btn" onClick={onClose}>
            关闭
          </button>
        </div>
        <div className="metric-row">
          <span>source: {chunk.source || '-'}</span>
          <span>rank: {chunk.rank ?? '-'}</span>
          <span>chunk_id: {chunk.chunk_id ?? '-'}</span>
        </div>
        <p className="muted">{chunk.file_name || chunk.file_path || chunk.id}</p>
        <pre className="raw-block">{chunk.content || chunk.preview || ''}</pre>
      </section>
    </div>
  );
}

function LlmInputPanel({ llmInput }) {
  if (!llmInput) {
    return (
      <section className="trace-block">
        <h4>传给大模型的内容（完整）</h4>
        <p className="muted">执行流程后显示模型输入 messages 与 contexts 全文。</p>
      </section>
    );
  }

  return (
    <section className="trace-block">
      <h4>传给大模型的内容（完整）</h4>
      <div className="metric-row">
        <span>provider: {llmInput.provider || '-'}</span>
        <span>model: {llmInput.model || '-'}</span>
        <span>base_url: {llmInput.base_url || '-'}</span>
        <span>temperature: {llmInput.temperature ?? '-'}</span>
        <span>context_count: {llmInput.context_count ?? '-'}</span>
      </div>
      <p className="muted">messages（原始请求）</p>
      <pre className="raw-block">{JSON.stringify(llmInput.messages || [], null, 2)}</pre>
      <p className="muted">contexts（送入模型的全文）</p>
      <pre className="raw-block">{JSON.stringify(llmInput.contexts || [], null, 2)}</pre>
    </section>
  );
}

function BaselineReportPanel({ latest, items }) {
  const [selectedFileName, setSelectedFileName] = useState('');
  const baselineItems = useMemo(
    () => sortReportsByCreatedAt((items || []).filter((item) => !item?.file_name?.includes('_es_ab'))),
    [items],
  );

  const effectiveSelectedFileName = useMemo(() => {
    if (!baselineItems.length) return '';
    if (selectedFileName && baselineItems.some((item) => item.file_name === selectedFileName)) {
      return selectedFileName;
    }
    return baselineItems[0].file_name;
  }, [baselineItems, selectedFileName]);

  const activeSummary = useMemo(() => {
    if (baselineItems.length) {
      const matched = baselineItems.find((item) => item.file_name === effectiveSelectedFileName);
      if (matched) return matched;
      return baselineItems[0];
    }
    if (latest && !latest?.file_name?.includes('_es_ab')) {
      return latest;
    }
    return null;
  }, [baselineItems, latest, effectiveSelectedFileName]);

  const methods = activeSummary?.metrics ? Object.keys(activeSummary.metrics) : [];
  const ks = collectKs(activeSummary);
  const metricNames = collectMetricNames(activeSummary?.metrics || {});
  const strictRows = methods.flatMap((method) => {
    const metrics = activeSummary?.metrics?.[method] || {};
    const cost = activeSummary?.method_cost_seconds?.[method];
    return ks.map((k) => ({
      method,
      k,
      cost,
      values: metricNames.reduce((acc, metricName) => {
        acc[metricName] = getMetricValue(metrics, metricName, k);
        return acc;
      }, {}),
    }));
  });

  return (
    <section className="workflow">
      <h2>公共基线报表（严格口径）</h2>
      {!activeSummary && <p className="muted">暂无基线报表，请先运行 eval 脚本。</p>}
      {!!activeSummary && (
        <>
          <div className="metric-row">
            <span>dataset: {activeSummary.dataset || '-'}</span>
            <span>report: {activeSummary.file_name || '-'}</span>
            <span>method: {activeSummary.method || '-'}</span>
            <span>top_k: {activeSummary.top_k ?? '-'}</span>
            <span>created_at: {formatDateTime(activeSummary.created_at)}</span>
            <span>cost_total: {activeSummary.cost_seconds_total ?? '-'} s</span>
          </div>
          <div className="metric-row">
            <span>corpus: {activeSummary.counts?.corpus ?? '-'}</span>
            <span>queries: {activeSummary.counts?.queries ?? '-'}</span>
            <span>qrels_queries: {activeSummary.counts?.qrels_queries ?? '-'}</span>
            <span>qrels_pairs: {activeSummary.counts?.qrels_pairs ?? '-'}</span>
          </div>
          <p className="muted">
            严格展示规则：每一行固定同一个 k，指标值直接读取评测报告原始字段（如 hit@k、recall@k、mrr@k）。
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>method</th>
                  <th>k</th>
                  {metricNames.map((metricName) => (
                    <th key={`baseline-head-${metricName}`}>{metricName}@k</th>
                  ))}
                  <th>cost(s)</th>
                </tr>
              </thead>
              <tbody>
                {!strictRows.length && (
                  <tr>
                    <td colSpan={metricNames.length + 3}>无可展示指标</td>
                  </tr>
                )}
                {strictRows.map((row) => (
                  <tr key={`metric-${row.method}-${row.k}`}>
                    <td>{row.method}</td>
                    <td>{row.k}</td>
                    {metricNames.map((metricName) => (
                      <td key={`metric-${row.method}-${row.k}-${metricName}`}>
                        {formatMetricValue(row.values[metricName])}
                      </td>
                    ))}
                    <td>{typeof row.cost === 'number' ? row.cost.toFixed(3) : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!!baselineItems.length && (
        <div className="history-list">
          <h4>历史报表</h4>
          <ul>
            {baselineItems.slice(0, 6).map((item) => (
              <li key={item.file_name}>
                <button
                  type="button"
                  className={`history-item-btn ${effectiveSelectedFileName === item.file_name ? 'is-active' : ''}`}
                  onClick={() => setSelectedFileName(item.file_name)}
                >
                  <span>{item.file_name}</span>
                  <small>{formatDateTime(item.created_at)}</small>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function AnalyzerComparisonPanel({ latest, items }) {
  const [selectedFileName, setSelectedFileName] = useState('');

  const abItems = useMemo(
    () =>
      sortReportsByCreatedAt(
        (items || []).filter(
          (item) =>
            item?.file_name?.includes('_es_ab') &&
            item?.metrics &&
            typeof item.metrics === 'object' &&
            item.metrics.ik &&
            item.metrics.standard,
        ),
      ),
    [items],
  );

  const effectiveSelectedFileName = useMemo(() => {
    if (!abItems.length) return '';
    if (selectedFileName && abItems.some((item) => item.file_name === selectedFileName)) {
      return selectedFileName;
    }
    return abItems[0].file_name;
  }, [abItems, selectedFileName]);

  const activeSummary = useMemo(() => {
    if (abItems.length) {
      const matched = abItems.find((item) => item.file_name === effectiveSelectedFileName);
      if (matched) return matched;
      return abItems[0];
    }
    if (
      latest?.file_name?.includes('_es_ab') &&
      latest?.metrics?.ik &&
      latest?.metrics?.standard
    ) {
      return latest;
    }
    return null;
  }, [abItems, effectiveSelectedFileName, latest]);

  const ikMetrics = activeSummary?.metrics?.ik || {};
  const stdMetrics = activeSummary?.metrics?.standard || {};
  const ks = collectKs(activeSummary);
  const metricNames = collectMetricNames({ ik: ikMetrics, standard: stdMetrics });
  const rows = ks.flatMap((k) =>
    metricNames.map((metricName) => {
      const ikValue = getMetricValue(ikMetrics, metricName, k);
      const stdValue = getMetricValue(stdMetrics, metricName, k);
      return {
        name: `${metricName}@${k}`,
        ikValue,
        stdValue,
      };
    }),
  );

  return (
    <section className="workflow">
      <h2>IK vs standard 对比（严格口径）</h2>
      {!activeSummary && <p className="muted">暂无 ES Analyzer A/B 报表，请先运行 run_es_analyzer_ab.py。</p>}
      {!!activeSummary && (
        <>
          <div className="metric-row">
            <span>dataset: {activeSummary.dataset || '-'}</span>
            <span>report: {activeSummary.file_name || '-'}</span>
            <span>created_at: {formatDateTime(activeSummary.created_at)}</span>
            <span>queries: {activeSummary.counts?.queries ?? '-'}</span>
          </div>
          <div className="metric-row">
            <span>ik_cost: {typeof activeSummary.method_cost_seconds?.ik === 'number' ? `${activeSummary.method_cost_seconds.ik.toFixed(3)} s` : '-'}</span>
            <span>standard_cost: {typeof activeSummary.method_cost_seconds?.standard === 'number' ? `${activeSummary.method_cost_seconds.standard.toFixed(3)} s` : '-'}</span>
            <span>total_cost: {typeof activeSummary.cost_seconds_total === 'number' ? `${activeSummary.cost_seconds_total.toFixed(3)} s` : '-'}</span>
          </div>
          <p className="muted">
            严格展示规则：逐项展示报告中的 metric@k 原值，不混用 @5 与 @10。
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>metric@k</th>
                  <th>IK</th>
                  <th>standard</th>
                  <th>IK-standard</th>
                </tr>
              </thead>
              <tbody>
                {!rows.length && (
                  <tr>
                    <td colSpan={4}>无可展示指标</td>
                  </tr>
                )}
                {rows.map(({ name, ikValue, stdValue }) => {
                  const delta = typeof ikValue === 'number' && typeof stdValue === 'number' ? ikValue - stdValue : null;
                  return (
                    <tr key={`ab-${name}`}>
                      <td>{name}</td>
                      <td>{formatMetricValue(ikValue)}</td>
                      <td>{formatMetricValue(stdValue)}</td>
                      <td className={delta == null ? '' : delta > 0 ? 'delta-pos' : delta < 0 ? 'delta-neg' : 'delta-zero'}>
                        {typeof delta === 'number' ? `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}` : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!!abItems.length && (
        <div className="history-list">
          <h4>A/B 报表历史</h4>
          <ul>
            {abItems.slice(0, 6).map((item) => (
              <li key={`ab-${item.file_name}`}>
                <button
                  type="button"
                  className={`history-item-btn ${effectiveSelectedFileName === item.file_name ? 'is-active' : ''}`}
                  onClick={() => setSelectedFileName(item.file_name)}
                >
                  <span>{item.file_name}</span>
                  <small>{formatDateTime(item.created_at)}</small>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState('demo');
  const [theme, setTheme] = useState(() => {
    if (typeof window === 'undefined') return 'light';
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  const [evalDocs, setEvalDocs] = useState(EVAL_DOC_LIBRARY);
  const visibleDocs = useMemo(
    () => (activePage === 'eval' ? evalDocs : DEMO_DOC_LIBRARY),
    [activePage, evalDocs],
  );
  const [selectedDocId, setSelectedDocId] = useState(DEFAULT_DOC.id);
  const selectedDoc = useMemo(
    () => visibleDocs.find((item) => item.id === selectedDocId) || visibleDocs[0] || DEFAULT_DOC,
    [visibleDocs, selectedDocId],
  );
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [docPath, setDocPath] = useState(DEFAULT_DOC.filePath);
  const [steps, setSteps] = useState(() => createSteps(DEFAULT_QUESTION, DEFAULT_DOC.filePath, DEFAULT_DOC.title));
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [evalHint, setEvalHint] = useState('');
  const [reportHint, setReportHint] = useState('');

  const [health, setHealth] = useState(null);
  const [chunkCount, setChunkCount] = useState(0);
  const [answer, setAnswer] = useState('');
  const [retrieval, setRetrieval] = useState(null);
  const [llmInput, setLlmInput] = useState(null);
  const [activeChunk, setActiveChunk] = useState(null);
  const [evalReports, setEvalReports] = useState([]);
  const [latestEvalReport, setLatestEvalReport] = useState(null);
  const [deletePath, setDeletePath] = useState(DEFAULT_DOC.filePath);
  const [myPermissions, setMyPermissions] = useState([]);
  const [permissionUserId, setPermissionUserId] = useState(DEFAULT_USER_ID);
  const [permissionLookup, setPermissionLookup] = useState([]);
  const [roleName, setRoleName] = useState('analyst');
  const [permissionCode, setPermissionCode] = useState('kb:export');
  const [permissionDescription, setPermissionDescription] = useState('Allow exporting indexed chunks');
  const [bindRoleName, setBindRoleName] = useState('analyst');
  const [bindPermissionCode, setBindPermissionCode] = useState('kb:export');
  const [assignUserId, setAssignUserId] = useState(DEFAULT_USER_ID);
  const [assignRoleName, setAssignRoleName] = useState('analyst');
  const [adminResult, setAdminResult] = useState(null);
  const [adminNotice, setAdminNotice] = useState('');
  const [adminError, setAdminError] = useState('');
  const [sessionUserId, setSessionUserId] = useState(DEFAULT_USER_ID);
  const [sessionLimit, setSessionLimit] = useState(20);
  const [messageLimit, setMessageLimit] = useState(100);
  const [sessions, setSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [sessionMessages, setSessionMessages] = useState([]);
  const [chatHint, setChatHint] = useState('');
  const [chatError, setChatError] = useState('');
  const [chatConversationId, setChatConversationId] = useState(() => createConversationId());
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [chatStreaming, setChatStreaming] = useState(false);
  const [chatDebugMode, setChatDebugMode] = useState(false);
  const [chatTrace, setChatTrace] = useState(null);
  const [chatLlmInput, setChatLlmInput] = useState(null);
  const [chatCitations, setChatCitations] = useState([]);

  const apiBase = useMemo(() => import.meta.env.VITE_API_BASE_URL || '', []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    refreshStatus();
    refreshEvalDatasets();
    refreshEvalReports();
    refreshMyPermissions();
    refreshSessions(DEFAULT_USER_ID, 20);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 1800);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (!adminNotice) return;
    const timer = setTimeout(() => setAdminNotice(''), 1800);
    return () => clearTimeout(timer);
  }, [adminNotice]);

  useEffect(() => {
    if (!chatHint) return;
    const timer = setTimeout(() => setChatHint(''), 1800);
    return () => clearTimeout(timer);
  }, [chatHint]);

  useEffect(() => {
    setDocPath(selectedDoc.filePath);
    setQuestion(selectedDoc.examples[0] || DEFAULT_QUESTION);
    setSteps(createSteps(selectedDoc.examples[0] || DEFAULT_QUESTION, selectedDoc.filePath, selectedDoc.title));
    setDeletePath(selectedDoc.filePath);
    setLlmInput(null);
    setActiveChunk(null);
    refreshStatus(selectedDoc.filePath);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDoc]);

  useEffect(() => {
    const first = visibleDocs[0];
    if (!first) return;
    setSelectedDocId(first.id);
  }, [visibleDocs]);

  useEffect(() => {
    if (activePage !== 'chat') return;
    if (!selectedSessionId) return;
    if (chatMessages.length) return;
    refreshSessionMessages(selectedSessionId, messageLimit, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePage, selectedSessionId]);

  function patchStep(id, patch) {
    setSteps((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  async function executeStep(id, fn) {
    const started = Date.now();
    patchStep(id, { status: 'running', error: '' });
    try {
      const output = await fn();
      patchStep(id, {
        status: 'done',
        output,
        durationMs: Date.now() - started,
      });
      return output;
    } catch (err) {
      patchStep(id, {
        status: 'error',
        error: String(err.message || err),
        durationMs: Date.now() - started,
      });
      throw err;
    }
  }

  async function refreshStatus(filePath = docPath) {
    try {
      const [healthData, statsData] = await Promise.all([
        apiGet(apiBase, '/health'),
        apiGet(apiBase, `/api/v1/debug/file-stats?file_path=${encodeURIComponent(filePath)}`),
      ]);
      setHealth(healthData);
      setChunkCount(statsData.chunk_count || 0);
    } catch {
      setHealth({ status: 'error' });
    }
  }

  async function refreshEvalDatasets() {
    try {
      const payload = await apiGet(apiBase, '/api/v1/eval/datasets/manifest');
      const items = Array.isArray(payload.items) ? payload.items : [];
      if (!items.length) {
        setEvalHint('未检测到 DuRetrieval 数据集清单。');
        return;
      }

      const mapped = items.map((item) => ({
        id: item.id,
        title: item.title || item.id,
        filePath: item.corpus_file_path_runtime || item.resolved_file_path || '',
        prepare: 'local',
        examples: item.sample_questions?.slice(0, 3)?.length
          ? item.sample_questions.slice(0, 3)
          : ['请概括这份语料包含的主题'],
        exists: item.exists,
        indexed: item.indexed,
        chunkCount: item.chunk_count || 0,
      }));

      setEvalDocs(mapped);
      setEvalHint('评测数据集清单已同步。');
    } catch (err) {
      setEvalHint(`评测数据集清单读取失败：${String(err.message || err)}`);
    }
  }

  async function refreshEvalReports() {
    try {
      const payload = await apiGet(apiBase, '/api/v1/eval/reports?limit=20');
      let latest = payload.latest || null;
      try {
        const latestPayload = await apiGet(apiBase, '/api/v1/eval/reports/latest');
        if (latestPayload?.latest?.summary) {
          latest = latestPayload.latest.summary;
        }
      } catch {
        // Ignore latest endpoint failures and fallback to list endpoint summary.
      }
      const items = Array.isArray(payload.items) ? payload.items : [];
      setEvalReports(sortReportsByCreatedAt(items));
      setLatestEvalReport(latest);
      setReportHint(items.length ? `已同步 ${items.length} 份基线报表。` : '暂无基线报表，请先运行评测脚本。');
    } catch (err) {
      setReportHint(`基线报表读取失败：${String(err.message || err)}`);
    }
  }

  async function refreshMyPermissions() {
    try {
      const payload = await apiGet(apiBase, '/api/v1/me/permissions');
      const list = Array.isArray(payload.permissions) ? payload.permissions : [];
      setMyPermissions(list);
      return list;
    } catch (err) {
      setAdminError(String(err.message || err));
      return [];
    }
  }

  async function runAdminAction(actionName, runner) {
    setAdminNotice('');
    setAdminError('');
    try {
      const payload = await runner();
      setAdminResult(payload);
      setAdminNotice(`${actionName} 已完成`);
      await refreshMyPermissions();
      return payload;
    } catch (err) {
      setAdminError(String(err.message || err));
      return null;
    }
  }

  async function queryUserPermissions(userId = permissionUserId) {
    const target = userId.trim();
    if (!target) {
      setAdminError('请输入要查询的用户 ID');
      return;
    }
    const payload = await runAdminAction('查询用户权限', () =>
      apiGet(apiBase, `/api/v1/admin/users/${encodeURIComponent(target)}/permissions`),
    );
    if (payload) {
      setPermissionLookup(Array.isArray(payload.permissions) ? payload.permissions : []);
    }
  }

  async function deleteIndexedFile() {
    const targetPath = deletePath.trim();
    if (!targetPath) {
      setAdminError('请输入需要删除索引的文件路径');
      return;
    }
    const payload = await runAdminAction('删除索引', () =>
      apiDelete(apiBase, '/api/v1/index/file', {
        file_path: targetPath,
      }),
    );
    if (payload) {
      await refreshStatus(targetPath);
      setChunkCount(0);
    }
  }

  async function refreshSessionMessages(sessionId = selectedSessionId, limit = messageLimit, syncToChat = false) {
    const targetSessionId = (sessionId || '').trim();
    if (!targetSessionId) {
      setSessionMessages([]);
      if (syncToChat) {
        setChatMessages([]);
      }
      return;
    }
    setChatError('');
    try {
      const payload = await apiGet(
        apiBase,
        `/api/v1/chat/sessions/${encodeURIComponent(targetSessionId)}/messages?limit=${limit}`,
      );
      const items = Array.isArray(payload.items) ? payload.items : [];
      setSessionMessages(items);
      if (syncToChat) {
        const normalized = items.map((item) => ({
          id: item.id || `${item.role || 'assistant'}-${item.created_at || Date.now()}`,
          role: item.role || 'assistant',
          content: item.content || '',
          citations: item.citations?.items || [],
          created_at: item.created_at || null,
        }));
        setChatConversationId(targetSessionId);
        setChatMessages(normalized);
        setChatTrace(null);
        setChatLlmInput(null);
        setChatCitations([]);
      }
      setChatHint(`已加载 ${items.length} 条会话消息`);
    } catch (err) {
      setChatError(String(err.message || err));
    }
  }

  async function refreshSessions(
    userId = sessionUserId,
    limit = sessionLimit,
    preferredSessionId = '',
    syncToChat = false,
  ) {
    const targetUserId = userId.trim();
    if (!targetUserId) {
      setChatError('请输入会话用户 ID');
      return;
    }
    setChatError('');
    try {
      const payload = await apiGet(
        apiBase,
        `/api/v1/chat/sessions?user_id=${encodeURIComponent(targetUserId)}&limit=${limit}`,
      );
      const items = Array.isArray(payload.items) ? payload.items : [];
      setSessions(items);
      if (!items.length) {
        setSelectedSessionId('');
        setSessionMessages([]);
        if (syncToChat) {
          setChatMessages([]);
        }
        setChatHint('当前用户暂无会话记录');
        return;
      }
      const preferred = preferredSessionId.trim();
      const hasPreferred = preferred && items.some((item) => item.id === preferred);
      const nextSessionId = hasPreferred ? preferred : items[0].id;
      setSelectedSessionId(nextSessionId);
      setChatHint(`已加载 ${items.length} 个会话`);
      await refreshSessionMessages(nextSessionId, messageLimit, syncToChat);
    } catch (err) {
      setChatError(String(err.message || err));
    }
  }

  function startNewConversation() {
    setSelectedSessionId('');
    setSessionMessages([]);
    setChatMessages([]);
    setChatInput('');
    setChatTrace(null);
    setChatLlmInput(null);
    setChatCitations([]);
    setChatConversationId(createConversationId());
    setChatHint('已创建新会话，可以直接提问。');
  }

  async function sendChatMessage() {
    const query = chatInput.trim();
    if (!query || chatStreaming) return;

    const conversationId = chatConversationId || createConversationId();
    if (!chatConversationId) {
      setChatConversationId(conversationId);
    }

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now() + 1}`;
    setChatInput('');
    setChatError('');
    setChatTrace(null);
    setChatLlmInput(null);
    setChatCitations([]);
    setChatStreaming(true);

    setChatMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        role: 'user',
        content: query,
        citations: [],
        created_at: new Date().toISOString(),
      },
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        citations: [],
        created_at: new Date().toISOString(),
      },
    ]);

    try {
      const payload = {
        query,
        conversation_id: conversationId,
        user_id: DEFAULT_USER_ID,
        debug: chatDebugMode,
      };

      const res = await fetch(resolveUrl(apiBase, '/api/v1/qa/ask'), {
        method: 'POST',
        headers: buildHeaders(true),
        body: JSON.stringify(payload),
      });

      if (!res.ok || !res.body) {
        const bodyText = await res.text();
        throw new Error(`qa failed: ${res.status} ${bodyText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let assistantText = '';
      let citations = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const extracted = extractSsePayloads(buffer);
        buffer = extracted.remaining;

        for (const raw of extracted.payloads) {
          let event;
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }

          if (event.type === 'context') {
            citations = event.citations || [];
            setChatCitations(citations);
            continue;
          }
          if (event.type === 'trace') {
            setChatTrace(event.trace || null);
            continue;
          }
          if (event.type === 'llm_input') {
            setChatLlmInput(event.llm_input || null);
            continue;
          }
          if (event.type === 'token') {
            assistantText += event.text || '';
            setChatMessages((prev) =>
              prev.map((item) => (item.id === assistantMessageId ? { ...item, content: assistantText } : item)),
            );
          }
        }
      }

      setChatMessages((prev) =>
        prev.map((item) =>
          item.id === assistantMessageId
            ? {
                ...item,
                content: assistantText || '(无输出)',
                citations,
              }
            : item,
        ),
      );

      await refreshSessions(sessionUserId, sessionLimit, conversationId, true);
    } catch (err) {
      setChatError(String(err.message || err));
      setChatMessages((prev) =>
        prev.map((item) =>
          item.id === assistantMessageId
            ? {
                ...item,
                content: `请求失败：${String(err.message || err)}`,
              }
            : item,
        ),
      );
    } finally {
      setChatStreaming(false);
    }
  }

  async function askWithTrace(queryText) {
    setAnswer('');
    setRetrieval(null);
    setLlmInput(null);
    setActiveChunk(null);

    const payload = {
      query: queryText,
      conversation_id: null,
      user_id: DEFAULT_USER_ID,
      debug: true,
    };

    const res = await fetch(resolveUrl(apiBase, '/api/v1/qa/ask'), {
      method: 'POST',
      headers: buildHeaders(true),
      body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body) {
      const bodyText = await res.text();
      throw new Error(`qa failed: ${res.status} ${bodyText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    let contextCount = 0;
    let citations = [];
    let trace = null;
    let llmInputPayload = null;
    let fullAnswer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const extracted = extractSsePayloads(buffer);
      buffer = extracted.remaining;

      for (const raw of extracted.payloads) {
        let event;
        try {
          event = JSON.parse(raw);
        } catch {
          continue;
        }

        if (event.type === 'context') {
          contextCount = event.count || 0;
          citations = event.citations || [];
          continue;
        }

        if (event.type === 'trace') {
          trace = event.trace || null;
          setRetrieval({
            contextCount,
            citations,
            trace: event.trace || null,
          });
          continue;
        }

        if (event.type === 'llm_input') {
          llmInputPayload = event.llm_input || null;
          setLlmInput(llmInputPayload);
          continue;
        }

        if (event.type === 'token') {
          fullAnswer += event.text || '';
          setAnswer(fullAnswer);
        }
      }
    }

    setRetrieval({ contextCount, citations, trace });

    return {
      context_count: contextCount,
      citation_count: citations.length,
      keyword_hits: trace?.keyword?.count ?? 0,
      vector_hits: trace?.vector?.count ?? 0,
      fusion_hits: trace?.fusion?.count ?? 0,
      overlap_rate: trace?.metrics?.overlap_rate ?? 0,
      fusion_gain: trace?.metrics?.fusion_gain ?? 0,
      timing_ms: trace?.timing_ms || null,
      flow: trace?.flow || null,
      fusion_preview: (trace?.fusion?.detailed_hits || []).slice(0, 5),
      llm_input: llmInputPayload,
      answer_length: fullAnswer.length,
      answer_text: fullAnswer,
    };
  }

  async function runDemo() {
    if (running) return;

    const queryText = question.trim() || DEFAULT_QUESTION;
    const activeDoc = selectedDoc;
    setRunning(true);
    setError('');
    setNotice('');
    setAnswer('');
    setRetrieval(null);
    setLlmInput(null);
    setActiveChunk(null);
    setSteps(createSteps(queryText, activeDoc.filePath, activeDoc.title));

    try {
      const sampleResult = await executeStep('download', async () => {
        if (activeDoc.prepare === 'remote_sample') {
          return apiPost(apiBase, '/api/v1/debug/sample-doc', {});
        }
        return {
          status: 'using_local',
          file_path: activeDoc.filePath,
          doc: activeDoc.title,
        };
      });
      const targetPath = sampleResult.file_path || activeDoc.filePath;
      setDocPath(targetPath);

      setSteps((prev) =>
        prev.map((item) => {
          if (item.id === 'index' || item.id === 'poll') {
            return {
              ...item,
              input: {
                ...item.input,
                file_path: targetPath,
              },
            };
          }
          return item;
        }),
      );

      const preStats = await apiGet(
        apiBase,
        `/api/v1/debug/file-stats?file_path=${encodeURIComponent(targetPath)}`,
      );

      let pollResult = preStats;
      if ((preStats.chunk_count || 0) > 0) {
        patchStep('index', {
          status: 'done',
          output: {
            status: 'skipped',
            reason: 'already_indexed',
            file_path: targetPath,
            chunk_count: preStats.chunk_count,
          },
          durationMs: 0,
        });
        patchStep('poll', {
          status: 'done',
          output: {
            ...preStats,
            polls: 0,
            status: 'already_indexed',
          },
          durationMs: 0,
        });
      } else {
        await executeStep('index', () =>
          apiPost(apiBase, '/api/v1/index/file', {
            file_path: targetPath,
          }),
        );

        const maxPolls = activePage === 'eval' ? 240 : 30;
        pollResult = await executeStep('poll', async () => {
          let latest = null;
          for (let i = 0; i < maxPolls; i += 1) {
            latest = await apiGet(apiBase, `/api/v1/debug/file-stats?file_path=${encodeURIComponent(targetPath)}`);
            if (latest.chunk_count > 0) {
              return {
                ...latest,
                polls: i + 1,
              };
            }
            await new Promise((resolve) => setTimeout(resolve, 1000));
          }
          throw new Error(`索引超时：${maxPolls} 秒内未检测到入库 chunk`);
        });
      }

      setChunkCount(pollResult.chunk_count || 0);

      const retrievalOutput = await executeStep('retrieve', () => askWithTrace(queryText));

      await executeStep('answer', async () => ({
        preview: retrievalOutput.answer_text.slice(0, 240),
        answer_length: retrievalOutput.answer_length,
        llm_input_messages: retrievalOutput.llm_input?.messages?.length || 0,
        llm_input_contexts: retrievalOutput.llm_input?.contexts?.length || 0,
      }));

      setNotice('流程已跑通，结果已可视化');
      await refreshStatus(targetPath);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setRunning(false);
    }
  }

  function openChunkDetail(sourceTitle, item) {
    setActiveChunk({
      source: sourceTitle,
      id: item.id,
      rank: item.rank ?? item.final_rank,
      file_name: item.file_name,
      file_path: item.file_path,
      chunk_id: item.chunk_id,
      preview: item.preview,
      content: item.content,
    });
  }

  return (
    <div className="app">
      <header className="hero">
        <button
          type="button"
          className="theme-icon-btn"
          onClick={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}
          disabled={running || chatStreaming}
          aria-label={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
          title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
        >
          {theme === 'dark' ? (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path
                d="M12 4.75a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0V5.5a.75.75 0 0 1 .75-.75Zm0 12.5a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0V18a.75.75 0 0 1 .75-.75ZM6.52 6.52a.75.75 0 0 1 1.06 0l1.06 1.06a.75.75 0 1 1-1.06 1.06L6.52 7.58a.75.75 0 0 1 0-1.06Zm8.84 8.84a.75.75 0 0 1 1.06 0l1.06 1.06a.75.75 0 1 1-1.06 1.06l-1.06-1.06a.75.75 0 0 1 0-1.06ZM4.75 12a.75.75 0 0 1 .75-.75H7a.75.75 0 0 1 0 1.5H5.5a.75.75 0 0 1-.75-.75Zm12.5 0a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5H18a.75.75 0 0 1-.75-.75ZM6.52 17.48a.75.75 0 0 1 0-1.06l1.06-1.06a.75.75 0 1 1 1.06 1.06l-1.06 1.06a.75.75 0 0 1-1.06 0Zm8.84-8.84a.75.75 0 0 1 0-1.06l1.06-1.06a.75.75 0 1 1 1.06 1.06l-1.06 1.06a.75.75 0 0 1-1.06 0ZM12 8.25A3.75 3.75 0 1 1 8.25 12 3.75 3.75 0 0 1 12 8.25Z"
                fill="currentColor"
              />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path
                d="M14.25 3.5a.75.75 0 0 1 .82-.7 8.5 8.5 0 1 0 6.13 12.13.75.75 0 0 1 1.4.56A10 10 0 1 1 14.95 2.68a.75.75 0 0 1-.7.82Z"
                fill="currentColor"
              />
            </svg>
          )}
        </button>
        <h1>Everything Ent Hybrid</h1>
        <p>
          {activePage === 'chat'
            ? '用户问答：会话列表 + 流式回答 + 持久化聊天历史。'
            : activePage === 'demo'
              ? '零配置体验：固定测试文档 + 一键跑通索引与问答 + 每一步输入输出可视化。'
              : 'DuRetrieval 评测页：使用 C-MTEB 与 mteb 两套语料快照执行索引与问答验证。'}
        </p>
        <div className="tab-row">
          <button
            type="button"
            className={activePage === 'chat' ? '' : 'ghost'}
            onClick={() => setActivePage('chat')}
            disabled={running || chatStreaming}
          >
            用户问答
          </button>
          <button
            type="button"
            className={activePage === 'demo' ? '' : 'ghost'}
            onClick={() => setActivePage('demo')}
            disabled={running || chatStreaming}
          >
            流程演示
          </button>
          <button
            type="button"
            className={activePage === 'eval' ? '' : 'ghost'}
            onClick={() => setActivePage('eval')}
            disabled={running || chatStreaming}
          >
            DuRetrieval 评测
          </button>
        </div>
      </header>

      <section className="summary">
        <article>
          <h2>服务状态</h2>
          <p>{health?.status || 'unknown'}</p>
        </article>
        <article>
          <h2>{activePage === 'chat' ? '当前会话' : '测试文档'}</h2>
          {activePage === 'chat' ? (
            <>
              <p>{chatConversationId || '-'}</p>
              <p className="muted">session_id: {selectedSessionId || 'new'}</p>
            </>
          ) : (
            <>
              <p>{selectedDoc.title}</p>
              <p className="muted">{docPath}</p>
            </>
          )}
        </article>
        <article>
          <h2>{activePage === 'chat' ? '消息数量' : '已入库 Chunk'}</h2>
          <p>{activePage === 'chat' ? chatMessages.length : chunkCount}</p>
        </article>
        <article>
          <h2>执行账号</h2>
          <p>{DEFAULT_USER_ID}</p>
        </article>
        <article>
          <h2>当前页面</h2>
          <p>{activePage === 'chat' ? '用户问答' : activePage === 'demo' ? '流程演示' : 'DuRetrieval 评测'}</p>
        </article>
      </section>

      {activePage === 'chat' && (
        <section className="chat-layout">
          <aside className="chat-sidebar">
            <h2>会话列表</h2>
            <label>
              user_id
              <input
                value={sessionUserId}
                onChange={(event) => setSessionUserId(event.target.value)}
                placeholder="admin"
              />
            </label>
            <div className="action-row">
              <button type="button" className="ghost" onClick={() => refreshSessions(sessionUserId, sessionLimit)}>
                刷新会话
              </button>
              <button type="button" className="ghost" onClick={startNewConversation} disabled={chatStreaming}>
                新建会话
              </button>
            </div>
            {!sessions.length && <p className="muted">暂无会话记录</p>}
            {!!sessions.length && (
              <ul className="compact-list">
                {sessions.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`history-item-btn ${selectedSessionId === item.id ? 'is-active' : ''}`}
                      onClick={() => {
                        setSelectedSessionId(item.id);
                        refreshSessionMessages(item.id, messageLimit, true);
                      }}
                    >
                      <span>{item.title || item.id}</span>
                      <small>{formatDateTime(item.created_at)}</small>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </aside>

          <section className="chat-main">
            <div className="chat-main-head">
              <div>
                <h2>用户问答窗口</h2>
                <p className="muted">conversation_id: {chatConversationId || '-'}</p>
              </div>
              <button type="button" className="ghost" onClick={() => setChatDebugMode((prev) => !prev)}>
                {chatDebugMode ? '关闭开发者模式' : '开启开发者模式'}
              </button>
            </div>

            <div className="chat-stream">
              {!chatMessages.length && <p className="muted">输入问题后即可开始对话，答案会流式返回。</p>}
              {!!chatMessages.length &&
                chatMessages.map((item) => (
                  <article key={item.id} className={`chat-message ${item.role === 'user' ? 'is-user' : 'is-assistant'}`}>
                    <div className="chat-avatar">{item.role === 'user' ? 'U' : 'AI'}</div>
                    <div className={`chat-bubble ${item.role === 'user' ? 'from-user' : 'from-assistant'}`}>
                      <div className="message-head">
                        <strong>{item.role === 'user' ? '用户' : '助手'}</strong>
                        <small>{formatDateTime(item.created_at)}</small>
                      </div>
                      <p className="chat-content">
                        {item.content || (item.role === 'assistant' && chatStreaming ? '正在思考中...' : '')}
                      </p>
                      {!!item.citations?.length && (
                        <div className="chat-citations">
                          {item.citations.slice(0, 5).map((citation, idx) => (
                            <span key={`${item.id}-citation-${idx}`}>
                              [{idx + 1}] {citation.file_name || citation.file_path || '-'}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </article>
                ))}
            </div>

            <section className="chat-compose">
              <div className="chat-compose-bar">
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="给知识库助手发送消息（Enter 发送，Shift+Enter 换行）"
                  rows={2}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault();
                      sendChatMessage();
                    }
                  }}
                />
                <button type="button" className="send-icon-btn" onClick={sendChatMessage} disabled={chatStreaming || !chatInput.trim()}>
                  {chatStreaming ? '回答中' : '发送'}
                </button>
              </div>
              <p className="muted chat-compose-tip">仅基于检索到的文档回答，并附带引用来源。</p>
            </section>

            {chatHint && <p className="notice">{chatHint}</p>}
            {chatError && <p className="error">{chatError}</p>}

            {chatDebugMode && (
              <section className="trace">
                <h2>开发者调试信息</h2>
                {!chatTrace && <p className="muted">发送问题后显示检索链路与模型输入。</p>}
                {!!chatTrace && (
                  <>
                    <div className="metric-row">
                      <span>context: {chatCitations.length}</span>
                      <span>keyword: {chatTrace.keyword?.count ?? 0}</span>
                      <span>vector: {chatTrace.vector?.count ?? 0}</span>
                      <span>fusion: {chatTrace.fusion?.count ?? 0}</span>
                      <span>overlap_rate: {formatPercent(chatTrace.metrics?.overlap_rate)}</span>
                      <span>total_ms: {formatMs(chatTrace.timing_ms?.total)}</span>
                    </div>
                    <div className="trace-grid trace-grid-2">
                      <section className="trace-block">
                        <h4>阶段耗时</h4>
                        <TimingList timing={chatTrace.timing_ms} />
                      </section>
                      <section className="trace-block">
                        <h4>候选流转</h4>
                        <FlowList flow={chatTrace.flow} />
                      </section>
                    </div>
                    <div className="trace-grid">
                      <HitList title="关键词召回" hits={chatTrace.keyword?.hits || []} onOpenChunk={openChunkDetail} />
                      <HitList title="向量召回" hits={chatTrace.vector?.hits || []} onOpenChunk={openChunkDetail} />
                      <HitList title="融合结果" hits={chatTrace.fusion?.hits || []} onOpenChunk={openChunkDetail} />
                    </div>
                  </>
                )}
                <ChunkDetailPanel chunk={activeChunk} onClose={() => setActiveChunk(null)} />
                <LlmInputPanel llmInput={chatLlmInput} />
              </section>
            )}
          </section>
        </section>
      )}

      {activePage !== 'chat' && (
        <>

      <section className="actions">
        <label>
          {activePage === 'demo' ? '选择测试文档' : '选择评测数据集'}
          <select value={selectedDocId} onChange={(event) => setSelectedDocId(event.target.value)}>
            {visibleDocs.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.title}
              </option>
            ))}
          </select>
        </label>
        <div className="chip-list">
          {selectedDoc.examples.map((item) => (
            <button key={item} type="button" className="chip" onClick={() => setQuestion(item)}>
              {item}
            </button>
          ))}
        </div>
        {activePage === 'eval' && (
          <p className="muted">
            当前问答基于本地数据文件：{selectedDoc.filePath}。如未入库，会先自动触发索引。          </p>
        )}
        <label>
          测试问题
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="请输入要测试的问题"
          />
        </label>
        <div className="action-row">
          <button type="button" onClick={runDemo} disabled={running}>
            {running ? '执行中...' : activePage === 'demo' ? '一键跑通测试流程' : '一键跑通评测问答流程'}
          </button>
          <button type="button" className="ghost" onClick={refreshStatus} disabled={running}>
            刷新状态
          </button>
          {activePage === 'eval' && (
            <button type="button" className="ghost" onClick={refreshEvalDatasets} disabled={running}>
              刷新评测清单
            </button>
          )}
          {activePage === 'eval' && (
            <button type="button" className="ghost" onClick={refreshEvalReports} disabled={running}>
              刷新基线报表
            </button>
          )}
        </div>
        {activePage === 'eval' && evalHint && <p className="muted">{evalHint}</p>}
        {activePage === 'eval' && reportHint && <p className="muted">{reportHint}</p>}
        {notice && <p className="notice">{notice}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      {activePage === 'eval' && (
        <>
          <section className="workflow">
            <h2>DuRetrieval 数据集状态</h2>
            <div className="dataset-grid">
              {evalDocs.map((item) => (
                <article key={item.id} className="step-card">
                  <h3>{item.title}</h3>
                  <p className="muted">{item.filePath}</p>
                  <div className="metric-row">
                    <span>exists: {item.exists ? 'yes' : 'no'}</span>
                    <span>indexed: {item.indexed ? 'yes' : 'no'}</span>
                    <span>chunk: {item.chunkCount ?? 0}</span>
                  </div>
                </article>
              ))}
            </div>
          </section>
          <BaselineReportPanel latest={latestEvalReport} items={evalReports} />
          <AnalyzerComparisonPanel latest={latestEvalReport} items={evalReports} />
        </>
      )}

      <section className="workflow">
        <h2>索引与权限管理</h2>
        <div className="trace-grid">
          <section className="trace-block">
            <h4>当前账号权限</h4>
            {!myPermissions.length && <p className="muted">暂无权限数据</p>}
            {!!myPermissions.length && (
              <div className="metric-row">
                {myPermissions.map((code) => (
                  <span key={`my-perm-${code}`}>{code}</span>
                ))}
              </div>
            )}
            <div className="action-row">
              <button type="button" className="ghost" onClick={refreshMyPermissions} disabled={running}>
                刷新我的权限
              </button>
            </div>
          </section>

          <section className="trace-block">
            <h4>索引删除</h4>
            <label>
              file_path
              <input
                value={deletePath}
                onChange={(event) => setDeletePath(event.target.value)}
                placeholder="/data/knowledge/example.md"
              />
            </label>
            <div className="action-row">
              <button type="button" className="ghost" onClick={deleteIndexedFile} disabled={running}>
                删除该文件索引
              </button>
            </div>
          </section>
        </div>

        <div className="trace-grid">
          <section className="trace-block">
            <h4>RBAC 初始化与查询</h4>
            <label>
              查询用户 ID
              <input
                value={permissionUserId}
                onChange={(event) => setPermissionUserId(event.target.value)}
                placeholder="admin"
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="ghost"
                onClick={() => runAdminAction('RBAC 初始化', () => apiPost(apiBase, '/api/v1/admin/bootstrap', {}))}
                disabled={running}
              >
                执行 RBAC Bootstrap
              </button>
              <button type="button" className="ghost" onClick={() => queryUserPermissions()} disabled={running}>
                查询用户权限
              </button>
            </div>
            {!!permissionLookup.length && (
              <div className="metric-row">
                {permissionLookup.map((code) => (
                  <span key={`lookup-perm-${code}`}>{code}</span>
                ))}
              </div>
            )}
          </section>

          <section className="trace-block">
            <h4>创建角色与权限</h4>
            <label>
              角色名
              <input value={roleName} onChange={(event) => setRoleName(event.target.value)} placeholder="analyst" />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="ghost"
                onClick={() =>
                  runAdminAction('创建角色', () =>
                    apiPost(apiBase, '/api/v1/admin/roles', {
                      name: roleName.trim(),
                    }),
                  )
                }
                disabled={running}
              >
                创建角色
              </button>
            </div>
            <label>
              权限码
              <input
                value={permissionCode}
                onChange={(event) => setPermissionCode(event.target.value)}
                placeholder="kb:export"
              />
            </label>
            <label>
              权限描述
              <input
                value={permissionDescription}
                onChange={(event) => setPermissionDescription(event.target.value)}
                placeholder="Allow exporting indexed chunks"
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="ghost"
                onClick={() =>
                  runAdminAction('创建权限', () =>
                    apiPost(apiBase, '/api/v1/admin/permissions', {
                      code: permissionCode.trim(),
                      description: permissionDescription.trim(),
                    }),
                  )
                }
                disabled={running}
              >
                创建权限
              </button>
            </div>
          </section>
        </div>

        <div className="trace-grid">
          <section className="trace-block">
            <h4>角色授权权限</h4>
            <label>
              角色名
              <input
                value={bindRoleName}
                onChange={(event) => setBindRoleName(event.target.value)}
                placeholder="analyst"
              />
            </label>
            <label>
              权限码
              <input
                value={bindPermissionCode}
                onChange={(event) => setBindPermissionCode(event.target.value)}
                placeholder="kb:export"
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="ghost"
                onClick={() =>
                  runAdminAction('绑定权限到角色', () =>
                    apiPost(
                      apiBase,
                      `/api/v1/admin/roles/${encodeURIComponent(bindRoleName.trim())}/permissions/${encodeURIComponent(bindPermissionCode.trim())}`,
                      {},
                    ),
                  )
                }
                disabled={running}
              >
                绑定权限到角色
              </button>
            </div>
          </section>

          <section className="trace-block">
            <h4>分配角色给用户</h4>
            <label>
              用户 ID
              <input
                value={assignUserId}
                onChange={(event) => setAssignUserId(event.target.value)}
                placeholder="alice"
              />
            </label>
            <label>
              角色名
              <input
                value={assignRoleName}
                onChange={(event) => setAssignRoleName(event.target.value)}
                placeholder="analyst"
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="ghost"
                onClick={() =>
                  runAdminAction('分配角色给用户', () =>
                    apiPost(
                      apiBase,
                      `/api/v1/admin/users/${encodeURIComponent(assignUserId.trim())}/roles/${encodeURIComponent(assignRoleName.trim())}`,
                      {},
                    ),
                  )
                }
                disabled={running}
              >
                分配角色
              </button>
            </div>
          </section>
        </div>
        {adminNotice && <p className="notice">{adminNotice}</p>}
        {adminError && <p className="error">{adminError}</p>}
        {!!adminResult && <pre>{JSON.stringify(adminResult, null, 2)}</pre>}
      </section>

      <section className="workflow">
        <h2>会话历史</h2>
        <div className="trace-grid">
          <section className="trace-block">
            <h4>会话列表</h4>
            <label>
              user_id
              <input
                value={sessionUserId}
                onChange={(event) => setSessionUserId(event.target.value)}
                placeholder="admin"
              />
            </label>
            <label>
              会话上限
              <input
                type="number"
                min={1}
                max={200}
                value={sessionLimit}
                onChange={(event) => setSessionLimit(Number(event.target.value) || 20)}
              />
            </label>
            <div className="action-row">
              <button type="button" className="ghost" onClick={() => refreshSessions()} disabled={running}>
                拉取会话
              </button>
            </div>
            {!sessions.length && <p className="muted">暂无会话</p>}
            {!!sessions.length && (
              <ul className="compact-list">
                {sessions.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`history-item-btn ${selectedSessionId === item.id ? 'is-active' : ''}`}
                      onClick={() => {
                        setSelectedSessionId(item.id);
                        refreshSessionMessages(item.id, messageLimit);
                      }}
                    >
                      <span>{item.title || item.id}</span>
                      <small>{formatDateTime(item.created_at)}</small>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="trace-block">
            <h4>会话消息</h4>
            <label>
              消息上限
              <input
                type="number"
                min={1}
                max={500}
                value={messageLimit}
                onChange={(event) => setMessageLimit(Number(event.target.value) || 100)}
              />
            </label>
            <div className="action-row">
              <button
                type="button"
                className="ghost"
                onClick={() => refreshSessionMessages()}
                disabled={running || !selectedSessionId}
              >
                拉取消息
              </button>
            </div>
            {!selectedSessionId && <p className="muted">请先选择会话</p>}
            {!!selectedSessionId && !sessionMessages.length && <p className="muted">该会话暂无消息</p>}
            {!!sessionMessages.length && (
              <ul className="message-list">
                {sessionMessages.map((item) => (
                  <li key={item.id}>
                    <div className="message-head">
                      <strong>{item.role || '-'}</strong>
                      <small>{formatDateTime(item.created_at)}</small>
                    </div>
                    <pre>{item.content || ''}</pre>
                    {!!item.citations && <pre>{JSON.stringify(item.citations, null, 2)}</pre>}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
        {chatHint && <p className="notice">{chatHint}</p>}
        {chatError && <p className="error">{chatError}</p>}
      </section>

      <section className="workflow">
        <h2>流程可视化（输入 / 输出）</h2>
        <div className="step-grid">
          {steps.map((step) => (
            <StepCard key={step.id} step={step} />
          ))}
        </div>
      </section>

      <section className="result">
        <h2>问答结果</h2>
        <div className="answer-box">{answer || '执行后显示答案'}</div>
      </section>

      <section className="trace">
        <h2>检索指标</h2>
        {!retrieval?.trace && <p className="muted">执行流程后显示召回与融合指标。</p>}
        {!!retrieval?.trace && (
          <>
            <div className="metric-row">
              <span>context: {retrieval.contextCount}</span>
              <span>keyword: {retrieval.trace.keyword?.count ?? 0}</span>
              <span>vector: {retrieval.trace.vector?.count ?? 0}</span>
              <span>fusion: {retrieval.trace.fusion?.count ?? 0}</span>
              <span>overlap_rate: {formatPercent(retrieval.trace.metrics?.overlap_rate)}</span>
              <span>fusion_gain: {retrieval.trace.metrics?.fusion_gain ?? 0}</span>
              <span>total_ms: {formatMs(retrieval.trace.timing_ms?.total)}</span>
            </div>
            <div className="trace-grid trace-grid-2">
              <section className="trace-block">
                <h4>阶段耗时</h4>
                <TimingList timing={retrieval.trace.timing_ms} />
              </section>
              <section className="trace-block">
                <h4>候选流转</h4>
                <FlowList flow={retrieval.trace.flow} />
              </section>
            </div>
            <section className="trace-block">
              <h4>融合贡献明细（Top Final）</h4>
              <FusionTable
                rows={retrieval.trace.fusion?.detailed_hits || []}
                onOpenChunk={openChunkDetail}
                sourceTitle="融合贡献明细"
              />
            </section>
            <section className="trace-block">
              <h4>未进入 Top-K 的候选</h4>
              <FusionTable
                rows={retrieval.trace.fusion?.dropped_candidates || []}
                onOpenChunk={openChunkDetail}
                sourceTitle="未进入Top-K"
              />
            </section>
            <div className="trace-grid">
              <HitList title="关键词召回" hits={retrieval.trace.keyword?.hits || []} onOpenChunk={openChunkDetail} />
              <HitList title="向量召回" hits={retrieval.trace.vector?.hits || []} onOpenChunk={openChunkDetail} />
              <HitList title="融合结果" hits={retrieval.trace.fusion?.hits || []} onOpenChunk={openChunkDetail} />
            </div>
          </>
        )}
        <ChunkDetailPanel chunk={activeChunk} onClose={() => setActiveChunk(null)} />
        <LlmInputPanel llmInput={llmInput} />
      </section>
        </>
      )}
    </div>
  );
}
