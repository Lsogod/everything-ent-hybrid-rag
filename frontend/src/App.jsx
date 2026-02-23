import { useEffect, useMemo, useState } from 'react';
import './App.css';

const DEFAULT_USER_ID = import.meta.env.VITE_USER_ID || 'admin';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || '';
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

function pickMetric(metrics, name, preferredK = 5) {
  if (!metrics || typeof metrics !== 'object') return '-';
  const exact = metrics[`${name}@${preferredK}`];
  if (typeof exact === 'number') return exact.toFixed(4);
  const key = Object.keys(metrics)
    .filter((item) => item.startsWith(`${name}@`))
    .sort((a, b) => {
      const aK = Number(a.split('@')[1] || 0);
      const bK = Number(b.split('@')[1] || 0);
      return aK - bK;
    })[0];
  if (!key) return '-';
  const fallback = metrics[key];
  return typeof fallback === 'number' ? fallback.toFixed(4) : '-';
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
  if (!chunk) return null;
  return (
    <section className="trace-block chunk-detail-panel">
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

  useEffect(() => {
    if (!items?.length) {
      setSelectedFileName('');
      return;
    }
    if (!selectedFileName || !items.some((item) => item.file_name === selectedFileName)) {
      setSelectedFileName(items[0].file_name);
    }
  }, [items, selectedFileName]);

  const activeSummary = useMemo(() => {
    if (items?.length) {
      const matched = items.find((item) => item.file_name === selectedFileName);
      if (matched) return matched;
      return items[0];
    }
    return latest || null;
  }, [items, latest, selectedFileName]);

  const methods = activeSummary?.metrics ? Object.keys(activeSummary.metrics) : [];

  return (
    <section className="workflow">
      <h2>公共基线报表</h2>
      {!activeSummary && <p className="muted">暂无基线报表，请先运行 eval 脚本。</p>}
      {!!activeSummary && (
        <>
          <div className="metric-row">
            <span>dataset: {activeSummary.dataset || '-'}</span>
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
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>method</th>
                  <th>hit@5</th>
                  <th>recall@5</th>
                  <th>mrr@10</th>
                  <th>map@10</th>
                  <th>ndcg@10</th>
                  <th>cost(s)</th>
                </tr>
              </thead>
              <tbody>
                {methods.map((method) => {
                  const metrics = activeSummary.metrics?.[method] || {};
                  const cost = activeSummary.method_cost_seconds?.[method];
                  return (
                    <tr key={`metric-${method}`}>
                      <td>{method}</td>
                      <td>{pickMetric(metrics, 'hit', 5)}</td>
                      <td>{pickMetric(metrics, 'recall', 5)}</td>
                      <td>{pickMetric(metrics, 'mrr', 10)}</td>
                      <td>{pickMetric(metrics, 'map', 10)}</td>
                      <td>{pickMetric(metrics, 'ndcg', 10)}</td>
                      <td>{typeof cost === 'number' ? cost.toFixed(3) : '-'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!!items?.length && (
        <div className="history-list">
          <h4>鍘嗗彶鎶ヨ〃</h4>
          <ul>
            {items.slice(0, 6).map((item) => (
              <li key={item.file_name}>
                <button
                  type="button"
                  className={`history-item-btn ${selectedFileName === item.file_name ? 'active' : ''}`}
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

  const apiBase = useMemo(() => import.meta.env.VITE_API_BASE_URL || '', []);

  useEffect(() => {
    refreshStatus();
    refreshEvalDatasets();
    refreshEvalReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 1800);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    setDocPath(selectedDoc.filePath);
    setQuestion(selectedDoc.examples[0] || DEFAULT_QUESTION);
    setSteps(createSteps(selectedDoc.examples[0] || DEFAULT_QUESTION, selectedDoc.filePath, selectedDoc.title));
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
      const items = Array.isArray(payload.items) ? payload.items : [];
      setEvalReports(items);
      setLatestEvalReport(payload.latest || null);
      setReportHint(items.length ? `已同步 ${items.length} 份基线报表。` : '暂无基线报表，请先运行评测脚本。');
    } catch (err) {
      setReportHint(`基线报表读取失败：${String(err.message || err)}`);
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
        <h1>Everything Ent Hybrid</h1>
        <p>
          {activePage === 'demo'
            ? '零配置体验：固定测试文档 + 一键跑通索引与问答 + 每一步输入输出可视化。'
            : 'DuRetrieval 评测页：使用 C-MTEB 与 mteb 两套语料快照执行索引与问答验证。'}
        </p>
        <div className="tab-row">
          <button
            type="button"
            className={activePage === 'demo' ? '' : 'ghost'}
            onClick={() => setActivePage('demo')}
            disabled={running}
          >
            流程演示
          </button>
          <button
            type="button"
            className={activePage === 'eval' ? '' : 'ghost'}
            onClick={() => setActivePage('eval')}
            disabled={running}
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
          <h2>测试文档</h2>
          <p>{selectedDoc.title}</p>
          <p className="muted">{docPath}</p>
        </article>
        <article>
          <h2>已入库 Chunk</h2>
          <p>{chunkCount}</p>
        </article>
        <article>
          <h2>执行账号</h2>
          <p>{DEFAULT_USER_ID}</p>
        </article>
        <article>
          <h2>当前页面</h2>
          <p>{activePage === 'demo' ? '流程演示' : 'DuRetrieval 评测'}</p>
        </article>
      </section>

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
        </>
      )}

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
    </div>
  );
}

