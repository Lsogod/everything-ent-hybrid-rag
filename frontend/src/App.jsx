import { useEffect, useMemo, useState } from 'react';
import './App.css';

const DEFAULT_USER_ID = import.meta.env.VITE_USER_ID || 'admin';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || '';
const DOC_LIBRARY = [
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
];

const DEFAULT_DOC = DOC_LIBRARY[0];
const DEFAULT_QUESTION = DEFAULT_DOC.examples[0];

function createSteps(query, filePath, docTitle) {
  return [
    {
      id: 'download',
      title: '1. 准备测试文档',
      status: 'idle',
      input: {
        mode: 'local_or_remote',
        endpoint: '/api/v1/debug/sample-doc (仅远程样例文档)',
        doc: docTitle,
        file_path: filePath,
      },
      output: null,
      error: '',
      durationMs: 0,
      explain: {
        what: '确保本次测试使用的文档可被后续索引与检索读取。',
        choose: '远程样例文档会先下载；你选择的本地文档则直接使用现有文件。',
        done: '返回有效 file_path，后续步骤都基于这个路径执行。',
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
        what: '把文档送入异步索引队列，触发解析、切分、向量化与写入 ES。',
        choose: '只索引当前选中的文档路径，不会一次索引全部知识库。',
        done: '返回 task_id，表示任务已进入队列等待 worker 执行。',
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
        what: '轮询查看该文档是否已完成入库，避免在未就绪时提问。',
        choose: '每 1 秒查询一次，最多 30 次；当 chunk_count > 0 判定完成。',
        done: '看到 indexed=true 且 chunk_count 大于 0。',
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
        what: '对问题并行执行关键词召回与向量召回，再用 RRF 融合排序。',
        choose: '先取两路候选并集，再按 1/(k+rank) 计算融合分，选择 Top-K。',
        done: '得到 fusion 命中列表、候选流转、耗时与融合贡献明细。',
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
        what: '将 Top-K 上下文拼装成 messages，发送给大模型生成答案。',
        choose: '仅使用检索得到的上下文，不额外引入外部知识。',
        done: '收到 token 流并结束，展示最终答案与完整 llm_input。',
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
    const n1 = remaining.indexOf('\n\n');
    const n2 = remaining.indexOf('\\n\\n');

    let splitAt = -1;
    let delimiterSize = 2;

    if (n1 !== -1 && n2 !== -1) {
      splitAt = Math.min(n1, n2);
      delimiterSize = splitAt === n1 ? 2 : 4;
    } else if (n1 !== -1) {
      splitAt = n1;
      delimiterSize = 2;
    } else if (n2 !== -1) {
      splitAt = n2;
      delimiterSize = 4;
    }

    if (splitAt === -1) break;

    const frame = remaining.slice(0, splitAt).trim();
    remaining = remaining.slice(splitAt + delimiterSize);

    if (!frame) continue;
    const lines = frame.split(/\r?\n|\\n/);
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

function HitList({ title, hits }) {
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

function FusionTable({ rows }) {
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
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LlmInputPanel({ llmInput }) {
  if (!llmInput) {
    return (
      <section className="trace-block">
        <h4>传给大模型的内容（完整）</h4>
        <p className="muted">执行流程后显示模型输入 messages 与上下文全文。</p>
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

export default function App() {
  const [selectedDocId, setSelectedDocId] = useState(DEFAULT_DOC.id);
  const selectedDoc = useMemo(
    () => DOC_LIBRARY.find((item) => item.id === selectedDocId) || DEFAULT_DOC,
    [selectedDocId],
  );
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [docPath, setDocPath] = useState(DEFAULT_DOC.filePath);
  const [steps, setSteps] = useState(() => createSteps(DEFAULT_QUESTION, DEFAULT_DOC.filePath, DEFAULT_DOC.title));
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  const [health, setHealth] = useState(null);
  const [chunkCount, setChunkCount] = useState(0);
  const [answer, setAnswer] = useState('');
  const [retrieval, setRetrieval] = useState(null);
  const [llmInput, setLlmInput] = useState(null);

  const apiBase = useMemo(() => import.meta.env.VITE_API_BASE_URL || '', []);

  useEffect(() => {
    refreshStatus();
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
    refreshStatus(selectedDoc.filePath);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDocId]);

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

  async function askWithTrace(queryText) {
    setAnswer('');
    setRetrieval(null);
    setLlmInput(null);

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

      await executeStep('index', () =>
        apiPost(apiBase, '/api/v1/index/file', {
          file_path: targetPath,
        }),
      );

      const pollResult = await executeStep('poll', async () => {
        let latest = null;
        for (let i = 0; i < 30; i += 1) {
          latest = await apiGet(apiBase, `/api/v1/debug/file-stats?file_path=${encodeURIComponent(targetPath)}`);
          if (latest.chunk_count > 0) {
            return {
              ...latest,
              polls: i + 1,
            };
          }
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        throw new Error('索引超时：30 秒内未检测到入库 chunk');
      });

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

  return (
    <div className="app">
      <header className="hero">
        <h1>Everything Ent Hybrid</h1>
        <p>零配置体验：固定测试文档 + 一键跑通索引与问答 + 每一步输入输出可视化。</p>
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
      </section>

      <section className="actions">
        <label>
          选择测试文档
          <select value={selectedDocId} onChange={(event) => setSelectedDocId(event.target.value)}>
            {DOC_LIBRARY.map((doc) => (
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
            {running ? '执行中...' : '一键跑通测试流程'}
          </button>
          <button type="button" className="ghost" onClick={refreshStatus} disabled={running}>
            刷新状态
          </button>
        </div>
        {notice && <p className="notice">{notice}</p>}
        {error && <p className="error">{error}</p>}
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
              <FusionTable rows={retrieval.trace.fusion?.detailed_hits || []} />
            </section>
            <section className="trace-block">
              <h4>未进入 Top-K 的候选</h4>
              <FusionTable rows={retrieval.trace.fusion?.dropped_candidates || []} />
            </section>
            <div className="trace-grid">
              <HitList title="关键词召回" hits={retrieval.trace.keyword?.hits || []} />
              <HitList title="向量召回" hits={retrieval.trace.vector?.hits || []} />
              <HitList title="融合结果" hits={retrieval.trace.fusion?.hits || []} />
            </div>
          </>
        )}
        <LlmInputPanel llmInput={llmInput} />
      </section>
    </div>
  );
}
