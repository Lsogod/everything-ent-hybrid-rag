import { useEffect, useMemo, useState, useRef } from 'react';
import './App.css';

const DEFAULT_USER_ID = import.meta.env.VITE_USER_ID || 'admin';
const DEFAULT_API_KEY = import.meta.env.VITE_API_KEY || '';
const apiBase = import.meta.env.VITE_API_BASE_URL || '';

// --- 原有逻辑数据与配置 ---
const DURETRIEVAL_DOCS = [
  {
    id: 'duretrieval_c_mteb',
    title: 'DuRetrieval (C-MTEB) snapshot',
    filePath: '/data/knowledge/datasets/duretrieval_c_mteb/corpus.md',
    prepare: 'local',
    examples: ['国家法定节假日共多少天', '功和功率的区别', '我国古代第一个有伟大成就的爱国诗人是( )'],
  },
  {
    id: 'duretrieval_mteb',
    title: 'DuRetrieval (mteb) snapshot',
    filePath: '/data/knowledge/datasets/duretrieval_mteb/corpus.md',
    prepare: 'local',
    examples: ['如何查看好友申请', '怎么屏蔽QQ新闻弹窗', '宝鸡装修房子多少钱'],
  },
];

const DEMO_DOC_LIBRARY = [
  {
    id: 'fastapi_en',
    title: 'FastAPI 官方 README（英文）',
    filePath: '/data/knowledge/fastapi_official_readme.md',
    prepare: 'remote_sample',
    examples: ['What is FastAPI and what are its key features?', 'How does FastAPI improve developer productivity?'],
  },
  {
    id: 'vue_zh',
    title: 'Vue 中文介绍',
    filePath: '/data/knowledge/vue_zh_introduction.md',
    prepare: 'local',
    examples: ['Vue 是什么？', 'Vue 文档对新手的学习建议是什么？'],
  },
  ...DURETRIEVAL_DOCS,
];

const DEFAULT_DOC = DEMO_DOC_LIBRARY[0];
const DEFAULT_QUESTION = DEFAULT_DOC.examples[0];

// --- 工具函数 ---
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

async function apiGet(path) {
  const res = await fetch(resolveUrl(apiBase, path), { headers: buildHeaders(false) });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(resolveUrl(apiBase, path), {
    method: 'POST',
    headers: buildHeaders(true),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
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

function formatMs(value) {
  return typeof value === 'number' ? `${value.toFixed(2)} ms` : '-';
}

function formatPercent(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '-';
}

function formatDateTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

// --- 子组件 (保留原有逻辑，应用新样式) ---

function StepCard({ step }) {
  const output = step.error ? { error: step.error } : step.output || { message: '等待执行' };
  return (
    <article className={`step-card step-${step.status}`}>
      <header>
        <h3>{step.title}</h3>
        <div className="step-meta">
          <span>{step.status === 'done' ? '已完成' : step.status === 'running' ? '执行中' : step.status === 'error' ? '失败' : '等待'}</span>
          <span>{step.durationMs ? `${step.durationMs} ms` : '-'}</span>
        </div>
      </header>
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
          {hits.slice(0, 5).map((hit, idx) => (
            <li key={idx}>
              <div className="hit-header">
                <strong>#{hit.rank}</strong>
                <span>{hit.file_name || hit.file_path || hit.id}</span>
              </div>
              <small>score={hit.score?.toFixed(4)} rrf={hit.rrf_score?.toFixed(4)}</small>
              <p className="hit-preview">{hit.preview}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function BaselineReportPanel({ latest, items }) {
  const [selectedFileName, setSelectedFileName] = useState('');
  const activeSummary = useMemo(() => {
    if (items?.length) {
      return items.find((item) => item.file_name === selectedFileName) || items[0];
    }
    return latest || null;
  }, [items, latest, selectedFileName]);

  const methods = activeSummary?.metrics ? Object.keys(activeSummary.metrics) : [];

  return (
    <section className="workflow">
      <div className="section-header">
        <h2>公共基线报表</h2>
      </div>
      {!activeSummary && <p className="muted">暂无报表数据</p>}
      {!!activeSummary && (
        <>
          <div className="metric-row">
            <span>Dataset: {activeSummary.dataset}</span>
            <span>Created: {formatDateTime(activeSummary.created_at)}</span>
            <span>Total Cost: {activeSummary.cost_seconds_total}s</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Hit@5</th>
                  <th>Recall@5</th>
                  <th>MRR@10</th>
                  <th>nDCG@10</th>
                  <th>Cost(s)</th>
                </tr>
              </thead>
              <tbody>
                {methods.map((m) => (
                  <tr key={m}>
                    <td>{m}</td>
                    <td>{activeSummary.metrics[m]['hit@5']?.toFixed(4)}</td>
                    <td>{activeSummary.metrics[m]['recall@5']?.toFixed(4)}</td>
                    <td>{activeSummary.metrics[m]['mrr@10']?.toFixed(4)}</td>
                    <td>{activeSummary.metrics[m]['ndcg@10']?.toFixed(4)}</td>
                    <td>{activeSummary.method_cost_seconds?.[m]?.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="history-list">
             <h4>历史快照</h4>
             <div className="chip-list">
               {items.map(item => (
                 <button 
                  key={item.file_name} 
                  className={`chip ${selectedFileName === item.file_name ? 'active' : ''}`}
                  onClick={() => setSelectedFileName(item.file_name)}
                 >
                   {item.file_name}
                 </button>
               ))}
             </div>
          </div>
        </>
      )}
    </section>
  );
}

// --- 主程序 ---

export default function App() {
  const [activePage, setActivePage] = useState('chat'); // 'chat', 'demo', 'eval'
  
  // Chat 状态
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesEndRef = useRef(null);

  // Demo & Eval 共享状态
  const [selectedDocId, setSelectedDocId] = useState(DEFAULT_DOC.id);
  const [evalDocs, setEvalDocs] = useState([]);
  const visibleDocs = useMemo(() => (activePage === 'eval' ? evalDocs : DEMO_DOC_LIBRARY), [activePage, evalDocs]);
  const selectedDoc = useMemo(() => visibleDocs.find(d => d.id === selectedDocId) || visibleDocs[0] || DEFAULT_DOC, [visibleDocs, selectedDocId]);
  
  const [question, setQuestion] = useState(DEFAULT_DOC.examples[0]);
  const [steps, setSteps] = useState([]);
  const [running, setRunning] = useState(false);
  const [retrieval, setRetrieval] = useState(null);
  const [evalReports, setEvalReports] = useState([]);
  const [latestReport, setLatestReport] = useState(null);

  // 初始化
  useEffect(() => {
    refreshSessions();
    refreshEvalDatasets();
    refreshEvalReports();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- 逻辑函数 ---

  const refreshSessions = async () => {
    try {
      const res = await apiGet('/api/v1/chat/sessions');
      setSessions(res.items || []);
    } catch (err) { console.error(err); }
  };

  const loadMessages = async (sid) => {
    try {
      const res = await apiGet(`/api/v1/chat/sessions/${sid}/messages`);
      setMessages(res.items || []);
    } catch (err) { console.error(err); }
  };

  const refreshEvalDatasets = async () => {
    try {
      const payload = await apiGet('/api/v1/eval/datasets/manifest');
      const items = Array.isArray(payload.items) ? payload.items : [];
      setEvalDocs(items.map(item => ({
        id: item.id,
        title: item.title || item.id,
        filePath: item.corpus_file_path_runtime || item.resolved_file_path || '',
        prepare: 'local',
        examples: item.sample_questions?.slice(0, 3) || ['请概括这份语料的内容'],
        indexed: item.indexed,
        exists: item.exists
      })));
    } catch (err) { console.error(err); }
  };

  const refreshEvalReports = async () => {
    try {
      const payload = await apiGet('/api/v1/eval/reports?limit=20');
      setEvalReports(payload.items || []);
      setLatestReport(payload.latest || null);
    } catch (err) { console.error(err); }
  };

  const createSteps = (q, path, title) => [
    { id: 'download', title: '1. 文档准备', status: 'idle', input: { path, title }, output: null },
    { id: 'index', title: '2. 索引任务', status: 'idle', input: { method: 'POST', path }, output: null },
    { id: 'poll', title: '3. 入库等待', status: 'idle', input: { endpoint: 'file-stats' }, output: null },
    { id: 'retrieve', title: '4. 检索与问答', status: 'idle', input: { query: q }, output: null },
  ];

  const patchStep = (id, patch) => setSteps(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));

  // --- 事件处理 ---

  const handleSendChat = async () => {
    if (!chatInput.trim() || isGenerating) return;
    const q = chatInput;
    setChatInput('');
    setIsGenerating(true);
    const sid = activeSessionId || Math.random().toString(36).substring(7);
    if (!activeSessionId) setActiveSessionId(sid);

    const userMsg = { id: Date.now(), role: 'user', content: q };
    const aiMsg = { id: Date.now() + 1, role: 'assistant', content: '', citations: null };
    setMessages(prev => [...prev, userMsg, aiMsg]);

    try {
      const res = await fetch(resolveUrl(apiBase, '/api/v1/qa/ask'), {
        method: 'POST',
        headers: buildHeaders(true),
        body: JSON.stringify({ query: q, conversation_id: sid, user_id: DEFAULT_USER_ID, debug: true })
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { payloads, remaining } = extractSsePayloads(buffer);
        buffer = remaining;
        for (const raw of payloads) {
          const ev = JSON.parse(raw);
          if (ev.type === 'token') {
            fullText += ev.text;
            setMessages(prev => prev.map(m => m.id === aiMsg.id ? { ...m, content: fullText } : m));
          } else if (ev.type === 'context') {
            setMessages(prev => prev.map(m => m.id === aiMsg.id ? { ...m, citations: { items: ev.citations } } : m));
          }
        }
      }
      refreshSessions();
    } catch (err) {
      setMessages(prev => prev.map(m => m.id === aiMsg.id ? { ...m, content: 'Error: ' + err.message } : m));
    } finally { setIsGenerating(false); }
  };

  const runDemoFlow = async () => {
    if (running) return;
    setRunning(true);
    setSteps(createSteps(question, selectedDoc.filePath, selectedDoc.title));
    setRetrieval(null);
    try {
      // 1. Download/Prepare
      patchStep('download', { status: 'running' });
      await new Promise(r => setTimeout(r, 800));
      patchStep('download', { status: 'done', output: { status: 'ok', file: selectedDoc.filePath } });

      // 2. Index
      patchStep('index', { status: 'running' });
      const idxRes = await apiPost('/api/v1/index/file', { file_path: selectedDoc.filePath });
      patchStep('index', { status: 'done', output: idxRes });

      // 3. Poll
      patchStep('poll', { status: 'running' });
      let stats = null;
      for(let i=0; i<30; i++) {
        stats = await apiGet(`/api/v1/debug/file-stats?file_path=${encodeURIComponent(selectedDoc.filePath)}`);
        if (stats.chunk_count > 0) break;
        await new Promise(r => setTimeout(r, 1000));
      }
      patchStep('poll', { status: 'done', output: stats });

      // 4. Retrieve (Ask)
      patchStep('retrieve', { status: 'running' });
      const res = await fetch(resolveUrl(apiBase, '/api/v1/qa/ask'), {
        method: 'POST',
        headers: buildHeaders(true),
        body: JSON.stringify({ query: question, debug: true })
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let trace = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { payloads, remaining } = extractSsePayloads(buffer);
        buffer = remaining;
        for (const raw of payloads) {
          const ev = JSON.parse(raw);
          if (ev.type === 'trace') trace = ev.trace;
        }
      }
      setRetrieval({ trace });
      patchStep('retrieve', { status: 'done', output: { trace_received: true } });
    } catch (err) {
      console.error(err);
    } finally { setRunning(false); }
  };

  return (
    <div className="app-container">
      {/* Sidebar - 只在 Chat 页面有明显作用 */}
      <aside className="sidebar">
        <div className="sidebar-header">
           <div className="logo">Everything Ent</div>
           <button className="new-chat-btn" onClick={() => { setActiveSessionId(null); setMessages([]); setActivePage('chat'); }}>
             + 新建对话
           </button>
        </div>
        <nav className="nav-menu">
           <button className={activePage === 'chat' ? 'active' : ''} onClick={() => setActivePage('chat')}>智能对话</button>
           <button className={activePage === 'demo' ? 'active' : ''} onClick={() => setActivePage('demo')}>流程演示</button>
           <button className={activePage === 'eval' ? 'active' : ''} onClick={() => setActivePage('eval')}>评测报告</button>
        </nav>
        {activePage === 'chat' && (
          <div className="session-list">
            <header>最近历史</header>
            {sessions.map(s => (
              <button key={s.id} className={activeSessionId === s.id ? 'active' : ''} onClick={() => { setActiveSessionId(s.id); loadMessages(s.id); }}>
                {s.title || '新会话'}
              </button>
            ))}
          </div>
        )}
      </aside>

      <main className="main-content">
        {activePage === 'chat' && (
          <div className="chat-view">
             <div className="chat-header">智能问答机器人</div>
             <div className="messages-container">
               {messages.length === 0 ? (
                 <div className="empty-state"><h3>欢迎体验 RAG 混合检索问答</h3><p>在下方输入框提问，我将基于知识库为您解答。</p></div>
               ) : (
                 messages.map(m => (
                   <div key={m.id} className="message-wrapper">
                     <div className={`message ${m.role === 'user' ? 'message-user' : 'message-ai'}`}>{m.content}</div>
                     {m.citations?.items && (
                       <div className="citations">
                         {m.citations.items.map((c, i) => <div key={i} className="citation-item">[{c.index}] {c.file_name}</div>)}
                       </div>
                     )}
                   </div>
                 ))
               )}
               <div ref={messagesEndRef} />
             </div>
             <div className="input-container">
               <div className="input-form">
                 <textarea placeholder="输入问题..." value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSendChat())}/>
                 <button onClick={handleSendChat} disabled={!chatInput.trim() || isGenerating}>发送</button>
               </div>
             </div>
          </div>
        )}

        {activePage === 'demo' && (
          <div className="demo-view scroll-view">
            <header className="page-header">
               <h2>流程演示</h2>
               <p>零配置体验：一键跑通“文档下载 -> 索引入库 -> 检索召回 -> LLM回答”全流程。</p>
            </header>
            <section className="config-card">
               <div className="input-group">
                 <label>选择文档</label>
                 <select value={selectedDocId} onChange={e => setSelectedDocId(e.target.value)}>
                   {DEMO_DOC_LIBRARY.map(d => <option key={d.id} value={d.id}>{d.title}</option>)}
                 </select>
               </div>
               <div className="chip-list">
                 {selectedDoc.examples.map(ex => <button key={ex} className="chip" onClick={() => setQuestion(ex)}>{ex}</button>)}
               </div>
               <div className="input-group">
                 <label>测试问题</label>
                 <textarea value={question} onChange={e => setQuestion(e.target.value)} />
               </div>
               <button className="primary-btn" onClick={runDemoFlow} disabled={running}>
                 {running ? '正在执行流程...' : '一键跑通测试流程'}
               </button>
            </section>

            <section className="workflow">
               <h3>流程可视化</h3>
               <div className="step-grid">
                 {steps.map(s => <StepCard key={s.id} step={s} />)}
               </div>
            </section>

            {retrieval && (
              <section className="trace-view">
                <h3>检索指标可视化</h3>
                <div className="metric-row">
                   <span>Keyword: {retrieval.trace.keyword?.count}</span>
                   <span>Vector: {retrieval.trace.vector?.count}</span>
                   <span>Fusion: {retrieval.trace.fusion?.count}</span>
                   <span>Total Time: {formatMs(retrieval.trace.timing_ms?.total)}</span>
                </div>
                <div className="trace-grid">
                   <HitList title="关键词召回" hits={retrieval.trace.keyword?.hits} />
                   <HitList title="向量召回" hits={retrieval.trace.vector?.hits} />
                </div>
              </section>
            )}
          </div>
        )}

        {activePage === 'eval' && (
          <div className="eval-view scroll-view">
            <header className="page-header">
              <h2>DuRetrieval 评测</h2>
              <p>基于标准数据集的召回率与 MRR 指标分析。</p>
            </header>
            <div className="dataset-grid">
              {evalDocs.map(d => (
                <div key={d.id} className="mini-card">
                  <h4>{d.title}</h4>
                  <p className="muted">{d.filePath}</p>
                  <div className="status-row">
                    <span className={d.exists ? 'ok' : 'err'}>文件:{d.exists ? '存在' : '缺失'}</span>
                    <span className={d.indexed ? 'ok' : 'err'}>索引:{d.indexed ? '已完成' : '未入库'}</span>
                  </div>
                </div>
              ))}
            </div>
            <BaselineReportPanel latest={latestReport} items={evalReports} />
          </div>
        )}
      </main>
    </div>
  );
}
