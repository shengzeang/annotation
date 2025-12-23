import React, { useState, useEffect } from 'react';
import axios from 'axios';

function JsonEditor({ value, onChange, rows = 6 }) {
  const [txt, setTxt] = useState(() => JSON.stringify(value || {}, null, 2));
  const [err, setErr] = useState(null);

  useEffect(() => setTxt(JSON.stringify(value || {}, null, 2)), [value]);

  const handle = (v) => {
    setTxt(v);
    try {
      const parsed = JSON.parse(v);
      setErr(null);
      onChange(parsed);
    } catch (e) {
      setErr(e.message);
    }
  };

  return (
    <div>
      <textarea rows={rows} value={txt} onChange={(e) => handle(e.target.value)} style={{ width: '100%' }} />
      {err && <div style={{ color: 'crimson', fontSize: 12 }}>JSON error: {err}</div>}
    </div>
  );
}

export default function NodeEditor({ node, updateNodeData, deleteNode, openConfirm }) {
  const [params, setParams] = useState(node.data?.params || {});
  const label = node.data?.label || node.type || 'Node';

  useEffect(() => {
    setParams(node.data?.params || {});
  }, [node]);

  const [classLists, setClassLists] = useState({ filters: [], routers: [], tasks: [] });
  useEffect(() => {
    let mounted = true;
    axios.get('http://localhost:5000/list_classes').then((res) => {
      if (!mounted) return;
      setClassLists(res.data || { filters: [], routers: [], tasks: [] });
    }).catch(() => {});
    return () => { mounted = false };
  }, []);

  const [classInfo, setClassInfo] = useState({});
  const fetchClassInfo = async (name) => {
    if (!name) return;
    if (classInfo[name]) return; // cached
    try {
      const res = await axios.get(`http://localhost:5000/class_info?class=${encodeURIComponent(name)}`);
      setClassInfo((s) => ({ ...s, [name]: res.data }));
    } catch (e) {
      setClassInfo((s) => ({ ...s, [name]: { error: e.message } }));
    }
  };

  // fetch info for relevant fields when they change
  useEffect(() => {
    fetchClassInfo(params.filter_class);
    fetchClassInfo(params.router_class);
    fetchClassInfo(params.task_class);
  }, [params.filter_class, params.router_class, params.task_class]);

  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));

  const handleSave = () => updateNodeData(node.id, { params });

  // Per-node typed forms
  const renderForm = () => {
    const renderClassHelp = (name) => {
      if (!name) return null;
      const info = classInfo[name];
      if (!info) return <div style={{ fontSize: 12, color: '#666', marginTop: 6 }}>Loading class info...</div>;
      if (info.error) return <div style={{ fontSize: 12, color: 'crimson', marginTop: 6 }}>Error fetching class info: {info.error}</div>;
      return (
        <div style={{ marginTop: 8, padding: 8, background: '#fafafa', border: '1px solid #eee', fontSize: 13 }}>
          {info.doc ? <div style={{ marginBottom: 8 }}>{info.doc}</div> : null}
          {info.init_params && info.init_params.length > 0 ? (
            <div style={{ marginBottom: 6 }}><strong>__init__ params:</strong> {info.init_params.join(', ')}</div>
          ) : null}
          {info.methods && info.methods.length > 0 ? (
            <div>
              <strong>Methods</strong>
              <ul style={{ marginTop: 6 }}>
                {info.methods.slice(0, 6).map((m) => (
                  <li key={m.name}><strong>{m.name}()</strong>{m.doc ? ` — ${m.doc.split('\n')[0]}` : ''}</li>
                ))}
                {info.methods.length > 6 ? <li>...more</li> : null}
              </ul>
            </div>
          ) : null}
        </div>
      );
    };
    switch (label) {
      case 'LoadData':
        return (
          <div>
            <div>
              <label>Dataset name or file path</label>
              <input value={params.dataset || ''} onChange={(e) => setParam('dataset', e.target.value)} style={{ width: '100%' }} />
            </div>
            <div style={{ marginTop: 8 }}>
              <label>Max samples</label>
              <input type="number" value={params.max_samples || ''} onChange={(e) => setParam('max_samples', Number(e.target.value) || 0)} style={{ width: '100%' }} />
            </div>
            <div style={{ marginTop: 8 }}>
              <label>Inline samples (JSON list) — optional</label>
              <JsonEditor value={params.samples || []} onChange={(v) => setParam('samples', v)} rows={8} />
            </div>
          </div>
        );

      case 'CandidateLLMs':
        return (
          <div>
            <label>Candidate LLMs (one per line)</label>
            <textarea value={(params.candidate_llms || []).join('\n')} onChange={(e) => setParam('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} rows={6} style={{ width: '100%' }} />
          </div>
        );

      case 'Task':
        const taskVal = params.task_class || '';
        const taskValid = !taskVal || classLists.tasks.includes(taskVal);
        return (
          <div>
            <label>Task class (module.ClassName)</label>
            <input list="tasks-list" value={params.task_class || ''} placeholder="e.g. tasks.QATask" onChange={(e) => setParam('task_class', e.target.value)} style={{ width: '100%' }} />
            <datalist id="tasks-list">
              {classLists.tasks.map((c) => <option key={c} value={c} />)}
            </datalist>
            <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>Help: Choose the Task class that generates prompts and parses outputs for annotation.</div>
            {!taskValid && <div style={{ color: 'crimson', fontSize: 12 }}>Task class not found in repository scan.</div>}
            {renderClassHelp(taskVal)}
            <div style={{ marginTop: 8 }}>
              <label>Task params (JSON)</label>
              <JsonEditor value={params.task_params || {}} onChange={(v) => setParam('task_params', v)} rows={6} />
            </div>
          </div>
        );

      case 'Filter':
        const filterVal = params.filter_class || '';
        const filterValid = !filterVal || classLists.filters.includes(filterVal);
        return (
          <div>
            <label>Filter class (module.ClassName)</label>
            <input list="filters-list" value={params.filter_class || ''} placeholder="e.g. filters.al_filter.ActiveLearningFilter" onChange={(e) => setParam('filter_class', e.target.value)} style={{ width: '100%' }} />
            <datalist id="filters-list">
              {classLists.filters.map((c) => <option key={c} value={c} />)}
            </datalist>
            <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>Help: Enter the full python module path to the filter class, e.g. <code>filters.al_filter.ActiveLearningFilter</code>.</div>
            {!filterValid && <div style={{ color: 'crimson', fontSize: 12 }}>Class not found in repository scan.</div>}
            {renderClassHelp(filterVal)}
            <div style={{ marginTop: 8 }}>
              <label>Filter params (JSON)</label>
              <JsonEditor value={params.filter_params || {}} onChange={(v) => setParam('filter_params', v)} rows={6} />
            </div>
          </div>
        );

      case 'Router':
        const routerVal = params.router_class || '';
        const routerValid = !routerVal || classLists.routers.includes(routerVal);
        return (
          <div>
            <label>Router class (module.ClassName)</label>
            <input list="routers-list" value={params.router_class || ''} placeholder="e.g. routers.knn_router.KNNRouter" onChange={(e) => setParam('router_class', e.target.value)} style={{ width: '100%' }} />
            <datalist id="routers-list">
              {classLists.routers.map((c) => <option key={c} value={c} />)}
            </datalist>
            <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>Help: Pick or type the router class to use for routing logic.</div>
            {!routerValid && <div style={{ color: 'crimson', fontSize: 12 }}>Router class not found in repository scan.</div>}
            {renderClassHelp(routerVal)}
            <div style={{ marginTop: 8 }}>
              <label>Router params (JSON)</label>
              <JsonEditor value={params.router_params || {}} onChange={(v) => setParam('router_params', v)} rows={6} />
            </div>
            <div style={{ marginTop: 8 }}>
              <label>Candidate LLMs (optional)</label>
                <textarea value={(params.candidate_llms || []).join('\n')} placeholder="e.g. Qwen/Qwen2.5-7B-Instruct" onChange={(e) => setParam('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} rows={4} style={{ width: '100%' }} />
            </div>
          </div>
        );

      case 'Annotate':
      case 'Annotation':
        return (
          <div>
            <label>Candidate LLMs (one per line)</label>
            <textarea value={(params.candidate_llms || []).join('\n')} onChange={(e) => setParam('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} rows={4} style={{ width: '100%' }} />
            <div style={{ marginTop: 8 }}>
              <label>LLM mode</label>
              <select value={params.llm_mode || 'local'} onChange={(e) => setParam('llm_mode', e.target.value)} style={{ width: '100%' }}>
                <option value="local">local</option>
                <option value="api">api</option>
              </select>
            </div>
            <div style={{ marginTop: 8 }}>
              <label>API config (JSON) — used if llm_mode=api</label>
              <JsonEditor value={params.api_config || {}} onChange={(v) => setParam('api_config', v)} rows={6} />
            </div>
            <div style={{ marginTop: 8 }}>
              <label>Task class</label>
              {(() => {
                const tval = params.task_class || '';
                const valid = !tval || classLists.tasks.includes(tval);
                return (
                  <div>
                    <input list="tasks-list" value={params.task_class || ''} placeholder="e.g. tasks.QATask" onChange={(e) => setParam('task_class', e.target.value)} style={{ width: '100%' }} />
                    {!valid && <div style={{ color: 'crimson', fontSize: 12 }}>Task class not found in repository scan.</div>}
                  </div>
                );
              })()}
            {renderClassHelp(params.task_class)}
            </div>
          </div>
        );

      case 'Output':
        return (
          <div>
            <label>Output path (optional)</label>
            <input value={params.path || ''} onChange={(e) => setParam('path', e.target.value)} style={{ width: '100%' }} />
          </div>
        );

      default:
        return (
          <div>
            <label>Parameters (JSON)</label>
            <JsonEditor value={params || {}} onChange={(v) => setParams(v)} rows={8} />
          </div>
        );
    }
  };

  // Global datalists so inputs can reference them even if their node form isn't mounted
  const globalDatalists = (
    <div style={{ display: 'none' }}>
      <datalist id="filters-list">
        {classLists.filters.map((c) => <option key={c} value={c} />)}
      </datalist>
      <datalist id="routers-list">
        {classLists.routers.map((c) => <option key={c} value={c} />)}
      </datalist>
      <datalist id="tasks-list">
        {classLists.tasks.map((c) => <option key={c} value={c} />)}
      </datalist>
    </div>
  );

  return (
    <div className="node-editor">
      {globalDatalists}
      <div className="node-meta">
        <h3 style={{ margin: 0 }}>Node</h3>
        <div className="muted">{label}</div>
      </div>

      <div style={{ marginTop: 4 }}>{renderForm()}</div>

      <div className="actions">
        <button className="btn btn-secondary" onClick={() => { setParams(node.data?.params || {}); }}>Reset</button>
        <button className="btn btn-danger" onClick={() => { openConfirm ? openConfirm(node.id, `Delete node "${node.data?.label || node.id}"?`) : (deleteNode && deleteNode(node.id)); }} style={{ marginLeft: 8 }}>Delete</button>
        <button className="btn btn-primary" onClick={handleSave} style={{ marginLeft: 8 }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ marginRight: 6 }}>
            <path d="M5 12h14" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 5v14" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Save
        </button>
      </div>
    </div>
  );
}
