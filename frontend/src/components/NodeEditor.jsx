import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Button from './ui/Button';
import Card from './ui/Card';

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
      <textarea rows={rows} value={txt} onChange={(e) => handle(e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
      {err && <div className="text-xs text-red-600">JSON error: {err}</div>}
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
      if (!info) return <div className="text-xs text-gray-600 mt-1.5">Loading class info...</div>;
      if (info.error) return <div className="text-xs text-red-600 mt-1.5">Error fetching class info: {info.error}</div>;
      return (
        <div className="mt-2 p-2 bg-gray-50 border border-gray-200 text-sm">
          {info.doc ? <div className="mb-2">{info.doc}</div> : null}
          {info.init_params && info.init_params.length > 0 ? (
            <div className="mb-1.5"><strong>__init__ params:</strong> {info.init_params.join(', ')}</div>
          ) : null}
          {info.methods && info.methods.length > 0 ? (
            <div>
              <strong>Methods</strong>
              <ul className="mt-1.5">
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
              <input value={params.dataset || ''} onChange={(e) => setParam('dataset', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
            </div>
            <div className="mt-2">
              <label>Max samples</label>
              <input type="number" value={params.max_samples || ''} onChange={(e) => setParam('max_samples', Number(e.target.value) || 0)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
            </div>
            <div className="mt-2">
              <label>Inline samples (JSON list) — optional</label>
              <JsonEditor value={params.samples || []} onChange={(v) => setParam('samples', v)} rows={8} />
            </div>
          </div>
        );

      case 'CandidateLLMs':
        return (
            <div>
            <label>Candidate LLMs (one per line)</label>
            <textarea value={(params.candidate_llms || []).join('\n')} onChange={(e) => setParam('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} rows={6} className="w-full border border-gray-200 rounded p-2 bg-white" />
          </div>
        );

      case 'Task':
        const taskVal = params.task_class || '';
        const taskValid = !taskVal || classLists.tasks.includes(taskVal);
        return (
          <div>
            <label>Task class (module.ClassName)</label>
            <input list="tasks-list" value={params.task_class || ''} placeholder="e.g. tasks.QATask" onChange={(e) => setParam('task_class', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
            <datalist id="tasks-list">
              {classLists.tasks.map((c) => <option key={c} value={c} />)}
            </datalist>
            <div className="text-xs text-gray-600 mt-1.5">Help: Choose the Task class that generates prompts and parses outputs for annotation.</div>
            {!taskValid && <div className="text-xs text-red-600">Task class not found in repository scan.</div>}
            {renderClassHelp(taskVal)}
            <div className="mt-2">
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
            <input list="filters-list" value={params.filter_class || ''} placeholder="e.g. filters.al_filter.ActiveLearningFilter" onChange={(e) => setParam('filter_class', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
            <datalist id="filters-list">
              {classLists.filters.map((c) => <option key={c} value={c} />)}
            </datalist>
            <div className="text-xs text-gray-600 mt-1.5">Help: Enter the full python module path to the filter class, e.g. <code>filters.al_filter.ActiveLearningFilter</code>.</div>
            {!filterValid && <div className="text-xs text-red-600">Class not found in repository scan.</div>}
            {renderClassHelp(filterVal)}
            <div className="mt-2">
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
            <input list="routers-list" value={params.router_class || ''} placeholder="e.g. routers.knn_router.KNNRouter" onChange={(e) => setParam('router_class', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
            <datalist id="routers-list">
              {classLists.routers.map((c) => <option key={c} value={c} />)}
            </datalist>
            <div className="text-xs text-gray-600 mt-1.5">Help: Pick or type the router class to use for routing logic.</div>
            {!routerValid && <div className="text-xs text-red-600">Router class not found in repository scan.</div>}
            {renderClassHelp(routerVal)}
            <div className="mt-2">
              <label>Router params (JSON)</label>
              <JsonEditor value={params.router_params || {}} onChange={(v) => setParam('router_params', v)} rows={6} />
            </div>
            <div className="mt-2">
              <label>Candidate LLMs (optional)</label>
                <textarea value={(params.candidate_llms || []).join('\n')} placeholder={"e.g. gpt2\ndistilgpt2"} onChange={(e) => setParam('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} rows={4} className="w-full border border-gray-200 rounded p-2 bg-white" />
            </div>
          </div>
        );

      case 'Annotate':
      case 'Annotation':
        return (
          <div>
            <label>Candidate LLMs (one per line)</label>
            <textarea value={(params.candidate_llms || []).join('\n')} onChange={(e) => setParam('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} rows={4} className="w-full border border-gray-200 rounded p-2 bg-white" />
            <div className="mt-2">
              <label>LLM mode</label>
              <select value={params.llm_mode || 'local'} onChange={(e) => setParam('llm_mode', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white">
                <option value="local">local</option>
                <option value="api">api</option>
              </select>
            </div>
            <div className="mt-2">
              <label>API config (JSON) — used if llm_mode=api</label>
              <JsonEditor value={params.api_config || {}} onChange={(v) => setParam('api_config', v)} rows={6} />
            </div>
            <div className="mt-2">
              <label>Task class</label>
              {(() => {
                const tval = params.task_class || '';
                const valid = !tval || classLists.tasks.includes(tval);
                return (
                  <div>
                    <input list="tasks-list" value={params.task_class || ''} placeholder="e.g. tasks.QATask" onChange={(e) => setParam('task_class', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
                    {!valid && <div className="text-xs text-red-600">Task class not found in repository scan.</div>}
                  </div>
                );
              })()}
            {renderClassHelp(params.task_class)}
            </div>
            <div className="mt-2">
              <label>Min confidence (optional, 0-1)</label>
              <input type="number" step="0.01" min="0" max="1" value={params.min_confidence !== undefined ? params.min_confidence : ''} onChange={(e) => {
                const v = e.target.value;
                if (v === '') setParam('min_confidence', undefined); else setParam('min_confidence', Number(v));
              }} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
            </div>
          </div>
        );

      case 'Output':
        return (
          <div>
              <label>Output path (optional)</label>
              <input value={params.path || ''} onChange={(e) => setParam('path', e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1 bg-white" />
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
    <div className="hidden">
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
    <Card className="node-editor">
      {globalDatalists}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Node</h3>
        <div className="text-sm text-slate-500">{label}</div>
      </div>

      <div className="mt-3">{renderForm()}</div>

      <div className="flex justify-end items-center gap-3 mt-4">
        <Button variant="ghost" onClick={() => { setParams(node.data?.params || {}); }}>Reset</Button>
        <Button variant="ghost" danger onClick={() => { openConfirm ? openConfirm(node.id, `Delete node \"${node.data?.label || node.id}\"?`) : (deleteNode && deleteNode(node.id)); }}>Delete</Button>
        <Button variant="primary" onClick={handleSave}>Save</Button>
      </div>
    </Card>
  );
}
