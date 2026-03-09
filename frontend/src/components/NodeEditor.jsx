import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Button from './ui/Button';

const TYPE_COLORS = {
  LoadData: '#3b82f6', Task: '#f97316', CandidateLLMs: '#8b5cf6',
  Filter: '#22c55e', Router: '#ec4899', Annotate: '#f59e0b', Output: '#14b8a6',
};

function Field({ label, help, error, children }) {
  return (
    <div className="field-group">
      <label className="field-label">{label}</label>
      {children}
      {help  && <div className="field-help">{help}</div>}
      {error && <div className="field-error">{error}</div>}
    </div>
  );
}

function TextInput({ value, onChange, placeholder, list }) {
  return (
    <input
      className="field-input"
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      list={list}
    />
  );
}

function NumberInput({ value, onChange, min, max, step, placeholder }) {
  return (
    <input
      type="number"
      className="field-input"
      value={value}
      onChange={onChange}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
    />
  );
}

function Textarea({ value, onChange, rows = 5, placeholder }) {
  return (
    <textarea
      className="field-input field-textarea"
      value={value}
      onChange={onChange}
      rows={rows}
      placeholder={placeholder}
    />
  );
}

function Select({ value, onChange, children }) {
  return (
    <select className="field-input field-select" value={value} onChange={onChange}>
      {children}
    </select>
  );
}

function JsonEditor({ value, onChange, rows = 5 }) {
  const [txt, setTxt] = useState(() => JSON.stringify(value || {}, null, 2));
  const [err, setErr] = useState(null);

  useEffect(() => {
    setTxt(JSON.stringify(value || {}, null, 2));
  }, [value]);

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
      <Textarea
        value={txt}
        onChange={(e) => handle(e.target.value)}
        rows={rows}
        placeholder="{}"
      />
      {err && <div className="field-error">JSON error: {err}</div>}
    </div>
  );
}

function ClassInfoBox({ name, info }) {
  if (!name) return null;
  if (!info) return <div className="field-help" style={{ marginTop: 6 }}>Loading class info…</div>;
  if (info.error) return <div className="field-error" style={{ marginTop: 6 }}>Error: {info.error}</div>;
  return (
    <div className="class-info-box">
      {info.doc && <div style={{ marginBottom: 6 }}>{info.doc}</div>}
      {info.init_params?.length > 0 && (
        <div style={{ marginBottom: 4 }}>
          <strong>Init params:</strong> <span style={{ color: '#6366f1' }}>{info.init_params.join(', ')}</span>
        </div>
      )}
      {info.methods?.length > 0 && (
        <div>
          <strong>Methods:</strong>
          {info.methods.slice(0, 5).map((m) => (
            <div key={m.name} style={{ marginTop: 2, paddingLeft: 8 }}>
              <code style={{ fontSize: 11 }}>{m.name}()</code>
              {m.doc ? <span style={{ color: '#64748b' }}> — {m.doc.split('\n')[0]}</span> : null}
            </div>
          ))}
          {info.methods.length > 5 && <div style={{ color: '#94a3b8', fontSize: 11 }}>…and {info.methods.length - 5} more</div>}
        </div>
      )}
    </div>
  );
}

export default function NodeEditor({ node, updateNodeData, deleteNode, openConfirm }) {
  const [params, setParams] = useState(node.data?.params || {});
  const label = node.data?.label || node.type || 'Node';
  const accent = TYPE_COLORS[label] || '#6366f1';

  useEffect(() => {
    setParams(node.data?.params || {});
  }, [node]);

  const [classLists, setClassLists] = useState({ filters: [], routers: [], tasks: [] });
  useEffect(() => {
    let mounted = true;
    axios.get('http://localhost:5000/list_classes').then((res) => {
      if (mounted) setClassLists(res.data || { filters: [], routers: [], tasks: [] });
    }).catch(() => {});
    return () => { mounted = false; };
  }, []);

  const [classInfo, setClassInfo] = useState({});
  const fetchClassInfo = async (name) => {
    if (!name || classInfo[name]) return;
    try {
      const res = await axios.get(`http://localhost:5000/class_info?class=${encodeURIComponent(name)}`);
      setClassInfo((s) => ({ ...s, [name]: res.data }));
    } catch (e) {
      setClassInfo((s) => ({ ...s, [name]: { error: e.message } }));
    }
  };

  useEffect(() => {
    fetchClassInfo(params.filter_class);
    fetchClassInfo(params.router_class);
    fetchClassInfo(params.task_class);
  }, [params.filter_class, params.router_class, params.task_class]);

  const set = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  const handleSave = () => updateNodeData(node.id, { params });

  const renderForm = () => {
    switch (label) {
      case 'LoadData':
        return (
          <>
            <Field label="Dataset name or path">
              <TextInput value={params.dataset || ''} onChange={(e) => set('dataset', e.target.value)} placeholder="e.g. squad" />
            </Field>
            <Field label="Max samples">
              <NumberInput value={params.max_samples || ''} onChange={(e) => set('max_samples', Number(e.target.value) || 0)} placeholder="e.g. 200" />
            </Field>
            <Field label="Inline samples (JSON list)" help="Optional: provide samples directly as a JSON array">
              <JsonEditor value={params.samples || []} onChange={(v) => set('samples', v)} rows={6} />
            </Field>
          </>
        );

      case 'CandidateLLMs':
        return (
          <Field label="Candidate LLMs" help="One model name per line">
            <Textarea
              value={(params.candidate_llms || []).join('\n')}
              onChange={(e) => set('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))}
              rows={6}
              placeholder={"gpt2\ndistilgpt2"}
            />
          </Field>
        );

      case 'Task':
        const taskVal = params.task_class || '';
        const taskValid = !taskVal || classLists.tasks.includes(taskVal);
        return (
          <>
            <Field
              label="Task class"
              help="Full Python module path, e.g. tasks.qa.QATask"
              error={!taskValid ? 'Class not found in repository scan' : null}
            >
              <TextInput value={taskVal} onChange={(e) => set('task_class', e.target.value)} placeholder="tasks.qa.QATask" list="tasks-list" />
              <ClassInfoBox name={taskVal} info={classInfo[taskVal]} />
            </Field>
            <Field label="Task params (JSON)">
              <JsonEditor value={params.task_params || {}} onChange={(v) => set('task_params', v)} rows={5} />
            </Field>
          </>
        );

      case 'Filter':
        const filterVal = params.filter_class || '';
        const filterValid = !filterVal || classLists.filters.includes(filterVal);
        return (
          <>
            <Field
              label="Filter class"
              help="e.g. filters.al_filter.ActiveLearningFilter"
              error={!filterValid ? 'Class not found in repository scan' : null}
            >
              <TextInput value={filterVal} onChange={(e) => set('filter_class', e.target.value)} placeholder="filters.al_filter.ActiveLearningFilter" list="filters-list" />
              <ClassInfoBox name={filterVal} info={classInfo[filterVal]} />
            </Field>
            <Field label="Filter params (JSON)">
              <JsonEditor value={params.filter_params || {}} onChange={(v) => set('filter_params', v)} rows={5} />
            </Field>
          </>
        );

      case 'Router':
        const routerVal = params.router_class || '';
        const routerValid = !routerVal || classLists.routers.includes(routerVal);
        return (
          <>
            <Field
              label="Router class"
              help="e.g. routers.knn_router.KNNRouter"
              error={!routerValid ? 'Class not found in repository scan' : null}
            >
              <TextInput value={routerVal} onChange={(e) => set('router_class', e.target.value)} placeholder="routers.knn_router.KNNRouter" list="routers-list" />
              <ClassInfoBox name={routerVal} info={classInfo[routerVal]} />
            </Field>
            <Field label="Router params (JSON)">
              <JsonEditor value={params.router_params || {}} onChange={(v) => set('router_params', v)} rows={5} />
            </Field>
            <Field label="Candidate LLMs (optional)" help="One model name per line">
              <Textarea
                value={(params.candidate_llms || []).join('\n')}
                onChange={(e) => set('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))}
                rows={4}
                placeholder={"gpt2\ndistilgpt2"}
              />
            </Field>
          </>
        );

      case 'Annotate':
      case 'Annotation':
        const annotateTaskVal = params.task_class || '';
        const annotateTaskValid = !annotateTaskVal || classLists.tasks.includes(annotateTaskVal);
        return (
          <>
            <Field label="Candidate LLMs" help="One model name per line">
              <Textarea
                value={(params.candidate_llms || []).join('\n')}
                onChange={(e) => set('candidate_llms', e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))}
                rows={4}
                placeholder={"gpt2\ndistilgpt2"}
              />
            </Field>
            <Field label="LLM mode">
              <Select value={params.llm_mode || 'local'} onChange={(e) => set('llm_mode', e.target.value)}>
                <option value="local">local</option>
                <option value="api">api</option>
              </Select>
            </Field>
            <Field label="API config (JSON)" help="Used when LLM mode is 'api'">
              <JsonEditor value={params.api_config || {}} onChange={(v) => set('api_config', v)} rows={5} />
            </Field>
            <Field
              label="Task class"
              error={!annotateTaskValid ? 'Class not found in repository scan' : null}
            >
              <TextInput value={annotateTaskVal} onChange={(e) => set('task_class', e.target.value)} placeholder="tasks.qa.QATask" list="tasks-list" />
              <ClassInfoBox name={annotateTaskVal} info={classInfo[annotateTaskVal]} />
            </Field>
            <Field label="Min confidence" help="Optional threshold 0–1">
              <NumberInput
                value={params.min_confidence !== undefined ? params.min_confidence : ''}
                onChange={(e) => {
                  const v = e.target.value;
                  set('min_confidence', v === '' ? undefined : Number(v));
                }}
                min={0} max={1} step={0.01}
                placeholder="0.0"
              />
            </Field>
          </>
        );

      case 'Output':
        return (
          <Field label="Output file path" help="e.g. out/annotations.json">
            <TextInput value={params.path || ''} onChange={(e) => set('path', e.target.value)} placeholder="out/annotations.json" />
          </Field>
        );

      default:
        return (
          <Field label="Parameters (JSON)">
            <JsonEditor value={params || {}} onChange={(v) => setParams(v)} rows={8} />
          </Field>
        );
    }
  };

  // Hidden datalists for autocomplete
  const datalists = (
    <div style={{ display: 'none' }}>
      <datalist id="filters-list">{classLists.filters.map((c) => <option key={c} value={c} />)}</datalist>
      <datalist id="routers-list">{classLists.routers.map((c) => <option key={c} value={c} />)}</datalist>
      <datalist id="tasks-list">{classLists.tasks.map((c) => <option key={c} value={c} />)}</datalist>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {datalists}

      {/* Header */}
      <div className="node-editor-header">
        <div className="node-editor-type-badge" style={{ background: accent + '18', color: accent }}>
          {label}
        </div>
        <div style={{ fontSize: 11, color: '#94a3b8', marginLeft: 'auto' }}>id: {node.id.slice(0, 8)}</div>
      </div>

      {/* Form */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
        {renderForm()}
      </div>

      {/* Actions */}
      <div className="node-editor-actions">
        <Button variant="ghost" onClick={() => setParams(node.data?.params || {})}>Reset</Button>
        <Button
          danger
          onClick={() => {
            if (openConfirm) openConfirm(node.id, `Delete node "${node.data?.label || node.id}"?`);
            else if (deleteNode) deleteNode(node.id);
          }}
        >
          Delete
        </Button>
        <div style={{ flex: 1 }} />
        <Button variant="primary" onClick={handleSave}>Save changes</Button>
      </div>
    </div>
  );
}
