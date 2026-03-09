import React from 'react';
import { Handle, Position } from 'reactflow';

// Color/accent mapping per node type
const TYPE_META = {
  LoadData:      { color: '#3b82f6', bg: '#eff6ff' },
  Task:          { color: '#f97316', bg: '#fff7ed' },
  CandidateLLMs: { color: '#8b5cf6', bg: '#f5f3ff' },
  Filter:        { color: '#22c55e', bg: '#f0fdf4' },
  Router:        { color: '#ec4899', bg: '#fdf2f8' },
  Annotate:      { color: '#f59e0b', bg: '#fffbeb' },
  Output:        { color: '#14b8a6', bg: '#f0fdfa' },
};

function NodeIcon({ label, color }) {
  switch (label) {
    case 'LoadData':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          <polyline points="7 10 12 15 17 10" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          <line x1="12" y1="15" x2="12" y2="3" stroke={color} strokeWidth="2.2" strokeLinecap="round"/>
        </svg>
      );
    case 'Task':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="3" width="18" height="18" rx="3" stroke={color} strokeWidth="2.2"/>
          <path d="M9 12l2 2 4-4" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    case 'CandidateLLMs':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <circle cx="6" cy="6" r="2.5" stroke={color} strokeWidth="2"/>
          <circle cx="18" cy="6" r="2.5" stroke={color} strokeWidth="2"/>
          <circle cx="12" cy="18" r="2.5" stroke={color} strokeWidth="2"/>
          <line x1="6" y1="8.5" x2="12" y2="15.5" stroke={color} strokeWidth="1.5"/>
          <line x1="18" y1="8.5" x2="12" y2="15.5" stroke={color} strokeWidth="1.5"/>
        </svg>
      );
    case 'Filter':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
        </svg>
      );
    case 'Router':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <circle cx="5" cy="12" r="2" stroke={color} strokeWidth="2"/>
          <circle cx="19" cy="6" r="2" stroke={color} strokeWidth="2"/>
          <circle cx="19" cy="18" r="2" stroke={color} strokeWidth="2"/>
          <path d="M7 12h4l4-5" stroke={color} strokeWidth="1.8" strokeLinecap="round"/>
          <path d="M7 12h4l4 5" stroke={color} strokeWidth="1.8" strokeLinecap="round"/>
        </svg>
      );
    case 'Annotate':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <path d="M12 20h9" stroke={color} strokeWidth="2.2" strokeLinecap="round"/>
          <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    case 'Output':
      return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          <polyline points="17 8 12 3 7 8" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          <line x1="12" y1="3" x2="12" y2="15" stroke={color} strokeWidth="2.2" strokeLinecap="round"/>
        </svg>
      );
    default:
      return null;
  }
}

function truncate(s, n = 28) {
  if (!s && s !== 0) return '';
  const str = String(s);
  return str.length > n ? str.slice(0, n - 1) + '\u2026' : str;
}

function getSummary(data) {
  const label = data?.label || '';
  const p = data?.params || {};
  try {
    if (label === 'Filter')         return truncate((p.filter_class || '').split('.').pop() || 'filter');
    if (label === 'Router')         return truncate((p.router_class || '').split('.').pop() || 'router');
    if (label === 'Task')           return truncate((p.task_class || '').split('.').pop() || 'task');
    if (label === 'CandidateLLMs')  return truncate(Array.isArray(p.candidate_llms) ? p.candidate_llms.join(', ') : String(p.candidate_llms || ''), 32);
    if (label === 'LoadData')       return truncate(p.dataset || ('max: ' + (p.max_samples || '\u2014')));
    if (label === 'Annotate')       return truncate((p.task_class || '').split('.').pop() || '');
    if (label === 'Output')         return truncate(p.path || 'output');
    const keys = Object.keys(p || {});
    return keys.length ? truncate(keys.slice(0, 3).join(', ')) : '';
  } catch (e) { return ''; }
}

export default function CompactNode({ id, data }) {
  const label = data?.label || id;
  const meta  = TYPE_META[label] || { color: '#6366f1', bg: '#eef2ff' };
  const summary = getSummary(data);
  const isCandidates = label === 'CandidateLLMs';
  const candList = isCandidates ? (data?.params?.candidate_llms || []) : [];

  // Progress
  const progress = data?.progress;
  const pct = progress && progress.total > 0
    ? Math.min(100, Math.round((progress.current / progress.total) * 100))
    : null;

  return (
    <div className="compact-node">
      {/* Accent top bar */}
      <div className="node-accent-bar" style={{ background: meta.color }} />

      <div className="node-body">
        {/* Title row */}
        <div className="node-title">
          <div className="node-icon" style={{ background: meta.bg }}>
            <NodeIcon label={label} color={meta.color} />
          </div>
          <span>{label}</span>
        </div>

        {/* Summary / content */}
        {isCandidates ? (
          <div style={{ marginTop: 8 }}>
            {(Array.isArray(candList) ? candList : []).slice(0, 5).map((c, i) => (
              <div key={i} style={{ fontSize: 10.5, color: '#64748b', lineHeight: 1.6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {c}
              </div>
            ))}
            {candList.length > 5 && (
              <div style={{ fontSize: 10, color: '#94a3b8' }}>+{candList.length - 5} more</div>
            )}
          </div>
        ) : (
          <>
            {summary ? <div className="node-summary">{summary}</div> : null}
            {pct !== null && ['Filter', 'Router', 'Annotate'].includes(label) ? (
              <div className="node-progress">
                <div className="node-progress-bar">
                  <div className="node-progress-fill" style={{ width: pct + '%', background: meta.color }} />
                </div>
                <div className="node-progress-text">{progress.current}/{progress.total} &middot; {pct}%</div>
              </div>
            ) : null}
          </>
        )}
      </div>

      {/* ReactFlow handles */}
      <Handle type="target" position={Position.Left} style={{ background: meta.color, border: '2px solid #fff', width: 10, height: 10 }} />
      <Handle type="source" position={Position.Right} style={{ background: meta.color, border: '2px solid #fff', width: 10, height: 10 }} />
    </div>
  );
}
