import React from 'react';
import { Handle, Position } from 'reactflow';

function short(s, n = 36) {
  if (!s && s !== 0) return '';
  const str = String(s);
  return str.length > n ? str.slice(0, n - 1) + '…' : str;
}

function summaryFromData(data) {
  const label = data?.label || '';
  const params = data?.params || {};
  try {
    if (label.includes('Filter')) {
      const cls = params.filter_class || params.filter || '';
      return short(cls.split('.').pop() || cls || 'filter');
    }
    if (label.includes('Router')) {
      const cls = params.router_class || params.router || '';
      return short(cls.split('.').pop() || cls || 'router');
    }
    if (label === 'Task') {
      const cls = params.task_class || params.task || '';
      return short(cls.split('.').pop() || cls || 'task');
    }
    if (label === 'CandidateLLMs') {
      const list = params.candidate_llms || params.candidate || [];
      if (Array.isArray(list)) return short(list.join(', '), 40);
      return short(list, 40);
    }
    if (label === 'LoadData') {
      const ds = params.dataset || params.samples || '';
      return short(ds || `max:${params.max_samples || ''}`);
    }
    if (label === 'Annotate' || label === 'Annotator') {
      const task = params.task_class || params.task || '';
      return short(task ? task.split('.').pop() : '', 40);
    }
    if (label === 'Output') {
      return short(params.path || params.output || 'out');
    }
    const keys = Object.keys(params || {});
    return keys.length ? short(keys.slice(0, 3).join(', ')) : '';
  } catch (e) {
    return '';
  }
}

function typeColor(label) {
  const map = {
    LoadData: '#E6F7FF',
    Task: '#FFF7E6',
    CandidateLLMs: '#F0F5FF',
    Filter: '#F6FFED',
    Router: '#FFF0F6',
    Annotate: '#FFF9E6',
    Output: '#F0FFF4',
  };
  return map[label] || '#ffffff';
}

function typeAccent(label) {
  const map = {
    LoadData: '#1890FF',
    Task: '#FA8C16',
    CandidateLLMs: '#2F54EB',
    Filter: '#52C41A',
    Router: '#EB2F96',
    Annotate: '#D48806',
    Output: '#13C2C2',
  };
  return map[label] || '#999';
}

export default function CompactNode({ id, data }) {
  const title = data?.label || id;
  const summary = summaryFromData(data);
  const isCandidate = title === 'CandidateLLMs';
  const candList = isCandidate && data?.params?.candidate_llms ? data.params.candidate_llms : null;
  const bg = typeColor(title);
  const accent = typeAccent(title);

  const icon = (label) => {
    const stroke = 'var(--accent)';
    switch (label) {
      case 'LoadData':
        return (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 3v10" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M8 7l4-4 4 4" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
        );
      case 'Task':
        return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" stroke={stroke} strokeWidth="1.2"/></svg>);
      case 'CandidateLLMs':
        return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><circle cx="7" cy="7" r="3" stroke={stroke} strokeWidth="1.2"/><circle cx="17" cy="17" r="3" stroke={stroke} strokeWidth="1.2"/></svg>);
      case 'Filter':
        return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M4 6h16" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/><path d="M10 12h4" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/><path d="M6 18h12" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/></svg>);
      case 'Router':
        return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 3v18" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/><path d="M4 10h8" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/><path d="M12 14h8" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/></svg>);
      case 'Annotate':
        return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M3 21l3-3 11-11 3 3L7 21H3z" stroke={stroke} strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/></svg>);
      case 'Output':
        return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M4 12h16" stroke={stroke} strokeWidth="1.5" strokeLinecap="round"/><path d="M12 5l7 7-7 7" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>);
      default:
        return null;
    }
  };

  return (
    <div className="compact-node p-2 rounded-md min-w-[140px] shadow-sm" data-id={id} style={{ ['--accent']: accent, ['--bg']: bg }}>
      <Handle type="target" position={Position.Left} className="bg-gray-700" />

      <div className="flex items-center gap-2.5 text-[13px] font-bold mb-0 text-[#0f172a]">
        <div className="w-[18px] h-[18px] flex items-center justify-center">{icon(title)}</div>
        <div>{title}</div>
      </div>

      {isCandidate && Array.isArray(candList) ? (
        <div className="flex flex-col gap-1 mt-2">
          {candList.slice(0, 6).map((c, i) => (
            <div key={i} className="text-[11px] text-[#333] truncate">{c}</div>
          ))}
          {candList.length > 6 ? (
            <div className="text-[11px] text-[#777]">+{candList.length - 6} more</div>
          ) : null}
        </div>
      ) : (
        <>
          {summary ? <div className="text-[12px] text-gray-600 mt-2">{summary}</div> : null}

          {(['Filter', 'Router', 'Annotate'].includes(title) && data && data.progress) ? (
            (() => {
              const pr = data.progress || {};
              const curr = pr.current || 0;
              const total = pr.total || 0;
              const pct = total > 0 ? Math.min(100, Math.round((curr / total) * 100)) : 0;
              return (
                <div className="mt-2">
                  <div className="h-2 bg-gray-200 rounded overflow-hidden">
                    <div className="h-full bg-[var(--accent)]" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="text-[11px] text-gray-500 mt-1">{curr}/{total} ({pct}%)</div>
                </div>
              );
            })()
          ) : null}
        </>
      )}

      <Handle type="source" position={Position.Right} className="bg-gray-700" />
    </div>
  );
}
