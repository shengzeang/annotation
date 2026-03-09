import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Button from './ui/Button';

export default function ReviewQueue({ items, onUpdate }) {
  const [list, setList] = useState(items || []);
  const [sel, setSel] = useState(0);
  const [selectedSet, setSelectedSet] = useState(new Set());

  useEffect(() => { setList(items || []); setSel(0); }, [items]);

  const cur = list[sel] || null;

  const handleChange = (k, v) => {
    setList((s) => s.map((it, i) => i === sel ? { ...it, [k]: v } : it));
  };

  const toggleSelect = (index) => {
    setSelectedSet((s) => {
      const next = new Set(s);
      if (next.has(index)) next.delete(index); else next.add(index);
      return next;
    });
  };

  const submitSingle = async (sample, action) =>
    axios.post('http://localhost:5000/submit_review', { sample, action });

  const submitAction = async (action) => {
    if (!cur) return;
    try {
      await submitSingle(cur, action);
      const next = list.filter((_, i) => i !== sel);
      setList(next);
      setSel((s) => Math.min(s, Math.max(0, next.length - 1)));
      if (onUpdate) onUpdate(next);
    } catch (e) {
      console.error(e);
      alert('Failed to submit: ' + e.message);
    }
  };

  const submitBulk = async (action) => {
    const indices = Array.from(selectedSet).sort((a, b) => b - a);
    if (!indices.length) return alert('No items selected');
    try {
      const results = await Promise.all(indices.map((i) => submitSingle(list[i], action).catch((e) => ({ error: e }))));
      const successIndices = results.map((r, idx) => (r?.data?.status === 'ok' ? indices[idx] : null)).filter((x) => x !== null);
      const next = list.filter((_, i) => !successIndices.includes(i));
      setList(next);
      setSelectedSet(new Set());
      setSel((s) => Math.min(s, Math.max(0, next.length - 1)));
      if (onUpdate) onUpdate(next);
      alert(successIndices.length === indices.length ? 'Bulk action completed.' : 'Some submissions failed.');
    } catch (e) {
      console.error(e);
      alert('Bulk action failed: ' + e.message);
    }
  };

  // Empty state — matches target image
  if (!list.length) {
    return (
      <div className="rp-queue-empty">
        <svg className="rp-empty-icon" width="36" height="36" viewBox="0 0 24 24" fill="none">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke="#9ca3af" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          <polyline points="22 4 12 14.01 9 11.01" stroke="#9ca3af" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <div className="rp-empty-label">No items in review queue</div>
        <div className="rp-empty-sublabel">Items needing human review will appear here</div>
      </div>
    );
  }

  return (
    <div>
      {/* List */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 11.5, fontWeight: 700, color: '#374151' }}>{list.length} items</span>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            style={{ fontSize: 11, color: '#6366f1', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600, padding: '2px 0' }}
            onClick={() => setSelectedSet(new Set(list.map((_, i) => i)))}
          >Select all</button>
          <span style={{ color: '#e2e8f0' }}>|</span>
          <button
            style={{ fontSize: 11, color: '#94a3b8', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600, padding: '2px 0' }}
            onClick={() => setSelectedSet(new Set())}
          >Clear</button>
        </div>
      </div>

      <div style={{ maxHeight: 200, overflowY: 'auto', marginBottom: 12 }}>
        {list.map((it, i) => (
          <div
            key={i}
            className={'review-item' + (i === sel ? ' active' : '')}
            onClick={() => setSel(i)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="checkbox"
                checked={selectedSet.has(i)}
                onChange={() => toggleSelect(i)}
                onClick={(e) => e.stopPropagation()}
                style={{ accentColor: '#6366f1', flexShrink: 0 }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: '#1e293b' }}>{it.id || it.qid || `item ${i + 1}`}</span>
                  <span className="review-badge" style={it._server
                    ? { background: '#f0fdf4', color: '#16a34a' }
                    : { background: '#fffbeb', color: '#d97706' }}>
                    {it._server ? 'server' : 'run'}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {String(it.annotation || '').slice(0, 60)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Detail view */}
      {cur && (
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 5 }}>Selected item</div>
          <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6, background: '#f8fafc', borderRadius: 7, padding: '9px 10px', marginBottom: 10, border: '1px solid #e2e8f0' }}>
            {cur.question || cur.q || cur.text || JSON.stringify(cur).slice(0, 200)}
          </div>

          <div className="field-group">
            <label className="field-label">Annotation</label>
            <input className="field-input" value={cur.annotation || ''} onChange={(e) => handleChange('annotation', e.target.value)} placeholder="Enter annotation…" />
          </div>

          <div className="field-group">
            <label className="field-label">Confidence (0–1)</label>
            <input type="number" step="0.01" min="0" max="1" className="field-input" value={cur.confidence || 0} onChange={(e) => handleChange('confidence', Number(e.target.value) || 0)} style={{ width: 90 }} />
          </div>

          <div className="field-group">
            <label className="field-label">Notes</label>
            <textarea rows={3} className="field-input field-textarea" value={cur.notes || ''} onChange={(e) => handleChange('notes', e.target.value)} placeholder="Optional notes…" />
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            <Button variant="primary" onClick={() => submitAction('update')}>Save</Button>
            <Button variant="primary" onClick={() => submitAction('approve')}>Approve</Button>
            <Button variant="ghost" onClick={() => submitAction('reject')}>Reject</Button>
          </div>

          {selectedSet.size > 0 && (
            <div style={{ display: 'flex', gap: 6, paddingTop: 8, borderTop: '1px solid #f1f5f9', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: '#94a3b8' }}>{selectedSet.size} selected:</span>
              <Button variant="primary" onClick={() => submitBulk('approve')}>Bulk Approve</Button>
              <Button variant="ghost" onClick={() => submitBulk('reject')}>Bulk Reject</Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
