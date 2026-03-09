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

  if (!list.length) {
    return (
      <div style={{ padding: '32px 16px', textAlign: 'center' }}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 12px', display: 'block', opacity: 0.25 }}>
          <path d="M9 12l2 2 4-4m6 2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" stroke="#374151" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <div style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>No items in review queue</div>
        <div style={{ fontSize: 11.5, color: '#cbd5e1', marginTop: 4 }}>Items needing human review will appear here</div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* List */}
      <div style={{ padding: '10px 16px 6px', borderBottom: '1px solid rgba(15,23,42,0.07)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <span style={{ fontSize: 11.5, fontWeight: 700, color: '#374151' }}>{list.length} items</span>
          <div style={{ display: 'flex', gap: 6 }}>
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
        <div style={{ maxHeight: 200, overflowY: 'auto' }}>
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
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {String(it.annotation || '').slice(0, 60)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detail view */}
      {cur && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 6 }}>Selected item</div>
          <div style={{ fontSize: 12.5, color: '#475569', lineHeight: 1.6, background: '#f8fafc', borderRadius: 8, padding: '10px 12px', marginBottom: 12, border: '1px solid #e2e8f0' }}>
            {cur.question || cur.q || cur.text || JSON.stringify(cur).slice(0, 300)}
          </div>

          <div className="field-group">
            <label className="field-label">Annotation</label>
            <input
              className="field-input"
              value={cur.annotation || ''}
              onChange={(e) => handleChange('annotation', e.target.value)}
              placeholder="Enter annotation…"
            />
          </div>

          <div className="field-group">
            <label className="field-label">Confidence (0–1)</label>
            <input
              type="number" step="0.01" min="0" max="1"
              className="field-input"
              value={cur.confidence || 0}
              onChange={(e) => handleChange('confidence', Number(e.target.value) || 0)}
              style={{ width: 100 }}
            />
          </div>

          <div className="field-group">
            <label className="field-label">Notes</label>
            <textarea
              rows={3}
              className="field-input field-textarea"
              value={cur.notes || ''}
              onChange={(e) => handleChange('notes', e.target.value)}
              placeholder="Optional notes…"
            />
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, marginTop: 8 }}>
            <Button variant="primary" onClick={() => submitAction('update')}>Save</Button>
            <Button variant="primary" onClick={() => submitAction('approve')}>Approve</Button>
            <Button variant="ghost" onClick={() => submitAction('reject')}>Reject</Button>
          </div>

          {selectedSet.size > 0 && (
            <div style={{ display: 'flex', gap: 7, marginTop: 10, paddingTop: 10, borderTop: '1px solid #f1f5f9' }}>
              <span style={{ fontSize: 11.5, color: '#94a3b8', alignSelf: 'center' }}>{selectedSet.size} selected:</span>
              <Button variant="primary" onClick={() => submitBulk('approve')}>Bulk Approve</Button>
              <Button variant="ghost" onClick={() => submitBulk('reject')}>Bulk Reject</Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
