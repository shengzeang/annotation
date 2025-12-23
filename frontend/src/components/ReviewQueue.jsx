import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function ReviewQueue({ items, onUpdate, onRefresh }) {
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

  const selectAll = () => {
    setSelectedSet(new Set(list.map((_, i) => i)));
  };

  const clearSelection = () => setSelectedSet(new Set());

  const submitSingle = async (sample, action) => {
    return axios.post('http://localhost:5000/submit_review', { sample, action });
  };

  const submitAction = async (action) => {
    if (!cur) return;
    try {
      await submitSingle(cur, action);
      // remove from local list
      const next = list.filter((_, i) => i !== sel);
      setList(next);
      setSel((s) => Math.min(s, Math.max(0, next.length - 1)));
      if (onUpdate) onUpdate(next);
      alert('Submission saved.');
    } catch (e) {
      console.error(e);
      alert('Failed to submit: ' + e.message);
    }
  };

  const submitBulk = async (action) => {
    const indices = Array.from(selectedSet).sort((a, b) => b - a); // descending so removals by index won't shift earlier ones
    if (!indices.length) return alert('No items selected');
    try {
      const promises = indices.map((i) => submitSingle(list[i], action).catch((e) => ({ error: e })));
      const results = await Promise.all(promises);
      // build next list removing successful submissions
      const failed = [];
      const successIndices = [];
      results.forEach((r, idx) => {
        if (r && r.data && r.data.status === 'ok') successIndices.push(indices[idx]); else failed.push(indices[idx]);
      });
      const next = list.filter((_, i) => !successIndices.includes(i));
      setList(next);
      setSelectedSet(new Set());
      setSel((s) => Math.min(s, Math.max(0, next.length - 1)));
      if (onUpdate) onUpdate(next);
      if (failed.length === 0) alert('Bulk submission saved.'); else alert('Some submissions failed.');
    } catch (e) {
      console.error(e);
      alert('Bulk submission failed: ' + e.message);
    }
  };

  const serverCount = list.filter((i) => i && i._server).length;
  const runOnlyCount = list.length - serverCount;

  if (!list || list.length === 0) return (
    <div style={{ marginTop: 12 }}>
      <h4 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>Review Queue <button className="btn btn-secondary" onClick={onRefresh}>Refresh</button></h4>
      <div style={{ color: '#666' }}>No items requiring human review.</div>
    </div>
  );

  return (
    <div style={{ marginTop: 12 }}>
      <h4>Review Queue ({list.length})</h4>
      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ width: 160, maxHeight: 420, overflow: 'auto', borderRight: '1px solid #eee' }}>
          {list.map((it, i) => (
            <div key={i} style={{ padding: 8, background: i === sel ? '#f3f4f6' : 'transparent', display: 'flex', gap: 8, alignItems: 'center' }}>
              <input type="checkbox" checked={selectedSet.has(i)} onChange={() => toggleSelect(i)} />
              <div style={{ flex: 1, cursor: 'pointer' }} onClick={() => setSel(i)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{it.id || it.qid || `item ${i}`}</div>
                  {it._server ? (
                    <div style={{ fontSize: 11, background: '#e6fffb', color: '#065f46', padding: '2px 6px', borderRadius: 10 }}>server</div>
                  ) : (
                    <div style={{ fontSize: 11, background: '#fff7ed', color: '#7c2d12', padding: '2px 6px', borderRadius: 10 }}>run-only</div>
                  )}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>{(it.annotation || '').toString().slice(0, 80)}</div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ flex: 1 }}>
          <div style={{ marginBottom: 8 }}>
            <strong>Selected</strong>
            <div style={{ fontSize: 13, color: '#333', marginTop: 6 }}>{cur && (cur.question || cur.q || cur.text || JSON.stringify(cur).slice(0,200))}</div>
          </div>

          <div style={{ marginTop: 8 }}>
            <label>Annotation</label>
            <input value={cur?.annotation || ''} onChange={(e) => handleChange('annotation', e.target.value)} style={{ width: '100%' }} />
          </div>

          <div style={{ marginTop: 8 }}>
            <label>Confidence</label>
            <input type="number" step="0.01" min="0" max="1" value={cur?.confidence || 0} onChange={(e) => handleChange('confidence', Number(e.target.value) || 0)} style={{ width: 120 }} />
          </div>

          <div style={{ marginTop: 8 }}>
            <label>Notes (optional)</label>
            <textarea value={cur?.notes || ''} onChange={(e) => handleChange('notes', e.target.value)} rows={4} style={{ width: '100%' }} />
          </div>

          <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
            <button className="btn btn-primary" onClick={() => submitAction('update')}>Save</button>
            <button className="btn btn-primary" onClick={() => submitAction('approve')}>Approve</button>
            <button className="btn btn-secondary" onClick={() => submitAction('reject')}>Reject</button>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
              <button className="btn btn-primary" onClick={() => submitBulk('approve')} disabled={selectedSet.size === 0}>Bulk Approve</button>
              <button className="btn btn-secondary" onClick={() => submitBulk('reject')} disabled={selectedSet.size === 0}>Bulk Reject</button>
            </div>
            <div style={{ marginLeft: '12px' }}>
              <span style={{ fontSize: 12, color: '#666' }}>Server: {serverCount} · Run-only: {runOnlyCount}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
