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
    <div className="mt-3">
      <div className="flex items-center gap-3">
        <h4 className="text-sm font-semibold">Remaining</h4>
        <button className="btn btn-ghost text-sm" onClick={onRefresh} disabled={!onRefresh}>Refresh</button>
      </div>
      <div className="text-sm text-slate-500 mt-2">No items requiring human review.</div>
    </div>
  );

  return (
    <div className="mt-3">
      <div className="flex items-start gap-4">
        <div className="min-w-[180px] max-w-[260px] max-h-[420px] overflow-auto border-r border-slate-100 pr-2">
          <div className="flex gap-2 mb-2 items-center">
            <button className="btn btn-ghost text-sm" onClick={selectAll}>Select All</button>
            <button className="btn btn-ghost text-sm" onClick={clearSelection}>Clear</button>
            <div className="ml-auto text-sm text-slate-500">{list.length} items</div>
          </div>
          <div className="space-y-2">
            {list.map((it, i) => (
              <div key={i} className={`p-2 ${i === sel ? 'bg-slate-100' : ''} flex gap-2 items-start rounded`}>
                <input type="checkbox" checked={selectedSet.has(i)} onChange={() => toggleSelect(i)} />
                <div className="flex-1 cursor-pointer" onClick={() => setSel(i)}>
                  <div className="flex items-center gap-2">
                    <div className="text-sm font-semibold">{it.id || it.qid || `item ${i}`}</div>
                    {it._server ? (
                      <div className="text-xs bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full">server</div>
                    ) : (
                      <div className="text-xs bg-amber-50 text-amber-700 px-2 py-0.5 rounded-full">run-only</div>
                    )}
                  </div>
                  <div className="text-sm text-slate-500">{(it.annotation || '').toString().slice(0, 80)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex-1">
          <div className="mb-2">
            <strong className="text-sm">Selected</strong>
            <div className="text-sm text-slate-700 mt-2">{cur && (cur.question || cur.q || cur.text || JSON.stringify(cur).slice(0,200))}</div>
          </div>

          <div className="mt-3">
            <label className="block text-sm mb-1">Annotation</label>
            <input className="w-full border border-slate-100 rounded px-2 py-1" value={cur?.annotation || ''} onChange={(e) => handleChange('annotation', e.target.value)} />
          </div>

          <div className="mt-3">
            <label className="block text-sm mb-1">Confidence</label>
            <input type="number" step="0.01" min="0" max="1" value={cur?.confidence || 0} onChange={(e) => handleChange('confidence', Number(e.target.value) || 0)} className="w-28 border border-slate-100 rounded px-2 py-1" />
          </div>

          <div className="mt-3">
            <label className="block text-sm mb-1">Notes (optional)</label>
            <textarea rows={4} className="w-full border border-slate-100 rounded px-2 py-1" value={cur?.notes || ''} onChange={(e) => handleChange('notes', e.target.value)} />
          </div>

          <div className="mt-4 flex items-center gap-2">
            <button className="btn btn-primary" onClick={() => submitAction('update')}>Save</button>
            <button className="btn btn-primary" onClick={() => submitAction('approve')}>Approve</button>
            <button className="btn btn-ghost" onClick={() => submitAction('reject')}>Reject</button>
            <div className="ml-auto flex gap-2">
              <button className="btn btn-primary" onClick={() => submitBulk('approve')} disabled={selectedSet.size === 0}>Bulk Approve</button>
              <button className="btn btn-ghost" onClick={() => submitBulk('reject')} disabled={selectedSet.size === 0}>Bulk Reject</button>
            </div>
            <div className="ml-3 text-sm text-slate-500">Server: {serverCount} · Run-only: {runOnlyCount}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
