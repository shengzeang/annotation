import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function CompletedSamples() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(null);

  const fetchItems = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:5000/completed_samples');
      if (res.data && Array.isArray(res.data.items)) {
        setItems(res.data.items);
      }
    } catch (e) {
      // Server may not be running or endpoint may not exist — fail silently
      if (e?.response?.status !== 404) {
        console.debug('CompletedSamples: could not fetch from server', e?.message);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchItems();
  }, []);

  if (!items.length && !loading) {
    return (
      <div style={{ padding: '32px 16px', textAlign: 'center' }}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 12px', display: 'block', opacity: 0.25 }}>
          <path d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" stroke="#374151" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <div style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>No completed samples yet</div>
        <div style={{ fontSize: 11.5, color: '#cbd5e1', marginTop: 4 }}>Completed annotations will appear here</div>
        <button
          onClick={fetchItems}
          style={{ marginTop: 12, fontSize: 12, color: '#6366f1', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 16px 8px', borderBottom: '1px solid rgba(15,23,42,0.07)' }}>
        <span style={{ fontSize: 11.5, fontWeight: 700, color: '#374151' }}>{items.length} completed</span>
        <button
          onClick={fetchItems}
          style={{ fontSize: 11, color: '#6366f1', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      <div style={{ padding: '10px 16px', maxHeight: 400, overflowY: 'auto' }}>
        {items.map((it, i) => (
          <div
            key={i}
            className="result-card"
            style={{ cursor: 'pointer' }}
            onClick={() => setExpanded(expanded === i ? null : i)}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div className="result-card-title">{it.id || it.qid || `Sample ${i + 1}`}</div>
              <span style={{
                fontSize: 10.5, fontWeight: 700, padding: '2px 8px', borderRadius: 999,
                background: it.action === 'approve' ? '#f0fdf4' : it.action === 'reject' ? '#fef2f2' : '#f1f5f9',
                color: it.action === 'approve' ? '#16a34a' : it.action === 'reject' ? '#ef4444' : '#64748b',
              }}>
                {it.action || 'saved'}
              </span>
            </div>
            {it.annotation && (
              <div className="result-card-meta" style={{ marginTop: 4 }}>
                {String(it.annotation).slice(0, 80)}{String(it.annotation).length > 80 ? '…' : ''}
              </div>
            )}
            {expanded === i && (
              <div style={{ marginTop: 10, fontSize: 12, color: '#374151', background: '#f8fafc', borderRadius: 7, padding: '8px 10px', border: '1px solid #e2e8f0', lineHeight: 1.6 }}>
                <pre style={{ margin: 0, fontFamily: 'inherit', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {JSON.stringify(it, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
