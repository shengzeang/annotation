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

  if (!items.length) {
    return (
      <div className="rp-queue-empty">
        {/* Sparkle / star icon */}
        <svg className="rp-empty-icon" width="36" height="36" viewBox="0 0 24 24" fill="none">
          <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6L12 2z" stroke="#9ca3af" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <div className="rp-empty-label">No completed samples yet</div>
        <div className="rp-empty-sublabel">Completed annotations will appear here</div>
        <button
          onClick={fetchItems}
          style={{ marginTop: 10, fontSize: 12, color: '#6366f1', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ fontSize: 11.5, fontWeight: 700, color: '#374151' }}>{items.length} completed</span>
        <button
          onClick={fetchItems}
          style={{ fontSize: 11, color: '#6366f1', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      <div>
        {items.map((it, i) => (
          <div
            key={i}
            className="result-card"
            onClick={() => setExpanded(expanded === i ? null : i)}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div className="result-card-title">{it.id || it.qid || `Sample ${i + 1}`}</div>
              <span style={{
                fontSize: 10.5, fontWeight: 700, padding: '2px 7px', borderRadius: 999,
                background: it.action === 'approve' ? '#f0fdf4' : it.action === 'reject' ? '#fef2f2' : '#f1f5f9',
                color: it.action === 'approve' ? '#16a34a' : it.action === 'reject' ? '#ef4444' : '#64748b',
              }}>
                {it.action || 'saved'}
              </span>
            </div>
            {it.annotation && (
              <div className="result-card-meta" style={{ marginTop: 3 }}>
                {String(it.annotation).slice(0, 80)}{String(it.annotation).length > 80 ? '…' : ''}
              </div>
            )}
            {expanded === i && (
              <div style={{ marginTop: 9, fontSize: 11.5, color: '#374151', background: '#f8fafc', borderRadius: 6, padding: '8px 10px', border: '1px solid #e2e8f0', lineHeight: 1.6 }}>
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
