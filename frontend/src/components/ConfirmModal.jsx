import React from 'react';

export default function ConfirmModal({ open, title = 'Confirm', message = '', onConfirm, onCancel }) {
  if (!open) return null;
  const overlay = {
    position: 'fixed',
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(15,23,42,0.45)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999,
  };
  const box = {
    background: 'white',
    padding: 18,
    borderRadius: 8,
    width: 360,
    boxShadow: '0 12px 40px rgba(2,6,23,0.4)',
  };
  const titleStyle = { margin: 0, fontSize: 16, fontWeight: 700 };
  const msgStyle = { marginTop: 8, color: '#334155', fontSize: 13 };
  const actions = { marginTop: 14, display: 'flex', justifyContent: 'flex-end', gap: 8 };

  return (
    <div style={overlay} role="dialog" aria-modal="true">
      <div style={box}>
        <h4 style={titleStyle}>{title}</h4>
        <div style={msgStyle}>{message}</div>
        <div style={actions}>
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
          <button className="btn btn-danger" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  );
}
