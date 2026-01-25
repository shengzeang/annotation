import React from 'react';
import Button from './ui/Button';
import Card from './ui/Card';

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

  return (
    <div style={overlay} role="dialog" aria-modal="true">
      <Card style={{ width: 360 }}>
        <h4 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>{title}</h4>
        <div style={{ marginTop: 8, color: '#334155', fontSize: 13 }}>{message}</div>
        <div style={{ marginTop: 14, display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <Button variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button variant="ghost" danger onClick={onConfirm}>Delete</Button>
        </div>
      </Card>
    </div>
  );
}
