import React from 'react';
import Button from './ui/Button';

export default function ConfirmModal({ open, title = 'Confirm', message = '', onConfirm, onCancel }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-box">
        <h4 className="modal-title">{title}</h4>
        <p className="modal-message">{message}</p>
        <div className="modal-actions">
          <Button variant="secondary" onClick={onCancel}>Cancel</Button>
          <Button danger onClick={onConfirm}>Delete</Button>
        </div>
      </div>
    </div>
  );
}
