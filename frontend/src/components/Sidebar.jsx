import React from 'react';
import { v4 as uuidv4 } from 'uuid';

// Node type metadata
const NODE_TYPES = [
  {
    type: 'LoadData',
    label: 'Load Data',
    desc: 'Input dataset',
    color: '#3b82f6',
    bg: 'rgba(59,130,246,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <polyline points="7 10 12 15 17 10" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <line x1="12" y1="15" x2="12" y2="3" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    type: 'Task',
    label: 'Task',
    desc: 'Task definition',
    color: '#f97316',
    bg: 'rgba(249,115,22,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="18" height="18" rx="3" stroke="#f97316" strokeWidth="2"/>
        <path d="M9 12l2 2 4-4" stroke="#f97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    type: 'CandidateLLMs',
    label: 'Candidate LLMs',
    desc: 'LLM model pool',
    color: '#8b5cf6',
    bg: 'rgba(139,92,246,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <circle cx="6" cy="6" r="3" stroke="#8b5cf6" strokeWidth="2"/>
        <circle cx="18" cy="6" r="3" stroke="#8b5cf6" strokeWidth="2"/>
        <circle cx="12" cy="18" r="3" stroke="#8b5cf6" strokeWidth="2"/>
        <line x1="6" y1="9" x2="12" y2="15" stroke="#8b5cf6" strokeWidth="1.5"/>
        <line x1="18" y1="9" x2="12" y2="15" stroke="#8b5cf6" strokeWidth="1.5"/>
      </svg>
    ),
  },
  {
    type: 'Filter',
    label: 'Filter',
    desc: 'Data filtering',
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
      </svg>
    ),
  },
  {
    type: 'Router',
    label: 'Router',
    desc: 'Sample routing',
    color: '#ec4899',
    bg: 'rgba(236,72,153,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <path d="M16 3h5v5" stroke="#ec4899" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M4 20L21 3" stroke="#ec4899" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M21 16v5h-5" stroke="#ec4899" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M15 15l6 6" stroke="#ec4899" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M4 4l5 5" stroke="#ec4899" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    type: 'Annotate',
    label: 'Annotate',
    desc: 'LLM annotation',
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <path d="M12 20h9" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round"/>
        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    type: 'Output',
    label: 'Output',
    desc: 'Save results',
    color: '#14b8a6',
    bg: 'rgba(20,184,166,0.15)',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <polyline points="17 8 12 3 7 8" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <line x1="12" y1="3" x2="12" y2="15" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
  },
];

export default function Sidebar({ setNodes }) {
  const handleAdd = (nodeType) => {
    const id = uuidv4();
    setNodes((nds) => [
      ...nds,
      {
        id,
        type: 'compact',
        position: { x: 280 + Math.random() * 120, y: 80 + Math.random() * 200 },
        data: { label: nodeType, params: {} },
      },
    ]);
  };

  return (
    <div className="sidebar-panel">
      {/* Logo / header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <svg width="26" height="26" viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="9" fill="#6366f1"/>
            <path d="M8 16h4l3-6 4 12 3-6h4" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <div>
            <div style={{ lineHeight: 1.2 }}>DataFlow</div>
            <div style={{ fontSize: 10, fontWeight: 400, color: '#64748b', letterSpacing: '0.04em' }}>Annotator</div>
          </div>
        </div>
      </div>

      {/* Node palette */}
      <div className="sidebar-section-title">Node Types</div>
      <div className="sidebar-scroll">
        {NODE_TYPES.map((n) => (
          <div
            key={n.type}
            className="node-palette-item"
            onClick={() => handleAdd(n.type)}
            title={`Add ${n.label} node`}
          >
            <div className="node-palette-icon" style={{ background: n.bg }}>
              {n.icon}
            </div>
            <div>
              <div className="node-palette-label">{n.label}</div>
              <div className="node-palette-desc">{n.desc}</div>
            </div>
          </div>
        ))}
      </div>

      {/* How to use */}
      <div className="sidebar-howto">
        <div style={{ fontWeight: 700, color: '#94a3b8', fontSize: 10, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>How to use</div>
        <div>• Click a node type to add it</div>
        <div>• Drag to reposition nodes</div>
        <div>• Drag between handles to connect</div>
        <div>• Click a node to edit params</div>
        <div>• Press Run to execute pipeline</div>
      </div>
    </div>
  );
}

export { NODE_TYPES };
