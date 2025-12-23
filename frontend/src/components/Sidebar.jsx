import React from 'react';
import { v4 as uuidv4 } from 'uuid';

const Svg = ({ children, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    {children}
  </svg>
);

const nodeTypes = [
  { type: 'LoadData', label: 'Load Data', icon: (<Svg><path d="M12 3v10" stroke="#111827" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M8 7l4-4 4 4" stroke="#111827" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></Svg>) },
  { type: 'Task', label: 'Task', icon: (<Svg><rect x="3" y="3" width="18" height="18" rx="2" stroke="#0f172a" strokeWidth="1.2"/></Svg>) },
  { type: 'CandidateLLMs', label: 'Candidate LLMs', icon: (<Svg><circle cx="7" cy="7" r="3" stroke="#0f172a" strokeWidth="1.2"/><circle cx="17" cy="17" r="3" stroke="#0f172a" strokeWidth="1.2"/></Svg>) },
  { type: 'Filter', label: 'Filter', icon: (<Svg><path d="M4 6h16" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/><path d="M10 12h4" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/><path d="M6 18h12" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/></Svg>) },
  { type: 'Router', label: 'Router', icon: (<Svg><path d="M12 3v18" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/><path d="M4 10h8" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/><path d="M12 14h8" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/></Svg>) },
  { type: 'Annotate', label: 'Annotate', icon: (<Svg><path d="M3 21l3-3 11-11 3 3L7 21H3z" stroke="#0f172a" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/></Svg>) },
  { type: 'Output', label: 'Output', icon: (<Svg><path d="M4 12h16" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round"/><path d="M12 5l7 7-7 7" stroke="#0f172a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></Svg>) },
];

export default function Sidebar({ setNodes, nodes }) {
  const handleAdd = (nodeType) => {
    const id = uuidv4();
    const base = {
      id,
      type: 'compact',
      position: { x: 250 + Math.random() * 100, y: 50 + Math.random() * 200 },
      data: { label: `${nodeType}`, params: {} },
    };
    setNodes((nds) => [...nds, base]);
  };

  return (
    <aside className="sidebar">
      <h3>Palette</h3>
      {nodeTypes.map((n) => (
        <div key={n.type} className="node-item" onClick={() => handleAdd(n.type)}>
          <div style={{ fontSize: 18 }}>{n.icon}</div>
          <div style={{ flex: 1 }}>{n.label}</div>
        </div>
      ))}
      <hr />
      <div className="howto">
        <strong style={{ display: 'block', marginBottom: 6 }}>How to use</strong>
        <div>- Click a node to add to canvas</div>
        <div>- Select a node to edit params</div>
        <div>- Connect edges to form flow</div>
        <div>- Click Run Graph to execute</div>
      </div>
    </aside>
  );
}
