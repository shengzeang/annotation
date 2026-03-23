import React, { useCallback, useState } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges,
} from 'reactflow';
import 'reactflow/dist/style.css';
import Sidebar from './components/Sidebar';
import CompactNode from './components/CompactNode';
import NodeEditor from './components/NodeEditor';
import ConfirmModal from './components/ConfirmModal';
import ReviewQueue from './components/ReviewQueue';
import CompletedSamples from './components/CompletedSamples';
import axios from 'axios';

// ─── Initial graph (matches target layout) ───────────────────
const initialNodes = [
  // Left column: input nodes
  { id: 'n1', type: 'compact', position: { x: 60,  y: 60  }, data: { label: 'LoadData',      params: { dataset: 'squad', max_samples: 200 } } },
  { id: 'n2', type: 'compact', position: { x: 60,  y: 230 }, data: { label: 'Task',           params: { task_class: 'tasks.qa.QATask' } } },
  { id: 'n3', type: 'compact', position: { x: 60,  y: 400 }, data: { label: 'CandidateLLMs', params: { candidate_llms: ['Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'] } } },
  // Pipeline row
  { id: 'n4', type: 'compact', position: { x: 320, y: 230 }, data: { label: 'Filter',    params: { filter_class: 'filters.al_filter.ActiveLearningFilter', filter_params: { method: 'alps', budget: 100, batch_size: 20 } } } },
  { id: 'n5', type: 'compact', position: { x: 530, y: 230 }, data: { label: 'Router',    params: { router_class: 'routers.knn_router.KNNRouter', router_params: { k: 5 }, candidate_llms: ['Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'] } } },
  { id: 'n6', type: 'compact', position: { x: 740, y: 230 }, data: { label: 'Annotate',  params: { candidate_llms: ['Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'], llm_mode: 'local', task_class: 'tasks.qa.QATask' } } },
  { id: 'n7', type: 'compact', position: { x: 950, y: 230 }, data: { label: 'Output',    params: { path: 'out/annotations.json' } } },
];

const initialEdges = [
  { id: 'e1-4', source: 'n1', target: 'n4', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
  { id: 'e4-5', source: 'n4', target: 'n5', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
  { id: 'e5-6', source: 'n5', target: 'n6', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
  { id: 'e6-7', source: 'n6', target: 'n7', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
  { id: 'e3-5', source: 'n3', target: 'n5', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
  { id: 'e3-6', source: 'n3', target: 'n6', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
  { id: 'e2-6', source: 'n2', target: 'n6', style: { stroke: '#94a3b8', strokeWidth: 1.8 } },
];

const nodeTypes = { compact: CompactNode };

// ─── App ──────────────────────────────────────────────────────
export default function App() {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [prevNodes, setPrevNodes] = useState(null);
  const [undoStack, setUndoStack] = useState([]);
  const [reviewItems, setReviewItems] = useState([]);
  const [confirm, setConfirm] = useState({ open: false, targetId: null, message: '' });
  const [running, setRunning] = useState(false);
  const [rightTab, setRightTab] = useState('properties'); // 'properties' | 'review' | 'completed'

  // Load persisted review queue
  React.useEffect(() => {
    let mounted = true;
    const fetchServerQueue = async () => {
      try {
        const res = await axios.get('http://localhost:5000/review_submissions');
        if (!mounted) return;
        if (res.data?.status === 'ok') {
          const serverItems = (res.data.items || []).map((it) => ({ ...it, _server: true }));
          setReviewItems((cur) => {
            const same = (a, b) => {
              if (!a || !b) return false;
              if (a.id && b.id && a.id === b.id) return true;
              if (a.qid && b.qid && a.qid === b.qid) return true;
              if (a.question && b.question && a.question === b.question) return true;
              if (a.text && b.text && a.text === b.text) return true;
              return false;
            };
            const runOnly = (cur || []).filter((x) => !x._server);
            const merged = [...serverItems];
            runOnly.forEach((r) => { if (!merged.find((s) => same(s, r))) merged.push(r); });
            return merged;
          });
        }
      } catch (e) { /* server may not be running */ }
    };
    fetchServerQueue();
    const iv = setInterval(fetchServerQueue, 10000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);

  // Snapshot management
  const pushSnapshot = React.useCallback(() => {
    setUndoStack((s) => {
      const next = [...s, { nodes: nodes.map((n) => ({ ...n })), edges: edges.map((e) => ({ ...e })) }];
      return next.length > 10 ? next.slice(next.length - 10) : next;
    });
  }, [nodes, edges]);

  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);

  const onConnect = useCallback((params) => {
    pushSnapshot();
    setEdges((eds) => addEdge({ ...params, style: { stroke: '#94a3b8', strokeWidth: 1.8 } }, eds));
  }, [pushSnapshot]);

  const onNodesDelete = useCallback((nodesToDelete) => {
    pushSnapshot();
    const removeIds = new Set(nodesToDelete.map((n) => n.id));
    setNodes((nds) => nds.filter((n) => !removeIds.has(n.id)));
    setEdges((eds) => eds.filter((e) => !removeIds.has(e.source) && !removeIds.has(e.target)));
  }, [pushSnapshot]);

  const onEdgesDelete = useCallback((edgesToDelete) => {
    pushSnapshot();
    const removeIds = new Set(edgesToDelete.map((e) => e.id));
    setEdges((eds) => eds.filter((e) => !removeIds.has(e.id)));
  }, [pushSnapshot]);

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
    setRightTab('properties');
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const deleteNodeById = useCallback((id) => {
    if (!id) return;
    pushSnapshot();
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id && e.id !== id));
    setSelectedNode((cur) => (cur?.id === id ? null : cur));
  }, [pushSnapshot]);

  const deleteSelected = useCallback(() => {
    if (!selectedNode) return;
    setConfirm({ open: true, targetId: selectedNode.id, message: `Delete node "${selectedNode.data?.label || selectedNode.id}"?` });
  }, [selectedNode]);

  const handleUndo = useCallback(() => {
    if (undoStack?.length > 0) {
      const last = undoStack[undoStack.length - 1];
      setNodes(last.nodes.map((n) => ({ ...n })));
      setEdges(last.edges.map((e) => ({ ...e })));
      setUndoStack((s) => s.slice(0, -1));
      setSelectedNode(null);
    } else if (prevNodes) {
      setNodes(prevNodes.map((n) => ({ ...n })));
      setPrevNodes(null);
    }
  }, [undoStack, prevNodes]);

  const compactLayout = useCallback(() => {
    setPrevNodes(nodes.map((n) => ({ ...n })));
    const nodeMap = new Map(nodes.map((n) => [n.id, { ...n, layer: 0 }]));
    const adj = new Map(); const inDegree = new Map();
    nodes.forEach((n) => { adj.set(n.id, []); inDegree.set(n.id, 0); });
    edges.forEach((e) => {
      if (!adj.has(e.source)) adj.set(e.source, []);
      adj.get(e.source).push(e.target);
      inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
    });
    const q = []; inDegree.forEach((deg, id) => { if (deg === 0) q.push(id); });
    let processed = 0;
    while (q.length) {
      const id = q.shift(); processed++;
      const node = nodeMap.get(id);
      (adj.get(id) || []).forEach((nb) => {
        const nbNode = nodeMap.get(nb);
        if (nbNode) nbNode.layer = Math.max(nbNode.layer, node.layer + 1);
        inDegree.set(nb, inDegree.get(nb) - 1);
        if (inDegree.get(nb) === 0) q.push(nb);
      });
    }
    if (processed < nodes.length) {
      const order = { LoadData: 0, Task: 1, CandidateLLMs: 2, Filter: 3, Router: 4, Annotate: 5, Output: 6 };
      nodes.forEach((n) => { nodeMap.get(n.id).layer = order[n.data?.label] ?? 10; });
    }
    const layers = [];
    nodeMap.forEach((n) => { const l = n.layer || 0; if (!layers[l]) layers[l] = []; layers[l].push(n); });
    const newNodes = nodes.map((n) => {
      const nObj = nodeMap.get(n.id);
      const layer = nObj.layer || 0;
      const column = layers[layer] || [];
      const index = column.findIndex((c) => c.id === n.id);
      return { ...n, position: { x: 60 + layer * 210, y: 60 + index * 170 } };
    });
    setNodes(newNodes);
  }, [nodes, edges]);

  const updateNodeData = (id, data) => {
    pushSnapshot();
    setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...data } } : n)));
    const n = nodes.find((x) => x.id === id);
    setSelectedNode(n ? { ...n, data: { ...n.data, ...data } } : null);
  };

  const handleRun = async () => {
    setRunning(true);
    try {
      const res = await axios.post('http://localhost:5000/run_graph', { nodes, edges });
      console.log('Run result:', res.data);
      const ctx = res.data?.context || res.data?.outputs || {};
      const found = [];
      Object.values(ctx).forEach((val) => {
        if (Array.isArray(val)) val.forEach((it) => { if (it?.needs_human) found.push({ ...it, _server: false }); });
      });
      if (found.length) {
        setReviewItems((cur) => {
          const same = (a, b) => {
            if (!a || !b) return false;
            if (a.id && b.id && a.id === b.id) return true;
            if (a.qid && b.qid && a.qid === b.qid) return true;
            if (a.question && b.question && a.question === b.question) return true;
            return a.text && b.text && a.text === b.text;
          };
          const curServer = (cur || []).filter((x) => x?._server);
          const runOnly = found.filter((f) => !curServer.find((s) => same(s, f)));
          return [...curServer, ...runOnly];
        });
      }
      alert('Pipeline run completed.');
    } catch (err) {
      console.error(err);
      alert('Run failed: ' + err.message);
    } finally {
      setRunning(false);
    }
  };

  const canUndo = !!(prevNodes || undoStack?.length > 0);

  return (
    <div style={{ height: '100vh', display: 'flex', overflow: 'hidden', background: '#f0f2f5' }}>
      <ReactFlowProvider>

        {/* ── Left sidebar (palette) ── */}
        <Sidebar setNodes={setNodes} nodes={nodes} />

        {/* ── Canvas ── */}
        <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodesDelete={onNodesDelete}
            onEdgesDelete={onEdgesDelete}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            fitView
            fitViewOptions={{ padding: 0.18 }}
            deleteKeyCode={null}
          >
            {/* Dotted grid background */}
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="#c7cfe0"
            />
            <Controls style={{ bottom: 16, left: 16, top: 'auto' }} />
            <MiniMap
              nodeColor={(node) => {
                const colors = {
                  LoadData: '#3b82f6', Task: '#f97316', CandidateLLMs: '#8b5cf6',
                  Filter: '#22c55e', Router: '#ec4899', Annotate: '#f59e0b', Output: '#14b8a6',
                };
                return colors[node.data?.label] || '#94a3b8';
              }}
              style={{ bottom: 16, right: 16, background: '#fff', borderRadius: 10, border: '1px solid rgba(15,23,42,0.08)', boxShadow: '0 2px 8px rgba(2,6,23,0.06)' }}
            />

            {/* ── Top toolbar ── */}
            <Panel position="top-left">
              <div className="toolbar">
                {/* Run Graph */}
                <button
                  className="btn btn-run"
                  onClick={handleRun}
                  disabled={running}
                  title="Run the pipeline"
                >
                  {running ? (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" className="spin-icon">
                        <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.35)" strokeWidth="2.5"/>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke="#fff" strokeWidth="2.5" strokeLinecap="round"/>
                      </svg>
                      Running…
                    </>
                  ) : (
                    <>
                      <svg width="11" height="13" viewBox="0 0 12 14" fill="white">
                        <path d="M1 1l10 6-10 6V1z"/>
                      </svg>
                      Run Graph
                    </>
                  )}
                </button>

                {/* Divider */}
                <span className="toolbar-sep">|</span>

                {/* Compact */}
                <button className="btn btn-toolbar" onClick={compactLayout} title="Auto-arrange nodes">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <rect x="3" y="3" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.8"/>
                    <rect x="14" y="3" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.8"/>
                    <rect x="3" y="14" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.8"/>
                    <rect x="14" y="14" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.8"/>
                  </svg>
                  Compact
                </button>

                {/* Undo */}
                <button
                  className="btn btn-toolbar"
                  onClick={handleUndo}
                  disabled={!canUndo}
                  title="Undo last action"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <path d="M9 14H4V9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M4 9a9 9 0 1 1 0 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                  </svg>
                  Undo
                </button>

                {/* Delete */}
                <button
                  className="btn btn-delete"
                  onClick={deleteSelected}
                  disabled={!selectedNode}
                  title="Delete selected node"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Delete
                </button>
              </div>
            </Panel>
          </ReactFlow>
        </div>

        {/* ── Right panel (tabs) ── */}
        <div className="right-panel">

          {/* Tab bar */}
          <div className="right-tabs">
            <button
              className={'right-tab' + (rightTab === 'properties' ? ' active' : '')}
              onClick={() => setRightTab('properties')}
            >
              Properties
            </button>
            <button
              className={'right-tab' + (rightTab === 'review' ? ' active' : '')}
              onClick={() => setRightTab('review')}
            >
              Review Queue
              {reviewItems.length > 0 && (
                <span className="right-tab-badge">{reviewItems.length}</span>
              )}
            </button>
            <button
              className={'right-tab' + (rightTab === 'completed' ? ' active' : '')}
              onClick={() => setRightTab('completed')}
            >
              Completed
            </button>
          </div>

          {/* Tab content */}
          <div className="right-panel-scroll">

            {rightTab === 'properties' && (
              <div className="rp-section">
                <div className="rp-section-header">
                  <span className="rp-section-title">Node Editor</span>
                </div>
                {selectedNode ? (
                  <NodeEditor
                    node={selectedNode}
                    updateNodeData={updateNodeData}
                    deleteNode={deleteNodeById}
                    openConfirm={(id, message) => setConfirm({ open: true, targetId: id, message })}
                  />
                ) : (
                  <div className="rp-empty-state">
                    <div className="rp-empty-text">Select a node to edit its params</div>
                  </div>
                )}
              </div>
            )}

            {rightTab === 'review' && (
              <div className="rp-section">
                <ReviewQueue items={reviewItems} onUpdate={(items) => setReviewItems(items)} />
              </div>
            )}

            {rightTab === 'completed' && (
              <div className="rp-section">
                <CompletedSamples />
              </div>
            )}

          </div>
        </div>

        {/* Confirm delete modal */}
        <ConfirmModal
          open={confirm.open}
          title="Delete node"
          message={confirm.message}
          onConfirm={() => {
            if (confirm.targetId) deleteNodeById(confirm.targetId);
            setConfirm({ open: false, targetId: null, message: '' });
          }}
          onCancel={() => setConfirm({ open: false, targetId: null, message: '' })}
        />
      </ReactFlowProvider>
    </div>
  );
}
