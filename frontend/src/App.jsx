import React, { useCallback, useState } from 'react';
import ReactFlow, {
  addEdge,
  Background,
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

// ─── Initial graph ────────────────────────────────────────────
const initialNodes = [
  { id: 'n1', type: 'compact', position: { x: 40, y: 60 },  data: { label: 'LoadData',      params: { dataset: 'squad', max_samples: 200 } } },
  { id: 'n2', type: 'compact', position: { x: 40, y: 200 }, data: { label: 'Task',           params: { task_class: 'tasks.qa.QATask' } } },
  { id: 'n3', type: 'compact', position: { x: 40, y: 340 }, data: { label: 'CandidateLLMs', params: { candidate_llms: ['gpt2', 'distilgpt2'] } } },
  { id: 'n4', type: 'compact', position: { x: 280, y: 200 }, data: { label: 'Filter',    params: { filter_class: 'filters.al_filter.ActiveLearningFilter', filter_params: { method: 'alps', budget: 100, batch_size: 20 } } } },
  { id: 'n5', type: 'compact', position: { x: 500, y: 200 }, data: { label: 'Router',    params: { router_class: 'routers.knn_router.KNNRouter', router_params: { k: 5 }, candidate_llms: ['gpt2', 'distilgpt2'] } } },
  { id: 'n6', type: 'compact', position: { x: 720, y: 200 }, data: { label: 'Annotate',  params: { candidate_llms: ['gpt2', 'distilgpt2'], llm_mode: 'local', task_class: 'tasks.qa.QATask' } } },
  { id: 'n7', type: 'compact', position: { x: 940, y: 200 }, data: { label: 'Output',    params: { path: 'out/annotations.json' } } },
];

const initialEdges = [
  { id: 'e1-4', source: 'n1', target: 'n4', style: { strokeWidth: 2 } },
  { id: 'e4-5', source: 'n4', target: 'n5', style: { strokeWidth: 2 } },
  { id: 'e5-6', source: 'n5', target: 'n6', style: { strokeWidth: 2 } },
  { id: 'e6-7', source: 'n6', target: 'n7', style: { strokeWidth: 2 } },
  { id: 'e3-5', source: 'n3', target: 'n5', style: { strokeWidth: 2, strokeDasharray: '5,4' } },
  { id: 'e3-6', source: 'n3', target: 'n6', style: { strokeWidth: 2, strokeDasharray: '5,4' } },
  { id: 'e2-6', source: 'n2', target: 'n6', style: { strokeWidth: 2, strokeDasharray: '5,4' } },
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
  const [rightTab, setRightTab] = useState('editor'); // 'editor' | 'review' | 'completed'
  const [confirm, setConfirm] = useState({ open: false, targetId: null, message: '' });
  const [running, setRunning] = useState(false);

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
    setEdges((eds) => addEdge({ ...params, style: { strokeWidth: 2 } }, eds));
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
    setRightTab('editor');
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
      return { ...n, position: { x: 40 + layer * 220, y: 80 + index * 100 } };
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
      // Extract human-review items
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
        setRightTab('review');
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
            fitView
            fitViewOptions={{ padding: 0.2 }}
            deleteKeyCode={null}
          >
            <Background color="#c7d2fe" gap={20} size={1} style={{ opacity: 0.4 }} />
            <Controls />
            <MiniMap
              nodeColor={(node) => {
                const colors = { LoadData: '#3b82f6', Task: '#f97316', CandidateLLMs: '#8b5cf6', Filter: '#22c55e', Router: '#ec4899', Annotate: '#f59e0b', Output: '#14b8a6' };
                return colors[node.data?.label] || '#94a3b8';
              }}
              style={{ background: '#fff', borderRadius: 10, border: '1px solid rgba(15,23,42,0.08)' }}
            />

            {/* Toolbar */}
            <Panel position="top-left">
              <div className="panel-actions">
                {/* Run */}
                <button
                  className="btn btn-primary"
                  onClick={handleRun}
                  disabled={running}
                  title="Run the pipeline"
                >
                  {running ? (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" className="spin-icon">
                        <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="2.5"/>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke="#fff" strokeWidth="2.5" strokeLinecap="round"/>
                      </svg>
                      <span>Running…</span>
                    </>
                  ) : (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="white">
                        <path d="M5 3v18l15-9L5 3z"/>
                      </svg>
                      <span>Run Pipeline</span>
                    </>
                  )}
                </button>

                <div style={{ width: 1, height: 20, background: 'rgba(15,23,42,0.1)', margin: '0 2px' }} />

                {/* Compact layout */}
                <button className="btn btn-secondary" onClick={compactLayout} title="Auto-arrange nodes">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <rect x="3" y="3" width="6" height="6" rx="1.5" stroke="#374151" strokeWidth="2"/>
                    <rect x="15" y="3" width="6" height="6" rx="1.5" stroke="#374151" strokeWidth="2"/>
                    <rect x="9" y="15" width="6" height="6" rx="1.5" stroke="#374151" strokeWidth="2"/>
                    <path d="M6 9v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3V9" stroke="#374151" strokeWidth="1.8" strokeLinecap="round"/>
                  </svg>
                  <span>Layout</span>
                </button>

                {/* Undo */}
                <button
                  className="btn btn-secondary"
                  onClick={handleUndo}
                  disabled={!canUndo}
                  title="Undo last action"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <path d="M21 7v6h-6" stroke="#374151" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M3 17a9 9 0 0 1 15.9-6.36L21 11" stroke="#374151" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>Undo</span>
                </button>

                {/* Delete */}
                <button
                  className="btn btn-danger"
                  onClick={deleteSelected}
                  disabled={!selectedNode}
                  title="Delete selected node"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <path d="M3 6h18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M8 6V4h8v2" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M19 6l-1 14H6L5 6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>Delete</span>
                </button>
              </div>
            </Panel>
          </ReactFlow>
        </div>

        {/* ── Right panel ── */}
        <div className="right-panel">
          {/* Tabs */}
          <div className="right-tabs">
            <div
              className={'right-tab' + (rightTab === 'editor' ? ' active' : '')}
              onClick={() => setRightTab('editor')}
            >
              Properties
            </div>
            <div
              className={'right-tab' + (rightTab === 'review' ? ' active' : '')}
              onClick={() => setRightTab('review')}
            >
              Review Queue
              {reviewItems.length > 0 && (
                <span style={{ marginLeft: 5, background: '#ef4444', color: '#fff', fontSize: 10, fontWeight: 700, borderRadius: 999, padding: '1px 5px' }}>
                  {reviewItems.length}
                </span>
              )}
            </div>
            <div
              className={'right-tab' + (rightTab === 'completed' ? ' active' : '')}
              onClick={() => setRightTab('completed')}
            >
              Completed
            </div>
          </div>

          {/* Tab bodies */}
          {rightTab === 'editor' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {selectedNode ? (
                <NodeEditor
                  node={selectedNode}
                  updateNodeData={updateNodeData}
                  deleteNode={deleteNodeById}
                  openConfirm={(id, message) => setConfirm({ open: true, targetId: id, message })}
                />
              ) : (
                <div style={{ padding: '32px 20px', textAlign: 'center' }}>
                  <svg width="44" height="44" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 12px', display: 'block', opacity: 0.2 }}>
                    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#374151" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M2 17l10 5 10-5" stroke="#374151" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M2 12l10 5 10-5" stroke="#374151" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <div style={{ fontSize: 13.5, fontWeight: 600, color: '#374151' }}>No node selected</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 5, lineHeight: 1.5 }}>
                    Click on a node in the canvas to edit its parameters
                  </div>
                </div>
              )}
            </div>
          )}

          {rightTab === 'review' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <ReviewQueue items={reviewItems} onUpdate={(items) => setReviewItems(items)} />
            </div>
          )}

          {rightTab === 'completed' && (
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <CompletedSamples />
            </div>
          )}
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
