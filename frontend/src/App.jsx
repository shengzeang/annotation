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
import axios from 'axios';
import { PlayIcon, TrashIcon, PencilSquareIcon, DocumentTextIcon } from '@heroicons/react/24/solid';

const initialNodes = [
  // Starting nodes
  {
    id: 'n1',
    type: 'compact',
    position: { x: 40, y: 60 },
    data: { label: 'LoadData', params: { dataset: 'squad', max_samples: 200 } },
  },
  {
    id: 'n2',
    type: 'compact',
    position: { x: 40, y: 200 },
    data: { label: 'Task', params: { task_class: 'tasks.qa.QATask' } },
  },
  {
    id: 'n3',
    type: 'compact',
    position: { x: 40, y: 340 },
    data: { label: 'CandidateLLMs', params: { candidate_llms: ['gpt2', 'distilgpt2'] } },
  },

  // Pipeline nodes
  {
    id: 'n4',
    type: 'compact',
    position: { x: 260, y: 200 },
    data: { label: 'Filter', params: { filter_class: 'filters.al_filter.ActiveLearningFilter', filter_params: { method: 'alps', budget: 100, batch_size: 10 } } },
  },
  {
    id: 'n5',
    type: 'compact',
    position: { x: 460, y: 200 },
    data: { label: 'Router', params: { router_class: 'routers.knn_router.KNNRouter', router_params: { k: 5 }, candidate_llms: ['gpt2', 'distilgpt2'] } },
  },
  {
    id: 'n6',
    type: 'compact',
    position: { x: 660, y: 200 },
    data: { label: 'Annotate', params: { candidate_llms: ['gpt2', 'distilgpt2'], llm_mode: 'local', task_class: 'tasks.qa.QATask', min_confidence: 0.5 } },
  },
  {
    id: 'n7',
    type: 'compact',
    position: { x: 860, y: 200 },
    data: { label: 'Output', params: { path: 'out/annotations.json' } },
  }
];

const nodeTypes = { compact: CompactNode };

// Connect starting nodes into the pipeline:
// LoadData -> Filter -> Router -> Annotate -> Output
// CandidateLLMs -> Router and CandidateLLMs -> Annotate
// Task -> Annotate
const initialEdges = [
  { id: 'e1-4', source: 'n1', target: 'n4' },
  { id: 'e4-5', source: 'n4', target: 'n5' },
  { id: 'e5-6', source: 'n5', target: 'n6' },
  { id: 'e6-7', source: 'n6', target: 'n7' },

  { id: 'e3-5', source: 'n3', target: 'n5' },
  { id: 'e3-6', source: 'n3', target: 'n6' },
  { id: 'e2-6', source: 'n2', target: 'n6' },
  ];
export default function App() {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [prevNodes, setPrevNodes] = useState(null);
  const [undoStack, setUndoStack] = useState([]);
  const [reviewItems, setReviewItems] = useState([]);
  const [rightView, setRightView] = useState('node'); // 'node' | 'results'

  const [runResults, setRunResults] = useState(null);
  const [displayResults, setDisplayResults] = useState(null);
  const [latestRunId, setLatestRunId] = useState(null);
  
  const [runCols, setRunCols] = useState({});
  const [runPage, setRunPage] = useState({});
  const [runPerPage, setRunPerPage] = useState({});
  
  // Transform runResults by fetching any saved output files so UI shows file contents
  React.useEffect(() => {
    let mounted = true;
    const fetchAll = async () => {
      if (!runResults) { setDisplayResults(null); return; }
      try {
        const entries = await Promise.all(Object.entries(runResults).map(async ([nodeId, out]) => {
          if (out && typeof out === 'object' && out.saved_to) {
            try {
              const resp = await axios.get('http://localhost:5000/read_output', { params: { run_id: latestRunId, node_id: nodeId } });
              if (resp.data && resp.data.status === 'ok') {
                // if the file contains array or {items: [...]}, normalize
                return [nodeId, resp.data.data];
              }
            } catch (e) {
              // fallthrough to return original
            }
          }
          return [nodeId, out];
        }));
        if (!mounted) return;
        setDisplayResults(Object.fromEntries(entries));
      } catch (e) {
        console.error('Failed to fetch output files', e);
        setDisplayResults(runResults);
      }
    };
    fetchAll();
    return () => { mounted = false; };
  }, [runResults, latestRunId]);
  // runProgress removed; node-local progress will be stored on each node's data.progress

  // load persisted review queue on mount
  React.useEffect(() => {
    let mounted = true;
    const fetchServerQueue = async () => {
      try {
        const res = await axios.get('http://localhost:5000/review_submissions');
        if (!mounted) return;
        if (res.data && res.data.status === 'ok') {
          const serverItems = (res.data.items || []).map((it) => ({ ...it, _server: true }));
          // merge: keep run-only items (those with _server !== true) and add/replace server items
          setReviewItems((cur) => {
            // helper to compare
            const same = (a, b) => {
              if (!a || !b) return false;
              if (a.id && b.id && a.id === b.id) return true;
              if (a.qid && b.qid && a.qid === b.qid) return true;
              if (a.question && b.question && a.question === b.question) return true;
              if (a.text && b.text && a.text === b.text) return true;
              return false;
            };
            const runOnly = (cur || []).filter((x) => !x._server);
            // for serverItems, prefer server copy; avoid duplicates
            const merged = [...serverItems];
            runOnly.forEach((r) => {
              const exists = merged.find((s) => same(s, r));
              if (!exists) merged.push(r);
            });
            return merged;
          });
        }
      } catch (e) {
        // ignore
      }
    };

    // initial fetch
    fetchServerQueue();
    // poll every 10s
    const iv = setInterval(fetchServerQueue, 10000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);

  // push a snapshot of current nodes+edges to undo stack (cap at 10 entries)
  const pushSnapshot = React.useCallback(() => {
    setUndoStack((s) => {
      const next = [
        ...s,
        {
          nodes: nodes.map((n) => ({ ...n })),
          edges: edges.map((e) => ({ ...e })),
        },
      ];
      if (next.length > 10) return next.slice(next.length - 10);
      return next;
    });
  }, [nodes, edges]);

  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);
  const onConnect = useCallback((params) => {
    // snapshot before adding an edge so it can be undone
    pushSnapshot();
    setEdges((eds) => addEdge(params, eds));
  }, [pushSnapshot]);

  const onElementsRemove = useCallback((elementsToRemove) => {
    pushSnapshot();
    const removeIds = new Set(elementsToRemove.map((el) => el.id));
    setNodes((nds) => nds.filter((n) => !removeIds.has(n.id)));
    setEdges((eds) => eds.filter((e) => !removeIds.has(e.id) && !removeIds.has(e.source) && !removeIds.has(e.target)));
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
    try { setRightView('node'); } catch (e) {}
  }, [setSelectedNode, setRightView]);

  const deleteNodeById = useCallback((id) => {
    if (!id) return;
    // push snapshot for undo
    pushSnapshot();
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id && e.id !== id));
    setSelectedNode((cur) => (cur && cur.id === id ? null : cur));
  }, [pushSnapshot]);

  const deleteSelected = useCallback(() => {
    if (!selectedNode) return;
    // show confirmation modal instead of deleting immediately
    setConfirm({ open: true, targetId: selectedNode.id, message: `Delete node "${selectedNode.data?.label || selectedNode.id}"?` });
  }, [selectedNode, deleteNodeById]);

  const [confirm, setConfirm] = React.useState({ open: false, targetId: null, message: '' });

  const handleConfirmCancel = () => setConfirm({ open: false, targetId: null, message: '' });
  const handleConfirmDelete = () => {
    if (confirm && confirm.targetId) deleteNodeById(confirm.targetId);
    setConfirm({ open: false, targetId: null, message: '' });
  };

  // Note: keyboard Delete/Backspace handler removed — deletions happen via Delete button only

  const handleRun = async () => {
    const payload = { nodes, edges };
    try {
    const res = await axios.post('http://localhost:5000/run_graph', payload);
    const run_id = res.data.run_id;
    setLatestRunId(run_id);
      // start polling progress if run_id provided
      if (run_id) {
        // clear previous node progress for Filter/Router/Annotate
        setNodes((prev) => prev.map((n) => {
          const label = n.data && n.data.label ? n.data.label : '';
          if (['Filter', 'Router', 'Annotate'].includes(label)) {
            const d = { ...n.data };
            delete d.progress;
            return { ...n, data: d };
          }
          return n;
        }));
        // Use Server-Sent Events to update node-local progress live
        try {
          const es = new EventSource(`http://localhost:5000/run_progress_stream?run_id=${run_id}`);
          es.onmessage = (ev) => {
            try {
              const data = JSON.parse(ev.data || '{}') || {};
              const nodeProgress = (data && data.nodes) ? data.nodes : {};
              if (nodeProgress && Object.keys(nodeProgress).length) {
                setNodes((prev) => prev.map((n) => {
                  const p = nodeProgress[n.id];
                  if (p) {
                    return { ...n, data: { ...n.data, progress: p } };
                  }
                  return n;
                }));
              }
              if (data.status && data.status !== 'running') {
                // fetch final outputs from server and show them in UI
                (async () => {
                          try {
                              const resp = await axios.get('http://localhost:5000/run_progress', { params: { run_id } });
                              if (resp.data && resp.data.status === 'ok') {
                                const prog = resp.data.progress || {};
                                setRunResults(prog.outputs || prog.context || {});
                              }
                  } catch (e) {
                    console.error('failed fetching final run results', e);
                  }
                })();
                try { es.close(); } catch (e) {}
              }
            } catch (e) {
              console.error('failed parsing progress stream', e);
            }
          };
          es.onerror = (err) => {
            console.error('progress stream error', err);
            try { es.close(); } catch (e) {}
          };
          window.__run_progress_es = es;
        } catch (e) {
          console.warn('SSE not available; no live node progress will be shown', e);
        }
      }
      // Run started; UI updates are handled via SSE — no blocking alert.
      console.log('Run result:', res.data);
      // extract human-review items from result context/outputs
      try {
        const ctx = res.data.context || res.data.outputs || {};
        const found = [];
        Object.values(ctx).forEach((val) => {
          if (Array.isArray(val)) {
            val.forEach((it) => { if (it && it.needs_human) found.push({ ...it, _server: false }); });
          }
        });
        // merge with current reviewItems (server items kept)
        setReviewItems((cur) => {
          const same = (a, b) => {
            if (!a || !b) return false;
            if (a.id && b.id && a.id === b.id) return true;
            if (a.qid && b.qid && a.qid === b.qid) return true;
            if (a.question && b.question && a.question === b.question) return true;
            if (a.text && b.text && a.text === b.text) return true;
            return false;
          };
          const curServer = (cur || []).filter((x) => x && x._server) || [];
          const runOnly = found.filter((f) => !curServer.find((s) => same(s, f)));
          return [...curServer, ...runOnly];
        });
      } catch (e) {
        console.error('Failed to extract review items', e);
      }
    } catch (err) {
      console.error(err);
      alert('Run failed: ' + err.message);
    }
  };

  const compactLayout = useCallback(() => {
    // Save previous node positions so layout can be undone
    setPrevNodes(nodes.map((n) => ({ ...n })));
    // Smart compact layout:
    // 1. Try to compute node layers via a topological-like pass using edges (sources -> targets)
    // 2. If graph has cycles or topological layering doesn't reach all nodes, fallback to grouping by node label types
    // 3. Place nodes column-by-column according to layer, with compact vertical spacing
    const nodeMap = new Map(nodes.map((n) => [n.id, { ...n, layer: 0 }]));

    // build adjacency and in-degree
    const adj = new Map();
    const inDegree = new Map();
    nodes.forEach((n) => {
      adj.set(n.id, []);
      inDegree.set(n.id, 0);
    });
    edges.forEach((e) => {
      if (!adj.has(e.source)) adj.set(e.source, []);
      adj.get(e.source).push(e.target);
      inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
    });

    // Kahn-like processing to compute layers
    const q = [];
    inDegree.forEach((deg, id) => { if (deg === 0) q.push(id); });
    let processed = 0;
    while (q.length) {
      const id = q.shift();
      processed += 1;
      const node = nodeMap.get(id);
      const neighbors = adj.get(id) || [];
      neighbors.forEach((nb) => {
        const nbNode = nodeMap.get(nb);
        if (nbNode) nbNode.layer = Math.max(nbNode.layer, node.layer + 1);
        inDegree.set(nb, inDegree.get(nb) - 1);
        if (inDegree.get(nb) === 0) q.push(nb);
      });
    }

    // If not all nodes processed (cycle), fallback to grouping by label order
    if (processed < nodes.length) {
      const labelOrder = {
        LoadData: 0,
        Task: 1,
        CandidateLLMs: 2,
        Filter: 3,
        Router: 4,
        Annotate: 5,
        Output: 6,
      };
      nodes.forEach((n) => {
        const label = n.data && n.data.label ? n.data.label : '';
        const ord = labelOrder[label] !== undefined ? labelOrder[label] : 10;
        nodeMap.get(n.id).layer = ord;
      });
    }

    // Group nodes by layer
    const layers = [];
    nodeMap.forEach((n) => {
      const l = n.layer || 0;
      if (!layers[l]) layers[l] = [];
      layers[l].push(n);
    });

    const startX = 40;
    const spacingX = 200;
    const spacingY = 88;
    const baseY = 100;

    // Normalize: compact each layer vertically
    const newNodes = nodes.map((n) => {
      const nObj = nodeMap.get(n.id);
      const layer = nObj.layer || 0;
      const column = layers[layer] || [];
      const index = column.findIndex((c) => c.id === n.id);
      const x = startX + layer * spacingX;
      const y = baseY + index * spacingY;
      return { ...n, position: { x, y } };
    });

    setNodes(newNodes);
  }, [nodes, edges, setNodes]);

  const undoLayout = useCallback(() => {
    if (prevNodes) {
      setNodes(prevNodes.map((n) => ({ ...n })));
      setPrevNodes(null);
    }
  }, [prevNodes, setNodes]);

  const handleUndo = useCallback(() => {
    // If we have undo history (deletes/edits), restore last snapshot
    if (undoStack && undoStack.length > 0) {
      const last = undoStack[undoStack.length - 1];
      setNodes(last.nodes.map((n) => ({ ...n })));
      setEdges(last.edges.map((e) => ({ ...e })));
      setUndoStack((s) => s.slice(0, -1));
      setSelectedNode(null);
      return;
    }
    // Otherwise fall back to layout undo
    undoLayout();
  }, [undoStack, setNodes, setEdges, setUndoStack, undoLayout]);

  const updateNodeData = (id, data) => {
    // snapshot before editing node params so it can be undone
    pushSnapshot();
    setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...data } } : n)));
    const n = nodes.find((x) => x.id === id);
    setSelectedNode(n ? { ...n, data: { ...n.data, ...data } } : null);
  };



  return (
    <div className="h-screen flex flex-row">
      <div className="w-52 p-3">
        <Sidebar setNodes={setNodes} setEdges={setEdges} nodes={nodes} />
      </div>

      <div className="flex-1 flex flex-col">
        <ReactFlowProvider>
          <div className="flex flex-row h-full">
            <div className="flex-1 flex flex-col">
              <div className="h-[60vh] min-h-[360px] border-b border-gray-200 relative">
                <div className="absolute top-[10px] right-[12px] z-50">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleRun}
                      title="Run the current graph"
                      className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-3 py-1.5 rounded"
                    >
                      <PlayIcon className="w-4 h-4" />
                      <span>Run Graph</span>
                    </button>

                    <button
                      onClick={compactLayout}
                      title="Rearrange nodes compactly"
                      className="inline-flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1.5 rounded"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-gray-700">
                        <path d="M12 4v16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
                      </svg>
                      <span>Compact</span>
                    </button>

                    <button
                      onClick={handleUndo}
                      disabled={!(prevNodes || (undoStack && undoStack.length > 0))}
                      title="Undo last action (deletes or layout)"
                      className={`inline-flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded ${!(prevNodes || (undoStack && undoStack.length > 0)) ? 'bg-gray-100 text-gray-400 cursor-not-allowed opacity-60' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M21 7v6h-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 17a9 9 0 0115.9-6.36L21 11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      <span>Undo</span>
                    </button>

                    <button
                      onClick={deleteSelected}
                      disabled={!selectedNode}
                      title="Delete selected node"
                      className={`inline-flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded ${!selectedNode ? 'bg-red-200 text-red-400 cursor-not-allowed opacity-60' : 'bg-red-600 hover:bg-red-700 text-white'}`}
                    >
                      <TrashIcon className="w-4 h-4" />
                      <span>Delete</span>
                    </button>
                  </div>
                </div>

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
                  className="w-full h-full"
                >
                  <Background />
                  <Controls />
                  <MiniMap />
                </ReactFlow>
              </div>
              <div className="flex-1 flex gap-3 p-3 overflow-auto border-t bg-white">
                <div className="flex-1 border border-gray-200 p-2 rounded bg-white overflow-auto">
                  <h4 className="mt-0 mb-2 text-sm font-semibold">Review Queue</h4>
                  <ReviewQueue items={reviewItems} onUpdate={(items) => setReviewItems(items)} />
                </div>
              </div>
            </div>
            <div className="w-[360px] border-l border-gray-200 p-3 bg-white rounded flex flex-col">
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => setRightView('node')}
                  className={`inline-flex items-center gap-2 px-3 py-1.5 rounded ${rightView === 'node' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'}`}
                >
                  <PencilSquareIcon className="w-4 h-4" />
                  <span>Node Editor</span>
                </button>

                <button
                  onClick={() => setRightView('results')}
                  className={`inline-flex items-center gap-2 px-3 py-1.5 rounded ${rightView === 'results' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'}`}
                >
                  <DocumentTextIcon className="w-4 h-4" />
                  <span>Annotation Results</span>
                </button>
              </div>

              <div className="flex-1 overflow-auto">
                {rightView === 'node' && (
                  <div>
                    <h3 className="mt-0 text-lg font-semibold">Node Editor</h3>
                    {selectedNode ? (
                      <NodeEditor node={selectedNode} updateNodeData={updateNodeData} deleteNode={deleteNodeById} openConfirm={(id, message) => setConfirm({ open: true, targetId: id, message })} />
                    ) : (
                      <div className="text-gray-600">Select a node to edit its params</div>
                    )}
                  </div>
                )}

                {/* Review Queue moved under canvas */}

                {rightView === 'results' && (
                  <div>
                      <div className="flex justify-between items-center mt-0 mb-3">
                        <span className="text-lg font-semibold">Annotation Results</span>
                        <button
                        onClick={() => {
                          try {
                            const blob = new Blob([JSON.stringify({ run_id: latestRunId, outputs: runResults }, null, 2)], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `run_outputs_${latestRunId || 'latest'}.json`;
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                            URL.revokeObjectURL(url);
                          } catch (e) { console.error('download failed', e); }
                        }}
                        className="inline-flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1 rounded"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M12 3v12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/><path d="M8 11l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
                        <span>Download JSON</span>
                      </button>
                    </div>

                    {(!displayResults || Object.keys(displayResults).length === 0) && (<div>No outputs</div>)}
                    {displayResults && Object.keys(displayResults).length > 0 && (
                      <div>
                        {Object.entries(displayResults).map(([nodeId, out]) => {
                          const node = nodes.find((n) => n.id === nodeId);
                          let title = node ? (node.data && node.data.label) || nodeId : nodeId;
                          try {
                            const nlabel = node && node.data && node.data.label;
                            if (nlabel === 'Output') title = 'Annotated';
                          } catch (e) {}
                          const rows = Array.isArray(out) ? out : (out && out.items) ? out.items : [out];
                          const getOriginal = (s) => (s && (s.question || s.text || s.input || s.source || s.context)) || (s && (s.id || s.qid)) || (typeof s === 'string' ? s : '');
                          const getAnnotation = (s) => (s && (s.annotation || s.prediction || s.answer || s.output)) || '';

                          return (
                            <div key={nodeId} className="mb-3">
                              <div className="font-semibold">{title} — {rows.length} items</div>
                              <div className="flex flex-col gap-2 mt-2">
                                {rows.map((r, i) => (
                                  <div key={i} className="border border-gray-200 p-2 rounded bg-gray-50 flex gap-3 items-start">
                                    <div className="flex-1">
                                      <div className="text-sm font-semibold mb-1">{getOriginal(r) || `item ${i}`}</div>
                                      <div className="text-sm text-slate-900"><strong>Annotation:</strong> {getAnnotation(r) || '(none)'}</div>
                                      {r.confidence !== undefined && (<div className="text-xs text-gray-500 mt-1">Confidence: {Number(r.confidence).toFixed(3)}</div>)}
                                      {r.notes && (<div className="text-xs text-gray-500 mt-1">Notes: {r.notes}</div>)}
                                    </div>
                                    <div className="w-[120px]" />
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <ConfirmModal open={confirm.open} title="Delete node" message={confirm.message} onConfirm={handleConfirmDelete} onCancel={handleConfirmCancel} />

        </ReactFlowProvider>
      </div>
    </div>
  );
}
