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
    data: { label: 'Filter', params: { filter_class: 'filters.al_filter.ActiveLearningFilter', filter_params: { method: 'alps', budget: 100, batch_size: 20 } } },
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
    data: { label: 'Annotate', params: { candidate_llms: ['gpt2', 'distilgpt2'], llm_mode: 'local', task_class: 'tasks.qa.QATask' } },
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
  }, []);

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
      alert('Run completed, check console for output.');
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
    <div style={{ height: '100vh', display: 'flex' }}>
      <ReactFlowProvider>
        <Sidebar setNodes={setNodes} setEdges={setEdges} nodes={nodes} />

        <div style={{ flex: 1 }}>
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
          >
            <Background />
            <Controls />
            <MiniMap />
            <Panel position="top-left">
              <div className="panel-actions">
                <button className="btn btn-primary" onClick={handleRun} title="Run the current graph">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 3v18l15-9L5 3z" fill="white"/></svg>
                  <span>Run Graph</span>
                </button>

                <button className="btn btn-secondary" onClick={compactLayout} title="Rearrange nodes compactly">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M12 2v20" stroke="#0f172a" strokeWidth="1.6" strokeLinecap="round"/></svg>
                  <span>Compact</span>
                </button>

                <button className="btn btn-secondary" onClick={handleUndo} disabled={!(prevNodes || (undoStack && undoStack.length > 0))} title="Undo last action (deletes or layout)">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M21 7v6h-6" stroke="#0f172a" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 17a9 9 0 0115.9-6.36L21 11" stroke="#0f172a" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  <span>Undo</span>
                </button>

                <button className="btn btn-danger" onClick={deleteSelected} disabled={!selectedNode} title="Delete selected node">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M3 6h18" stroke="white" strokeWidth="1.6" strokeLinecap="round"/><path d="M8 6v12" stroke="white" strokeWidth="1.6" strokeLinecap="round"/><path d="M16 6v12" stroke="white" strokeWidth="1.6" strokeLinecap="round"/></svg>
                  <span>Delete</span>
                </button>
              </div>
            </Panel>
          </ReactFlow>
        </div>

        <ConfirmModal open={confirm.open} title="Delete node" message={confirm.message} onConfirm={handleConfirmDelete} onCancel={handleConfirmCancel} />

        <div style={{ width: 320, borderLeft: '1px solid #ddd', padding: 8 }}>
          <h3>Node Editor</h3>
          {selectedNode ? (
            <NodeEditor node={selectedNode} updateNodeData={updateNodeData} deleteNode={deleteNodeById} openConfirm={(id, message) => setConfirm({ open: true, targetId: id, message })} />
          ) : (
            <div>Select a node to edit its params</div>
          )}
          <ReviewQueue items={reviewItems} onUpdate={(items) => setReviewItems(items)} />
        </div>
      </ReactFlowProvider>
    </div>
  );
}
