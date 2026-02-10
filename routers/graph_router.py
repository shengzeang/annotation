from typing import List, Dict, Any, Optional
import numpy as np
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from base_structure.base_router import BaseRouter

try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class GraphRouter(BaseRouter):
    """Graph-based router: builds a bipartite graph between samples and models and propagates scores.
    Routing decisions are based on semantic similarity and graph-based score propagation"""
    def __init__(self, annotator, candidate_llms, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", topk: int = 8, alpha: float = 0.85, device: Optional[str] = None, train_budget: int = 50):
        self.annotator = annotator
        self.candidate_llms = candidate_llms
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(self.encoder_name, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.sample_texts: List[str] = []
        self.sample_embs: Optional[np.ndarray] = None
        self.model_list: List[str] = []
        self.sample_to_model_edges: List[List[str]] = []
        self.train_budget = train_budget

    @property
    def if_train(self):
        """Signals if router requires training phase before inference"""
        self.ready = False
        return True

    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Converts texts into vector embeddings using pooled output or masked mean pooling."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.encoder(**enc)
                if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    emb = out.pooler_output
                else:
                    last = out.last_hidden_state
                    attn = enc.get('attention_mask', None)
                    if attn is not None:
                        mask = attn.unsqueeze(-1).expand(last.size()).float()
                        emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    else:
                        emb = last.mean(1)
            all_embs.append(emb.cpu().numpy())
        if len(all_embs) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(all_embs)

    def _personalized_propagation(self, pref: np.ndarray, max_iter: int = 20, tol: float = 1e-6) -> np.ndarray:
        # pref: preference over samples shape (N,)
        """Personalized PageRank returns final relevance distribution"""
        N = len(pref)
        if N == 0:
            return np.zeros(0, dtype=np.float32)
        # build transition matrix P over samples using neighbor_idx
        rows = []
        cols = []
        data_vals = []
        for i, nbrs in enumerate(self._neighbor_idx):
            if len(nbrs) == 0:
                continue
            weight = 1.0 / len(nbrs)
            for j in nbrs:
                rows.append(j)
                cols.append(i)
                data_vals.append(weight)
        if len(rows) == 0:
            return pref
        if SCIPY_AVAILABLE:
            P = sp.csr_matrix((data_vals, (rows, cols)), shape=(N, N), dtype=np.float32)
            r = pref.copy().astype(np.float32)
            teleport = (1.0 - self.alpha) * pref
            for _ in range(max_iter):
                r_new = self.alpha * (P @ r) + teleport
                if np.linalg.norm(r_new - r) < tol:
                    r = r_new
                    break
                r = r_new
            return r
        else:
            P = np.zeros((N, N), dtype=np.float32)
            for rr, cc, dv in zip(rows, cols, data_vals):
                P[rr, cc] = dv
            r = pref.copy().astype(np.float32)
            teleport = (1.0 - self.alpha) * pref
            for _ in range(max_iter):
                r_new = self.alpha * (P @ r) + teleport
                if np.linalg.norm(r_new - r) < tol:
                    r = r_new
                    break
                r = r_new
            return r

    def score(self, sample: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        """Compute routing scores for candidate models given a new input sample,
        based on similarity to past routed samples and graph-based score propagation.
        Return ranked list of candidate models with normalized routing scores."""
        # if graph not built, fallback to simple token overlap
        if self.sample_embs is None or len(self.sample_embs) == 0:
            sample_words = set(sample.lower().split())
            scores = []
            for cand in candidate_llms:
                cand_words = set(cand.lower().split('/'))
                overlap = len(sample_words & cand_words)
                score = 1.0 - np.exp(-overlap)
                scores.append({'model': cand, 'score': score})
            scores.sort(key=lambda x: x['score'], reverse=True)
            return scores

        q_emb = self._encode_texts([sample])[0].astype(np.float32)
        a = self.sample_embs
        a_norm = np.linalg.norm(a, axis=1)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            sims = np.zeros(a.shape[0], dtype=np.float32)
        else:
            sims = (a @ q_emb) / (a_norm * (q_norm + 1e-12) + 1e-12)
        # preference over samples
        pref = np.maximum(sims, 0.0)
        if pref.sum() == 0:
            pref = np.ones_like(pref) / float(pref.size)
        else:
            pref = pref / pref.sum()
        # propagate
        r = self._personalized_propagation(pref)
        # aggregate scores per model
        model_scores = {m: 0.0 for m in candidate_llms}
        for idx, score_val in enumerate(r):
            routed = self.sample_to_model_edges[idx]
            if routed in model_scores:
                model_scores[routed] += float(score_val)
        # normalize
        total = sum(model_scores.values()) + 1e-12
        out = []
        for c in candidate_llms:
            out.append({'model': c, 'score': float(model_scores.get(c, 0.0) / total)})
        out.sort(key=lambda x: x['score'], reverse=True)
        return out

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        if self.sample_embs is None:
            raise RuntimeError("No graph built to save. Call build_graph_from_annotations first.")
        np.save(os.path.join(dirpath, 'graph_sample_embs.npy'), self.sample_embs)
        with open(os.path.join(dirpath, 'graph_sample_texts.json'), 'w', encoding='utf-8') as f:
            json.dump(self.sample_texts, f)
        with open(os.path.join(dirpath, 'graph_routes.json'), 'w', encoding='utf-8') as f:
            json.dump(self.sample_to_model_edges, f)
        with open(os.path.join(dirpath, 'graph_models.json'), 'w', encoding='utf-8') as f:
            json.dump(self.model_list, f)
        meta = {
            'encoder_name': self.encoder_name,
            'topk': int(self.topk),
            'alpha': float(self.alpha),
            'embedding_dim': int(self.sample_embs.shape[1]) if self.sample_embs is not None and self.sample_embs.size>0 else None
        }
        with open(os.path.join(dirpath, 'meta_graph.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, dirpath: str, device: Optional[str] = None):
        meta_path = os.path.join(dirpath, 'meta_graph.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        router = cls(encoder_name=meta.get('encoder_name'), topk=meta.get('topk', 8), alpha=meta.get('alpha', 0.85), device=device)
        emb_path = os.path.join(dirpath, 'graph_sample_embs.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(emb_path)
        router.sample_embs = np.load(emb_path)
        texts_path = os.path.join(dirpath, 'graph_sample_texts.json')
        routes_path = os.path.join(dirpath, 'graph_routes.json')
        models_path = os.path.join(dirpath, 'graph_models.json')
        if not os.path.exists(texts_path) or not os.path.exists(routes_path) or not os.path.exists(models_path):
            raise FileNotFoundError('graph files missing')
        with open(texts_path, encoding='utf-8') as f:
            router.sample_texts = json.load(f)
        with open(routes_path, encoding='utf-8') as f:
            router.sample_to_model_edges = json.load(f)
        with open(models_path, encoding='utf-8') as f:
            router.model_list = json.load(f)
        # rebuild neighbor idx from sample_embs
        a = router.sample_embs
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        sim = (a @ a.T) / (norms * norms.T)
        N = sim.shape[0]
        topk = min(router.topk + 1, N)
        idx = np.argpartition(-sim, range(topk), axis=1)[:, :topk]
        neigh_idx = []
        for i in range(N):
            row = idx[i]
            row = row[row != i]
            if len(row) > router.topk:
                row = row[:router.topk]
            neigh_idx.append(row.tolist())
        router._neighbor_idx = neigh_idx
        return router


    def build_from_annotations(self, annotations, out_dir: str):
        """Given previous texts and model chosen,
        Builds graph structure with each sample connected to top k most similar samples"""
        samples = []
        sample_model_edges = []
        models_set = set()
        for d in annotations:
            txt = d.get('text') or f"Q: {d.get('question','')}\nContext: {d.get('context','')}"
            route = d.get('route')
            if route is None:
                continue
            samples.append(txt)
            sample_model_edges.append(route)
            models_set.add(route)
        if len(samples) == 0:
            raise ValueError("No routed samples found in annotations to build Graph")
        model_list = sorted(list(models_set))
        self.sample_texts = samples
        self.sample_to_model_edges = sample_model_edges
        self.model_list = model_list
        embs = self._encode_texts(self.sample_texts)
        self.sample_embs = embs.astype(np.float32)

        # build sample-sample topk neighbor indices using cosine similarity
        a = self.sample_embs
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        sim = (a @ a.T) / (norms * norms.T)
        N = sim.shape[0]
        topk = min(self.topk + 1, N)
        idx = np.argpartition(-sim, range(topk), axis=1)[:, :topk]
        # Remove self index and limit topk
        neigh_idx = []
        for i in range(N):
            row = idx[i]
            row = row[row != i]
            if len(row) > self.topk:
                row = row[:self.topk]
            neigh_idx.append(row.tolist())
        self._neighbor_idx = neigh_idx
        self.save(out_dir)
