from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import json
import torch

from .base_router import BaseRouter


class KNNRouter(BaseRouter):
    """KNN-based router using historical annotated samples."""
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 5, device: Optional[str] = None, ann_path: Optional[str] = None):
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(self.encoder_name, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.k = int(k)
        self.sample_embs: Optional[np.ndarray] = None
        self.routes: List[str] = []
        self.ann_path = ann_path

    @property
    def if_train(self):
        self.ready = False
        return True

    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
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

    def build_index_from_annotations(self, ann_path: str):
        with open(ann_path, encoding='utf-8') as f:
            data = json.load(f)
        samples = []
        routes = []
        for d in data:
            sample_text = d.get("text") or f"Q: {d.get('question','')}\nContext: {d.get('context','')}"
            routed = d.get('route')
            if routed is None:
                continue
            samples.append(sample_text)
            routes.append(routed)
        if len(samples) == 0:
            raise ValueError("No routed samples found in annotations to build KNN index")
        embs = self._encode_texts(samples)
        self.sample_embs = embs.astype(np.float32)
        self.routes = routes

    def score(self, sample: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
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
        k = min(self.k, sims.shape[0])
        top_idx = np.argsort(-sims)[:k]
        weights = sims[top_idx]
        cand_scores = {c: 0.0 for c in candidate_llms}
        total_weight = float(np.sum(np.abs(weights))) + 1e-12
        for idx, w in zip(top_idx, weights):
            routed = self.routes[idx]
            if routed in cand_scores:
                cand_scores[routed] += float(max(w, 0.0))
        out = []
        for c in candidate_llms:
            score = cand_scores.get(c, 0.0) / total_weight
            out.append({'model': c, 'score': float(score)})
        out.sort(key=lambda x: x['score'], reverse=True)
        return out

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        if self.sample_embs is None:
            raise RuntimeError("No index to save. Call build_index_from_annotations first.")
        np.save(os.path.join(dirpath, 'sample_embs.npy'), self.sample_embs)
        with open(os.path.join(dirpath, 'routes.json'), 'w', encoding='utf-8') as f:
            json.dump(self.routes, f)
        meta = {
            'encoder_name': self.encoder_name,
            'k': int(self.k),
            'embedding_dim': int(self.sample_embs.shape[1]) if self.sample_embs is not None and self.sample_embs.size>0 else None
        }
        with open(os.path.join(dirpath, 'meta_knn.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, dirpath: str, device: Optional[str] = None):
        meta_path = os.path.join(dirpath, 'meta_knn.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        router = cls(encoder_name=meta.get('encoder_name'), k=meta.get('k', 5), device=device)
        emb_path = os.path.join(dirpath, 'sample_embs.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(emb_path)
        router.sample_embs = np.load(emb_path)
        routes_path = os.path.join(dirpath, 'routes.json')
        if not os.path.exists(routes_path):
            raise FileNotFoundError(routes_path)
        with open(routes_path, encoding='utf-8') as f:
            router.routes = json.load(f)
        return router


    def build_from_annotations(self, out_dir: str):
        with open(self.ann_path, encoding='utf-8') as f:
            data = json.load(f)
        samples = []
        routes = []
        for d in data:
            sample_text = d.get("text") or f"Q: {d.get('question','')}\nContext: {d.get('context','')}"
            routed = d.get('route')
            if routed is None:
                continue
            samples.append(sample_text)
            routes.append(routed)
        if len(samples) == 0:
            raise ValueError("No routed samples found in annotations to build KNN index")
        embs = self._encode_texts(samples)
        self.sample_embs = embs.astype(np.float32)
        self.routes = routes
        self.save(out_dir)
