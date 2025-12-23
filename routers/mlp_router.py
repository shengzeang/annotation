from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import json

from base_structure.base_router import BaseRouter


class MLPRouter(BaseRouter):
    """
    A simple MLP-based router that uses encoded features of (sample, candidate_name)
    pairs and predicts a score in [0,1]. Supports training from labeled pairs and
    scoring new samples.
    """
    def __init__(self, annotator, candidate_llms, hidden_dim: int = 64, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None, train_budget: int = 50):
        self.annotator = annotator
        self.candidate_llms = candidate_llms
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(self.encoder_name, trust_remote_code=True)
        self.model: Optional[nn.Module] = None
        self.hidden_dim = hidden_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.train_budget = train_budget

    @property
    def if_train(self):
        self.ready = False
        return True

    class _Net(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

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
                    # mean pooling
                    last = out.last_hidden_state
                    attn = enc.get('attention_mask', None)
                    if attn is not None:
                        mask = attn.unsqueeze(-1).expand(last.size()).float()
                        emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    else:
                        emb = last.mean(1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    def _featurize(self, sample: str, candidate: str) -> np.ndarray:
        se = self._encode_texts([sample])[0]
        ce = self._encode_texts([candidate])[0]
        prod = se * ce
        diff = np.abs(se - ce)
        feat = np.concatenate([se, ce, prod, diff], axis=0)
        return feat

    def train(self, pairs: List[Dict[str, Any]], epochs: int = 5, lr: float = 1e-3, batch_size: int = 32):
        samples = [p['sample'] for p in pairs]
        cands = [p['candidate'] for p in pairs]
        scores = np.array([p['score'] for p in pairs], dtype=np.float32)
        sample_embs = self._encode_texts(samples)
        cand_embs = self._encode_texts(cands)
        feats = []
        for se, ce in zip(sample_embs, cand_embs):
            feats.append(np.concatenate([se, ce, se*ce, np.abs(se-ce)], axis=0))
        X = np.vstack(feats)
        input_dim = X.shape[1]
        self.model = MLPRouter._Net(input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(scores).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss/len(dataset):.6f}")

    def score(self, sample: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("MLPRouter model not trained. Call train() first.")
        sample_emb = self._encode_texts([sample])[0]
        cand_embs = self._encode_texts(candidate_llms)
        feats = np.vstack([np.concatenate([sample_emb, ce, sample_emb*ce, np.abs(sample_emb-ce)], axis=0) for ce in cand_embs])
        xb = torch.from_numpy(feats).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(xb).cpu().numpy()
        return [{'model': c, 'score': float(s)} for c,s in zip(candidate_llms, preds.tolist())]

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        if self.model is None:
            raise RuntimeError("No trained model to save")
        state_path = os.path.join(dirpath, "mlp_state.pt")
        torch.save(self.model.state_dict(), state_path)
        try:
            emb_dim = self._encode_texts(["__dummy_for_meta__"]).shape[1]
        except Exception:
            emb_dim = None
        meta = {
            "encoder_name": self.encoder_name,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
            "embedding_dim": int(emb_dim) if emb_dim is not None else None,
            "state_path": os.path.basename(state_path)
        }
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, dirpath: str, device: Optional[str] = None):
        meta_path = os.path.join(dirpath, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        router = cls(hidden_dim=meta.get("hidden_dim", 64), encoder_name=meta.get("encoder_name"), device=device)
        state_file = meta.get("state_path", "mlp_state.pt")
        state_path = os.path.join(dirpath, state_file)
        if not os.path.exists(state_path):
            raise FileNotFoundError(state_path)
        emb_dim = meta.get("embedding_dim", None)
        if emb_dim is None:
            dummy_emb = router._encode_texts(["__dummy_for_meta__"])
            emb_dim = int(dummy_emb.shape[1])
        input_dim = int(4 * int(emb_dim))
        router.model = MLPRouter._Net(input_dim, router.hidden_dim).to(router.device)
        state = torch.load(state_path, map_location=router.device)
        router.model.load_state_dict(state)
        router.model.eval()
        return router


    def build_from_annotations(self, annotations, out_dir: str, epochs: int = 200, batch_size: int = 32):
        pairs = []
        positive_score = 1.0
        negative_score = 0.0
        for d in annotations:
            sample_text = d.get("text") or f"Q: {d.get('question','')}\nContext: {d.get('context','')}"
            routed = d.get("route")
            for cand in self.candidate_llms:
                score = positive_score if cand == routed else negative_score
                pairs.append({"sample": sample_text, "candidate": cand, "score": float(score)})
        self.train(pairs, epochs=epochs, batch_size=batch_size)
        self.save(out_dir)
