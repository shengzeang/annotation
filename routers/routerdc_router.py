from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json

from ..base_structure.base_router import BaseRouter


class RouterDCRouter(BaseRouter):
    """
    Router-DC-style
    
    For each LLM, compute a "model identity vector" from our_anno.json
    During inference, encode the query and pick the most similar LLM.(comparing the query embdding with the model embedding)
    """

    def __init__(self, candidate_llms: List[str], encoder_name: str ="sentence-transformers/all-MiniLM-L6-v2", device: str =None, epochs: int =10, lr: float =0.01, ann_path: Optional[str] = None):
        self.candidate_llms = candidate_llms
        self.model_index = {m: i for i, m in enumerate(candidate_llms)}
        self.num_models = len(candidate_llms)
        self.ann_path = ann_path

        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

        # Hyperparameters
        self.lr = lr
        self.epochs = epochs
        self.temperature = 0.07

        # Will be initialized after the first encode (because we need embedding dim)
        self.model_embs = None   # nn.Parameter (M × D)

    @property
    def if_train(self):
        self.ready = False
        return True

    #text embeddings from annotation(same as mlprouter)
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

    #method 2: using sample-LLM constrastve loss (as per RouterDC) to generate model/identity vectors
    def build_from_annotations(self, out_dir: str):
        with open(self.ann_path) as f:
            data = json.load(f)
        queries = []
        labels = []
        # Load (query, model_index)
        for d in data:
            route = d.get("route")
            if route not in self.model_index:
                continue
            text = d.get("text") or f"Q: {d.get('question')}\nContext: {d.get('context')}"
            queries.append(text)
            labels.append(self.model_index[route])

        print(f"[RouterDC] Encoding {len(queries)} queries...")
        X_np = self._encode_texts(queries)       
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)

        N, D = X.shape
        # Initialize model identity embeddings
        self.model_embs = nn.Parameter(torch.randn(self.num_models, D, device=self.device))
        optimizer = torch.optim.Adam([self.model_embs], lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        #train contrastive model. pull vector closer to corresponding query
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # logits: (N × M)
            logits = (X @ self.model_embs.t()) / self.temperature
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}  Loss = {loss.item():.4f}")

    def score(self, sample, candidate_llms):
        assert self.model_embs is not None, "Router not trained."

        # Encode sample
        q = self._encode_texts([sample])[0]
        q = torch.tensor(q, dtype=torch.float32, device=self.device)

        # Normalize query
        q = q / (q.norm() + 1e-12)
        # Normalize each model embedding
        model_embs = self.model_embs / (self.model_embs.norm(dim=1, keepdim=True) + 1e-12)
        #Cosine similarities
        sims = (q @ model_embs.t()).detach().cpu().numpy().tolist()
        return [{"model": m, "score": float(sims[self.model_index[m]])}
                for m in candidate_llms]