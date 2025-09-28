# ==============================
# Active Learning 模块
# ==============================

import numpy as np
from typing import List, Sequence
from abc import ABC, abstractmethod
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class DataPool:
    """数据池, 包含文本和对应ID"""
    def __init__(self, texts: List[str], ids: List[str]):
        self.texts = texts
        self.ids = ids


class Embeddings(ABC):
    """嵌入器基类：文本转向量接口"""
    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回 shape=[N, D] 的向量"""
        return NotImplementedError


class Selector(ABC):
    """主动学习采样器基类，需实现 select_indices"""
    def __init__(self, emb: Embeddings, budget: int, batch_size: int = 32, seed: int = 42):
        self.emb = emb
        self.budget = budget
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def run(self, unlabeled: DataPool, labeled_ids: List[str] = None) -> List[str]:
        """给定未标注池和已标注ID, 返回采样ID列表"""
        picked: List[str] = []
        labeled_ids = set(labeled_ids or [])
        X = self.emb.encode(unlabeled.texts)
        id_arr = np.array(unlabeled.ids)
        while len(picked) < self.budget:
            X_remain = X[~np.isin(id_arr, picked + list(labeled_ids))]
            ids_remain = id_arr[~np.isin(id_arr, picked + list(labeled_ids))]
            order = self.select_indices(X_remain, picked_ids=picked, all_X=X, all_ids=id_arr)
            chosen = ids_remain[order[: self.batch_size]].tolist()
            picked.extend(chosen)
        return picked[: self.budget]

    @abstractmethod
    def select_indices(self, X_remain: np.ndarray, picked_ids: List[str], all_X: np.ndarray, all_ids: np.ndarray) -> np.ndarray:
        """返回未标注池的采样顺序索引，越靠前越优"""
        return NotImplementedError


class BertEmbeddings(Embeddings):
    """BERT向量嵌入"""
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model(**enc)
        return out.pooler_output.cpu().numpy()


class BertKM(Selector):
    """BERT+KMeans采样器"""
    def __init__(self, emb: Embeddings, budget: int, batch_size: int = 32, k_factor: float = 1.0):
        super().__init__(emb, budget, batch_size)
        self.k_factor = k_factor

    def select_indices(self, X_remain, picked_ids, all_X, all_ids):
        k = max(1, int(self.batch_size * self.k_factor))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_remain)
        centers = km.cluster_centers_
        nn_idx, _ = pairwise_distances_argmin_min(centers, X_remain)
        return nn_idx


class SurprisalEmbeddings(Embeddings):
    """基于MLM困惑度的Surprisal嵌入"""
    def __init__(self, model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 batch_size: int = 32,
                 device: str = None,
                 no_mask_eval_15pct: bool = True,
                 mlm_probability: float = 0.15,
                 hist_bins: int = 32, loss_clip: float = 10.0,
                 seed: int = 42):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.no_mask_eval_15pct = no_mask_eval_15pct
        self.mlm_probability = mlm_probability
        self.hist_bins = hist_bins
        self.loss_clip = loss_clip
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def _prep_batch(self, texts: Sequence[str]):
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _random_15pct_mask_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special = []
        for row in input_ids.tolist():
            special.append(self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True))
        special = torch.tensor(special, dtype=torch.bool, device=input_ids.device)
        if self.tokenizer.pad_token_id is not None:
            pad = input_ids.eq(self.tokenizer.pad_token_id)
        else:
            pad = torch.zeros_like(input_ids, dtype=torch.bool)
        eligible = ~(special | pad)
        prob = torch.full_like(input_ids, fill_value=self.mlm_probability, dtype=torch.float32)
        bern = torch.bernoulli(prob).bool().to(input_ids.device)
        mask = bern & eligible
        for b in range(B):
            if not mask[b].any() and eligible[b].any():
                idx = torch.nonzero(eligible[b], as_tuple=False).squeeze(1)
                j = idx[self.rng.integers(0, len(idx))]
                mask[b, j] = True
        return mask

    @torch.no_grad()
    def _get_token_losses(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        if self.no_mask_eval_15pct:
            labels = input_ids.clone()
            sel = self._random_15pct_mask_positions(input_ids)
            labels[~sel] = -100
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        else:
            inputs_masked, labels = self._mlm_mask_inputs(input_ids.clone())
            outputs = self.model(input_ids=inputs_masked, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
        V = logits.size(-1)
        loss_fct = CrossEntropyLoss(reduction="none")
        loss_flat = loss_fct(logits.view(-1, V), labels.view(-1))
        loss = loss_flat.view(B, L)
        if self.no_mask_eval_15pct:
            loss = torch.where(labels.eq(-100), torch.zeros_like(loss), loss)
        return loss

    def _mlm_mask_inputs(self, input_ids: torch.Tensor):
        labels = input_ids.clone()
        prob = torch.full_like(input_ids, fill_value=self.mlm_probability, dtype=torch.float32)
        special = []
        for row in input_ids.tolist():
            special.append(self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True))
        special = torch.tensor(special, dtype=torch.bool, device=input_ids.device)
        if self.tokenizer.pad_token_id is not None:
            pad = input_ids.eq(self.tokenizer.pad_token_id)
        else:
            pad = torch.zeros_like(input_ids, dtype=torch.bool)
        prob.masked_fill_(special | pad, 0.0)
        masked_indices = torch.bernoulli(prob).bool().to(input_ids.device)
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full_like(input_ids, 0.8, dtype=torch.float32)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = (torch.bernoulli(torch.full_like(input_ids, 0.5, dtype=torch.float32)).bool()
                          & masked_indices & ~indices_replaced)
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, device=input_ids.device, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        return input_ids, labels

    def _loss_histogram(self, loss_row: torch.Tensor) -> np.ndarray:
        vals = loss_row[loss_row > 0].clamp(max=self.loss_clip).detach().cpu().numpy()
        if vals.size == 0:
            h = np.zeros(self.hist_bins, dtype=np.float32)
            h[0] = 1.0
            return h
        hist, _ = np.histogram(vals, bins=self.hist_bins, range=(0.0, self.loss_clip), density=False)
        h = hist.astype(np.float32)
        h /= (h.sum() + 1e-8)
        return h

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.model.eval()
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i: i + self.batch_size]
            batch = self._prep_batch(chunk)
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            loss = self._get_token_losses(input_ids, attn)
            for b in range(loss.size(0)):
                vecs.append(self._loss_histogram(loss[b]))
        X = np.stack(vecs, axis=0)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X


class ALPS(Selector):
    """ALPS采样器"""
    def select_indices(self, X_remain, picked_ids, all_X, all_ids):
        km = KMeans(n_clusters=self.batch_size, n_init=10, random_state=0).fit(X_remain)
        nn_idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X_remain)
        centroids_set = np.unique(nn_idx)
        m = self.batch_size - len(centroids_set)
        if m > 0:
            pool = np.delete(np.arange(len(X_remain)), centroids_set)
            p = np.random.choice(len(pool), m, replace=False)
            nn_idx = np.concatenate((centroids_set, pool[p]), axis=None)
        return nn_idx
