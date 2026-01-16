# 任务抽象接口（预留，便于未来多任务扩展）
from base_structure.base_task import Task
# ==============================
# Active Learning 模块
# ==============================

import numpy as np
from typing import List, Sequence
from abc import ABC, abstractmethod
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import logging

logger = logging.getLogger(__name__)




def _safe_kmeans_fit(X, n_clusters, mb_batch, use_mini=True, timeout=10.0):
    """Attempt to fit MiniBatchKMeans or KMeans in a thread with timeout.
    Returns cluster centers or None on timeout/error.
    """
    # Inline synchronous fit (no threads/processes). This may be slower
    # but avoids process/handle duplication issues on Windows.
    if timeout is not None:
        logger.debug('_safe_kmeans_fit: timeout parameter provided (%.2fs) but inline fit cannot be preempted', timeout)
    try:
        if use_mini:
            mb = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=mb_batch, max_iter=10)
            mb.fit(X)
            centers = mb.cluster_centers_
        else:
            km = KMeans(n_clusters=n_clusters, n_init=1, random_state=0, max_iter=100)
            km.fit(X)
            centers = km.cluster_centers_
        return centers
    except Exception as e:
        logger.exception('_safe_kmeans_fit: inline fit failed: %s', e)
        return None


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
    def __init__(self, emb: Embeddings, budget: int, batch_size: int = 32, seed: int = 42, force_fallback: bool = False):
        self.emb = emb
        self.budget = budget
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        # When True, skip clustering and use simple deterministic fallback selection
        self.force_fallback = force_fallback

    def run(self, unlabeled: DataPool, labeled_ids: List[str] = None) -> List[str]:
        """给定未标注池和已标注ID, 返回采样ID列表"""
        picked: List[str] = []
        labeled_ids = set(labeled_ids or [])
        import time
        logger.info("Selector.run: encoding %d texts...", len(unlabeled.texts))
        t0 = time.time()
        X = self.emb.encode(unlabeled.texts)
        logger.info("Selector.run: encoding done in %.2fs, embeddings shape=%s", time.time() - t0, getattr(X, 'shape', None))
        id_arr = np.array(unlabeled.ids)
        iter_count = 0
        # safety cap to avoid infinite loops
        max_iters = max(1000, int(self.budget * 10))
        while len(picked) < self.budget and iter_count < max_iters:
            iter_count += 1
            mask = ~np.isin(id_arr, picked + list(labeled_ids))
            X_remain = X[mask]
            ids_remain = id_arr[mask]
            logger.info("Selector.run iter %d: remaining=%d picked=%d", iter_count, len(ids_remain), len(picked))
            if len(ids_remain) == 0:
                break
            # use the selector's strategy to choose indices from the remaining pool
            import time as _time
            t_sel = _time.time()
            try:
                order = self.select_indices(X_remain, picked, X, id_arr)
                order = np.asarray(order, dtype=int)
                logger.info("Selector.select_indices done in %.2fs (remain=%d)", _time.time() - t_sel, len(ids_remain))
            except Exception as _e:
                logger.exception("Selector.select_indices failed: %s", _e)
                # fallback: pick first up-to-batch_size items
                order = np.arange(len(ids_remain), dtype=int)
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
        logger.info('SurprisalEmbeddings: loaded model %s on %s', model_name, self.device)
        logger.info('BertEmbeddings: loaded model %s on %s', model_name, self.device)

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
        logger.info('BertKM.select_indices: X_remain.shape=%s k=%d', getattr(X_remain, 'shape', None), k)
        if getattr(self, 'force_fallback', False):
            logger.info('BertKM.select_indices: force_fallback enabled, using deterministic fallback')
            return np.arange(min(k, X_remain.shape[0]), dtype=int)
        # use MiniBatchKMeans for speed on CPU; fall back to KMeans if needed
        mb_batch = min(64, max(1, X_remain.shape[0] // 2))
        logger.info('BertKM.select_indices: attempting safe clustering n_clusters=%d batch_size=%d', k, mb_batch)
        centers = _safe_kmeans_fit(X_remain, k, mb_batch, use_mini=True, timeout=10.0)
        if centers is None:
            logger.info('BertKM.select_indices: MiniBatchKMeans timed out or failed, trying KMeans safely')
            centers = _safe_kmeans_fit(X_remain, k, mb_batch, use_mini=False, timeout=10.0)
        if centers is None:
            logger.warning('BertKM.select_indices: clustering failed; falling back to simple selection')
            # fallback: return the first min(k, remaining) indices
            return np.arange(min(k, X_remain.shape[0]), dtype=int)
        logger.info('BertKM.select_indices: clustering produced centers.shape=%s', getattr(centers, 'shape', None))
        nn_idx, _ = pairwise_distances_argmin_min(centers, X_remain)
        logger.info('BertKM.select_indices: returning %d indices', len(nn_idx))
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
        import logging, time
        logger = logging.getLogger(__name__)
        self.model.eval()
        vecs = []
        total = len(texts)
        logger.info('SurprisalEmbeddings: encoding %d texts in batches of %d', total, self.batch_size)
        t0 = time.time()
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i: i + self.batch_size]
            batch = self._prep_batch(chunk)
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            loss = self._get_token_losses(input_ids, attn)
            for b in range(loss.size(0)):
                vecs.append(self._loss_histogram(loss[b]))
            logger.info('SurprisalEmbeddings: processed %d/%d examples (%.2fs elapsed)', min(i + self.batch_size, total), total, time.time() - t0)
        X = np.stack(vecs, axis=0) if len(vecs) > 0 else np.zeros((0, self.hist_bins), dtype=np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8) if X.size > 0 else X
        logger.info('SurprisalEmbeddings: finished encoding in %.2fs', time.time() - t0)
        return X


class ALPS(Selector):
    """ALPS采样器"""
    def select_indices(self, X_remain, picked_ids, all_X, all_ids):
        # fast-path: when batch_size <= 1, no clustering needed
        if self.batch_size <= 1:
            logger.info('ALPS.select_indices: batch_size <=1, fast-path returning first index')
            return np.array([0], dtype=int)
        if getattr(self, 'force_fallback', False):
            logger.info('ALPS.select_indices: force_fallback enabled, using deterministic fallback')
            nn_idx = np.arange(min(self.batch_size, len(X_remain)), dtype=int)
            # pad if needed
            if len(nn_idx) < self.batch_size:
                pool = np.delete(np.arange(len(X_remain)), nn_idx)
                m = self.batch_size - len(nn_idx)
                if len(pool) > 0:
                    p = np.random.choice(len(pool), min(m, len(pool)), replace=False)
                    nn_idx = np.concatenate((nn_idx, pool[p]), axis=None)
            return nn_idx
        # use MiniBatchKMeans for ALPS as well for performance
        logger.info('ALPS.select_indices: X_remain.shape=%s batch_size=%d', getattr(X_remain, 'shape', None), self.batch_size)
        mb_batch = min(64, max(1, X_remain.shape[0] // 2))
        logger.info('ALPS.select_indices: attempting safe clustering n_clusters=%d batch_size=%d', self.batch_size, mb_batch)
        centers = _safe_kmeans_fit(X_remain, self.batch_size, mb_batch, use_mini=True, timeout=10.0)
        if centers is None:
            logger.info('ALPS.select_indices: MiniBatchKMeans timed out or failed, trying KMeans safely')
            centers = _safe_kmeans_fit(X_remain, self.batch_size, mb_batch, use_mini=False, timeout=10.0)
        if centers is None:
            logger.warning('ALPS.select_indices: clustering failed; falling back to simple selection')
            # fallback: choose first batch_size indices
            nn_idx = np.arange(min(self.batch_size, len(X_remain)), dtype=int)
            centroids_set = np.unique(nn_idx)
            m = self.batch_size - len(centroids_set)
            if m > 0 and len(X_remain) > len(centroids_set):
                pool = np.delete(np.arange(len(X_remain)), centroids_set)
                p = np.random.choice(len(pool), m, replace=False)
                nn_idx = np.concatenate((centroids_set, pool[p]), axis=None)
            return nn_idx
        logger.info('ALPS.select_indices: clustering produced centers.shape=%s', getattr(centers, 'shape', None))
        nn_idx, _ = pairwise_distances_argmin_min(centers, X_remain)
        centroids_set = np.unique(nn_idx)
        m = self.batch_size - len(centroids_set)
        if m > 0:
            pool = np.delete(np.arange(len(X_remain)), centroids_set)
            p = np.random.choice(len(pool), m, replace=False)
            nn_idx = np.concatenate((centroids_set, pool[p]), axis=None)
        return nn_idx
