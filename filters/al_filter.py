from typing import List, Dict, Any
import time
import logging

from base_structure.base_filter import BaseFilter
from base_structure.active_learning import DataPool, BertEmbeddings, BertKM, SurprisalEmbeddings, ALPS

logger = logging.getLogger(__name__)

class ActiveLearningFilter(BaseFilter):
    """
    Active Learning filter implementation.
    """
    def __init__(self, method="alps", budget=100, batch_size=10, model_name="bert-base-uncased", force_fallback=True):
        self.method = method.lower()
        self.budget = budget
        self.batch_size = batch_size
        self.model_name = model_name
        self.force_fallback = force_fallback

        if self.method == "bertkm":
            self.emb = BertEmbeddings(model_name=self.model_name)
            self.selector = BertKM(self.emb, budget=self.budget, batch_size=self.batch_size, force_fallback=self.force_fallback)
        elif self.method == "alps":
            self.emb = SurprisalEmbeddings(model_name=self.model_name, batch_size=self.batch_size)
            self.selector = ALPS(self.emb, budget=self.budget, batch_size=self.batch_size, force_fallback=self.force_fallback)
        else:
            raise ValueError(f"Unknown active learning method: {self.method}")

    def filter(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the active learning filter to the dataset.

        Args:
            raw_dataset (List[Dict[str, Any]]): The input dataset.

        Returns:
            List[Dict[str, Any]]: The filtered dataset.
        """
        start = time.time()
        # also log to root logger to ensure visibility in different logging configs
        logging.getLogger().info('ActiveLearningFilter.filter called (method=%s)', self.method)
        try:
            total = len(raw_dataset)
            logger.info('ActiveLearningFilter.start: method=%s budget=%s batch_size=%s samples=%d', self.method, self.budget, self.batch_size, total)
            # show a few sample ids/questions for context
            try:
                preview = [raw_dataset[i].get('id') or raw_dataset[i].get('question') for i in range(min(3, total))]
                logger.info('ActiveLearningFilter.preview: %s', preview)
            except Exception:
                pass

            texts = [d["text"] if "text" in d else f"Q: {d['question']}\nContext: {d['context']}" for d in raw_dataset]
            ids = [str(d.get("id", i)) for i, d in enumerate(raw_dataset)]
            pool = DataPool(texts, ids)
            logger.info('ActiveLearningFilter: DataPool created (texts=%d ids=%d)', len(pool.texts), len(pool.ids))
            # report initial progress (encoding done)
            try:
                if hasattr(self, 'progress_cb') and callable(getattr(self, 'progress_cb')):
                    try:
                        self.progress_cb(1, max(1, total), {'phase': 'start'})
                    except Exception:
                        pass
            except Exception:
                pass

            t0 = time.time()
            logger.info('ActiveLearningFilter: calling selector.run()')
            try:
                picked_ids = set(self.selector.run(pool))
            finally:
                logger.info('ActiveLearningFilter.selector.run finished (elapsed=%.2fs)', time.time() - t0)

            picked_data = [d for i, d in enumerate(raw_dataset) if str(d.get("id", i)) in picked_ids]
            elapsed = time.time() - start
            logger.info('ActiveLearningFilter.done: picked=%d elapsed=%.2fs', len(picked_data), elapsed)
            try:
                if hasattr(self, 'progress_cb') and callable(getattr(self, 'progress_cb')):
                    try:
                        self.progress_cb(len(picked_data), max(1, total), {'phase': 'done'})
                    except Exception:
                        pass
            except Exception:
                pass
            return picked_data
        except Exception as e:
            logger.exception('ActiveLearningFilter.filter failed: %s', e)
            return []