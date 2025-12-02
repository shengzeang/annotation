from typing import List, Dict, Any
from base_structure.base_filter import BaseFilter
from base_structure.active_learning import DataPool, BertEmbeddings, BertKM, SurprisalEmbeddings, ALPS

class ActiveLearningFilter(BaseFilter):
    """
    Active Learning filter implementation.
    """
    def __init__(self, method="bertkm", budget=100, batch_size=20, model_name="bert-base-uncased"):
        self.method = method.lower()
        self.budget = budget
        self.batch_size = batch_size
        self.model_name = model_name

        if self.method == "bertkm":
            self.emb = BertEmbeddings(model_name=self.model_name)
            self.selector = BertKM(self.emb, budget=self.budget, batch_size=self.batch_size)
        elif self.method == "alps":
            self.emb = SurprisalEmbeddings(model_name=self.model_name, batch_size=self.batch_size)
            self.selector = ALPS(self.emb, budget=self.budget, batch_size=self.batch_size)
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
        texts = [d["text"] if "text" in d else f"Q: {d['question']}\nContext: {d['context']}" for d in raw_dataset]
        ids = [str(d.get("id", i)) for i, d in enumerate(raw_dataset)]
        pool = DataPool(texts, ids)
        picked_ids = set(self.selector.run(pool))
        picked_data = [d for i, d in enumerate(raw_dataset) if str(d.get("id", i)) in picked_ids]
        return picked_data