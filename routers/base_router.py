from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseRouter(ABC):
    """Abstract router interface for scoring candidate LLMs."""
    @property
    def if_train(self):
        """Whether this router requires training."""
        raise NotImplementedError()
    
    @abstractmethod
    def build_from_annotations(self, out_dir: str, **kwargs):
        """Build internal index from annotated data."""
        raise NotImplementedError()

    @abstractmethod
    def score(self, sample: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        """Return list of {'model': name, 'score': float} for the candidates."""
        raise NotImplementedError()

    def choose_best(self, sample: str, candidate_llms: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
        scored = self.score(sample, candidate_llms)
        best = max(scored, key=lambda x: x.get('score', 0.0))
        return best['model'], scored
    
    # Main function entry point
    def route(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """see if the router needs training, if so, train it"""
        if self.if_train and not self.ready:
            self.build_from_annotations(out_dir="./")
            self.ready = True

        routed = []
        for d in tqdm(dataset):
            # If a router is provided (e.g., MLPRouter), use it to score candidates
            scores = self.score(d.get('text', ''), self.candidate_llms)
            # scores -> list of {model, score}
            best = max(scores, key=lambda x: x.get('score', 0.0))
            chosen = best.get('model')
            routed.append({**d, 'route': chosen, 'route_scores': scores})
        return routed
