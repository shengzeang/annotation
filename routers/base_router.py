from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Abstract router interface for scoring candidate LLMs."""
    @abstractmethod
    def score(self, sample: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        """Return list of {'model': name, 'score': float} for the candidates."""
        raise NotImplementedError()

    def choose_best(self, sample: str, candidate_llms: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
        scored = self.score(sample, candidate_llms)
        best = max(scored, key=lambda x: x.get('score', 0.0))
        return best['model'], scored
