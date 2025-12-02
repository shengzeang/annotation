from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseFilter(ABC):
    """Abstract base class for filters."""

    @abstractmethod
    def filter(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the filter to the dataset.

        Args:
            dataset (List[Dict[str, Any]]): The input dataset.

        Returns:
            List[Dict[str, Any]]: The filtered dataset.
        """
        pass
