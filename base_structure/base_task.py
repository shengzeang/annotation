"""
Abstraction for annotation tasks.

Defines the base interface for annotation operations.
Includes abstract methods of generating LLMs prompts and parsing results
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class Task(ABC):
    """Abstract base class for annotation tasks"""
    @abstractmethod
    #Generate LLM input
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        """Generate prompt based on sample and optional RAG retrieval"""
        pass
    #Parse LLM output
    @abstractmethod
    def parse_output(self, output: str) -> Dict[str, Any]:
        """Parse LLM output and return structured annotation result"""
        pass