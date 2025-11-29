"""
Facade module for filters. This file keeps the old import path for backward compatibility
and re-exports the filter classes defined in separate files.
"""
from filters.al_filter import ActiveLearningFilter
from filters.llm_filter import LLMNaiveFilter

__all__ = [
    "ActiveLearningFilter",
    "LLMNaiveFilter",
]
