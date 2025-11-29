"""
Facade module for routers. This file keeps the old import path for backward compatibility
and re-exports the router classes and helpers defined in separate files.
"""
from routers.llm_router import LLMRouter
from routers.mlp_router import MLPRouter, build_mlprouter_from_annotations
from routers.knn_router import KNNRouter, build_knnrouter_from_annotations
from routers.graph_router import GraphRouter, build_graphrouter_from_annotations

__all__ = [
    "LLMRouter",
    "MLPRouter",
    "KNNRouter",
    "GraphRouter",
    "build_mlprouter_from_annotations",
    "build_knnrouter_from_annotations",
    "build_graphrouter_from_annotations",
]

