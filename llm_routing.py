"""
Facade module for routers. This file keeps the old import path for backward compatibility
and re-exports the router classes and helpers defined in separate files.
"""
from routers.llm_router import LLMRouter
from routers.mlp_router import MLPRouter, synthesize_pairs_from_annotations, train_mlprouter_from_annotations
from routers.knn_router import KNNRouter, build_knn_from_annotations
from routers.graph_router import GraphRouter, build_graph_from_annotations

__all__ = [
    "LLMRouter",
    "MLPRouter",
    "KNNRouter",
    "GraphRouter",
    "train_mlprouter_from_annotations",
    "build_knn_from_annotations",
    "build_graph_from_annotations",
]

