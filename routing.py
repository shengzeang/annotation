"""
Facade module for routers. This file keeps the old import path for backward compatibility
and re-exports the router classes and helpers defined in separate files.
"""
from routers.llm_router import LLMRouter
from routers.mlp_router import MLPRouter
from routers.knn_router import KNNRouter
from routers.graph_router import GraphRouter
from routers.cascade_router import CascadeRouter
from routers.routerdc_router import RouterDCRouter

__all__ = [
    "LLMRouter",
    "MLPRouter",
    "KNNRouter",
    "GraphRouter",
    "CascadeRouter",
    "RouterDCRouter",
]

