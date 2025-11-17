"""
Graph engine utilities for building Godot-aware project graphs.

This package encapsulates the parsing, normalization, and persistence logic
required to keep the vector database in sync with rich structural metadata.
"""

from .schema import GraphArtifact, GraphEdge, GraphSummary, GraphPayload
from .builder import GraphBuilder
from .indexer import GodotGraphIndexer

__all__ = [
    "GraphArtifact",
    "GraphEdge",
    "GraphSummary",
    "GraphPayload",
    "GraphBuilder",
    "GodotGraphIndexer",
]





