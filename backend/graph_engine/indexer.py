"""
High-level coordinator for feeding graph payloads into vector managers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from .builder import GraphBuilder
from .schema import GraphPayload


class GodotGraphIndexer:
    """Coordinates graph building and persistence via the configured vector manager."""

    def __init__(self, vector_manager, project_root: Optional[str] = None):
        self.vector_manager = vector_manager
        self.builder = GraphBuilder(project_root=project_root)

    def ingest(self, files: Iterable[Dict], user_id: str, project_id: str) -> GraphPayload:
        """Build graph payload from files and push it to the vector store."""
        payload = self.builder.build(files)
        sync_method = getattr(self.vector_manager, "sync_graph_payload", None)
        if callable(sync_method) and payload.artifacts:
            try:
                sync_method(payload, user_id=user_id, project_id=project_id)
            except Exception as exc:
                print(f"GraphIndexer: Failed to sync graph payload: {exc}")
        return payload






