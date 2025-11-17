"""
Typed structures shared across the Godot graph indexing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


NodeKind = Literal[
    "script",
    "scene",
    "resource",
    "scene_node",
    "symbol",
    "input_action",
    "autoload",
    "test_case",
    "runtime_event",
]

EdgeKind = Literal[
    "extends",
    "preload",
    "load",
    "instantiates",
    "attached_script",
    "uses_resource",
    "connects_signal",
    "defines_signal",
    "emits_signal",
    "exports_property",
    "group_member",
    "collision_layer",
    "collision_mask",
    "input_action",
    "autoload",
    "test_covers",
    "runtime_trace",
]


@dataclass
class GraphArtifact:
    """Normalized node/vertex stored per file or sub-entity."""

    file_path: str
    artifact_type: NodeKind
    name: str
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None
    last_modified: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def to_properties(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata or {}
        data["tags"] = self.tags or []
        return data


@dataclass
class GraphEdge:
    """Structured relationship between artifacts."""

    source_file: str
    target_file: str
    relationship_type: EdgeKind
    weight: float = 1.0
    source_symbol: Optional[str] = None
    target_symbol: Optional[str] = None
    signal_name: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None

    def to_properties(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSummary:
    """Aggregated insights computed while building the graph."""

    total_artifacts: int = 0
    total_edges: int = 0
    scene_files: int = 0
    script_files: int = 0
    resource_files: int = 0
    signal_connections: int = 0
    autoloads: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphPayload:
    """Complete payload emitted by the builder for persistence."""

    artifacts: List[GraphArtifact] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    summary: GraphSummary = field(default_factory=GraphSummary)

    def extend(self, other: "GraphPayload") -> None:
        self.artifacts.extend(other.artifacts)
        self.edges.extend(other.edges)
        self.summary.total_artifacts += other.summary.total_artifacts
        self.summary.total_edges += other.summary.total_edges
        self.summary.scene_files += other.summary.scene_files
        self.summary.script_files += other.summary.script_files
        self.summary.resource_files += other.summary.resource_files
        self.summary.signal_connections += other.summary.signal_connections
        self.summary.autoloads = list(
            sorted(set(self.summary.autoloads + other.summary.autoloads))
        )



