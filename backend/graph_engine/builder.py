"""
GraphBuilder orchestrates parsing files into artifact/edge payloads.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .parsers import (
    ScriptParser,
    SceneParser,
    ProjectSettingsParser,
    classify_extension,
    iter_godot_files,
)
from .schema import GraphArtifact, GraphEdge, GraphPayload, GraphSummary


class GraphBuilder:
    """Builds a Godot-aware graph payload from raw file contents."""

    TEXT_EXTENSIONS = {".gd", ".tscn", ".scn", ".tres", ".res", ".gdshader", ".glsl", ".cfg"}

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root
        self.script_parser = ScriptParser()
        self.scene_parser = SceneParser()
        self.settings_parser = ProjectSettingsParser()

    def build(self, files: Iterable[Dict]) -> GraphPayload:
        payload = GraphPayload(summary=GraphSummary())
        normalized_files = list(iter_godot_files(files))
        autoloads: Dict[str, str] = {}

        for file_data in normalized_files:
            file_path = file_data["file_path"]
            content = file_data["content"]
            metadata = file_data.get("metadata") or {}
            ext = classify_extension(file_path)
            checksum = self._compute_hash(content)
            size_bytes = len(content.encode("utf-8")) if isinstance(content, str) else len(content)
            last_modified = metadata.get("last_modified") or time.time()

            artifact_metadata = {
                "checksum": checksum,
                "structure": {},
            }

            if ext in {".gd"}:
                parsed = self.script_parser.parse(file_path, content)
                artifact_metadata["structure"] = {
                    "class_name": parsed.class_name,
                    "extends": parsed.extends,
                    "functions": parsed.functions,
                    "signals": parsed.signals,
                    "exports": parsed.exports,
                    "emits": parsed.emitted_signals,
                    "node_accesses": parsed.node_accesses,
                    "signal_handlers": parsed.signal_handlers,
                }
                payload.summary.script_files += 1
                summary = self._summarize_script(parsed)
                artifact = GraphArtifact(
                    file_path=file_path,
                    artifact_type="script",
                    name=parsed.class_name or os.path.basename(file_path),
                    summary=summary,
                    metadata=artifact_metadata,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    last_modified=last_modified,
                )
                payload.artifacts.append(artifact)
                payload.summary.total_artifacts += 1
                payload.edges.extend(self._edges_from_script(parsed))

            elif ext in {".tscn", ".scn"}:
                parsed = self.scene_parser.parse(file_path, content)
                artifact_metadata["structure"] = {
                    "nodes": parsed.nodes,
                    "connections": parsed.connections,
                }
                payload.summary.scene_files += 1
                summary = self._summarize_scene(parsed)
                artifact = GraphArtifact(
                    file_path=file_path,
                    artifact_type="scene",
                    name=os.path.basename(file_path),
                    summary=summary,
                    metadata=artifact_metadata,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    last_modified=last_modified,
                )
                payload.artifacts.append(artifact)
                payload.summary.total_artifacts += 1
                payload.edges.extend(self._edges_from_scene(parsed))
                payload.summary.signal_connections += len(parsed.connections)

            elif ext in {".tres", ".res"}:
                artifact_metadata["structure"] = {"type_hint": metadata.get("type")}
                payload.summary.resource_files += 1
                artifact = GraphArtifact(
                    file_path=file_path,
                    artifact_type="resource",
                    name=os.path.basename(file_path),
                    summary=f"Resource: {metadata.get('type', 'unknown')}",
                    metadata=artifact_metadata,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    last_modified=last_modified,
                )
                payload.artifacts.append(artifact)
                payload.summary.total_artifacts += 1

            elif os.path.basename(file_path) == "project.godot":
                parsed_settings = self.settings_parser.parse(content)
                autoloads.update(parsed_settings.autoloads)
                if parsed_settings.autoloads:
                    payload.artifacts.append(
                        GraphArtifact(
                            file_path=file_path,
                            artifact_type="autoload",
                            name="project_autoloads",
                            summary="Project autoload singletons",
                            metadata={"autoloads": parsed_settings.autoloads},
                        )
                    )
                    payload.summary.total_artifacts += 1
                if parsed_settings.input_actions:
                    payload.artifacts.append(
                        GraphArtifact(
                            file_path=file_path,
                            artifact_type="input_action",
                            name="input_actions",
                            summary="Input map",
                            metadata={"input_map": parsed_settings.input_actions},
                        )
                    )
                    payload.summary.total_artifacts += 1

        if autoloads:
            payload.summary.autoloads = list(sorted(autoloads.keys()))

        payload.summary.total_edges = len(payload.edges)
        return payload

    def _edges_from_script(self, parsed: "ScriptParser") -> List[GraphEdge]:
        edges: List[GraphEdge] = []
        for ref in parsed.preloads:
            edges.append(
                GraphEdge(
                    source_file=parsed.file_path,
                    target_file=ref,
                    relationship_type="preload",
                    weight=1.0,
                )
            )
        for ref in parsed.loads:
            edges.append(
                GraphEdge(
                    source_file=parsed.file_path,
                    target_file=ref,
                    relationship_type="load",
                    weight=0.8,
                )
            )
        if parsed.extends:
            edges.append(
                GraphEdge(
                    source_file=parsed.file_path,
                    target_file=parsed.extends,
                    relationship_type="extends",
                    weight=1.2,
                )
            )
        for conn in parsed.signal_connections:
            edges.append(
                GraphEdge(
                    source_file=parsed.file_path,
                    target_file=conn.get("target", ""),
                    relationship_type="connects_signal",
                    signal_name=conn.get("signal"),
                    weight=1.3,
                )
            )
        if parsed.signals:
            for signal in parsed.signals:
                edges.append(
                    GraphEdge(
                        source_file=parsed.file_path,
                        target_file=parsed.file_path,
                        relationship_type="defines_signal",
                        signal_name=signal,
                        weight=0.7,
                    )
                )
        if parsed.emitted_signals:
            for signal in parsed.emitted_signals:
                edges.append(
                    GraphEdge(
                        source_file=parsed.file_path,
                        target_file=parsed.file_path,
                        relationship_type="emits_signal",
                        signal_name=signal,
                        weight=0.7,
                    )
                )
        return edges

    def _edges_from_scene(self, parsed: "SceneParser") -> List[GraphEdge]:
        edges: List[GraphEdge] = []
        for attached in parsed.attached_scripts:
            script_path = attached.get("script_path")
            if script_path:
                edges.append(
                    GraphEdge(
                        source_file=parsed.file_path,
                        target_file=script_path,
                        relationship_type="attached_script",
                        weight=1.0,
                        context=attached.get("node_path") or attached.get("node"),
                    )
                )
        for ext in parsed.ext_resources.values():
            target = ext.get("path")
            if target:
                edges.append(
                    GraphEdge(
                        source_file=parsed.file_path,
                        target_file=target,
                        relationship_type="uses_resource",
                        weight=0.8,
                        context=ext.get("type"),
                    )
                )
        for connection in parsed.connections:
            edges.append(
                GraphEdge(
                    source_file=parsed.file_path,
                    target_file=parsed.file_path,
                    relationship_type="connects_signal",
                    signal_name=connection.get("signal"),
                    source_symbol=connection.get("from_path") or connection.get("from"),
                    target_symbol=connection.get("to_path") or connection.get("to"),
                    context=connection.get("method"),
                    weight=1.1,
                )
            )
            src_artifact = self._resolve_node_artifact(parsed, connection.get("from_path"))
            dst_artifact = self._resolve_node_artifact(parsed, connection.get("to_path"))
            if src_artifact and dst_artifact:
                edges.append(
                    GraphEdge(
                        source_file=src_artifact,
                        target_file=dst_artifact,
                        relationship_type="connects_signal",
                        signal_name=connection.get("signal"),
                        source_symbol=connection.get("from_path"),
                        target_symbol=connection.get("to_path"),
                        context=connection.get("method"),
                        weight=1.4,
                    )
                )
        for node_path, instance_path in parsed.node_instances.items():
            if instance_path:
                edges.append(
                    GraphEdge(
                        source_file=parsed.file_path,
                        target_file=instance_path,
                        relationship_type="instantiates",
                        weight=1.05,
                        context=node_path,
                    )
                )
        for node in parsed.nodes:
            groups = node.get("groups") or []
            node_path = node.get("path") or node.get("name")
            for group_name in groups:
                target = f"group::{group_name}"
                edges.append(
                    GraphEdge(
                        source_file=parsed.file_path,
                        target_file=target,
                        relationship_type="group_member",
                        weight=0.4,
                        context=node_path,
                    )
                )
            properties = node.get("properties") or {}
            for prop_key in ("collision_layer", "collision_mask"):
                if prop_key in properties:
                    target = f"{prop_key}::{properties.get(prop_key)}"
                    relationship_type = "collision_layer" if prop_key == "collision_layer" else "collision_mask"
                    edges.append(
                        GraphEdge(
                            source_file=parsed.file_path,
                            target_file=target,
                            relationship_type=relationship_type,
                            weight=0.3,
                            context=node_path,
                        )
                    )
        return edges

    @staticmethod
    def _resolve_node_artifact(parsed_scene: "SceneParser", node_path: Optional[str]) -> Optional[str]:
        if not node_path:
            return None
        if node_path in parsed_scene.node_scripts:
            return parsed_scene.node_scripts[node_path]
        if node_path in parsed_scene.node_instances:
            return parsed_scene.node_instances[node_path]
        if node_path == "." and parsed_scene.node_scripts.get("."):
            return parsed_scene.node_scripts["."]
        return None

    @staticmethod
    def _summarize_script(parsed) -> str:
        parts = []
        if parsed.class_name:
            parts.append(f"class {parsed.class_name}")
        if parsed.extends:
            parts.append(f"extends {parsed.extends}")
        if parsed.exports:
            parts.append(f"exports: {', '.join(parsed.exports)}")
        if parsed.signals:
            parts.append(f"signals: {', '.join(parsed.signals)}")
        return "; ".join(parts) or "GDScript file"

    @staticmethod
    def _summarize_scene(parsed) -> str:
        node_count = len(parsed.nodes)
        connection_count = len(parsed.connections)
        return f"Scene with {node_count} nodes, {connection_count} signal connections"

    @staticmethod
    def _compute_hash(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest() if isinstance(content, str) else ""



