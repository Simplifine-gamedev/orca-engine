"""
Lightweight Godot-aware parsers that extract structural signals for graphs.

The goal is not to implement a full AST but to capture enough structure to
produce reliable edges and metadata for search/contextualization.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


AUTLOAD_SECTION = "autoload"
INPUT_SECTION = "input"


@dataclass
class ParsedScript:
    file_path: str
    class_name: Optional[str]
    extends: Optional[str]
    functions: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    preloads: List[str] = field(default_factory=list)
    loads: List[str] = field(default_factory=list)
    signal_connections: List[Dict[str, str]] = field(default_factory=list)
    emitted_signals: List[str] = field(default_factory=list)
    node_accesses: List[str] = field(default_factory=list)
    signal_handlers: List[str] = field(default_factory=list)


@dataclass
class ParsedScene:
    file_path: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, str]] = field(default_factory=list)
    ext_resources: Dict[str, Dict[str, str]] = field(default_factory=dict)
    attached_scripts: List[Dict[str, str]] = field(default_factory=list)
    node_paths: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    node_scripts: Dict[str, str] = field(default_factory=dict)
    node_instances: Dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedProjectSettings:
    autoloads: Dict[str, str] = field(default_factory=dict)
    input_actions: Dict[str, Dict] = field(default_factory=dict)


class ScriptParser:
    RE_CLASS = re.compile(r"class\s+([A-Za-z_][\w]*)")
    RE_EXTENDS_STR = re.compile(r'extends\s+["\']([^"\']+)["\']')
    RE_EXTENDS_TYPE = re.compile(r"extends\s+([A-Za-z_][\w]*)")
    RE_FUNC = re.compile(r"func\s+([A-Za-z_][\w]*)\s*\(")
    RE_SIGNAL = re.compile(r"signal\s+([A-Za-z_][\w]*)")
    RE_EXPORT = re.compile(r"@export\s+(?:var|multiline)\s+([A-Za-z_][\w]*)")
    RE_PRELOAD = re.compile(r'preload\(["\']([^"\']+)["\']\)')
    RE_LOAD = re.compile(r'load\(["\']([^"\']+)["\']\)')
    RE_CONNECT = re.compile(
        r'\.connect\(\s*["\']([^"\']+)["\'],\s*([^) ,]+)(?:,\s*["\']([^"\']+)["\'])?'
    )
    RE_EMIT = re.compile(r'emit_signal\(\s*["\']([^"\']+)["\']')
    RE_DOLLAR = re.compile(r'\$([A-Za-z0-9_\/]+)')
    RE_GET_NODE = re.compile(r'get_node\(\s*["\']([^"\']+)["\']\)')

    def parse(self, file_path: str, content: str) -> ParsedScript:
        class_name = self._match_first(self.RE_CLASS, content)
        extends_match = self._match_first(self.RE_EXTENDS_STR, content)
        if not extends_match:
            extends_match = self._match_first(self.RE_EXTENDS_TYPE, content)
        functions = self.RE_FUNC.findall(content)
        signals = self.RE_SIGNAL.findall(content)
        exports = self.RE_EXPORT.findall(content)
        preloads = [self._normalize_path(path) for path in self.RE_PRELOAD.findall(content)]
        loads = [self._normalize_path(path) for path in self.RE_LOAD.findall(content)]

        connections = []
        for match in self.RE_CONNECT.findall(content):
            signal_name, target, method = match
            connections.append(
                {
                    "signal": signal_name.strip(),
                    "target": target.strip(),
                    "method": (method or "").strip(),
                }
            )

        emitted = list(dict.fromkeys(self.RE_EMIT.findall(content)))
        node_accesses = list(
            dict.fromkeys(self.RE_DOLLAR.findall(content) + self.RE_GET_NODE.findall(content))
        )
        signal_handlers = [fn for fn in functions if fn.startswith("_on_")]

        return ParsedScript(
            file_path=file_path,
            class_name=class_name,
            extends=extends_match,
            functions=functions,
            signals=signals,
            exports=exports,
            preloads=preloads,
            loads=loads,
            signal_connections=connections,
            emitted_signals=emitted,
            node_accesses=node_accesses,
            signal_handlers=signal_handlers,
        )

    @staticmethod
    def _match_first(pattern: re.Pattern, text: str) -> Optional[str]:
        match = pattern.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _normalize_path(path: str) -> str:
        if path.startswith("res://"):
            return path[6:]
        return path


class SceneParser:
    ATTR_PAIR_RE = re.compile(
        r'(\w+)=("([^"]*)"|ExtResource\("([^"]*)"\)|NodePath\("([^"]*)"\)|[^ \]]+)'
    )
    CONNECTION_RE = re.compile(r"\[connection\s+(.+?)\]")
    EXT_RESOURCE_RE = re.compile(r"\[ext_resource\s+(.+?)\]")
    NODE_HEADER_RE = re.compile(r"\[node\s+(.+?)\]")
    EXT_RESOURCE_VALUE_RE = re.compile(r'ExtResource\("([^"]+)"\)')
    ARRAY_VALUE_RE = re.compile(r'"([^"]+)"')

    def parse(self, file_path: str, content: str) -> ParsedScene:
        scene = ParsedScene(file_path=file_path)
        current_node: Optional[Dict[str, Any]] = None

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("[ext_resource"):
                attrs = self._parse_attributes(line)
                ext_id = attrs.get("id")
                if ext_id:
                    scene.ext_resources[ext_id] = {
                        "path": self._normalize(attrs.get("path", "")),
                        "type": attrs.get("type", ""),
                    }
                current_node = None
                continue

            if line.startswith("[node "):
                attrs = self._parse_attributes(line)
                name = attrs.get("name", "")
                node_type = attrs.get("type", "")
                parent = attrs.get("parent")
                if parent is None:
                    parent = ""
                node_path = self._compose_node_path(parent, name)
                node_info: Dict[str, Any] = {
                    "name": name,
                    "type": node_type,
                    "parent": parent,
                    "path": node_path,
                    "groups": [],
                }
                scene.nodes.append(node_info)
                scene.node_paths[node_path] = node_info

                instance_id = attrs.get("instance")
                if instance_id:
                    instance_path = self._resolve_resource(scene.ext_resources, instance_id)
                    if instance_path:
                        node_info["instance_path"] = instance_path
                        scene.node_instances[node_path] = instance_path

                script_header = attrs.get("script")
                if script_header:
                    script_path = self._resolve_resource(scene.ext_resources, script_header)
                    if script_path:
                        node_info["script_path"] = script_path
                        scene.node_scripts[node_path] = script_path
                        scene.attached_scripts.append(
                            {
                                "node": node_info["name"],
                                "node_path": node_path,
                                "resource_id": script_header,
                                "script_path": script_path,
                            }
                        )

                current_node = node_info
                continue

            if line.startswith("[connection "):
                attrs = self._parse_attributes(line)
                connection = {
                    "signal": attrs.get("signal"),
                    "from": attrs.get("from"),
                    "to": attrs.get("to"),
                    "method": attrs.get("method"),
                    "from_path": attrs.get("from"),
                    "to_path": attrs.get("to"),
                }
                scene.connections.append(connection)
                current_node = None
                continue

            if line.startswith("["):
                current_node = None
                continue

            if current_node:
                self._apply_node_property(current_node, line, scene)

        return scene

    def _parse_attributes(self, block_line: str) -> Dict[str, str]:
        attrs: Dict[str, str] = {}
        match = self.NODE_HEADER_RE.match(block_line)
        if not match:
            match = self.CONNECTION_RE.match(block_line) or self.EXT_RESOURCE_RE.match(block_line)
        payload = match.group(1) if match else ""
        for attr in self.ATTR_PAIR_RE.finditer(payload):
            key = attr.group(1)
            if attr.group(4):
                attrs[key] = attr.group(4)
            elif attr.group(5):
                attrs[key] = attr.group(5)
            elif attr.group(3) is not None:
                attrs[key] = attr.group(3)
            else:
                value = attr.group(2).strip('"')
                attrs[key] = value
        return attrs

    def _apply_node_property(self, node_info: Dict[str, Any], line: str, scene: ParsedScene) -> None:
        if "=" not in line:
            return
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        raw_value = raw_value.strip()

        if key == "groups":
            node_info["groups"] = self.ARRAY_VALUE_RE.findall(raw_value)
            return

        if key in {"collision_layer", "collision_mask"}:
            try:
                node_info[key] = int(raw_value)
            except ValueError:
                node_info[key] = raw_value
            return

        if key in {"monitoring", "monitorable"}:
            node_info[key] = raw_value.lower() == "true"
            return

        if key == "script":
            resource_id = self._extract_ext_id(raw_value)
            if resource_id:
                script_path = self._resolve_resource(scene.ext_resources, resource_id)
                if script_path and node_info.get("script_path") != script_path:
                    node_info["script_path"] = script_path
                    scene.node_scripts[node_info["path"]] = script_path
                    scene.attached_scripts.append(
                        {
                            "node": node_info["name"],
                            "node_path": node_info["path"],
                            "resource_id": resource_id,
                            "script_path": script_path,
                        }
                    )
            return

        if key == "instance":
            resource_id = self._extract_ext_id(raw_value)
            if resource_id:
                instance_path = self._resolve_resource(scene.ext_resources, resource_id)
                if instance_path and node_info.get("instance_path") != instance_path:
                    node_info["instance_path"] = instance_path
                    scene.node_instances[node_info["path"]] = instance_path
            return

        node_info.setdefault("properties", {})[key] = raw_value

    @staticmethod
    def _extract_ext_id(value: str) -> Optional[str]:
        match = SceneParser.EXT_RESOURCE_VALUE_RE.search(value)
        return match.group(1) if match else None

    @staticmethod
    def _compose_node_path(parent: str, name: str) -> str:
        if parent == "" or parent is None:
            return "."
        if parent == ".":
            return name
        return f"{parent.rstrip('/')}/{name}"

    @staticmethod
    def _normalize(path: str) -> str:
        if path.startswith("res://"):
            return path[6:]
        return path

    def _resolve_resource(self, resources: Dict[str, Dict[str, str]], resource_id: str) -> str:
        info = resources.get(resource_id)
        if not info:
            return ""
        return info.get("path", "")


class ProjectSettingsParser:
    SECTION_PATTERN = re.compile(r"\[([^\]]+)\]")
    KEY_VALUE_PATTERN = re.compile(r'([^=]+)=\s*(.+)')

    def parse(self, content: str) -> ParsedProjectSettings:
        parsed = ParsedProjectSettings()
        current_section: Optional[str] = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                current_section = section_match.group(1)
                continue
            if current_section is None:
                continue
            kv_match = self.KEY_VALUE_PATTERN.match(line)
            if not kv_match:
                continue
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            if current_section.startswith(AUTLOAD_SECTION):
                parsed.autoloads[key] = value.strip('"')
            elif current_section.startswith(INPUT_SECTION):
                parsed.input_actions[key] = {"raw": value}
        return parsed


def iter_godot_files(files: Iterable[Dict]) -> Iterable[Dict]:
    """Normalize incoming file data structures from various call sites."""
    for file_data in files:
        if not file_data:
            continue
        file_path = file_data.get("path") or file_data.get("file_path")
        if not file_path:
            continue
        content = file_data.get("content")
        if content is None:
            continue
        normalized_path = file_path
        if normalized_path.startswith("res://"):
            normalized_path = normalized_path[6:]
        yield {
            "file_path": normalized_path.replace("\\", "/"),
            "content": content,
            "hash": file_data.get("hash"),
            "metadata": file_data.get("metadata") or {},
        }


def classify_extension(file_path: str) -> str:
    _, ext = os.path.splitext(file_path.lower())
    return ext



