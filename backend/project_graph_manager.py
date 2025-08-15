#!/usr/bin/env python3
"""
Project Graph Manager for Godot projects
Creates and manages a graph structure of project relationships
"""

import json
import os
import re
import time
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx

@dataclass
class GraphNode:
    """Represents a node in the project graph"""
    file_path: str
    node_type: str  # 'scene', 'script', 'resource', 'image', 'audio', 'shader'
    modality: str   # 'text', 'image', 'audio'
    size: int
    last_modified: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class GraphEdge:
    """Represents a relationship between two files"""
    source: str
    target: str
    relationship_type: str  # 'script_attached', 'resource_used', 'scene_instanced', 'texture_used', etc.
    strength: float = 1.0   # Relationship strength (0-1)
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ProjectGraphManager:
    """Manages the graph structure of a Godot project"""
    
    def __init__(self, project_root: str, user_id: str):
        self.project_root = project_root
        self.user_id = user_id
        self.graph = nx.DiGraph()  # Directed graph for file relationships
        
        # Relationship patterns for different file types
        self.relationship_patterns = {
            # Scene files (.tscn)
            'scene': {
                'script_attached': r'script = ExtResource\(\s*"?([^")]+)"?\s*\)',
                'scene_instanced': r'instance = ExtResource\(\s*"?([^")]+)"?\s*\)',
                'texture_used': r'texture = ExtResource\(\s*"?([^")]+)"?\s*\)',
                'material_used': r'material = ExtResource\(\s*"?([^")]+)"?\s*\)',
                'sound_used': r'stream = ExtResource\(\s*"?([^")]+)"?\s*\)',
                'external_resource': r'\[ext_resource[^]]*path="([^"]+)"[^]]*type="([^"]+)"',
                'external_resource_id': r'ExtResource\(\s*"?([^")]+)"?\s*\)'
            },
            
            # Script files (.gd, .cs)
            'script': {
                'class_extends': r'extends\s+([A-Za-z_][A-Za-z0-9_]*)',
                'preload_resource': r'preload\(\s*["\']([^"\']+)["\']\s*\)',
                'load_resource': r'load\(\s*["\']([^"\']+)["\']\s*\)',
                'get_node': r'get_node\(\s*["\']([^"\']+)["\']\s*\)',
                'scene_change': r'get_tree\(\)\.change_scene.*["\']([^"\']+)["\']',
                'signal_connect': r'connect\(\s*["\']([^"\']+)["\']\s*,\s*([^,)]+)'
            },
            
            # Resource files (.tres, .res)
            'resource': {
                'script_reference': r'script = ExtResource\(\s*"([^"]+)"\s*\)',
                'texture_reference': r'texture = ExtResource\(\s*"([^"]+)"\s*\)',
                'external_resource': r'path="([^"]+)".*type="([^"]+)"'
            }
        }
    
    def add_node(self, file_path: str, node_type: str, modality: str, size: int, last_modified: float, metadata: Dict = None):
        """Add a node to the graph"""
        node = GraphNode(
            file_path=file_path,
            node_type=node_type,
            modality=modality,
            size=size,
            last_modified=last_modified,
            metadata=metadata or {}
        )
        
        self.graph.add_node(file_path, **asdict(node))
    
    def add_edge(self, source: str, target: str, relationship_type: str, strength: float = 1.0, metadata: Dict = None):
        """Add an edge (relationship) to the graph"""
        edge = GraphEdge(
            source=source,
            target=target,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata or {}
        )
        
        self.graph.add_edge(source, target, **asdict(edge))
    
    def _parse_scene_resources(self, content: str) -> Dict[str, str]:
        """Parse ExtResource IDs to file paths mapping from scene file"""
        resource_map = {}
        
        # Find all ext_resource definitions
        ext_resource_pattern = r'\[ext_resource[^]]*path="([^"]+)"[^]]*id="([^"]+)"'
        matches = re.finditer(ext_resource_pattern, content, re.MULTILINE)
        
        for match in matches:
            path = match.group(1)
            resource_id = match.group(2)
            
            # Convert res:// paths to relative paths
            if path.startswith('res://'):
                path = path[6:]  # Remove 'res://'
            
            resource_map[resource_id] = path
        
        return resource_map

    def analyze_file_relationships(self, file_path: str, content: str) -> List[Tuple[str, str, float]]:
        """Analyze a file's content to find relationships with other files"""
        relationships = []
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Determine file category
        if file_ext in ['.tscn', '.scn']:
            category = 'scene'
        elif file_ext in ['.gd', '.cs', '.js']:
            category = 'script'
        elif file_ext in ['.tres', '.res']:
            category = 'resource'
        else:
            return relationships
        
        patterns = self.relationship_patterns.get(category, {})
        
        # For scene files, first map ExtResource IDs to paths
        resource_map = {}
        if category == 'scene':
            resource_map = self._parse_scene_resources(content)
        
        for relationship_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                if relationship_type == 'external_resource':
                    # Handle external resource references
                    resource_path = match.group(1)
                    resource_type = match.group(2) if len(match.groups()) > 1 else 'unknown'
                    
                    # Convert relative path to absolute
                    if resource_path.startswith('res://'):
                        resource_path = resource_path[6:]  # Remove 'res://'
                    
                    if not os.path.isabs(resource_path):
                        resource_path = os.path.join(self.project_root, resource_path)
                    
                    relationships.append((resource_path, relationship_type, 0.8))
                    
                elif relationship_type == 'external_resource_id':
                    # Handle ExtResource ID references (for scenes)
                    resource_id = match.group(1)
                    if resource_id in resource_map:
                        resource_path = resource_map[resource_id]
                        if not os.path.isabs(resource_path):
                            resource_path = os.path.join(self.project_root, resource_path)
                        relationships.append((resource_path, 'external_reference', 0.7))
                    
                elif relationship_type == 'signal_connect':
                    # Handle signal connections (node relationships)
                    signal_name = match.group(1)
                    target_node = match.group(2) if len(match.groups()) > 1 else ''
                    relationships.append((target_node, f"signal_{signal_name}", 0.6))
                    
                else:
                    # Handle other relationships
                    target_path = match.group(1)
                    
                    # Convert res:// paths
                    if target_path.startswith('res://'):
                        target_path = target_path[6:]  # Remove 'res://'
                    
                    # Convert relative path to absolute if it looks like a file path
                    if ('/' in target_path or '.' in target_path) and not os.path.isabs(target_path):
                        target_path = os.path.join(self.project_root, target_path)
                    
                    # Determine strength based on relationship type
                    strength = {
                        'script_attached': 0.9,
                        'scene_instanced': 0.8,
                        'preload_resource': 0.9,
                        'load_resource': 0.7,
                        'texture_used': 0.6,
                        'class_extends': 0.8,
                        'get_node': 0.5,
                        'scene_change': 0.7
                    }.get(relationship_type, 0.5)
                    
                    relationships.append((target_path, relationship_type, strength))
        
        return relationships
    
    def analyze_node_structure(self, file_path: str, content: str) -> Dict:
        """Analyze the internal structure of a file (nodes, classes, functions)"""
        file_ext = os.path.splitext(file_path)[1].lower()
        structure = {}
        
        if file_ext == '.tscn':
            # Analyze scene structure
            structure['nodes'] = self._extract_scene_nodes(content)
            structure['signals'] = self._extract_scene_signals(content)
            
        elif file_ext == '.gd':
            # Analyze GDScript structure
            structure['classes'] = self._extract_gdscript_classes(content)
            structure['functions'] = self._extract_gdscript_functions(content)
            structure['signals'] = self._extract_gdscript_signals(content)
            structure['exports'] = self._extract_gdscript_exports(content)
            
        return structure
    
    def _extract_scene_nodes(self, content: str) -> List[Dict]:
        """Extract node information from a .tscn file"""
        nodes = []
        node_pattern = r'\[node name="([^"]+)".*type="([^"]+)"'
        
        for match in re.finditer(node_pattern, content):
            nodes.append({
                'name': match.group(1),
                'type': match.group(2)
            })
        
        return nodes
    
    def _extract_scene_signals(self, content: str) -> List[Dict]:
        """Extract signal connections from a .tscn file"""
        signals = []
        signal_pattern = r'\[connection signal="([^"]+)" from="([^"]+)" to="([^"]+)" method="([^"]+)"'
        
        for match in re.finditer(signal_pattern, content):
            signals.append({
                'signal': match.group(1),
                'from': match.group(2),
                'to': match.group(3),
                'method': match.group(4)
            })
        
        return signals
    
    def _extract_gdscript_classes(self, content: str) -> List[str]:
        """Extract class names from GDScript"""
        classes = []
        class_pattern = r'class\s+([A-Za-z_][A-Za-z0-9_]*)'
        
        for match in re.finditer(class_pattern, content):
            classes.append(match.group(1))
        
        return classes
    
    def _extract_gdscript_functions(self, content: str) -> List[str]:
        """Extract function names from GDScript"""
        functions = []
        func_pattern = r'func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
        
        for match in re.finditer(func_pattern, content):
            functions.append(match.group(1))
        
        return functions
    
    def _extract_gdscript_signals(self, content: str) -> List[str]:
        """Extract signal definitions from GDScript"""
        signals = []
        signal_pattern = r'signal\s+([A-Za-z_][A-Za-z0-9_]*)'
        
        for match in re.finditer(signal_pattern, content):
            signals.append(match.group(1))
        
        return signals
    
    def _extract_gdscript_exports(self, content: str) -> List[str]:
        """Extract exported variables from GDScript"""
        exports = []
        export_pattern = r'@export\s+var\s+([A-Za-z_][A-Za-z0-9_]*)'
        
        for match in re.finditer(export_pattern, content):
            exports.append(match.group(1))
        
        return exports
    
    def build_project_graph(self, file_contents: Dict[str, str], file_metadata: Dict[str, Dict]):
        """Build the complete project graph from file contents"""
        print(f"🌐 Building project graph with {len(file_contents)} files...")
        
        # First pass: Add all nodes
        for file_path, content in file_contents.items():
            metadata = file_metadata.get(file_path, {})
            
            # Determine node type and modality
            file_ext = os.path.splitext(file_path)[1].lower()
            node_type = self._get_node_type(file_ext)
            modality = self._get_modality(file_ext)
            
            # Analyze file structure
            structure = self.analyze_node_structure(file_path, content)
            metadata.update({'structure': structure})
            
            self.add_node(
                file_path=file_path,
                node_type=node_type,
                modality=modality,
                size=len(content.encode('utf-8')) if isinstance(content, str) else len(content),
                last_modified=metadata.get('last_modified', time.time()),
                metadata=metadata
            )
        
        # Second pass: Add relationships
        for file_path, content in file_contents.items():
            if isinstance(content, str):  # Only analyze text files for relationships
                relationships = self.analyze_file_relationships(file_path, content)
                
                for target_path, relationship_type, strength in relationships:
                    # Check if target exists in our graph or filesystem
                    if target_path in self.graph.nodes or os.path.exists(target_path):
                        self.add_edge(file_path, target_path, relationship_type, strength)
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _get_node_type(self, file_ext: str) -> str:
        """Determine node type from file extension"""
        type_mapping = {
            '.tscn': 'scene', '.scn': 'scene',
            '.gd': 'script', '.cs': 'script', '.js': 'script',
            '.tres': 'resource', '.res': 'resource',
            '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image',
            '.wav': 'audio', '.ogg': 'audio', '.mp3': 'audio',
            '.glsl': 'shader', '.shader': 'shader'
        }
        return type_mapping.get(file_ext, 'unknown')
    
    def _get_modality(self, file_ext: str) -> str:
        """Determine modality from file extension"""
        modality_mapping = {
            '.tscn': 'text', '.scn': 'text', '.gd': 'text', '.cs': 'text', '.js': 'text',
            '.tres': 'text', '.res': 'text', '.glsl': 'text', '.shader': 'text',
            '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image',
            '.wav': 'audio', '.ogg': 'audio', '.mp3': 'audio'
        }
        return modality_mapping.get(file_ext, 'text')
    
    def find_connected_files(self, file_path: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """Find all files connected to a given file within max_depth"""
        if file_path not in self.graph.nodes:
            return {}
        
        connected = defaultdict(list)
        
        # Use BFS to find connected nodes
        from collections import deque
        queue = deque([(file_path, 0)])
        visited = {file_path}
        
        while queue:
            current_file, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get outgoing connections (files this file uses)
            for neighbor in self.graph.successors(current_file):
                edge_data = self.graph[current_file][neighbor]
                relationship = edge_data.get('relationship_type', 'unknown')
                
                connected[f"uses_{relationship}"].append(neighbor)
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
            
            # Get incoming connections (files that use this file)
            for predecessor in self.graph.predecessors(current_file):
                edge_data = self.graph[predecessor][current_file]
                relationship = edge_data.get('relationship_type', 'unknown')
                
                connected[f"used_by_{relationship}"].append(predecessor)
                
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append((predecessor, depth + 1))
        
        return dict(connected)
    
    def get_central_files(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the most central/important files in the project"""
        if self.graph.number_of_nodes() == 0:
            return []
        
        # Calculate different centrality measures
        centralities = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph)
        }
        
        # Combine centrality scores
        combined_scores = {}
        for node in self.graph.nodes:
            score = (
                centralities['degree'].get(node, 0) * 0.4 +
                centralities['betweenness'].get(node, 0) * 0.3 +
                centralities['pagerank'].get(node, 0) * 0.3
            )
            combined_scores[node] = score
        
        # Sort by score and return top k
        sorted_files = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:top_k]
    
    def get_graph_summary(self) -> Dict:
        """Get a summary of the project graph"""
        if self.graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        # Node type distribution
        node_types = defaultdict(int)
        modalities = defaultdict(int)
        
        for node in self.graph.nodes(data=True):
            node_data = node[1]
            node_types[node_data.get('node_type', 'unknown')] += 1
            modalities[node_data.get('modality', 'unknown')] += 1
        
        # Relationship type distribution
        relationship_types = defaultdict(int)
        for edge in self.graph.edges(data=True):
            edge_data = edge[2]
            relationship_types[edge_data.get('relationship_type', 'unknown')] += 1
        
        # Graph metrics
        try:
            avg_clustering = nx.average_clustering(self.graph.to_undirected())
        except:
            avg_clustering = 0
        
        central_files = self.get_central_files(5)
        
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'modalities': dict(modalities),
            'relationship_types': dict(relationship_types),
            'avg_clustering': avg_clustering,
            'central_files': [{'file': f, 'centrality': c} for f, c in central_files],
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def export_graph(self) -> Dict:
        """Export the graph structure for visualization or storage"""
        return {
            'nodes': [
                {
                    'id': node,
                    **data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **data
                }
                for source, target, data in self.graph.edges(data=True)
            ],
            'summary': self.get_graph_summary()
        }