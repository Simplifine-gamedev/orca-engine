#!/usr/bin/env python3
"""
Enhanced Godot Project Parser
© 2025 Simplifine Corp. 

Comprehensive Godot project analysis that captures:
- Signal flows and connections across the entire project
- Scene-script relationships and node hierarchies  
- Resource dependency chains (textures→materials→meshes→scenes)
- Autoload/singleton tracking and global state
- Runtime behavior patterns (_ready, _process, etc.)
- Cross-file dependency mapping
- Export variable relationships
- Input action mappings
- Group memberships and collision layers
"""

import os
import re
import json
import hashlib
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import configparser

@dataclass
class GodotNode:
    """Enhanced node representation with full Godot context"""
    file_path: str
    node_type: str  # 'script', 'scene', 'resource', 'autoload', 'input_action'
    name: str
    class_name: Optional[str] = None
    extends: Optional[str] = None
    
    # Script-specific
    functions: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    signal_handlers: List[str] = field(default_factory=list)
    runtime_methods: List[str] = field(default_factory=list)  # _ready, _process, etc.
    
    # Scene-specific
    scene_nodes: List[Dict] = field(default_factory=list)
    node_scripts: Dict[str, str] = field(default_factory=dict)  # node_path -> script_path
    
    # Resource-specific
    resource_type: Optional[str] = None
    resource_dependencies: List[str] = field(default_factory=list)
    
    # Relationships
    preloads: List[str] = field(default_factory=list)
    loads: List[str] = field(default_factory=list)
    node_accesses: List[str] = field(default_factory=list)
    
    # Metadata
    groups: List[str] = field(default_factory=list)
    collision_layers: List[int] = field(default_factory=list)
    collision_masks: List[int] = field(default_factory=list)
    size_bytes: int = 0
    last_modified: float = 0.0
    checksum: str = ""

@dataclass
class GodotConnection:
    """Enhanced connection with full signal flow context"""
    source_file: str
    target_file: str
    connection_type: str  # 'signal_flow', 'script_attached', 'resource_used', 'extends', 'preload', 'scene_instance'
    
    # Signal-specific
    signal_name: Optional[str] = None
    handler_method: Optional[str] = None
    source_node_path: Optional[str] = None
    target_node_path: Optional[str] = None
    
    # Metadata
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedGodotParser:
    """Comprehensive Godot project parser with deep understanding"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.nodes: Dict[str, GodotNode] = {}
        self.connections: List[GodotConnection] = []
        self.autoloads: Dict[str, str] = {}  # name -> script_path
        self.input_actions: Dict[str, Dict] = {}
        self.signal_flows: Dict[str, List[Dict]] = defaultdict(list)  # signal_name -> [flow_info]
        
        # Enhanced regex patterns
        self.patterns = {
            # Script patterns
            'class_name': re.compile(r'class_name\s+([A-Za-z_]\w*)'),
            'extends': re.compile(r'extends\s+([A-Za-z_]\w*|"[^"]+")'),
            'function': re.compile(r'func\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:->.*?)?:'),
            'signal_def': re.compile(r'signal\s+([A-Za-z_]\w*)(?:\([^)]*\))?'),
            'export_var': re.compile(r'@export(?:\([^)]*\))?\s+var\s+([A-Za-z_]\w*)'),
            'preload': re.compile(r'preload\s*\(\s*["\']([^"\']+)["\']\s*\)'),
            'load': re.compile(r'load\s*\(\s*["\']([^"\']+)["\']\s*\)'),
            'get_node': re.compile(r'get_node\s*\(\s*["\']([^"\']+)["\']\s*\)'),
            'dollar_node': re.compile(r'\$([A-Za-z0-9_/]+)'),
            'signal_connect': re.compile(r'([A-Za-z_]\w*)\.connect\s*\(\s*["\']([^"\']+)["\']\s*,\s*([^,)]+)(?:\s*,\s*["\']([^"\']+)["\']\s*)?\)'),
            'signal_emit': re.compile(r'emit_signal\s*\(\s*["\']([^"\']+)["\']\s*(?:,.*?)?\)'),
            'scene_change': re.compile(r'get_tree\s*\(\s*\)\.change_scene(?:_to_file)?\s*\(\s*["\']([^"\']+)["\']\s*\)'),
            
            # Runtime method patterns
            'ready': re.compile(r'func\s+_ready\s*\('),
            'process': re.compile(r'func\s+_process\s*\('),
            'physics_process': re.compile(r'func\s+_physics_process\s*\('),
            'input': re.compile(r'func\s+_input\s*\('),
            'unhandled_input': re.compile(r'func\s+_unhandled_input\s*\('),
            
            # Scene patterns
            'ext_resource': re.compile(r'\[ext_resource[^]]*path="([^"]+)"[^]]*type="([^"]+)"[^]]*id="([^"]+)"'),
            'node_header': re.compile(r'\[node[^]]*name="([^"]+)"(?:[^]]*type="([^"]+)")?(?:[^]]*parent="([^"]*)")?'),
            'script_attach': re.compile(r'script\s*=\s*ExtResource\s*\(\s*"([^"]+)"\s*\)'),
            'scene_instance': re.compile(r'instance\s*=\s*ExtResource\s*\(\s*"([^"]+)"\s*\)'),
            'connection': re.compile(r'\[connection[^]]*signal="([^"]+)"[^]]*from="([^"]*)"[^]]*to="([^"]*)"[^]]*method="([^"]+)"'),
            
            # Resource patterns
            'resource_script': re.compile(r'script\s*=\s*ExtResource\s*\(\s*"([^"]+)"\s*\)'),
            'resource_texture': re.compile(r'(?:texture|albedo_texture)\s*=\s*ExtResource\s*\(\s*"([^"]+)"\s*\)'),
            'resource_material': re.compile(r'material\s*=\s*ExtResource\s*\(\s*"([^"]+)"\s*\)'),
            
            # Group and collision patterns
            'groups': re.compile(r'groups\s*=\s*PackedStringArray\s*\(\s*\[([^\]]*)\]'),
            'collision_layer': re.compile(r'collision_layer\s*=\s*(\d+)'),
            'collision_mask': re.compile(r'collision_mask\s*=\s*(\d+)'),
        }
    
    def parse_project(self) -> Dict[str, Any]:
        """Parse entire Godot project and build comprehensive graph"""
        print(f"🔍 Starting enhanced Godot project analysis: {self.project_root}")
        
        # Step 1: Parse project settings for autoloads and input actions
        self._parse_project_settings()
        
        # Step 2: Scan all project files
        all_files = self._scan_project_files()
        
        # Step 3: Parse each file type with enhanced analysis
        for file_path in all_files:
            self._parse_file(file_path)
        
        # Step 4: Build cross-file relationships and signal flows
        self._build_signal_flows()
        self._build_dependency_chains()
        
        # Step 5: Generate comprehensive graph data
        return self._generate_graph_payload()
    
    def _parse_project_settings(self):
        """Parse project.godot for autoloads and input actions"""
        project_file = os.path.join(self.project_root, "project.godot")
        if not os.path.exists(project_file):
            return
        
        try:
            # Read project.godot manually since it has a special format
            with open(project_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            current_section = None
            
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                # Parse section headers
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]
                    continue
                
                # Parse key=value pairs
                if '=' in line and current_section:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse autoloads
                    if current_section == 'autoload':
                        # Remove asterisk prefix if present
                        clean_path = value.lstrip('*').strip('"')
                        if clean_path.startswith('res://'):
                            clean_path = clean_path[6:]
                        self.autoloads[key] = clean_path
                        print(f"📡 Found autoload: {key} -> {clean_path}")
                    
                    # Parse input actions
                    elif current_section == 'input':
                        # This is an input action definition
                        self.input_actions[key] = {'definition': value}
                        print(f"🎮 Found input action: {key}")
                    
                    # Parse input action details
                    elif current_section.startswith('input/') or key in ['move_left', 'move_right', 'move_up', 'move_down', 'start_game']:
                        # Handle input action definitions that span multiple lines
                        if key not in self.input_actions:
                            self.input_actions[key] = {}
                        self.input_actions[key]['definition'] = value
                        print(f"🎮 Found input action: {key}")
                    
        except Exception as e:
            print(f"⚠️ Error parsing project settings: {e}")
    
    def _scan_project_files(self) -> List[str]:
        """Scan project for all relevant files"""
        files = []
        
        for root, dirs, filenames in os.walk(self.project_root):
            # Skip hidden and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'build', 'dist', '__pycache__'}]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.project_root)
                
                # Include all Godot-relevant files
                ext = os.path.splitext(filename)[1].lower()
                if ext in {'.gd', '.cs', '.tscn', '.scn', '.tres', '.res', '.gdshader', '.glsl', '.cfg', '.import'}:
                    files.append(rel_path)
        
        print(f"📁 Found {len(files)} project files to analyze")
        return files
    
    def _parse_file(self, rel_path: str):
        """Parse individual file with enhanced analysis"""
        full_path = os.path.join(self.project_root, rel_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Error reading {rel_path}: {e}")
            return
        
        # Get file metadata
        stat = os.stat(full_path)
        checksum = hashlib.md5(content.encode()).hexdigest()
        
        ext = os.path.splitext(rel_path)[1].lower()
        
        if ext == '.gd':
            self._parse_gdscript(rel_path, content, stat.st_size, stat.st_mtime, checksum)
        elif ext in {'.tscn', '.scn'}:
            self._parse_scene(rel_path, content, stat.st_size, stat.st_mtime, checksum)
        elif ext in {'.tres', '.res'}:
            self._parse_resource(rel_path, content, stat.st_size, stat.st_mtime, checksum)
    
    def _parse_gdscript(self, file_path: str, content: str, size: int, mtime: float, checksum: str):
        """Enhanced GDScript parsing with deep analysis"""
        
        # Extract basic structure
        class_name = self._extract_match(self.patterns['class_name'], content)
        extends = self._extract_match(self.patterns['extends'], content)
        
        # Extract all functions and categorize them
        functions = self.patterns['function'].findall(content)
        signal_handlers = [f for f in functions if f.startswith('_on_')]
        runtime_methods = []
        
        # Check for runtime methods
        for pattern_name in ['ready', 'process', 'physics_process', 'input', 'unhandled_input']:
            if self.patterns[pattern_name].search(content):
                runtime_methods.append(f'_{pattern_name}')
        
        # Extract signals and exports
        signals = self.patterns['signal_def'].findall(content)
        exports = self.patterns['export_var'].findall(content)
        
        # Extract dependencies
        preloads = [self._normalize_path(p) for p in self.patterns['preload'].findall(content)]
        loads = [self._normalize_path(p) for p in self.patterns['load'].findall(content)]
        
        # Extract node accesses
        node_accesses = (self.patterns['get_node'].findall(content) + 
                        self.patterns['dollar_node'].findall(content))
        
        # Extract groups and collision info
        groups = []
        collision_layers = []
        collision_masks = []
        
        groups_match = self.patterns['groups'].search(content)
        if groups_match:
            groups_str = groups_match.group(1)
            groups = [g.strip(' "') for g in groups_str.split(',') if g.strip()]
        
        layer_match = self.patterns['collision_layer'].search(content)
        if layer_match:
            collision_layers = [int(layer_match.group(1))]
            
        mask_match = self.patterns['collision_mask'].search(content)
        if mask_match:
            collision_masks = [int(mask_match.group(1))]
        
        # Create enhanced node
        node = GodotNode(
            file_path=file_path,
            node_type='script',
            name=class_name or os.path.basename(file_path),
            class_name=class_name,
            extends=extends,
            functions=functions,
            signals=signals,
            exports=exports,
            signal_handlers=signal_handlers,
            runtime_methods=runtime_methods,
            preloads=preloads,
            loads=loads,
            node_accesses=node_accesses,
            groups=groups,
            collision_layers=collision_layers,
            collision_masks=collision_masks,
            size_bytes=size,
            last_modified=mtime,
            checksum=checksum
        )
        
        self.nodes[file_path] = node
        
        # Extract signal connections for flow analysis
        self._extract_signal_connections(file_path, content)
        
        print(f"📜 Parsed script: {file_path} ({len(functions)} functions, {len(signals)} signals)")
    
    def _parse_scene(self, file_path: str, content: str, size: int, mtime: float, checksum: str):
        """Enhanced scene parsing with full node hierarchy"""
        
        # Extract external resources
        ext_resources = {}
        for match in self.patterns['ext_resource'].finditer(content):
            path, res_type, res_id = match.groups()
            ext_resources[res_id] = {
                'path': self._normalize_path(path),
                'type': res_type
            }
        
        # Extract scene nodes and their relationships
        scene_nodes = []
        node_scripts = {}
        
        current_node = None
        for line in content.split('\n'):
            line = line.strip()
            
            # Parse node headers
            node_match = self.patterns['node_header'].search(line)
            if node_match:
                node_name, node_type, parent = node_match.groups()
                current_node = {
                    'name': node_name,
                    'type': node_type or 'Node',
                    'parent': parent or '',
                    'path': parent + '/' + node_name if parent else node_name
                }
                scene_nodes.append(current_node)
                continue
            
            # Parse script attachments
            if current_node:
                script_match = self.patterns['script_attach'].search(line)
                if script_match:
                    resource_id = script_match.group(1)
                    if resource_id in ext_resources:
                        script_path = ext_resources[resource_id]['path']
                        node_scripts[current_node['path']] = script_path
                        
                        # Create connection
                        self.connections.append(GodotConnection(
                            source_file=file_path,
                            target_file=script_path,
                            connection_type='script_attached',
                            target_node_path=current_node['path'],
                            metadata={'node_type': current_node['type']}
                        ))
        
        # Extract scene connections (signal flows) from [connection] blocks
        for match in self.patterns['connection'].finditer(content):
            signal_name, from_path, to_path, method = match.groups()
            
            # Find the actual script files involved
            from_script = node_scripts.get(from_path, file_path)
            to_script = node_scripts.get(to_path, file_path)
            
            self.connections.append(GodotConnection(
                source_file=from_script,
                target_file=to_script,
                connection_type='signal_flow',
                signal_name=signal_name,
                handler_method=method,
                source_node_path=from_path,
                target_node_path=to_path
            ))
            
            # Track signal flow
            self.signal_flows[signal_name].append({
                'scene': file_path,
                'from_node': from_path,
                'to_node': to_path,
                'from_script': from_script,
                'to_script': to_script,
                'method': method
            })
        
        # ENHANCED: Also parse connections from the [connection] lines at the end
        connection_lines = re.findall(r'\[connection signal="([^"]+)" from="([^"]*)" to="([^"]*)" method="([^"]+)"\]', content)
        for signal_name, from_path, to_path, method in connection_lines:
            # Resolve script files
            from_script = node_scripts.get(from_path, file_path) if from_path else file_path
            to_script = node_scripts.get(to_path, file_path) if to_path else file_path
            
            # If to_path is "." it means the main scene script
            if to_path == ".":
                # Look for the main scene script in ext_resources
                for res_id, res_info in ext_resources.items():
                    if res_info['type'] == 'Script' and res_info['path'].endswith('.gd'):
                        to_script = res_info['path']
                        break
            
            self.connections.append(GodotConnection(
                source_file=from_script,
                target_file=to_script,
                connection_type='signal_flow',
                signal_name=signal_name,
                handler_method=method,
                source_node_path=from_path,
                target_node_path=to_path
            ))
            
            # Track signal flow
            self.signal_flows[signal_name].append({
                'scene': file_path,
                'from_node': from_path,
                'to_node': to_path,
                'from_script': from_script,
                'to_script': to_script,
                'method': method,
                'connection_type': 'scene_connection'
            })
            
            print(f"🔗 Found signal connection: {signal_name} from {from_path} to {to_path} -> {method}")
        
        # Create scene node
        node = GodotNode(
            file_path=file_path,
            node_type='scene',
            name=os.path.basename(file_path),
            scene_nodes=scene_nodes,
            node_scripts=node_scripts,
            size_bytes=size,
            last_modified=mtime,
            checksum=checksum
        )
        
        self.nodes[file_path] = node
        
        print(f"🎬 Parsed scene: {file_path} ({len(scene_nodes)} nodes, {len(node_scripts)} scripts)")
    
    def _parse_resource(self, file_path: str, content: str, size: int, mtime: float, checksum: str):
        """Enhanced resource parsing with dependency tracking"""
        
        # Determine resource type from content
        resource_type = 'Resource'
        if 'Material' in content:
            resource_type = 'Material'
        elif 'Texture' in content:
            resource_type = 'Texture'
        elif 'Mesh' in content:
            resource_type = 'Mesh'
        elif 'PackedScene' in content:
            resource_type = 'PackedScene'
        
        # Extract resource dependencies
        dependencies = []
        
        # Find texture dependencies
        for match in self.patterns['resource_texture'].finditer(content):
            resource_id = match.group(1)
            # Would need to resolve ExtResource ID to actual path
            dependencies.append(f"texture_dependency_{resource_id}")
        
        # Find material dependencies  
        for match in self.patterns['resource_material'].finditer(content):
            resource_id = match.group(1)
            dependencies.append(f"material_dependency_{resource_id}")
        
        # Create resource node
        node = GodotNode(
            file_path=file_path,
            node_type='resource',
            name=os.path.basename(file_path),
            resource_type=resource_type,
            resource_dependencies=dependencies,
            size_bytes=size,
            last_modified=mtime,
            checksum=checksum
        )
        
        self.nodes[file_path] = node
        
        print(f"📦 Parsed resource: {file_path} (type: {resource_type})")
    
    def _extract_signal_connections(self, file_path: str, content: str):
        """Extract signal connections from script content"""
        
        # Find signal.connect() calls
        for match in self.patterns['signal_connect'].finditer(content):
            source_obj, signal_name, target_obj, method = match.groups()
            
            # Try to resolve target object to actual file
            target_file = self._resolve_node_reference(target_obj, file_path)
            
            self.connections.append(GodotConnection(
                source_file=file_path,
                target_file=target_file or file_path,
                connection_type='signal_flow',
                signal_name=signal_name,
                handler_method=method or f"_on_{signal_name}",
                metadata={'source_object': source_obj, 'target_object': target_obj}
            ))
        
        # Find emit_signal() calls
        for signal_name in self.patterns['signal_emit'].findall(content):
            self.signal_flows[signal_name].append({
                'emitter_script': file_path,
                'signal': signal_name,
                'type': 'emission'
            })
    
    def _build_signal_flows(self):
        """Build comprehensive signal flow mapping"""
        print(f"🔗 Building signal flow map ({len(self.signal_flows)} unique signals)")
        
        for signal_name, flows in self.signal_flows.items():
            emitters = [f for f in flows if f.get('type') == 'emission']
            handlers = [f for f in flows if f.get('method')]
            
            print(f"  📡 {signal_name}: {len(emitters)} emitters → {len(handlers)} handlers")
    
    def _build_dependency_chains(self):
        """Build resource dependency chains"""
        print("🔗 Building dependency chains...")
        
        # Add autoload connections
        for autoload_name, script_path in self.autoloads.items():
            if script_path in self.nodes:
                self.nodes[script_path].node_type = 'autoload'
                print(f"  🌐 Autoload: {autoload_name} -> {script_path}")
        
        # Add preload/load connections
        for file_path, node in self.nodes.items():
            for preload_path in node.preloads:
                if preload_path in self.nodes:
                    self.connections.append(GodotConnection(
                        source_file=file_path,
                        target_file=preload_path,
                        connection_type='preload',
                        strength=0.9
                    ))
            
            for load_path in node.loads:
                if load_path in self.nodes:
                    self.connections.append(GodotConnection(
                        source_file=file_path,
                        target_file=load_path,
                        connection_type='load',
                        strength=0.7
                    ))
    
    def _generate_graph_payload(self) -> Dict[str, Any]:
        """Generate comprehensive graph payload"""
        
        # Convert nodes to serializable format
        nodes_data = []
        for file_path, node in self.nodes.items():
            node_data = asdict(node)
            nodes_data.append(node_data)
        
        # Convert connections to serializable format
        connections_data = []
        for conn in self.connections:
            conn_data = asdict(conn)
            connections_data.append(conn_data)
        
        # Generate summary statistics
        summary = {
            'total_files': len(self.nodes),
            'scripts': len([n for n in self.nodes.values() if n.node_type == 'script']),
            'scenes': len([n for n in self.nodes.values() if n.node_type == 'scene']),
            'resources': len([n for n in self.nodes.values() if n.node_type == 'resource']),
            'autoloads': len(self.autoloads),
            'input_actions': len(self.input_actions),
            'total_connections': len(self.connections),
            'signal_flows': len(self.signal_flows),
            'unique_signals': len(self.signal_flows)
        }
        
        payload = {
            'project_root': self.project_root,
            'nodes': nodes_data,
            'connections': connections_data,
            'autoloads': self.autoloads,
            'input_actions': self.input_actions,
            'signal_flows': dict(self.signal_flows),
            'summary': summary,
            'version': '2.0.0'
        }
        
        print(f"✅ Generated enhanced graph: {summary}")
        return payload
    
    # Helper methods
    def _extract_match(self, pattern: re.Pattern, text: str) -> Optional[str]:
        """Extract first match from pattern"""
        match = pattern.search(text)
        return match.group(1) if match else None
    
    def _normalize_path(self, path: str) -> str:
        """Normalize Godot resource path"""
        if path.startswith('res://'):
            return path[6:]
        return path
    
    def _resolve_node_reference(self, node_ref: str, context_file: str) -> Optional[str]:
        """Try to resolve node reference to actual file"""
        # This would need more sophisticated logic to resolve
        # $NodeName, get_node("NodeName"), self, etc. to actual script files
        # For now, return None to indicate same file
        return None


# Integration function for the existing system
def create_enhanced_godot_graph(project_root: str, user_id: str, project_id: str) -> Dict[str, Any]:
    """Create enhanced Godot graph for integration with existing system"""
    parser = EnhancedGodotParser(project_root)
    graph_data = parser.parse_project()
    
    # Add integration metadata
    graph_data['user_id'] = user_id
    graph_data['project_id'] = project_id
    graph_data['created_at'] = time.time()
    
    return graph_data


if __name__ == "__main__":
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        parser = EnhancedGodotParser(project_path)
        result = parser.parse_project()
        
        # Save result for inspection
        with open('enhanced_graph_output.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"📊 Enhanced graph saved to enhanced_graph_output.json")
