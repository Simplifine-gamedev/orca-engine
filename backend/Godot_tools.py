


# NEW set of tools :)
godot_tools = [
    {
        "type": "function",
        "function": {
            "name": "project_manager",
            "description": "Project context, filesystem, asset library, and update checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # Context
                            "context.get",
                            # Filesystem
                            "fs.list", "fs.read", "fs.write", "fs.write_lines", "fs.replace_string",
                            "fs.copy", "fs.move", "fs.delete",
                            "fs.mkdir", "fs.symlink", "fs.refresh",
                            # Project-level ops
                            "project.analyze_dir", "project.copy_dir", "project.update_refs",
                            # Asset library
                            "assets.search", "assets.install",
                            # Updates
                            "updates.check"
                        ],
                        "description": "Operation selector"
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    # Context.get options
                    "context_mode": {
                        "type": "string",
                        "enum": ["structure", "hierarchy", "find_scenes", "patterns"],
                        "description": "When op == 'context.get'"
                    },
                    "project_root": {"type": "string"},
                    "scene_path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "include_details": {"type": "boolean", "default": True},

                    # fs.list
                    "dir": {"type": "string", "description": "Directory to list (alternative to 'path')"},
                    "path": {"type": "string", "description": "Directory path to list (alternative to 'dir')"},
                    "filter": {"type": "string", "description": "Basic file filter pattern (e.g., '*.gd')"},
                    "file_patterns": {"type": "array", "items": {"type": "string"}, "description": "Array of file patterns to filter by (e.g., ['*.glb', '*.gltf'])"},
                    "recursive": {"type": "boolean", "default": False, "description": "If true, recursively list all subdirectories"},
                    "full_paths": {"type": "boolean", "default": True, "description": "If true, return full paths; otherwise relative paths"},

                    # fs.read / fs.write (whole file replacement)
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "New content for fs.write (whole file replacement)"},
                    
                    # fs.write_lines (line range editing)
                    "start_line": {"type": "integer", "description": "Starting line number for fs.write_lines (1-based)"},
                    "end_line": {"type": "integer", "description": "Ending line number for fs.write_lines (1-based, inclusive)"},
                    "lines_content": {"type": "string", "description": "New content for the specified line range"},
                    
                    # fs.replace_string (precise string replacement)
                    "find_string": {"type": "string", "description": "String to find for fs.replace_string"},
                    "replace_string": {"type": "string", "description": "String to replace with for fs.replace_string"},
                    "replace_all": {"type": "boolean", "default": False, "description": "Replace all occurrences (default: false, replace first only)"},
                    "case_sensitive": {"type": "boolean", "default": True, "description": "Case sensitive search (default: true)"},

                    # fs.copy / fs.move / fs.delete / fs.mkdir / fs.symlink
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                    "target": {"type": "string"},
                    "link_path": {"type": "string"},

                    # project.* ops
                    "target_path": {"type": "string"},
                    "source_addon": {"type": "string"},
                    "target_addon": {"type": "string"},
                    "old_path": {"type": "string"},
                    "new_path": {"type": "string"},
                    "file_patterns": {"type": "array", "items": {"type": "string"}},

                    # assets.search
                    "asset_query": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["2d_tools", "3d_tools", "shaders", "materials", "tools", "scripts", "misc", "templates", "demos", "plugins"]
                    },
                    "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                    "support_level": {"type": "string", "enum": ["all", "official", "featured", "community", "testing"], "default": "all"},
                    "godot_version": {"type": "string", "default": "4.3"},
                    "sort_by": {"type": "string", "enum": ["rating", "updated", "name", "cost"], "default": "rating"},
                    "sort_reverse": {"type": "boolean", "default": False},
                    "asset_type": {"type": "string", "enum": ["any", "addon", "project"], "default": "any"},
                    "cost_filter": {"type": "string", "enum": ["all", "free", "paid"], "default": "all"},

                    # assets.install
                    "asset_id": {"type": "string"},
                    "project_path": {"type": "string"},
                    "install_location": {"type": "string", "default": "addons/"},
                    "create_backup": {"type": "boolean", "default": True},

                    # updates.check
                    "force_check": {"type": "boolean", "default": False},
                    "show_notification": {"type": "boolean", "default": True}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "scene_manager",
            "description": "Scenes, nodes, properties, resources, collisions, signals, groups, selection, and bulk/copy configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # Scene ops
                            "scene.open", "scene.create", "scene.save_as", "scene.instantiate", "scene.instantiate_batch",
                            "scene.analyze", "scene.info", "scene.nodes.get_all",
                            "scene.nodes.find_by_type", "editor.selection.get",
                            "scene.bulk_configure", "scene.copy_configuration",
                            # Node CRUD & type
                            "node.create", "node.create_batch", "node.delete", "node.delete_batch", "node.move",
                            "node.type.change", "node.type.set", "node.rename",
                            # Advanced batch operations
                            "node.create_and_configure_batch", "node.assign_resources_batch", "node.set_transforms_batch",
                            # Pattern-based operations
                            "node.props.set_pattern", "node.delete_pattern", "node.assign_resource_pattern",
                            # Groups
                            "groups.add", "groups.remove", "groups.list",
                            # Props & methods  
                            "node.props.get", "node.props.set_batch", "node.mesh.set_properties", "node.method.call",
                            # Resources & collisions
                            "node.assign_resource", "node.add_collision",
                            # Signals & connections
                            "signals.list_node_signals", "signals.list_connections",
                            "signals.list_incoming_connections", "signals.connect",
                            "signals.disconnect", "signals.validate", "signals.open_dialog",
                            "signals.trace.start", "signals.trace.stop", "signals.trace.events"
                        ],
                        "description": "Operation selector"
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    # Scene ops
                    "path": {"type": "string"},
                    "root_type": {"type": "string"},
                    "include_current_as_child": {"type": "boolean", "default": False},
                    "scene_path": {"type": "string"},
                    "parent_node": {"type": "string", "description": "Parent node path (required for instantiate/create operations)"},
                    "instance_name": {"type": "string"},
                    "await_import": {"type": "boolean", "default": True, "description": "Whether to wait for import before loading (recommended for GLB/GLTF files)"},
                    "timeout_ms": {"type": "integer", "default": 30000, "description": "Timeout in milliseconds for import waiting (30 seconds default)"},
                    "skip_import_wait": {"type": "boolean", "default": False, "description": "Skip import waiting entirely - use for problematic files that cause import loops"},
                    "scope": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    
                    # Batch scene instantiation
                    "instantiate_batch": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scene_path": {"type": "string", "description": "Path to scene/GLB file to instantiate"},
                                "parent_node": {"type": "string", "description": "Parent node path"},
                                "instance_name": {"type": "string", "description": "Optional name for the instance"}
                            },
                            "required": ["scene_path", "parent_node"]
                        }
                    },
                    
                    # Pattern-based operations
                    "node_pattern": {"type": "string", "description": "Node path pattern with wildcards (e.g., 'Hallway//Column*_Left')"},
                    "property_pattern": {"type": "string", "description": "Property to set on pattern-matched nodes"},
                    "value_pattern": {"description": "Value to set on pattern-matched nodes"},
                    "resource_path_pattern": {"type": "string", "description": "Resource path for pattern-based assignment"},

                    # Node create/find
                    "type": {"type": "string"},
                    "name": {"type": "string"},
                    
                    # Node batch operations
                    "node_paths": {"type": "array", "items": {"type": "string"}, "description": "Array of node paths to delete (e.g., ['Floor/Cube', 'UI/Button1', 'Player/Weapon'])"},
                    "ignore_missing": {"type": "boolean", "default": True, "description": "If true, continue deleting other nodes even if some don't exist"},
                    "skip_scene_root": {"type": "boolean", "default": True, "description": "If true, automatically skip scene root nodes for safety"},
                    "nodes_to_create": {"type": "array", "items": {"type": "object"}, "description": "Array of node specs to create: [{type: 'MeshInstance3D', name: 'Floor', parent: 'World'}, {type: 'Camera3D', name: 'MainCamera'}]"},
                    "stop_on_error": {"type": "boolean", "default": True, "description": "If true, stop batch operation on first error"},

                    # Node move/type/rename
                    "new_parent": {"type": "string"},
                    "new_type": {"type": "string"},
                    "preserve_children": {"type": "boolean", "default": True},
                    "strategy": {"type": "string", "enum": ["wrap_root", "swap"], "default": "wrap_root"},
                    "type_name": {"type": "string"},
                    "script_path": {"type": "string"},
                    "new_name": {"type": "string"},

                    # Groups
                    "group": {"type": "string"},
                    "groups": {"type": "array", "items": {"type": "string"}},

                    # Properties & batch ops
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "property": {"type": "string"},
                                "value": {}
                            },
                            "required": ["path", "property", "value"]
                        }
                    },
                    "property": {"type": "string"},
                    
                    # Mesh property updates
                    "mesh_property": {"type": "string", "enum": ["radius", "size", "height", "top_radius", "bottom_radius", "radial_segments", "rings"], "description": "Mesh property to update (e.g., 'radius' for SphereMesh)"},
                    "mesh_value": {"description": "New value for the mesh property (e.g., 0.038 for radius, or {x: 1, y: 2, z: 3} for size)"},

                    # Node property inspection (node.props.get) - filtering/pagination
                    "include": {"type": "array", "items": {"type": "string"}, "description": "Exact property names to include"},
                    "ensure": {"type": "array", "items": {"type": "string"}, "description": "Always-include property names (bypass limits)"},
                    "prefix": {"type": "string", "description": "Include properties beginning with this prefix"},
                    "offset": {"type": "integer", "default": 0, "description": "Skip first N matching properties"},
                    "editor_only": {"type": "boolean", "default": True, "description": "When false, include non-editor properties"},
                    "max_properties": {"type": "integer", "default": -1, "description": "Set -1 for no limit (default: unlimited for AI debugging)"},

                    # Scene listing controls (scene.nodes.get_all)
                    "owned_only": {"type": "boolean", "default": True, "description": "Only include nodes owned by the edited scene"},
                    "max_nodes": {"type": "integer", "default": 500, "description": "Limit traversal size"},

                    # Methods
                    "method": {"type": "string"},
                    "args": {"type": "array", "items": {}},

                    # Resource assignment
                    "resource": {},

                    # Collision
                    "node_path": {"type": "string"},
                    "shape_type": {"type": "string", "enum": ["rectangle", "circle", "capsule", "box", "box3d", "sphere", "sphere3d", "capsule3d", "convex", "convex3d", "trimesh", "trimesh3d"]},

                    # Bulk/copy config
                    "transformations": {"type": "object"},
                    "validation": {"type": "boolean", "default": True},
                    "source_config_scene": {"type": "string", "description": "Source node path for copy_configuration (will be mapped to 'source')"},
                    "source": {"type": "string", "description": "Source node path (alternative to source_config_scene)"},

                    # Advanced batch operations
                    "templates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Node type (e.g., 'MeshInstance3D')"},
                                "name": {"type": "string", "description": "Name pattern with {i} placeholder (e.g., 'Column{i}_Left')"},
                                "parent": {"type": "string", "description": "Parent node path"},
                                "count": {"type": "integer", "description": "Number of nodes to create"},
                                "mesh": {"type": "string", "description": "Mesh resource path"},
                                "material": {"type": "string", "description": "Material resource path"},
                                "properties": {"type": "object", "description": "Properties to set on each node"},
                                "positions": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": {"type": "string", "enum": ["linear", "grid", "circle", "custom"]},
                                        "start": {"type": "object", "description": "Starting position {x, y, z}"},
                                        "spacing": {"type": "object", "description": "Spacing between nodes {x, y, z}"},
                                        "grid_size": {"type": "object", "description": "Grid dimensions {x, y, z} for grid pattern"},
                                        "radius": {"type": "number", "description": "Radius for circle pattern"},
                                        "custom_positions": {"type": "array", "items": {"type": "object"}, "description": "Custom position list"}
                                    }
                                }
                            },
                            "required": ["type", "name", "parent", "count"]
                        }
                    },
                    "batch_resources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_paths": {"type": "array", "items": {"type": "string"}},
                                "property": {"type": "string"},
                                "resource_path": {"type": "string"}
                            }
                        }
                    },
                    "batch_transforms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_paths": {"type": "array", "items": {"type": "string"}},
                                "positions": {"type": "array", "items": {"type": "object"}},
                                "rotations": {"type": "array", "items": {"type": "object"}},
                                "scales": {"type": "array", "items": {"type": "object"}}
                            }
                        }
                    },

                    # Signals
                    "signal_name": {"type": "string"},
                    "source_path": {"type": "string"},
                    "target_path": {"type": "string"},
                    "binds": {"type": "array", "items": {}},
                    "flags": {"type": "integer"},
                    "node_paths": {"type": "array", "items": {"type": "string"}},
                    "signals": {"type": "array", "items": {"type": "string"}},
                    "include_args": {"type": "boolean", "default": False},
                    "max_events": {"type": "integer", "default": 100},
                    "trace_id": {"type": "string"},
                    "since_index": {"type": "integer"}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "script_manager",
            "description": "Scripts and classes: attach/detach/reload, class registry, and compilation checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "script.get_for_node", "script.attach", "script.detach", "script.reload",
                            "classes.refresh", "classes.custom_list", "classes.available",
                            "compile.check"
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    "path": {"type": "string", "description": "Node path (for get/attach/detach)"},
                    "script_path": {"type": "string", "description": "Script file path (attach/reload)"},
                    "pattern": {"type": "string", "description": "Filter for custom classes"},
                    "check_path": {"type": "string", "description": "Script file path to check (only used when op='compile.check' and check_all=false)"},
                    "check_all": {"type": "boolean", "default": False},
                    "check_mode": {"type": "string", "enum": ["scripts", "output"], "default": "scripts"}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "resource_manager",
            "description": "Create/inspect/modify/assign resources, load & assign, import options and reimport, image ops, and spritesheet slicing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "res.create", "res.inspect", "res.modify", "res.assign",
                            "res.copy_from_template", "res.refresh",
                            "res.load_and_assign", "res.create_and_assign",
                            "import.set_options", "import.reimport",
                            "shader.clear_cache", "shader.force_recompile", "shader.debug_cache",
                            "image.generate_or_edit", "image.save", "image.slice_spritesheet"
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    # Resources
                    "type": {"type": "string", "description": "Resource type (e.g., 'StandardMaterial3D', 'BoxMesh', 'SphereMesh')"},
                    "resource_type": {"type": "string", "description": "Resource type for res.create_and_assign (e.g., 'StandardMaterial3D', 'BoxMesh', 'SphereMesh')"},
                    "props": {"type": "object", "description": "Properties to set/modify. For BoxMesh use: {size: {x: 50, y: 0.2, z: 50}}. For SphereMesh: {radius: 1.0, radial_segments: 16, rings: 8}. For materials: {albedo_color: {r: 1, g: 0, b: 0, a: 1}, metallic: 0.5, albedo_texture: {path: 'res://texture.png'}} or {albedo_texture: 'res://texture.png'}"},
                    "properties": {"type": "object", "description": "Properties for res.create_and_assign - same format as props"},
                    "save_path": {"type": "string"},
                    "target": {"type": "string", "description": "Resource target path to inspect/modify"},
                    "path": {"type": "string", "description": "Node path (for res.assign)"},
                    "property": {"type": "string", "description": "Node property to assign to"},
                    "resource": {"type": "object", "description": "Inline resource spec or {path: 'res://...'}"},
                    "resource_path": {"type": "string", "description": "Path to load resource from (res.load_and_assign)"},
                    "node_path": {"type": "string", "description": "Node path to assign loaded resource to"},
                    "await_import": {"type": "boolean", "default": True, "description": "Whether to wait for import before loading (recommended for GLB/GLTF files)"},
                    "timeout_ms": {"type": "integer", "default": 30000, "description": "Timeout in milliseconds for import waiting (30 seconds default)"},
                    "force_reimport": {"type": "boolean", "default": False, "description": "Force reimport the resource before loading"},
                    "skip_import_wait": {"type": "boolean", "default": False, "description": "Skip import waiting entirely - use for problematic files that cause import loops"},
                    "source_template": {"type": "string", "description": "Resource template path to clone from"},

                    # Import pipeline
                    "import_path": {"type": "string", "description": "Source asset to set options/reimport"},
                    "import_options": {"type": "object", "description": "Importer-specific options (e.g., texture flags)"},
                    
                    # Shader cache management
                    "shader_path": {"type": "string", "description": "Specific shader file path for cache operations"},
                    "cache_type": {"type": "string", "enum": ["user", "project", "all"], "default": "all", "description": "Which shader cache to clear"},
                    "force_recompile_all": {"type": "boolean", "default": False, "description": "Force recompile all shaders in project"},

                    # Image generate/edit
                    "description": {"type": "string", "description": "Text description of the image to generate or edit"},
                    "images": {"type": "array", "items": {"type": "string"}, "description": "Array of image IDs to edit (leave empty to generate new image). Use image IDs from previous conversation like 'generated_abc123' or 'edited_def456'"},
                    "style": {"type": "string", "description": "Art style for the image (e.g., 'pixel art', 'photorealistic', '3D render')"},
                    "size": {"type": "string"},
                    "exact_size": {"type": "string"},
                    "tile_size": {"type": "string"},
                    "grid": {"type": "string"},
                    "resize_filter": {"type": "string", "enum": ["nearest", "bilinear", "bicubic", "lanczos"], "default": "lanczos"},
                    "path_to_save": {"type": "string"},

                    # Image save
                    "image_id": {"type": "string", "description": "ID of the image from conversation (e.g., 'generated_abc123', 'edited_def456')"},
                    "path": {"type": "string", "description": "Path where to save the image in the project (e.g., 'res://textures/floor_texture.png')"},
                    "format": {"type": "string", "enum": ["png", "jpg", "jpeg"], "default": "png"},

                    # Spritesheet slicing
                    "sheet_base64": {"type": "string"},
                    "sheet_path": {"type": "string"},
                    "margin": {"type": "integer", "default": 0},
                    "spacing": {"type": "integer", "default": 0},
                    "auto_detect": {"type": "boolean", "default": True},
                    "bg_tolerance": {"type": "integer", "default": 24},
                    "alpha_threshold": {"type": "integer", "default": 1},
                    "tight_crop": {"type": "boolean", "default": True},
                    "padding": {"type": "integer", "default": 0},
                    "fuzzy": {"type": "integer", "default": 2},
                    "normalize_to": {"type": "string"}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "settings_manager",
            "description": "ProjectSettings, InputMap actions, Autoloads (singletons), and layer/mask names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # ProjectSettings
                            "project_settings.get", "project_settings.set", "project_settings.list",
                            "project_settings.get_many", "project_settings.search",
                            # Autoloads
                            "autoload.add", "autoload.remove",
                            # Layer names
                            "layers.get_names", "layers.set_name"
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    # ProjectSettings
                    "key": {"type": "string", "description": "e.g., 'application/config/name'"},
                    "value": {},
                    "prefix": {"type": "string", "description": "List settings whose key starts with this prefix"},
                    "keys": {"type": "array", "items": {"type": "string"}, "description": "Array of keys for get_many operation"},
                    "query": {"type": "string", "description": "Search term for project_settings.search"},
                    "search_in_values": {"type": "boolean", "default": False, "description": "Search in setting values too"},
                    "keys_only": {"type": "boolean", "default": False, "description": "Return only keys without values"},
                    "offset": {"type": "integer", "default": 0, "description": "Pagination offset"},
                    "limit": {"type": "integer", "default": 200, "description": "Max results per page"},

                    # Autoloads
                    "autoload_name": {"type": "string"},
                    "autoload_path": {"type": "string"},
                    "autoload_is_singleton": {"type": "boolean", "default": True},

                    # Layer names (2D/3D physics/render layers)
                    "layer_scope": {
                        "type": "string",
                        "enum": ["2d_physics", "3d_physics", "2d_render", "3d_render"]
                    },
                    "layer_index": {"type": "integer"},
                    "layer_name": {"type": "string"}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "animation_manager",
            "description": "Create and edit animations on AnimationPlayer: animations, tracks, and keyframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "animation.create", "animation.remove",
                            "animation.set_meta",
                            "track.add", "track.remove",
                            "key.insert", "key.remove"
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    "player_path": {"type": "string", "description": "Node path to AnimationPlayer"},
                    "animation_name": {"type": "string"},
                    "length": {"type": "number"},
                    "loop": {"type": "boolean", "default": False},
                    "speed_scale": {"type": "number", "default": 1.0},

                    # Tracks
                    "track_type": {
                        "type": "string",
                        "enum": ["property", "method", "bezier", "audio", "animation"],
                        "description": "Commonly 'property' or 'method'"
                    },
                    "track_path": {"type": "string", "description": "Node path affected by the track (for property tracks)"},
                    "track_property": {"type": "string", "description": "e.g., 'position', 'modulate:a'"},
                    "track_index": {"type": "integer"},

                    # Keys
                    "time": {"type": "number"},
                    "value": {},
                    "transition": {"type": "number", "description": "Optional easing/transition"},
                    "in_handle": {"type": "object"},
                    "out_handle": {"type": "object"}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_manager",
            "description": "Search across the project (semantic/keyword/hybrid) and Godot docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["project.search", "docs.search"]},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "include_graph": {"type": "boolean", "default": True},
                    "modality_filter": {"type": "string", "enum": ["text", "image", "audio"]},
                    "project_root": {"type": "string"},
                    "project_id": {"type": "string"},
                    "trace_dependencies": {"type": "boolean", "default": False},
                    "search_mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid", "grep", "auto"], "default": "semantic"},

                    # Grep-specific options (only for grep mode)
                    "case_sensitive": {"type": "boolean", "default": True, "description": "Case-sensitive grep search"},
                    "whole_words": {"type": "boolean", "default": False, "description": "Match whole words only"},
                    "file_extensions": {"type": "array", "items": {"type": "string"}, "description": "Limit grep to specific file extensions (e.g., ['gd', 'tres', 'tscn'])"},

                    # Docs filters
                    "section_filter": {"type": "string", "enum": ["overview", "methods", "properties", "signals"]},
                    "class_filter": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                    "code_examples_only": {"type": "boolean", "default": False}
                },
                "required": ["op", "query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "runtime_manager",
            "description": "Run/stop/status, error summaries/details, screenshots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["game.start", "game.stop", "game.status", "errors.summary", "errors.details", "errors.test", "errors.debug", "screenshot.capture", "console.get_output", "input.test_action", "input.test_key"]},
                    "scene_path": {"type": "string"},
                    "clear_errors": {"type": "boolean", "default": True},

                    "include_warnings": {"type": "boolean", "default": True},
                    "file_filter": {"type": "string"},
                    "max_count": {"type": "integer", "default": 20},
                    "message_contains": {"type": "string"},
                    "group_duplicates": {"type": "boolean", "default": True},

                    # Screenshot parameters
                    "filename": {"type": "string", "default": "screenshot_debug.png"},
                    "target": {"type": "string", "enum": ["editor", "game", "both"], "default": "game"},
                    "return_base64": {"type": "boolean", "default": True},
                    
                    # Console output parameters
                    "output_type": {"type": "string", "enum": ["all", "print", "error", "warning"], "default": "all"},
                    "max_lines": {"type": "integer", "default": 50, "description": "Maximum console lines to retrieve"},
                    "since_timestamp": {"type": "integer", "description": "Only get output since this timestamp (milliseconds)"},
                    
                    # Input testing parameters
                    "action_name": {"type": "string", "description": "Input action name to test"},
                    "key_code": {"type": "integer", "description": "Key code to test (e.g., 32 for space)"},
                    "test_duration": {"type": "number", "default": 1.0, "description": "How long to test input (seconds)"}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "runtime_inspector",
            "description": "Inspect and modify runtime node properties, materials, shaders, environment settings, and mesh data during play. Includes advanced diagnostics for conflict detection and script behavior analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # Runtime node properties
                            "runtime.node.get_props", "runtime.node.set_prop", 
                            "runtime.node.get_tree", "runtime.node.find_by_type",
                            # Material/shader inspection
                            "runtime.material.get", "runtime.material.set_param",
                            "runtime.material.list_params", "runtime.material.get_shader_code",
                            # Mesh inspection
                            "runtime.mesh.get_arrays", "runtime.mesh.get_uv_info",
                            "runtime.mesh.get_surface_count", "runtime.mesh.get_surface_material",
                            # Environment/lighting
                            "runtime.environment.get", "runtime.environment.set",
                            "runtime.camera.get_exposure",
                            # Debug helpers
                            "runtime.watch.add", "runtime.watch.remove", "runtime.watch.get_values",
                            "runtime.breakpoint.set", "runtime.breakpoint.clear",
                            # Help/info
                            "runtime.debug.info", "help", "runtime.debug.tree_dump",
                            # ADVANCED DIAGNOSTICS - The "System Observatory"
                            "runtime.node.diagnose", "runtime.property.trace",
                            "runtime.node.list_scripts", "runtime.script.analyze_effects",
                            "runtime.node.get_full_state", "runtime.script.toggle"
                        ],
                        "description": "Runtime inspection operation"
                    },
                    
                    # Common parameters
                    "node_path": {"type": "string", "description": "Path to node in remote/runtime scene tree"},
                    "property": {"type": "string", "description": "Property name to get/set"},
                    "value": {"description": "Value to set for property"},
                    
                    # Material/shader params
                    "material_property": {"type": "string", "description": "Material property path (e.g., 'material_override', 'surface_material_override/0')"},
                    "shader_param": {"type": "string", "description": "Shader parameter name"},
                    "shader_value": {"description": "Shader parameter value"},
                    "include_shader_code": {"type": "boolean", "default": False, "description": "Include shader source code"},
                    
                    # Mesh params
                    "surface_index": {"type": "integer", "description": "Mesh surface index"},
                    "array_type": {"type": "string", "enum": ["vertex", "normal", "tangent", "color", "tex_uv", "tex_uv2", "custom0", "custom1", "custom2", "custom3", "bones", "weights", "index"], "description": "Type of mesh array to retrieve"},
                    
                    # Environment params
                    "env_property": {"type": "string", "description": "Environment property (e.g., 'tonemap_mode', 'tonemap_exposure', 'tonemap_white')"},
                    "env_value": {"description": "Environment property value"},
                    
                    # Watch/breakpoint params
                    "watch_id": {"type": "string", "description": "Unique ID for watch"},
                    "watch_expression": {"type": "string", "description": "Expression to watch"},
                    "breakpoint_line": {"type": "integer", "description": "Line number for breakpoint"},
                    "breakpoint_file": {"type": "string", "description": "Script file for breakpoint"},
                    
                    # Search/filter params
                    "type_filter": {"type": "string", "description": "Node type to filter by"},
                    "max_depth": {"type": "integer", "default": 10, "description": "Maximum tree depth to traverse"},
                    "include_internal": {"type": "boolean", "default": False, "description": "Include internal nodes"},
                    
                    # ADVANCED DIAGNOSTICS params (System Observatory features)
                    "script_path": {"type": "string", "description": "Path to script to analyze or toggle"},
                    "enabled": {"type": "boolean", "description": "Enable/disable script (for runtime.script.toggle)"},
                    "trace_duration": {"type": "number", "default": 1.0, "description": "How long to trace property changes (seconds)"},
                    "include_callstack": {"type": "boolean", "default": True, "description": "Include script callstack in trace"},
                    "compare_to_editor": {"type": "boolean", "default": False, "description": "Compare runtime values to editor values (for diagnose)"}
                },
                "required": ["op"]
            }
        }
    }
]
