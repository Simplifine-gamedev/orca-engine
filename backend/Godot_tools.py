


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
                            "scene.open", "scene.create", "scene.save_as", "scene.instantiate",
                            "scene.analyze", "scene.info", "scene.nodes.get_all",
                            "scene.nodes.find_by_type", "editor.selection.get",
                            "scene.bulk_configure", "scene.copy_configuration",
                            # Node CRUD & type
                            "node.create", "node.delete", "node.move",
                            "node.type.change", "node.type.set", "node.rename",
                            # Groups
                            "groups.add", "groups.remove", "groups.list",
                            # Props & methods
                            "node.props.get", "node.props.set_batch", "node.method.call",
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

                    # Node create/find
                    "type": {"type": "string"},
                    "name": {"type": "string"},

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

                    # Node property inspection (node.props.get) - filtering/pagination
                    "include": {"type": "array", "items": {"type": "string"}, "description": "Exact property names to include"},
                    "ensure": {"type": "array", "items": {"type": "string"}, "description": "Always-include property names (bypass limits)"},
                    "prefix": {"type": "string", "description": "Include properties beginning with this prefix"},
                    "offset": {"type": "integer", "default": 0, "description": "Skip first N matching properties"},
                    "editor_only": {"type": "boolean", "default": True, "description": "When false, include non-editor properties"},
                    "max_properties": {"type": "integer", "default": 50, "description": "Set -1 for no limit"},

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
                            "res.load_and_assign",
                            "import.set_options", "import.reimport",
                            "image.generate_or_edit", "image.save", "image.slice_spritesheet"
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    # Resources
                    "type": {"type": "string", "description": "Resource type (e.g., 'StandardMaterial3D', 'BoxMesh', 'SphereMesh')"},
                    "props": {"type": "object", "description": "Properties to set/modify. For BoxMesh use: {size: {x: 50, y: 0.2, z: 50}}. For SphereMesh: {radius: 1.0, radial_segments: 16, rings: 8}. For materials: {albedo_color: {r: 1, g: 0, b: 0, a: 1}, metallic: 0.5}"},
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
                    "image_id": {"type": "string"},
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
                            # InputMap
                            "inputmap.add_action", "inputmap.erase_action",
                            "inputmap.action_add_event", "inputmap.action_erase_event",
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

                    # InputMap
                    "action": {"type": "string"},
                    "event": {
                        "type": "object",
                        "description": "Serialized input event (e.g., {type:'key', scancode:'Key.SPACE', device:0})"
                    },

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
                    "search_mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid", "auto"], "default": "semantic"},

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
            "description": "Run/stop/status, error summaries/details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["game.start", "game.stop", "game.status", "errors.summary", "errors.details", "errors.test", "errors.debug"]},
                    "scene_path": {"type": "string"},
                    "clear_errors": {"type": "boolean", "default": True},

                    "include_warnings": {"type": "boolean", "default": True},
                    "file_filter": {"type": "string"},
                    "max_count": {"type": "integer", "default": 20},
                    "message_contains": {"type": "string"},
                    "group_duplicates": {"type": "boolean", "default": True}

                    # TEMPORARILY DISABLED - Screenshot parameters
                    # "filename": {"type": "string", "default": "screenshot_debug.png"},
                    # "target": {"type": "string", "enum": ["editor", "game", "both"], "default": "editor"},
                    # "return_base64": {"type": "boolean", "default": False}
                },
                "required": ["op"]
            }
        }
    }
]
