


# NEW set of tools :)
# CRITICAL: This array is used by long-running Cloud Run instances
# Any in-place mutations will corrupt it over time, causing "Missing required parameter: 'tools[0].type'" errors
# Always deep copy before passing to LiteLLM to prevent provider adapters from mutating the global
_godot_tools_template = [
    {
        "type": "function",
        "function": {
            "name": "project_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Manages project context, filesystem operations, asset library, and updates. Common operations: 'context.get' (analyze project), 'fs.read' (read files), 'fs.list' (list directories), 'assets.search' (find assets).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # Context
                            "context.get",
                            # Filesystem
                            "fs.list", "fs.read", "fs.read_with_context", "fs.write", "fs.write_lines", "fs.replace_string",
                            "fs.copy", "fs.move", "fs.delete",
                            "fs.mkdir", "fs.symlink", "fs.refresh",
                            # Project-level ops
                            "project.analyze_dir", "project.copy_dir", "project.update_refs",
                            # Asset library
                            "assets.search", "assets.install",
                            # Updates
                            "updates.check"
                        ],
                        "description": "**REQUIRED** operation type. Examples: 'context.get' (get project structure), 'fs.read' (read file), 'fs.read_with_context' (read file with relationships), 'fs.list' (list directory), 'fs.write' (write file), 'assets.search' (search assets). NEVER call project_manager without specifying op parameter."
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
                    "include_context": {"type": "boolean", "default": False, "description": "For fs.read: include relationship/edge information (signals emitted/received, dependencies, etc.)"},
                    "content": {"type": "string", "maxLength": 8000, "description": "New content for fs.write. IMPORTANT: Keep content under 8000 characters to prevent JSON corruption. For large scripts, make smaller incremental edits instead of full file replacements. Break complex features into multiple smaller operations."},
                    
                    # fs.write_lines (line range editing)
                    "start_line": {"type": "integer", "description": "Starting line number for fs.write_lines (1-based)"},
                    "end_line": {"type": "integer", "description": "Ending line number for fs.write_lines (1-based, inclusive)"},
                    "lines_content": {"type": "string", "maxLength": 5000, "description": "New content for the specified line range. Keep under 5000 characters. For large changes, break into multiple smaller line-range edits."},
                    
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
            "description": "REQUIRED: Always specify 'op' parameter. Manages scenes, nodes, properties, resources, collisions, signals, groups. Common operations: 'scene.open' (open scene), 'node.create' (create node), 'node.props.get' (get properties), 'scene.analyze' (analyze scene). IMPORTANT: Use 'node.fix_physics_body' to match CollisionShape2D size with Sprite2D texture size! Use 'node.shape.set' to directly set shape properties (size/radius/height).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # Scene ops
                            "scene.open", "scene.create", "scene.save_as", "scene.instantiate", "scene.instantiate_batch",
                            "scene.analyze", "scene.info", "scene.nodes.get_all",
                            "scene.nodes.find_by_type", "editor.selection.get",
                            "scene.bulk_configure", "scene.copy_configuration", "scene.validate_physics_bodies",
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
                            "node.assign_resource", "node.add_collision", "node.fix_physics_body", "node.shape.set",
                            # Signals & connections
                            "signals.list_node_signals", "signals.list_connections",
                            "signals.list_incoming_connections", "signals.connect",
                            "signals.disconnect", "signals.validate", "signals.open_dialog"
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
                    
                    # Physics body auto-fix (for node.fix_physics_body) - USE THIS when sprite and collision don't match!
                    "fix_mode": {"type": "string", "enum": ["auto", "collision_to_sprite", "sprite_to_collision", "align_only"], "default": "auto", "description": "IMPORTANT: Use node.fix_physics_body to match CollisionShape2D with Sprite2D! 'auto' (fix all issues), 'collision_to_sprite' (resize collision to match sprite), 'sprite_to_collision' (resize/create sprite to match collision), 'align_only' (just align positions). This automatically reads sprite texture size and sets collision shape to match!"},
                    "create_missing": {"type": "boolean", "default": True, "description": "Whether to create missing components (CollisionShape2D or Sprite2D)"},
                    
                    # Direct shape property setting (for node.shape.set)
                    "shape_property": {"type": "string", "enum": ["size", "radius", "height"], "description": "For node.shape.set: Which shape property to set. Use 'size' for RectangleShape2D (Vector2), 'radius' for CircleShape2D (float), 'radius'+'height' for CapsuleShape2D."},
                    "shape_value": {"description": "For node.shape.set: The value to set. For 'size' use {x: width, y: height}. For 'radius' use a number. For 'height' use a number."},

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
                    "include_args": {"type": "boolean", "default": False}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "script_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Scripts and classes: attach/detach/reload, class registry, and compilation checks. Common operations: 'script.attach' (attach script), 'script.detach' (detach script), 'classes.refresh' (refresh classes).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
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
            "description": "REQUIRED: Always specify 'op' parameter. Create/inspect/modify/assign resources, load & assign, import options and reimport, image ops, spritesheet slicing, and BATCH image creation. IMPORTANT: If 'image.generate_or_edit', 'image.create_isolated_object', or 'image.create_batch' includes 'path_to_save', images are automatically saved - do NOT call 'image.save' separately. Common operations: 'res.create' (create resource), 'res.inspect' (inspect resource), 'image.generate_or_edit' (single image), 'image.create_isolated_object' (single isolated object), 'image.create_batch' (multiple images in parallel - great for game assets like weapons, characters, items).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "res.create", "res.inspect", "res.modify", "res.assign",
                            "res.copy_from_template", "res.refresh",
                            "res.load_and_assign", "res.create_and_assign",
                            "import.set_options", "import.reimport",
                            "shader.clear_cache", "shader.force_recompile", "shader.debug_cache",
                            "image.generate_or_edit", "image.create_isolated_object", "image.save", "image.slice_spritesheet", "image.create_batch"
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},

                    # Resources
                    "type": {"type": "string", "description": "Resource type (e.g., 'StandardMaterial3D', 'BoxMesh', 'SphereMesh')"},
                    "resource_type": {"type": "string", "description": "Resource type for res.create_and_assign (e.g., 'StandardMaterial3D', 'BoxMesh', 'SphereMesh')"},
                    "props": {"type": "object", "description": "Properties to set/modify. For BoxMesh use: {size: {x: 50, y: 0.2, z: 50}}. For SphereMesh: {radius: 1.0, radial_segments: 16, rings: 8}. For materials: {albedo_color: {r: 1, g: 0, b: 0, a: 1}, metallic: 0.5, albedo_texture: {path: 'res://texture.png'}} or {albedo_texture: 'res://texture.png'}"},
                    "properties": {"type": "object", "description": "Properties for res.create_and_assign - same format as props"},
                    "save_path": {"type": "string", "description": "CRITICAL: Path to save resource file (e.g., 'res://resources/my_mesh.tres'). ALWAYS provide this to persist resources to disk. Without save_path, resources are only created in memory and will be lost."},
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
                    "description": {"type": "string", "maxLength": 2000, "description": "Text description of the image to generate or edit. Keep descriptions concise (under 2000 chars) for reliable JSON parsing. Focus on key visual elements and style."},
                    "images": {"type": "array", "items": {"type": "string"}, "description": "Array of image IDs to edit (leave empty to generate new image). Use image IDs from previous conversation like 'generated_abc123' or 'edited_def456'"},
                    "style": {"type": "string", "description": "Art style for the image (e.g., 'pixel art', 'photorealistic', '3D render')"},
                    "image_type": {"type": "string", "enum": ["general", "texture", "sprite", "icon", "background"], "default": "general", "description": "Type of image being generated: 'texture' (seamless tileable material), 'sprite' (character/object), 'icon' (UI symbol), 'background' (scene backdrop), 'general' (any image)"},
                    "size": {"type": "string"},
                    "exact_size": {"type": "string"},
                    "tile_size": {"type": "string"},
                    "grid": {"type": "string"},
                    "resize_filter": {"type": "string", "enum": ["nearest", "bilinear", "bicubic", "lanczos"], "default": "lanczos"},
                    "path_to_save": {"type": "string", "description": "OPTIONAL: Path to save the generated/edited image (e.g., 'res://art/texture.png'). If provided, the image will be automatically saved. If not provided, use the separate 'image.save' operation later."},

                    # Isolated object creation (for icons, symbols, etc.)
                    "object_description": {"type": "string", "maxLength": 1500, "description": "Description of the standalone object to create (icons, symbols, characters, etc.). For image.create_isolated_object operation only. Will automatically add instructions for white background and transparent extraction."},
                    "input_image_path": {"type": "string", "description": "Path to input image for editing existing drawings into isolated objects. Used with image.create_isolated_object when editing existing images. Can be a file path or image ID."},
                    "input_image_id": {"type": "string", "description": "Reference to an image from conversation context. Accepts exact IDs like 'img_123_foo', generated IDs like 'generated_ab12cd34', or numeric forms '#1', '[1]', '1' referring to the Nth most recent image."},
                    "white_threshold": {"type": "integer", "default": 240, "minimum": 180, "maximum": 255, "description": "RGB threshold for white background detection (180-255). Higher values = more strict white detection. Used with image.create_isolated_object."},
                    "target_resolution": {"type": "integer", "default": 128, "description": "Target resolution for the isolated object (e.g., 8, 16, 32, 64, 128, 256, 512, 1024). Default is 128x128 for optimal icon/sprite size. Use -1 for original size. Maintains aspect ratio when resizing. Used with image.create_isolated_object."},
                    # For isolated objects specifically
                    "object_type": {"type": "string", "enum": ["sprite", "icon"], "default": "sprite", "description": "Type of isolated object: 'sprite' (game characters/objects), 'icon' (UI symbols/buttons). Used with image.create_isolated_object only."},

                    # Image save
                    "image_id": {"type": "string", "description": "ID of the image from conversation (e.g., 'generated_abc123', 'edited_def456')"},
                    "path": {"type": "string", "description": "Path where to save the image in the project (e.g., 'res://textures/floor_texture.png')"},
                    "format": {"type": "string", "enum": ["png", "jpg", "jpeg"], "default": "png"},
                    "target_resolution": {"type": "integer", "description": "Target resolution for saved image (e.g., 8, 16, 32, 64, 128, 256, 512, 1024, 2048). Use -1 or omit for original size. Maintains aspect ratio when resizing. NOTE: Only use this operation if the image generation did NOT include path_to_save - otherwise the image is already saved automatically."},

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
                    "normalize_to": {"type": "string"},

                    # Batch image creation
                    "image_requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string", "maxLength": 1500, "description": "Description of the image to create"},
                                "method": {"type": "string", "enum": ["general", "isolated_object", "isolated"], "default": "general", "description": "Creation method: 'general' for textures/backgrounds/sprites, 'isolated_object'/'isolated' for transparent objects/icons"},
                                "style": {"type": "string", "description": "Art style override for this specific image (optional, uses global_style if not specified)"},
                                "image_type": {"type": "string", "enum": ["general", "texture", "sprite", "icon", "background"], "default": "general", "description": "Type hint for image generation"},
                                "size": {"type": "string", "description": "Size override for this image (optional, uses global_size if not specified)"},
                                "exact_size": {"type": "string", "description": "Exact pixel size for this image"},
                                "input_image_id": {"type": "string", "description": "Optional image ID to edit instead of creating new"},
                                "object_type": {"type": "string", "enum": ["sprite", "icon"], "default": "sprite", "description": "For isolated objects: 'sprite' or 'icon'"},
                                "target_resolution": {"type": "integer", "default": 128, "description": "For isolated objects: target resolution"},
                                "white_threshold": {"type": "integer", "default": 240, "description": "For isolated objects: background removal threshold"},
                                "path_to_save": {"type": "string", "description": "Optional specific path to save this image"},
                                "tile_size": {"type": "string", "description": "For spritesheets: tile size"},
                                "grid": {"type": "string", "description": "For spritesheets: grid layout"},
                                "resize_filter": {"type": "string", "enum": ["nearest", "bilinear", "bicubic", "lanczos"], "default": "lanczos"}
                            },
                            "required": ["description", "method"]
                        },
                        "description": "Array of image creation requests. Each request specifies 'method' ('general' or 'isolated_object') and 'description'. Example: [{method: 'isolated_object', description: 'assault rifle icon', object_type: 'icon'}, {method: 'general', description: 'desert battlefield background', image_type: 'background'}]"
                    },
                    "global_style": {"type": "string", "description": "Default art style applied to all images in batch (e.g., 'pixel art', 'low poly', 'photorealistic'). Individual requests can override this."},
                    "global_size": {"type": "string", "default": "1024x1024", "description": "Default size for all images in batch. Individual requests can override this."},
                    "max_parallel": {"type": "integer", "default": 4, "minimum": 1, "maximum": 8, "description": "Maximum number of images to process in parallel (1-8, default 4)"},
                    "timeout_per_image": {"type": "integer", "default": 120, "minimum": 30, "maximum": 300, "description": "Timeout per image in seconds (30-300, default 120)"},
                    "save_base_path": {"type": "string", "description": "Base directory path for auto-saving batch images. If specified, images without individual path_to_save will be auto-saved here with generated names."}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "settings_manager",
            "description": "REQUIRED: Always specify 'op' parameter. ProjectSettings, InputMap actions, Autoloads (singletons), and layer/mask names. Common operations: 'project_settings.get' (get setting), 'project_settings.set' (set setting), 'autoload.add' (add autoload).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
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
            "description": "REQUIRED: Always specify 'op' parameter. Create and edit animations on AnimationPlayer: animations, tracks, and keyframes. Common operations: 'animation.create' (create animation), 'track.add' (add track), 'key.insert' (insert keyframe).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
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
            "description": "REQUIRED: Always specify 'op' parameter. Search across project and Godot docs with ENHANCED CONTEXT. Operations: 'project.search' (search files with context), 'docs.search' (search Godot docs), 'scene.composition_tree' (analyze scene structure).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {"type": "string", "enum": ["project.search", "docs.search", "scene.composition_tree"]},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "include_graph": {"type": "boolean", "default": True},
                    "include_context": {"type": "boolean", "default": True, "description": "Include enhanced context with signals, dependencies, and relationships - RECOMMENDED for world-class results"},
                    "modality_filter": {"type": "string", "enum": ["text", "image", "audio"]},
                    "project_root": {"type": "string"},
                    "project_id": {"type": "string"},
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
                "required": ["op"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Explore the Godot project graph (scene↔script relationships, signals, resources). Use this before editing to understand connected files.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["graph.neighbors", "graph.walk"],
                        "description": "Graph operations. 'graph.neighbors' inspects a specific file; 'graph.walk' traverses multiple hops automatically."
                    },
                    "file_path": {"type": "string", "description": "Primary file to explore (e.g., 'res://player.gd')."},
                    "file_paths": {"type": "array", "items": {"type": "string"}, "description": "Optional list of additional files to include."},
                    "start_file": {"type": "string", "description": "graph.walk: single starting file/scene."},
                    "start_files": {"type": "array", "items": {"type": "string"}, "description": "graph.walk: list of starting files/scenes."},
                    "project_root": {"type": "string", "description": "Project root path (used to derive project_id if not provided)."},
                    "project_id": {"type": "string", "description": "Project identifier (MD5 hash of project root)."},
                    "depth": {"type": "integer", "default": 1, "minimum": 1, "maximum": 3, "description": "Neighbor expansion depth (currently limited to 1-3)."},
                    "edge_types": {"type": "array", "items": {"type": "string"}, "description": "Filter to specific relationship types (e.g., ['attached_script','scene_ref'])."},
                    "max_nodes": {"type": "integer", "default": 12, "minimum": 1, "maximum": 50, "description": "Maximum nodes to return per file."},
                    "max_edges": {"type": "integer", "default": 24, "minimum": 1, "maximum": 200, "description": "Maximum edges to return per file."},
                    "graph_preview": {"type": "boolean", "default": True, "description": "Return trimmed preview for UI display."},
                    "include_summary": {"type": "boolean", "default": True, "description": "Include aggregate graph statistics."},
                    "include_raw": {"type": "boolean", "default": False, "description": "Return the full raw graph context (can be large)."}
                },
                "required": ["op"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "runtime_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Run/stop/status, error summaries/details. TEMP: screenshots disabled.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # TEMP: Removed 'screenshot.capture' from enum to disable screenshot tool
                    "op": {"type": "string", "enum": ["game.start", "game.stop", "game.status", "errors.summary", "errors.details", "errors.test", "errors.debug", "console.get_output", "input.test_action", "input.test_key"]},
                    "scene_path": {"type": "string"},
                    "clear_errors": {"type": "boolean", "default": True},

                    "include_warnings": {
                        "type": "boolean",
                        "default": True,
                        "description": "errors.* ops: include warnings alongside errors when summarizing logs."
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "errors.* ops: only include entries whose file path contains this substring (e.g., 'player.gd')."
                    },
                    "max_count": {
                        "type": "integer",
                        "default": 20,
                        "description": "errors.* ops: limit number of error instances to return after filtering."
                    },
                    "message_contains": {
                        "type": "string",
                        "description": "errors.* ops: only include messages that contain this substring."
                    },
                    "group_duplicates": {
                        "type": "boolean",
                        "default": True,
                        "description": "errors.summary: when true, collapse identical errors into a single entry with counts."
                    },

                    # Screenshot parameters (TEMP disabled)
                    # "filename": {"type": "string", "default": "screenshot_debug.png"},
                    # "target": {"type": "string", "enum": ["editor", "game", "both"], "default": "game"},
                    # "return_base64": {"type": "boolean", "default": True},
                    
                    # Console output parameters
                    "lookback_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "description": "errors.* ops: only include runtime errors from the last N seconds (0 = all history)."
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["all", "print", "error", "warning"],
                        "default": "all",
                        "description": "console.get_output: which messages to return (filters by Godot log category)."
                    },
                    "max_lines": {
                        "type": "integer",
                        "default": 50,
                        "description": "console.get_output: maximum number of filtered game log lines to return."
                    },
                    "since_timestamp": {
                        "type": "integer",
                        "description": "console.get_output: reserved for future incremental log streaming (currently ignored)."
                    },
                    
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
            "name": "terminal_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Execute standard CLI/terminal commands on the local machine. Supports common command line operations like file management, version control, text processing, and system utilities. All operations run on the local machine (frontend). Common operations: 'execute' (run command), 'history' (command history), 'clear' (clear output).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "execute", "history", "clear", "status", "pwd", "cd", "allowed_commands"
                        ],
                        "description": "**REQUIRED** terminal operation. 'execute' runs commands, 'history' shows command history, 'clear' clears output, 'status' shows terminal status, 'pwd' gets current directory, 'cd' changes directory, 'allowed_commands' lists allowed commands"
                    },
                    "dry_run": {"type": "boolean", "default": False, "description": "If true, don't actually execute the command, just validate it"},
                    
                    # Command execution
                    "command": {"type": "string", "description": "CLI command to execute (for 'execute' operation). Examples: 'ls -la' (list files), 'grep -r \"pattern\" .' (search text), 'git status' (version control), 'pwd' (current directory), 'find . -name \"*.txt\"' (find files), 'cat filename.txt' (view file)"},
                    "working_directory": {"type": "string", "description": "Working directory for command execution (defaults to project root)"},
                    "timeout": {"type": "integer", "default": 30, "description": "Command timeout in seconds (max 300s for safety)"},
                    "capture_output": {"type": "boolean", "default": True, "description": "Whether to capture and return command output"},
                    "shell": {"type": "boolean", "default": False, "description": "Whether to execute command through shell (use with caution)"},
                    "env_vars": {"type": "object", "description": "Additional environment variables to set for command execution"},
                    
                    # Directory operations  
                    "path": {"type": "string", "description": "Directory path for 'cd' operation"},
                    
                    # History operations
                    "max_history": {"type": "integer", "default": 50, "description": "Maximum number of history entries to return"},
                    "filter_pattern": {"type": "string", "description": "Filter history entries by pattern (regex supported)"}
                },
                "required": ["op"]
            }
        }
    },
    
    {
        "type": "function",
        "function": {
            "name": "runtime_inspector",
            "description": "REQUIRED: Always specify 'op' parameter. Inspect and modify runtime node properties, materials, shaders, environment settings during play. Common operations: 'runtime.node.get_props' (get node properties), 'runtime.node.set_prop' (set property), 'runtime.material.get' (get material).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            # Basic runtime inspection
                            "runtime.node.get_props", "runtime.node.set_prop", 
                            "runtime.node.get_tree", "runtime.node.find_by_type",
                            # Help/info
                            "runtime.debug.info", "help"
                        ],
                        "description": "Runtime inspection operation"
                    },
                    
                    # Common parameters
                    "node_path": {"type": "string", "description": "Path to node in remote/runtime scene tree"},
                    "property": {"type": "string", "description": "Property name to get/set"},
                    "value": {"description": "Value to set for property"},
                    
                    
                    
                    # Search/filter params
                    "type_filter": {"type": "string", "description": "Node type to filter by"},
                    "max_depth": {"type": "integer", "default": 10, "description": "Maximum tree depth to traverse"},
                    "include_internal": {"type": "boolean", "default": False, "description": "Include internal nodes"},
                    
                },
                "required": ["op"]
            }
        }
    },
    
    {
        "type": "function",
        "function": {
            "name": "2d_animation_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Create and manage 2D sprite animations using the AI animation server. Generate complete animation sets (idle, walk, attack, etc.) from text descriptions with optional style reference images. Animations are stored in Supabase and return sprite sheets, transparent GIFs, and individual frames. Common operations: 'create' (create animation project), 'status' (check progress), 'edit' (modify animations), 'list_jobs' (list recent projects).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "list_my_animations",  # List all animations with permanent refs (P1, P1.1, etc.)
                            "create",              # Create new animation project
                            "status",              # Check job/project status
                            "edit",                # Edit existing animation
                            "list_jobs",           # List recent animation projects
                            "add_branch",          # Add animation to existing project
                            "download"             # Download animation files to project
                        ],
                        "description": "Animation operations. 'list_my_animations' shows all animations with PERMANENT refs (P1=project, P1.1=animation). 'create' generates new project. 'download' fetches files using refs. 'edit' modifies animations. 'add_branch' adds animation to project. Refs like P10 and P10.1 are stable - same ref always means same animation."
                    },
                    
                    # CREATE operation parameters
                    "user_request": {"type": "string", "maxLength": 2000, "description": "Natural language description of the animations to create. Example: 'Create a pixel-art knight with idle, walk, and attack animations'. Keep descriptions focused and under 2000 chars for reliable processing."},
                    "reference_image_ids": {"type": "array", "items": {"type": "string"}, "description": "OPTIONAL: Array of image IDs from conversation to use as style reference. The system will check if the image is suitable (isolated object on white background). If suitable, it uses the image directly. If not (complex background, multiple objects), it auto-generates an isolated version. Can reference numbered images like '#1', named IDs like 'generated_abc123', or recent images."},
                    "reference_description": {"type": "string", "maxLength": 1000, "description": "OPTIONAL: Describe what the character/object is (e.g., 'a pixel-art knight', 'blue robot'). Used for quality checking and if regeneration is needed. If the reference image isn't suitable (not isolated on white bg), this description guides the auto-generated replacement."},
                    "target_resolution": {"type": "string", "enum": ["8x8", "64x64", "128x128", "256x256", "512x512"], "default": "128x128", "description": "Frame resolution for sprite animations. '64x64' for retro pixel-art, '128x128' for balanced quality (default), '256x256' for HD sprites, '512x512' for high-detail sprites."},
                    "execute_immediately": {"type": "boolean", "default": True, "description": "Whether to start animation generation immediately (recommended). If false, only creates the graph plan without generating videos."},
                    "upload_to_supabase": {"type": "boolean", "default": True, "description": "Whether to upload results to Supabase for persistent storage (recommended). Required for 'edit' operation to work later."},
                    
                    # AUTO-EXPORT parameters (optional - if specified, animations are automatically saved when complete)
                    # STANDARD FILE NAMING: <name>_sheet.png, <name>_frames.tres, <name>.tscn, <name>.gd
                    "export_destination": {"type": "string", "description": "OPTIONAL: Folder path to auto-save animations. E.g., 'res://sprites/knight/'. Creates folder if needed. If specified, exports automatically when complete."},
                    "export_resolution": {"type": "integer", "default": 64, "minimum": 32, "maximum": 512, "description": "Sprite resolution. 64 for pixel art (default), 128 for balanced, 256 for HD."},
                    "export_format": {"type": "string", "enum": ["sprite_sheet", "frames", "gif", "godot_template"], "default": "godot_template", "description": "ALWAYS use 'godot_template' - creates ready-to-use Godot files. Creates: <name>_sheet.png, <name>_frames.tres, <name>.tscn, <name>.gd"},
                    "export_template_type": {"type": "string", "enum": ["player", "character", "object", "effect", "simple"], "default": "character", "description": "Scene type: 'player' (CharacterBody2D with keyboard controls), 'character' (CharacterBody2D, no controls - for NPCs/enemies), 'object' (StaticBody2D with collision - for fire pits, chests, etc.), 'effect' (Node2D - for VFX like explosions), 'simple' (just AnimatedSprite2D, no physics)."},
                    "export_resource_name": {"type": "string", "description": "REQUIRED for godot_template. Base name for files. E.g., 'knight' creates: knight_sheet.png, knight_frames.tres, knight.tscn, knight.gd"},
                    "export_fps": {"type": "integer", "default": 10, "minimum": 1, "maximum": 60, "description": "Animation playback FPS. 10 for most animations, lower for pixel art."},
                    
                    # STATUS operation parameters
                    "job_id": {"type": "string", "description": "Job ID returned from 'create' operation. Used to check generation progress."},
                    
                    # EDIT operation parameters
                    "project_id": {"type": "string", "description": "Project ID (UUID). Can also use animation_ref instead for easier reference."},
                    "animation_ref": {"type": "string", "description": "Animation or project reference (e.g., 'P1.1', 'P10.2' for animation, 'P1' for whole project). Permanent refs from list_my_animations."},
                    "edit_request": {"type": "string", "maxLength": 1500, "description": "What to change. Examples: 'make it faster', 'add 4 more frames', 'make the attack swing wider'."},
                    "auto_regenerate": {"type": "boolean", "default": True, "description": "Regenerate animation immediately (recommended). If false, only updates prompts."},
                    
                    # ADD_BRANCH operation parameters
                    "branch_request": {"type": "string", "maxLength": 1500, "description": "Description of new animation to add. Example: 'add a death animation' or 'add running animation'. Use with project_id='P1' to specify which project."},
                    "project_ref": {"type": "string", "description": "Project reference from list_my_animations (e.g., 'P1', 'P2'). Easier than using raw project_id UUID."},
                    
                    # LIST_JOBS parameters
                    "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100, "description": "Maximum number of recent jobs to return for 'list_jobs' operation."},
                    
                    # LIST_MY_ANIMATIONS parameters  
                    "refresh": {"type": "boolean", "default": True, "description": "For 'list_my_animations': whether to fetch fresh data from Supabase (recommended). If false, uses cached data."},
                    "user_id": {"type": "string", "description": "User ID to fetch animations for (optional, defaults to current user)."},
                    
                    # DOWNLOAD operation parameters
                    "animation_number": {"type": "string", "description": "Reference to download. Use 'P1.1', 'P10.2' for single animations, or 'P1', 'P10' for entire projects (all animations). Refs are permanent - get them from list_my_animations."},
                    "animation_url": {"type": "string", "description": "Direct Supabase URL to download. Use this if you have the URL directly instead of a reference."},
                    "destination_path": {"type": "string", "description": "Local project path. For single files: 'res://sprites/hero_idle.png'. For projects: 'res://sprites/knight/' (directory)."},
                    "file_type": {"type": "string", "enum": ["sprite_sheet", "animated_gif", "thumbnail", "frames", "all"], "default": "sprite_sheet", "description": "What to download: 'sprite_sheet' (PNG), 'animated_gif', 'thumbnail' (64x64), 'frames' (individual PNGs), 'all'."}
                },
                "required": ["op"]
            }
        }
    }
]

# CRITICAL PROTECTION: Create the public godot_tools from a deep copy
# This prevents any accidental mutations during module initialization
# At runtime, app.py MUST deepcopy again before passing to LiteLLM
import copy
godot_tools = copy.deepcopy(_godot_tools_template)

# Diagnostic: Validate tools structure at module load time
# This catches any corruption during initialization
if not godot_tools or not isinstance(godot_tools, list) or len(godot_tools) == 0:
    raise RuntimeError("CRITICAL: godot_tools is empty or invalid at module load!")
if "type" not in godot_tools[0] or godot_tools[0]["type"] != "function":
    raise RuntimeError(f"CRITICAL: godot_tools[0] is malformed at module load: {godot_tools[0]}")

print(f"✅ Godot tools loaded successfully: {len(godot_tools)} tools registered")
print(f"✅ Tools validation: First tool type='{godot_tools[0].get('type')}', name='{godot_tools[0].get('function', {}).get('name')}'")
print("⚠️  WARNING: Always deepcopy godot_tools before passing to LiteLLM to prevent corruption!")
