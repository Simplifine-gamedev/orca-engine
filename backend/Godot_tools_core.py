# NEW minimal tools set: Search, Read/Write, CLI, Images, Animations
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
            "name": "2d_animation_batch_manager",
            "description": "REQUIRED: Always specify 'op' parameter. Create and manage BATCH 2D sprite animation jobs in PARALLEL. Mirrors 2d_animation_manager 'create' semantics per item, but processes multiple characters/objects at once. Returns a batch job with child job IDs for tracking.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "create_batch",      # Create multiple animation jobs in parallel
                            "status",            # Aggregate status for a batch or explicit job_ids
                            "list_jobs",         # List recent batch jobs (alias of single jobs)
                            "download"           # Download results for all jobs in a batch (optional)
                        ]
                    },
                    "dry_run": {"type": "boolean", "default": False},
                    
                    # CREATE_BATCH parameters
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "user_request": {"type": "string", "maxLength": 2000, "description": "Natural language description per item, e.g., 'pixel-art clownfish with idle and swim'"},
                                "animation_preset": {"type": "string", "enum": ["auto", "rpg_topdown", "platformer", "custom"], "default": "auto"},
                                "reference_image_ids": {"type": "array", "items": {"type": "string"}},
                                "reference_description": {"type": "string", "maxLength": 1000},
                                "target_resolution": {"type": "string", "enum": ["8x8", "64x64", "128x128", "256x256", "512x512"], "default": "128x128"},
                                
                                # Optional auto-export for each item
                                "export_destination": {"type": "string"},
                                "export_resolution": {"type": "integer", "default": 128, "minimum": 32, "maximum": 512},
                                "export_format": {"type": "string", "enum": ["sprite_sheet", "frames", "gif", "godot_template"], "default": "godot_template"},
                                "export_template_type": {"type": "string", "enum": ["player", "character", "object", "effect", "simple", "rpg_character"], "default": "character"},
                                "export_resource_name": {"type": "string"},
                                "export_fps": {"type": "integer", "default": 10, "minimum": 1, "maximum": 60}
                            },
                            "required": ["user_request"]
                        },
                        "description": "Array of per-item animation creation requests. Each item mirrors 2d_animation_manager 'create' parameters."
                    },
                    "max_parallel": {"type": "integer", "default": 4, "minimum": 1, "maximum": 8, "description": "Maximum number of items to process in parallel (1-8, default 4)"},
                    "timeout_per_job": {"type": "integer", "default": 120, "minimum": 30, "maximum": 600, "description": "Timeout per single job create call (seconds)"},
                    
                    # STATUS parameters
                    "batch_job_id": {"type": "string", "description": "Batch ID returned from 'create_batch'. If provided, aggregates child job statuses."},
                    "job_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit list of job IDs to aggregate status for"},
                    
                    # DOWNLOAD parameters (optional, not required to implement fully at start)
                    "destination_base_path": {"type": "string", "description": "Base directory to save results for all jobs when using 'download'"},
                    "file_type": {"type": "string", "enum": ["sprite_sheet", "animated_gif", "thumbnail", "frames", "all"], "default": "sprite_sheet"}
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
                    "command": {"type": "string", "description": "CLI command to execute (for 'execute' operation). Examples: 'ls -la' (list files), 'grep -r \"pattern\" .' (search text), 'git status' (version control), 'pwd' (current directory), 'find . -name \"*.txt\"' (find files)"},
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
                    "animation_preset": {"type": "string", "enum": ["auto", "rpg_topdown", "platformer", "custom"], "default": "auto", "description": "Animation preset to use. 'rpg_topdown' = optimized for top-down RPGs (generates idle, walk_side, walk_up; auto-mirrors walk_left; skips walk_down which doesn't work well). 'platformer' = side-view animations. 'auto' = AI decides based on request. 'custom' = no preset, AI designs from scratch."},
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
                    "export_template_type": {"type": "string", "enum": ["player", "character", "object", "effect", "simple", "rpg_character"], "default": "character", "description": "Scene type: 'player' (CharacterBody2D with keyboard controls), 'character' (CharacterBody2D, no controls - for NPCs/enemies), 'object' (StaticBody2D with collision - for fire pits, chests, etc.), 'effect' (Node2D - for VFX like explosions), 'simple' (just AnimatedSprite2D, no physics), 'rpg_character' (Top-down RPG character with 8-dir movement, auto-mirroring left from right)."},
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
    },
    {
        "type": "function",
        "function": {
            "name": "explore_codebase",
            "description": """Launch the Explorer Agent to deeply investigate a topic in the Godot project.

The Explorer will:
1. Perform multiple semantic and keyword searches
2. Read and analyze relevant files  
3. Follow references between files using the project graph
4. Compile findings with FULL CITATIONS (file paths + line numbers)

Use this tool when you need thorough investigation of:
- How a feature is implemented
- Where specific functionality exists
- Connections between different parts of the codebase
- Understanding complex systems or patterns

The Explorer returns a comprehensive report with cited findings. This is a streaming operation - you'll see the Explorer's progress as it works.""",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {
                        "type": "string",
                        "maxLength": 1000,
                        "description": "The specific question or topic to explore. Be as specific as possible for best results. Examples: 'How does the player movement system work?', 'Where is the inventory system implemented?', 'What signals does the GameManager emit?'"
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of directories or file patterns to focus the search on. Example: ['res://scripts/player/', 'res://systems/']"
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["quick", "normal", "thorough"],
                        "default": "normal",
                        "description": "'quick' (3-5 tool calls, fast overview), 'normal' (5-10 tool calls, balanced), 'thorough' (10-15 tool calls, deep investigation)"
                    }
                },
                "required": ["question"]
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

print(f"✅ Godot minimal tools loaded: {len(godot_tools)} tools registered")
print(f"✅ Tools validation: First tool type='{godot_tools[0].get('type')}', name='{godot_tools[0].get('function', {}).get('name')}'")
print("⚠️  WARNING: Always deepcopy godot_tools before passing to LiteLLM to prevent corruption!")


