# --- Individual Tool Definitions (Original 22 Tools) ---
godot_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_project_context",
            "description": "Get comprehensive project structure and context. ALWAYS use this FIRST before creating scenes, nodes, or scripts to understand what already exists in the project. This prevents duplicates and naming conflicts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["structure", "hierarchy", "find_scenes", "patterns"],
                        "description": "Operation type: 'structure' for full project overview, 'hierarchy' for specific scene details, 'find_scenes' to check existing scenes, 'patterns' to understand project conventions"
                    },
                    "project_root": {
                        "type": "string",
                        "description": "Project root path (optional, uses current project by default)"
                    },
                    "scene_path": {
                        "type": "string",
                        "description": "Scene path for 'hierarchy' operation"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern for 'find_scenes' operation"
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed analysis (default true)"
                    }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Get information about the current scene including root node and structure",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_node_type",
            "description": "Safely change a node's type by creating a replacement and reparenting children",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to node to replace"},
                    "new_type": {"type": "string", "description": "New node type (e.g., 'Node3D')"},
                    "preserve_children": {"type": "boolean", "default": True},
                    "strategy": {"type": "string", "enum": ["wrap_root", "swap"], "default": "wrap_root"}
                },
                "required": ["path", "new_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_resource",
            "description": "Create a Resource (e.g., BoxMesh, StandardMaterial3D) with optional properties and optional save path",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "properties": {"type": "object"},
                    "save_path": {"type": "string"}
                },
                "required": ["type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_resource_to_node_property",
            "description": "Assign a resource (by path, RID from create_resource, or inline spec) to a node property",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "property": {"type": "string"},
                    "resource": {}
                },
                "required": ["path", "property", "resource"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_scene_with_root",
            "description": "Create a new scene with a specific root type and save it; optionally attach current scene under it",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_root_type": {"type": "string"},
                    "new_scene_path": {"type": "string"},
                    "include_current_as_child": {"type": "boolean", "default": False}
                },
                "required": ["new_root_type", "new_scene_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory within the project",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file within the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move/rename a file within the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False}
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file within the project",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_symlink",
            "description": "Create a symlink within the project (platform dependent)",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "link_path": {"type": "string"}
                },
                "required": ["target", "link_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_filesystem",
            "description": "Refresh the editor's file system view",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "universal_resource_manager",
            "description": "Smart resource operations: create, inspect, modify, copy_from_template with type-aware handling",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["create", "inspect", "modify", "assign", "copy_from_template"]},
                    "type": {"type": "string"},
                    "target": {"type": "string"},
                    "properties": {"type": "object"},
                    "source_template": {"type": "string"},
                    "path": {"type": "string"},
                    "property": {"type": "string"},
                    "resource": {"type": "object"},
                    "save_path": {"type": "string"}
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "universal_scene_manager",
            "description": "Smart scene operations: analyze, bulk_configure, copy_configuration with validation",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["analyze", "bulk_configure", "copy_configuration"]},
                    "scope": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    "transformations": {"type": "object"},
                    "validation": {"type": "boolean", "default": True},
                    "source": {"type": "string"}
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "universal_project_manager",
            "description": "Smart project operations: analyze_directory, copy_directory, update_references with dependency awareness",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["analyze_directory", "copy_directory", "update_references"]},
                    "target_path": {"type": "string"},
                    "source_addon": {"type": "string"},
                    "target_addon": {"type": "string"},
                    "old_path": {"type": "string"},
                    "new_path": {"type": "string"},
                    "file_patterns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_nodes",
            "description": "Get all nodes in the current scene with their information",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_nodes_by_type",
            "description": "Search for nodes by their type (e.g., 'Node2D', 'Button', 'CharacterBody2D')",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The node type to search for"
                    }
                },
                "required": ["type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_editor_selection",
            "description": "Get currently selected nodes in the editor",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_node_properties",
            "description": "Get properties of a specific node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_node",
            "description": "Create a new node in the scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of node to create (e.g., 'Node2D', 'Button', 'CharacterBody2D')"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the new node"
                    },
                    "parent": {
                        "type": "string",
                        "description": "Parent node path (optional)"
                    }
                },
                "required": ["type", "name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_node",
            "description": "Delete a node from the scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node to delete"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_set_node_properties",
            "description": "Apply multiple property changes then optionally save once.",
            "parameters": {
                "type": "object",
                "properties": {
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
                    "save_after": {
                        "type": "boolean",
                        "description": "Save scene once after all operations are applied.",
                        "default": True
                    }
                },
                "required": ["operations"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_node",
            "description": "Move a node to a different parent",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node to move"
                    },
                    "new_parent": {
                        "type": "string",
                        "description": "Path to the new parent node"
                    }
                },
                "required": ["path", "new_parent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_node_method",
            "description": "Call a method on a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    },
                    "method": {
                        "type": "string",
                        "description": "Method name to call"
                    },
                    "method_args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments for the method call"
                    }
                },
                "required": ["path", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_classes",
            "description": "Get list of all available node classes in Godot",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_node_script",
            "description": "Get the script attached to a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "attach_script",
            "description": "Attach a script to a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    },
                    "script_path": {
                        "type": "string",
                        "description": "Path to the script file"
                    }
                },
                "required": ["path", "script_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detach_script",
            "description": "Remove script from a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the node"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reload_script",
            "description": "Reload a script to refresh class_name registration",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_path": {"type": "string"},
                    "path": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_global_classes",
            "description": "Force refresh of global class registrations",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_custom_classes",
            "description": "List custom global classes with class_name",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_node_type",
            "description": "Set node type via script or class replacement",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "type_name": {"type": "string"},
                    "script_path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_scene",
            "description": "Manage scene operations (open, create, save, instantiate)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["open", "create_new", "save_as", "instantiate"],
                        "description": "Scene operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Scene file path"
                    },
                    "parent_node": {
                        "type": "string",
                        "description": "Parent node path for instantiate operations"
                    },
                    "instance_name": {
                        "type": "string",
                        "description": "Name for the instantiated scene (optional)"
                    }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_and_assign_resource",
            "description": "Load a resource (mesh, texture, material, etc.) from file and assign it to a node property",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_path": {
                        "type": "string",
                        "description": "Path to the resource file (e.g., 'res://models/sword.glb', 'res://textures/wood.png')"
                    },
                    "node_path": {
                        "type": "string",
                        "description": "Path to the target node"
                    },
                    "property": {
                        "type": "string",
                        "description": "Property name to assign the resource to (e.g., 'mesh', 'texture', 'material')"
                    }
                },
                "required": ["resource_path", "node_path", "property"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_collision_shape",
            "description": "Add a collision shape to a physics body node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_path": {
                        "type": "string",
                        "description": "Path to the physics body node"
                    },
                    "shape_type": {
                        "type": "string",
                        "enum": ["rectangle", "circle", "capsule"],
                        "description": "Type of collision shape to create"
                    }
                },
                "required": ["node_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_project_files",
            "description": "List files and directories in the current directory (like ls/dir command). Shows immediate contents only unless recursive=true is explicitly needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {
                        "type": "string",
                        "description": "Directory to list (default: res:// root). Use this to navigate into subdirectories like 'res://scripts' or 'res://assets'."
                    },
                    "filter": {
                        "type": "string",
                        "description": "File filter (e.g., '*.gd', '*.tscn')"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list ALL files in the entire project tree (use sparingly, only when you need a complete overview). Default false shows only current directory contents.",
                        "default": False
                    },
                    "full_paths": {
                        "type": "boolean",
                        "description": "If true, return full paths (recommended for navigation)",
                        "default": True
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content. Optional start_line/end_line to fetch a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path to read" },
                    "start_line": { "type": "integer", "description": "Starting line (1-indexed)" },
                    "end_line": { "type": "integer", "description": "Ending line (inclusive)" }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edit",
            "description": "Apply AI-powered edits to a file. Supports partial edits by line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Description of the edit to apply"
                    },
                    "lines": {
                        "type": "string",
                        "enum": ["all", "range"],
                        "description": "Edit scope: 'all' for whole file (default), or 'range' to edit only a specific set of lines",
                        "default": "all"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "When lines='range', the 1-based start line (inclusive)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "When lines='range', the 1-based end line (inclusive)"
                    }
                },
                "required": ["path", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_compilation_errors",
            "description": "Check for errors in the project. Can check script compilation errors or all output panel errors (runtime errors, warnings, shader errors, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Script file path to check (only used when check_mode='scripts' and check_all=false)"
                    },
                    "check_all": {
                        "type": "boolean",
                        "description": "If true, check all script files in the project instead of a specific file (only for check_mode='scripts')",
                        "default": False
                    },
                    "check_mode": {
                        "type": "string",
                        "enum": ["scripts", "output"],
                        "description": "Mode of checking: 'scripts' for script compilation errors only, 'output' for all errors/warnings from the output panel",
                        "default": "scripts"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_operation",
            "description": "Dynamic image generation and editing. Specify which images to use as inputs, or leave empty for pure text generation. Perfect for: generating new images, editing specific uploaded images, combining multiple images, or modifying existing images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the desired image or modification."
                    },
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of image identifiers to use as inputs. Leave empty [] for pure text generation. Include specific image IDs when you want to edit/combine existing images from the conversation."
                    },
                    "style": {
                        "type": "string",
                        "description": "Art style (e.g., 'realistic', 'anime', 'pixel_art', 'cartoon', 'photographic')"
                        },
                        "path_to_save": {
                            "type": "string",
                            "description": "Optional. Local path on the Godot editor machine where the client should save the generated image (e.g., 'res://art/output.png' or absolute). The server will not write this path; it's forwarded for the client to handle."
                        },
                        "path": {
                            "type": "string",
                            "description": "Alias for path_to_save. If provided, the editor will save the generated image to this path."
                        },
                        "size": {
                            "type": "string",
                            "description": "Provider size hint. Supports square values like '256x256', '512x512', '1024x1024'."
                        },
                        "exact_size": {
                            "type": "string",
                            "description": "Exact output dimensions in pixels, e.g., '64x64'. The server will resize to this exactly using the chosen filter."
                        },
                        "tile_size": {
                            "type": "string",
                            "description": "Tile pixel size 'WxH', e.g., '32x32'. Combine with grid to compute exact_size automatically."
                        },
                        "grid": {
                            "type": "string",
                            "description": "Grid 'colsxrows', e.g., '2x2'. When used with tile_size, final exact size = tile_size * grid."
                        },
                        "resize_filter": {
                            "type": "string",
                            "enum": ["nearest", "bilinear", "bicubic", "lanczos"],
                            "description": "Resampling filter to reach exact_size. Defaults to 'lanczos'; for pixel art use 'nearest'."
                        }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_image_to_path",
            "description": "Save a generated/attached image by its in-conversation image_id to a local path on the Godot editor machine. This is a frontend tool and will be executed by the editor; the server does not write files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "The image identifier, e.g. 'gen_img_...' returned when an image was generated/attached."},
                    "path": {"type": "string", "description": "Destination file path on the editor (e.g., 'res://art/output.png' or absolute)."},
                    "format": {"type": "string", "enum": ["png", "jpg", "jpeg"], "description": "Preferred format hint; the editor may ignore and write original bytes.", "default": "png"}
                },
                "required": ["image_id", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "editor_introspect",
            "description": "Signal and scene inspection/manipulation with optional lightweight tracing. Multiplexed tool to keep the surface small.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "list_node_signals",
                            "list_signal_connections",
                            "list_incoming_connections",
                            "validate_signal_connection",
                            "connect_signal",
                            "disconnect_signal",
                            "rename_node",
                            "open_connections_dialog",
                            "start_signal_trace",
                            "stop_signal_trace",
                            "get_trace_events",
                            "refresh_resources",
                            "slice_spritesheet"
                        ],
                        "description": "Operation to perform"
                    },
                    "sheet_path": { "type": "string", "description": "Spritesheet image path for slice_spritesheet (e.g., 'res://art/hero_sheet.png')" },
                    "tile_size": { "type": "string", "description": "Tile size 'WxH' for slicing (e.g., '32x32'). Vector2i also supported by frontend." },
                    "grid": { "type": "string", "description": "Grid 'colsxrows' (e.g., '3x3'). Optional; auto-computed if omitted." },
                    "margin": { "type": "integer", "description": "Outer margin (pixels) around the sheet", "default": 0 },
                    "spacing": { "type": "integer", "description": "Spacing (pixels) between tiles", "default": 0 },
                    "out_dir": { "type": "string", "description": "Output directory for sliced frames (default: <sheet_dir>/slices)" },
                    "path": { "type": "string", "description": "Node path" },
                    "signal_name": { "type": "string", "description": "Signal name" },
                    "source_path": { "type": "string", "description": "Source node path (for connect/disconnect/validate)" },
                    "target_path": { "type": "string", "description": "Target node path" },
                    "method": { "type": "string", "description": "Target method name" },
                    "binds": { "type": "array", "items": {}, "description": "Optional bound args" },
                    "flags": { "type": "integer", "description": "Connect flags (e.g., CONNECT_DEFERRED)" },
                    "new_name": { "type": "string", "description": "New name for rename_node" },
                    "node_paths": { "type": "array", "items": { "type": "string" }, "description": "Nodes to trace" },
                    "signals": { "type": "array", "items": { "type": "string" }, "description": "Signals to trace" },
                    "include_args": { "type": "boolean", "default": False, "description": "Include args in trace events" },
                    "max_events": { "type": "integer", "default": 100, "description": "Max buffered events" },
                    "trace_id": { "type": "string", "description": "Existing trace ID" },
                    "since_index": { "type": "integer", "description": "Fetch events/logs since index" }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "slice_spritesheet",
            "description": "Backend spritesheet slicer. Takes a sheet (base64 or path) and returns frames (row/col indexed) as base64 PNGs with robust auto-detection of grid/margins.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_base64": {"type": "string", "description": "Base64-encoded PNG/JPG spritesheet (preferred)."},
                    "sheet_path": {"type": "string", "description": "Optional local path for dev; prefer sheet_base64."},
                    "tile_size": {"type": "string", "description": "Tile size 'WxH' (e.g., '32x32'). Optional if auto_detect."},
                    "grid": {"type": "string", "description": "Grid 'colsxrows' (e.g., '3x3'). Optional if auto_detect."},
                    "margin": {"type": "integer", "description": "Outer margin (pixels).", "default": 0},
                    "spacing": {"type": "integer", "description": "Spacing between tiles (pixels).", "default": 0},
                    "auto_detect": {"type": "boolean", "description": "Infer margins/spacing/grid from image content.", "default": True},
                    "bg_tolerance": {"type": "integer", "description": "Color tolerance for background detection (0..50).", "default": 24},
                    "alpha_threshold": {"type": "integer", "description": "Alpha <= threshold treated as background (0..255).", "default": 1},
                    "tight_crop": {"type": "boolean", "description": "Crop to non-transparent content inside each cell.", "default": True},
                    "padding": {"type": "integer", "description": "Padding around cropped content on final tile.", "default": 0},
                    "fuzzy": {"type": "integer", "description": "Extra pixels to expand cell bounds to avoid cutoffs.", "default": 2},
                    "normalize_to": {"type": "string", "description": "Final tile canvas 'WxH'. Defaults to tile_size if omitted."}
                },
                "required": ["sheet_base64"]
            }
        }
    },
    # Deprecated tools removed: create_script_file, delete_file_safe
    {
        "type": "function",
        "function": {
            "name": "search_across_project",
            "description": "Semantic search across the user's current Godot project. Returns the most relevant files by meaning (not keyword) and can include graph context (connected files and central project files). Use this to locate where behavior is implemented, find related assets, or navigate large projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of what to find (e.g., 'player movement', 'where damage is applied', 'UI theme resource')."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                        "default": 5
                    },
                    "include_graph": {
                        "type": "boolean",
                        "description": "Include graph context: connected files per result and central files (default true).",
                        "default": True
                    },
                    "modality_filter": {
                        "type": "string",
                        "description": "Optional filter: 'text' (scripts/scenes), 'image', 'audio'."
                    },
                    "project_root": {
                        "type": "string",
                        "description": "Absolute path to the project root. Defaults to active project if omitted."
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Stable project identifier to segregate indexes across machines (optional)."
                    },
                    "trace_dependencies": {
                        "type": "boolean",
                        "description": "Enable multi-hop dependency tracing to show what functions call/affect each other (default false).",
                        "default": False
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "description": "Search mode: 'semantic' (AI understanding), 'keyword' (exact text), 'hybrid' (both). Default: semantic.",
                        "default": "semantic"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_across_godot_docs",
            "description": "Search the latest Godot documentation with multiple modes and intelligent filtering. Perfect for learning Godot patterns and finding working code examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find in the docs (natural language or specific terms)."},
                    "max_results": {"type": "integer", "description": "Maximum results (default 5)", "default": 5},
                    "search_mode": {
                        "type": "string", 
                        "enum": ["auto", "semantic", "keyword", "hybrid"],
                        "description": "Search mode: 'auto' (best mode detection), 'semantic' (meaning-based), 'keyword' (exact terms), 'hybrid' (both)",
                        "default": "auto"
                    },
                    "section_filter": {
                        "type": "string",
                        "enum": ["overview", "methods", "properties", "signals"],
                        "description": "Filter by documentation section"
                    },
                    "class_filter": {
                        "type": "string",
                        "description": "Filter by specific class (e.g., 'CharacterBody3D', 'Camera3D')"
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "description": "Filter by difficulty level"
                    },
                    "code_examples_only": {
                        "type": "boolean",
                        "description": "Only return documentation with code examples",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_godot_assets",
            "description": "Search the Godot Asset Library for plugins, templates, demos, and other assets",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms for assets (e.g., 'dialogue system', 'platformer', 'inventory')"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["2d_tools", "3d_tools", "shaders", "materials", "tools", "scripts", "misc", "templates", "demos", "plugins"],
                        "description": "Filter by asset category"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of results to return (1-100)"
                    },
                    "support_level": {
                        "type": "string",
                        "enum": ["all", "official", "featured", "community", "testing"],
                        "default": "all",
                        "description": "Filter by support level - official, featured, community, or testing assets"
                    },
                    "godot_version": {
                        "type": "string",
                        "default": "4.3",
                        "description": "Godot engine version to filter assets for (e.g., '4.3', '4.2', '4.1', '3.5'). Defaults to current stable version."
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["rating", "updated", "name", "cost"],
                        "default": "rating",
                        "description": "Sort results by rating, last updated date, name (alphabetical), or cost"
                    },
                    "sort_reverse": {
                        "type": "boolean",
                        "default": False,
                        "description": "Reverse the sort order (e.g., highest to lowest rating, newest to oldest)"
                    },
                    "asset_type": {
                        "type": "string",
                        "enum": ["any", "addon", "project"],
                        "default": "any",
                        "description": "Filter by asset type - any, addon (plugins/tools), or project (templates/demos)"
                    },
                    "cost_filter": {
                        "type": "string",
                        "enum": ["all", "free", "paid"],
                        "default": "all",
                        "description": "Filter by cost - show all, only free assets, or only paid assets"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "install_godot_asset",
            "description": "Download and install an asset from the Godot Asset Library into the current project",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "The asset ID from search results"
                    },
                    "project_path": {
                        "type": "string", 
                        "description": "Path to the Godot project (e.g., 'res://' or absolute path)"
                    },
                    "install_location": {
                        "type": "string",
                        "default": "addons/",
                        "description": "Where to install the asset (addons/, scripts/, scenes/, etc.)"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "default": True,
                        "description": "Create a backup before installation in case of conflicts"
                    }
                },
                "required": ["asset_id", "project_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_game",
            "description": "Start the game/scene for testing and debugging. Clears error log by default for clean testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_path": {
                        "type": "string",
                        "description": "Path to the scene to run (optional, uses current scene if not provided)"
                    },
                    "clear_errors": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Whether to clear previous errors before starting for clean testing"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "stop_game",
            "description": "Stop the currently running game/scene",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_status", 
            "description": "Check if a game is currently running and which scene",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_runtime_errors_summary",
            "description": "Get a smart summary of runtime errors with deduplication. Shows total error counts, unique error types, and most frequent errors. Perfect for getting an overview without being overwhelmed by hundreds of duplicate errors.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "include_warnings": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include warnings in addition to errors"
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "Only show errors from a specific file (optional)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_runtime_errors_detailed",
            "description": "Get detailed runtime error information with smart filtering and grouping. Use this after get_runtime_errors_summary to investigate specific error types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_warnings": {
                        "type": "boolean",
                        "default": True, 
                        "description": "Include warnings in addition to errors"
                    },
                    "max_count": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of errors to return"
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "Only show errors from a specific file (optional)"
                    },
                    "message_contains": {
                        "type": "string",
                        "description": "Only show errors containing this text (optional)"
                    },
                    "group_duplicates": {
                        "type": "boolean",
                        "default": True,
                        "description": "Group identical errors and show frequency counts vs individual instances"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the editor or running game with multiple capture modes",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename for the screenshot",
                        "default": "screenshot_debug.png"
                    },
                    "target": {
                        "type": "string",
                        "enum": ["editor", "game", "both"],
                        "description": "What to capture: editor viewport, game viewport, or both",
                        "default": "editor"
                    },
                    "return_base64": {
                        "type": "boolean",
                        "description": "Return base64 data instead of saving to file (for immediate chat display)",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "generate_3d_model",
    #         "description": "Generate a 3D model from a text description. Creates a GLB file that can be imported into Godot.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "prompt": {
    #                     "type": "string",
    #                     "description": "Text description of the 3D model to generate (e.g., 'a low-poly tree', 'a simple spaceship', 'a medieval sword')"
    #                 },
    #                 "model": {
    #                     "type": "string",
    #                     "enum": ["fast", "good", "pretty good"],
    #                     "default": "fast",
    #                     "description": "Generation quality/speed tradeoff. 'fast' for quick results, 'good' for better quality, 'pretty good' for highest quality."
    #                 },
    #                 "save_path": {
    #                     "type": "string",
    #                     "description": "Optional path where to save the generated model in the project (e.g., 'res://models/generated_tree.glb')"
    #                 }
    #             },
    #             "required": ["prompt"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "check_for_app_updates",
            "description": "Check if a newer version of Orca Engine is available and show update notification to user",
            "parameters": {
                "type": "object",
                "properties": {
                    "force_check": {
                        "type": "boolean",
                        "description": "Force immediate check even if recently checked",
                        "default": False
                    },
                    "show_notification": {
                        "type": "boolean", 
                        "description": "Show update popup to user if update is available",
                        "default": True
                    }
                },
                "required": []
            }
        }
    }
]