"""
© 2025 Simplifine Corp. Original backend contribution for Orca Engine.
Free personal/non-commercial use only under Company Non‑Commercial License.
Commercial use requires a separate license from Simplifine. See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""
from flask import Flask, request, Response, jsonify, redirect, session, stream_with_context
import openai
from litellm import completion
import litellm
import json
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import requests
from threading import Lock
import uuid
import time
import tempfile
import hashlib
from cloud_vector_manager import CloudVectorManager
from auth_manager import AuthManager

# Load environment variables from .env file
load_dotenv()

# --- Global State & Configuration ---


# Stop mechanism for streaming requests
stop_requests_lock = Lock()
ACTIVE_REQUESTS = {}  # request_id -> {"stop": False, "timestamp": time.time()}

def cleanup_old_requests():
    """Clean up requests older than 5 minutes to prevent memory leaks"""
    current_time = time.time()
    with stop_requests_lock:
        to_remove = []
        for req_id, data in ACTIVE_REQUESTS.items():
            if current_time - data["timestamp"] > 300:  # 5 minutes
                to_remove.append(req_id)
        for req_id in to_remove:
            del ACTIVE_REQUESTS[req_id]

app = Flask(__name__)
# Secret must be stable across restarts in production. Require env in production, random only in DEV_MODE.
_dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'
DEPLOYMENT_MODE = os.getenv('DEPLOYMENT_MODE', 'oss').lower()  # 'oss' or 'cloud'
REQUIRE_SERVER_API_KEY = os.getenv('REQUIRE_SERVER_API_KEY', 'false').lower() == 'true'
SERVER_API_KEY = os.getenv('SERVER_API_KEY')
_secret_env = os.getenv('FLASK_SECRET_KEY')
if _secret_env:
    app.secret_key = _secret_env
elif _dev_mode:
    app.secret_key = os.urandom(24)
else:
    raise ValueError("FLASK_SECRET_KEY must be set in production")

# Multi-provider model configuration using LiteLLM
MODEL_MAP = {
    "gemini-2.5": os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-pro"),
    "claude-4": os.getenv("CLAUDE_MODEL", "anthropic/claude-sonnet-4-20250514"),
    "gpt-5": os.getenv("OPENAI_MODEL", "openai/gpt-5"),
    "gpt-4o": os.getenv("GPT4O_MODEL", "openai/gpt-4o"),
}

# Default model and allowed models
DEFAULT_MODEL = "gpt-5"
ALLOWED_CHAT_MODELS = set(MODEL_MAP.keys())

# Keep OpenAI client for image operations (LiteLLM doesn't support images yet)
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("WARNING: OPENAI_API_KEY not set - image operations will fail")
    client = None
else:
    client = openai.OpenAI(api_key=api_key)

def get_validated_chat_model(requested: str | None) -> str:
    """Return a valid chat model limited to ALLOWED_CHAT_MODELS.
    Falls back to default MODEL if the requested one is not allowed or missing.
    """
    try:
        if requested and requested in ALLOWED_CHAT_MODELS:
            return MODEL_MAP[requested]
    except Exception:
        pass
    return MODEL_MAP[DEFAULT_MODEL]

def get_model_friendly_name(model_id: str) -> str:
    """Get friendly name for a model ID"""
    for friendly, real_id in MODEL_MAP.items():
        if real_id == model_id:
            return friendly
    return model_id

# Initialize Authentication Manager
auth_manager = AuthManager()

# Initialize Cloud Vector Manager
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
CLOUD_VECTOR_ENABLED = bool(GCP_PROJECT_ID)
cloud_vector_manager = None
if CLOUD_VECTOR_ENABLED:
    cloud_vector_manager = CloudVectorManager(GCP_PROJECT_ID, client)

# Load system prompt from file (once at startup)
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
SYSTEM_PROMPT = None
try:
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read().strip()
        if SYSTEM_PROMPT:
            print(f"SYSTEM_PROMPT: Loaded ({len(SYSTEM_PROMPT)} chars)")
        else:
            print("SYSTEM_PROMPT: File is empty; no system message will be prepended")
except Exception as e:
    print(f"SYSTEM_PROMPT: Failed to load: {e}")

def verify_authentication():
    """Verify user authentication from request (with dev mode bypass)"""
    # DEV MODE: Allow bypass for testing
    if os.getenv('DEV_MODE', 'false').lower() == 'true':
        # Check headers first, then JSON
        user_id = request.headers.get('X-User-ID')
        data = request.json if request.json else {}
        if not user_id:
            user_id = data.get('user_id')
        # Fallback: derive a dev user from machine_id to allow zero-click local runs
        machine_id_dev = request.headers.get('X-Machine-ID') or data.get('machine_id')
        if user_id or machine_id_dev:
            effective_user = user_id or f"dev_{machine_id_dev}"
            print(f"🧪 DEV MODE: Bypassing auth for user {effective_user}")
            return {
                "id": effective_user,
                "name": "Dev User",
                "email": "dev@example.com",
                "provider": "dev_mode"
            }, None, None
    
    auth_header = request.headers.get('Authorization', '')
    machine_id = request.headers.get('X-Machine-ID') or (request.json.get('machine_id') if request.json else None)
    
    if not machine_id:
        return None, {"error": "machine_id required", "success": False}, 401
    
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        user = auth_manager.verify_session(machine_id, token)
        if user:
            return user, None, None
    
    # Guest fallback if allowed
    # Default: OSS mode allows guests; cloud mode disables by default unless explicitly enabled
    default_allow = (DEPLOYMENT_MODE != 'cloud')
    allow_guests = os.getenv('ALLOW_GUESTS', 'true' if default_allow else 'false').lower() == 'true'
    request_allows_guest = (request.headers.get('X-Allow-Guest', 'true').lower() == 'true')
    if allow_guests and request_allows_guest:
        guest_name = request.headers.get('X-Guest-Name')
        guest_result = auth_manager.create_or_get_guest_session(machine_id, guest_name)
        if guest_result.get('success'):
            return guest_result['user'], None, None
        else:
            return None, {"error": f"Guest session failed: {guest_result.get('error','unknown')}", "success": False}, 401
    
    return None, {"error": "Authentication required", "success": False}, 401


def verify_server_key_if_required():
    """Optional server-side API key gate for sensitive endpoints."""
    if not REQUIRE_SERVER_API_KEY:
        return None
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({"success": False, "error": "Server API key required"}), 401
    token = auth_header[7:]
    if not SERVER_API_KEY or token != SERVER_API_KEY:
        return jsonify({"success": False, "error": "Invalid server API key"}), 403
    return None

# Image handling will use OpenAI's native ID system - no local registry needed

# --- Helper Functions ---

# --- Asset Processing Function ---
def process_asset_internal(arguments: dict) -> dict:
    """Process assets using various AI and image processing techniques"""
    try:
        operation = arguments.get('operation', '')
        input_path = arguments.get('input_path', '')
        
        if not operation:
            return {"success": False, "error": "No operation specified"}
        
        # For now, return a placeholder for asset processing operations
        # This would be expanded to include actual asset processing logic
        operations = {
            'remove_background': 'Background removal using AI',
            'auto_crop': 'Intelligent sprite boundary detection',
            'generate_spritesheet': 'Automatic sprite sheet generation',
            'style_transfer': 'Apply consistent art style',
            'batch_process': 'Process multiple assets',
            'classify': 'Classify asset types',
            'create_variants': 'Generate asset variations'
        }
        
        if operation not in operations:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
        return {
            "success": True,
            "message": f"Asset processing '{operation}' completed",
            "operation": operation,
            "description": operations[operation],
            "input_path": input_path
        }
            
    except Exception as e:
        return {"success": False, "error": f"Asset processing failed: {str(e)}"}

# --- Dynamic Image Operation Function ---
def image_operation_internal(arguments: dict, conversation_messages: list = None) -> dict:
    """Dynamic image generation or editing using OpenAI Images API.

    Behavior:
    - If no image IDs provided: generate a new image from prompt.
    - If one or more image IDs provided: edit the first matching image using the prompt.
    The conversation can carry prior images with unique `name` and base64 data which
    the model can reference via the `images` array in `arguments`.
    """

    try:
        description = arguments.get('description', '')
        style = arguments.get('style', '')
        image_ids = arguments.get('images', []) or []
        size = arguments.get('size', '1024x1024')  # optional
        # For cloud safety: we do not write to local file systems from the server.
        # If provided, path_to_save is simply echoed back so the Godot editor can
        # save client-side after receiving the image.
        path_to_save = arguments.get('path_to_save')

        print("IMAGE_OP DEBUG: Incoming arguments:")
        print(f"  - description len: {len(description)} | style: '{style}' | size: {size}")
        if isinstance(image_ids, list):
            print(f"  - requested image ids: {image_ids} (count={len(image_ids)})")
        else:
            print(f"  - images field type: {type(image_ids)} -> {image_ids}")

        if not description:
            return {"success": False, "error": "No description provided for image operation"}

        prompt_text = description
        if style:
            prompt_text += f", {style} style"

        # Gather available images from prior conversation messages
        available_images = {}
        if conversation_messages:
            print(f"IMAGE_OP DEBUG: conversation_messages count: {len(conversation_messages)}")
            cm_index = -1
            for msg in conversation_messages:
                cm_index += 1
                if not isinstance(msg, dict):
                    continue
                if 'images' in msg and isinstance(msg['images'], list):
                    print(f"    - msg[{cm_index}] has images: {len(msg['images'])}")
                    for img in msg['images']:
                        name = img.get('name')
                        b64 = img.get('base64_data')
                        if name and b64:
                            available_images[name] = img
                            print(f"      -> cached image '{name}' (base64 len={len(b64)})")
                # Also log tool/assistant markers present in content
                content_preview = str(msg.get('content', ''))[:120].replace('\n', ' ')
                if 'image_name' in content_preview or 'Image ID' in content_preview:
                    print(f"    - msg[{cm_index}] content mentions image id: '{content_preview}'")
        else:
            print("IMAGE_OP DEBUG: No conversation_messages provided or empty")

        selected_images = []
        for img_id in image_ids:
            # Accept both exact and numeric-suffixed IDs (e.g., 'generated_123' vs 'generated_123.0')
            match = None
            if img_id in available_images:
                match = available_images[img_id]
            else:
                # Try tolerant matching
                for key in available_images.keys():
                    if str(key).startswith(str(img_id)):
                        match = available_images[key]
                        print(f"IMAGE_OP DEBUG: tolerant match for '{img_id}' -> '{key}'")
                        break
            if match:
                selected_images.append(match)
                print(f"IMAGE_OP: Selected input image '{img_id}'")
            else:
                print(f"IMAGE_OP: Warning - requested image '{img_id}' not found in conversation context")

        print(f"IMAGE_OP DEBUG: available_images keys: {list(available_images.keys())}")
        print(f"IMAGE_OP DEBUG: selected_images count: {len(selected_images)}")

        # If no images selected, do text-to-image generation
        if not selected_images:
            print("IMAGE_OP: Generating new image from prompt using Images API")
            gen = client.images.generate(model="gpt-image-1", prompt=prompt_text, size=size)
            if not gen.data or not getattr(gen.data[0], 'b64_json', None):
                return {"success": False, "error": "Image generation returned no data"}

            image_base64 = gen.data[0].b64_json
            result = {
                "success": True,
                "image_data": image_base64,
                "prompt": description,
                "style": style,
                "format": "png",
                "input_images": 0,
                "requested_images": len(image_ids)
            }
            if path_to_save:
                result["path_to_save"] = path_to_save
            return result

        # If images provided, do an edit on the first one
        print("IMAGE_OP: Performing edit on provided image using Images API")
        first_img = selected_images[0]
        try:
            img_bytes = base64.b64decode(first_img['base64_data'])
            print(f"IMAGE_OP DEBUG: decoded first image bytes: {len(img_bytes)}")
        except Exception as decode_err:
            return {"success": False, "error": f"Failed to decode input image '{first_img.get('name','unknown')}': {decode_err}"}

        # Re-encode to PNG to ensure a valid image mimetype and structure
        try:
            pil_image = Image.open(io.BytesIO(img_bytes))
            print(f"IMAGE_OP DEBUG: PIL loaded size: {pil_image.size} | mode: {pil_image.mode}")
        except Exception as pil_err:
            return {"success": False, "error": f"Failed to load input image: {pil_err}"}

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                pil_image.save(tmp, format="PNG")

            with open(temp_path, "rb") as img_fh:
                # Prefer images.edits if available in SDK; otherwise fall back to images.edit
                images_api = getattr(client, 'images')
                print(f"IMAGE_OP DEBUG: Using images API method: {'edits' if hasattr(images_api, 'edits') else 'edit'} | prompt len={len(prompt_text)}")
                if hasattr(images_api, 'edits'):
                    edit = images_api.edits(model="gpt-image-1", image=img_fh, prompt=prompt_text, size=size)
                else:
                    # Older SDKs
                    edit = images_api.edit(model="gpt-image-1", image=img_fh, prompt=prompt_text, size=size)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        if not edit.data or not getattr(edit.data[0], 'b64_json', None):
            print("IMAGE_OP DEBUG: Edit API returned no data or missing b64_json")
            return {"success": False, "error": "Image edit returned no data"}

        image_base64 = edit.data[0].b64_json
        result = {
            "success": True,
            "image_data": image_base64,
            "prompt": description,
            "style": style,
            "format": "png",
            "input_images": 1,
            "requested_images": len(image_ids)
        }
        if path_to_save:
            result["path_to_save"] = path_to_save
        return result

    except Exception as e:
        print(f"IMAGE_OP ERROR: {str(e)}")
        return {"success": False, "error": f"Image operation failed: {str(e)}"}

# --- Search Across Project Function ---
def search_across_project_internal(arguments: dict, current_user: dict = None) -> dict:
    """Execute search across project using the cloud vector system"""
    try:
        query = arguments.get('query', '')
        if not query:
            return {"success": False, "error": "Query parameter is required"}
        
        # Get parameters
        max_results = arguments.get('max_results', 5)
        include_graph = bool(arguments.get('include_graph', True))
        project_root = arguments.get('project_root')
        project_id = arguments.get('project_id')
        
        # Get authentication
        if current_user is None:
            user, error_response, status_code = verify_authentication()
            if error_response:
                return {"success": False, "error": "Authentication required"}
        else:
            user = current_user
        
        # Ensure a project_root is present or fall back to environment/CWD (never require request context here)
        if not project_root:
            # Try environment variable first, then current working directory
            project_root = os.getenv('PROJECT_ROOT') or os.getcwd()
        
        if not project_root:
            return {
                "success": False,
                "error": "project_root not provided and no fallback available"
            }
        
        # Generate project ID if not provided
        if not project_id:
            project_id = hashlib.md5(project_root.encode()).hexdigest()
        
        # Search using cloud vector manager
        results = cloud_vector_manager.search(query, user['id'], project_id, max_results)
        
        # Format results
        formatted_results = {
            "similar_files": [
                {
                    "file_path": r['file_path'],
                    "similarity": r['similarity'],
                    "modality": "text",
                    "chunk_index": r['chunk']['chunk_index'] if r.get('chunk') else 0,
                    "chunk_start": r['chunk']['start_line'] if r.get('chunk') else None,
                    "chunk_end": r['chunk']['end_line'] if r.get('chunk') else None,
                }
                for r in results
            ],
            "central_files": [],
            "graph_summary": {}
        }
        graph_context = {}
        if include_graph and results:
            files = [r['file_path'] for r in results]
            graph_context = cloud_vector_manager.get_graph_context_for_files(files, user['id'], project_id)
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "include_graph": include_graph,
            "graph": graph_context,
            "file_count": len(results),
            "message": f"Found {len(results)} relevant files for query: {query}"
        }
        
    except Exception as e:
        print(f"SEARCH_PROJECT_INTERNAL_ERROR: {e}")
        return {"success": False, "error": f"Search failed: {str(e)}"}

# --- Note: Script generation now handled by dedicated /generate_script endpoint ---

# --- Tool Execution Function ---
def execute_godot_tool(function_name: str, arguments: dict) -> dict:
    """Execute backend-specific tools"""
    if function_name == "image_operation":
        return image_operation_internal(arguments)
    elif function_name == "asset_processor":
        return process_asset_internal(arguments)
    elif function_name == "search_across_project":
        return search_across_project_internal(arguments)
    else:
        # This shouldn't happen if we filter correctly
        print(f"WARNING: Unknown backend tool called: {function_name}")

    return {"success": False, "error": f"Unknown backend tool called: {function_name}"}

# --- Individual Tool Definitions (Original 22 Tools) ---
godot_tools = [
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
            "name": "set_node_property",
            "description": "Set a property on a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    },
                    "property": {
                        "type": "string",
                        "description": "Property name to set"
                    },
                    "value": {
                        "description": "Value to set for the property"
                    }
                },
                "required": ["path", "property", "value"]
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
                        "description": "Parent node for instantiate operations"
                    }
                },
                "required": ["operation"]
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
            "description": "Apply AI-powered edits to a file",
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
            "description": "Check for compilation errors in script files. Can check a specific file or all script files in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Script file path to check (not required if check_all is true)"
                    },
                    "check_all": {
                        "type": "boolean",
                        "description": "If true, check all script files in the project instead of a specific file",
                        "default": False
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
                            "get_trace_events"
                        ],
                        "description": "Operation to perform"
                    },
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
                    }
                },
                "required": ["query"]
            }
        }
    }
]


@app.route('/stop', methods=['POST'])
def stop_chat():
    """Stop a streaming chat request"""
    data = request.json
    request_id = data.get('request_id')
    
    if not request_id:
        return jsonify({"error": "No request_id provided"}), 400
    
    with stop_requests_lock:
        if request_id in ACTIVE_REQUESTS:
            ACTIVE_REQUESTS[request_id]["stop"] = True
            print(f"STOP_REQUEST: Marked request {request_id} for stopping")
            return jsonify({"success": True, "message": "Stop signal sent"})
        else:
            print(f"STOP_REQUEST: Request {request_id} not found in active requests")
            return jsonify({"success": False, "message": "Request not found or already completed"}), 404

@app.route('/chat', methods=['POST'])
def chat():
    # Optional server key gate
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    """
    Main chat endpoint that handles the full conversation flow:
    1. Receives messages from Godot
    2. Calls OpenAI API 
    3. Executes any tool calls
    4. Streams the final response back to Godot
    """
    # Verify authentication
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    data = request.json
    messages = data.get('messages', [])
    model = get_validated_chat_model(data.get('model'))  # Restrict to allowed models

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Generate unique request ID and register it
    request_id = str(uuid.uuid4())
    with stop_requests_lock:
        ACTIVE_REQUESTS[request_id] = {"stop": False, "timestamp": time.time()}
    
    # Clean up old requests
    cleanup_old_requests()
    
    print(f"CHAT_START: New request {request_id} registered")

    def generate_stream():
        def check_stop():
            """Check if this request should be stopped"""
            with stop_requests_lock:
                return ACTIVE_REQUESTS.get(request_id, {}).get("stop", False)
        
        try:
            # Send request_id first so frontend can use it for stop requests
            yield json.dumps({"request_id": request_id, "status": "started"}) + '\n'
            
            # Filter out any None or invalid messages from the start
            conversation_messages = []
            for msg in messages:
                if msg is not None and isinstance(msg, dict) and msg.get('role'):
                    conversation_messages.append(msg)
                else:
                    print(f"STARTUP_FILTER: Skipping invalid message: {type(msg)} = {msg}")
                
                # Check for stop even during initial processing
                if check_stop():
                    print(f"STOP_DETECTED: Request {request_id} stopped during message filtering")
                    yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                    return

            # Log incoming headers for project root troubleshooting
            try:
                prj_hdr = request.headers.get('X-Project-Root')
                print(f"CHAT_HEADERS: X-Project-Root={prj_hdr} X-User-ID={request.headers.get('X-User-ID')} X-Machine-ID={request.headers.get('X-Machine-ID')}")
            except Exception:
                pass

            while True:  # Loop to handle tool calling and responses
                # Check for stop before each major operation
                if check_stop():
                    print(f"STOP_DETECTED: Request {request_id} stopped before OpenAI call")
                    yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                    return
                
                print(f"CONVERSATION_LOOP: Starting OpenAI call with {len(conversation_messages)} messages")
                if conversation_messages:
                    last_msg = conversation_messages[-1]
                    if last_msg and isinstance(last_msg, dict):
                        print(f"CONVERSATION_LOOP: Last message: {last_msg.get('role', 'unknown')}")
                    else:
                        print(f"CONVERSATION_LOOP: Last message is invalid: {type(last_msg)}")
                    
                # Debug logs for OpenAI messages have been quieted to reduce console noise.
                
                # Clean messages for OpenAI (preserve vision content, remove custom fields)
                openai_messages = []
                # Only forward tool-role messages if there's a preceding assistant message with tool_calls
                prior_assistant_with_tools = False
                for msg in conversation_messages:
                    if msg is None:
                        # print(f"CLEAN_MESSAGES: Skipping None message")
                        continue
                    if not isinstance(msg, dict):
                        # print(f"CLEAN_MESSAGES: Skipping non-dict message: {type(msg)}")
                        continue
                        
                    role = msg['role']
                    if role == 'tool' and not prior_assistant_with_tools:
                        # Skip stray tool messages that are not responses to assistant tool_calls
                        continue
                    clean_msg = {
                        'role': role,
                        'content': msg.get('content') # Use .get for safety
                    }
                    # Include standard OpenAI fields
                    if 'tool_calls' in msg:
                        # Ensure tool calls have required 'type' field for OpenAI
                        fixed_tool_calls = []
                        for tool_call in msg['tool_calls']:
                            if isinstance(tool_call, dict):
                                fixed_tool_call = tool_call.copy()
                                # Add 'type' field if missing
                                if 'type' not in fixed_tool_call:
                                    fixed_tool_call['type'] = 'function'
                                fixed_tool_calls.append(fixed_tool_call)
                        clean_msg['tool_calls'] = fixed_tool_calls
                        if role == 'assistant' and fixed_tool_calls:
                            prior_assistant_with_tools = True
                    if 'tool_call_id' in msg:
                        clean_msg['tool_call_id'] = msg['tool_call_id']
                    if 'name' in msg:
                        clean_msg['name'] = msg['name']
                    
                    # Remove custom frontend fields that may contain large data
                    # Keep only standard OpenAI message format
                    openai_messages.append(clean_msg)

                # Prepend system prompt if available
                if SYSTEM_PROMPT:
                    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
                    openai_messages = [system_msg] + openai_messages
                
                # Debug: Check total token usage
                total_chars = sum(len(str(msg.get('content', ''))) for msg in openai_messages)
                model_friendly = get_model_friendly_name(model)
                print(f"LITELLM_PREP: Sending {len(openai_messages)} messages to {model_friendly} ({model}), total chars: {total_chars}")
                if total_chars > 100000:
                    print(f"LITELLM_PREP: WARNING - Very large message content ({total_chars} chars), may hit token limits!")
                
                # Resilient model call with brief retry and cross-provider fallback
                attempts = 0
                max_attempts = 2
                providers_tried = set()
                model_try = model
                while True:
                    try:
                        response = completion(
                            model=model_try,
                            messages=openai_messages,
                            tools=godot_tools,
                            tool_choice="auto",
                            stream=True
                        )
                        break
                    except Exception as e:
                        err_name = e.__class__.__name__
                        overloaded = "Overloaded" in str(e)
                        transient = err_name in ("InternalServerError", "RateLimitError", "ServiceUnavailableError") or overloaded
                        if transient and attempts < max_attempts:
                            attempts += 1
                            yield json.dumps({
                                "status": "retrying_provider",
                                "provider": get_model_friendly_name(model_try),
                                "attempt": attempts
                            }) + '\n'
                            time.sleep(0.6 * attempts)
                            continue

                        # Fallback to a different provider if available
                        providers_tried.add(model_try)
                        fallback = None
                        try:
                            if model_try == MODEL_MAP.get("claude-4") and MODEL_MAP.get("gpt-4o") not in providers_tried:
                                fallback = MODEL_MAP.get("gpt-4o")
                            elif model_try == MODEL_MAP.get("gpt-4o") and MODEL_MAP.get("gpt-5") not in providers_tried:
                                fallback = MODEL_MAP.get("gpt-5")
                            elif model_try == MODEL_MAP.get("gpt-5") and MODEL_MAP.get("gemini-2.5") not in providers_tried:
                                fallback = MODEL_MAP.get("gemini-2.5")
                        except Exception:
                            fallback = None

                        if fallback and fallback != model_try:
                            yield json.dumps({
                                "status": "switching_model",
                                "from": get_model_friendly_name(model_try),
                                "to": get_model_friendly_name(fallback)
                            }) + '\n'
                            model_try = fallback
                            attempts = 0
                            continue

                        # No retries/fallbacks left – bubble up to main handler
                        raise

                full_text_response = ""
                tool_call_aggregator = {}
                tool_ids = {}
                current_tool_index = None
                chunk_count = 0
                
                for chunk in response:
                    # Check for stop during streaming - this is critical for mid-stream stopping
                    if check_stop():
                        print(f"STOP_DETECTED: Request {request_id} stopped during streaming")
                        yield json.dumps({"status": "stopped", "message": "Request stopped during streaming"}) + '\n'
                        return
                    
                    chunk_count += 1
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        
                        # Handle streaming text content
                        content = getattr(delta, 'content', None) if hasattr(delta, 'content') else delta.get('content')
                        if content:
                            full_text_response += content
                            yield json.dumps({
                                "content_delta": content,
                                "status": "streaming"
                            }) + '\n'
                        
                        # Handle tool calls (LiteLLM format)
                        tool_calls = getattr(delta, 'tool_calls', None) if hasattr(delta, 'tool_calls') else delta.get('tool_calls')
                        if tool_calls:
                            for tool_call in tool_calls:
                                # Handle both pydantic and dict formats
                                if hasattr(tool_call, 'index'):
                                    index = tool_call.index
                                    tc_id = getattr(tool_call, 'id', None)
                                    fn = getattr(tool_call, 'function', None)
                                    fn_name = getattr(fn, 'name', None) if fn else None
                                    fn_args = getattr(fn, 'arguments', '') if fn else ''
                                else:
                                    index = tool_call.get('index', 0)
                                    tc_id = tool_call.get('id')
                                    fn = tool_call.get('function', {})
                                    fn_name = fn.get('name')
                                    fn_args = fn.get('arguments', '')
                                
                                current_tool_index = index
                                
                                # Use index as key for consistent accumulation
                                key = f"tool_call_{index}"
                                if key not in tool_call_aggregator:
                                    tool_call_aggregator[key] = {
                                        "name": "",
                                        "arguments": ""
                                    }
                                    tool_ids[key] = tc_id or f"call_{index}"
                                
                                # Accumulate function name and arguments
                                if fn_name:
                                    tool_call_aggregator[key]["name"] = fn_name
                                if fn_args:
                                    tool_call_aggregator[key]["arguments"] += fn_args
                
                print(f"RESPONSE_DEBUG: Processed {chunk_count} chunks, text_length: {len(full_text_response)}, tools: {len(tool_call_aggregator)}")
                if tool_call_aggregator:
                    print(f"RESPONSE_DEBUG: Tool calls: {[f['name'] for f in tool_call_aggregator.values()]}")
                if not full_text_response and not tool_call_aggregator:
                    print("RESPONSE_DEBUG: WARNING - OpenAI responded with NO content and NO tool calls!")

                # Now that we've processed all chunks, handle the results

                # --- Backend-Only Tool Execution (Image Generation + Search) ---
                backend_tools_detected = [func.get("name") for func in tool_call_aggregator.values() if func.get("name") in ["image_operation", "search_across_project"]]
                print(f"BACKEND_DETECTION: Found {len(backend_tools_detected)} backend tools: {backend_tools_detected}")
                
                if any(func.get("name") in ["image_operation", "search_across_project"] for func in tool_call_aggregator.values()):
                    # This is a backend-only tool call, so we will execute it,
                    # add the results to the conversation, and loop again for the AI's final response.
                    
                    # We need the original tool call data to append to the history
                    original_tool_calls_for_history = []
                    
                    # Execute image operation
                    tool_results_for_history = []
                    
                    for i, func in tool_call_aggregator.items():
                        tool_id = tool_ids[i]
                        original_tool_calls_for_history.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": func["name"], "arguments": func["arguments"]},
                        })
                        
                        if func["name"] == "image_operation":
                            # Check for stop before tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "image_operation", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"])
                            except json.JSONDecodeError:
                                arguments = {}
                            
                            # AI now intelligently specifies which images to use via the 'images' parameter

                            # Execute the operation with conversation context
                            image_result = image_operation_internal(arguments, conversation_messages)
                            
                            # Check for stop after tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Yield result to frontend immediately (include tool_call_id for consistent UI handling)
                            yield json.dumps({
                                "tool_executed": "image_operation",
                                "tool_result": image_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # Prepare tool result for conversation history (exclude massive image data)
                            tool_result_for_openai = {
                                "success": image_result.get("success"),
                                "description": image_result.get("description"),
                                "style": image_result.get("style"),
                                "input_images": image_result.get("input_images", 0),
                                "requested_images": image_result.get("requested_images", 0)
                            }
                            # Exclude the massive 'image_data' field to save tokens
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "image_operation",
                                "content": json.dumps(tool_result_for_openai),
                            })
                        
                        elif func["name"] == "search_across_project":
                            # Check for stop before tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "search_across_project", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            # Ensure project_root is provided (fallback to header)
                            if not arguments.get('project_root'):
                                hdr_root = request.headers.get('X-Project-Root')
                                if hdr_root:
                                    arguments['project_root'] = hdr_root
                                    print(f"SEARCH_TOOL_FIX: Injected project_root from header: {hdr_root}")
                                else:
                                    # As a final fallback, try environment/project cwd
                                    try:
                                        cwd_root = os.getenv('PROJECT_ROOT') or os.getcwd()
                                        arguments['project_root'] = cwd_root
                                        print(f"SEARCH_TOOL_FIX: Fallback project_root from CWD: {cwd_root}")
                                    except Exception:
                                        pass
                            
                            # Execute the search operation (pass current user context)
                            search_result = search_across_project_internal(arguments, current_user=user)
                            # If search failed due to missing project_root, synthesize a minimal result to satisfy toolcall contract
                            if not search_result.get('success', False):
                                msg = search_result.get('error') or search_result.get('message') or 'Search failed'
                                search_result = {
                                    'success': False,
                                    'query': arguments.get('query'),
                                    'results': {'similar_files': [], 'central_files': [], 'graph_summary': {}},
                                    'file_count': 0,
                                    'message': f"search_across_project error: {msg}"
                                }
                            
                            # Check for stop after tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Yield result to frontend immediately
                            yield json.dumps({"tool_executed": "search_across_project", "tool_result": search_result, "tool_call_id": tool_id, "status": "tool_completed"}) + '\n'
                            
                            # Prepare tool result for conversation history
                            tool_result_for_openai = {
                                "success": search_result.get("success"),
                                "query": search_result.get("query"),
                                "file_count": search_result.get("file_count", 0),
                                "message": search_result.get("message"),
                                "similar_files": search_result.get("results", {}).get("similar_files", [])[:3]  # Limit to first 3 for token efficiency
                            }
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "search_across_project",
                                "content": json.dumps(tool_result_for_openai),
                            })
                
                    # Add the assistant's decision to call the tool to history
                    assistant_message = {"role": "assistant", "content": None, "tool_calls": original_tool_calls_for_history}
                    conversation_messages.append(assistant_message)
                    print(f"CONVERSATION_ADD: Added assistant message with tool calls")
                    
                    # Add the results of the tool call to history
                    for tool_result in tool_results_for_history:
                        if tool_result is None:
                            print(f"CONVERSATION_ADD: ERROR - Attempting to add None tool result!")
                            continue
                        conversation_messages.append(tool_result)
                                                    # print(f"CONVERSATION_ADD: Added tool result: {tool_result.get('name', 'unknown')}")

                    # Now, loop again to get the final text response from the AI
                    # print("CONVERSATION_LOOP: Backend tool executed. Continuing loop for final AI response.")
                    continue

                # --- Frontend Tool Calls & Final Text Responses ---
                
                print(f"FRONTEND_PROCESSING: Reached frontend tool processing. tool_call_aggregator has {len(tool_call_aggregator)} tools")
                
                # If we get here, it means no backend tools were called.
                # It's either a final text response or tool calls for the frontend.
                
                # Append assistant message (will include tool calls if any)
                assistant_message = {
                    "role": "assistant",
                    "content": full_text_response if full_text_response else None,
                }

                if tool_call_aggregator:
                    print(f"FRONTEND_PROCESSING: Processing {len(tool_call_aggregator)} frontend tool calls")
                    # Prepare tool calls for both history and frontend
                    tool_calls_for_history = []
                    tool_calls_for_frontend = []
                    for i, func in tool_call_aggregator.items():
                        tool_id = tool_ids[i]
                        print(f"FRONTEND_PROCESSING: Processing tool {func['name']} with id {tool_id}")
                        tool_calls_for_history.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": func["name"], "arguments": func["arguments"]},
                        })
                        tool_calls_for_frontend.append({
                            "id": tool_id,
                            "function": {
                                "name": func["name"],
                                "arguments": func["arguments"]
                            }
                        })
                    
                    assistant_message["tool_calls"] = tool_calls_for_history
                    conversation_messages.append(assistant_message)
                    print(f"CONVERSATION_ADD: Added frontend assistant message with {len(tool_calls_for_history)} tool calls")
                    
                    print(f"FRONTEND_PROCESSING: Sending {len(tool_calls_for_frontend)} tool calls to frontend")
                    # Yield tool calls to the frontend in the format it expects
                    frontend_response = {
                        "status": "executing_tools",
                        "assistant_message": {
                            "role": "assistant",
                            "content": full_text_response or None,
                            "tool_calls": tool_calls_for_frontend
                        }
                    }
                    yield json.dumps(frontend_response) + '\n'
                    # Signal graceful end of this stream so the frontend doesn't treat the disconnect as an error.
                    yield json.dumps({"status": "completed"}) + '\n'
                    print(f"FRONTEND_PROCESSING: Tool calls sent, closing stream awaiting frontend tool execution")
                    break  # Exit loop after sending tools to frontend

                # If no tools, it's a final text response. Append and break.
                print(f"FRONTEND_PROCESSING: No tools detected, treating as final text response")
                conversation_messages.append(assistant_message)
                print(f"CONVERSATION_ADD: Added final text response message")
                yield json.dumps({"status": "completed"}) + '\n'
                break # Exit loop
        
        except Exception as e:
            print(f"ERROR: Exception in stream generation: {e}")
            yield json.dumps({"error": str(e), "status": "error"}) + '\n'
        
        finally:
            # Clean up this request from active requests
            with stop_requests_lock:
                if request_id in ACTIVE_REQUESTS:
                    del ACTIVE_REQUESTS[request_id]
                    print(f"CLEANUP: Removed request {request_id} from active requests")

    # Preserve request context during streaming to avoid 'Working outside of request context'.
    return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')

@app.route('/generate_script', methods=['POST'])
def generate_script():
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    """Generate script content using AI"""
    data = request.json
    script_type = data.get('script_type', '')
    node_type = data.get('node_type', 'Node') 
    description = data.get('description', '')
    
    print(f"GENERATE_SCRIPT: Received request for {script_type} script")
    
    if not script_type or not description:
        return jsonify({"error": "Missing script_type or description"}), 400
    
    # Generate script using AI
    script_prompt = f"""
    Create a GDScript for a {node_type} that serves as a {script_type}.
    
    Requirements: {description}
    
    CRITICAL REQUIREMENTS:
    - Return ONLY raw GDScript code
    - NO markdown formatting (no ```, no ```gdscript, no ```gd)
    - NO explanations or comments outside the code
    - Use GODOT 4 syntax: "extends RefCounted" (NOT "extends Reference")
    - Use GODOT 4 syntax: "extends Node" (NOT "extends KinematicBody2D")
    - Use GODOT 4 syntax: "extends CharacterBody2D" (NOT "extends KinematicBody2D")
    - Use GODOT 4 syntax: "extends RigidBody2D" (NOT "extends RigidBody2D")
    - Ensure proper GDScript syntax for Godot 4.x
    - Start directly with "extends" or class declaration
    
    Example format:
    extends RefCounted
    
    func my_function():
        pass
    """
    
    try:
        response = completion(
            model=get_validated_chat_model(data.get('model')),
            messages=[{"role": "user", "content": script_prompt}]
        )
        
        script_content = response.choices[0].message.content
        
        # Clean up any markdown wrappers that might have leaked through
        script_content = script_content.strip()
        
        # Remove markdown code blocks if they exist
        if script_content.startswith('```'):
            lines = script_content.split('\n')
            # Remove first line if it's a code block marker
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's a closing code block marker
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            script_content = '\n'.join(lines)
        
        # Remove any remaining ``` markers
        script_content = script_content.replace('```gdscript', '').replace('```gd', '').replace('```', '')
        script_content = script_content.strip()
        
        print(f"GENERATE_SCRIPT: Cleaned script content (first 200 chars): {script_content[:200]}")
        
        return jsonify({
            "success": True,
            "script_content": script_content,
            "script_type": script_type,
            "node_type": node_type
        })
        
    except Exception as e:
        print(f"GENERATE_SCRIPT_ERROR: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/predict_code_edit', methods=['POST'])
def predict_code_edit():
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    """
    Uses OpenAI's Predicted Outputs feature to edit code files.
    Provides the original file content as prediction and returns the complete edited file.
    """
    data = request.json
    file_content = data.get('file_content', '')
    prompt = data.get('prompt')
    
    print(f"APPLY_EDIT_REQUEST: Received request with prompt: '{prompt}' for file content length: {len(file_content)}")

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    # Simple, clear prompt for file editing
    user_prompt = f"Edit this file according to the request: {prompt}\n\nReturn the complete edited file content."

    try:
        # Simple approach: provide file content in context and ask for complete edited file
        full_prompt = f"""Edit this file according to the request: {prompt}

Original file content:
```
{file_content}
```

Return the complete edited file content (no explanations, just the code):"""

        response = completion(
            model=get_validated_chat_model(data.get('model')),
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        edited_content = response.choices[0].message.content
        print(f"APPLY_EDIT_MODEL: Received edited file content (length: {len(edited_content)})")
        
        # Return the complete edited file content
        return jsonify({
            "edited_content": edited_content,
            "success": True
        })
        
    except Exception as e:
        print(f"APPLY_EDIT_ERROR: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/auth/login', methods=['GET'])
def auth_login():
    """Start OAuth authentication process"""
    try:
        machine_id = request.args.get('machine_id')
        provider = request.args.get('provider', 'google')  # Default to Google
        
        if not machine_id:
            return jsonify({"error": "machine_id parameter required"}), 400
        
        if provider == 'google':
            auth_url = auth_manager.get_google_auth_url(machine_id)
        elif provider == 'github':
            auth_url = auth_manager.get_github_auth_url(machine_id)
        elif provider == 'microsoft':
            auth_url = auth_manager.get_microsoft_auth_url(machine_id)
        elif provider == 'guest':
            # Create/return guest session immediately
            result = auth_manager.create_or_get_guest_session(machine_id)
            if result.get('success'):
                # Return a small HTML page that the editor can parse, similar to callback flow
                user = result['user']
                token = result['token']
                return f"""
                <html>
                <body>
                    <h1>Guest Session Ready</h1>
                    <p>Welcome, {user['name']}!</p>
                    <script>
                        // Communicate token and user back to opener if needed
                        try {{
                            window.opener && window.opener.postMessage({{
                                type: 'auth_success',
                                provider: 'guest',
                                user: {json.dumps(user)},
                                token: '{token}'
                            }}, '*');
                        }} catch(e) {{}}
                        window.close();
                    </script>
                </body>
                </html>
                """
            else:
                return jsonify({"error": result.get('error','Guest session failed'), "success": False}), 500
        else:
            return jsonify({"error": "Unsupported provider"}), 400
        
        return redirect(auth_url)
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/auth/callback', methods=['GET'])
@app.route('/api/auth/callback', methods=['GET'])
def auth_callback():
    """Handle OAuth callback"""
    try:
        state = request.args.get('state')
        code = request.args.get('code')
        error = request.args.get('error')
        
        if error:
            return f"<html><body><h1>Authentication Error</h1><p>{error}</p></body></html>", 400
        
        if not state or not code:
            return "<html><body><h1>Authentication Error</h1><p>Missing state or code parameter</p></body></html>", 400
        
        # Determine provider from pending auth
        pending = auth_manager.pending_auth.get(state)
        if not pending:
            return "<html><body><h1>Authentication Error</h1><p>Invalid or expired state</p></body></html>", 400
        
        provider = pending['provider']
        
        print(f"Processing {provider} callback for state: {state}")
        
        if provider == 'google':
            result = auth_manager.handle_google_callback(state, code)
        elif provider == 'github':
            result = auth_manager.handle_github_callback(state, code)
        elif provider == 'microsoft':
            result = auth_manager.handle_microsoft_callback(state, code)
        else:
            return "<html><body><h1>Authentication Error</h1><p>Invalid provider</p></body></html>", 400
        
        print(f"Auth result: {result}")
        
        if result['success']:
            user = result['user']
            return f"""
            <html>
            <body>
                <h1>Authentication Successful!</h1>
                <p>Welcome, {user['name']}!</p>
                <p>You can now close this window and return to Godot.</p>
                <script>window.close();</script>
            </body>
            </html>
            """
        else:
            return f"<html><body><h1>Authentication Failed</h1><p>{result['error']}</p></body></html>", 400
            
    except Exception as e:
        return f"<html><body><h1>Authentication Error</h1><p>{str(e)}</p></body></html>", 500

@app.route('/auth/status', methods=['POST'])
def auth_status():
    """Check authentication status for a machine"""
    try:
        data = request.json
        machine_id = data.get('machine_id')
        require_provider = data.get('require_provider')
        allow_guest = data.get('allow_guest', True)
        
        if not machine_id:
            return jsonify({"error": "machine_id required", "success": False}), 400
        
        user_data = auth_manager.get_user_by_machine_id(machine_id)
        
        if user_data:
            user = user_data['user']
            # If a specific provider is required, ensure it matches
            if require_provider and user.get('provider') != require_provider:
                return jsonify({
                    "success": False,
                    "error": "Authenticated session exists but with a different provider",
                    "current_provider": user.get('provider')
                }), 401
            # Optionally disallow guest while polling for OAuth
            if not allow_guest and user.get('provider') == 'guest':
                return jsonify({
                    "success": False,
                    "error": "Guest session present; awaiting OAuth provider"
                }), 401
            return jsonify({
                "success": True,
                "user": user_data['user'],
                "token": user_data['token']
            })
        else:
            return jsonify({
                "success": False,
                "error": "Not authenticated"
            }), 401
            
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/auth/providers', methods=['GET'])
def auth_providers():
    """List available authentication providers (including guest)."""
    try:
        return jsonify({
            'success': True,
            'providers': auth_manager.get_available_providers()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/auth/guest', methods=['POST'])
def auth_guest():
    """Create or get a guest session for a machine."""
    try:
        data = request.json or {}
        machine_id = data.get('machine_id') or request.headers.get('X-Machine-ID')
        guest_name = data.get('guest_name') or request.headers.get('X-Guest-Name')
        if not machine_id:
            return jsonify({"success": False, "error": "machine_id required"}), 400
        result = auth_manager.create_or_get_guest_session(machine_id, guest_name)
        status = 200 if result.get('success') else 500
        return jsonify(result), status
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    """Logout user"""
    try:
        data = request.json
        machine_id = data.get('machine_id')
        user_id = data.get('user_id')
        
        if not machine_id:
            return jsonify({"error": "machine_id required", "success": False}), 400
        
        success = auth_manager.logout_user(machine_id, user_id)
        
        return jsonify({
            "success": success,
            "message": "Logged out successfully" if success else "No active session found"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/embed', methods=['POST'])
def embed_endpoint():
    # Optional server key gate
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    """
    Cloud embedding endpoint for managing project file embeddings
    
    Actions:
    - index_project: Index all project files
    - index_file: Index specific file
    - search: Search for similar files
    - status: Get project summary
    - clear: Clear project index
    """
    try:
        # Verify authentication
        user, error_response, status_code = verify_authentication()
        if error_response:
            return jsonify(error_response), status_code
        
        data = request.json
        action = data.get('action')
        project_root = data.get('project_root')
        project_id = data.get('project_id')
        
        if not action:
            return jsonify({"error": "No action specified"}), 400
        
        if not project_root:
            return jsonify({"error": "project_root required"}), 400
        
        # Generate project ID if not provided
        if not project_id:
            project_id = hashlib.md5(project_root.encode()).hexdigest()
        
        if not CLOUD_VECTOR_ENABLED:
            return jsonify({
                "success": False,
                "error": "Cloud vector indexing disabled (set GCP_PROJECT_ID or run in cloud mode)",
                "action": action
            }), 501

        if action == 'index_project':
            force_reindex = data.get('force_reindex', False)
            max_workers = data.get('max_workers')
            try:
                stats = cloud_vector_manager.index_project(project_root, user['id'], project_id, force_reindex, max_workers=max_workers)
            except TypeError:
                stats = cloud_vector_manager.index_project(project_root, user['id'], project_id, force_reindex)
            return jsonify({
                "success": True,
                "action": "index_project",
                "stats": stats,
                "project_id": project_id
            })
        
        elif action == 'index_file':
            file_path = data.get('file_path')
            if not file_path:
                return jsonify({"error": "file_path required for index_file action"}), 400
            
            indexed = cloud_vector_manager.index_file(file_path, user['id'], project_id, project_root)
            return jsonify({
                "success": True,
                "action": "index_file",
                "file_path": file_path,
                "indexed": indexed
            })
            
        elif action == 'index_files':
            # Cloud-ready batch file indexing
            files = data.get('files', [])
            if not files:
                return jsonify({"error": "files array required for index_files action"}), 400
            
            batch_info = data.get('batch_info', {})
            max_workers = data.get('max_workers')
            try:
                stats = cloud_vector_manager.index_files_with_content(files, user['id'], project_id, max_workers=max_workers)
            except TypeError:
                stats = cloud_vector_manager.index_files_with_content(files, user['id'], project_id)
            
            return jsonify({
                "success": True,
                "action": "index_files",
                "stats": stats,
                "batch_info": batch_info,
                "project_id": project_id
            })
        
        elif action == 'update_file':
            file_path = data.get('file_path')
            if not file_path:
                return jsonify({"error": "file_path required for update_file action"}), 400

            # update_file is equivalent to index_file with fresh content check
            indexed = cloud_vector_manager.index_file(file_path, user['id'], project_id, project_root)
            return jsonify({
                "success": True,
                "action": "update_file",
                "file_path": file_path,
                "indexed": indexed
            })

        elif action == 'remove_file':
            file_path = data.get('file_path')
            if not file_path:
                return jsonify({"error": "file_path required for remove_file action"}), 400

            removed = cloud_vector_manager.remove_file(user['id'], project_id, file_path)
            return jsonify({
                "success": removed,
                "action": "remove_file",
                "file_path": file_path
            })

        elif action == 'search':
            query = data.get('query')
            if not query:
                return jsonify({"error": "query required for search action"}), 400
            
            max_results = data.get('k', 5)
            include_graph = bool(data.get('include_graph', False))
            results = cloud_vector_manager.search(query, user['id'], project_id, max_results)
            # Filter out Godot sidecar UID files
            results = [r for r in results if not str(r.get('file_path','')).endswith('.uid')]
            
            return jsonify({
                "success": True,
                "action": "search",
                "query": query,
                "results": results,
                "graph": cloud_vector_manager.get_graph_context_for_files(
                    [r.get('file_path') for r in results], user['id'], project_id
                ) if include_graph else {}
            })
        
        elif action == 'status':
            stats = cloud_vector_manager.get_stats(user['id'], project_id)
            return jsonify({
                "success": True,
                "action": "status",
                "stats": stats,
                "project_id": project_id
            })
        
        elif action == 'clear':
            cloud_vector_manager.clear_project(user['id'], project_id)
            return jsonify({
                "success": True,
                "action": "clear",
                "message": "Project index cleared successfully"
            })
        
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
    
    except Exception as e:
        print(f"EMBED_ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/search_project', methods=['POST'])
def search_project():
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    """
    Search across project files using semantic similarity
    Used by the search_across_project tool
    """
    try:
        # Verify authentication
        user, error_response, status_code = verify_authentication()
        if error_response:
            return jsonify(error_response), status_code
        
        data = request.json
        query = data.get('query')
        project_root = data.get('project_root')
        project_id = data.get('project_id')
        
        # Fallback to header if not provided in body
        if not project_root:
            try:
                project_root = request.headers.get('X-Project-Root') or project_root
                print(f"SEARCH_PROJECT_HEADERS: X-Project-Root={request.headers.get('X-Project-Root')}")
            except Exception:
                pass

        if not query:
            return jsonify({"error": "Query required"}), 400
        if not project_root:
            return jsonify({"error": "project_root required (pass in body or X-Project-Root header)"}), 400

        # Generate project ID if not provided
        if not project_id:
            project_id = hashlib.md5(project_root.encode()).hexdigest()

        # Get search parameters
        max_results = data.get('max_results', 5)

        # Search using cloud vector manager
        if not CLOUD_VECTOR_ENABLED or cloud_vector_manager is None:
            return jsonify({
                "success": False,
                "error": "Cloud vector search disabled (set GCP_PROJECT_ID or run in cloud mode)"
            }), 501
        results = cloud_vector_manager.search(query, user['id'], project_id, max_results)
        # Filter out Godot sidecar UID files
        results = [r for r in results if not str(r.get('file_path','')).endswith('.uid')]

        # Format results
        formatted_results = {
            "similar_files": [
                {
                    "file_path": r['file_path'],
                    "similarity": r['similarity'],
                    "modality": "text",
                    "chunk_index": r['chunk']['chunk_index'] if r.get('chunk') else 0,
                    "chunk_start": r['chunk']['start_line'] if r.get('chunk') else None,
                    "chunk_end": r['chunk']['end_line'] if r.get('chunk') else None,
                }
                for r in results
            ],
            "central_files": [],
            "graph_summary": {}
        }

        return jsonify({
            "success": True,
            "query": query,
            "results": formatted_results,
            "include_graph": False
        })

    except Exception as e:
        print(f"SEARCH_PROJECT_ERROR: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    try:
        models = []
        for friendly_name, model_id in MODEL_MAP.items():
            models.append({
                "id": friendly_name,
                "name": friendly_name,
                "provider": model_id.split('/')[0] if '/' in model_id else 'unknown',
                "model_id": model_id
            })
        
        return jsonify({
            "success": True,
            "models": models,
            "default_model": DEFAULT_MODEL
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for testing"""
    return jsonify({
        "status": "healthy", 
        "service": "godot-ai-multi-model-service",
        "providers": ["openai", "anthropic", "google"],
        "available_models": list(MODEL_MAP.keys())
    })



if __name__ == '__main__':
    # Local dev only; in production use Gunicorn (configured in Dockerfile)
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)