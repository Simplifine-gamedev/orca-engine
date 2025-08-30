"""
© 2025 Simplifine Corp. Original backend contribution for this Godot fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""
from flask import Flask, request, Response, jsonify, redirect, session, stream_with_context, g
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
try:
    from weaviate_vector_manager import WeaviateVectorManager
except Exception:
    WeaviateVectorManager = None
try:
    from local_vector_manager import LocalVectorManager
except Exception:
    LocalVectorManager = None
from auth_manager import AuthManager

# Load environment variables from .env file
load_dotenv()

# Configure Vertex AI credentials and settings
VERTEX_AI_PROJECT = os.getenv('VERTEX_AI_PROJECT')
VERTEX_AI_LOCATION = os.getenv('VERTEX_AI_LOCATION', 'global')
VERTEX_AI_CREDENTIALS_PATH = os.getenv('VERTEX_AI_CREDENTIALS_PATH')

# Set up Vertex AI authentication
if VERTEX_AI_PROJECT:
    os.environ['VERTEXAI_PROJECT'] = VERTEX_AI_PROJECT
    os.environ['VERTEXAI_LOCATION'] = VERTEX_AI_LOCATION
    
    if VERTEX_AI_CREDENTIALS_PATH:
        # Use explicit credentials file if provided
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = VERTEX_AI_CREDENTIALS_PATH
        print(f"VERTEX_AI: Using credentials from {VERTEX_AI_CREDENTIALS_PATH}")
    else:
        # Use default GCP authentication (gcloud CLI credentials)
        print("VERTEX_AI: Using default GCP authentication (gcloud CLI credentials)")
    
    print(f"VERTEX_AI: Configured for project {VERTEX_AI_PROJECT} in location {VERTEX_AI_LOCATION}")
else:
    print("WARNING: VERTEX_AI_PROJECT not set - Vertex AI models will fail")

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

def _estimate_token_count(text: str) -> int:
    """Rough estimation of token count (approximately 4 characters per token)"""
    return len(text) // 4

def _manage_conversation_length_fallback(messages: list, model: str) -> list:
    """Manage conversation length to prevent token limit exceeded errors"""
    # Model-specific token limits (leaving safety margin)
    TOKEN_LIMITS = {
        "anthropic/claude-sonnet-4-20250514": 180000,  # 200k limit - 20k safety margin
        "openai/gpt-5": 120000,  # 128k limit - 8k safety margin  
        "openai/gpt-4o": 120000,  # 128k limit - 8k safety margin
        "openai/gpt-4-turbo": 120000,  # 128k limit - 8k safety margin
        "anthropic/claude-3-5-sonnet-20241022": 180000,  # 200k limit - 20k safety margin
    }
    
    limit = TOKEN_LIMITS.get(model, 100000)  # Default conservative limit
    
    # Calculate current token usage
    total_tokens = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = str(msg.get('content', ''))
            total_tokens += _estimate_token_count(content)
    
    if total_tokens <= limit:
        return messages  # No pruning needed
    
    print(f"CONVERSATION_PRUNE: Token count {total_tokens} exceeds limit {limit}, pruning conversation")
    
    # Keep system message (first) and recent messages, prune middle
    if len(messages) <= 3:
        return messages  # Too short to prune meaningfully
    
    system_msg = messages[0] if messages and messages[0].get('role') == 'system' else None
    recent_count = min(10, len(messages) // 2)  # Keep last 10 messages or half, whichever is smaller
    recent_messages = messages[-recent_count:]
    
    # Create pruned conversation with summary
    pruned_messages = []
    if system_msg:
        pruned_messages.append(system_msg)
    
    # Add context summary
    pruned_messages.append({
        "role": "assistant",
        "content": f"[Previous conversation context was automatically pruned due to length. Continuing from recent messages. Total messages pruned: {len(messages) - len(recent_messages) - (1 if system_msg else 0)}]"
    })
    
    # Add recent messages
    pruned_messages.extend(recent_messages)
    
    # Verify we're under the limit
    pruned_tokens = sum(_estimate_token_count(str(msg.get('content', ''))) for msg in pruned_messages)
    print(f"CONVERSATION_PRUNE: Reduced from {total_tokens} to {pruned_tokens} tokens ({len(messages)} to {len(pruned_messages)} messages)")
    
    return pruned_messages

app = Flask(__name__)

# Add request logging for debugging
@app.before_request
def log_request_info():
    print(f"DEBUG REQUEST: {request.method} {request.url} from {request.environ.get('REMOTE_ADDR')}")

# Secret must be stable across restarts in production. Require env in production, random only in DEV_MODE.
_dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'
DEPLOYMENT_MODE = os.getenv('DEPLOYMENT_MODE', 'oss').lower()  # 'oss' or 'cloud'
REQUIRE_SERVER_API_KEY = os.getenv('REQUIRE_SERVER_API_KEY', 'false').lower() == 'true'
SERVER_API_KEY = os.getenv('SERVER_API_KEY')

# Optional 3D Model Generation Service Integration
# Only enabled when all required environment variables are set
MODEL_3D_SERVICE_URL = os.getenv('MODEL_3D_SERVICE_URL')
MODEL_3D_SECRET_KEY = os.getenv('MODEL_3D_SECRET_KEY')  
MODEL_3D_ENABLED = (
    os.getenv('MODEL_3D_ENABLED', 'false').lower() == 'true' and
    MODEL_3D_SERVICE_URL and 
    MODEL_3D_SECRET_KEY
)
_secret_env = os.getenv('FLASK_SECRET_KEY')
if _secret_env:
    app.secret_key = _secret_env
elif _dev_mode:
    app.secret_key = os.urandom(24)
else:
    raise ValueError("FLASK_SECRET_KEY must be set in production")

# Multi-provider model configuration using LiteLLM
# Base models (always available)
BASE_MODEL_MAP = {
    "gemini-2.5": os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-pro"),
    "claude-4": os.getenv("CLAUDE_MODEL", f"vertex_ai/claude-sonnet-4@20250514"),
    "gpt-5": os.getenv("OPENAI_MODEL", "openai/gpt-5"),
    "gpt-4o": os.getenv("GPT4O_MODEL", "openai/gpt-4o"),
}

# Dynamic model map that includes base + cerebras models
MODEL_MAP = BASE_MODEL_MAP.copy()

def fetch_cerebras_models():
    """Fetch available models from Cerebras API"""
    cerebras_api_key = os.getenv('CEREBRAS_API_KEY')
    if not cerebras_api_key:
        print("WARNING: CEREBRAS_API_KEY not set - Cerebras models will not be available")
        return {}
    
    try:
        import requests
        response = requests.get(
            'https://api.cerebras.ai/v1/models',
            headers={'Authorization': f'Bearer {cerebras_api_key}'},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        cerebras_models = {}
        for model in data.get('data', []):
            model_id = model.get('id', '')
            if model_id:
                # Use simple display name for frontend
                display_name = f"[FAST] {model_id}"
                cerebras_models[display_name] = f"cerebras/{model_id}"
        
        print(f"CEREBRAS_MODELS: Loaded {len(cerebras_models)} models from API")
        return cerebras_models
    except Exception as e:
        print(f"WARNING: Failed to fetch Cerebras models: {e}")
        return {}

# Load Cerebras models at startup
cerebras_models = fetch_cerebras_models()
MODEL_MAP.update(cerebras_models)

# Ensure LiteLLM has access to Cerebras API key
cerebras_api_key = os.getenv('CEREBRAS_API_KEY')
if cerebras_api_key:
    # Make sure LiteLLM can access the Cerebras API key
    os.environ['CEREBRAS_API_KEY'] = cerebras_api_key
    print(f"CEREBRAS_SETUP: API key configured for LiteLLM")
else:
    print("WARNING: CEREBRAS_API_KEY not found in environment - Cerebras models will fail")

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
            result = MODEL_MAP[requested]
            return result
    except Exception:
        pass
    return MODEL_MAP[DEFAULT_MODEL]

def get_model_friendly_name(model_id: str) -> str:
    """Get friendly name for a model ID"""
    for friendly, real_id in MODEL_MAP.items():
        if real_id == model_id:
            return friendly
    # If not found in MODEL_MAP, check if it's a Cerebras model directly
    if model_id.startswith("cerebras/"):
        return f"[FAST] {model_id.replace('cerebras/', '')}"
    return model_id

# Initialize Authentication Manager
auth_manager = AuthManager()

# Initialize Vector Manager with priority: Weaviate -> Local
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
cloud_vector_manager = None

# Try Weaviate first (fastest option with advanced features)
if WEAVIATE_URL and WEAVIATE_API_KEY and WeaviateVectorManager and client is not None:
    try:
        cloud_vector_manager = WeaviateVectorManager(WEAVIATE_URL, WEAVIATE_API_KEY, client)
        print(f"VECTOR_INDEX: Using Weaviate at {WEAVIATE_URL} (function-level indexing + signal flows)")
    except Exception as e:
        print(f"VECTOR_INDEX: Weaviate init failed: {e}")

# Fallback to local
if cloud_vector_manager is None:
    if LocalVectorManager and client is not None:
        try:
            cloud_vector_manager = LocalVectorManager(client)
            print("VECTOR_INDEX: Using local JSON index (no external vector DB configured)")
        except Exception as e:
            print(f"VECTOR_INDEX ERROR: Failed to init LocalVectorManager: {e}")
    else:
        print("VECTOR_INDEX: LocalVectorManager unavailable or OpenAI client missing; semantic indexing disabled")

# Initialize Conversation Memory Manager with Weaviate
conversation_memory = None
try:
    from conversation_memory import ConversationMemoryManager
    conversation_memory = ConversationMemoryManager(weaviate_manager=cloud_vector_manager)
    if conversation_memory.enabled and cloud_vector_manager:
        print("CONVERSATION_MEMORY: Initialized with Weaviate backend")
    elif conversation_memory.enabled:
        print("CONVERSATION_MEMORY: Enabled but no Weaviate backend available")
    else:
        print("CONVERSATION_MEMORY: Disabled via configuration")
except Exception as e:
    print(f"CONVERSATION_MEMORY: Failed to initialize: {e}")

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
        size = arguments.get('size', '1024x1024')  # optional, may be arbitrary WxH
        # Exact pixel control parameters (optional)
        exact_size = arguments.get('exact_size') or arguments.get('size_exact')
        # Optional high-level spritesheet spec (lets the model follow an explicit layout)
        spritesheet = arguments.get('spritesheet') or {}
        tile_size = arguments.get('tile_size') or (spritesheet.get('tile_size') if isinstance(spritesheet, dict) else None)  # e.g., "32x32"
        grid = arguments.get('grid') or (spritesheet.get('grid') if isinstance(spritesheet, dict) else None)            # e.g., "2x2" (cols x rows)
        # Additional layout hints
        ss_order = (spritesheet.get('order') or 'row-major') if isinstance(spritesheet, dict) else 'row-major'
        ss_margin = int(spritesheet.get('margin') or 0) if isinstance(spritesheet, dict) else 0
        ss_spacing = int(spritesheet.get('spacing') or 0) if isinstance(spritesheet, dict) else 0
        ss_row_labels = spritesheet.get('row_labels') if isinstance(spritesheet, dict) else None
        ss_normalize_to = spritesheet.get('normalize_to') if isinstance(spritesheet, dict) else None
        resize_filter = (arguments.get('resize_filter') or '').lower()  # nearest|bilinear|bicubic|lanczos
        # For cloud safety: we do not write to local file systems from the server.
        # If provided, path_to_save is simply echoed back so the Godot editor can
        # save client-side after receiving the image.
        # Allow both 'path_to_save' and the more concise 'path'
        path_to_save = arguments.get('path_to_save') or arguments.get('path')

        print("IMAGE_OP DEBUG: Incoming arguments:")
        print(f"  - description len: {len(description)} | style: '{style}' | size: {size}")
        if exact_size:
            print(f"  - exact_size: {exact_size}")
        if tile_size or grid:
            print(f"  - tile_size: {tile_size} | grid: {grid}")
        if isinstance(image_ids, list):
            print(f"  - requested image ids: {image_ids} (count={len(image_ids)})")
        else:
            print(f"  - images field type: {type(image_ids)} -> {image_ids}")

        if not description:
            return {"success": False, "error": "No description provided for image operation"}

        prompt_text = description
        if style:
            prompt_text += f", {style} style"

        # If spritesheet spec is provided, append strict layout constraints for better consistency
        if isinstance(spritesheet, dict) and (grid or (spritesheet.get('rows') and spritesheet.get('cols'))):
            # Normalize grid string
            if not grid and spritesheet.get('rows') and spritesheet.get('cols'):
                grid = f"{int(spritesheet['cols'])}x{int(spritesheet['rows'])}"
            prompt_text += "\n\nSPRITESHEET CONSTRAINTS:" \
                           f"\n- Grid: {grid or 'unspecified'} ({ss_order})" \
                           f"\n- Tile size: {tile_size or 'consistent tiles'}" \
                           f"\n- Spacing: {ss_spacing}px, Margins: {ss_margin}px" \
                           "\n- Each tile must be contained entirely within its cell with uniform padding, aligned to a fixed grid, and with transparent background." \
                           "\n- Left-to-right within a row is chronological frame order. Top-to-bottom is row order." \
                           "\n- Keep palette and proportions consistent across all cells."
            if isinstance(ss_row_labels, list) and ss_row_labels:
                try:
                    # Render row labels as strict instructions
                    labels = ', '.join([str(x) for x in ss_row_labels])
                    prompt_text += f"\n- Row labels (top→bottom): {labels}"
                except Exception:
                    pass

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

        # Helpers for size parsing and provider compatibility
        def _parse_size_str(val: str | None) -> tuple[int | None, int | None]:
            try:
                if not val:
                    return None, None
                parts = str(val).lower().replace(' ', '').split('x')
                if len(parts) != 2:
                    return None, None
                return int(float(parts[0])), int(float(parts[1]))
            except Exception:
                return None, None

        def _compute_exact_size() -> tuple[int | None, int | None]:
            # Priority: exact_size -> tile_size+grid -> size if arbitrary WxH
            w, h = _parse_size_str(exact_size) if exact_size else (None, None)
            if w and h:
                return w, h
            if tile_size and grid:
                tw, th = _parse_size_str(tile_size)
                try:
                    gc, gr = [int(x) for x in str(grid).lower().split('x')]
                except Exception:
                    gc, gr = None, None
                if tw and th and gc and gr:
                    # Account for optional spacing/margins for better sheet planning
                    total_w = tw * gc + ss_spacing * (gc - 1) + ss_margin * 2
                    total_h = th * gr + ss_spacing * (gr - 1) + ss_margin * 2
                    return total_w, total_h
            # If size is an arbitrary WxH (not provider-supported), use that as target
            sw, sh = _parse_size_str(size)
            if sw and sh:
                return sw, sh
            return None, None

        def _choose_provider_size(target_w: int | None, target_h: int | None) -> str:
            # Only pass provider-supported sizes to avoid 400 errors.
            # Allowed: '1024x1024', '1024x1536' (portrait), '1536x1024' (landscape), and 'auto'.
            allowed = {"1024x1024", "1024x1536", "1536x1024", "auto"}
            # If the user requested an allowed value, honor it directly.
            if isinstance(size, str) and size.lower() in allowed:
                return size.lower()
            sw, sh = _parse_size_str(size)
            if sw and sh:
                candidate = f"{sw}x{sh}".lower()
                if candidate in allowed:
                    return candidate
            # Otherwise infer orientation from target exact size if available
            if target_w and target_h:
                if abs(float(target_w) / float(target_h) - 1.0) < 0.15:
                    return "1024x1024"
                return "1536x1024" if target_w > target_h else "1024x1536"
            # Fallback to square if no hints
            return "1024x1024"

        def _maybe_resize_b64_to_exact(b64_png: str, target_w: int | None, target_h: int | None) -> tuple[str, int | None, int | None]:
            if not b64_png or not (target_w and target_h):
                return b64_png, None, None
            try:
                raw = base64.b64decode(b64_png)
                im = Image.open(io.BytesIO(raw))
                if im.size == (target_w, target_h):
                    return b64_png, im.size[0], im.size[1]
                # Choose filter
                filt = Image.NEAREST if (resize_filter == 'nearest' or 'pixel' in (style or '').lower()) else (
                    Image.BILINEAR if resize_filter == 'bilinear' else (
                    Image.BICUBIC if resize_filter == 'bicubic' else Image.LANCZOS))
                resized = im.resize((int(target_w), int(target_h)), filt)
                out_buf = io.BytesIO()
                resized.save(out_buf, format='PNG')
                out_b64 = base64.b64encode(out_buf.getvalue()).decode('utf-8')
                return out_b64, resized.size[0], resized.size[1]
            except Exception as re:
                print(f"IMAGE_OP RESIZE WARNING: {re}")
                return b64_png, None, None

        # Determine target exact size and provider size
        t_w, t_h = _compute_exact_size()
        provider_size = _choose_provider_size(t_w, t_h)

        # If no images selected, do text-to-image generation
        if not selected_images:
            print("IMAGE_OP: Generating new image from prompt using Images API")
            gen = client.images.generate(model="gpt-image-1", prompt=prompt_text, size=provider_size)
            if not gen.data or not getattr(gen.data[0], 'b64_json', None):
                return {"success": False, "error": "Image generation returned no data"}

            image_base64 = gen.data[0].b64_json
            # Resize to exact target if requested
            image_base64, out_w, out_h = _maybe_resize_b64_to_exact(image_base64, t_w, t_h)
            result = {
                "success": True,
                "image_data": image_base64,
                "prompt": description,
                "style": style,
                "format": "png",
                "width": out_w,
                "height": out_h,
                "input_images": 0,
                "requested_images": len(image_ids)
            }
            # Provide a compact slice hint for downstream tools (frontend will use editor_introspect.slice_spritesheet)
            if grid or tile_size:
                result["slice_hint"] = {
                    "grid": grid,
                    "tile_size": tile_size or ss_normalize_to,
                    "normalize_to": ss_normalize_to or tile_size,
                    "order": ss_order,
                    "spacing": ss_spacing,
                    "margin": ss_margin,
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
                    edit = images_api.edits(model="gpt-image-1", image=img_fh, prompt=prompt_text, size=provider_size)
                else:
                    # Older SDKs
                    edit = images_api.edit(model="gpt-image-1", image=img_fh, prompt=prompt_text, size=provider_size)
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
        # Resize to exact target if requested
        image_base64, out_w, out_h = _maybe_resize_b64_to_exact(image_base64, t_w, t_h)
        result = {
            "success": True,
            "image_data": image_base64,
            "prompt": description,
            "style": style,
            "format": "png",
            "width": out_w,
            "height": out_h,
            "input_images": 1,
            "requested_images": len(image_ids)
        }
        if grid or tile_size:
            result["slice_hint"] = {
                "grid": grid,
                "tile_size": tile_size or ss_normalize_to,
                "normalize_to": ss_normalize_to or tile_size,
                "order": ss_order,
                "spacing": ss_spacing,
                "margin": ss_margin,
            }
        if path_to_save:
            result["path_to_save"] = path_to_save
        return result

    except Exception as e:
        print(f"IMAGE_OP ERROR: {str(e)}")
        return {"success": False, "error": f"Image operation failed: {str(e)}"}

# --- Backend Spritesheet Slicing Function ---
def slice_spritesheet_internal(arguments: dict) -> dict:
    """Robust spritesheet slicer (backend executed, no file writes).

    Args:
      - sheet_base64 (preferred) OR sheet_path (absolute/res:// on editor side)
      - tile_size: 'WxH' (optional if auto_detect)
      - grid: 'colsxrows' (optional if auto_detect)
      - margin, spacing: ints
      - auto_detect: bool (default True)
      - bg_tolerance: int (default 24)
      - alpha_threshold: int (default 1)
      - tight_crop: bool (default True)
      - padding: int (default 0)
      - fuzzy: int (default 2)
      - normalize_to: 'WxH' (optional; default tile_size)
    Returns:
      { success, frames:[{row,col,filename,width,height,base64_data}], grid_cols, grid_rows, tile_size, message }
    """
    try:
        b64 = arguments.get('sheet_base64')
        sheet_path = arguments.get('sheet_path')
        if not b64 and not sheet_path:
            return {"success": False, "error": "sheet_base64 or sheet_path required"}

        # Load image into PIL
        if b64:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert('RGBA')
        else:
            # Backend should avoid reading editor paths; this is best-effort for local dev
            img = Image.open(sheet_path).convert('RGBA')

        def _parse_wh(s):
            if not s:
                return None, None
            parts = str(s).lower().replace(' ', '').split('x')
            if len(parts) != 2:
                return None, None
            return int(float(parts[0])), int(float(parts[1]))

        tw, th = _parse_wh(arguments.get('tile_size'))
        nw, nh = _parse_wh(arguments.get('normalize_to'))
        grid = arguments.get('grid')
        cols = rows = 0
        if grid and isinstance(grid, str) and 'x' in grid:
            try:
                parts = [int(x) for x in grid.lower().split('x')]
                cols, rows = parts[0], parts[1]
            except Exception:
                cols = rows = 0
        margin = int(arguments.get('margin') or 0)
        spacing = int(arguments.get('spacing') or 0)
        auto_detect = bool(arguments.get('auto_detect', True))
        bg_tol = int(arguments.get('bg_tolerance') or 24)
        alpha_thresh = int(arguments.get('alpha_threshold') or 1)
        tight_crop = bool(arguments.get('tight_crop', True))
        padding = int(arguments.get('padding') or 0)
        fuzzy = int(arguments.get('fuzzy') or 2)

        if not nw or not nh:
            nw, nh = (tw or 32), (th or 32)

        W, H = img.size
        px = img.load()
        # Estimate background (average corners)
        corners = [px[0, 0], px[W-1, 0], px[0, H-1], px[W-1, H-1]]
        def _to_rgba(c):
            if len(c) == 4:
                return c
            return Image.new('RGBA', (1,1), c).getpixel((0,0))
        corners = [_to_rgba(c) for c in corners]
        bg = (
            sum(c[0] for c in corners)/4.0,
            sum(c[1] for c in corners)/4.0,
            sum(c[2] for c in corners)/4.0,
            sum((c[3] if len(c)>3 else 255) for c in corners)/4.0,
        )
        def _is_bg(c):
            if len(c) == 4 and c[3] <= alpha_thresh:
                return True
            return (abs(c[0]-bg[0]) <= bg_tol and abs(c[1]-bg[1]) <= bg_tol and abs(c[2]-bg[2]) <= bg_tol)

        # Auto grid/margins if requested or missing
        if auto_detect or cols <= 0 or rows <= 0 or not tw or not th:
            # Project to axes
            col_ne = [0]*W
            row_ne = [0]*H
            for x in range(W):
                col_ne[x] = 1 if any(not _is_bg(px[x, y]) for y in range(H)) else 0
            for y in range(H):
                row_ne[y] = 1 if any(not _is_bg(px[x, y]) for x in range(W)) else 0
            # Margins from outer empties
            left = 0
            while left < W and col_ne[left] == 0: left += 1
            right = W - 1
            while right >= 0 and col_ne[right] == 0: right -= 1
            top = 0
            while top < H and row_ne[top] == 0: top += 1
            bottom = H - 1
            while bottom >= 0 and row_ne[bottom] == 0: bottom -= 1
            if left < right and top < bottom:
                margin = max(margin, min(left, top))
            # Spacing via median empty run
            def _est_space(flags):
                gaps = []
                run = 0
                prev = False
                for f in flags:
                    if f == 0:
                        run += 1; prev = True
                    else:
                        if prev and run > 0: gaps.append(run)
                        run = 0; prev = False
                return int(sorted(gaps)[len(gaps)//2]) if gaps else 0
            if spacing == 0:
                spacing = max(0, min(_est_space(col_ne), _est_space(row_ne)))
            # Infer cols/rows from usable area if missing
            if not tw or not th:
                # approximate from dominant non-empty stride
                tw = tw or max(8, (right-left+1)//3)
                th = th or max(8, (bottom-top+1)//3)
            if cols <= 0 or rows <= 0:
                usable_w = W - margin*2 + spacing
                usable_h = H - margin*2 + spacing
                cols = max(1, (usable_w + spacing)//(tw + spacing))
                rows = max(1, (usable_h + spacing)//(th + spacing))

        frames = []
        for r in range(rows):
            for c in range(cols):
                ox = margin + c*(tw + spacing)
                oy = margin + r*(th + spacing)
                fx = max(0, ox - fuzzy)
                fy = max(0, oy - fuzzy)
                fw = min(W - fx, tw + fuzzy*2)
                fh = min(H - fy, th + fuzzy*2)
                if fw <= 0 or fh <= 0:
                    continue
                cell = img.crop((fx, fy, fx+fw, fy+fh))
                if tight_crop:
                    # alpha-based crop
                    cp = cell.load()
                    cw, ch = cell.size
                    minx, miny, maxx, maxy = cw, ch, -1, -1
                    for yy in range(ch):
                        for xx in range(cw):
                            a = cp[xx, yy][3] if len(cp[xx, yy]) == 4 else 255
                            if a > alpha_thresh:
                                if xx < minx: minx = xx
                                if yy < miny: miny = yy
                                if xx > maxx: maxx = xx
                                if yy > maxy: maxy = yy
                    if maxx >= minx and maxy >= miny:
                        cell = cell.crop((minx, miny, maxx+1, maxy+1))
                # Center on normalized canvas
                canvas = Image.new('RGBA', (nw + padding*2, nh + padding*2), (0,0,0,0))
                dx = (canvas.size[0] - cell.size[0])//2
                dy = (canvas.size[1] - cell.size[1])//2
                canvas.alpha_composite(cell, (dx, dy))
                out_buf = io.BytesIO()
                canvas.save(out_buf, format='PNG')
                frames.append({
                    'row': r,
                    'col': c,
                    'filename': f'frame_{r:02d}_{c:02d}.png',
                    'width': canvas.size[0],
                    'height': canvas.size[1],
                    'base64_data': base64.b64encode(out_buf.getvalue()).decode('utf-8')
                })

        return {
            'success': True,
            'frames': frames,
            'grid_cols': cols,
            'grid_rows': rows,
            'tile_size': f"{nw}x{nh}",
            'message': f'Sliced {len(frames)} frames ({cols}x{rows})'
        }
    except Exception as e:
        print(f"SLICE_SPRITESHEET_ERROR: {e}")
        return {"success": False, "error": str(e)}
# --- Graph-Enhanced Search Intelligence ---
def _enhance_search_with_graph(initial_results: list, query: str, user_id: str, project_id: str, 
                              max_results: int, vector_manager, include_graph: bool) -> dict:
    """
    GRAPH-ENHANCED RANKING: The heart of intelligent Godot project search
    
    Combines:
    1. Semantic similarity (embeddings) 
    2. Graph centrality (structural importance)
    3. Relationship strength (how files connect)
    4. Godot-specific intelligence (scene-script pairs, etc.)
    """
    if not initial_results:
        return {"similar_files": [], "central_files": [], "graph_summary": {}}
    
    try:
        # Step 1: Get all file paths for graph analysis
        file_paths = [r['file_path'] for r in initial_results]
        
        # Step 2: Get graph context for all files
        graph_context = {}
        if include_graph:
            graph_context = vector_manager.get_graph_context_for_files(file_paths, user_id, project_id)
        
        # Step 3: Calculate enhanced scores
        enhanced_files = []
        for result in initial_results:
            file_path = result['file_path']
            base_similarity = result['similarity']
            
            # Calculate centrality score (how connected this file is)
            centrality_score = _calculate_centrality_score(file_path, graph_context)
            
            # Calculate relationship strength to other results
            relationship_score = _calculate_relationship_score(file_path, file_paths, graph_context)
            
            # Add Godot-specific intelligence boost
            godot_boost = _calculate_godot_context_boost(file_path, query, file_paths)
            
            # ENHANCED RANKING FORMULA
            enhanced_score = (
                base_similarity * 0.5 +          # Semantic similarity (primary)
                centrality_score * 0.2 +         # Structural importance  
                relationship_score * 0.2 +       # Connection strength
                godot_boost * 0.1                # Godot-specific intelligence
            )
            
            # Create explanation for transparency
            explanation_parts = []
            if centrality_score > 0.1:
                explanation_parts.append(f"Hub file ({centrality_score:.2f} centrality)")
            if relationship_score > 0.1:
                explanation_parts.append(f"Connected to results ({relationship_score:.2f})")
            if godot_boost > 0.1:
                explanation_parts.append(f"Godot pattern match ({godot_boost:.2f})")
            
            ranking_explanation = "; ".join(explanation_parts) if explanation_parts else "Semantic match"
            
            enhanced_result = result.copy()
            enhanced_result.update({
                'enhanced_score': enhanced_score,
                'centrality_score': centrality_score,
                'relationship_score': relationship_score,
                'godot_boost': godot_boost,
                'ranking_explanation': ranking_explanation
            })
            enhanced_files.append(enhanced_result)
        
        # Step 4: MULTI-HOP CONTEXT EXPANSION
        # Add contextually relevant files through intelligent graph traversal
        expanded_files = _expand_with_multi_hop_context(enhanced_files, graph_context, query)
        
        # Step 5: Sort by enhanced score and limit to max_results
        expanded_files.sort(key=lambda x: x['enhanced_score'], reverse=True)
        final_results = expanded_files[:max_results]
        
        # Step 6: Find central files (high centrality, architectural importance)
        central_files = []
        if include_graph:
            # Include both original and expanded files for centrality analysis
            all_expanded_paths = [f['file_path'] for f in expanded_files]
            # Get files already in similar_files to avoid duplication
            similar_file_paths = {f['file_path'] for f in final_results}
            
            for file_path in all_expanded_paths:
                # Skip if already in similar_files section
                if file_path in similar_file_paths:
                    continue
                    
                centrality = _calculate_centrality_score(file_path, graph_context)
                if centrality > 0.3:  # High centrality threshold
                    central_files.append({
                        'file_path': file_path,
                        'centrality': centrality,
                        'connections': len(graph_context.get(file_path, {}).get('edges', [])),
                        'role': _identify_architectural_role(file_path, graph_context)
                    })
        
        central_files.sort(key=lambda x: x['centrality'], reverse=True)
        
        # Step 7: Generate graph summary
        total_files = len(file_paths)
        total_connections = sum(len(ctx.get('edges', [])) for ctx in graph_context.values())
        avg_centrality = sum(_calculate_centrality_score(fp, graph_context) for fp in file_paths) / max(1, len(file_paths))
        
        graph_summary = {
            'total_files': total_files,
            'total_connections': total_connections,
            'avg_centrality': avg_centrality,
            'architecture_detected': len(central_files) > 0
        }
        
        return {
            "similar_files": final_results,
            "central_files": central_files[:5],  # Limit central files
            "graph_summary": graph_summary
        }
        
    except Exception as e:
        print(f"GRAPH_ENHANCE_ERROR: {e}")
        # Fallback to original results on error
        return {
            "similar_files": initial_results[:max_results], 
            "central_files": [], 
            "graph_summary": {}
        }

def _calculate_centrality_score(file_path: str, graph_context: dict) -> float:
    """Calculate how central/important a file is in the project graph"""
    file_ctx = graph_context.get(file_path, {})
    edges = file_ctx.get('edges', [])
    
    if not edges:
        return 0.0
    
    # Simple degree centrality (could be enhanced with PageRank later)
    connection_count = len(edges)
    
    # Weight by relationship types
    weighted_score = 0.0
    for edge in edges:
        rel_type = edge.get('type', 'reference')
        if rel_type == 'extends':
            weighted_score += 1.5  # Inheritance is important
        elif rel_type == 'preload':
            weighted_score += 1.2  # Preloads indicate dependency
        elif rel_type == 'scene_ref':
            weighted_score += 1.0  # Scene references
        else:
            weighted_score += 0.8  # General references
    
    # Normalize to 0-1 range (assuming max ~20 connections for most files)
    return min(1.0, weighted_score / 20.0)

def _calculate_relationship_score(file_path: str, all_files: list, graph_context: dict) -> float:
    """Calculate how strongly this file relates to other search results"""
    file_ctx = graph_context.get(file_path, {})
    edges = file_ctx.get('edges', [])
    
    if not edges:
        return 0.0
    
    # Count connections to other files in the result set
    connections_to_results = 0
    for edge in edges:
        target = edge.get('target')
        source = edge.get('source')
        connected_file = target if target != file_path else source
        
        if connected_file in all_files:
            connections_to_results += edge.get('weight', 1.0)
    
    # Normalize by total result count
    return min(1.0, connections_to_results / max(1, len(all_files)))

def _calculate_godot_context_boost(file_path: str, query: str, all_files: list) -> float:
    """
    GODOT-SPECIFIC INTELLIGENCE: Boost files based on patterns and HP system awareness
    
    Enhanced with SIGNAL FLOW INTELLIGENCE for HP system scenarios!
    """
    boost = 0.0
    file_lower = file_path.lower()
    query_lower = query.lower()
    
    # HP SYSTEM INTELLIGENCE - Critical for adding HP to one-hit games!
    hp_damage_terms = ['health', 'hp', 'damage', 'hit', 'hurt', 'collision', 'die', 'death', 'game_over', 'gameover']
    if any(term in query_lower for term in hp_damage_terms):
        print(f"HP_SYSTEM_QUERY: Analyzing {file_path} for HP system relevance")
        
        # PLAYER FILES - Central to HP systems
        if 'player' in file_lower:
            boost += 0.6  # Major boost - player files are critical for HP
            print(f"HP_BOOST: Player file {file_path} +0.6")
        
        # ENEMY/DAMAGE SOURCES - Need to understand what currently kills player
        if any(enemy in file_lower for enemy in ['enemy', 'bullet', 'projectile', 'hazard', 'trap']):
            boost += 0.5  # Major boost - damage sources
            print(f"HP_BOOST: Damage source {file_path} +0.5")
        
        # UI FILES - Need to add health bars/displays
        if any(ui in file_lower for ui in ['ui', 'hud', 'health', 'bar', 'display', 'interface']):
            boost += 0.4  # Important for HP display
            print(f"HP_BOOST: UI file {file_path} +0.4")
        
        # GAME MANAGER - Handles game over logic that needs to change  
        if any(mgr in file_lower for mgr in ['game', 'main', 'manager', 'controller']):
            boost += 0.4  # Important for game state management
            print(f"HP_BOOST: Game manager {file_path} +0.4")
        
        # COLLISION/DAMAGE LOGIC - The core of what needs to change
        if any(col in file_lower for col in ['collision', 'damage', 'hit', 'area']):
            boost += 0.5  # Critical - this is where one-hit logic likely lives
            print(f"HP_BOOST: Collision/damage logic {file_path} +0.5")
    
    # Scene-Script pair detection
    base_name = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
    script_pair = f"{base_name}.gd"
    scene_pair = f"{base_name}.tscn"
    
    if file_path.endswith('.gd') and scene_pair in all_files:
        boost += 0.3  # Script with matching scene
    elif file_path.endswith('.tscn') and script_pair in all_files:
        boost += 0.3  # Scene with matching script
    
    # ENHANCED Query-specific boosting
    if 'player' in query_lower:
        if 'player' in file_lower:
            boost += 0.4
    elif 'movement' in query_lower or 'move' in query_lower:
        if any(word in file_lower for word in ['movement', 'move', 'kinematic', 'character']):
            boost += 0.3
    elif 'ui' in query_lower or 'interface' in query_lower:
        if any(word in file_lower for word in ['ui', 'menu', 'button', 'panel', 'interface', 'hud']):
            boost += 0.3
    elif 'audio' in query_lower or 'sound' in query_lower:
        if any(word in file_lower for word in ['audio', 'sound', 'music', 'sfx']):
            boost += 0.3
    elif 'signal' in query_lower or 'connect' in query_lower:
        # SIGNAL SYSTEM QUERIES - boost files likely to contain signal logic
        if any(sig in file_lower for sig in ['player', 'game', 'manager', 'controller']):
            boost += 0.4
    
    # File type relevance
    if 'script' in query_lower and file_path.endswith('.gd'):
        boost += 0.2
    elif 'scene' in query_lower and file_path.endswith('.tscn'):
        boost += 0.2
    elif 'resource' in query_lower and file_path.endswith(('.tres', '.res')):
        boost += 0.2
    
    # Architectural patterns
    if any(pattern in file_lower for pattern in ['manager', 'controller', 'system', 'service']):
        boost += 0.2  # Likely architectural components
    
    return min(1.0, boost)  # Cap at 1.0

def _expand_with_multi_hop_context(enhanced_files: list, graph_context: dict, query: str) -> list:
    """
    MULTI-HOP CONTEXT EXPANSION: Intelligently traverse the graph to find related files
    
    This finds files that should be included based on:
    1. Scene-Script pairs (if you find Player.gd, also include Player.tscn)  
    2. Base classes and extensions
    3. Dependencies and dependents
    4. Related components in the same architectural layer
    """
    if not graph_context:
        return enhanced_files
    
    try:
        existing_files = {f['file_path'] for f in enhanced_files}
        expansion_candidates = {}  # file_path -> {'reason': str, 'score': float, 'source': str}
        
        for result in enhanced_files:
            file_path = result['file_path']
            file_ctx = graph_context.get(file_path, {})
            edges = file_ctx.get('edges', [])
            
            # 1. SCENE-SCRIPT PAIR EXPANSION
            base_name = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
            if file_path.endswith('.gd'):
                # Script found, look for matching scene
                scene_pair = f"{base_name}.tscn"
                if scene_pair not in existing_files:
                    _add_expansion_candidate(expansion_candidates, scene_pair, 
                                           "Scene-Script Pair", 0.8, file_path)
            elif file_path.endswith('.tscn'):
                # Scene found, look for matching script
                script_pair = f"{base_name}.gd"
                if script_pair not in existing_files:
                    _add_expansion_candidate(expansion_candidates, script_pair,
                                           "Scene-Script Pair", 0.8, file_path)
            
            # 2. INHERITANCE CHAIN EXPANSION
            for edge in edges:
                if edge.get('type') == 'extends':
                    target = edge.get('target')
                    source = edge.get('source')
                    related_file = target if target != file_path else source
                    
                    if related_file and related_file not in existing_files:
                        if edge.get('source') == file_path:
                            # This file extends another (find base class)
                            _add_expansion_candidate(expansion_candidates, related_file,
                                                   "Base Class", 0.7, file_path)
                        else:
                            # Another file extends this (find derived class)
                            _add_expansion_candidate(expansion_candidates, related_file,
                                                   "Derived Class", 0.6, file_path)
            
            # 3. DEPENDENCY EXPANSION
            for edge in edges:
                if edge.get('type') in ['preload', 'load']:
                    target = edge.get('target')
                    source = edge.get('source')
                    related_file = target if target != file_path else source
                    
                    if related_file and related_file not in existing_files:
                        weight = edge.get('weight', 1.0)
                        reason = "Preloaded Dependency" if edge.get('type') == 'preload' else "Dynamic Dependency"
                        _add_expansion_candidate(expansion_candidates, related_file,
                                               reason, 0.5 * weight, file_path)
            
            # 4. ARCHITECTURAL LAYER EXPANSION
            # If we find a manager, look for related systems
            if 'manager' in file_path.lower() or 'controller' in file_path.lower():
                for edge in edges:
                    target = edge.get('target')
                    source = edge.get('source')
                    related_file = target if target != file_path else source
                    
                    if (related_file and related_file not in existing_files and
                        any(pattern in related_file.lower() for pattern in ['system', 'service', 'handler'])):
                        _add_expansion_candidate(expansion_candidates, related_file,
                                               "Related System", 0.4, file_path)
        
        # 5. QUERY-SPECIFIC INTELLIGENT EXPANSION
        query_lower = query.lower()
        if 'player' in query_lower:
            # Look for player-related files in the graph
            for file_path, ctx in graph_context.items():
                if (file_path not in existing_files and 'player' in file_path.lower()):
                    _add_expansion_candidate(expansion_candidates, file_path,
                                           "Query Pattern Match", 0.6, "query_expansion")
        
        # 6. Convert candidates to enhanced results
        expanded_results = enhanced_files.copy()
        for candidate_path, info in expansion_candidates.items():
            # Create a synthetic result for the expanded file
            expanded_result = {
                'file_path': candidate_path,
                'similarity': 0.0,  # No direct semantic match
                'enhanced_score': info['score'] * 0.8,  # Slightly lower than direct matches
                'centrality_score': _calculate_centrality_score(candidate_path, graph_context),
                'relationship_score': info['score'],
                'godot_boost': _calculate_godot_context_boost(candidate_path, query, list(existing_files)),
                'ranking_explanation': f"Expanded: {info['reason']} (from {info['source']})",
                'expansion_source': info['source'],
                'expansion_reason': info['reason']
            }
            
            # Recalculate enhanced score with expansion context
            expanded_result['enhanced_score'] = (
                expanded_result['similarity'] * 0.3 +          # Lower semantic weight for expansions
                expanded_result['centrality_score'] * 0.3 +    # Higher structural weight  
                expanded_result['relationship_score'] * 0.3 +  # Higher relationship weight
                expanded_result['godot_boost'] * 0.1
            )
            
            expanded_results.append(expanded_result)
        
        print(f"MULTI_HOP: Expanded {len(enhanced_files)} results to {len(expanded_results)} with {len(expansion_candidates)} contextual additions")
        return expanded_results
        
    except Exception as e:
        print(f"MULTI_HOP_ERROR: {e}")
        return enhanced_files  # Fallback to original results

def _add_expansion_candidate(candidates: dict, file_path: str, reason: str, score: float, source: str):
    """Helper to add or update expansion candidates with best score"""
    if file_path not in candidates or candidates[file_path]['score'] < score:
        candidates[file_path] = {
            'reason': reason,
            'score': score, 
            'source': source
        }

def _identify_architectural_role(file_path: str, graph_context: dict) -> str:
    """Identify the architectural role of a file based on its connections"""
    file_ctx = graph_context.get(file_path, {})
    edges = file_ctx.get('edges', [])
    file_lower = file_path.lower()
    
    # Count incoming vs outgoing connections
    incoming = len([e for e in edges if e.get('target') == file_path])
    outgoing = len([e for e in edges if e.get('source') == file_path])
    
    # Pattern matching
    if 'singleton' in file_lower or 'autoload' in file_lower:
        return 'Singleton/Autoload'
    elif 'manager' in file_lower and outgoing > incoming:
        return 'System Manager'
    elif 'base' in file_lower or 'abstract' in file_lower:
        return 'Base Class'
    elif incoming > outgoing * 2:
        return 'Dependency Hub'
    elif outgoing > incoming * 2:
        return 'Consumer'
    elif file_path.endswith('.tscn'):
        return 'Scene'
    elif file_path.endswith('.gd'):
        return 'Script'
    else:
        return 'Resource'

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
        trace_dependencies = bool(arguments.get('trace_dependencies', False))
        search_mode = arguments.get('search_mode', 'semantic')
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
        
        # Smart search mode detection with proper logging
        detected_mode = search_mode
        if search_mode == 'semantic':
            query_lower = query.lower()
            
            # Explicit keyword request detection
            if any(word in query_lower for word in ['keyword', 'exact', 'literal', 'find exact']):
                detected_mode = 'keyword'
                print(f"SEARCH_MODE_DETECTION: User explicitly requested KEYWORD search: {query}")
            
            # Hybrid search detection (semantic + keyword)
            elif (any(pattern in query_lower for pattern in ['func ', 'function ', 'def ', 'class ', 'signal ']) or
                  any(api in query_lower for api in ['set_velocity', 'move_and_slide', 'get_node', 'emit_signal']) or
                  '"' in query or "'" in query or
                  # Common exact searches
                  query_lower in ['_ready', '_process', '_physics_process', '_input', 'add_vectors', 'try_jump']):
                detected_mode = 'hybrid'
                print(f"SEARCH_MODE_DETECTION: Auto-detected HYBRID search: {query}")
            else:
                print(f"SEARCH_MODE_DETECTION: Using SEMANTIC search: {query}")
        else:
            print(f"SEARCH_MODE_DETECTION: Using explicit {detected_mode.upper()} search: {query}")
        
        # Search using cloud vector manager with detected/selected mode
        if detected_mode == 'keyword':
            # Extract the actual search term from keyword requests
            search_term = query
            query_lower = query.lower()
            
            # Clean up keyword search queries
            if 'keyword' in query_lower:
                # Extract term after "keyword"
                parts = query_lower.split('keyword')
                if len(parts) > 1:
                    search_term = parts[1].strip().split()[0] if parts[1].strip() else query
            elif 'search for' in query_lower:
                # Extract term after "search for"
                parts = query_lower.split('search for')
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    if remaining.startswith('keyword'):
                        remaining = remaining.replace('keyword', '').strip()
                    search_term = remaining.split()[0] if remaining else query
            
            print(f"SEARCH_EXECUTION: Keyword search - extracted term '{search_term}' from query '{query}'")
            initial_results = cloud_vector_manager.keyword_search(search_term, user['id'], project_id, max_results * 2)
            print(f"SEARCH_EXECUTION: Performed pure KEYWORD search")
        elif detected_mode == 'hybrid' and hasattr(cloud_vector_manager, 'hybrid_search'):
            initial_results = cloud_vector_manager.hybrid_search(query, user['id'], project_id, max_results * 2)
            print(f"SEARCH_EXECUTION: Performed HYBRID search")
        elif trace_dependencies and hasattr(cloud_vector_manager, 'search_with_dependency_context'):
            # Use enhanced search with dependency tracing
            initial_results = cloud_vector_manager.search_with_dependency_context(
                query, user['id'], project_id, max_results * 2, include_dependencies=True
            )
            print(f"SEARCH_EXECUTION: Performed DEPENDENCY-TRACED search")
        else:
            # Use standard semantic search
            initial_results = cloud_vector_manager.search(query, user['id'], project_id, max_results * 2)  # Get more for reranking
            print(f"SEARCH_EXECUTION: Performed standard SEMANTIC search")
        
        # NUCLEAR FILTER: Remove .import files from ALL search results (in case old junk persists in database)
        initial_results = [r for r in initial_results if not r.get('file_path', '').endswith('.import')]
        print(f"SEARCH_FILTER: Blocked .import files, {len(initial_results)} results remaining")
        
        # GRAPH-ENHANCED RANKING: Combine semantic similarity with graph intelligence
        enhanced_results = _enhance_search_with_graph(
            initial_results, query, user['id'], project_id, max_results, 
            cloud_vector_manager, include_graph
        )
        
        # Format results with enhanced scoring
        formatted_results = {
            "similar_files": [
                {
                    "file_path": r['file_path'],
                    "similarity": r['similarity'],
                    "enhanced_score": r.get('enhanced_score', r['similarity']),
                    "centrality_score": r.get('centrality_score', 0.0),
                    "relationship_score": r.get('relationship_score', 0.0),
                    "godot_boost": r.get('godot_boost', 0.0),
                    "ranking_explanation": r.get('ranking_explanation', ''),
                    "modality": "text",
                    "chunk_index": r['chunk']['chunk_index'] if r.get('chunk') else 0,
                    "chunk_start": r['chunk']['start_line'] if r.get('chunk') else None,
                    "chunk_end": r['chunk']['end_line'] if r.get('chunk') else None,
                    "line_count": r.get('file_line_count')
                }
                for r in enhanced_results['similar_files']
            ],
            "central_files": enhanced_results.get('central_files', []),
            "graph_summary": enhanced_results.get('graph_summary', {})
        }
        
        # Get graph context for final results
        graph_context = {}
        if include_graph and enhanced_results['similar_files']:
            files = [r['file_path'] for r in enhanced_results['similar_files']]
            graph_context = cloud_vector_manager.get_graph_context_for_files(files, user['id'], project_id)
        
        return {
            "success": True,
            "query": query,
            "search_mode": detected_mode,
            "results": formatted_results,
            "include_graph": include_graph,
            "trace_dependencies": trace_dependencies,
            "graph": graph_context,
            "file_count": len(enhanced_results['similar_files']),
            "message": f"Found {len(enhanced_results['similar_files'])} relevant files using {detected_mode.upper()} search for query: {query}"
        }
        
    except Exception as e:
        print(f"SEARCH_PROJECT_INTERNAL_ERROR: {e}")
        return {"success": False, "error": f"Search failed: {str(e)}"}

# --- Game Testing and Error Analysis Tools ---

def start_game_internal(arguments: dict) -> dict:
    """Start the game for testing - frontend only operation"""
    return {
        "success": False,
        "frontend_only": True,
        "message": "Game control is only available from the Godot editor frontend. Use the 'run_scene' tool in the editor.",
        "suggested_tool": "run_scene",
        "arguments_to_forward": arguments
    }

def stop_game_internal(arguments: dict) -> dict:
    """Stop the running game - frontend only operation"""
    return {
        "success": False, 
        "frontend_only": True,
        "message": "Game control is only available from the Godot editor frontend. Use the 'stop_game' tool in the editor.",
        "suggested_tool": "stop_game",
        "arguments_to_forward": arguments
    }

def get_game_status_internal(arguments: dict) -> dict:
    """Get game status - frontend only operation"""
    return {
        "success": False,
        "frontend_only": True, 
        "message": "Game status is only available from the Godot editor frontend. Use the 'get_game_status' tool in the editor.",
        "suggested_tool": "get_game_status",
        "arguments_to_forward": arguments
    }

def get_runtime_errors_summary_internal(arguments: dict) -> dict:
    """Get runtime errors summary - frontend only operation"""
    return {
        "success": False,
        "frontend_only": True,
        "message": "Runtime error analysis is only available from the Godot editor frontend. Use the 'get_runtime_errors_summary' tool in the editor.",
        "suggested_tool": "get_runtime_errors_summary", 
        "arguments_to_forward": arguments,
        "note": "This tool provides smart error deduplication showing total counts, unique error types, and frequency analysis."
    }

def get_runtime_errors_detailed_internal(arguments: dict) -> dict:
    """Get detailed runtime errors - frontend only operation"""
    return {
        "success": False,
        "frontend_only": True,
        "message": "Detailed runtime error analysis is only available from the Godot editor frontend. Use the 'get_runtime_errors_detailed' tool in the editor.",
        "suggested_tool": "get_runtime_errors_detailed",
        "arguments_to_forward": arguments,
        "note": "This tool provides filtered error details with options for grouping duplicates and searching by message content."
    }

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
    elif function_name == "search_across_godot_docs":
        return search_across_godot_docs_internal(arguments)
    elif function_name == "search_godot_assets":
        return search_godot_assets_internal(arguments)
    elif function_name == "install_godot_asset":
        return install_godot_asset_internal(arguments)
    # Note: Game testing tools (start_game, stop_game, etc.) are frontend-only and not executed in backend
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
            "description": "Search the latest Godot documentation (tutorials and class reference) using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find in the docs (natural language)."},
                    "max_results": {"type": "integer", "description": "Maximum results (default 5)", "default": 5}
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

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a user (frontend handles the actual clearing)"""
    # This endpoint exists mainly for potential future server-side conversation management
    # Currently, conversation clearing is handled by the frontend
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
        
    print(f"CONVERSATION_CLEAR: Clear conversation request from user {user.get('id', 'unknown')}")
    return jsonify({
        "success": True, 
        "message": "Conversation clear signal received",
        "note": "Conversation clearing is handled by the frontend"
    })

@app.route('/memory_stats', methods=['GET'])
def get_memory_stats():
    """Get conversation memory management statistics"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        if conversation_memory:
            stats = conversation_memory.get_stats()
            return jsonify({
                "success": True,
                "stats": stats,
                "message": "Memory statistics retrieved successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Conversation memory not initialized",
                "stats": {"enabled": False, "weaviate_connected": False}
            })
    except Exception as e:
        print(f"MEMORY_STATS_ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/memory_cleanup', methods=['POST'])
def cleanup_memory():
    """Clean up old conversation summaries"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        if not conversation_memory:
            return jsonify({"success": False, "error": "Conversation memory not initialized"}), 500
            
        data = request.get_json() or {}
        days_old = data.get('days_old')  # Use None to trigger config default
        
        from memory_config import MemoryConfig
        actual_days = days_old if days_old is not None else MemoryConfig.CLEANUP_DAYS_DEFAULT
        conversation_memory.cleanup_old_summaries(days_old)
        return jsonify({
            "success": True,
            "message": f"Cleaned up summaries older than {actual_days} days"
        })
    except Exception as e:
        print(f"MEMORY_CLEANUP_ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/summarize_conversation', methods=['POST'])
def summarize_conversation():
    """Summarize a chunk of conversation messages using AI"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        if not conversation_memory:
            return jsonify({"success": False, "error": "Conversation memory not initialized"}), 500
        
        if not conversation_memory.enabled:
            return jsonify({"success": False, "error": "Conversation summarization disabled"}), 400
            
        data = request.get_json() or {}
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"success": False, "error": "No messages provided"}), 400
            
        user_id = user.get('id', 'unknown') if user else 'unknown'
        
        # Create summary using AI models
        import asyncio
        summary = asyncio.run(conversation_memory.summarize_conversation_chunk(messages, user_id))
        
        return jsonify({
            "success": True,
            "summary": summary,
            "original_message_count": len(messages),
            "summary_tokens": conversation_memory.estimate_tokens(summary),
            "message": "Conversation summarized successfully"
        })
        
    except Exception as e:
        print(f"SUMMARIZE_CONVERSATION_ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/update_conversation_summary', methods=['POST'])
def update_conversation_summary():
    """Update summary when messages are edited"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        if not conversation_memory:
            return jsonify({"success": False, "error": "Conversation memory not initialized"}), 500
        
        if not conversation_memory.enabled:
            return jsonify({"success": False, "error": "Conversation summarization disabled"}), 400
            
        data = request.get_json() or {}
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"success": False, "error": "No messages provided"}), 400
            
        user_id = user.get('id', 'unknown') if user else 'unknown'
        
        # Update summary for edited messages
        import asyncio
        result = asyncio.run(conversation_memory.update_summary_for_edited_messages(messages, user_id))
        
        if result["success"]:
            return jsonify({
                "success": True,
                "summary": result["summary"],
                "was_updated": result["was_updated"],
                "previous_summary_found": result["previous_summary_found"],
                "original_message_count": len(messages),
                "summary_tokens": conversation_memory.estimate_tokens(result["summary"]),
                "message": result["message"]
            })
        else:
            return jsonify(result), 500
        
    except Exception as e:
        print(f"UPDATE_CONVERSATION_ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/search_conversation_history', methods=['POST']) 
def search_conversation_history():
    """Search for similar conversations in history"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        if not conversation_memory:
            return jsonify({"success": False, "error": "Conversation memory not initialized"}), 500
            
        data = request.get_json() or {}
        query = data.get('query', '')
        max_results = min(data.get('max_results', 5), 20)  # Cap at 20
        
        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400
            
        user_id = user.get('id', 'unknown') if user else 'unknown'
        
        # Search for similar conversations
        similar_conversations = conversation_memory.search_similar_conversations(query, user_id, max_results)
        
        return jsonify({
            "success": True,
            "similar_conversations": similar_conversations,
            "query": query,
            "total_found": len(similar_conversations),
            "message": "Conversation search completed successfully"
        })
        
    except Exception as e:
        print(f"SEARCH_CONVERSATION_ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

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
    
    # Robust JSON parse: tolerate stray control chars or accidental non-JSON bytes
    try:
        data = request.get_json()
    except Exception:
        raw = request.get_data(cache=False, as_text=True)
        # Remove ASCII control characters except whitespace/newlines/tabs
        filtered = ''.join(ch for ch in raw if ord(ch) >= 32 or ch in '\n\r\t')
        import json as _json
        try:
            data = _json.loads(filtered)
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body"}), 400

    messages = data.get('messages', [])
    requested_model = data.get('model')
    model = get_validated_chat_model(requested_model)  # Restrict to allowed models

    if not messages:
        # Return a minimal NDJSON-friendly error envelope so the frontend doesn't try to parse HTML.
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
                    pass
                # Check for stop even during initial processing
                if check_stop():
                    print(f"STOP_DETECTED: Request {request_id} stopped during message filtering")
                    yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                    return

            # Log incoming headers for project root troubleshooting
            try:
                prj_hdr = request.headers.get('X-Project-Root')
                print(f"CHAT_HEADERS: X-Project-Root={prj_hdr} X-User-ID={request.headers.get('X-User-ID')} X-Machine-ID={request.headers.get('X-Machine-ID')}")
                
                # Store project root in Flask g for access by internal functions
                if prj_hdr:
                    g.project_root = prj_hdr
            except Exception:
                pass

            # Helper to ensure tool call arguments are valid JSON strings.
            # Prevents downstream provider adapters (e.g., Gemini) from failing to parse
            # arguments when malformed content leaks into tool calls.
            def _sanitize_tool_arguments(arguments_value):
                try:
                    import json as _json
                    import re as _re
                    if isinstance(arguments_value, dict):
                        return _json.dumps(arguments_value, separators=(",", ":"))
                    s = str(arguments_value or "")
                    if not s:
                        return "{}"
                    try:
                        obj = _json.loads(s)
                        return _json.dumps(obj, separators=(",", ":"))
                    except Exception:
                        pass
                    start = s.find('{')
                    end = s.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        inner = s[start:end + 1]
                        try:
                            obj = _json.loads(inner)
                            return _json.dumps(obj, separators=(",", ":"))
                        except Exception:
                            s = inner
                    s2 = _re.sub(r"</?[^>]+>", "", s)
                    s2 = s2.replace("\n", " ").strip()
                    try:
                        obj = _json.loads(s2)
                        return _json.dumps(obj, separators=(",", ":"))
                    except Exception:
                        return "{}"
                except Exception:
                    return "{}"

            while True:  # Loop to handle tool calling and responses
                # Check for stop before each major operation
                if check_stop():
                    print(f"STOP_DETECTED: Request {request_id} stopped before OpenAI call")
                    yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                    return
                
                # Check and manage conversation length before making API call
                # Note: With frontend-managed conversations, we use a simpler fallback approach
                # Frontend can call /summarize_conversation endpoint for intelligent summarization
                if conversation_memory and conversation_memory.enabled:
                    user_id = user.get('id', 'unknown') if user else 'unknown'
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        conversation_messages = loop.run_until_complete(
                            conversation_memory.manage_conversation_length(conversation_messages, model, user_id)
                        )
                        loop.close()
                    except Exception as e:
                        print(f"CONVERSATION_MANAGE_ERROR: Failed intelligent management, using fallback: {e}")
                        conversation_messages = _manage_conversation_length_fallback(conversation_messages, model)
                else:
                    # Use simple fallback when conversation memory is not available
                    conversation_messages = _manage_conversation_length_fallback(conversation_messages, model)
                
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
                                # Sanitize function.arguments to valid JSON string if present
                                try:
                                    fn = fixed_tool_call.get('function') or {}
                                    if isinstance(fn, dict) and 'arguments' in fn:
                                        fn_args = fn.get('arguments')
                                        fn['arguments'] = _sanitize_tool_arguments(fn_args)
                                        fixed_tool_call['function'] = fn
                                except Exception:
                                    pass
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
                
                # Resilient model call with 5 retries (1 second each) then fallback to GPT-5
                attempts = 0
                max_attempts = 5  # 5 retry attempts as requested
                providers_tried = set()
                model_try = model
                
                # We need to retry the ENTIRE streaming process, not just the initial call
                while True:
                    try:
                        response = completion(
                            model=model_try,
                            messages=openai_messages,
                            tools=godot_tools,
                            tool_choice="auto",
                            stream=True
                        )
                        
                        # Process the stream inside the try block to catch streaming errors
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
                        
                        # Successfully processed the stream, break out of retry loop
                        break
                        
                    except Exception as e:
                        print(f"STREAM_ERROR: {e}")
                        err_name = e.__class__.__name__
                        overloaded = "Overloaded" in str(e)
                        transient = err_name in ("InternalServerError", "RateLimitError", "ServiceUnavailableError") or overloaded
                        
                        # Check for stop during retry loop
                        if check_stop():
                            print(f"STOP_DETECTED: Request {request_id} stopped during retry")
                            yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                            return
                        
                        # Special handling for rate limit errors
                        is_rate_limit = "RateLimitError" in err_name and ("limit exceeded" in str(e) or "too many tokens" in str(e) or "rate limit" in str(e).lower())
                        
                        if transient and attempts < max_attempts:
                            attempts += 1
                            
                            if is_rate_limit:
                                yield json.dumps({
                                    "status": "rate_limit_hit",
                                    "provider": get_model_friendly_name(model_try), 
                                    "attempt": attempts,
                                    "max_attempts": max_attempts,
                                    "error": str(e)[:100],
                                    "message": "Rate limit exceeded, retrying..."
                                }) + '\n'
                            else:
                                yield json.dumps({
                                    "status": "retrying_provider",
                                    "provider": get_model_friendly_name(model_try),
                                    "attempt": attempts,
                                    "max_attempts": max_attempts,
                                    "error": str(e)[:100]  # Show error snippet
                                }) + '\n'
                            
                            print(f"RETRY: Attempt {attempts}/{max_attempts} after error: {e}")
                            time.sleep(1.0)  # Fixed 1 second delay as requested
                            continue

                        # After 5 retries, fallback to GPT-5 if not already tried
                        providers_tried.add(model_try)
                        
                        # Notify about model switching if it's due to rate limits
                        if is_rate_limit and model_try != 'openai/gpt-5':
                            yield json.dumps({
                                "status": "provider_switched",
                                "from_provider": get_model_friendly_name(model_try),
                                "to_provider": "OpenAI GPT-5",
                                "reason": "Rate limit exceeded",
                                "message": f"Switching from {get_model_friendly_name(model_try)} to OpenAI GPT-5 due to rate limits"
                            }) + '\n'
                        
                        # Always try GPT-5 after retries exhausted
                        if MODEL_MAP.get("gpt-5") not in providers_tried:
                            fallback = MODEL_MAP.get("gpt-5")
                            yield json.dumps({
                                "status": "switching_model",
                                "from": get_model_friendly_name(model_try),
                                "to": "gpt-5",
                                "reason": f"Provider overloaded after {max_attempts} retries"
                            }) + '\n'
                            print(f"SWITCHING: From {model_try} to {fallback} after {max_attempts} failed attempts")
                            model_try = fallback
                            attempts = 0  # Reset attempts for new provider
                            continue

                        # No retries/fallbacks left – bubble up to main handler
                        raise

                # Now that we've processed all chunks, handle the results

                # --- Backend-Only Tool Execution (Image Generation + Search + Docs) ---
                backend_tools_detected = [
                    func.get("name")
                    for func in tool_call_aggregator.values()
                    if func.get("name") in [
                        "image_operation",
                        "search_across_project",
                        "search_across_godot_docs",
                        "slice_spritesheet",
                        "search_godot_assets",
                        "install_godot_asset",
                        # Note: Game testing tools are frontend-only, not backend
                    ]
                ]
                print(f"BACKEND_DETECTION: Found {len(backend_tools_detected)} backend tools: {backend_tools_detected}")
                
                if any(
                    func.get("name") in [
                        "image_operation",
                        "search_across_project",
                        "search_across_godot_docs",
                        "slice_spritesheet",
                        "search_godot_assets",
                        "install_godot_asset",
                        # Note: Game testing tools are frontend-only, not backend
                    ]
                    for func in tool_call_aggregator.values()
                ):
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
                            "function": {"name": func["name"], "arguments": _sanitize_tool_arguments(func["arguments"])},
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
                            # Execute the operation with conversation context (with cooperative cancellation)
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_image_op():
                                try:
                                    _tool_result_holder["result"] = image_operation_internal(arguments, conversation_messages)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_image_op, daemon=True)
                            t.start()
                            # Poll for stop while tool runs
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during image_operation")
                                    # We don't kill the thread; we just stop streaming and drop the result
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                time.sleep(0.1)
                            image_result = _tool_result_holder["result"] or {"success": False, "error": "image_operation returned no result"}
                            
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
                            
                            # Execute the search operation (pass current user context) with cancellation poll
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_search():
                                try:
                                    _tool_result_holder["result"] = search_across_project_internal(arguments, current_user=user)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_search, daemon=True)
                            t.start()
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_across_project")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                time.sleep(0.05)
                            search_result = _tool_result_holder["result"] or {"success": False, "error": "search_across_project returned no result"}
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
                        elif func["name"] == "search_across_godot_docs":
                            # Check for stop before tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "search_across_godot_docs", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            # Execute docs search with cancellation poll
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_docs():
                                try:
                                    _tool_result_holder["result"] = search_across_godot_docs_internal(arguments)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_docs, daemon=True)
                            t.start()
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_across_godot_docs")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                time.sleep(0.05)
                            docs_result = _tool_result_holder["result"] or {"success": False, "error": "search_across_godot_docs returned no result"}
                        elif func["name"] == "slice_spritesheet":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            yield json.dumps({"tool_starting": "slice_spritesheet", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_slice():
                                try:
                                    _tool_result_holder["result"] = slice_spritesheet_internal(arguments)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_slice, daemon=True)
                            t.start()
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during slice_spritesheet")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                time.sleep(0.05)
                            slice_result = _tool_result_holder["result"] or {"success": False, "error": "slice_spritesheet returned no result"}
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            yield json.dumps({"tool_executed": "slice_spritesheet", "tool_result": slice_result, "tool_call_id": tool_id, "status": "tool_completed"}) + '\n'
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "slice_spritesheet",
                                "content": json.dumps({
                                    "success": slice_result.get("success"),
                                    "grid_cols": slice_result.get("grid_cols"),
                                    "grid_rows": slice_result.get("grid_rows"),
                                    "tile_size": slice_result.get("tile_size"),
                                    "frames_count": len(slice_result.get("frames", []))
                                }),
                            })
                        
                        elif func["name"] == "search_godot_assets":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "search_godot_assets", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_asset_search():
                                try:
                                    _tool_result_holder["result"] = search_godot_assets_internal(arguments)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_asset_search, daemon=True)
                            t.start()
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_godot_assets")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                time.sleep(0.05)
                            
                            asset_search_result = _tool_result_holder["result"] or {"success": False, "error": "search_godot_assets returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Yield result to frontend immediately
                            yield json.dumps({
                                "tool_executed": "search_godot_assets",
                                "tool_result": asset_search_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # Prepare result for conversation history (limit assets to avoid token bloat)
                            assets_summary = {
                                "success": asset_search_result.get("success"),
                                "query": asset_search_result.get("query"),
                                "total_found": asset_search_result.get("total_found", 0),
                                "assets": asset_search_result.get("assets", [])[:5]  # Limit to 5 for history
                            }
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "search_godot_assets",
                                "content": json.dumps(assets_summary)
                            })
                        
                        elif func["name"] == "install_godot_asset":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "install_godot_asset", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            
                            # Ensure project_path is provided (get from Flask context before threading)
                            # Handle both missing project_path and res:// paths
                            project_path_arg = arguments.get('project_path', '')
                            if not project_path_arg or project_path_arg == 'res://':
                                if hasattr(g, 'project_root') and g.project_root:
                                    arguments['project_path'] = g.project_root
                                    print(f"ASSET_INSTALL_PREP: Injected project_root from Flask context: {g.project_root} (was: {project_path_arg})")
                                else:
                                    print(f"ASSET_INSTALL_ERROR: Cannot resolve project path. Flask g.project_root not available")
                            elif project_path_arg.startswith('res://'):
                                # Handle res://subdirectory paths  
                                if hasattr(g, 'project_root') and g.project_root:
                                    relative_path = project_path_arg[6:]  # Remove 'res://'
                                    resolved_path = os.path.join(g.project_root, relative_path) if relative_path else g.project_root
                                    arguments['project_path'] = resolved_path
                                    print(f"ASSET_INSTALL_PREP: Converted res:// path '{project_path_arg}' to '{resolved_path}'")
                            
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_asset_install():
                                try:
                                    _tool_result_holder["result"] = install_godot_asset_internal(arguments)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_asset_install, daemon=True)
                            t.start()
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during install_godot_asset")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                time.sleep(0.1)  # Slightly longer delay for install operations
                            
                            install_result = _tool_result_holder["result"] or {"success": False, "error": "install_godot_asset returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Yield result to frontend immediately
                            yield json.dumps({
                                "tool_executed": "install_godot_asset",
                                "tool_result": install_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # Prepare result for conversation history
                            install_summary = {
                                "success": install_result.get("success"),
                                "message": install_result.get("message"),
                                "asset_name": install_result.get("installation_info", {}).get("asset_name") if install_result.get("installation_info") else None,
                                "installed_to": install_result.get("installation_info", {}).get("installed_to") if install_result.get("installation_info") else None,
                                "is_plugin": install_result.get("installation_info", {}).get("is_plugin") if install_result.get("installation_info") else False
                            }
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool", 
                                "name": "install_godot_asset",
                                "content": json.dumps(install_summary)
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
                            "function": {"name": func["name"], "arguments": _sanitize_tool_arguments(func["arguments"])},
                        })
                        tool_calls_for_frontend.append({
                            "id": tool_id,
                            "function": {
                                "name": func["name"],
                                "arguments": _sanitize_tool_arguments(func["arguments"]) 
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
                    # Signal that the stream is ending but the overall task is waiting on the frontend.
                    yield json.dumps({"status": "awaiting_frontend_action"}) + '\n'
                    print(f"FRONTEND_PROCESSING: Tool calls sent, stream closing. Awaiting frontend tool execution in next request.")
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
        # Add retry logic for script generation
        attempts = 0
        max_attempts = 5
        model_for_script = data.get('model', DEFAULT_MODEL)

        while True:
            try:
                response = completion(
                    model=get_validated_chat_model(model_for_script),
                    messages=[{"role": "user", "content": script_prompt}]
                )
                break
            except Exception as e:
                err_name = e.__class__.__name__
                overloaded = "Overloaded" in str(e)
                transient = err_name in ("InternalServerError", "RateLimitError", "ServiceUnavailableError") or overloaded

                if transient and attempts < max_attempts:
                    attempts += 1
                    print(f"GENERATE_SCRIPT: Retry {attempts}/{max_attempts} after error: {str(e)[:100]}")
                    time.sleep(1.0)
                    continue

                # After 5 retries, try GPT-5
                if attempts >= max_attempts and model_for_script != 'gpt-5':
                    print(f"GENERATE_SCRIPT: Switching to GPT-5 after {max_attempts} failed attempts")
                    model_for_script = 'gpt-5'
                    attempts = 0
                    continue

                raise

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

def _analyze_gdscript_indentation(file_content: str, edit_prompt: str) -> str:
    """
    Analyze GDScript indentation patterns and provide context to help AI preserve structure.
    This is a dynamic analysis that understands the actual code structure.
    """
    lines = file_content.split('\n')
    indentation_rules = []
    
    # Detect base indentation (tabs vs spaces and size)
    indent_char = None
    indent_size = 4  # Default for GDScript
    
    for line in lines:
        if line.strip() and line.startswith((' ', '\t')):
            if line.startswith('\t'):
                indent_char = '\t'
                indent_size = 1
                break
            elif line.startswith(' '):
                indent_char = ' '
                # Count leading spaces
                spaces = 0
                for char in line:
                    if char == ' ':
                        spaces += 1
                    else:
                        break
                if spaces > 0:
                    indent_size = spaces
                    break
    
    if not indent_char:
        indent_char = '\t'  # Default for GDScript
    
    # Analyze code structure patterns
    current_indent_level = 0
    function_contexts = []
    class_contexts = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        # Calculate current indentation level
        leading_whitespace = len(line) - len(line.lstrip())
        if indent_char == '\t':
            line_indent_level = leading_whitespace
        else:
            line_indent_level = leading_whitespace // indent_size
        
        # Track function definitions
        if stripped.startswith('func '):
            func_name = stripped.split('(')[0].replace('func ', '')
            function_contexts.append({
                'name': func_name,
                'line': i + 1,
                'indent_level': line_indent_level,
                'body_indent': line_indent_level + 1
            })
        
        # Track class definitions  
        elif stripped.startswith('class '):
            class_name = stripped.split(':')[0].replace('class ', '')
            class_contexts.append({
                'name': class_name,
                'line': i + 1,
                'indent_level': line_indent_level,
                'body_indent': line_indent_level + 1
            })
        
        # Track control structures
        elif any(stripped.startswith(keyword + ' ') or stripped.startswith(keyword + ':') 
                for keyword in ['if', 'elif', 'else', 'for', 'while', 'match']):
            # These increase indentation for their body
            pass
    
    # Build indentation guidance
    indent_type = 'tabs' if indent_char == '\t' else f'{indent_size} spaces'
    rules = [
        f"- Use {indent_type} for indentation",
        "- GDScript uses indentation to define code blocks (like Python)",
        "- Function bodies must be indented one level deeper than the function definition",
        "- Class bodies must be indented one level deeper than the class definition", 
        "- Control structures (if/for/while) increase indentation by one level for their body",
        "- Statements at the same logical level should have the same indentation"
    ]
    
    # Add context about existing functions if relevant
    if function_contexts and any(keyword in edit_prompt.lower() 
                               for keyword in ['function', 'func', 'method', 'add', 'insert']):
        rules.append("- Current functions in this file:")
        for func in function_contexts[-3:]:  # Show last 3 functions for context
            rules.append(f"  • {func['name']}() at line {func['line']}, body indented to level {func['body_indent']}")
    
    # Analyze the specific area being edited if possible
    if any(keyword in edit_prompt.lower() for keyword in ['line', 'after', 'before', 'around']):
        # Try to extract line numbers or context from the prompt
        import re
        line_numbers = re.findall(r'line\s*(\d+)', edit_prompt.lower())
        if line_numbers:
            target_line = int(line_numbers[0]) - 1  # Convert to 0-based
            if 0 <= target_line < len(lines):
                target_line_content = lines[target_line].strip()
                original_line = lines[target_line]
                target_indent = len(original_line) - len(original_line.lstrip())
                
                if indent_char == '\t':
                    target_level = target_indent
                    indent_example = '\t' * target_indent
                else:
                    target_level = target_indent // indent_size
                    indent_example = ' ' * target_indent
                
                rules.append(f"- Target area context: Line {target_line + 1} ('{target_line_content}') uses indent level {target_level}")
                indent_type = 'tabs' if indent_char == '\t' else 'spaces'
                rules.append(f"- EXACT indentation for this area: '{indent_example}' ({target_indent} {indent_type})")
                rules.append(f"- When modifying this line, preserve the EXACT leading whitespace: '{original_line[:target_indent]}'")
                rules.append(f"- MANDATORY: Every line you output must start with exactly {target_indent} {indent_type}")
                rules.append(f"- COPY this exact indentation: '{repr(original_line[:target_indent])}'")
                rules.append(f"- New code in this area should match indent level {target_level} or follow the logical structure")
                
                # Add context about surrounding lines for better understanding
                if target_line > 0:
                    prev_line = lines[target_line - 1]
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    prev_indent_type = 'tabs' if indent_char == '\t' else 'spaces'
                    rules.append(f"- Previous line indent: {prev_indent} {prev_indent_type}")
                
                if target_line < len(lines) - 1:
                    next_line = lines[target_line + 1]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    next_indent_type = 'tabs' if indent_char == '\t' else 'spaces'
                    rules.append(f"- Next line indent: {next_indent} {next_indent_type}")
    
    return '\n'.join(rules)

@app.route('/predict_code_edit', methods=['POST'])
def predict_code_edit():
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    """
    Optimized AI-powered apply edit endpoint.
    - Simplified to ask for edited code directly (no complex JSON schemas)
    - Faster response parsing
    - Better error handling
    Supports both full-file edits and range edits.
    """
    data = request.json
    file_content = data.get('file_content', '')
    prompt = data.get('prompt')
    # Optional range-edit context from frontend
    lines_mode = (data.get('lines') or data.get('mode') or 'all').lower()
    start_line = int(data.get('start_line') or 0)
    end_line = int(data.get('end_line') or 0)
    pre_text = data.get('pre_text') or ''
    post_text = data.get('post_text') or ''
    path = data.get('path') or ''
    
    print(f"APPLY_EDIT_REQUEST: '{prompt}' for {path} (content_len={len(file_content)})")

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        import json as _json
        # OPTIMIZATION: Simpler, direct prompts without JSON schemas
        is_range = (lines_mode == 'range') or (start_line > 0 and end_line >= start_line)
        
        # Analyze indentation for GDScript files FIRST (before building prompts)
        indentation_context = ""
        if path and path.endswith('.gd'):
            indentation_context = _analyze_gdscript_indentation(file_content, prompt)
            print(f"GDSCRIPT INDENTATION ANALYSIS for {path}:")
            print(indentation_context)
        
        # Build a simple, clear prompt
        if is_range:
            # For range edits, provide context about the specific lines
            indentation_reminder = f"\n\nCRITICAL INDENTATION RULES:\n{indentation_context}" if indentation_context else ""
            full_prompt = (
                f"Task: {prompt}\n\n"
                f"Edit the following code segment (lines {start_line}-{end_line}):\n"
                f"{file_content}\n\n"
                f"CRITICAL: You must preserve EXACT indentation. Look at the existing lines and match their indentation precisely. Count the tabs/spaces and use exactly the same amount.{indentation_reminder}\n\n"
                f"Reply with ONLY the edited code for this segment."
            )
        else:
            # For full file edits, provide the complete file
            # Add line numbers for context if file is large
            if len(file_content.split('\n')) > 50:
                lines = file_content.split('\n')
                numbered_content = '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))
                full_prompt = (
                    f"Task: {prompt}\n\n"
                    f"Current file content (with line numbers for reference):\n"
                    f"{numbered_content}\n\n"
                    "Reply with ONLY the complete edited file content. No explanations or markdown."
                )
            else:
                indentation_reminder = f"\n\nCRITICAL: {indentation_context}" if indentation_context else ""
                full_prompt = (
                    f"Task: {prompt}\n\n"
                    f"Current file content:\n"
                    f"{file_content}\n\n"
                    f"IMPORTANT: Reply with the COMPLETE edited file content. You must include ALL original lines plus your changes.{indentation_reminder}\n\n"
                    "Output format: Just the complete file content, no explanations, no markdown, no truncation."
                )

        # Indentation context already analyzed above
        
        # OPTIMIZATION: Add temperature and timeout settings
        # Use claude-4 by default for apply_edit as it's often faster
        model_for_edit = data.get('model', 'claude-4')
        
        # Add retry logic for apply_edit as well
        attempts = 0
        max_attempts = 5
        while True:
            try:
                # Enhanced system prompt for GDScript indentation awareness
                system_prompt = "You are a code editor. Output only edited code, no explanations."
                if indentation_context:
                    system_prompt += f"\n\nCRITICAL INDENTATION REQUIREMENTS:\n{indentation_context}\n\nYou MUST preserve exact indentation. Copy the whitespace characters exactly as shown. This is non-negotiable."
                
                response = completion(
                    model=get_validated_chat_model(model_for_edit),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2,  # Even lower temperature for precise indentation
                    max_tokens=16000  # Higher limit to ensure complete file generation
                )
                break
            except Exception as e:
                err_name = e.__class__.__name__
                overloaded = "Overloaded" in str(e)
                transient = err_name in ("InternalServerError", "RateLimitError", "ServiceUnavailableError") or overloaded
                
                if transient and attempts < max_attempts:
                    attempts += 1
                    print(f"APPLY_EDIT: Retry {attempts}/{max_attempts} after error: {str(e)[:100]}")
                    time.sleep(1.0)  # 1 second delay
                    continue
                
                # After 5 retries, try GPT-5
                if attempts >= max_attempts and model_for_edit != 'gpt-5':
                    print(f"APPLY_EDIT: Switching to GPT-5 after {max_attempts} failed attempts")
                    model_for_edit = 'gpt-5'
                    attempts = 0  # Reset for new provider
                    continue
                
                # If all retries exhausted, return error
                raise

        raw = response.choices[0].message.content
        print(f"APPLY_EDIT: Response length: {len(raw)}")
        print(f"APPLY_EDIT: Raw response preview: {raw[:200]}")
        
        # Check if response is suspiciously short for a full file edit
        if not is_range and len(raw) < len(file_content) * 0.5:
            print(f"WARNING: AI response ({len(raw)} chars) is much shorter than original file ({len(file_content)} chars)")
            print(f"This suggests the AI didn't complete the task properly")

        # OPTIMIZATION: Simple response cleaning instead of complex JSON parsing
        edited_content = raw.strip()
        
        # Remove markdown code fences if present
        if edited_content.startswith('```'):
            lines = edited_content.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            edited_content = '\n'.join(lines)
        
        # Remove any language-specific markdown markers
        for marker in ['```python', '```py', '```gdscript', '```gd', '```csharp', '```cs', '```cpp', '```c++', '```json', '```javascript', '```js', '```typescript', '```ts', '```java', '```go', '```rust', '```ruby', '```php', '```swift', '```kotlin', '```scala', '```r', '```julia', '```matlab', '```perl', '```lua', '```dart', '```haskell', '```clojure', '```elixir', '```erlang', '```ocaml', '```fsharp', '```nim', '```crystal', '```zig', '```vlang', '```']:
            edited_content = edited_content.replace(marker, '')
        
        edited_content = edited_content.strip()
        
        # Build the full edited content based on edit mode
        import difflib
        
        if is_range:
            # For range edits, splice the result back into the full file
            original_full = (pre_text or '') + ('\n' if pre_text and file_content else '') + (file_content or '') + ('\n' if file_content and post_text else '') + (post_text or '')
            
            full_edited_content = (pre_text or '')
            if full_edited_content and edited_content and not full_edited_content.endswith('\n'):
                full_edited_content += '\n'
            full_edited_content += edited_content
            if post_text:
                if full_edited_content and not full_edited_content.endswith('\n'):
                    full_edited_content += '\n'
                full_edited_content += post_text
        else:
            # For full file edits, the response is the complete new file
            original_full = file_content or ''
            full_edited_content = edited_content

        # Generate both unified diff and inline diff for user review
        diff_lines = list(difflib.unified_diff(
            (original_full or '').splitlines(),
            (full_edited_content or '').splitlines(),
            fromfile=f"{path} (original)" if path else 'original',
            tofile=f"{path} (modified)" if path else 'modified',
            lineterm=''
        ))
        diff_text = "\n".join(diff_lines)
        
        # Generate inline diff using SequenceMatcher for better quality
        import difflib
        original_lines = (original_full or '').splitlines()
        edited_lines = (full_edited_content or '').splitlines()
        
        # Debug: Check for whitespace issues
        print(f"DIFF DEBUG: Comparing {len(original_lines)} vs {len(edited_lines)} lines")
        if len(original_lines) > 20 and len(edited_lines) > 20:
            # Sample a few lines to check for whitespace differences
            for i in [20, 21, 22, 23, 24]:
                if i < len(original_lines) and i < len(edited_lines):
                    if original_lines[i] != edited_lines[i]:
                        print(f"DIFF DEBUG: Line {i+1} differs:")
                        print(f"  Original: {repr(original_lines[i])}")
                        print(f"  Edited:   {repr(edited_lines[i])}")
        
        inline_diff_lines = []
        matcher = difflib.SequenceMatcher(None, original_lines, edited_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Unchanged lines
                for i in range(i1, i2):
                    inline_diff_lines.append({"type": "equal", "content": original_lines[i]})
            elif tag == 'delete':
                # Lines removed
                for i in range(i1, i2):
                    inline_diff_lines.append({"type": "delete", "content": original_lines[i]})
            elif tag == 'insert':
                # Lines added
                for j in range(j1, j2):
                    inline_diff_lines.append({"type": "insert", "content": edited_lines[j]})
            elif tag == 'replace':
                # Lines changed - show as delete + insert
                for i in range(i1, i2):
                    inline_diff_lines.append({"type": "delete", "content": original_lines[i]})
                for j in range(j1, j2):
                    inline_diff_lines.append({"type": "insert", "content": edited_lines[j]})
        
        # Convert inline diff to text format for frontend
        inline_diff_text = ""
        for line in inline_diff_lines:
            if line["type"] == "equal":
                inline_diff_text += "  " + line["content"] + "\n"
            elif line["type"] == "delete":
                inline_diff_text += "- " + line["content"] + "\n"
            elif line["type"] == "insert":
                inline_diff_text += "+ " + line["content"] + "\n"
        
        # DEBUG: Log the diff generation
        print(f"PREDICT_CODE_EDIT DIFF: Generated inline_diff_text length: {len(inline_diff_text)}")
        print(f"PREDICT_CODE_EDIT DIFF: Original lines: {len(original_lines)}, Edited lines: {len(edited_lines)}")
        print(f"PREDICT_CODE_EDIT DIFF: Diff operations count: {len(inline_diff_lines)}")
        if inline_diff_text:
            print(f"PREDICT_CODE_EDIT DIFF: Preview: {inline_diff_text[:300]}")
        else:
            print("PREDICT_CODE_EDIT DIFF: WARNING - inline_diff_text is EMPTY!")

        return jsonify({
            "success": True,
            "status": "pending_user_action",
            "pending_user_action": True,
            "applied": False,
            "mode": "range" if is_range else "full",
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "structured_edits": {},  # No longer using structured edits
            "full_edited_content": full_edited_content,
            "edited_content": full_edited_content,  # For compatibility
            "diff": diff_text,
            "inline_diff": inline_diff_text,  # New inline diff format
            "inline_diff_data": inline_diff_lines,  # Structured diff data
            "original_content": original_full  # Include for frontend diff display
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
        # Robust JSON parse: fall back to raw body parsing to tolerate stray control chars
        try:
            data = request.get_json()
        except Exception:
            raw = request.get_data(cache=False, as_text=True)
            # Remove ASCII control characters except whitespace/newlines/tabs
            filtered = ''.join(ch for ch in raw if ord(ch) >= 32 or ch in '\n\r\t')
            data = json.loads(filtered)
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

@app.route('/index_status', methods=['POST'])
def check_index_status():
    """Check if project is already indexed and up-to-date"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    try:
        # Verify authentication
        user, error_response, status_code = verify_authentication()
        if error_response:
            return jsonify(error_response), status_code
            
        data = request.json or {}
        project_root = data.get('project_root')
        
        # Fallback to header if not provided in body
        if not project_root:
            project_root = request.headers.get('X-Project-Root')
        
        if not project_root:
            return jsonify({"error": "project_root required"}), 400
        
        project_id = hashlib.md5(project_root.encode()).hexdigest()
        
        if not cloud_vector_manager:
            return jsonify({"indexed": False, "error": "Vector search unavailable"}), 501
        
        # Check if project has any indexed files
        try:
            stats = cloud_vector_manager.get_project_stats(user['id'], project_id)
            indexed_files = stats.get('total_files', 0)
            
            return jsonify({
                "success": True,
                "indexed": indexed_files > 0,
                "stats": stats,
                "project_id": project_id
            })
        except AttributeError:
            # Fallback for managers that don't have get_project_stats
            return jsonify({
                "success": True, 
                "indexed": False,  # Conservative: assume not indexed if we can't check
                "message": "Index status check not supported by current vector manager",
                "project_id": project_id
            })
        
    except Exception as e:
        print(f"INDEX_STATUS ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/reindex_project', methods=['POST'])
def reindex_project():
    """Re-index entire project (clear + fresh index)"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
        
    try:
        # Verify authentication
        user, error_response, status_code = verify_authentication()
        if error_response:
            return jsonify(error_response), status_code
            
        data = request.json or {}
        project_root = data.get('project_root')
        
        # Fallback to header if not provided in body
        if not project_root:
            project_root = request.headers.get('X-Project-Root')
        
        if not project_root:
            return jsonify({"error": "project_root required (pass in body or X-Project-Root header)"}), 400
        
        project_id = hashlib.md5(project_root.encode()).hexdigest()
        
        if not cloud_vector_manager:
            return jsonify({"success": False, "error": "Vector search unavailable"}), 501
        
        # Step 1: Clear existing data
        print(f"REINDEX: Clearing project data for {project_root}")
        cloud_vector_manager.clear_project(user['id'], project_id)
        
        # Step 2: Trigger fresh indexing by calling the embed endpoint internally
        # This will cause the frontend to scan and send files
        return jsonify({
            "success": True, 
            "action": "reindex_project",
            "message": f"Project cleared. Please trigger indexing from Godot to complete re-indexing.",
            "project_id": project_id
        })
        
    except Exception as e:
        print(f"REINDEX ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/clear_project_debug', methods=['POST'])
def clear_project_debug():
    """Debug endpoint to clear project data - bypasses auth for testing"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
        
    # FOR DEBUG: Create a fake user to bypass auth
    user = {"id": "debug_user"}
        
    data = request.json or {}
    project_root = data.get('project_root') or os.getcwd()
    project_id = hashlib.md5(project_root.encode()).hexdigest()
    
    if cloud_vector_manager:
        cloud_vector_manager.clear_project(user['id'], project_id)
        return jsonify({"success": True, "message": f"Cleared project data for {project_id}"})
    else:
        return jsonify({"success": False, "message": "No vector manager available"})

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
        # Parse JSON body first, then auth (we may need machine_id)
        try:
            data = request.get_json()
        except Exception:
            raw = request.get_data(cache=False, as_text=True)
            filtered = ''.join(ch for ch in raw if ord(ch) >= 32 or ch in '\n\r\t')
            data = json.loads(filtered)

        # Verify authentication (allow guest fallback for indexing/search)
        user, error_response, status_code = verify_authentication()
        if error_response:
            # Attempt guest fallback using machine id if provided
            machine_id = request.headers.get('X-Machine-ID') or (data.get('machine_id') if isinstance(data, dict) else None)
            guest_name = request.headers.get('X-Guest-Name')
            if machine_id:
                guest_result = auth_manager.create_or_get_guest_session(machine_id, guest_name)
                if guest_result.get('success'):
                    user = guest_result['user']
                else:
                    return jsonify(error_response), status_code
            else:
                return jsonify(error_response), status_code
        
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
        
        if cloud_vector_manager is None:
            return jsonify({
                "success": False,
                "error": "Vector indexing unavailable (configure Weaviate or ensure local index + OPENAI_API_KEY)",
                "action": action
            }), 501

        if action == 'index_project':
            # In cloud deployment, frontend should send files via index_files action
            if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('GAE_ENV', '').startswith('standard'):
                return jsonify({
                    "success": False,
                    "action": "index_project",
                    "error": "index_project not supported in cloud deployment. Frontend should use index_files action with file content.",
                    "project_id": project_id
                }), 400
            
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
            # In cloud deployment, frontend should send files via index_files action
            if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('GAE_ENV', '').startswith('standard'):
                return jsonify({
                    "success": False,
                    "action": "index_file",
                    "error": "index_file not supported in cloud deployment. Frontend should use index_files action with file content.",
                    "project_id": project_id
                }), 400
            
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
            # Default to lighter graph for speed; clients can request deeper
            graph_depth = int(data.get('graph_depth', 1))
            graph_edge_kinds = data.get('graph_edge_kinds') or []
            results = cloud_vector_manager.search(query, user['id'], project_id, max_results)
            # Filter out Godot sidecar UID files
            results = [r for r in results if not str(r.get('file_path','')).endswith('.uid')]
            
            return jsonify({
                "success": True,
                "action": "search",
                "query": query,
                "results": results,
                "graph": (
                    cloud_vector_manager.get_graph_context_expanded(
                        [r.get('file_path') for r in results], user['id'], project_id,
                        depth=graph_depth, kinds=graph_edge_kinds
                    ) if include_graph else {}
                )
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
        if cloud_vector_manager is None:
            return jsonify({
                "success": False,
                "error": "Vector search unavailable (configure Weaviate or ensure local index + OPENAI_API_KEY)"
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
                    # If backend provides file_line_count, pass it through; else leave None
                    "line_count": r.get('file_line_count')
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
        # Refresh Cerebras models in case they changed
        global MODEL_MAP
        fresh_cerebras = fetch_cerebras_models()
        MODEL_MAP = BASE_MODEL_MAP.copy()
        MODEL_MAP.update(fresh_cerebras)
        
        # Update allowed models
        global ALLOWED_CHAT_MODELS
        ALLOWED_CHAT_MODELS = set(MODEL_MAP.keys())
        
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

# --- Docs search (shared corpus)
DOCS_USER_ID = os.getenv('DOCS_USER_ID', 'public_docs')
DOCS_PROJECT_ID = os.getenv('DOCS_PROJECT_ID', 'godot_docs_latest')
DOCS_DATASET = os.getenv('EMBED_DATASET', 'godot_embeddings')
DOCS_TABLE = os.getenv('EMBED_TABLE', 'embeddings')

def _search_godot_docs_bq(query: str, max_results: int = 5) -> list[dict]:
    if cloud_vector_manager is None:
        return []
    try:
        return cloud_vector_manager.search(query, DOCS_USER_ID, DOCS_PROJECT_ID, max_results)
    except Exception as e:
        print(f"DOCS_SEARCH_FALLBACK: {e}")
        return []

def search_across_godot_docs_internal(arguments: dict) -> dict:
    try:
        query = arguments.get('query', '')
        if not query:
            return {"success": False, "error": "Query parameter is required"}
        max_results = int(arguments.get('max_results', 5))
        results = _search_godot_docs_bq(query, max_results)
        formatted = []
        for r in results:
            fp = r.get('file_path', '')
            # Prefer plain content; fall back to content_preview from search, then nested chunk content
            raw_content = (
                r.get('content')
                or r.get('content_preview')
                or (r.get('chunk', {}) or {}).get('content')
                or ''
            )
            formatted.append({
                'title': fp,
                'snippet': raw_content[:400],
                'full_content': raw_content,
                'similarity': r.get('similarity', 0.0),
                'source': 'godot_docs',
                'file_path': fp,
            })
        return {
            'success': True,
            'query': query,
            'results': formatted,
            'file_count': len(formatted)
        }
    except Exception as e:
        print(f"DOCS_SEARCH_ERROR: {e}")
        return {"success": False, "error": f"Docs search failed: {str(e)}"}

# --- Asset Library Functions ---

def search_godot_assets_internal(arguments: dict) -> dict:
    """Search the Godot Asset Library for plugins, templates, and other assets"""
    try:
        query = arguments.get('query', '')
        if not query:
            return {"success": False, "error": "Query parameter is required"}
        
        category = arguments.get('category')
        max_results = arguments.get('max_results', 10)
        support_level = arguments.get('support_level', 'all')  # official, featured, community, testing, all
        godot_version = arguments.get('godot_version', '4.3')  # Default to current stable version
        sort_by = arguments.get('sort_by', 'rating')
        sort_reverse = arguments.get('sort_reverse', False)
        asset_type = arguments.get('asset_type', 'any')
        cost_filter = arguments.get('cost_filter', 'all')
        
        print(f"ASSET_SEARCH: Searching for '{query}' in Godot Asset Library (version: {godot_version}, sort: {sort_by}, type: {asset_type}, support: {support_level})")
        
        # Godot Asset Library API endpoint
        base_url = "https://godotengine.org/asset-library/api/asset"
        params = {
            'filter': query,
            'max_results': min(max_results, 100),  # Cap at 100 for better search flexibility
            'godot_version': godot_version,  # Filter by Godot version to get relevant results
            'sort': sort_by,
            'reverse': str(sort_reverse).lower()  # Convert boolean to lowercase string
        }
        
        # Category mapping (Godot Asset Library category IDs)
        category_map = {
            '2d_tools': '1',
            '3d_tools': '2', 
            'shaders': '3',
            'materials': '4',
            'tools': '5',
            'scripts': '6',
            'misc': '7',
            'templates': '8',
            'demos': '9',
            'plugins': '10'
        }
        
        if category and category.lower() in category_map:
            params['category'] = category_map[category.lower()]
        
        # Asset type filtering
        if asset_type != 'any':
            params['type'] = asset_type
        
        # Support level filtering (map to API values)
        support_level_map = {
            'official': 'official',
            'featured': 'featured', 
            'community': 'community',
            'testing': 'testing'
        }
        if support_level != 'all' and support_level in support_level_map:
            params['support'] = support_level_map[support_level]
        
        # Cost filtering
        if cost_filter == 'free':
            params['cost'] = 'MIT'  # Free assets typically use MIT license
        elif cost_filter == 'paid':
            params['cost'] = 'Non-free'  # Paid/commercial assets
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        results = response.json()
        
        # Format results for better readability
        formatted_assets = []
        for asset in results.get('result', []):
            formatted_asset = {
                'id': str(asset.get('asset_id', '')),
                'title': asset.get('title', 'Unknown'),
                'description': asset.get('description', ''),
                'category': asset.get('category', 'Unknown'),
                'author': asset.get('author', 'Unknown'),
                'version': asset.get('version', '1.0'),
                'godot_version': asset.get('godot_version', 'Unknown'),
                'rating': asset.get('rating', 0),
                'cost': asset.get('cost', 'Free'),
                'download_url': asset.get('download_url', ''),
                'browse_url': asset.get('browse_url', ''),
                'icon_url': asset.get('icon_url', '')
            }
            formatted_assets.append(formatted_asset)
        
        print(f"ASSET_SEARCH: Found {len(formatted_assets)} assets")
        
        return {
            "success": True,
            "query": query,
            "assets": formatted_assets,
            "total_found": len(formatted_assets),
            "search_params": {
                "category": category,
                "godot_version": godot_version,
                "sort_by": sort_by,
                "sort_reverse": sort_reverse,
                "asset_type": asset_type,
                "support_level": support_level,
                "cost_filter": cost_filter,
                "max_results": max_results
            }
        }
        
    except Exception as e:
        print(f"ASSET_SEARCH_ERROR: {e}")
        return {"success": False, "error": f"Asset search failed: {str(e)}"}

def install_godot_asset_internal(arguments: dict) -> dict:
    """Download and install an asset from the Godot Asset Library"""
    try:
        asset_id = arguments.get('asset_id')
        project_path = arguments.get('project_path', '')
        install_location = arguments.get('install_location', 'addons/')
        create_backup = arguments.get('create_backup', True)
        
        if not asset_id:
            return {"success": False, "error": "asset_id is required"}
        
        if not project_path:
            return {"success": False, "error": "project_path is required"}
        
        # Validate that we have a real filesystem path
        if project_path.startswith('res://'):
            return {"success": False, "error": f"Invalid project_path '{project_path}' - res:// paths should have been resolved to real filesystem paths before calling this function"}
        
        if project_path == 'res://':
            return {"success": False, "error": "project_path cannot be 'res://' - a real filesystem path is required"}
        
        # Check if we're running in cloud mode (project path won't exist on cloud server)
        is_cloud_mode = not os.path.exists(project_path)
        
        if is_cloud_mode:
            print(f"ASSET_INSTALL: Cloud mode detected - project path {project_path} not accessible from server")
        else:
            print(f"ASSET_INSTALL: Local mode detected - project path {project_path} exists")
        
        print(f"ASSET_INSTALL: Installing asset {asset_id} to {project_path}")
        
        # Get asset details from API
        asset_url = f"https://godotengine.org/asset-library/api/asset/{asset_id}"
        asset_response = requests.get(asset_url, timeout=30)
        asset_response.raise_for_status()
        asset_data = asset_response.json()
        
        if not asset_data:
            return {"success": False, "error": f"Asset {asset_id} not found"}
        
        asset_name = asset_data.get('title', f'asset_{asset_id}')
        download_url = asset_data.get('download_url')
        
        if not download_url:
            return {"success": False, "error": f"No download URL found for asset {asset_name}"}
        
        print(f"ASSET_INSTALL: Downloading {asset_name} from {download_url}")
        
        # Download the asset ZIP (allow redirects for GitHub repo renames)
        zip_response = requests.get(download_url, timeout=120, allow_redirects=True)  # Longer timeout for downloads
        zip_response.raise_for_status()
        
        if len(zip_response.content) == 0:
            return {"success": False, "error": f"Downloaded file for {asset_name} is empty"}
        
        # Cloud Mode: Return asset data for client-side installation
        if is_cloud_mode:
            import base64
            asset_b64 = base64.b64encode(zip_response.content).decode('utf-8')
            
            installation_info = {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "version": asset_data.get('version', '1.0'),
                "author": asset_data.get('author', 'Unknown'),
                "intended_path": os.path.join(project_path, install_location.strip('/')),
                "install_location": install_location,
                "is_plugin": False,  # Will be determined client-side
                "godot_version": asset_data.get('godot_version', 'Unknown'),
                "description": asset_data.get('description', '')[:200] + '...' if len(asset_data.get('description', '')) > 200 else asset_data.get('description', ''),
                "cloud_mode": True
            }
            
            print(f"ASSET_INSTALL: Cloud mode - returning asset data for client-side installation")
            
            return {
                "success": True,
                "message": f"Downloaded {asset_name} - ready for client installation",
                "installation_info": installation_info,
                "asset_data": asset_b64,
                "cloud_mode": True
            }
        
        # Local Mode: Direct installation (existing logic)
        # Prepare installation directory
        install_path = os.path.join(project_path, install_location.strip('/'))
        print(f"ASSET_INSTALL: Creating installation directory: {install_path}")
        
        try:
            os.makedirs(install_path, exist_ok=True)
            print(f"ASSET_INSTALL: Directory created/verified: {install_path}")
        except Exception as dir_error:
            print(f"ASSET_INSTALL: Failed to create directory {install_path}: {dir_error}")
            return {"success": False, "error": f"Failed to create installation directory: {str(dir_error)}"}
        
        # Create backup if requested and directory exists
        backup_path = None
        if create_backup and os.path.exists(install_path) and os.listdir(install_path):
            backup_dir = os.path.join(project_path, '.asset_backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{asset_name}_{int(time.time())}")
            import shutil
            shutil.copytree(install_path, backup_path)
            print(f"ASSET_INSTALL: Created backup at {backup_path}")
        
        # Extract ZIP file
        import zipfile
        
        extracted_files = []
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
            # List all files that will be extracted
            file_list = zip_file.namelist()
            print(f"ASSET_INSTALL: Extracting {len(file_list)} files")
            
            for file_info in zip_file.infolist():
                # Skip directories and hidden files
                if file_info.is_dir() or file_info.filename.startswith('.'):
                    continue
                    
                # Extract file
                extracted_path = zip_file.extract(file_info, install_path)
                extracted_files.append(extracted_path)
            
        # Verify installation
        if not extracted_files:
            return {"success": False, "error": f"No files were extracted from {asset_name}"}
        
        # Check for plugin.cfg if this looks like a plugin
        plugin_cfg_path = None
        for file_path in extracted_files:
            if file_path.endswith('plugin.cfg'):
                plugin_cfg_path = file_path
                break
        
        installation_info = {
            "asset_id": asset_id,
            "asset_name": asset_name,
            "version": asset_data.get('version', '1.0'),
            "author": asset_data.get('author', 'Unknown'),
            "installed_to": install_path,
            "files_extracted": len(extracted_files),
            "is_plugin": plugin_cfg_path is not None,
            "plugin_config": plugin_cfg_path,
            "backup_created": backup_path,
            "godot_version": asset_data.get('godot_version', 'Unknown'),
            "description": asset_data.get('description', '')[:200] + '...' if len(asset_data.get('description', '')) > 200 else asset_data.get('description', '')
        }
        
        print(f"ASSET_INSTALL: Successfully installed {asset_name}")
        
        return {
            "success": True,
            "message": f"Successfully installed {asset_name}",
            "installation_info": installation_info
        }
        
    except Exception as e:
        import traceback
        print(f"ASSET_INSTALL_ERROR: {e}")
        print(f"ASSET_INSTALL_TRACEBACK: {traceback.format_exc()}")
        return {"success": False, "error": f"Asset installation failed: {str(e)}"}


@app.route('/search_docs', methods=['POST'])
def search_docs():
    """HTTP endpoint to search across the shared Godot docs corpus.
    Thin wrapper around search_across_godot_docs_internal.
    """
    # Optional server key gate
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate

    try:
        data = request.json or {}
        query = data.get('query', '')
        if not query:
            return jsonify({"success": False, "error": "Query parameter is required"}), 400
        max_results = int(data.get('max_results', 5))
        result = search_across_godot_docs_internal({
            'query': query,
            'max_results': max_results,
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# --- Optional 3D Model Generation Endpoints ---
# Only available when properly configured via environment variables

def _forward_to_3d_service(endpoint: str, method: str = 'GET', **kwargs):
    """Helper to forward requests to 3D model service with authentication"""
    if not MODEL_3D_ENABLED:
        return jsonify({
            'error': '3D model generation not available',
            'message': 'Service not configured. Contact administrator.',
            'available': False
        }), 503
    
    try:
        url = f"{MODEL_3D_SERVICE_URL}/{endpoint.lstrip('/')}"
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {MODEL_3D_SECRET_KEY}'
        headers['X-Forwarded-For'] = request.environ.get('REMOTE_ADDR', 'unknown')
        headers['User-Agent'] = 'Godot-AI-Backend/1.0'
        
        kwargs['headers'] = headers
        kwargs['timeout'] = kwargs.get('timeout', 60)
        
        if method.upper() == 'POST':
            response = requests.post(url, **kwargs)
        else:
            response = requests.get(url, **kwargs)
            
        # Filter headers to avoid conflicts
        safe_headers = {}
        for key, value in response.headers.items():
            if key.lower() not in ['content-length', 'content-encoding', 'transfer-encoding', 'connection']:
                safe_headers[key] = value
        
        return Response(
            response.content,
            status=response.status_code,
            headers=safe_headers,
            mimetype=response.headers.get('content-type', 'application/json')
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': '3D service unavailable',
            'message': 'Failed to connect to 3D model generation service',
            'available': False
        }), 502

@app.route('/api/3d/health', methods=['GET'])
def model_3d_health():
    """Check 3D model generation service availability"""
    print("DEBUG: /api/3d/health endpoint hit!")
    if not MODEL_3D_ENABLED:
        return jsonify({
            'available': False,
            'message': '3D model generation not configured',
            'config_required': ['MODEL_3D_SERVICE_URL', 'MODEL_3D_SECRET_KEY', 'MODEL_3D_ENABLED=true']
        })
    
    return _forward_to_3d_service('health')

@app.route('/api/3d/generate/text', methods=['POST'])
def generate_3d_from_text():
    """Generate 3D model from text prompt"""
    print("DEBUG: /api/3d/generate/text endpoint hit!")
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    if not MODEL_3D_ENABLED:
        return jsonify({
            'error': '3D model generation not available',
            'message': 'Service not configured'
        }), 503
    
    try:
        data = request.json or {}
        if not data.get('prompt'):
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Add user_id based on IP (as expected by Point-E service)
        user_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        data['user_id'] = f"user_{user_ip.replace('.', '_')}"
        
        return _forward_to_3d_service(
            'generate/text',
            method='POST',
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
    except Exception as e:
        return jsonify({
            'error': 'Generation failed',
            'message': str(e)
        }), 500

@app.route('/api/3d/generate/image', methods=['POST'])
def generate_3d_from_image():
    """Generate 3D model from image"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    if not MODEL_3D_ENABLED:
        return jsonify({
            'error': '3D model generation not available',
            'message': 'Service not configured'
        }), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image file is required'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Validate file size (max 10MB)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'Image file too large (max 10MB)'}), 413
        
        # Forward multipart data
        files = {'image': (file.filename, file, file.content_type)}
        
        # Add user_id based on IP (as expected by Point-E service)
        user_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        data = {'user_id': f"user_{user_ip.replace('.', '_')}"}
        
        return _forward_to_3d_service(
            'generate/image',
            method='POST',
            files=files,
            data=data
        )
        
    except Exception as e:
        return jsonify({
            'error': 'Generation failed',
            'message': str(e)
        }), 500

@app.route('/api/3d/download/<path:filename>', methods=['GET'])
def download_3d_model(filename: str):
    """Download generated 3D model file"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    return _forward_to_3d_service(f'download/{filename}')

@app.route('/api/3d/models/<user_id>', methods=['GET'])
def list_user_3d_models(user_id: str):
    """List 3D models for a user"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    # Basic authorization - users can only see their own models
    if hasattr(g, 'user_id') and g.user_id != user_id:
        return jsonify({'error': 'Access denied'}), 403
    
    return _forward_to_3d_service(f'models/{user_id}')

if __name__ == '__main__':
    # Print 3D service status on startup
    if MODEL_3D_ENABLED:
        print(f"3D_GENERATION: Enabled, forwarding to {MODEL_3D_SERVICE_URL}")
    else:
        print("3D_GENERATION: Disabled (configure MODEL_3D_* environment variables to enable)")
    
    # Local dev only; in production use Gunicorn (configured in Dockerfile)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)