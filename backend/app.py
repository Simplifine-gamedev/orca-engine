"""
© 2025 Simplifine Corp. Original backend contribution for this Godot fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""
from flask import Flask, request, Response, jsonify, redirect, session, stream_with_context, g
from flask_cors import CORS
import openai
from litellm import completion, token_counter, get_max_tokens
import litellm
import json
import os
from typing import Optional
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
import logging
import copy
import jwt
from jwt import PyJWKClient, InvalidTokenError, PyJWKClientError
from Godot_tools import godot_tools
try:
    from weaviate_vector_manager import WeaviateVectorManager
except Exception:
    WeaviateVectorManager = None
try:
    from local_vector_manager import LocalVectorManager
except Exception:
    LocalVectorManager = None
from auth_manager import AuthManager
from auto_update_manager import auto_update_manager
from tool_logger import log_tool_call, log_tool_result
from version_checker import version_checker
from todo_store import TodoStore
from autumn_integration import AutumnPricingService

# app = Flask(__name__)
# CORS(app)

# Load environment variables from .env file
load_dotenv()

# --- LiteLLM Custom Logging Setup ---
# Initialize custom logger for LiteLLM API calls
# Auto-detect logging server URL based on DEV_MODE
_dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'

# Choose logging server URL based on environment

# Summarization Configuration Constants
UNIVERSAL_SUMMARIZATION_TRIGGER = 150000  # Trigger summarization at 150k tokens
EMERGENCY_THRESHOLD = 180000  # Emergency threshold at 180k tokens
if _dev_mode:
    # Development mode: use local logging server
    LOGGING_SERVER_URL = 'http://localhost:3031'
    print(f"🧪 DEV_MODE: Using local logging server: {LOGGING_SERVER_URL}")
else:
    # Production mode: use configured cloud logging server
    LOGGING_SERVER_URL = os.getenv('LOGGING_SERVER_URL')
    if LOGGING_SERVER_URL:
        print(f"☁️  PRODUCTION: Using cloud logging server: {LOGGING_SERVER_URL}")
    else:
        print("ℹ️  PRODUCTION: LOGGING_SERVER_URL not configured")

# Initialize the logger if we have a URL and logging is enabled
detailed_logging_enabled = os.getenv('DETAILED_LOGGING', 'false').lower()
if detailed_logging_enabled == 'false':
    print("ℹ️  LiteLLM logging disabled via DETAILED_LOGGING=false")
    litellm_logger = None
elif LOGGING_SERVER_URL and detailed_logging_enabled in ['true', 'auto']:
    try:
        from litellm_callback import GodotLiteLLMLogger
        litellm_logger = GodotLiteLLMLogger(LOGGING_SERVER_URL)
        litellm.drop_params = True  # CRITICAL: Fix GPT-5.x temperature errors
        litellm.callbacks = [litellm_logger]
        mode = "DEV" if _dev_mode else "PROD"
        logging_mode = "FORCED" if detailed_logging_enabled == 'true' else "AUTO"
        print(f"✅ LiteLLM logging enabled ({mode}/{logging_mode}): {LOGGING_SERVER_URL}")
    except ImportError as e:
        print(f"⚠️  LiteLLM logging disabled: {e}")
        litellm_logger = None
else:
    print("ℹ️  LiteLLM logging disabled: no server URL available or DETAILED_LOGGING=false")
    litellm.drop_params = True  # CRITICAL: Fix GPT-5.x temperature errors even without logging
    litellm_logger = None

# Print final logging status
if detailed_logging_enabled == 'true':
    print("🔍 DETAILED_LOGGING: ENABLED (forced via DETAILED_LOGGING=true)")
elif detailed_logging_enabled == 'auto':
    cloud_detected = bool(os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('CLOUD_RUN_JOB'))
    mode = "CLOUD" if cloud_detected else "LOCAL"
    print(f"📊 DETAILED_LOGGING: AUTO ({mode} mode)")
else:
    print("🔇 DETAILED_LOGGING: DISABLED (default - set DETAILED_LOGGING=true to enable)")
# --- End LiteLLM Setup ---

# Vertex AI configuration for Claude models (ENABLED)
VERTEX_AI_PROJECT = os.getenv('VERTEXAI_PROJECT')
VERTEX_AI_LOCATION = os.getenv('VERTEXAI_LOCATION', 'us-east5')  # Claude 4 works in us-east5
VERTEX_AI_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Set up Vertex AI authentication for Claude models
if VERTEX_AI_PROJECT:
    os.environ['VERTEXAI_PROJECT'] = VERTEX_AI_PROJECT
    os.environ['VERTEXAI_LOCATION'] = VERTEX_AI_LOCATION
    
    if VERTEX_AI_CREDENTIALS_PATH:
        # Use explicit credentials file if provided
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = VERTEX_AI_CREDENTIALS_PATH
        print(f"VERTEX_AI: Using credentials from {VERTEX_AI_CREDENTIALS_PATH}")
    else:
        # Auto-detect vertex-ai-key.json in current directory
        import os.path
        key_file = os.path.join(os.path.dirname(__file__), 'vertex-ai-key.json')
        if os.path.exists(key_file):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file
            print(f"VERTEX_AI: Auto-detected credentials at {key_file}")
        else:
            print("VERTEX_AI: Using default GCP authentication (gcloud CLI credentials)")
    
    print(f"VERTEX_AI: Configured for project {VERTEX_AI_PROJECT} in location {VERTEX_AI_LOCATION}")
    print("VERTEX_AI: Claude models will use Vertex AI by default")
else:
    print("WARNING: VERTEXAI_PROJECT not set - Claude models will use direct Anthropic API")

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

# --- Simple, environment-aware logging ---
_is_cloud = bool(os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('CLOUD_RUN_JOB'))
_structured_mode = (os.getenv('STRUCTURED_LOGS', 'auto').lower())
if _structured_mode == 'auto':
    STRUCTURED_LOGS = _is_cloud
else:
    STRUCTURED_LOGS = (_structured_mode == 'json')

def _anon(val: str | None) -> str | None:
    try:
        if not val:
            return None
        return hashlib.sha256(str(val).encode('utf-8')).hexdigest()[:16]
    except Exception:
        return None

def _should_emit_for_local_request() -> bool:
    try:
        # Check DETAILED_LOGGING environment variable first (overrides all other logic)
        detailed_logging_env = os.getenv('DETAILED_LOGGING', '').lower()
        if detailed_logging_env == 'true':
            return True
        elif detailed_logging_env == 'false':
            return False
        # If DETAILED_LOGGING not set, use existing logic
        
        # Always emit in cloud (structured logs)
        if STRUCTURED_LOGS:
            return True
        # Local dev: only when detailed_log=true is present
        if request:
            q = (request.args.get('detailed_log') or '').strip().lower()
            h = (request.headers.get('X-Detailed-Log') or '').strip().lower()
            return q == 'true' or h == 'true'
        return False
    except Exception:
        # Be conservative locally
        return False

def debug_print(message: str) -> None:
    """Print debug messages only when detailed logging is enabled"""
    try:
        if _should_emit_for_local_request():
            print(message)
    except Exception:
        pass

def log_event(event_name: str, props: dict | None = None, severity: str = 'INFO') -> None:
    try:
        # Gate local printing unless detailed_log=true
        if not _should_emit_for_local_request():
            return
        payload = {
            'event': event_name,
            'severity': severity,
            'ts': int(time.time() * 1000),
            'service': 'godot-ai-multi-model-service',
            'component': 'backend',
        }
        # Attach request context if available
        try:
            payload['method'] = request.method
            payload['path'] = request.path
            payload['request_id'] = getattr(g, 'request_id', None)
        except Exception:
            pass
        if props:
            payload['props'] = props
        if STRUCTURED_LOGS:
            print(json.dumps(payload, separators=(",", ":")))
        else:
            # Simple human-readable line for local dev
            msg = f"[{payload.get('severity')}] {payload.get('event')} path={payload.get('path')} props={props or {}}"
            print(msg)
    except Exception:
        pass

def _count_tokens_for_messages(messages: list, model: str) -> int:
    """Use LiteLLM's actual token counter for accurate counting"""
    try:
        # Use LiteLLM's built-in token counter
        token_count = token_counter(model=model, messages=messages)
        return token_count
    except Exception as e:
        print(f"TOKEN_COUNTER: Error using LiteLLM token_counter ({e}), falling back to estimation")
        # Fallback to rough estimation if token_counter fails
        total = 0
        for msg in messages:
            content = str(msg.get('content', ''))
            total += len(content) // 2  # Conservative estimate
        return total

def _get_model_token_limit(model: str) -> int:
    """Get INPUT context window limit for model (manually set, with safety margins)"""
    
    # CRITICAL: These are INPUT context limits with safety margins built in
    # We use 180k for Claude-4 (actual: 200k) and OpenAI (actual: 128k) for safety
    
    input_context_limits = {
        # Anthropic models - 180k with 20k safety margin
        "anthropic/claude-sonnet-4-20250514": 180000,
        "anthropic/claude-3-5-sonnet-20241022": 180000,
        "anthropic/claude-3-5-sonnet": 180000,
        "anthropic/claude-3-opus": 180000,
        
        # Vertex AI Claude models - same limits as direct Anthropic
        "vertex_ai/claude-sonnet-4@20250514": 180000,
        "vertex_ai/claude-3-5-sonnet@20240620": 180000,
        "vertex_ai/claude-3-opus@20240229": 180000,
        
        # OpenAI models - 120k with 8k safety margin
        "openai/gpt-5.1": 120000,
        "openai/gpt-5": 120000,
        "openai/gpt-4o": 120000,
        "openai/gpt-4o-mini": 120000,
        "openai/gpt-4-turbo": 120000,
        
        # Google models - conservative limits
        "gemini/gemini-2.5-pro": 1800000,  # 2M actual, using 1.8M
        "gemini/gemini-2.0-flash-exp": 900000,  # 1M actual, using 900k
        
        # Cerebras models
        "cerebras/llama-3.3-70b": 120000,
        "cerebras/llama3.1-8b": 120000,
    }
    
    # Try exact match first
    if model in input_context_limits:
        return input_context_limits[model]
    
    # Try partial match (handles model variants)
    for key, limit in input_context_limits.items():
        if key in model or model in key:
            return limit
    
    # Conservative default for unknown models
    print(f"⚠️ TOKEN_LIMIT: Unknown model {model}, using conservative 100k default")
    return 100000

def _detect_and_fix_orphaned_tool_calls(messages: list, error_message: str) -> tuple[list, bool]:
    """
    CRITICAL RECOVERY MECHANISM: Detect orphaned tool calls and create placeholder results.
    
    PURPOSE:
    When backend tool execution fails (due to network issues, service interruptions, etc.),
    we may end up with an assistant message containing tool_calls but missing the corresponding
    tool result messages. This violates OpenAI's API contract and causes the dreaded error:
    
    "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'"
    
    When this happens, users get completely stuck - they can't send new messages, stop the
    conversation, or recover in any way. The UI becomes completely unresponsive.
    
    SOLUTION:
    This function detects the specific error, extracts the missing tool_call_ids, and creates
    placeholder tool result messages that gracefully explain what happened. This allows the
    conversation to continue and users to try their request again.
    
    SAFETY:
    - Only triggers on the exact error signature to avoid false positives
    - Creates appropriate placeholder results based on tool type
    - Limited to 2 recovery attempts per conversation to prevent infinite loops
    - Logs recovery attempts for monitoring
    
    Args:
        messages: The conversation messages list
        error_message: The error message from LiteLLM
        
    Returns:
        (fixed_messages, was_fixed): Fixed messages and whether recovery succeeded
    """
    try:
        error_str = str(error_message).lower()
        if "tool_calls" not in error_str or "tool_call_id" not in error_str:
            return messages, False
        
        # Extract missing tool_call_ids from error message
        import re
        # Pattern matches: "The following tool_call_ids did not have response messages: tool_id1, tool_id2"
        # Use the original error message (not lowercased) to preserve tool call ID case
        pattern = r"the following tool_call_ids did not have response messages:\s*([^\n]+)"
        match = re.search(pattern, str(error_message), re.IGNORECASE)
        
        if not match:
            return messages, False
        
        # Parse the missing tool call IDs (preserve original case)
        missing_ids_str = match.group(1).strip()
        missing_tool_call_ids = [tid.strip() for tid in missing_ids_str.split(',')]
        
        print(f"TOOL_CALL_RECOVERY: Detected orphaned tool calls: {missing_tool_call_ids}")
        
        # Find the assistant messages with these tool calls to understand what tools were called
        tool_call_details = {}
        for msg in messages:
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tool_call in msg['tool_calls']:
                    tc_id = tool_call.get('id')
                    if tc_id in missing_tool_call_ids:
                        tool_call_details[tc_id] = {
                            'name': tool_call.get('function', {}).get('name', 'unknown_tool'),
                            'arguments': tool_call.get('function', {}).get('arguments', '{}')
                        }
        
        if not tool_call_details:
            print(f"TOOL_CALL_RECOVERY: Could not find details for missing tool calls")
            return messages, False
        
        # Create placeholder tool result messages for the missing tool calls
        fixed_messages = messages.copy()
        
        for tool_call_id, details in tool_call_details.items():
            tool_name = details['name']
            
            # Create an appropriate placeholder result based on the tool type
            if tool_name == 'image_operation':
                placeholder_result = {
                    "success": False,
                    "error": "Tool execution was interrupted. Please try the image operation again.",
                    "recovery_mode": True
                }
            elif tool_name == 'search_across_project':
                placeholder_result = {
                    "success": False,
                    "error": "Search was interrupted. Please try searching again.",
                    "recovery_mode": True,
                    "results": {"similar_files": [], "central_files": [], "graph_summary": {}}
                }
            elif tool_name == 'search_across_godot_docs':
                placeholder_result = {
                    "success": False,
                    "error": "Docs search was interrupted. Please try searching again.",
                    "recovery_mode": True,
                    "results": []
                }
            else:
                # Generic placeholder for any other tool
                placeholder_result = {
                    "success": False,
                    "error": f"Tool '{tool_name}' execution was interrupted. Please try again.",
                    "recovery_mode": True
                }
            
            # Add the placeholder tool result message
            tool_result_msg = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(placeholder_result)
            }
            
            fixed_messages.append(tool_result_msg)
            print(f"TOOL_CALL_RECOVERY: Added placeholder result for {tool_name} (ID: {tool_call_id})")
        
        print(f"TOOL_CALL_RECOVERY: Successfully created {len(tool_call_details)} placeholder results")
        return fixed_messages, True
        
    except Exception as recovery_error:
        print(f"TOOL_CALL_RECOVERY: Failed to fix orphaned tool calls: {recovery_error}")
        return messages, False

def _detect_and_fix_duplicate_tool_results(messages: list, error_message: str) -> tuple[list, bool]:
    """
    CRITICAL RECOVERY MECHANISM: Detect and fix duplicate tool result errors.
    
    PURPOSE:
    When frontend sends duplicate tool results or conversation state gets corrupted,
    we may end up with multiple tool result messages for the same tool_call_id.
    This violates both OpenAI and Anthropic API contracts causing errors like:
    
    Anthropic: "each tool_use must have a single result. Found multiple tool_result blocks with id: XXX"
    OpenAI: "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'"
    
    SOLUTION:
    This function detects duplicate tool results and keeps only the LAST (most recent) result
    for each tool_call_id, which typically contains the final/complete result.
    
    Args:
        messages: The conversation messages list  
        error_message: The error message from LiteLLM
        
    Returns:
        (fixed_messages, was_fixed): Fixed messages and whether recovery succeeded
    """
    try:
        error_str = str(error_message).lower()
        
        # Check for duplicate tool result signatures
        is_duplicate_error = (
            ("multiple" in error_str and "tool_result" in error_str and "blocks with id" in error_str) or
            ("each tool_use must have a single result" in error_str) or
            ("found multiple" in error_str and "tool_result" in error_str)
        )
        
        if not is_duplicate_error:
            return messages, False
        
        # Extract the problematic tool call ID from error message
        import re
        id_pattern = r"with id:\s*([^\s\"}']+)"
        match = re.search(id_pattern, str(error_message))
        
        if not match:
            print("DUPLICATE_TOOL_RECOVERY: Could not extract tool call ID from error message")
            return messages, False
        
        problematic_id = match.group(1)
        print(f"DUPLICATE_TOOL_RECOVERY: Detected duplicate results for tool_call_id: {problematic_id}")
        
        # Find all tool result messages with this ID
        duplicate_tool_results = []
        for i, msg in enumerate(messages):
            if (msg.get('role') == 'tool' and 
                msg.get('tool_call_id') == problematic_id):
                duplicate_tool_results.append((i, msg))
        
        if len(duplicate_tool_results) <= 1:
            print(f"DUPLICATE_TOOL_RECOVERY: Found {len(duplicate_tool_results)} results for {problematic_id} - no duplicates to fix")
            return messages, False
        
        print(f"DUPLICATE_TOOL_RECOVERY: Found {len(duplicate_tool_results)} duplicate results for {problematic_id}")
        
        # Keep only the LAST result (most recent/complete) and remove others
        fixed_messages = []
        indices_to_remove = set()
        
        # Mark all but the last duplicate for removal
        for i, (msg_index, msg) in enumerate(duplicate_tool_results[:-1]):  # All except last
            indices_to_remove.add(msg_index)
            print(f"DUPLICATE_TOOL_RECOVERY: Marking duplicate tool result at index {msg_index} for removal")
        
        # Rebuild messages without duplicates
        for i, msg in enumerate(messages):
            if i not in indices_to_remove:
                fixed_messages.append(msg)
        
        removed_count = len(messages) - len(fixed_messages)
        print(f"DUPLICATE_TOOL_RECOVERY: Removed {removed_count} duplicate tool results, kept the most recent one")
        
        return fixed_messages, True
        
    except Exception as recovery_error:
        print(f"DUPLICATE_TOOL_RECOVERY: Failed to fix duplicate tool results: {recovery_error}")
        return messages, False

def _generate_ai_summary(messages: list, summary_type: str, model: str = "openai/gpt-4o-mini") -> str:
    """Generate PRODUCTION-GRADE AI summary with technical details preserved"""
    try:
        # Use provided model (claude-4 for quality)
        summary_model = model
        
        # COMPREHENSIVE SUMMARIZATION PROMPT: Preserve maximum context for AI continuity
        summary_prompt = f"""Summarize this Godot development conversation for an AI assistant. This summary must preserve ALL important context for seamless continuation.

COMPREHENSIVE COVERAGE REQUIRED - Include ALL of these when present:

🎯 **PROJECT CONTEXT & GOALS**:
- What the user is building/trying to achieve
- Current project state and progress made
- Planned features or next steps mentioned

🛠️ **TECHNICAL DETAILS** (be specific):
- File paths: res://scripts/Player.gd, res://scenes/MainMenu.tscn
- Function names: _physics_process(), handle_input(), calculate_damage()
- Node structures: CharacterBody2D > CollisionShape2D > Sprite2D
- Property values: speed=300, health=100, emission_energy=15
- Scene hierarchies and component relationships

🐛 **PROBLEMS & SOLUTIONS**:
- Error messages encountered (exact text when possible)
- Root causes identified  
- Solutions implemented
- Code changes made (before/after when significant)

💡 **CODE IMPLEMENTATIONS**:
- Scripts created/modified with key logic
- Important variable names and their purposes
- Signal connections and data flows
- Resource references and dependencies

🎮 **GAME MECHANICS**:
- Player controls and movement systems
- Combat/interaction systems
- UI elements and their functions
- Audio/visual effects implemented

📝 **USER PREFERENCES & PATTERNS**:
- Coding style preferences shown
- Architectural decisions made
- Tools/approaches they prefer

CRITICAL: Be thorough! This summary replaces {len(messages)} messages - include enough detail for the AI to continue helping effectively without losing context.

Conversation ({len(messages)} messages):
"""
        
        # Extract ACTUAL CONVERSATION CONTENT (not metadata!)
        conversation_sample = []
        
        # Strategy: Sample the entire conversation to capture full context
        # Include: First 10, middle 10, last 10 for comprehensive coverage
        indices_to_sample = []
        
        # First 10 messages
        for i in range(min(10, len(messages))):
            indices_to_sample.append(i)
        
        # Middle 10 messages
        if len(messages) > 20:
            mid_start = len(messages) // 2 - 5
            for i in range(mid_start, min(mid_start + 10, len(messages))):
                if i not in indices_to_sample:
                    indices_to_sample.append(i)
        
        # Last 10 messages
        for i in range(max(0, len(messages) - 10), len(messages)):
            if i not in indices_to_sample:
                indices_to_sample.append(i)
        
        # Extract actual message content
        for i in sorted(indices_to_sample):
            msg = messages[i]
            role = msg.get('role', '')
            content = str(msg.get('content', ''))
            
            # Limit individual message length but keep substance
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            
            conversation_sample.append(f"[{i+1}] {role}: {content}")
        
        # Add sampled conversation to prompt
        summary_prompt += f"\n\n=== CONVERSATION SAMPLE ({len(conversation_sample)} key messages from {len(messages)} total) ===\n"
        summary_prompt += "\n\n".join(conversation_sample)
        
        # Call LLM for summarization with VERY generous token budget for comprehensive detail
        response = completion(
            model=summary_model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=5000,  # INCREASED: Much larger budget for comprehensive context preservation
            temperature=0.1,  # Minimum precision (GPT-5.x doesn't support 0.0)
            timeout=120,  # INCREASED: 2 minutes for comprehensive summaries in GCP
            request_timeout=120  # CRITICAL: Explicit request timeout for GCP Cloud Run
        )
        
        ai_summary = response.choices[0].message.content.strip()
        
        # Format the summary nicely
        formatted_summary = f"[{summary_type}] Conversation history ({len(messages)} messages condensed):\n\n"
        formatted_summary += ai_summary
        formatted_summary += f"\n\n📌 PRESERVED CONTEXT: Recent messages contain the most current state. This summary provides historical context for continuity."
        
        print(f"✅ AI_SUMMARY: Generated {len(ai_summary)} char summary using {summary_model}")
        return formatted_summary
        
    except Exception as e:
        print(f"⚠️ AI_SUMMARY: Failed to generate AI summary ({e}), using fallback")
        # Fallback to simple placeholder
        return f"[{summary_type}] {len(messages)} messages summarized for token efficiency. Key context preserved in recent messages."

def _manage_conversation_length_fallback(messages: list, model: str) -> tuple[list, bool]:
    """PRODUCTION-GRADE: Incremental summarization using LiteLLM's actual token counting
    
    TOKEN-ONLY TRIGGERING: Summarization is triggered purely based on token count,
    not message count. This prevents premature summarization of short conversations.
    
    Manages conversation length by creating AI-generated summaries of older messages
    while preserving recent messages for context. Uses actual token counting for accuracy.
    
    Returns: (managed_messages, was_summarized)
    """
    
    # Get ACTUAL model token limit using LiteLLM
    model_limit = _get_model_token_limit(model)
    print(f"TOKEN_LIMIT: Model {model} has limit of {model_limit} tokens (via LiteLLM)")

    # Test-mode override to trigger summarization sooner for local testing
    # Configure via ENV:
    #   SUMMARIZATION_TEST_MODE=true
    #   SUMMARIZATION_TEST_TRIGGER_PCT=0.08              (default 8%)
    #   SUMMARIZATION_TEST_EMERGENCY_PCT=0.15            (default 15%)
    #   SUMMARIZATION_TEST_MESSAGE_COUNT=8               (default 8 messages)
    #   SUMMARIZATION_TEST_KEEP_RECENT_NORMAL=12         (default 12)
    #   SUMMARIZATION_TEST_KEEP_RECENT_EMERGENCY=8       (default 8)
    #   SUMMARIZATION_TEST_MIN_INITIAL=3                 (default 3)
    #   SUMMARIZATION_TEST_MIN_INCREMENTAL=3             (default 3)
    
    # Test-mode override to trigger summarization sooner for local testing
    # Set SUMMARIZATION_TEST_MODE=true in environment to enable testing with lower barriers
    test_mode = os.getenv('SUMMARIZATION_TEST_MODE', 'false').lower() == 'true'

    if test_mode:
        try:
            trigger_ratio = float(os.getenv('SUMMARIZATION_TEST_TRIGGER_PCT', '0.4'))  # 40% instead of 50%
        except Exception:
            trigger_ratio = 0.4
        try:
            emergency_ratio = float(os.getenv('SUMMARIZATION_TEST_EMERGENCY_PCT', '0.6'))  # 60% instead of 75%
        except Exception:
            emergency_ratio = 0.6
        # Message count limit removed - using token-only triggering
        try:
            keep_recent_normal = int(os.getenv('SUMMARIZATION_TEST_KEEP_RECENT_NORMAL', '25'))  # 25 instead of 40
        except Exception:
            keep_recent_normal = 25
        try:
            keep_recent_emergency = int(os.getenv('SUMMARIZATION_TEST_KEEP_RECENT_EMERGENCY', '20'))  # 20 instead of 30
        except Exception:
            keep_recent_emergency = 20
        try:
            min_initial_needed = int(os.getenv('SUMMARIZATION_TEST_MIN_INITIAL', '8'))  # 8 instead of 10
        except Exception:
            min_initial_needed = 8
        try:
            min_incremental_needed = int(os.getenv('SUMMARIZATION_TEST_MIN_INCREMENTAL', '10'))  # 10 instead of 15
        except Exception:
            min_incremental_needed = 10
    else:
        # PRODUCTION SETTINGS: Conservative thresholds for real use
        trigger_ratio = 0.5          # 50% of token limit
        emergency_ratio = 0.75       # 75% of token limit
        # Message count limit removed - using token-only triggering
        keep_recent_normal = 40      # Keep last 40 messages
        keep_recent_emergency = 30   # Keep last 30 messages in emergency
        min_initial_needed = 10      # Need 10+ messages for first summary
        min_incremental_needed = 15  # Need 15+ messages for incremental summary

    # Use global universal trigger thresholds (defined at top of file)
    
    # Calculate ACTUAL token usage using LiteLLM's token_counter
    total_tokens = _count_tokens_for_messages(messages, model)
    model_limit = _get_model_token_limit(model)
    print(f"TOKEN_COUNT: Conversation using {total_tokens} tokens (limit: {model_limit}, trigger: {UNIVERSAL_SUMMARIZATION_TRIGGER})")
    
    # TOKEN-ONLY TRIGGER: Use only token count, ignore message count
    token_count_trigger = total_tokens > UNIVERSAL_SUMMARIZATION_TRIGGER
    
    if not token_count_trigger:
        return messages, False  # No management needed yet
    
    # Check if this is an emergency (over emergency threshold)
    is_emergency = total_tokens >= EMERGENCY_THRESHOLD
    
    if is_emergency:
        print(f"🚨 CONVERSATION_EMERGENCY: {total_tokens} tokens exceeds emergency limit ({EMERGENCY_THRESHOLD})! Forcing aggressive summarization")
    else:
        print(f"CONVERSATION_MANAGE: {total_tokens} tokens exceeds threshold ({UNIVERSAL_SUMMARIZATION_TRIGGER}), starting smart summarization")
    
    # PRODUCTION-GRADE INCREMENTAL SUMMARIZATION
    # Strategy: Keep system message + summaries + recent N messages
    # When summarizing: Create numbered summaries (Summary 1, Summary 2, etc.)

    initial_recent_messages_to_keep = keep_recent_emergency if is_emergency else keep_recent_normal

    # No model-specific overrides - universal behavior for all models
    
    recent_messages_to_keep = initial_recent_messages_to_keep
    
    # In test mode, use relaxed barriers but don't force summarization inappropriately
    if test_mode:
        print(f"🧪 TEST_MODE: Using testing barriers (60 messages, 40% tokens)")
    
    if len(messages) <= recent_messages_to_keep + 5:
        return messages, False  # Too short to summarize meaningfully
    
    # Separate system messages (NEVER summarize these!)
    system_messages = []
    conversation_messages = []
    for msg in messages:
        if msg.get('role') == 'system':
            system_messages.append(msg)
        else:
            conversation_messages.append(msg)
    
    # Find existing summaries
    existing_summaries = []
    last_summary_index = -1
    for i, msg in enumerate(conversation_messages):
        content = str(msg.get('content', ''))
        if msg.get('role') == 'assistant' and ('[SUMMARY' in content or 'CONVERSATION CONTEXT SUMMARY' in content):
            existing_summaries.append(msg)
            last_summary_index = i
    
    # Calculate how many messages to summarize
    total_conv_messages = len(conversation_messages)
    
    # DYNAMIC TOKEN-AWARE RECENT MESSAGE CALCULATION
    # If last k messages are >80% of tokens, reduce k to prevent over-preservation
    actual_recent_keep = recent_messages_to_keep
    if total_conv_messages > recent_messages_to_keep:
        # Calculate tokens for last k messages
        last_k_messages = conversation_messages[-recent_messages_to_keep:]
        last_k_tokens = _count_tokens_for_messages(last_k_messages, model)
        total_tokens_current = _count_tokens_for_messages(conversation_messages, model)
        
        if total_tokens_current > 0:
            last_k_percentage = (last_k_tokens / total_tokens_current) * 100
            print(f"🔍 TOKEN_BALANCE: Last {recent_messages_to_keep} messages = {last_k_tokens} tokens ({last_k_percentage:.1f}% of total)")
            
            # If recent messages are >80% of tokens, reduce them to leave room for meaningful summarization
            if last_k_percentage > 80 and recent_messages_to_keep > 1:
                # Reduce recent keep to maximum 70% of total messages or 50% of tokens, whichever is smaller
                max_by_message_count = max(1, int(total_conv_messages * 0.7))
                
                # Binary search to find optimal recent_keep that's ~50% of tokens
                target_percentage = 50
                left, right = 1, recent_messages_to_keep
                best_keep = recent_messages_to_keep
                
                for _ in range(10):  # Max 10 iterations
                    mid = (left + right) // 2
                    if mid >= total_conv_messages:
                        right = mid - 1
                        continue
                        
                    test_messages = conversation_messages[-mid:]
                    test_tokens = _count_tokens_for_messages(test_messages, model)
                    test_percentage = (test_tokens / total_tokens_current) * 100
                    
                    if test_percentage <= target_percentage:
                        best_keep = mid
                        left = mid + 1
                    else:
                        right = mid - 1
                
                actual_recent_keep = min(best_keep, max_by_message_count)
                print(f"📉 DYNAMIC_ADJUST: Reduced recent_keep from {recent_messages_to_keep} to {actual_recent_keep} (to prevent >80% token concentration)")
    
    recent_start_index = total_conv_messages - actual_recent_keep
    
    # CORE LOGIC VERIFICATION: 
    # Given n total messages (0,1,2,...,n-1):
    # - Messages 0 to (n-k-1) will be SUMMARIZED  → conversation_messages[:recent_start_index] 
    # - Messages (n-k) to (n-1) will be PRESERVED → conversation_messages[recent_start_index:]
    print(f"📊 SUMMARIZATION_LOGIC: n={total_conv_messages} messages total")
    print(f"📊   → Summarize: messages 0 to {recent_start_index-1} ({recent_start_index} messages)")  
    print(f"📊   → Preserve: messages {recent_start_index} to {total_conv_messages-1} ({actual_recent_keep} messages)")
    print(f"📊 CALCULATION: total_conv={total_conv_messages}, recent_keep={actual_recent_keep}, recent_start={recent_start_index}, last_summary_idx={last_summary_index}")
    
    if last_summary_index >= 0:
        # We have existing summaries - summarize messages between last summary and recent
        messages_to_summarize = conversation_messages[last_summary_index + 1:recent_start_index]
        
        if len(messages_to_summarize) < min_incremental_needed:  # Not enough to warrant new summary
            print(f"SUMMARIZATION_SKIP: Only {len(messages_to_summarize)} messages since last summary (need {min_incremental_needed}+)")
            # Emergency: just trim older messages more aggressively
            if is_emergency:
                print(f"EMERGENCY_TRIM: Keeping only existing summaries + last {actual_recent_keep} messages")
                
                # Add explanation for emergency trim if we have summaries
                if existing_summaries:
                    emergency_explanation = {
                        "role": "system", 
                        "content": "Above summaries were generated by an automatic summary generation system. This is only visible to you and the user will be seeing the actual messages that they have sent before, carry on the conversation."
                    }
                    return system_messages + existing_summaries + [emergency_explanation] + conversation_messages[-actual_recent_keep:], True
                else:
                    return system_messages + conversation_messages[-actual_recent_keep:], True
            print(f"SUMMARIZATION_SKIP: Returning original messages unchanged")
            return messages, False  # Keep as-is
        
        # Create REAL AI summary using the same model or fallback to gpt-4o-mini for speed
        summary_number = len(existing_summaries) + 1
        # Use current model if it's OpenAI, otherwise use fast gpt-4o-mini
        summary_model = model if model.startswith('openai/') else 'openai/gpt-4o-mini'
        print(f"📝 CONVERSATION_MANAGE: Generating AI summary #{summary_number} of {len(messages_to_summarize)} messages using {summary_model}...")
        
        summary_content = _generate_ai_summary(messages_to_summarize, f"INCREMENTAL SUMMARY {summary_number}", summary_model)
        
        new_summary = {"role": "assistant", "content": summary_content}
        
        # Add explanation message for the AI assistant
        summary_explanation = {
            "role": "system", 
            "content": "Above is the summary generated by an automatic summary generation system. This is only visible to you and the user will be seeing the actual messages that they have sent before, carry on the conversation."
        }
        
        # Build result: System + all summaries + new summary + explanation + recent messages
        result_messages = system_messages + existing_summaries + [new_summary, summary_explanation] + conversation_messages[recent_start_index:]
        
        print(f"✅ CONVERSATION_MANAGE: Created incremental summary #{summary_number}, total summaries: {len(existing_summaries) + 1}")
    else:
        # No existing summaries - create first summary
        messages_to_summarize = conversation_messages[:recent_start_index]
        
        if len(messages_to_summarize) < min_initial_needed:  # Need at least N messages to summarize
            print(f"CONVERSATION_MANAGE: Only {len(messages_to_summarize)} messages to summarize, skipping (need {min_initial_needed}+)")
            return messages, False
        
        # Use current model if it's OpenAI, otherwise use fast gpt-4o-mini
        summary_model = model if model.startswith('openai/') else 'openai/gpt-4o-mini'
        print(f"📝 CONVERSATION_MANAGE: Generating AI summary of {len(messages_to_summarize)} messages using {summary_model}...")
        
        # Create REAL AI summary using the selected model (or fast fallback)
        summary_content = _generate_ai_summary(messages_to_summarize, "INITIAL SUMMARY", summary_model)
        
        new_summary = {"role": "assistant", "content": summary_content}
        
        # Add explanation message for the AI assistant
        summary_explanation = {
            "role": "system", 
            "content": "Above is the summary generated by an automatic summary generation system. This is only visible to you and the user will be seeing the actual messages that they have sent before, carry on the conversation."
        }
        
        # Build result: System + new summary + explanation + recent messages  
        result_messages = system_messages + [new_summary, summary_explanation] + conversation_messages[recent_start_index:]
        
        print(f"✅ CONVERSATION_MANAGE: Created first summary, structure: 1 summary + {len(conversation_messages[recent_start_index:])} recent messages")
        print(f"✅ LOGIC_VERIFIED: Summarized messages 0-{recent_start_index-1}, preserved messages {recent_start_index}-{total_conv_messages-1}")
    
    # Verify we're under the limit using ACTUAL token counting
    result_tokens = _count_tokens_for_messages(result_messages, model)
    reduction_percent = ((total_tokens - result_tokens) / total_tokens * 100) if total_tokens > 0 else 0
    result_percent_of_limit = (result_tokens / model_limit * 100)
    
    print(f"✅ CONVERSATION_MANAGE: Reduced from {total_tokens} to {result_tokens} tokens ({len(messages)} to {len(result_messages)} messages)")
    print(f"📊 CONVERSATION_MANAGE: Saved {reduction_percent:.1f}% of tokens through summarization")
    print(f"📈 CONVERSATION_MANAGE: Now at {result_percent_of_limit:.1f}% of model limit ({result_tokens}/{model_limit})")
    print(f"🏗️  CONVERSATION_MANAGE: Structure: {len(system_messages)} system + {len(existing_summaries) + (1 if last_summary_index >= 0 or len(messages_to_summarize) >= 10 else 0)} summaries + {len(result_messages) - len(system_messages) - len(existing_summaries) - (1 if last_summary_index >= 0 or len(messages_to_summarize) >= 10 else 0)} recent")
    
    # SAFETY CHECK: If still over 85% after summarization, force more aggressive trimming
    if result_tokens > model_limit * 0.85:
        print(f"⚠️ STILL TOO LARGE: {result_tokens} tokens > 85% limit! Force trimming to last 15 messages...")
        
        # Extract summaries and add explanation
        summaries = [m for m in result_messages if m.get('role') == 'assistant' and '[SUMMARY' in str(m.get('content', ''))]
        force_trim_explanation = {
            "role": "system", 
            "content": "Above summaries were generated by an automatic summary generation system. This is only visible to you and the user will be seeing the actual messages that they have sent before, carry on the conversation."
        }
        
        if summaries:
            result_messages = system_messages + summaries + [force_trim_explanation] + result_messages[-15:]
        else:
            result_messages = system_messages + result_messages[-15:]
            
        result_tokens = _count_tokens_for_messages(result_messages, model)
        print(f"✅ FORCE_TRIM: Now {result_tokens} tokens ({(result_tokens/model_limit*100):.1f}% of limit)")
    
    return result_messages, True  # Return True to indicate summarization happened

app = Flask(__name__)

# Configure CORS for development and production
CORS(app, origins=["*"])

# Add request logging for debugging
@app.before_request
def log_request_info():
    try:
        if _should_emit_for_local_request():
            debug_print(f"DEBUG REQUEST: {request.method} {request.url} from {request.environ.get('REMOTE_ADDR')}")
        
        # CRITICAL: Enhanced GCP debugging - log all requests in cloud to detect hangs
        is_gcp_cloud = bool(os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('CLOUD_RUN_JOB'))
        if is_gcp_cloud:
            print(f"GCP_REQUEST_START: {request.method} {request.endpoint} from {request.environ.get('REMOTE_ADDR')} at {time.time()}")
            # Track request start time for hang detection
            g.gcp_request_start = time.time()
    except Exception:
        pass
    try:
        g.request_id = str(uuid.uuid4())
        g.request_started_at = time.time()
        
        # CRITICAL: Validate tools array hasn't been corrupted (sample every 100th request for performance)
        if hash(g.request_id) % 100 == 0:
            try:
                if not godot_tools or not isinstance(godot_tools, list) or len(godot_tools) == 0:
                    print(f"🚨 CORRUPTION DETECTED: godot_tools is empty or invalid!")
                elif "type" not in godot_tools[0]:
                    print(f"🚨 CORRUPTION DETECTED: godot_tools[0] missing 'type' field! Keys: {list(godot_tools[0].keys())}")
                elif godot_tools[0].get("type") != "function":
                    print(f"🚨 CORRUPTION DETECTED: godot_tools[0].type = '{godot_tools[0].get('type')}' (expected 'function')")
            except Exception as e:
                print(f"🚨 CORRUPTION DETECTED: Exception validating tools: {e}")
        
        # Generate conversation ID for logging (STABLE across entire conversation)
        try:
            user_id = request.headers.get('X-User-ID', 'anonymous')
            machine_id = request.headers.get('X-Machine-ID', 'unknown')
            
            # SIMPLE & STABLE: Just use user+machine (no time buckets)
            # This ensures the ENTIRE conversation thread has the same ID
            conv_seed = f"conv_{user_id}_{machine_id}"
            g.conversation_id = hashlib.md5(conv_seed.encode()).hexdigest()[:16]
            
            # Store for debugging
            g.conversation_seed = conv_seed
            
        except Exception:
            g.conversation_id = 'fallback_conv'
        log_event('request_start', {
            'content_length': request.content_length,
            'query_len': (len(request.query_string) if request.query_string else 0),
            'ip_h': _anon(request.environ.get('REMOTE_ADDR')),
        })
    except Exception:
        pass

@app.before_request 
def version_compatibility_check():
    """Check version compatibility before processing requests"""
    try:
        version_check_result = check_version_compatibility()
        if version_check_result:
            return version_check_result
    except Exception as e:
        print(f"VERSION_CHECK_ERROR: {e}")
        # Continue processing request if version check fails
        pass

@app.after_request
def log_request_end(response):
    try:
        started = getattr(g, 'request_started_at', None)
        dur_ms = int((time.time() - started) * 1000) if started else None
        
        # CRITICAL: Enhanced GCP hang detection
        is_gcp_cloud = bool(os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('CLOUD_RUN_JOB'))
        if is_gcp_cloud:
            gcp_start = getattr(g, 'gcp_request_start', None)
            if gcp_start:
                gcp_duration = time.time() - gcp_start
                print(f"GCP_REQUEST_END: {request.method} {request.endpoint} completed in {gcp_duration:.2f}s (status: {response.status_code})")
                # Log slow requests that could indicate hanging issues
                if gcp_duration > 30:
                    print(f"GCP_SLOW_REQUEST: ⚠️ Request took {gcp_duration:.2f}s - potential hang risk!")
                elif gcp_duration > 60:
                    print(f"GCP_VERY_SLOW_REQUEST: 🚨 Request took {gcp_duration:.2f}s - serious hang risk!")
        
        # Add anonymized hints only; never content
        uid = request.headers.get('X-User-ID') if request else None
        mid = request.headers.get('X-Machine-ID') if request else None
        log_event('request_end', {
            'status': response.status_code,
            'duration_ms': dur_ms,
            'user_h': _anon(uid),
            'machine_h': _anon(mid),
        })
    except Exception:
        pass
    return response

# Secret must be stable across restarts in production. Require env in production, random only in DEV_MODE.
# _dev_mode already defined above in LiteLLM logging setup section
DEPLOYMENT_MODE = os.getenv('DEPLOYMENT_MODE', 'oss').lower()  # 'oss' or 'cloud'
REQUIRE_SERVER_API_KEY = os.getenv('REQUIRE_SERVER_API_KEY', 'false').lower() == 'true'
SERVER_API_KEY = os.getenv('SERVER_API_KEY')

# Optional 3D Model Generation Service Integration
# Only enabled when all required environment variables are set
MODEL_3D_SERVICE_URL = os.getenv('MODEL_3D_SERVICE_URL')
MODEL_3D_SECRET_KEY = os.getenv('MODEL_3D_SECRET_KEY')  
_model_3d_enabled_env = os.getenv('MODEL_3D_ENABLED', 'false').lower()
print(f"DEBUG ENV: MODEL_3D_ENABLED={_model_3d_enabled_env}, MODEL_3D_SERVICE_URL={MODEL_3D_SERVICE_URL}, MODEL_3D_SECRET_KEY={'[SET]' if MODEL_3D_SECRET_KEY else '[NOT SET]'}")
MODEL_3D_ENABLED = bool(
    _model_3d_enabled_env == 'true' and
    MODEL_3D_SERVICE_URL and 
    MODEL_3D_SECRET_KEY
)
print(f"DEBUG RESULT: MODEL_3D_ENABLED={MODEL_3D_ENABLED}")

# Supabase Crash Reporting Integration (matches logging_server.py pattern)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for backend writes
CRASH_REPORTS_TABLE = os.getenv('CRASH_REPORTS_TABLE', 'crash_reports')
SUPABASE_CRASH_REPORTING_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)
ENFORCE_SUPABASE_IDENTITY = os.getenv(
    'ENFORCE_SUPABASE_IDENTITY',
    'true' if DEPLOYMENT_MODE == 'cloud' else 'false'
).lower() == 'true'
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/jwks" if SUPABASE_URL else None
SUPABASE_JWK_CLIENT = PyJWKClient(SUPABASE_JWKS_URL) if SUPABASE_JWKS_URL else None

if SUPABASE_CRASH_REPORTING_ENABLED:
    print(f"SUPABASE_CRASH_REPORTING: Enabled - storing to '{CRASH_REPORTS_TABLE}' table at {SUPABASE_URL}")
else:
    print("SUPABASE_CRASH_REPORTING: Disabled (set SUPABASE_URL and SUPABASE_SERVICE_KEY to enable)")
_secret_env = os.getenv('FLASK_SECRET_KEY')
if _secret_env:
    app.secret_key = _secret_env
elif _dev_mode:
    app.secret_key = os.urandom(24)
else:
    raise ValueError("FLASK_SECRET_KEY must be set in production")

# Multi-provider model configuration using LiteLLM
# Claude automatically uses Vertex AI (leverages your GCP credits)
def _get_claude_model():
    """
    Get Claude model - Vertex AI by default (uses your GCP credits).
    
    The frontend will only see 'claude-4' as an option, but the backend
    automatically uses Vertex AI for Claude to leverage your GCP credits.
    
    To switch to direct Anthropic API later, set: CLAUDE_PROVIDER=anthropic
    """
    # Check if user explicitly wants direct Anthropic
    if os.getenv("CLAUDE_PROVIDER", "").lower() == "anthropic":
        print("CLAUDE_CONFIG: Using direct Anthropic API (CLAUDE_PROVIDER=anthropic)")
        return os.getenv("CLAUDE_MODEL", "anthropic/claude-sonnet-4-20250514")
    
    # Default to Vertex AI if project is configured (uses GCP credits)
    if os.getenv('VERTEXAI_PROJECT'):
        print(f"CLAUDE_CONFIG: Using Vertex AI for Claude (Project: {os.getenv('VERTEXAI_PROJECT')})")
        return os.getenv("CLAUDE_MODEL", "vertex_ai/claude-sonnet-4@20250514")
    
    # Fallback to direct Anthropic if no Vertex project
    print("CLAUDE_CONFIG: No VERTEXAI_PROJECT found, falling back to direct Anthropic API")
    return os.getenv("CLAUDE_MODEL", "anthropic/claude-sonnet-4-20250514")

BASE_MODEL_MAP = {
    "gemini-2.5": os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-pro"),
    "claude-4": _get_claude_model(),  # Dynamic Claude selection (Vertex AI by default)
    "gpt-5.1": os.getenv("GPT51_MODEL", "openai/gpt-5.1"),
    "gpt-5": os.getenv("OPENAI_MODEL", "openai/gpt-5"),
    "gpt-4o": os.getenv("GPT4O_MODEL", "openai/gpt-4o"),
}

# Import LiteLLM supports_reasoning for dynamic thinking model detection
try:
    import litellm
    _litellm_supports_reasoning = getattr(litellm, 'supports_reasoning', None)
except ImportError:
    _litellm_supports_reasoning = None

def _create_thinking_variants(base_models: dict) -> dict:
    """Create thinking and non-thinking variants for models that support reasoning"""
    expanded_models = {}
    
    if not _litellm_supports_reasoning:
        print("WARNING: LiteLLM supports_reasoning not available - no thinking variants will be created")
        return base_models.copy()
    
    for friendly_name, model_id in base_models.items():
        try:
            if _litellm_supports_reasoning(model_id):
                # Add thinking variant
                thinking_name = f"{friendly_name} (thinking)"
                expanded_models[thinking_name] = model_id
                
                # Keep the original name as non-thinking (no suffix)
                expanded_models[friendly_name] = model_id
                
                print(f"THINKING_MODELS: Created variants for {friendly_name}: {thinking_name}, {friendly_name}")
            else:
                # Model doesn't support thinking, keep as-is
                expanded_models[friendly_name] = model_id
                print(f"THINKING_MODELS: No thinking support for {friendly_name}, keeping original")
        except Exception as e:
            print(f"WARNING: Failed to check reasoning support for {model_id}: {e}")
            # On error, keep the original model
            expanded_models[friendly_name] = model_id
    
    return expanded_models

# Dynamic model map that includes base + cerebras models + thinking variants
MODEL_MAP = _create_thinking_variants(BASE_MODEL_MAP)

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

# Load Cerebras models at startup and add thinking variants
cerebras_models = fetch_cerebras_models()
cerebras_with_thinking = _create_thinking_variants(cerebras_models)
MODEL_MAP.update(cerebras_with_thinking)

# Ensure LiteLLM has access to Cerebras API key
cerebras_api_key = os.getenv('CEREBRAS_API_KEY')
if cerebras_api_key:
    # Make sure LiteLLM can access the Cerebras API key
    os.environ['CEREBRAS_API_KEY'] = cerebras_api_key
    print(f"CEREBRAS_SETUP: API key configured for LiteLLM")
else:
    print("WARNING: CEREBRAS_API_KEY not found in environment - Cerebras models will fail")

# Ensure LiteLLM has access to Anthropic API key
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_api_key:
    # Make sure LiteLLM can access the Anthropic API key
    os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
    print(f"ANTHROPIC_SETUP: API key configured for LiteLLM")
else:
    print("WARNING: ANTHROPIC_API_KEY not found in environment - Anthropic models will fail")

# Default model and allowed models
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.1")
if DEFAULT_MODEL not in MODEL_MAP:
    if "gpt-5" in MODEL_MAP:
        fallback_default = "gpt-5"
    else:
        fallback_default = next(iter(MODEL_MAP), None)
    if fallback_default is None:
        raise RuntimeError("MODEL_MAP is empty; no models available for DEFAULT_MODEL")
    print(f"WARNING: DEFAULT_MODEL '{DEFAULT_MODEL}' not found in MODEL_MAP; falling back to '{fallback_default}'")
    DEFAULT_MODEL = fallback_default


def _get_openai_preferred_model() -> tuple[str, str]:
    """Return (friendly_name, provider_id) for the preferred OpenAI-tier model."""
    for candidate in ("gpt-5.1", "gpt-5", DEFAULT_MODEL):
        model_id = MODEL_MAP.get(candidate)
        if model_id:
            return candidate, model_id
    fallback_name, fallback_id = next(iter(MODEL_MAP.items()), (None, None))
    if fallback_name is None or fallback_id is None:
        raise RuntimeError("MODEL_MAP is empty; no models configured")
    return fallback_name, fallback_id


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

def _is_thinking_mode(model_friendly_name: str) -> bool:
    """Check if a model name indicates thinking mode is enabled"""
    return "(thinking)" in model_friendly_name.lower()

def _get_reasoning_params(model_friendly_name: str, model_id: str | None = None) -> dict:
    """Get reasoning/thinking params for LiteLLM based on model/provider.

    Rules (per LiteLLM v1.63+ docs):
    - Use reasoning_effort="low" when thinking is requested
    - Anthropic can additionally accept thinking={"type":"enabled","budget_tokens":1024}
    - Always keep temperature=1.0 when thinking is enabled
    - Pass drop_params=True so LiteLLM drops unsupported thinking params on provider switch
    """
    if not _is_thinking_mode(model_friendly_name):
        return {}

    params: dict = {
        "reasoning_effort": "low",
        "temperature": 1.0,
        "drop_params": True,
    }

    try:
        if model_id and str(model_id).startswith("anthropic/"):
            # Enable Anthropic thinking blocks explicitly
            params["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    except Exception:
        pass

    return params

# Initialize Authentication Manager
auth_manager = AuthManager()

# Initialize Autumn Pricing Service
pricing_service = AutumnPricingService()

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

# Load system prompts from files (once at startup)
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
SYSTEM_PROMPT_ASK_PATH = os.path.join(os.path.dirname(__file__), 'system_prompt_ask_mode.txt')
SYSTEM_PROMPT = None
SYSTEM_PROMPT_ASK = None

try:
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read().strip()
        if SYSTEM_PROMPT:
            print(f"SYSTEM_PROMPT (Agent): Loaded ({len(SYSTEM_PROMPT)} chars)")
        else:
            print("SYSTEM_PROMPT (Agent): File is empty; no system message will be prepended")
except Exception as e:
    print(f"SYSTEM_PROMPT (Agent): Failed to load: {e}")

try:
    with open(SYSTEM_PROMPT_ASK_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_ASK = f.read().strip()
        if SYSTEM_PROMPT_ASK:
            print(f"SYSTEM_PROMPT (Ask): Loaded ({len(SYSTEM_PROMPT_ASK)} chars)")
        else:
            print("SYSTEM_PROMPT (Ask): File is empty; will use agent prompt")
except Exception as e:
    print(f"SYSTEM_PROMPT (Ask): Failed to load: {e}")
    SYSTEM_PROMPT_ASK = SYSTEM_PROMPT  # Fallback to agent prompt

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
            return _enforce_supabase_identity({
                "id": effective_user,
                "name": "Dev User",
                "email": "dev@example.com",
                "provider": "dev_mode"
            })
    
    auth_header = request.headers.get('Authorization', '')
    machine_id = request.headers.get('X-Machine-ID') or (request.json.get('machine_id') if request.json else None)
    
    if not machine_id:
        return None, {"error": "machine_id required", "success": False}, 401
    
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        user = auth_manager.verify_session(machine_id, token)
        if user:
            return _enforce_supabase_identity(user)
    
    # If Supabase is configured, require Supabase authentication (no guest fallback)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase_user_id = request.headers.get('X-Supabase-User-ID')
        if not supabase_user_id:
            return None, {"error": "You need to login to the app", "success": False}, 401
        # Validate the Supabase user ID
        supabase_email = request.headers.get('X-Supabase-Email')
        ok, profile, error_message = verify_supabase_user_id(supabase_user_id.split(',')[0].strip() if supabase_user_id else None, supabase_email)
        if not ok:
            print(f"SUPABASE_AUTH_REQUIRED: Validation failed - {error_message}")
            return None, {"error": "You need to login to the app", "success": False}, 401
        # Create a user dict from Supabase profile
        supabase_user = {
            "id": profile.get('id', supabase_user_id.split(',')[0].strip()),
            "name": profile.get('email', supabase_email) or "Supabase User",
            "email": profile.get('email', supabase_email) or "",
            "provider": "supabase"
        }
        print(f"SUPABASE_AUTH_SUCCESS: Authenticated as {supabase_user['email']}")
        return supabase_user, None, None
    
    # Guest fallback if Supabase not configured and guests are allowed
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


def verify_supabase_user_id(user_id: str | None, email: str | None = None) -> tuple[bool, dict | None, str | None]:
    """Confirm that a Supabase auth user exists using the Admin API."""
    if not user_id:
        return False, None, "Supabase user_id required"

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False, None, "Supabase admin credentials are not configured"

    admin_url = f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}"
    headers = {
        'apikey': SUPABASE_SERVICE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
    }

    print(f"SUPABASE_CHECK: Checking user_id='{user_id}' at {admin_url}")
    try:
        response = requests.get(admin_url, headers=headers, timeout=5)
        print(f"SUPABASE_CHECK: Response status={response.status_code}, body={response.text[:200]}")
    except requests.RequestException as exc:
        print(f"SUPABASE_CHECK_ERROR: {exc}")
        return False, None, f"Supabase verification error: {exc}"

    if response.status_code == 200:
        profile = response.json()
        profile_email = profile.get('email')
        if email and profile_email and profile_email.lower() != email.lower():
            return False, None, "Supabase email mismatch"
        return True, profile, None

    if response.status_code == 404:
        return False, None, f"Supabase user not found (user_id: {user_id})"

    snippet = response.text[:200] if response.text else ""
    return False, None, f"Supabase verification failed (HTTP {response.status_code}) {snippet}"


def verify_supabase_jwt(token: str) -> tuple[bool, dict | None, str | None]:
    """Validate a Supabase access token using the project's JWKS."""
    if not SUPABASE_JWK_CLIENT:
        return False, None, "Supabase JWKS URL not configured"
    try:
        signing_key = SUPABASE_JWK_CLIENT.get_signing_key_from_jwt(token)
    except (PyJWKClientError, InvalidTokenError) as exc:
        return False, None, f"Supabase JWKS error: {exc}"
    try:
        decoded = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return True, decoded, None
    except InvalidTokenError as exc:
        return False, None, f"Supabase token decode error: {exc}"


def _enforce_supabase_identity(user: dict):
    """
    Simple validation: if X-Supabase-User-ID header is present, verify the user exists in Supabase.
    Returns (user, None, None) on success or (None, error_body, status_code) on failure.
    """
    supabase_user_id_raw = request.headers.get('X-Supabase-User-ID')
    if not supabase_user_id_raw:
        return user, None, None
    
    # Handle duplicate user IDs (take first one if comma-separated)
    supabase_user_id = supabase_user_id_raw.split(',')[0].strip()
    if supabase_user_id != supabase_user_id_raw:
        print(f"SUPABASE_CLEANUP: Fixed duplicated user_id '{supabase_user_id_raw}' -> '{supabase_user_id}'")
    
    # Simple check: verify user exists in Supabase
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase_email = request.headers.get('X-Supabase-Email')
        ok, profile, error_message = verify_supabase_user_id(supabase_user_id, supabase_email)
        if not ok:
            print(f"SUPABASE_VALIDATION_FAILED: {error_message}")
            return None, {"success": False, "error": error_message or "Supabase user verification failed"}, 401
        print(f"SUPABASE_VALIDATION_SUCCESS: User {supabase_user_id} verified")
        # Enrich user with Supabase info
        enriched_user = dict(user)
        enriched_user['supabase_user_id'] = supabase_user_id
        enriched_user['supabase_email'] = profile.get('email', supabase_email) if profile else supabase_email
        return enriched_user, None, None
    else:
        # If Supabase not configured, skip validation
        print("SUPABASE_VALIDATION_SKIPPED: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        return user, None, None


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

def check_version_compatibility():
    """Check if frontend version is compatible with backend."""
    frontend_version = request.headers.get('X-Frontend-Version')
    frontend_api_version = request.headers.get('X-Frontend-API-Version')
    
    # Skip version checking for health endpoint and version endpoint itself
    if request.endpoint in ['health_check', 'get_version_info']:
        return None
    
    # Skip if no version headers (for backwards compatibility with old frontends)
    if not frontend_version or not frontend_api_version:
        # Log old client for monitoring during transition
        log_event('legacy_client_detected', {
            'user_agent': request.headers.get('User-Agent', ''),
            'endpoint': request.endpoint
        }, 'INFO')
        return None
    
    compatible, error = version_checker.check_compatibility(frontend_version, frontend_api_version)
    
    if not compatible:
        return jsonify({
            "success": False,
            "error": "Version compatibility check failed",
            "version_error": error,
            "backend_version": version_checker.backend_version,
            "backend_api_version": version_checker.api_version,
            "frontend_version": frontend_version,
            "frontend_api_version": frontend_api_version,
            "compatibility_info": version_checker.get_compatibility_status(frontend_version, frontend_api_version)
        }), 409  # 409 Conflict for version mismatch
    
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
            debug_print(f"IMAGE_OP DEBUG: conversation_messages count: {len(conversation_messages)}")
            cm_index = -1
            for msg in conversation_messages:
                cm_index += 1
                if not isinstance(msg, dict):
                    continue
                    
                # Debug: Log all message details
                role = msg.get('role', 'unknown')
                print(f"    - msg[{cm_index}] role={role}, has_images={'images' in msg}, keys={list(msg.keys())}")
                
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
                if 'image_name' in content_preview or 'Image ID' in content_preview or 'image_id' in content_preview:
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
                        debug_print(f"IMAGE_OP DEBUG: tolerant match for '{img_id}' -> '{key}'")
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
            
            # Generate unique image ID for conversation tracking
            image_id = f"generated_{uuid.uuid4().hex[:8]}"
            
            result = {
                "success": True,
                "image_id": image_id,
                "image_name": image_id,  # For backward compatibility
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
        
        # Generate unique image ID for edited image
        image_id = f"edited_{uuid.uuid4().hex[:8]}"
        
        result = {
            "success": True,
            "image_id": image_id,
            "image_name": image_id,  # For backward compatibility
            "image_data": image_base64,
            "prompt": description,
            "style": style,
            "format": "png",
            "width": out_w,
            "height": out_h,
            "input_images": 1,
            "requested_images": len(image_ids),
            "edited_from": image_ids[0] if image_ids else None  # Track source image
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
                
                # Generate unique ID for each frame
                frame_id = f"frame_{r:02d}_{c:02d}_{uuid.uuid4().hex[:6]}"
                
                frames.append({
                    'row': r,
                    'col': c,
                    'frame_id': frame_id,
                    'image_id': frame_id,  # Consistent with other image operations
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
def _trim_graph_context(graph_context: dict, max_nodes: int = 12, max_edges: int = 24) -> dict:
    """Return a lightweight graph snapshot to avoid overwhelming clients."""
    trimmed = {}
    for file_path, payload in (graph_context or {}).items():
        nodes = payload.get('nodes', [])
        edges = payload.get('edges', [])
        trimmed[file_path] = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes[:max_nodes],
            "edges": edges[:max_edges],
        }
    return trimmed


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
    Simplified Godot context boost - only basic scene-script pairing
    All hardcoded biases removed for neutral, predictable behavior
    """
    boost = 0.0
    
    # Scene-Script pair detection (this is objective structural information)
    base_name = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
    script_pair = f"{base_name}.gd"
    scene_pair = f"{base_name}.tscn"
    
    if file_path.endswith('.gd') and scene_pair in all_files:
        boost += 0.1  # Script with matching scene - minimal boost
    elif file_path.endswith('.tscn') and script_pair in all_files:
        boost += 0.1  # Scene with matching script - minimal boost
    
    return boost

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
        graph_preview = bool(arguments.get('graph_preview', False))
        
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
        
        # Use search mode as specified - no hardcoded auto-detection
        detected_mode = search_mode
        print(f"SEARCH_MODE: Using {detected_mode.upper()} search: {query}")
        
        # Search using cloud vector manager with specified mode
        if detected_mode == 'keyword':
            initial_results = cloud_vector_manager.keyword_search(query, user['id'], project_id, max_results * 2)
            print(f"SEARCH_EXECUTION: Performed KEYWORD search")
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
            graph_context = cloud_vector_manager.get_graph_context_for_files(
                files, user['id'], project_id
            )
        
        if graph_preview and graph_context:
            graph_payload = _trim_graph_context(graph_context)
        else:
            graph_payload = {}
        
        return {
            "success": True,
            "query": query,
            "search_mode": detected_mode,
            "results": formatted_results,
            "include_graph": include_graph,
            "trace_dependencies": trace_dependencies,
            "graph": graph_payload,
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

def generate_3d_model_internal(arguments: dict) -> dict:
    """Generate a 3D model from text prompt using the AI 3D service"""
    try:
        prompt = arguments.get('prompt', '')
        if not prompt:
            return {"success": False, "error": "Prompt is required for 3D model generation"}
        
        model = arguments.get('model', 'fast')
        save_path = arguments.get('save_path', '')
        
        print(f"3D_GENERATION: Generating model for prompt: '{prompt}' using model: {model}")
        

        # 3D Generation API endpoint - use local server for development
        if os.getenv('DEV_MODE', 'false').lower() == 'true':
            api_url = "http://127.0.0.1:3030/api/generate-3d"
            base_3d_url = "http://127.0.0.1:3030"
        else:
            # Use environment variable for 3D service URL, fallback to default project
            model_3d_service_url = os.getenv('MODEL_3D_SERVICE_URL', 'https://ai-3d-proxy-976792908107.us-central1.run.app')
            api_url = f"{model_3d_service_url}/api/generate-3d"
            base_3d_url = model_3d_service_url
        
        # Generate unique user ID for tracking
        user_id = f"godot_user_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        
        # Prepare request payload
        payload = {
            "user_id": user_id,
            "prompt": prompt,
            "model": model,
            "output_format": "glb"
        }
        
        print(f"3D_GENERATION: Sending request to {api_url}")
        
        # Make request to 3D generation service
        response = requests.post(api_url, json=payload, timeout=120)  # 2 minute timeout for 3D generation
        response.raise_for_status()
        
        result = response.json()
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error from 3D service')
            print(f"3D_GENERATION_ERROR: {error_msg}")
            return {"success": False, "error": f"3D generation failed: {error_msg}"}
        
        job_id = result.get('job_id', '')
        download_url = result.get('download_url', '')
        generation_time = result.get('generation_time', 0)
        model_used = result.get('model', model)
        
        if not download_url:
            return {"success": False, "error": "3D service did not provide download URL"}
        
        # Construct full download URL if it's relative
        if download_url.startswith('/'):
            full_download_url = base_3d_url + download_url
        else:
            full_download_url = download_url
        
        print(f"3D_GENERATION: Model generated successfully - Job ID: {job_id}, Download URL: {full_download_url}")
        
        # Download the GLB file
        print(f"3D_GENERATION: Downloading GLB file from {full_download_url}")
        glb_response = requests.get(full_download_url, timeout=60)
        glb_response.raise_for_status()
        
        if len(glb_response.content) == 0:
            return {"success": False, "error": "Downloaded GLB file is empty"}
        
        # Convert to base64 for transmission to frontend
        glb_base64 = base64.b64encode(glb_response.content).decode('utf-8')
        
        result_data = {
            "success": True,
            "prompt": prompt,
            "model": model_used,
            "generation_time": generation_time,
            "job_id": job_id,
            "download_url": download_url,
            "glb_data": glb_base64,
            "file_size": len(glb_response.content),
            "format": "glb"
        }
        
        if save_path:
            result_data["save_path"] = save_path
        
        print(f"3D_GENERATION: Successfully generated 3D model - Size: {len(glb_response.content)} bytes")
        return result_data
        
    except Exception as e:
        print(f"3D_GENERATION_ERROR: {str(e)}")
        return {"success": False, "error": f"3D model generation failed: {str(e)}"}

# --- Note: Script generation now handled by dedicated /generate_script endpoint ---

# --- Tool Execution Function ---
def check_for_app_updates_internal(arguments: dict) -> dict:
    """Check for Orca Engine updates and optionally show notification"""
    try:
        force_check = arguments.get('force_check', False)
        show_notification = arguments.get('show_notification', True)
        
        print(f"UPDATE_TOOL: Checking for updates (force={force_check}, notify={show_notification})")
        
        # Check for updates
        update_info = auto_update_manager.check_for_updates(force=force_check)
        
        if update_info:
            result = {
                "success": True,
                "update_available": True,
                "update_info": {
                    "version": update_info.version,
                    "current_version": auto_update_manager.current_version,
                    "download_url": update_info.download_url,
                    "file_size_mb": round(update_info.file_size / (1024 * 1024), 1),
                    "release_notes": update_info.release_notes,
                    "is_critical": update_info.is_critical,
                    "published_at": update_info.published_at
                },
                "message": f"Update available: v{update_info.version}",
                "show_popup": show_notification
            }
            
            if show_notification:
                result["popup_config"] = {
                    "title": "Update Available" if not update_info.is_critical else "Critical Update Available",
                    "message": f"Orca Engine v{update_info.version} is now available.\n\nCurrent version: v{auto_update_manager.current_version}",
                    "buttons": ["Install Now", "Later"],
                    "default_button": 0 if update_info.is_critical else 1,
                    "icon": "warning" if update_info.is_critical else "info"
                }
            
            return result
        else:
            return {
                "success": True,
                "update_available": False,
                "current_version": auto_update_manager.current_version,
                "message": "You have the latest version of Orca Engine",
                "show_popup": False
            }
            
    except Exception as e:
        print(f"UPDATE_TOOL_ERROR: {e}")
        return {
            "success": False,
            "error": f"Update check failed: {str(e)}"
        }


# --- New Consolidated Tool Handlers ---

def project_manager_internal(arguments: dict) -> dict:
    """Handle project_manager tool operations"""
    try:
        # CRITICAL: Check for tool generation failures first
        if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
            failure_type = arguments.get("_tool_gen_failure")
            original_size = arguments.get("_original_size", "unknown")
            
            error_msg = (
                f"Tool call failed due to large content ({original_size} characters). "
                "This is likely because the content was very big, make the content smaller and apply fewer edits. "
                "Break large operations into multiple smaller steps."
            )
            
            print(f"🚨 TOOL_GEN_FAILURE_DETECTED: {failure_type} with size {original_size}")
            return {"success": False, "error": error_msg}
        
        op = arguments.get('op', '')
        if not op:
            # DEBUG: Show what arguments were actually received when op is missing
            print("🚨 PROJECT_MANAGER ERROR: Missing 'op' parameter!")
            print("📋 RECEIVED ARGUMENTS:")
            for key, value in arguments.items():
                if isinstance(value, str) and len(value) > 200:
                    display_value = value[:200] + "... (truncated)"
                else:
                    display_value = value
                print(f"   {key}: {display_value}")
            print("❌ Tool execution failed due to missing 'op' parameter")
            return {"success": False, "error": "Operation 'op' parameter is required"}
        
        if op == "assets.search":
            # Route to existing asset search function
            search_args = {
                'query': arguments.get('asset_query', ''),
                'category': arguments.get('category'),
                'max_results': arguments.get('max_results', 10),
                'support_level': arguments.get('support_level', 'all'),
                'godot_version': arguments.get('godot_version', '4.3'),
                'sort_by': arguments.get('sort_by', 'rating'),
                'sort_reverse': arguments.get('sort_reverse', False),
                'asset_type': arguments.get('asset_type', 'any'),
                'cost_filter': arguments.get('cost_filter', 'all')
            }
            return search_godot_assets_internal(search_args)
            
        elif op == "assets.install":
            # Route to existing asset install function
            install_args = {
                'asset_id': arguments.get('asset_id'),
                'project_path': arguments.get('project_path'),
                'install_location': arguments.get('install_location', 'addons/'),
                'create_backup': arguments.get('create_backup', True)
            }
            return install_godot_asset_internal(install_args)
            
        elif op == "updates.check":
            # Route to existing update check function
            update_args = {
                'force_check': arguments.get('force_check', False),
                'show_notification': arguments.get('show_notification', True)
            }
            return check_for_app_updates_internal(update_args)
            
        elif op == "templates.list":
            # List available base game templates
            from template_manager import list_templates, get_categories
            category = arguments.get('template_category')
            templates = list_templates(category)
            categories = get_categories()
            return {
                "success": True,
                "templates": templates,
                "categories": categories,
                "count": len(templates),
                "filtered_by": category if category else "all"
            }
            
        elif op == "templates.install":
            # Install a base game template
            from template_manager import install_template
            template_id = arguments.get('template_id')
            target_path = arguments.get('target_path')
            
            if not template_id:
                return {
                    "success": False,
                    "error": "Missing required parameter 'template_id'"
                }
            
            if not target_path:
                return {
                    "success": False,
                    "error": "Missing required parameter 'target_path'"
                }
            
            # Install the template
            result = install_template(template_id, target_path)
            return result
            
        elif op in ["context.get", "fs.list", "fs.read", "fs.write_lines", "fs.replace_string", 
                   "fs.copy", "fs.move", "fs.delete", "fs.mkdir", "fs.symlink", "fs.refresh", 
                   "project.analyze_dir", "project.copy_dir", "project.update_refs"]:
            # These are frontend-only operations
            return {
                "success": False,
                "frontend_only": True,
                "message": f"Operation '{op}' is handled by the frontend. This should not be executed on the backend.",
                "operation": op,
                "arguments_to_forward": arguments
            }
        
        elif op == "fs.write":
            # CRITICAL FIX: Handle fs.write properly in backend to avoid false success reports
            file_path = arguments.get('path', '')
            content = arguments.get('content', '')
            encoding = arguments.get('encoding', 'utf-8')
            
            if not file_path:
                return {
                    "success": False,
                    "error": "Missing required parameter 'path' for fs.write operation"
                }
            
            # CRITICAL: Detect missing content parameter (likely due to JSON corruption)
            if content == '' or content is None:
                print(f"🚨 MISSING CONTENT: fs.write called for {file_path} with no content parameter!")
                print("🚨 This is likely due to JSON corruption from large content generation")
                return {
                    "success": False,
                    "error": (
                        f"Missing required parameter 'content' for fs.write operation on {file_path}. "
                        "This is likely because the content was very big, make the content smaller and apply fewer edits. "
                        "The content parameter was lost during tool call generation."
                    )
                }
                
            try:
                # Convert to absolute path if relative
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write the file
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                print(f"BACKEND: Successfully wrote file {file_path} ({len(content)} characters)")
                
                return {
                    "success": True,
                    "message": f"File written successfully: {file_path}",
                    "path": file_path,
                    "bytes_written": len(content.encode(encoding))
                }
            except Exception as e:
                error_msg = f"Failed to write file {file_path}: {str(e)}"
                print(f"BACKEND ERROR: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "path": file_path
                }
            
        else:
            return {"success": False, "error": f"Unknown project_manager operation: {op}"}
            
    except Exception as e:
        print(f"PROJECT_MANAGER_ERROR: {e}")
        return {"success": False, "error": f"Project manager operation failed: {str(e)}"}

def graph_manager_internal(arguments: dict, current_user: dict = None) -> dict:
    """Handle graph_manager tool operations"""
    try:
        def _normalize_graph_path(path_value: str, project_root_value: Optional[str]) -> str:
            if not path_value:
                return ""
            normalized = str(path_value).strip().replace("\\", "/")
            if normalized.startswith("res://"):
                normalized = normalized[6:]
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if project_root_value:
                try:
                    pr_norm = os.path.abspath(project_root_value)
                    candidate = normalized
                    if os.path.isabs(candidate):
                        candidate_abs = os.path.abspath(candidate)
                    else:
                        candidate_abs = os.path.abspath(os.path.join(pr_norm, candidate))
                    if os.path.commonprefix([candidate_abs, pr_norm]) == pr_norm:
                        normalized = os.path.relpath(candidate_abs, pr_norm).replace("\\", "/")
                except Exception:
                    pass
                base_name = os.path.basename(project_root_value.rstrip("/\\"))
                if base_name and base_name in normalized:
                    idx = normalized.find(base_name)
                    trimmed = normalized[idx + len(base_name):].lstrip("/\\")
                    if trimmed:
                        normalized = trimmed
            return normalized.strip("/")

        op = arguments.get('op', '')
        if not op:
            return {"success": False, "error": "Operation 'op' parameter is required"}

        if cloud_vector_manager is None:
            return {"success": False, "error": "Graph intelligence unavailable (vector manager not configured)"}

        user_obj = current_user
        if user_obj is None:
            user_obj, error_response, status_code = verify_authentication()
            if error_response:
                return {"success": False, "error": "Authentication required to access graph data"}

        user_id = user_obj.get('id') or 'guest'

        project_root = arguments.get('project_root') or getattr(g, 'project_root', None)
        project_id = arguments.get('project_id')
        if not project_id and project_root:
            project_id = hashlib.md5(project_root.encode()).hexdigest()
        if not project_id:
            return {"success": False, "error": "project_id or project_root required for graph operations"}

        if op == "graph.neighbors":
            raw_paths = []
            primary_path = arguments.get('file_path')
            if primary_path:
                raw_paths.append(primary_path)
            extra_files = arguments.get('file_paths') or []
            if isinstance(extra_files, list):
                raw_paths.extend([fp for fp in extra_files if fp])
            normalized_map = {}
            ordered_originals = []
            for raw in raw_paths:
                if raw not in normalized_map:
                    normalized_map[raw] = _normalize_graph_path(raw, project_root)
                    ordered_originals.append(raw)
            # Ensure at least one target
            normalized_values = [norm for norm in normalized_map.values() if norm]
            if not normalized_values:
                return {"success": False, "error": "Provide 'file_path' or 'file_paths' for graph.neighbors"}

            depth = max(1, int(arguments.get('depth', 1)))
            edge_types = arguments.get('edge_types') or arguments.get('kinds')
            max_nodes = max(1, min(int(arguments.get('max_nodes', 12)), 100))
            max_edges = max(1, min(int(arguments.get('max_edges', 24)), 400))
            include_summary = bool(arguments.get('include_summary', True))
            include_raw = bool(arguments.get('include_raw', False))

            if depth > 1 and hasattr(cloud_vector_manager, 'get_graph_context_expanded'):
                raw_context = cloud_vector_manager.get_graph_context_expanded(
                    normalized_values,
                    user_id,
                    project_id,
                    depth=depth - 1,  # depth includes starting nodes
                    kinds=edge_types,
                    max_nodes=max_nodes,
                    max_edges=max_edges,
                )
            else:
                raw_context = cloud_vector_manager.get_graph_context_for_files(
                    normalized_values, user_id, project_id
                )

            rekeyed_context = {}
            for original, normalized in normalized_map.items():
                if not normalized:
                    continue
                rekeyed_context[original] = raw_context.get(normalized, {"nodes": [], "edges": []})

            trimmed = _trim_graph_context(rekeyed_context, max_nodes=max_nodes, max_edges=max_edges)

            response = {
                "success": True,
                "graph": trimmed,
                "requested_files": ordered_originals,
                "returned_files": [f for f in ordered_originals if trimmed.get(f, {}).get("total_nodes")],
                "depth": depth
            }

            if include_summary:
                total_nodes = sum(entry.get("total_nodes", len(entry.get("nodes", []))) for entry in trimmed.values())
                total_edges = sum(entry.get("total_edges", len(entry.get("edges", []))) for entry in trimmed.values())
                response["summary"] = {
                    "requested_files": len(ordered_originals),
                    "returned_files": len([f for f in trimmed if trimmed[f].get("total_nodes")]),
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "edge_types": edge_types or "all"
                }

            if include_raw:
                response["graph_full"] = rekeyed_context

            return response

        elif op == "graph.walk":
            raw_paths = (
                arguments.get('start_files')
                or arguments.get('file_paths')
                or []
            )
            if not raw_paths:
                single = arguments.get('start_file') or arguments.get('file_path')
                if single:
                    raw_paths = [single]

            normalized_map = {}
            for raw in raw_paths:
                normalized = _normalize_graph_path(raw, project_root)
                if normalized:
                    normalized_map[raw] = normalized

            if not normalized_map:
                return {"success": False, "error": "Provide at least one 'start_file' or 'file_path' for graph.walk"}

            normalized_values = list(dict.fromkeys(normalized_map.values()))
            depth = max(0, int(arguments.get('depth', 2)))
            edge_types = arguments.get('edge_types') or arguments.get('kinds')
            max_nodes = max(1, min(int(arguments.get('max_nodes', 50)), 500))
            max_edges = max(1, min(int(arguments.get('max_edges', 200)), 1000))
            include_summary = bool(arguments.get('include_summary', True))
            include_raw = bool(arguments.get('include_raw', False))

            expanded_context = cloud_vector_manager.get_graph_context_expanded(
                normalized_values,
                user_id,
                project_id,
                depth=depth,
                kinds=edge_types,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )

            trimmed = _trim_graph_context(expanded_context, max_nodes=max_nodes, max_edges=max_edges)

            inverse_map = {}
            for original, normalized in normalized_map.items():
                if normalized not in inverse_map:
                    inverse_map[normalized] = original

            display_graph = {}
            for norm_key, payload in trimmed.items():
                display_key = inverse_map.get(norm_key)
                if not display_key:
                    display_key = norm_key if norm_key.startswith("res://") else f"res://{norm_key}"
                display_graph[display_key] = payload

            response = {
                "success": True,
                "graph": display_graph,
                "requested_files": list(normalized_map.keys()),
                "visited_files": list(display_graph.keys()),
                "depth": depth,
            }

            if include_summary:
                total_nodes = sum(entry.get("total_nodes", len(entry.get("nodes", []))) for entry in display_graph.values())
                total_edges = sum(entry.get("total_edges", len(entry.get("edges", []))) for entry in display_graph.values())
                response["summary"] = {
                    "requested_files": len(normalized_map),
                    "returned_files": len(display_graph),
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "edge_types": edge_types or "all",
                }

            if include_raw:
                raw_display = {}
                for norm_key, payload in expanded_context.items():
                    display_key = inverse_map.get(norm_key)
                    if not display_key:
                        display_key = norm_key if norm_key.startswith("res://") else f"res://{norm_key}"
                    raw_display[display_key] = payload
                response["graph_full"] = raw_display

            return response

        return {"success": False, "error": f"Unknown graph_manager operation: {op}"}

    except Exception as e:
        print(f"GRAPH_MANAGER_ERROR: {e}")
        return {"success": False, "error": f"Graph manager operation failed: {str(e)}"}

def search_manager_internal(arguments: dict, current_user: dict = None) -> dict:
    """Handle search_manager tool operations"""
    try:
        # CRITICAL: Check for tool generation failures first
        if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
            failure_type = arguments.get("_tool_gen_failure")
            original_size = arguments.get("_original_size", "unknown")
            
            error_msg = (
                f"Tool call failed due to large content ({original_size} characters). "
                "This is likely because the content was very big, make the content smaller and apply fewer edits."
            )
            
            print(f"🚨 SEARCH_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
            return {"success": False, "error": error_msg}
        
        op = arguments.get('op', '')
        if not op:
            # DEBUG: Show what arguments were actually received when op is missing
            print("🚨 SEARCH_MANAGER ERROR: Missing 'op' parameter!")
            print("📋 RECEIVED ARGUMENTS:")
            for key, value in arguments.items():
                if isinstance(value, str) and len(value) > 200:
                    display_value = value[:200] + "... (truncated)"
                else:
                    display_value = value
                print(f"   {key}: {display_value}")
            print("❌ Tool execution failed due to missing 'op' parameter")
            return {"success": False, "error": "Operation 'op' parameter is required"}
            
        if op == "project.search":
            # Route to existing project search function
            search_args = {
                'query': arguments.get('query'),
                'max_results': arguments.get('max_results', 5),
                'include_graph': arguments.get('include_graph', True),
                'modality_filter': arguments.get('modality_filter'),
                'project_root': arguments.get('project_root'),
                'project_id': arguments.get('project_id'),
                'trace_dependencies': arguments.get('trace_dependencies', False),
                'search_mode': arguments.get('search_mode', 'semantic')
            }
            return search_across_project_internal(search_args, current_user)
            
        elif op == "docs.search":
            # Route to existing docs search function
            docs_args = {
                'query': arguments.get('query'),
                'max_results': arguments.get('max_results', 5),
                'section_filter': arguments.get('section_filter'),
                'class_filter': arguments.get('class_filter'),
                'difficulty': arguments.get('difficulty'),
                'code_examples_only': arguments.get('code_examples_only', False)
            }
            return search_across_godot_docs_internal(docs_args)
            
        else:
            return {"success": False, "error": f"Unknown search_manager operation: {op}"}
            
    except Exception as e:
        print(f"SEARCH_MANAGER_ERROR: {e}")
        return {"success": False, "error": f"Search manager operation failed: {str(e)}"}

def resource_manager_internal(arguments: dict, conversation_messages: list = None) -> dict:
    """Handle resource_manager tool operations"""
    try:
        # CRITICAL: Check for tool generation failures first
        if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
            failure_type = arguments.get("_tool_gen_failure")
            original_size = arguments.get("_original_size", "unknown")
            
            error_msg = (
                f"Tool call failed due to large content ({original_size} characters). "
                "This is likely because the content was very big, make the content smaller and apply fewer edits."
            )
            
            print(f"🚨 RESOURCE_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
            return {"success": False, "error": error_msg}
        
        # Accept both 'op' and legacy 'operation'
        op = arguments.get('op', '') or arguments.get('operation', '')
        if not op:
            # DEBUG: Show what arguments were actually received when op is missing
            print("🚨 RESOURCE_MANAGER ERROR: Missing 'op' parameter!")
            print("📋 RECEIVED ARGUMENTS:")
            for key, value in arguments.items():
                if isinstance(value, str) and len(value) > 200:
                    display_value = value[:200] + "... (truncated)"
                else:
                    display_value = value
                print(f"   {key}: {display_value}")
            print("❌ Tool execution failed due to missing 'op' parameter")
            return {"success": False, "error": "Operation 'op' parameter is required"}
            
        # Backend-processed image operations
        if op == "image.generate_or_edit":
            # Route to existing image operation function
            image_args = {
                'description': arguments.get('description'),
                'images': arguments.get('images', []),
                'style': arguments.get('style'),
                'size': arguments.get('size'),
                'exact_size': arguments.get('exact_size'),
                'tile_size': arguments.get('tile_size'),
                'grid': arguments.get('grid'),
                'resize_filter': arguments.get('resize_filter', 'lanczos'),
                'path_to_save': arguments.get('path_to_save')
            }
            return image_operation_internal(image_args, conversation_messages)
            
        elif op == "image.slice_spritesheet":
            # Route to existing spritesheet slicing function
            slice_args = {
                'sheet_base64': arguments.get('sheet_base64'),
                'sheet_path': arguments.get('sheet_path'),
                'tile_size': arguments.get('tile_size'),
                'grid': arguments.get('grid'),
                'margin': arguments.get('margin', 0),
                'spacing': arguments.get('spacing', 0),
                'auto_detect': arguments.get('auto_detect', True),
                'bg_tolerance': arguments.get('bg_tolerance', 24),
                'alpha_threshold': arguments.get('alpha_threshold', 1),
                'tight_crop': arguments.get('tight_crop', True),
                'padding': arguments.get('padding', 0),
                'fuzzy': arguments.get('fuzzy', 2),
                'normalize_to': arguments.get('normalize_to')
            }
            return slice_spritesheet_internal(slice_args)
            
        
        # All resource operations should be handled by frontend (direct Godot engine access needed)
        elif op in ["res.create", "res.inspect", "res.modify", "res.assign", "res.copy_from_template", 
                   "res.refresh", "res.load_and_assign", "import.set_options", "import.reimport", "image.save"]:
            # These are frontend-only operations
            return {
                "success": False,
                "frontend_only": True,
                "message": f"Operation '{op}' is handled by the frontend. This should not be executed on the backend.",
                "operation": op,
                "arguments_to_forward": arguments
            }
            
        else:
            return {"success": False, "error": f"Unknown resource_manager operation: {op}"}
            
    except Exception as e:
        print(f"RESOURCE_MANAGER_ERROR: {e}")
        return {"success": False, "error": f"Resource manager operation failed: {str(e)}"}

def scene_manager_internal(arguments: dict) -> dict:
    """Handle scene_manager tool operations - all frontend-only"""
    # CRITICAL: Check for tool generation failures first
    if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
        failure_type = arguments.get("_tool_gen_failure")
        original_size = arguments.get("_original_size", "unknown")
        
        error_msg = (
            f"Tool call failed due to large content ({original_size} characters). "
            "This is likely because the content was very big, make the content smaller and apply fewer edits."
        )
        
        print(f"🚨 SCENE_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
        return {"success": False, "error": error_msg}
    
    op = arguments.get('op', '')
    return {
        "success": False,
        "frontend_only": True,
        "message": f"All scene_manager operations are handled by the frontend. Operation '{op}' should not be executed on the backend.",
        "operation": op,
        "arguments_to_forward": arguments
    }

def script_manager_internal(arguments: dict) -> dict:
    """Handle script_manager tool operations - all frontend-only"""
    # CRITICAL: Check for tool generation failures first
    if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
        failure_type = arguments.get("_tool_gen_failure")
        original_size = arguments.get("_original_size", "unknown")
        
        error_msg = (
            f"Tool call failed due to large content ({original_size} characters). "
            "This is likely because the content was very big, make the content smaller and apply fewer edits."
        )
        
        print(f"🚨 SCRIPT_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
        return {"success": False, "error": error_msg}
    
    op = arguments.get('op', '')
    return {
        "success": False,
        "frontend_only": True,
        "message": f"All script_manager operations are handled by the frontend. Operation '{op}' should not be executed on the backend.",
        "operation": op,
        "arguments_to_forward": arguments
    }

def settings_manager_internal(arguments: dict) -> dict:
    """Handle settings_manager tool operations - all frontend-only"""
    # CRITICAL: Check for tool generation failures first
    if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
        failure_type = arguments.get("_tool_gen_failure")
        original_size = arguments.get("_original_size", "unknown")
        
        error_msg = (
            f"Tool call failed due to large content ({original_size} characters). "
            "This is likely because the content was very big, make the content smaller and apply fewer edits."
        )
        
        print(f"🚨 SETTINGS_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
        return {"success": False, "error": error_msg}
    
    op = arguments.get('op', '')
    return {
        "success": False,
        "frontend_only": True,
        "message": f"All settings_manager operations are handled by the frontend. Operation '{op}' should not be executed on the backend.",
        "operation": op,
        "arguments_to_forward": arguments
    }

def runtime_manager_internal(arguments: dict) -> dict:
    """Handle runtime_manager tool operations"""
    try:
        # CRITICAL: Check for tool generation failures first
        if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
            failure_type = arguments.get("_tool_gen_failure")
            original_size = arguments.get("_original_size", "unknown")
            
            error_msg = (
                f"Tool call failed due to large content ({original_size} characters). "
                "This is likely because the content was very big, make the content smaller and apply fewer edits."
            )
            
            print(f"🚨 RUNTIME_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
            return {"success": False, "error": error_msg}
        
        op = arguments.get('op', '')
        if not op:
            return {"success": False, "error": "Operation 'op' parameter is required"}
            
        if op in ["game.start", "game.stop", "game.status", "errors.summary", "errors.details", "screenshot.capture", "console.get_output", "input.test_action", "input.test_key"]:
            # These are frontend-only operations
            return {
                "success": False,
                "frontend_only": True,
                "message": f"Operation '{op}' is handled by the frontend. This should not be executed on the backend.",
                "operation": op,
                "arguments_to_forward": arguments
            }
            
        else:
            return {"success": False, "error": f"Unknown runtime_manager operation: {op}"}
            
    except Exception as e:
        print(f"RUNTIME_MANAGER_ERROR: {e}")
        return {"success": False, "error": f"Runtime manager operation failed: {str(e)}"}

def runtime_inspector_internal(arguments: dict) -> dict:
    """Handle runtime inspection operations for debugging during play"""
    try:
        # CRITICAL: Check for tool generation failures first
        if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
            failure_type = arguments.get("_tool_gen_failure")
            original_size = arguments.get("_original_size", "unknown")
            
            error_msg = (
                f"Tool call failed due to large content ({original_size} characters). "
                "This is likely because the content was very big, make the content smaller and apply fewer edits."
            )
            
            print(f"🚨 INSPECTOR_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
            return {"success": False, "error": error_msg}
        
        op = arguments.get('op', '')
        if not op:
            return {"success": False, "error": "Operation 'op' parameter is required"}
            
        # All runtime inspector operations are frontend-only but we add metadata
        # to help the frontend know what to do
        return {
            "success": False,
            "frontend_only": True,
            "message": f"Runtime inspection operation '{op}' is handled by the frontend",
            "operation": op,
            "arguments_to_forward": arguments,
            "requires_game_running": True  # Signal that game must be running
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def terminal_manager_internal(arguments: dict) -> dict:
    """Handle terminal_manager tool operations - all frontend-only"""
    try:
        # CRITICAL: Check for tool generation failures first
        if isinstance(arguments, dict) and arguments.get("_tool_gen_failure"):
            failure_type = arguments.get("_tool_gen_failure")
            original_size = arguments.get("_original_size", "unknown")
            
            error_msg = (
                f"Tool call failed due to large content ({original_size} characters). "
                "This is likely because the content was very big, make the content smaller and apply fewer edits."
            )
            
            print(f"🚨 TERMINAL_TOOL_GEN_FAILURE: {failure_type} with size {original_size}")
            return {"success": False, "error": error_msg}
        
        op = arguments.get('op', '')
        if not op:
            return {"success": False, "error": "Operation 'op' parameter is required"}
            
        # All terminal operations are frontend-only (need local machine access)
        return {
            "success": False,
            "frontend_only": True,
            "message": f"Terminal operation '{op}' is handled by the frontend with local machine CLI access",
            "operation": op,
            "arguments_to_forward": arguments,
            "requires_local_machine": True,  # Signal that this needs local terminal access
            "context": "generic_cli"  # Clear context for AI understanding
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def _resolve_project_root_argument(arguments: dict) -> Optional[str]:
    project_root = arguments.get("project_root")
    if project_root:
        return project_root
    try:
        return getattr(g, "project_root", None)
    except RuntimeError:
        return None

def todo_manager_internal(arguments: dict, current_user: Optional[dict] = None) -> dict:
    """Lightweight todo management shared by agent + UI."""
    op = arguments.get("op")
    if not op:
        return {"success": False, "error": "Missing 'op' parameter"}

    project_root = _resolve_project_root_argument(arguments)
    created_by = (current_user or {}).get("id") or "agent"

    if op == "todo.list":
        todos = TodoStore.list(project_root)
        return {"success": True, "todos": todos, "project_root": project_root}

    if op == "todo.add":
        content = (arguments.get("content") or "").strip()
        if not content:
            return {"success": False, "error": "content parameter required"}
        status = arguments.get("status", "pending")
        todo = TodoStore.add(project_root, content, status=status, created_by=created_by)
        return {"success": True, "todo": todo, "project_root": project_root}

    if op == "todo.add_batch":
        items = arguments.get("items") or []
        if not isinstance(items, list):
            return {"success": False, "error": "items must be an array"}
        created = TodoStore.add_batch(project_root, items, created_by=created_by)
        return {"success": True, "todos": created, "project_root": project_root}

    if op == "todo.update":
        todo_id = arguments.get("todo_id")
        if not todo_id:
            return {"success": False, "error": "todo_id parameter required"}
        updated = TodoStore.update(
            project_root,
            todo_id,
            content=arguments.get("content"),
            status=arguments.get("status"),
        )
        if not updated:
            return {"success": False, "error": f"Todo '{todo_id}' not found"}
        return {"success": True, "todo": updated, "project_root": project_root}

    if op == "todo.remove":
        todo_id = arguments.get("todo_id")
        if not todo_id:
            return {"success": False, "error": "todo_id parameter required"}
        removed = TodoStore.remove(project_root, todo_id)
        return {"success": removed, "removed": removed, "project_root": project_root}

    if op == "todo.clear":
        TodoStore.clear(project_root)
        return {"success": True, "project_root": project_root}

    return {"success": False, "error": f"Unsupported todo_manager op: {op}"}

def capture_screenshot_internal(arguments: dict) -> dict:
    """Capture screenshot from editor or running game"""
    return {
        "success": False,
        "frontend_only": True,
        "message": "Screenshot capture is handled by the frontend",
        "operation": "screenshot.capture",
        "arguments_to_forward": arguments
    }

def execute_godot_tool(function_name: str, arguments: dict) -> dict:
    """Execute backend-specific tools"""
    
    # ============ CLEAR TOOL CALL LOGGING ============
    print("=" * 80)
    print(f"🔧 TOOL CALLED: {function_name}")
    print("📋 ARGUMENTS:")
    for key, value in arguments.items():
        # Truncate very long values for readability
        if isinstance(value, str) and len(value) > 200:
            display_value = value[:200] + "... (truncated)"
        else:
            display_value = value
        print(f"   {key}: {display_value}")
    print("=" * 80)
    # ===============================================
    
    # New consolidated tools
    if function_name == "project_manager":
        return project_manager_internal(arguments)
    elif function_name == "scene_manager":
        return scene_manager_internal(arguments)
    elif function_name == "script_manager":
        return script_manager_internal(arguments)
    elif function_name == "resource_manager":
        return resource_manager_internal(arguments)
    elif function_name == "settings_manager":
        return settings_manager_internal(arguments)
    elif function_name == "search_manager":
        return search_manager_internal(arguments, None)
    elif function_name == "graph_manager":
        return graph_manager_internal(arguments, None)
    elif function_name == "runtime_manager":
        return runtime_manager_internal(arguments)
    elif function_name == "runtime_inspector":
        return runtime_inspector_internal(arguments)
    elif function_name == "terminal_manager":
        return terminal_manager_internal(arguments)
    elif function_name == "todo_manager":
        return todo_manager_internal(arguments, None)
    # Legacy individual tools (maintain backward compatibility)
    elif function_name == "image_operation":
        return image_operation_internal(arguments)
    elif function_name == "asset_processor":
        return process_asset_internal(arguments)
    elif function_name == "search_across_project":
        return search_across_project_internal(arguments, None)
    elif function_name == "search_across_godot_docs":
        return search_across_godot_docs_internal(arguments)
    elif function_name == "search_godot_assets":
        return search_godot_assets_internal(arguments)
    elif function_name == "install_godot_asset":
        return install_godot_asset_internal(arguments)
    elif function_name == "generate_3d_model":
        return generate_3d_model_internal(arguments)
    elif function_name == "check_for_app_updates":
        return check_for_app_updates_internal(arguments)
    # Note: Game testing tools (start_game, stop_game, etc.) are frontend-only and not executed in backend
    # Note: get_project_context is also frontend-only and provides structure data directly
    else:
        # This shouldn't happen if we filter correctly
        print(f"WARNING: Unknown backend tool called: {function_name}")

    return {"success": False, "error": f"Unknown backend tool called: {function_name}"}



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
            # Idempotent behavior: treat missing as already stopped/completed
            print(f"STOP_REQUEST: Request {request_id} not found; treating as already completed")
            return jsonify({"success": True, "message": "Already completed"}), 200

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

@app.route('/todo_manager', methods=['POST'])
def todo_manager_endpoint():
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate

    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON payload"}), 400

    prj_hdr = request.headers.get('X-Project-Root')
    if prj_hdr:
        g.project_root = prj_hdr

    result = todo_manager_internal(data, user)
    status = 200 if result.get("success", False) else 400
    return jsonify(result), status

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
        has_existing_summary = data.get('has_existing_summary', False)
        summary_message_index = data.get('summary_message_index', -1)
        recent_messages_to_keep = data.get('recent_messages_to_keep', 20)
        
        if not messages:
            return jsonify({"success": False, "error": "No messages provided"}), 400
            
        user_id = user.get('id', 'unknown') if user else 'unknown'
        
        print(f"SUMMARIZATION: Processing {len(messages)} messages, existing_summary={has_existing_summary}, keep_recent={recent_messages_to_keep}")
        
        # Create summary using AI models with enhanced context
        import asyncio
        summary = asyncio.run(conversation_memory.summarize_conversation_chunk(
            messages, user_id, 
            has_existing_summary=has_existing_summary,
            is_incremental=has_existing_summary
        ))
        
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
    # DEBUGGING: Log chat requests  
    from datetime import datetime
    print(f"🔥 CHAT_DEBUG: /chat endpoint hit - {datetime.now()}")
    
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
    
    # Check Autumn quota BEFORE processing request (don't track yet - track after success)
    allowed, pricing_info = pricing_service.check_quota(user['id'], "ai-requests", 1)
    if not allowed:
        return jsonify({
            "error": "Monthly request limit exceeded",
            "pricing_info": pricing_info,
            "upgrade_url": f"{request.host_url}pricing",
            "success": False
        }), 429
    
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
    context = data.get('context') or {}
    requested_model = data.get('model')
    chat_mode = str((data.get('mode') or 'agent')).lower().strip()
    print(f"CHAT_MODE_DEBUG: Raw mode from request: {repr(data.get('mode'))}, parsed chat_mode: '{chat_mode}'")
    model = get_validated_chat_model(requested_model)  # Restrict to allowed models
    
    # CRITICAL FIX: Preserve original friendly name for thinking mode detection
    # instead of doing unreliable reverse lookup
    model_friendly_name = requested_model if requested_model in ALLOWED_CHAT_MODELS else DEFAULT_MODEL
    print(f"MODEL_SELECTION: Frontend requested '{requested_model}' -> using friendly_name '{model_friendly_name}', thinking_mode: {_is_thinking_mode(model_friendly_name)}; chat_mode={chat_mode}")

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
            try:
                # Lightweight chat_start event (no content)
                log_event('chat_start', {
                    'model': model_friendly_name,
                    'messages_count': len(messages) if isinstance(messages, list) else None,
                })
            except Exception:
                pass
            
            # Filter out any None or invalid messages from the start
            conversation_messages = []
            # Track tool usage and simple per-request caching to avoid infinite loops
            tool_call_counts: dict[str, int] = {}
            tool_result_cache: dict[str, dict] = {}
            # Track recovery attempts to prevent infinite recovery loops
            recovery_attempts = 0
            max_recovery_attempts = 2
            
            # CRITICAL FIX: Track conversation state for better debugging
            total_user_messages = 0
            total_assistant_messages = 0
            total_tool_messages = 0
            
            # Log tool results that come from frontend
            for msg in messages:
                if msg is not None and isinstance(msg, dict) and msg.get('role'):
                    conversation_messages.append(msg)
                    
                    # Track message types for debugging
                    role = msg.get('role', '')
                    if role == 'user':
                        total_user_messages += 1
                    elif role == 'assistant':
                        total_assistant_messages += 1
                    elif role == 'tool':
                        total_tool_messages += 1
                    
                    # Log tool results when they come back from frontend
                    if msg.get('role') == 'tool':
                        try:
                            tool_name = msg.get('name', 'unknown_tool')
                            tool_call_id = msg.get('tool_call_id', 'unknown_id')
                            content = msg.get('content', '{}')
                            
                            
                            # CRITICAL FIX: Validate tool call ID presence
                            if not tool_call_id or tool_call_id == 'unknown_id':
                                print(f"TOOL_RESULT_ERROR: Missing tool_call_id for {tool_name}")
                            
                            # Parse tool result
                            try:
                                result = json.loads(content) if isinstance(content, str) else content
                            except Exception:
                                result = {'content': str(content)}
                            
                            log_tool_result(tool_name, tool_call_id, result, duration_ms=0)
                        except Exception as e:
                            print(f"⚠️  Error logging frontend tool result: {e}")
                else:
                    pass
                # Check for stop even during initial processing
                if check_stop():
                    print(f"STOP_DETECTED: Request {request_id} stopped during message filtering")
                    yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                    return

            # CRITICAL DEBUG: Log conversation state before processing
            print(f"CONVERSATION_INIT: Starting conversation with {len(conversation_messages)} messages")
            print(f"CONVERSATION_BREAKDOWN: {total_user_messages} user, {total_assistant_messages} assistant, {total_tool_messages} tool messages")
            
            # Validate conversation structure for tool call consistency
            unmatched_tool_calls = []
            for i, msg in enumerate(conversation_messages):
                if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                    for tool_call in msg['tool_calls']:
                        tc_id = tool_call.get('id', '')
                        if tc_id:
                            # Look for matching tool response
                            found_response = False
                            for j in range(i + 1, len(conversation_messages)):
                                if (conversation_messages[j].get('role') == 'tool' and 
                                    conversation_messages[j].get('tool_call_id') == tc_id):
                                    found_response = True
                                    break
                            if not found_response:
                                unmatched_tool_calls.append(tc_id)
            
            if unmatched_tool_calls:
                print(f"CONVERSATION_WARNING: Found {len(unmatched_tool_calls)} unmatched tool calls: {unmatched_tool_calls}")
            else:
                print(f"CONVERSATION_VALID: All tool calls have matching responses")

            # Log incoming headers for project root troubleshooting
            try:
                prj_hdr = request.headers.get('X-Project-Root')
                print(f"CHAT_HEADERS: X-Project-Root={prj_hdr} X-User-ID={request.headers.get('X-User-ID')} X-Machine-ID={request.headers.get('X-Machine-ID')}")
                
                # Store project root in Flask g for access by internal functions
                if prj_hdr:
                    g.project_root = prj_hdr
            except Exception:
                pass

            # Attach editor-provided context as a high-signal system message early
            if isinstance(context, dict) and context:
                try:
                    # Keep it compact and explicit; avoid leaking huge blobs
                    minimized = {
                        'type': 'godot_editor_context',
                        'project_root': context.get('project_root'),
                        'current_file': context.get('current_file'),
                        'cursor': context.get('cursor'),
                        'open_files': context.get('open_files'),
                        'selected_nodes': context.get('selected_nodes'),
                        'selected_text': context.get('selected_text')[:4000] if isinstance(context.get('selected_text'), str) else None,
                    }

                    # Include optional project structure fields if present (frontend may include these)
                    try:
                        def _copy_if_present(key, limiter=None):
                            if key in context and context.get(key) is not None:
                                val = context.get(key)
                                if limiter and isinstance(val, list):
                                    minimized[key] = val[:limiter]
                                else:
                                    minimized[key] = val

                        # Basic metadata
                        _copy_if_present('project_name')

                        # Summaries (cap list sizes just in case)
                        _copy_if_present('scenes', limiter=50)
                        _copy_if_present('scenes_count')
                        _copy_if_present('scripts', limiter=50)
                        _copy_if_present('scripts_count')
                        _copy_if_present('folders', limiter=100)
                        _copy_if_present('folders_count')

                        # Optional configuration blocks
                        _copy_if_present('autoloads')
                        _copy_if_present('input_actions')
                    except Exception as ie:
                        print(f"CONTEXT_MINIMIZE_WARN: Failed copying project structure fields: {ie}")

                    # Prepend as a system message for routing, before the main system prompt
                    conversation_messages = [{
                        'role': 'system',
                        'content': json.dumps(minimized, ensure_ascii=False)
                    }] + conversation_messages
                except Exception as e:
                    print(f"CONTEXT_ATTACH_WARN: Failed to attach context: {e}")

            # Helper to preserve critical parameters when JSON parsing fails
            def _try_preserve_critical_params(corrupted_json: str) -> str:
                """
                Try to extract critical parameters from corrupted JSON using regex patterns.
                Focus on preserving 'op', 'query', 'path', and other essential parameters.
                """
                import re
                try:
                    preserved = {}
                    
                    # Try to extract 'op' parameter (most critical for tool routing)
                    op_match = re.search(r'"op"\s*:\s*"([^"]+)"', corrupted_json, re.IGNORECASE)
                    if op_match:
                        preserved["op"] = op_match.group(1)
                    
                    # Try to extract 'query' parameter (critical for search tools)
                    query_match = re.search(r'"query"\s*:\s*"([^"]+)"', corrupted_json, re.IGNORECASE)
                    if query_match:
                        preserved["query"] = query_match.group(1)
                    
                    # Try to extract 'path' parameter (critical for file operations)
                    path_match = re.search(r'"path"\s*:\s*"([^"]+)"', corrupted_json, re.IGNORECASE)
                    if path_match:
                        preserved["path"] = path_match.group(1)
                    
                    # Try to extract other common parameters (but avoid large content fields)
                    for param in ["scene_path", "node_path", "type", "name", "operation", "description", "asset_query"]:
                        match = re.search(rf'"{param}"\s*:\s*"([^"]+)"', corrupted_json, re.IGNORECASE)
                        if match:
                            value = match.group(1)
                            # Truncate very long values to prevent re-corruption
                            if len(value) > 1000:
                                print(f"PARAM_RECOVERY: Truncating large '{param}' value from {len(value)} to 1000 chars")
                                value = value[:1000] + "... [truncated due to size]"
                            preserved[param] = value
                    
                    # CRITICAL: If we found an 'op' parameter but no other params, try to infer missing ones
                    if "op" in preserved and len(preserved) == 1:
                        op_value = preserved["op"]
                        
                        # For filesystem operations, try to extract path from the large content
                        if op_value.startswith("fs."):
                            # Look for common path patterns in the corrupted content
                            path_patterns = [
                                r'res://[^"\s]+\.(?:gd|tscn|cs|tres|png|jpg|wav)',  # Godot resource paths
                                r'"(?:res://)?[^"\s]*?\.(?:gd|tscn|cs|tres|png|jpg|wav)"'  # Quoted paths
                            ]
                            for pattern in path_patterns:
                                match = re.search(pattern, corrupted_json)
                                if match:
                                    preserved["path"] = match.group(0).strip('"')
                                    print(f"PARAM_RECOVERY: Inferred path '{preserved['path']}' for {op_value} operation")
                                    break
                    
                    if preserved:
                        import json as _json
                        result = _json.dumps(preserved, separators=(",", ":"))
                        print(f"PARAM_RECOVERY: Successfully preserved {len(preserved)} parameters: {list(preserved.keys())}")
                        return result
                    
                    return "{}"
                except Exception as e:
                    print(f"PARAM_RECOVERY_ERROR: Failed to preserve parameters: {e}")
                    return "{}"

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
                    
                    # ENHANCED: Handle large content that corrupts JSON
                    # If the string is very large, it likely contains content that's breaking JSON
                    if len(s) > 10000:  # 10KB threshold
                        print(f"TOOL_ARGS_WARNING: Very large tool arguments ({len(s)} chars) - attempting robust parsing")
                        
                        # ROBUST APPROACH: Try parsing as-is first, without regex extraction
                        try:
                            # Claude usually generates valid JSON even for large content
                            obj = _json.loads(s)
                            print(f"TOOL_ARGS_SUCCESS: Parsed large arguments successfully")
                            return _json.dumps(obj, separators=(",", ":"))
                        except Exception as e:
                            print(f"TOOL_ARGS_ERROR: Direct parsing failed: {e}")
                            # Try to fix common JSON issues in large content
                            try:
                                # Remove any trailing commas that break JSON
                                s_fixed = _re.sub(r',\s*}', '}', s)
                                s_fixed = _re.sub(r',\s*]', ']', s_fixed)
                                obj = _json.loads(s_fixed)
                                print(f"TOOL_ARGS_SUCCESS: Parsed after fixing trailing commas")
                                return _json.dumps(obj, separators=(",", ":"))
                            except Exception as e2:
                                print(f"TOOL_ARGS_ERROR: Failed even after comma fix: {e2}")
                                # CRITICAL: Try to preserve at least the 'op' parameter before complete fallback
                                preserved_args = _try_preserve_critical_params(s)
                                if preserved_args != "{}":
                                    print(f"TOOL_ARGS_RECOVERY: Preserved critical parameters: {preserved_args}")
                                    return preserved_args
                                # Last resort: Log the problematic section for debugging
                                error_context = s[max(0, 110):min(len(s), 150)]
                                print(f"TOOL_ARGS_DEBUG: Context around error position 116: ...{error_context}...")
                                # Fall through to other parsing attempts below
                    
                    # Original parsing logic for normal-sized content
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
                        # CRITICAL: Try to preserve essential parameters before complete fallback
                        preserved_args = _try_preserve_critical_params(s)
                        if preserved_args != "{}":
                            print(f"TOOL_ARGS_RECOVERY: Preserved parameters from final parsing attempt")
                            return preserved_args
                        # Mark this as a large content failure for special error handling
                        return '{"_tool_gen_failure": "large_content", "_original_size": ' + str(len(s)) + '}'
                except Exception as e:
                    print(f"TOOL_ARGS_CRITICAL_ERROR: Complete sanitization failure: {e}")
                    # Even in complete failure, try to preserve critical params if we have the original string
                    if isinstance(arguments_value, str) and len(arguments_value) > 0:
                        preserved_args = _try_preserve_critical_params(arguments_value)
                        if preserved_args != "{}":
                            print(f"TOOL_ARGS_RECOVERY: Preserved parameters from exception handler")
                            return preserved_args
                        # Mark as tool generation failure
                        size = len(str(arguments_value))
                        return f'{{"_tool_gen_failure": "critical_error", "_original_size": {size}}}'
                    return '{"_tool_gen_failure": "unknown"}'

            conversation_turn = 0
            max_conversation_turns = 10  # Prevent infinite loops
            
            while True:  # Loop to handle tool calling and responses
                conversation_turn += 1
                print(f"CONVERSATION_TURN: Starting turn {conversation_turn}/{max_conversation_turns}")
                
                # Prevent infinite conversation loops
                if conversation_turn > max_conversation_turns:
                    print(f"CONVERSATION_TURN: Max turns ({max_conversation_turns}) reached, ending conversation")
                    yield json.dumps({"status": "completed", "message": "Maximum conversation turns reached"}) + '\n'
                    return
                
                # Check for stop before each major operation
                if check_stop():
                    print(f"STOP_DETECTED: Request {request_id} stopped before OpenAI call")
                    yield json.dumps({"status": "stopped", "message": "Request stopped"}) + '\n'
                    return
                
                # CRITICAL: Always use fallback for production-grade incremental summarization
                # The conversation_memory system is experimental, fallback is proven
                original_message_count = len(conversation_messages)
                
                # Check if we WILL summarize (before actually doing it)
                total_tokens = _count_tokens_for_messages(conversation_messages, model)
                # Use global universal trigger threshold
                
                will_summarize = total_tokens > UNIVERSAL_SUMMARIZATION_TRIGGER
                
                if will_summarize:
                    # SEND STATUS BEFORE SUMMARIZATION STARTS
                    yield json.dumps({
                        "status": "summarizing_starting",
                        "message": f"Starting summarization of {len(conversation_messages)} messages...",
                        "original_count": len(conversation_messages)
                    }) + '\n'
                    print(f"📤 SENT: summarizing_starting status to frontend BEFORE blocking")
                
                # ALWAYS use fallback for now (it's more reliable and tested)
                # Track if summarization happened by checking return value
                original_message_count = len(conversation_messages)
                conversation_messages, summarization_was_attempted = _manage_conversation_length_fallback(conversation_messages, model)
                
                # SEND COMPLETION STATUS AFTER SUMMARIZATION WITH SUMMARIZED MESSAGES
                if will_summarize and summarization_was_attempted:
                    yield json.dumps({
                        "status": "summarizing",
                        "message": f"Summarization completed: {original_message_count} → {len(conversation_messages)} messages",
                        "original_count": original_message_count,
                        "new_count": len(conversation_messages),
                        "action": "replace_conversation_history", 
                        "new_messages": conversation_messages  # Always send summarized conversation for frontend storage
                    }) + '\n'
                    print(f"📤 SENT: summarizing completed status with {len(conversation_messages)} summarized messages to frontend ({original_message_count} → {len(conversation_messages)} messages)")
                
                # SAFETY CHECK: Verify we're under limit using ACTUAL token counting
                model_limit = _get_model_token_limit(model)
                post_summary_tokens = _count_tokens_for_messages(conversation_messages, model)
                percent_of_limit = (post_summary_tokens / model_limit * 100)
                
                print(f"🔍 SAFETY_CHECK: Post-summary token count: {post_summary_tokens} ({percent_of_limit:.1f}% of {model_limit} limit)")
                
                if post_summary_tokens > model_limit * 0.9:  # Still over 90% after summarization!
                    print(f"⚠️ CONVERSATION_SAFETY: Post-summary still at {post_summary_tokens} tokens (>90% of limit)! Force trimming...")
                    # Emergency: Keep only system + last 20 messages
                    system_msgs = [m for m in conversation_messages if m.get('role') == 'system']
                    conversation_messages = system_msgs + conversation_messages[-20:]
                    final_tokens = _count_tokens_for_messages(conversation_messages, model)
                    print(f"CONVERSATION_SAFETY: Emergency trim to {len(conversation_messages)} messages ({final_tokens} tokens, {(final_tokens/model_limit*100):.1f}% of limit)")
                
                # CRITICAL: Update frontend conversation history with summary
                # ALWAYS notify if summarization was attempted (tracked above)
                if summarization_was_attempted:
                    messages_removed = max(0, original_message_count - len(conversation_messages))
                    
                    # REMOVED: Legacy summarization notification (conflicts with new streamlined approach)
                    # The new approach sends completion status after summarization finishes
                
                # Define recursive stripper function once for both counting and sending
                def _strip_heavy_fields_recursive(value):
                    try:
                        import re as _re
                        preserve_keys = {"inline_diff", "diff", "original_content", "edited_content"}
                        if isinstance(value, dict):
                            for k in list(value.keys()):
                                if isinstance(k, str):
                                    lk = k.lower()
                                    if k not in preserve_keys and (
                                        lk == "image_data" or lk == "base64" or lk == "data_uri" or
                                        "base64" in lk or lk.endswith("_data") or lk.endswith("_bytes") or lk.endswith("_b64")
                                    ):
                                        # Try to downscale image data instead of removing completely
                                        field_value = value[k]
                                        if isinstance(field_value, str) and len(field_value) > 50000:
                                            # Looks like large base64 image data
                                            try:
                                                import base64
                                                from PIL import Image
                                                import io
                                                
                                                # Try to decode as base64 image
                                                img_bytes = base64.b64decode(field_value)
                                                img = Image.open(io.BytesIO(img_bytes))
                                                
                                                # Downscale to max 256px for AI  
                                                MAX_AI_SIZE = 256
                                                if img.width > MAX_AI_SIZE or img.height > MAX_AI_SIZE:
                                                    aspect = img.width / img.height
                                                    if img.width > img.height:
                                                        new_size = (MAX_AI_SIZE, int(MAX_AI_SIZE / aspect))
                                                    else:
                                                        new_size = (int(MAX_AI_SIZE * aspect), MAX_AI_SIZE)
                                                    
                                                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                                                    
                                                    # Re-encode to base64
                                                    buffer = io.BytesIO()
                                                    img.save(buffer, format='PNG')
                                                    small_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                                    
                                                    print(f"AI_FIELD_DOWNSCALE: {k} {len(field_value)} -> {len(small_b64)} chars ({img.width}x{img.height})")
                                                    value[k] = small_b64
                                                    continue
                                                    
                                                # Keep original if already small enough
                                                continue
                                            except Exception:
                                                # Not valid image data, remove the field
                                                value.pop(k, None)
                                                continue
                                        else:
                                            # Remove non-image or small data fields
                                            value.pop(k, None)
                                            continue
                                value[k] = _strip_heavy_fields_recursive(value[k])
                            return value
                        elif isinstance(value, list):
                            for i in range(len(value)):
                                value[i] = _strip_heavy_fields_recursive(value[i])
                            return value
                        elif isinstance(value, str):
                            s = value
                            if s.startswith("data:image/") and ";base64," in s:
                                # For AI backend: downscale large images instead of stripping completely
                                try:
                                    # Extract base64 data from data URI
                                    header, b64_data = s.split(',', 1)
                                    if len(b64_data) > 50000:  # > ~37KB base64 = large image
                                        import base64
                                        from PIL import Image
                                        import io
                                        
                                        # Decode, downscale, re-encode
                                        img_bytes = base64.b64decode(b64_data)
                                        img = Image.open(io.BytesIO(img_bytes))
                                        
                                        # Downscale to max 256px for AI (better than 128px for analysis)
                                        MAX_AI_SIZE = 256
                                        if img.width > MAX_AI_SIZE or img.height > MAX_AI_SIZE:
                                            aspect = img.width / img.height
                                            if img.width > img.height:
                                                new_size = (MAX_AI_SIZE, int(MAX_AI_SIZE / aspect))
                                            else:
                                                new_size = (int(MAX_AI_SIZE * aspect), MAX_AI_SIZE)
                                            
                                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                                            
                                            # Re-encode to base64
                                            buffer = io.BytesIO()
                                            img.save(buffer, format='PNG')
                                            small_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                            
                                            print(f"AI_IMAGE_DOWNSCALE: {len(b64_data)} -> {len(small_b64)} chars ({img.width}x{img.height})")
                                            return f"{header},{small_b64}"
                                        
                                        return s  # Keep original if already small enough
                                except Exception as e:
                                    print(f"AI_IMAGE_DOWNSCALE_ERROR: {e}")
                                    return "[IMAGE_PROCESSING_ERROR]"
                            elif len(s) > 100000 and _re.search(r"[A-Za-z0-9+/=]{1000,}", s):
                                return "[LARGE_DATA_STRIPPED]"
                            return s
                        else:
                            return value
                    except Exception:
                        return value

                # IMPORTANT: Recompute token count using sanitized messages to avoid base64/data_uri inflation
                import copy as _copy_for_count
                messages_for_count = []
                for m in conversation_messages:
                    if not isinstance(m, dict):
                        continue
                    m2 = _copy_for_count.deepcopy(m)
                    c = m2.get('content')
                    if isinstance(c, (dict, list, str)):
                        # IMPORTANT: operate on a deep-copied content to avoid mutating originals
                        c_copy = _copy_for_count.deepcopy(c)
                        m2['content'] = _strip_heavy_fields_recursive(c_copy)
                    messages_for_count.append(m2)

                print(f"CONVERSATION_LOOP: Starting OpenAI call with {len(messages_for_count)} messages")
                if conversation_messages:
                    last_msg = conversation_messages[-1]
                    if last_msg and isinstance(last_msg, dict):
                        last_role = last_msg.get('role', 'unknown')
                        print(f"CONVERSATION_LOOP: Last message: {last_role}")
                        
                        # CRITICAL DEBUG: Check if this is a continuation after frontend tools
                        if last_role == 'tool':
                            tool_name = last_msg.get('name', 'unknown')
                            tool_id = last_msg.get('tool_call_id', 'unknown')
                            print(f"CONVERSATION_CONTINUATION: Detected tool result continuation - {tool_name} (ID: {tool_id})")
                            print(f"CONVERSATION_CONTINUATION: This should trigger AI's final response")
                        elif last_role == 'assistant' and last_msg.get('tool_calls'):
                            num_tool_calls = len(last_msg.get('tool_calls', []))
                            print(f"CONVERSATION_CONTINUATION: Last message has {num_tool_calls} tool calls - expecting tool responses next")
                    else:
                        print(f"CONVERSATION_LOOP: Last message is invalid: {type(last_msg)}")
                    
                # Debug logs for OpenAI messages have been quieted to reduce console noise.
                
                # Clean messages for OpenAI with intelligent image management
                # (using the _strip_heavy_fields_recursive function defined above)
                openai_messages_send = []
                recent_images = []  # Track recent images for context
                prior_assistant_with_tools = False
                
                for i, msg in enumerate(conversation_messages):
                    if msg is None or not isinstance(msg, dict):
                        continue
                        
                    role = msg['role']
                    if role == 'tool' and not prior_assistant_with_tools:
                        continue
                    
                    clean_msg = {
                        'role': role,
                        'content': None
                    }
                    # Deep copy content to avoid in-place mutation of original conversation payload
                    try:
                        import copy as _copy_content
                        content_src = msg.get('content')
                        if isinstance(content_src, (dict, list)):
                            clean_msg['content'] = _copy_content.deepcopy(content_src)
                        else:
                            clean_msg['content'] = content_src
                    except Exception:
                        clean_msg['content'] = msg.get('content')
                    
                    # CRITICAL FIX: Convert tool screenshot results to proper IMAGE format
                    if role == 'tool' and clean_msg['content']:
                        content = clean_msg['content']
                        if isinstance(content, str):
                            # Parse tool content JSON to extract image data
                            try:
                                import json as tool_json
                                tool_data = tool_json.loads(content)
                                
                                # Check if this is a screenshot tool result with image data
                                if isinstance(tool_data, dict):
                                    extracted_image = None
                                    
                                    # Extract image from top-level image_data
                                    if 'image_data' in tool_data and len(str(tool_data['image_data'])) > 1000:
                                        extracted_image = {
                                            'base64_data': tool_data['image_data'],
                                            'mime_type': tool_data.get('mime_type', 'image/png'),
                                            'name': tool_data.get('image_name', tool_data.get('image_id', 'screenshot'))
                                        }
                                        # Clean the tool content 
                                        tool_data['image_data'] = f"[CONVERTED TO IMAGE ATTACHMENT]"
                                        
                                    # Extract image from screenshots array
                                    elif 'screenshots' in tool_data and isinstance(tool_data['screenshots'], list):
                                        for screenshot in tool_data['screenshots']:
                                            if isinstance(screenshot, dict) and 'image_data' in screenshot:
                                                if len(str(screenshot['image_data'])) > 1000:
                                                    extracted_image = {
                                                        'base64_data': screenshot['image_data'],
                                                        'mime_type': screenshot.get('mime_type', 'image/png'),
                                                        'name': screenshot.get('prompt', 'screenshot').replace(' ', '_')
                                                    }
                                                    screenshot['image_data'] = f"[CONVERTED TO IMAGE ATTACHMENT]"
                                                    break
                                    
                                    # STRIP IMAGE DATA COMPLETELY - tool messages can't use vision format
                                    # The AI already received the image, no need to send it back
                                    if extracted_image:
                                        print(f"TOOL_IMAGE_STRIP: Removed {len(extracted_image['base64_data'])} chars of base64 for '{extracted_image['name']}'")
                                    
                                    # Always use clean JSON without base64
                                    clean_msg['content'] = tool_json.dumps(tool_data)
                                    
                            except Exception as e:
                                print(f"TOOL_CONTENT_PARSE_ERROR: {e}")
                                # Fallback: just strip without converting
                                if '"image_data":"' in content and len(content) > 10000:
                                    import re
                                    content = re.sub(r'"image_data":"[^"]{1000,}"', '"image_data":"[STRIPPED]"', content)
                                    clean_msg['content'] = content
                    
                    # Handle images intelligently - AGGRESSIVE filtering to prevent token explosion
                    if 'images' in msg and isinstance(msg['images'], list):
                        images = msg['images']
                        # Only include images from the LAST message and only if it's a user message
                        is_last_message = i == len(conversation_messages) - 1
                        
                        if is_last_message and role == 'user' and len(images) <= 1:
                            # Only send 1 most recent user image to avoid token explosion
                            content_array = []
                            
                            if clean_msg['content']:
                                content_array.append({
                                    "type": "text", 
                                    "text": clean_msg['content']
                                })
                            
                            # Add ONLY the first image - but NEVER include base64 to prevent context explosion
                            if len(images) > 0 and images[0].get('base64_data'):
                                img = images[0]
                                # CRITICAL FIX: Replace base64 with small placeholder to prevent 500k+ char explosion
                                content_array.append({
                                    "type": "text",
                                    "text": f"[Image: {img.get('name', 'attached_image')} - {img.get('mime_type', 'image/png')} - available for reference]"
                                })
                                recent_images.append(img.get('name', 'recent_image'))
                            
                            clean_msg['content'] = content_array
                        else:
                            # For ALL other cases, strip images completely and just reference
                            image_names = [img.get('name', 'image') for img in images[:3]]  # Max 3 names
                            if clean_msg['content']:
                                clean_msg['content'] += f"\n[Referenced images: {', '.join(image_names)}]"
                            else:
                                clean_msg['content'] = f"[Images: {', '.join(image_names)}]"
                    
                    # Handle tool calls
                    if 'tool_calls' in msg:
                        fixed_tool_calls = []
                        for tool_call in msg['tool_calls']:
                            if isinstance(tool_call, dict):
                                fixed_tool_call = tool_call.copy()
                                if 'type' not in fixed_tool_call:
                                    fixed_tool_call['type'] = 'function'
                                try:
                                    fn = fixed_tool_call.get('function') or {}
                                    if isinstance(fn, dict) and 'arguments' in fn:
                                        fn_args = fn.get('arguments')
                                        # First ensure valid JSON string
                                        sanitized_args = _sanitize_tool_arguments(fn_args)
                                        # Then recursively strip heavy/base64/data_uri fields inside arguments
                                        try:
                                            import json as _json
                                            parsed_args = _json.loads(sanitized_args)
                                            parsed_args = _strip_heavy_fields_recursive(parsed_args)
                                            fn['arguments'] = _json.dumps(parsed_args, separators=(",", ":"))
                                        except Exception:
                                            fn['arguments'] = sanitized_args
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
                    
                    # FINAL SAFETY: Apply downscaling to ALL message content including vision parts
                    content_to_clean = clean_msg.get('content')
                    if isinstance(content_to_clean, (dict, list, str)):
                        if role == 'user' and isinstance(content_to_clean, list):
                            # User messages with vision format - downscale image_url parts
                            cleaned_content = []
                            for part in content_to_clean:
                                if isinstance(part, dict) and part.get('type') == 'image_url':
                                    # CRITICAL: Apply downscaling to image_url data URIs
                                    image_url_dict = part.get('image_url', {})
                                    url = image_url_dict.get('url', '')
                                    
                                    if url.startswith('data:image/') and ';base64,' in url:
                                        # Apply the same downscaling logic
                                        downscaled_url = _strip_heavy_fields_recursive(url)
                                        if downscaled_url != url:
                                            print(f"USER_IMAGE_DOWNSCALE: Applied to user message image")
                                            part = dict(part)
                                            part['image_url'] = {'url': downscaled_url}
                                    
                                    cleaned_content.append(part)
                                else:
                                    cleaned_content.append(_strip_heavy_fields_recursive(part) if isinstance(part, (dict, list)) else part)
                            clean_msg['content'] = cleaned_content
                        elif role != 'user':
                            # Non-user messages: strip everything
                            clean_msg['content'] = _strip_heavy_fields_recursive(content_to_clean)
                        # For user string content, keep as-is (might be legitimate text with references)
                    openai_messages_send.append(clean_msg)

                # Build a sanitized copy for counting purposes (do NOT modify messages for sending)
                def _strip_images_for_count(value):
                    try:
                        if isinstance(value, dict):
                            out = {}
                            for k, v in value.items():
                                if k == 'type' and v == 'image_url':
                                    out[k] = v
                                    out['image_url'] = {'url': '[IMAGE_DATA_REDACTED]'}
                                elif isinstance(v, str) and v.startswith('data:image/') and ';base64,' in v:
                                    out[k] = '[DATA_URI_STRIPPED]'
                                else:
                                    out[k] = _strip_images_for_count(v)
                            return out
                        elif isinstance(value, list):
                            return [_strip_images_for_count(x) for x in value]
                        elif isinstance(value, str):
                            if value.startswith('data:image/') and ';base64,' in value:
                                return '[DATA_URI_STRIPPED]'
                            return value
                        else:
                            return value
                    except Exception:
                        return value

                openai_messages_count = []
                import copy as _copy
                for m in openai_messages_send:
                    m2 = _copy.deepcopy(m)
                    c = m2.get('content')
                    if isinstance(c, (dict, list, str)):
                        m2['content'] = _strip_images_for_count(c)
                    openai_messages_count.append(m2)

                # Prepend appropriate system prompt based on mode
                if chat_mode == 'ask' and SYSTEM_PROMPT_ASK:
                    system_msg = {"role": "system", "content": SYSTEM_PROMPT_ASK}
                    print(f"SYSTEM_PROMPT: Using ASK mode prompt ({len(SYSTEM_PROMPT_ASK)} chars)")
                elif SYSTEM_PROMPT:
                    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
                    print(f"SYSTEM_PROMPT: Using AGENT mode prompt ({len(SYSTEM_PROMPT)} chars)")
                else:
                    system_msg = None
                
                if system_msg:
                    openai_messages_send = [system_msg] + openai_messages_send
                    openai_messages_count = [system_msg] + openai_messages_count
                
                # VISION_NORMALIZE: Enforce OpenAI vision format and downscale data URIs before sending
                def _normalize_openai_vision(messages):
                    normalized = []
                    for msg in messages:
                        try:
                            m = dict(msg)
                            content = m.get('content')
                            if isinstance(content, list):
                                first_text = None
                                first_image = None
                                for part in content:
                                    if not isinstance(part, dict):
                                        continue
                                    t = part.get('type')
                                    if t == 'text' and first_text is None:
                                        first_text = {"type": "text", "text": str(part.get('text', ''))}
                                    elif t == 'image_url' and first_image is None:
                                        url = str(part.get('image_url', {}).get('url', ''))
                                        # Downscale large data URIs
                                        if url.startswith('data:image/') and ';base64,' in url:
                                            url = _strip_heavy_fields_recursive(url)
                                        first_image = {"type": "image_url", "image_url": {"url": url}}
                                new_content = []
                                if first_text:
                                    new_content.append(first_text)
                                if first_image:
                                    new_content.append(first_image)
                                if new_content:
                                    m['content'] = new_content
                            normalized.append(m)
                        except Exception:
                            normalized.append(msg)
                    return normalized

                print("VISION_NORMALIZE: Normalizing user messages to OpenAI image_url format")
                openai_messages_send = _normalize_openai_vision(openai_messages_send)
                
                # FINAL VALIDATION: Check message sizes before sending
                total_message_chars = sum(len(str(msg.get('content', ''))) for msg in openai_messages_send)
                if total_message_chars > 500000:
                    print(f"FINAL_CHECK: WARNING - Still have large message content ({total_message_chars} chars)")
                    # Try to find and log which message is causing the issue
                    for i, msg in enumerate(openai_messages_send):
                        msg_chars = len(str(msg.get('content', '')))
                        if msg_chars > 100000:
                            print(f"FINAL_CHECK: Message {i} ({msg.get('role', 'unknown')}) has {msg_chars} chars")
                else:
                    print(f"FINAL_CHECK: Message content size looks good ({total_message_chars} chars)")
                
                # Debug: Check total token usage with ACTUAL LiteLLM counting
                total_tokens = _count_tokens_for_messages(openai_messages_count, model)
                total_chars = sum(len(str(msg.get('content', ''))) for msg in openai_messages_count)
                model_limit = _get_model_token_limit(model)
                percent_used = (total_tokens / model_limit * 100) if model_limit > 0 else 0
                print(f"LITELLM_PREP: Sending {len(openai_messages_send)} messages to {model_friendly_name} ({model})")
                print(f"LITELLM_PREP: Token usage: {total_tokens} tokens ({percent_used:.1f}% of {model_limit} limit), {total_chars} chars")
                if total_tokens > 150000:
                    print(f"LITELLM_PREP: WARNING - High token count ({total_tokens} tokens), may hit limits!")
                if total_chars > 500000:
                    print(f"LITELLM_PREP: CRITICAL - Massive char count ({total_chars} chars), likely has unprocessed base64!")
                
                # REMOVED: Emergency pruning - let backend conversation management handle this gracefully
                if total_chars > 300000:
                    print(f"⚠️ LARGE_CONVERSATION: {len(openai_messages_send)} messages, {total_chars} chars")
                    print("ℹ️ Backend conversation management will handle via intelligent summarization")
                
                # Resilient model call with 5 retries (1 second each) then fallback to a preferred OpenAI model
                attempts = 0
                max_attempts = 5  # 5 retry attempts as requested
                providers_tried = set()
                model_try = model
                
                # We need to retry the ENTIRE streaming process, not just the initial call
                while True:
                    try:
                        # Get reasoning parameters based on model name and provider id
                        # Use preserved friendly name instead of unreliable reverse lookup
                        reasoning_params = _get_reasoning_params(model_friendly_name, model_try)
                        
                        # SIMPLE APPROACH: Different tools array for ask vs agent mode
                        if chat_mode == 'ask':
                            # ASK MODE: Ultra-minimal read-only tools only
                            safe_tools = [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "project_manager", 
                                        "description": "Read-only project operations: analyze structure, list directories, read files, search assets",
                                        "parameters": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "op": {
                                                    "type": "string",
                                                    "enum": ["context.get", "fs.list", "fs.read", "assets.search", "project.analyze_dir"],
                                                    "description": "Read-only operations only"
                                                },
                                                "path": {"type": "string"},
                                                "dir": {"type": "string"}, 
                                                "project_root": {"type": "string"},
                                                "context_mode": {"type": "string", "enum": ["structure", "hierarchy", "find_scenes", "patterns"]},
                                                "pattern": {"type": "string"},
                                                "asset_query": {"type": "string"},
                                                "max_results": {"type": "integer", "default": 10}
                                            },
                                            "required": ["op"]
                                        }
                                    }
                                },
                                {
                                    "type": "function", 
                                    "function": {
                                        "name": "search_manager",
                                        "description": "Search project files and Godot documentation",
                                        "parameters": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "op": {"type": "string", "enum": ["project.search", "docs.search"]},
                                                "query": {"type": "string"},
                                                "max_results": {"type": "integer", "default": 5},
                                                "search_mode": {"type": "string", "enum": ["semantic", "keyword", "grep"], "default": "semantic"}
                                            },
                                            "required": ["op", "query"]
                                        }
                                    }
                                }
                            ]
                            print(f"ASK_MODE_TOOLS: Using minimal read-only toolset with {len(safe_tools)} tools")
                        else:
                            # AGENT MODE: Full tools
                            safe_tools = copy.deepcopy(godot_tools)
                            print(f"AGENT_MODE_TOOLS: Using full toolset with {len(safe_tools)} tools")
                        
                        completion_params = {
                            "model": model_try,
                            "messages": openai_messages_send,
                            "tools": safe_tools,
                            "tool_choice": "auto", 
                            "stream": True,
                            "timeout": 300,  # CRITICAL: 5 minute timeout for streaming responses
                            "request_timeout": 300  # CRITICAL: Explicit request timeout for GCP Cloud Run
                        }
                        
                        # EXPERIMENTAL: Always enable thinking mode when requested, regardless of tools
                        # Anthropic should support thinking + tools together
                        if reasoning_params:
                            completion_params.update(reasoning_params)
                            print(f"THINKING_MODE: Enabled for {model_friendly_name} with params: {reasoning_params}")
                        else:
                            print(f"THINKING_MODE: Not requested for {model_friendly_name}")
                        
                        # Execute request; if provider rejects thinking params, retry without them
                        try:
                            response = completion(**completion_params)
                        except Exception as e_comp:
                            err_msg = str(e_comp).lower()
                            if reasoning_params and ("reasoning" in err_msg or "thinking" in err_msg or "unsupported" in err_msg or "invalid" in err_msg):
                                print("THINKING_MODE: Provider rejected thinking params; retrying without reasoning/thinking")
                                completion_params_fallback = dict(completion_params)
                                completion_params_fallback.pop("reasoning_effort", None)
                                completion_params_fallback.pop("thinking", None)
                                completion_params_fallback.pop("reasoning", None)
                                response = completion(**completion_params_fallback)
                            else:
                                raise
                        
                        # Process the stream inside the try block to catch streaming errors
                        full_text_response = ""
                        full_reasoning_content = ""
                        thinking_blocks = []
                        tool_call_aggregator = {}
                        tool_ids = {}
                        current_tool_index = None
                        chunk_count = 0
                        
                        # CRITICAL FIX: Track tool call IDs for debugging orphaned tool calls
                        processed_tool_call_ids = set()
                        print(f"STREAM_PROCESSING: Starting stream processing for conversation turn {conversation_turn}")
                        
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
                                
                                # Handle reasoning content (thinking mode)
                                reasoning_content = getattr(delta, 'reasoning_content', None) if hasattr(delta, 'reasoning_content') else delta.get('reasoning_content')
                                if reasoning_content:
                                    full_reasoning_content += reasoning_content
                                    yield json.dumps({
                                        "reasoning_delta": reasoning_content,
                                        "status": "thinking"
                                    }) + '\n'
                                
                                # Handle thinking blocks (Anthropic-specific)
                                thinking_blocks_delta = getattr(delta, 'thinking_blocks', None) if hasattr(delta, 'thinking_blocks') else delta.get('thinking_blocks')
                                if thinking_blocks_delta:
                                    thinking_blocks.extend(thinking_blocks_delta)
                                    yield json.dumps({
                                        "thinking_blocks_delta": thinking_blocks_delta,
                                        "status": "thinking_blocks"
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
                                            final_tool_id = tc_id or f"call_{index}_{int(time.time() * 1000)}"
                                            tool_ids[key] = final_tool_id
                                            # Track all tool call IDs being processed
                                            processed_tool_call_ids.add(final_tool_id)
                                            print(f"TOOL_CALL_TRACK: Registered tool call ID: {final_tool_id}")
                                        
                                        # Accumulate function name and arguments
                                        if fn_name:
                                            tool_call_aggregator[key]["name"] = fn_name
                                            # CRITICAL FIX: Send executing_tools IMMEDIATELY when tool name arrives
                                            # Don't wait for arguments to finish - show placeholder now!
                                            if len(tool_call_aggregator[key]["name"]) > 0 and key not in locals().get('instant_notified', set()):  
                                                if 'instant_notified' not in locals():
                                                    instant_notified = set()
                                                instant_notified.add(key)
                                                
                                                # Send immediate notification to frontend for instant UI feedback
                                                # Use "tool_starting" status to create placeholder without triggering execution
                                                early_tool_response = {
                                                    "status": "tool_starting",
                                                    "tool_starting": fn_name,
                                                    "tool_id": tool_ids[key]
                                                }
                                                early_response_str = json.dumps(early_tool_response) + '\n'
                                                print(f"⚡ INSTANT_TOOL_NOTIFICATION: Sending tool_starting for {fn_name} immediately ({len(early_response_str)} bytes)")
                                                yield early_response_str
                                            else:
                                                tool_call_aggregator[key]["name"] = fn_name
                                        if fn_args:
                                            # DEBUG: Log accumulated argument length
                                            current_len = len(tool_call_aggregator[key]["arguments"])
                                            tool_call_aggregator[key]["arguments"] += fn_args
                                            new_len = len(tool_call_aggregator[key]["arguments"])
                                            
                                            # CRITICAL: Log size thresholds for monitoring
                                            if new_len > 50000:  # 50KB - Dangerous size!
                                                print(f"🚨 TOOL_ARGS_OVERFLOW: {fn_name} arguments reached {new_len} chars - CRITICAL SIZE!")
                                                print(f"🚨 This may cause empty arguments bug - monitoring for corruption...")
                                                # Log the structure to help debug
                                                try:
                                                    import json as _json_check
                                                    _json_check.loads(tool_call_aggregator[key]["arguments"])
                                                    print(f"✅ TOOL_ARGS_VALID: Large arguments still parse as valid JSON")
                                                except Exception as parse_error:
                                                    print(f"💥 TOOL_ARGS_CORRUPTED: Large arguments are malformed JSON: {parse_error}")
                                                    print(f"💥 This will likely cause empty arguments fallback!")
                                            elif new_len > 15000 and current_len < 15000:
                                                print(f"TOOL_ARGS_ACCUMULATING: {fn_name} arguments now {new_len} chars (getting very large!)")
                                            elif new_len > 5000 and current_len < 5000:
                                                print(f"TOOL_ARGS_ACCUMULATING: {fn_name} arguments now {new_len} chars (crossed 5KB threshold)")
                        
                        print(f"RESPONSE_DEBUG: Processed {chunk_count} chunks, text_length: {len(full_text_response)}, tools: {len(tool_call_aggregator)}")
                        try:
                            # chat_stream_summary with minimal fields for analytics
                            log_event('chat_stream_summary', {
                                'model': model_friendly_name,
                                'chunks': chunk_count,
                                'used_tools': [f.get('name') for f in tool_call_aggregator.values()] if tool_call_aggregator else [],
                                'text_len': len(full_text_response or ''),
                            })
                        except Exception:
                            pass
                        if tool_call_aggregator:
                            tool_names = [f.get('name', 'unknown') for f in tool_call_aggregator.values()]
                            tool_ids_list = list(tool_ids.values())
                            print(f"RESPONSE_DEBUG: Tool calls: {tool_names}")
                            print(f"TOOL_CALL_IDS: Processed IDs: {tool_ids_list}")
                            
                            # CRITICAL FIX: Validate all tool calls have proper IDs
                            for i, (key, tool_func) in enumerate(tool_call_aggregator.items()):
                                if key not in tool_ids or not tool_ids[key]:
                                    missing_id = f"missing_id_{int(time.time() * 1000)}_{i}"
                                    tool_ids[key] = missing_id
                                    print(f"TOOL_CALL_RECOVERY: Generated missing ID: {missing_id} for tool: {tool_func.get('name', 'unknown')}")
                        
                        if not full_text_response and not tool_call_aggregator:
                            print("RESPONSE_DEBUG: WARNING - OpenAI responded with NO content and NO tool calls!")
                        
                        # Successfully processed the stream, break out of retry loop
                        break
                        
                    except Exception as e:
                        print(f"STREAM_ERROR: {e}")
                        err_name = e.__class__.__name__
                        error_str = str(e)
                        overloaded = "Overloaded" in error_str
                        transient = err_name in ("InternalServerError", "RateLimitError", "ServiceUnavailableError") or overloaded
                        
                        # CRITICAL: Context window exceeded - trigger IMMEDIATE summarization!
                        is_context_exceeded = ("ContextWindowExceededError" in err_name or 
                                              "prompt is too long" in error_str or
                                              "maximum context length" in error_str.lower() or
                                              "tokens >" in error_str)
                        
                        if is_context_exceeded and attempts == 0:  # First attempt with this error
                            print(f"🚨 CONTEXT_OVERFLOW: Context window exceeded! Triggering emergency summarization...")
                            
                            # Get actual model limit
                            actual_limit = _get_model_token_limit(model_try)
                            current_tokens = _count_tokens_for_messages(openai_messages_send, model_try)
                            print(f"🚨 OVERFLOW_STATS: {current_tokens} tokens used, limit is {actual_limit}")
                            
                            # Notify frontend
                            yield json.dumps({
                                "status": "emergency_summarizing",
                                "message": f"Conversation too large ({current_tokens} tokens > {actual_limit} limit) - condensing...",
                                "original_count": len(openai_messages_send),
                                "tokens_used": current_tokens,
                                "tokens_limit": actual_limit
                            }) + '\n'
                            
                            # Force AGGRESSIVE summarization to get well under limit
                            original_count = len(openai_messages_send)
                            system_msgs = [m for m in openai_messages_send if m.get('role') == 'system']
                            other_msgs = [m for m in openai_messages_send if m.get('role') != 'system']
                            
                            # Emergency: Keep only last 15 messages (aggressive!)
                            openai_messages_send = system_msgs + other_msgs[-15:]
                            final_tokens = _count_tokens_for_messages(openai_messages_send, model_try)
                            
                            print(f"✅ EMERGENCY_SUMMARIZE: Reduced {current_tokens} → {final_tokens} tokens ({original_count} → {len(openai_messages_send)} messages)")
                            print(f"✅ EMERGENCY_RESULT: Now at {(final_tokens/actual_limit*100):.1f}% of limit - safe to retry")
                            
                            # Notify frontend of completion
                            # REMOVED: Legacy emergency summarization notification
                            # Emergency trimming is handled internally, no need to notify frontend
                            
                            # Retry immediately with trimmed conversation
                            attempts = 0  # Reset attempts for retry
                            continue
                        
                        # CRITICAL RECOVERY: Try duplicate tool result recovery FIRST (most common issue)
                        error_str_lower = error_str.lower()
                        
                        # 1. DUPLICATE TOOL RESULTS RECOVERY (fixes David's issue)
                        if (recovery_attempts < max_recovery_attempts and 
                            (("multiple" in error_str_lower and "tool_result" in error_str_lower) or
                             ("each tool_use must have a single result" in error_str_lower))):
                            
                            recovery_attempts += 1
                            print(f"DUPLICATE_TOOL_RECOVERY: Detected duplicate tool results error (attempt {recovery_attempts}/{max_recovery_attempts}), attempting recovery...")
                            
                            # Try to fix duplicate tool results  
                            fixed_messages, was_fixed = _detect_and_fix_duplicate_tool_results(openai_messages_send, str(e))
                            
                            if was_fixed:
                                # Update messages and retry immediately with fixed messages
                                openai_messages_send = fixed_messages
                                print(f"DUPLICATE_TOOL_RECOVERY: Fixed duplicate tool results, retrying request...")
                                
                                # Send recovery status to frontend
                                yield json.dumps({
                                    "status": "recovering", 
                                    "message": f"Recovered from duplicate tool results (attempt {recovery_attempts}), continuing...",
                                    "recovery_type": "duplicate_tool_results",
                                    "recovery_attempt": recovery_attempts
                                }) + '\n'
                                
                                # Log recovery event for analytics
                                try:
                                    log_event('duplicate_tool_recovery', {
                                        'attempt': recovery_attempts,
                                        'model': model_friendly_name,
                                        'error_snippet': str(e)[:100]
                                    })
                                except Exception:
                                    pass
                                
                                # Retry immediately with fixed messages
                                continue
                            else:
                                print(f"DUPLICATE_TOOL_RECOVERY: Failed to fix duplicate tool results, trying orphaned tool call recovery...")
                        
                        # 2. ORPHANED TOOL CALLS RECOVERY (handles missing results)
                        elif (recovery_attempts < max_recovery_attempts and 
                            "tool_calls" in error_str_lower and "tool_call_id" in error_str_lower and 
                            "must be followed by tool messages" in error_str_lower):
                            
                            recovery_attempts += 1
                            print(f"TOOL_CALL_RECOVERY: Detected orphaned tool calls error (attempt {recovery_attempts}/{max_recovery_attempts}), attempting recovery...")
                            
                            # Try to fix the orphaned tool calls
                            fixed_messages, was_fixed = _detect_and_fix_orphaned_tool_calls(openai_messages_send, str(e))
                            
                            if was_fixed:
                                # Update messages and retry immediately with the fixed messages
                                openai_messages_send = fixed_messages
                                print(f"TOOL_CALL_RECOVERY: Fixed orphaned tool calls, retrying request...")
                                
                                # Send recovery status to frontend
                                yield json.dumps({
                                    "status": "recovering",
                                    "message": f"Recovered from interrupted tool execution (attempt {recovery_attempts}), continuing...",
                                    "recovery_type": "orphaned_tool_calls",
                                    "recovery_attempt": recovery_attempts
                                }) + '\n'
                                
                                # Log recovery event for analytics
                                try:
                                    log_event('tool_call_recovery', {
                                        'attempt': recovery_attempts,
                                        'model': model_friendly_name,
                                        'error_snippet': str(e)[:100]
                                    })
                                except Exception:
                                    pass
                                
                                # Retry immediately with fixed messages (don't increment attempts counter)
                                continue
                            else:
                                print(f"TOOL_CALL_RECOVERY: Failed to fix orphaned tool calls, falling back to normal error handling")
                        elif recovery_attempts >= max_recovery_attempts and ("tool_calls" in error_str or "tool_result" in error_str):
                            print(f"TOOL_CALL_RECOVERY: Maximum recovery attempts ({max_recovery_attempts}) reached, giving up on recovery")
                        
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
                                    "provider": model_friendly_name, 
                                    "attempt": attempts,
                                    "max_attempts": max_attempts,
                                    "error": str(e)[:100],
                                    "message": "Rate limit exceeded, retrying..."
                                }) + '\n'
                            else:
                                yield json.dumps({
                                    "status": "retrying_provider",
                                    "provider": model_friendly_name,
                                    "attempt": attempts,
                                    "max_attempts": max_attempts,
                                    "error": str(e)[:100]  # Show error snippet
                                }) + '\n'
                            
                            print(f"RETRY: Attempt {attempts}/{max_attempts} after error: {e}")
                            time.sleep(1.0)  # Fixed 1 second delay as requested
                            continue

                        # After 5 retries, fallback to preferred OpenAI model if not already tried
                        providers_tried.add(model_try)
                        
                        fallback_friendly, fallback_model_id = _get_openai_preferred_model()

                        # Notify about model switching if it's due to rate limits
                        if is_rate_limit and model_try != fallback_model_id:
                            yield json.dumps({
                                "status": "provider_switched",
                                "from_provider": model_friendly_name,
                                "to_provider": fallback_friendly,
                                "reason": "Rate limit exceeded",
                                "message": f"Switching from {model_friendly_name} to {fallback_friendly} due to rate limits"
                            }) + '\n'
                        
                        # Always try preferred OpenAI model after retries exhausted
                        if fallback_model_id not in providers_tried:
                            yield json.dumps({
                                "status": "switching_model",
                                "from": model_friendly_name,
                                "to": fallback_friendly,
                                "reason": f"Provider overloaded after {max_attempts} retries"
                            }) + '\n'
                            print(f"SWITCHING: From {model_try} to {fallback_model_id} after {max_attempts} failed attempts")
                            model_try = fallback_model_id
                            attempts = 0  # Reset attempts for new provider
                            continue

                        # No retries/fallbacks left – send clear error status then bubble up to main handler
                        yield json.dumps({
                            "status": "error",
                            "error_type": "provider_exhausted",
                            "message": f"All AI providers failed after {max_attempts} attempts. Please try again later.",
                            "providers_tried": list(providers_tried)
                        }) + '\n'
                        raise

                # Now that we've processed all chunks, handle the results

                # --- Smart Backend Tool Detection ---
                def _needs_backend_processing(func_name: str, func_args: str) -> bool:
                    """Determine if a tool call needs backend processing based on the operation"""
                    # Legacy individual tools - always backend
                    if func_name in ["image_operation", "search_across_project", "search_across_godot_docs", 
                                   "slice_spritesheet", "search_godot_assets", "install_godot_asset", "generate_3d_model"]:
                        return True
                    
                    # Parse arguments to check operation
                    try:
                        import json as _json_parse
                        args = _json_parse.loads(func_args) if func_args else {}
                        op = args.get('op', '')
                        
                        # project_manager: only specific operations need backend
                        if func_name == "project_manager":
                            return op in ["assets.search", "assets.install", "updates.check"]
                        
                        # search_manager: check search mode to decide frontend vs backend
                        elif func_name == "search_manager":
                            if op == "docs.search":
                                return True  # Docs search always on backend
                            elif op == "project.search":
                                # Grep search needs frontend (local filesystem), others need backend
                                search_mode = args.get('search_mode', 'semantic')
                                return search_mode != 'grep'  # Backend for semantic/keyword/hybrid, frontend for grep
                            return False
                        
                        # resource_manager: only image operations need backend
                        elif func_name == "resource_manager":
                            return op in ["image.generate_or_edit", "image.slice_spritesheet"]
                        
                        # graph_manager: neighbors require backend graph data
                        elif func_name == "graph_manager":
                            return True
                        
                        # runtime_manager: all operations are frontend-only
                        elif func_name == "runtime_manager":
                            return False
                        
                        # runtime_inspector: all operations are frontend-only
                        elif func_name == "runtime_inspector":
                            return False
                        
                        # terminal_manager: all operations are frontend-only (need local machine access)
                        elif func_name == "terminal_manager":
                            return False
                        
                        # todo_manager: always backend (uses in-memory store)
                        elif func_name == "todo_manager":
                            return True
                        
                        # All other tools are frontend-only
                        return False
                        
                    except Exception as e:
                        print(f"BACKEND_DETECTION_ERROR: Failed to parse {func_name} args: {e}")
                        # Conservative: if we can't parse, assume frontend
                        return False
                
                # Detect tools that actually need backend processing
                backend_tools_detected = []
                backend_calls = {}
                
                for k, func in tool_call_aggregator.items():
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "")
                    
                    # DEBUG: Log raw arguments before detection
                    if len(func_args) > 1000:
                        print(f"BACKEND_DETECTION_DEBUG: {func_name} has {len(func_args)} char arguments")
                        print(f"BACKEND_DETECTION_DEBUG: First 200 chars: {func_args[:200]}")
                        print(f"BACKEND_DETECTION_DEBUG: Last 200 chars: {func_args[-200:]}")
                        # Try to parse to see what the issue is
                        try:
                            test_parse = json.loads(func_args)
                            print(f"BACKEND_DETECTION_DEBUG: Arguments parse OK, keys: {list(test_parse.keys())}")
                        except Exception as e:
                            print(f"BACKEND_DETECTION_DEBUG: Arguments parse FAILED: {e}")
                            # Log character at position 60-61 where error occurs
                            if len(func_args) > 61:
                                print(f"BACKEND_DETECTION_DEBUG: Chars around position 60: '{func_args[55:65]}'")
                    
                    if _needs_backend_processing(func_name, func_args):
                        backend_tools_detected.append(func_name)
                        backend_calls[k] = func
                        print(f"BACKEND_DETECTION: {func_name} needs backend processing")
                    else:
                        print(f"BACKEND_DETECTION: {func_name} will be handled by frontend")
                
                print(f"BACKEND_DETECTION: {len(backend_calls)} tools need backend processing: {backend_tools_detected}")
                
                # Simple guardrails to prevent infinite repeated backend calls
                # Build a canonical cache key from tool name and normalized arguments
                def _make_tool_cache_key(name: str, args_raw: str) -> str:
                    try:
                        a = json.loads(args_raw) if args_raw else {}
                    except Exception:
                        a = {"raw": str(args_raw)}
                    # Only include stable fields
                    canonical = {
                        "name": name,
                        "query": a.get("query"),
                        "pattern": a.get("pattern"),
                        "operation": a.get("operation"),
                        "max_results": a.get("max_results"),
                    }
                    return json.dumps(canonical, sort_keys=True)
                if backend_calls:
                    # This is a backend-only tool call, so we will execute it,
                    # add the results to the conversation, and loop again for the AI's final response.
                    
                    # CRITICAL LOOP PREVENTION: Check if these tools were already called recently
                    skip_duplicate_tools = []
                    for i, func in backend_calls.items():
                        cache_key = _make_tool_cache_key(func["name"], func["arguments"])
                        current_count = tool_call_counts.get(cache_key, 0)
                        
                        if current_count >= 3:
                            print(f"LOOP_PREVENTION: Tool {func['name']} already called {current_count} times with same args - SKIPPING to prevent infinite loop!")
                            skip_duplicate_tools.append(i)
                        else:
                            tool_call_counts[cache_key] = current_count + 1
                            print(f"LOOP_DETECTION: {func['name']} call count: {current_count + 1}")
                    
                    # Remove duplicate tools from backend_calls
                    for skip_key in skip_duplicate_tools:
                        del backend_calls[skip_key]
                    
                    if not backend_calls:
                        print(f"LOOP_PREVENTION: All tools were duplicates, skipping execution")
                        # Send a message to break the loop
                        yield json.dumps({"status": "completed", "message": "Prevented infinite tool loop"}) + '\n'
                        break
                    
                    # CRITICAL: Send executing_tools status first so frontend creates assistant message with tool calls
                    original_tool_calls_for_history = []
                    for i, func in backend_calls.items():
                        tool_id = tool_ids[i]
                        
                        # CRITICAL: Detect tool argument corruption before sanitization
                        original_args = func["arguments"]
                        sanitized_args = _sanitize_tool_arguments(original_args)
                        
                        # Log corruption detection
                        if len(original_args) > 1000 and sanitized_args == "{}":
                            print(f"🚨 TOOL_CORRUPTION_DETECTED: {func['name']} arguments ({len(original_args)} chars) sanitized to empty!")
                            print(f"🚨 Original args preview: {original_args[:200]}...")
                            print(f"🚨 This is the ROOT CAUSE of the 'missing op parameter' bug!")
                        
                        original_tool_calls_for_history.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": func["name"], "arguments": sanitized_args},
                        })
                    
                    # Send executing_tools to frontend BEFORE executing backend tools
                    yield json.dumps({
                        "status": "executing_tools",
                        "assistant_message": {
                            "role": "assistant", 
                            "content": full_text_response or None,
                            "tool_calls": [{"id": tc["id"], "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in original_tool_calls_for_history]
                        }
                    }) + '\n'
                    
                    # Execute tools
                    tool_results_for_history = []
                    
                    for i, func in backend_calls.items():
                        tool_id = tool_ids[i]
                        
                        if func["name"] == "image_operation":
                            # Check for stop before tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "image_operation", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            
                            # ============ CLEAR TOOL CALL LOGGING ============
                            print("=" * 80)
                            print(f"BACKEND TOOL CALLED: image_operation")
                            print(f"TOOL ID: {tool_id}")
                            print("ARGUMENTS:")
                            try:
                                arguments = json.loads(func["arguments"])
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except json.JSONDecodeError:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            # ===============================================
                            
                            # EARLY VALIDATION: Check for oversized arguments that could corrupt JSON
                            total_args_size = sum(len(str(v)) for v in arguments.values() if v is not None)
                            if total_args_size > 10000:  # 10KB threshold
                                print(f"⚠️  LARGE_ARGS_WARNING: Tool {func['name']} has {total_args_size} chars of arguments - potential corruption risk")
                                # Check for specific large content fields
                                if 'content' in arguments and isinstance(arguments['content'], str) and len(arguments['content']) > 5000:
                                    print(f"⚠️  LARGE_CONTENT_WARNING: 'content' field has {len(arguments['content'])} chars - this may corrupt tool calls")
                                    print("💡 SUGGESTION: Break large files into smaller chunks or use incremental operations")
                            
                            # Log tool call
                            log_tool_call("image_operation", tool_id, arguments)
                            
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
                            # Poll for stop while tool runs with timeout protection
                            timeout_start = time.time()
                            max_timeout = 300  # 5 minutes max for image operations
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during image_operation")
                                    # We don't kill the thread; we just stop streaming and drop the result
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: image_operation exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Tool execution timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.1)
                            image_result = _tool_result_holder["result"] or {"success": False, "error": "image_operation returned no result"}
                            
                            # Check for stop after tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Log tool result
                            log_tool_result("image_operation", tool_id, image_result, duration_ms=0)
                            
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
                                "image_id": image_result.get("image_id"),
                                "image_name": image_result.get("image_name"),
                                "description": image_result.get("description"),
                                "prompt": image_result.get("prompt"),
                                "style": image_result.get("style"),
                                "format": image_result.get("format"),
                                "width": image_result.get("width"),
                                "height": image_result.get("height"),
                                "input_images": image_result.get("input_images", 0),
                                "requested_images": image_result.get("requested_images", 0),
                                "edited_from": image_result.get("edited_from")
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
                            
                            # ============ CLEAR TOOL CALL LOGGING ============
                            print("=" * 80)
                            print(f"🔍 BACKEND TOOL CALLED: search_across_project")
                            print(f"🆔 TOOL ID: {tool_id}")
                            print("📋 ARGUMENTS:")
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except Exception:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            # ===============================================
                            
                            # Log tool call
                            log_tool_call("search_across_project", tool_id, arguments)
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
                            timeout_start = time.time()
                            max_timeout = 120  # 2 minutes max for project search operations
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_across_project")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: search_across_project exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Search operation timed out after {max_timeout} seconds"}) + '\n'
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
                            
                            # Log tool result 
                            log_tool_result("search_across_project", tool_id, search_result, duration_ms=0)
                            
                            # CRITICAL FREEZE FIX: Only remove the massive graph object, keep file metadata for AI
                            search_result_for_frontend = dict(search_result)
                            search_result_for_frontend.pop("graph", None)  # Remove only the 100KB+ graph object
                            
                            # Limit number of files but keep their metadata for AI
                            if "results" in search_result_for_frontend and isinstance(search_result_for_frontend["results"], dict):
                                results_copy = dict(search_result_for_frontend["results"])
                                
                                # Limit quantity but preserve quality (metadata intact for AI)
                                if "similar_files" in results_copy and len(results_copy["similar_files"]) > 10:
                                    results_copy["similar_files"] = results_copy["similar_files"][:10]
                                if "central_files" in results_copy and len(results_copy["central_files"]) > 5:
                                    results_copy["central_files"] = results_copy["central_files"][:5]
                                    
                                search_result_for_frontend["results"] = results_copy
                                print(f"FREEZE_FIX: Limited to {len(results_copy.get('similar_files', []))} files, keeping metadata for AI")
                            
                            # Yield result to frontend immediately (stripped version)
                            yield json.dumps({"tool_executed": "search_across_project", "tool_result": search_result_for_frontend, "tool_call_id": tool_id, "status": "tool_completed"}) + '\n'
                            
                            # Prepare tool result for AI model - include essential file info + graph intelligence
                            similar_files_full = search_result.get("results", {}).get("similar_files", [])[:10]
                            
                            # Keep essential fields + graph scores for AI decision making
                            similar_files_for_ai = []
                            for file_info in similar_files_full:
                                similar_files_for_ai.append({
                                    "file_path": file_info.get("file_path"),               # ESSENTIAL
                                    "chunk_start": file_info.get("chunk_start"),           # ESSENTIAL (line number)
                                    "chunk_end": file_info.get("chunk_end"),               # ESSENTIAL
                                    "similarity": file_info.get("similarity", 0.0),        # USEFUL (how relevant)
                                    "centrality_score": file_info.get("centrality_score"), # USEFUL (architectural importance)
                                    "ranking_explanation": file_info.get("ranking_explanation", "")  # USEFUL (why selected)
                                })
                            
                            tool_result_for_openai = {
                                "success": search_result.get("success"),
                                "query": search_result.get("query"),
                                "file_count": search_result.get("file_count", 0),
                                "message": search_result.get("message"),
                                "similar_files": similar_files_for_ai,                    # Files with metadata!
                                "graph_summary": search_result.get("results", {}).get("graph_summary", {})  # Graph stats
                            }
                            # Add central files for AI (architecturally important)
                            central_files = search_result.get("results", {}).get("central_files", [])
                            if central_files:
                                tool_result_for_openai["central_files"] = central_files[:5]
                            
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
                            
                            # ============ CLEAR TOOL CALL LOGGING ============
                            print("=" * 80)
                            print(f"📚 BACKEND TOOL CALLED: search_across_godot_docs")
                            print(f"🆔 TOOL ID: {tool_id}")
                            print("📋 ARGUMENTS:")
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except Exception:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            # ===============================================
                            
                            # Log tool call
                            log_tool_call("search_across_godot_docs", tool_id, arguments)
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
                            timeout_start = time.time()
                            max_timeout = 120  # 2 minutes max for docs search
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_across_godot_docs")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: search_across_godot_docs exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Docs search timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.05)
                            docs_result = _tool_result_holder["result"] or {"success": False, "error": "search_across_godot_docs returned no result"}
                            # Normalize failure into minimal contract-compliant payload
                            if not docs_result.get('success', False):
                                msg = docs_result.get('error') or docs_result.get('message') or 'Docs search failed'
                                docs_result = {
                                    'success': False,
                                    'query': arguments.get('query'),
                                    'results': [],
                                    'message': f"search_across_godot_docs error: {msg}"
                                }

                            # Check for stop after tool execution
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return

                            # Log tool result
                            log_tool_result("search_across_godot_docs", tool_id, docs_result, duration_ms=0)
                            
                            # Yield result to frontend immediately
                            yield json.dumps({"tool_executed": "search_across_godot_docs", "tool_result": docs_result, "tool_call_id": tool_id, "status": "tool_completed"}) + '\n'

                            # Prepare compact tool result for model history
                            tool_result_for_openai = {
                                "success": docs_result.get("success"),
                                "query": arguments.get('query'),
                                "source": docs_result.get("source"),
                                "top": docs_result.get("results", [])[:3]
                            }

                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "search_across_godot_docs",
                                "content": json.dumps(tool_result_for_openai),
                            })
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
                            timeout_start = time.time()
                            max_timeout = 180  # 3 minutes max for spritesheet processing
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during slice_spritesheet")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: slice_spritesheet exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Spritesheet processing timed out after {max_timeout} seconds"}) + '\n'
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
                            timeout_start = time.time()
                            max_timeout = 120  # 2 minutes max for asset search
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_godot_assets")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: search_godot_assets exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Asset search timed out after {max_timeout} seconds"}) + '\n'
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
                            timeout_start = time.time()
                            max_timeout = 300  # 5 minutes max for asset install (downloads can be slow)
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during install_godot_asset")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: install_godot_asset exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Asset install timed out after {max_timeout} seconds"}) + '\n'
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
                        
                        elif func["name"] == "generate_3d_model":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "generate_3d_model", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_3d_generation():
                                try:
                                    _tool_result_holder["result"] = generate_3d_model_internal(arguments)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_3d_generation, daemon=True)
                            t.start()
                            timeout_start = time.time()
                            max_timeout = 600  # 10 minutes max for 3D model generation (very slow operation)
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during generate_3d_model")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: generate_3d_model exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"3D model generation timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.1)  # Longer delay for 3D generation
                            
                            model_3d_result = _tool_result_holder["result"] or {"success": False, "error": "generate_3d_model returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Yield result to frontend immediately
                            yield json.dumps({
                                "tool_executed": "generate_3d_model",
                                "tool_result": model_3d_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # Prepare result for conversation history (exclude massive GLB data)
                            model_3d_summary = {
                                "success": model_3d_result.get("success"),
                                "prompt": model_3d_result.get("prompt"),
                                "model": model_3d_result.get("model"),
                                "generation_time": model_3d_result.get("generation_time"),
                                "job_id": model_3d_result.get("job_id"),
                                "format": model_3d_result.get("format"),
                                "file_size": model_3d_result.get("file_size")
                            }
                            # Exclude the massive 'glb_data' field to save tokens
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "generate_3d_model",
                                "content": json.dumps(model_3d_summary)
                            })
                        
                        elif func["name"] == "check_for_app_updates":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({"tool_starting": "check_for_app_updates", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            try:
                                arguments = json.loads(func["arguments"]) if func.get("arguments") else {}
                            except Exception:
                                arguments = {}
                            
                            update_result = check_for_app_updates_internal(arguments)
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # Yield result to frontend immediately
                            yield json.dumps({
                                "tool_executed": "check_for_app_updates",
                                "tool_result": update_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # Prepare result for conversation history
                            update_summary = {
                                "success": update_result.get("success"),
                                "update_available": update_result.get("update_available"),
                                "current_version": update_result.get("current_version"),
                                "message": update_result.get("message")
                            }
                            if update_result.get("update_available"):
                                update_summary["new_version"] = update_result.get("update_info", {}).get("version")
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "check_for_app_updates",
                                "content": json.dumps(update_summary)
                            })
                        
                        elif func["name"] == "project_manager":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            # ============ CLEAR TOOL CALL LOGGING ============
                            print("=" * 80)
                            print(f"📁 BACKEND TOOL CALLED: project_manager")
                            print(f"🆔 TOOL ID: {tool_id}")
                            print("📋 ARGUMENTS:")
                            try:
                                arguments = json.loads(func["arguments"])
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except json.JSONDecodeError:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            # ===============================================
                            
                            # CRITICAL FIX: Emit tool_starting for immediate frontend feedback
                            yield json.dumps({"tool_starting": "project_manager", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            
                            # Inject project_root/project_path from Flask context if not provided
                            if not arguments.get('project_root') and not arguments.get('project_path') and hasattr(g, 'project_root') and g.project_root:
                                arguments['project_root'] = g.project_root
                                arguments['project_path'] = g.project_root  # Some operations expect project_path
                                print(f"PROJECT_MANAGER: Injected project_root from Flask context: {g.project_root}")
                            
                            # CRITICAL FIX: Use threading + polling to allow tool_starting to be sent immediately
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_project_manager():
                                try:
                                    _tool_result_holder["result"] = project_manager_internal(arguments)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_project_manager, daemon=True)
                            t.start()
                            # Poll for stop while tool runs - this loop allows tool_starting to be sent
                            timeout_start = time.time()
                            max_timeout = 120  # 2 minutes max for project manager operations
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during project_manager")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: project_manager exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Project manager operation timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.05)  # Yield control to allow streaming
                            pm_result = _tool_result_holder["result"] or {"success": False, "error": "project_manager returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({
                                "tool_executed": "project_manager",
                                "tool_result": pm_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "project_manager",
                                "content": json.dumps(pm_result)
                            })
                        
                        elif func["name"] == "search_manager":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            # ============ CLEAR TOOL CALL LOGGING ============
                            print("=" * 80)
                            print(f"🔎 BACKEND TOOL CALLED: search_manager")
                            print(f"🆔 TOOL ID: {tool_id}")
                            print("📋 ARGUMENTS:")
                            try:
                                arguments = json.loads(func["arguments"])
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except json.JSONDecodeError:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            # ===============================================
                            
                            # CRITICAL FIX: Emit tool_starting for immediate frontend feedback
                            yield json.dumps({"tool_starting": "search_manager", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            
                            # Inject project_root from Flask context if not provided
                            if not arguments.get('project_root') and hasattr(g, 'project_root') and g.project_root:
                                arguments['project_root'] = g.project_root
                                print(f"SEARCH_MANAGER: Injected project_root from Flask context: {g.project_root}")
                            
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_search_manager():
                                try:
                                    _tool_result_holder["result"] = search_manager_internal(arguments, user)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_search_manager, daemon=True)
                            t.start()
                            timeout_start = time.time()
                            max_timeout = 120  # 2 minutes max for search manager operations
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during search_manager")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: search_manager exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Search manager operation timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.05)
                            sm_result = _tool_result_holder["result"] or {"success": False, "error": "search_manager returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            # CRITICAL FREEZE FIX: Only remove the massive graph object, keep file metadata for AI
                            # Handle both project search (dict) and docs search (list) formats
                            sm_result_for_frontend = dict(sm_result)
                            sm_result_for_frontend.pop("graph", None)  # Remove only the 100KB+ graph object
                            
                            # Handle different result formats
                            if "results" in sm_result_for_frontend:
                                if isinstance(sm_result_for_frontend["results"], dict):
                                    # PROJECT SEARCH: {"results": {"similar_files": [...]}}
                                    results_dict = dict(sm_result_for_frontend["results"])
                                    
                                    # Limit quantity but preserve quality (metadata intact for AI)
                                    if "similar_files" in results_dict and len(results_dict["similar_files"]) > 10:
                                        results_dict["similar_files"] = results_dict["similar_files"][:10]
                                    if "central_files" in results_dict and len(results_dict["central_files"]) > 5:
                                        results_dict["central_files"] = results_dict["central_files"][:5]
                                    
                                    sm_result_for_frontend["results"] = results_dict
                                    print(f"FREEZE_FIX: Limited project search to {len(results_dict.get('similar_files', []))} files")
                                
                                elif isinstance(sm_result_for_frontend["results"], list):
                                    # DOCS SEARCH: {"results": [...]}
                                    # CRITICAL: Strip full_content which can be 10,000+ chars per result!
                                    print(f"FREEZE_FIX: Processing docs search with {len(sm_result_for_frontend['results'])} results")
                                    lightweight_docs = []
                                    total_stripped = 0
                                    for doc in sm_result_for_frontend["results"][:10]:
                                        if isinstance(doc, dict):
                                            lightweight_doc = dict(doc)
                                            # Check if full_content exists and how big it is
                                            if "full_content" in lightweight_doc:
                                                content_size = len(str(lightweight_doc["full_content"]))
                                                total_stripped += content_size
                                                lightweight_doc.pop("full_content", None)  # REMOVE: 10KB+ per doc!
                                                print(f"FREEZE_FIX: Removed full_content ({content_size} chars) from doc: {lightweight_doc.get('title', 'unknown')}")
                                            lightweight_docs.append(lightweight_doc)
                                        else:
                                            lightweight_docs.append(doc)
                                    sm_result_for_frontend["results"] = lightweight_docs
                                    print(f"FREEZE_FIX: Stripped {total_stripped} total chars of full_content from {len(lightweight_docs)} docs results")
                            
                            yield json.dumps({
                                "tool_executed": "search_manager",
                                "tool_result": sm_result_for_frontend,  # STRIPPED version
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # For AI model history, handle both project search and docs search formats
                            sm_result_for_history = {
                                "success": sm_result.get("success"),
                                "query": sm_result.get("query"),
                                "file_count": sm_result.get("file_count", 0),
                                "message": sm_result.get("message"),
                                "search_mode": sm_result.get("search_mode", "unknown")
                            }
                            
                            # Handle different result formats
                            results_data = sm_result.get("results")
                            if isinstance(results_data, dict):
                                # PROJECT SEARCH: Extract file info with graph metadata
                                similar_files_full = results_data.get("similar_files", [])[:10]
                                similar_files_for_ai = []
                                for file_info in similar_files_full:
                                    similar_files_for_ai.append({
                                        "file_path": file_info.get("file_path"),
                                        "chunk_start": file_info.get("chunk_start"),
                                        "chunk_end": file_info.get("chunk_end"),
                                        "similarity": file_info.get("similarity", 0.0),
                                        "centrality_score": file_info.get("centrality_score"),
                                        "ranking_explanation": file_info.get("ranking_explanation", "")
                                    })
                                sm_result_for_history["similar_files"] = similar_files_for_ai
                                sm_result_for_history["graph_summary"] = results_data.get("graph_summary", {})
                                
                                # Add central files
                                central_files = results_data.get("central_files", [])
                                if central_files:
                                    sm_result_for_history["central_files"] = central_files[:5]
                                
                                # ADDED: Send lightweight graph relationships (not the massive nested object!)
                                # Extract key connections from the graph for AI to understand project structure
                                full_graph = sm_result.get("graph", {})
                                if full_graph and isinstance(full_graph, dict):
                                    graph_relationships = {}
                                    for file_path, file_context in list(full_graph.items())[:10]:  # Max 10 files
                                        if isinstance(file_context, dict):
                                            edges = file_context.get("edges", [])
                                            if edges:
                                                # Keep only essential edge info
                                                lightweight_edges = []
                                                for edge in edges[:5]:  # Max 5 edges per file
                                                    lightweight_edges.append({
                                                        "type": edge.get("type"),  # extends, preload, scene_ref, etc.
                                                        "target": edge.get("target") or edge.get("source"),  # Connected file
                                                        "weight": edge.get("weight", 1.0)
                                                    })
                                                graph_relationships[file_path] = lightweight_edges
                                    
                                    if graph_relationships:
                                        sm_result_for_history["graph_relationships"] = graph_relationships
                                        print(f"AI_CONTEXT: Added graph relationships for {len(graph_relationships)} files")
                            
                            elif isinstance(results_data, list):
                                # DOCS SEARCH: Results are a flat list
                                # Strip full_content but keep snippet for AI
                                docs_for_ai = []
                                for doc in results_data[:10]:
                                    if isinstance(doc, dict):
                                        doc_minimal = {
                                            "title": doc.get("title"),
                                            "snippet": doc.get("snippet"),  # Keep short preview
                                            "similarity": doc.get("similarity"),
                                            "class_name": doc.get("class_name"),
                                            "section": doc.get("section")
                                            # STRIPPED: full_content (10,000+ chars per doc!)
                                        }
                                        docs_for_ai.append(doc_minimal)
                                    else:
                                        docs_for_ai.append(doc)
                                sm_result_for_history["results"] = docs_for_ai
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "search_manager",
                                "content": json.dumps(sm_result_for_history)
                            })
                        
                        elif func["name"] == "graph_manager":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            print("=" * 80)
                            print(f"🕸️ BACKEND TOOL CALLED: graph_manager")
                            print(f"🆔 TOOL ID: {tool_id}")
                            try:
                                arguments = json.loads(func["arguments"])
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except json.JSONDecodeError:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            
                            yield json.dumps({"tool_starting": "graph_manager", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            
                            if not arguments.get('project_root') and hasattr(g, 'project_root') and g.project_root:
                                arguments['project_root'] = g.project_root
                                print(f"GRAPH_MANAGER: Injected project_root from Flask context: {g.project_root}")
                            
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_graph_manager():
                                try:
                                    _tool_result_holder["result"] = graph_manager_internal(arguments, user)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_graph_manager, daemon=True)
                            t.start()
                            timeout_start = time.time()
                            max_timeout = 60
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during graph_manager")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: graph_manager exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Graph manager operation timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.05)
                            graph_result = _tool_result_holder["result"] or {"success": False, "error": "graph_manager returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            graph_result_for_frontend = dict(graph_result)
                            graph_result_for_frontend.pop("graph_full", None)
                            
                            yield json.dumps({
                                "tool_executed": "graph_manager",
                                "tool_result": graph_result_for_frontend,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "graph_manager",
                                "content": json.dumps(graph_result)
                            })
                        
                        elif func["name"] == "todo_manager":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            print("=" * 80)
                            print(f"📝 BACKEND TOOL CALLED: todo_manager")
                            print(f"🆔 TOOL ID: {tool_id}")
                            try:
                                arguments = json.loads(func["arguments"])
                                for key, value in arguments.items():
                                    print(f"   {key}: {value}")
                            except json.JSONDecodeError:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            
                            yield json.dumps({"tool_starting": "todo_manager", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            
                            todo_result = todo_manager_internal(arguments, user)
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({
                                "tool_executed": "todo_manager",
                                "tool_result": todo_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "todo_manager",
                                "content": json.dumps(todo_result)
                            })
                        
                        elif func["name"] == "resource_manager":
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped before tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped before tool execution"}) + '\n'
                                return
                            
                            # ============ CLEAR TOOL CALL LOGGING ============
                            print("=" * 80)
                            print(f"BACKEND TOOL CALLED: resource_manager")
                            print(f"TOOL ID: {tool_id}")
                            print("📋 ARGUMENTS:")
                            try:
                                arguments = json.loads(func["arguments"])
                                for key, value in arguments.items():
                                    if isinstance(value, str) and len(value) > 200:
                                        display_value = value[:200] + "... (truncated)"
                                    else:
                                        display_value = value
                                    print(f"   {key}: {display_value}")
                            except json.JSONDecodeError:
                                arguments = {}
                                print("   (Failed to parse arguments)")
                            print("=" * 80)
                            # ===============================================
                            
                            # CRITICAL FIX: Emit tool_starting for immediate frontend feedback
                            yield json.dumps({"tool_starting": "resource_manager", "tool_id": tool_id, "status": "tool_starting"}) + '\n'
                            
                            # Execute resource_manager with threading support for image operations
                            from threading import Thread
                            _tool_result_holder = {"done": False, "result": None}
                            def _run_resource_mgr():
                                try:
                                    _tool_result_holder["result"] = resource_manager_internal(arguments, conversation_messages)
                                finally:
                                    _tool_result_holder["done"] = True
                            t = Thread(target=_run_resource_mgr, daemon=True)
                            t.start()
                            timeout_start = time.time()
                            max_timeout = 300  # 5 minutes max for resource manager operations (image processing can be slow)
                            while not _tool_result_holder["done"]:
                                if check_stop():
                                    print(f"STOP_DETECTED: Request {request_id} stopping during resource_manager")
                                    yield json.dumps({"status": "stopped", "message": "Request stopped during tool execution"}) + '\n'
                                    return
                                # CRITICAL: Add timeout protection to prevent infinite hangs in GCP
                                if time.time() - timeout_start > max_timeout:
                                    print(f"TIMEOUT_PROTECTION: resource_manager exceeded {max_timeout}s, aborting to prevent hang")
                                    yield json.dumps({"status": "error", "message": f"Resource manager operation timed out after {max_timeout} seconds"}) + '\n'
                                    return
                                time.sleep(0.1)
                            rm_result = _tool_result_holder["result"] or {"success": False, "error": "resource_manager returned no result"}
                            
                            if check_stop():
                                print(f"STOP_DETECTED: Request {request_id} stopped after tool execution")
                                yield json.dumps({"status": "stopped", "message": "Request stopped after tool execution"}) + '\n'
                                return
                            
                            yield json.dumps({
                                "tool_executed": "resource_manager",
                                "tool_result": rm_result,
                                "tool_call_id": tool_id,
                                "status": "tool_completed"
                            }) + '\n'
                            
                            # Log tool result for debugging
                            log_tool_result("resource_manager", tool_id, rm_result, duration_ms=0)
                            
                            # Prepare compact tool result for conversation history to avoid token bloat
                            tool_result_for_openai = {
                                "success": rm_result.get("success"),
                                # Pass through high-signal fields only
                                "op": (arguments.get("op") if isinstance(arguments, dict) else None),
                                "image_id": rm_result.get("image_id"),
                                "image_name": rm_result.get("image_name"),
                                "message": rm_result.get("message"),
                                "prompt": rm_result.get("prompt"),
                                "style": rm_result.get("style"),
                                "format": rm_result.get("format"),
                                "width": rm_result.get("width"),
                                "height": rm_result.get("height"),
                                "input_images": rm_result.get("input_images", 0),
                                "requested_images": rm_result.get("requested_images", 0),
                                "edited_from": rm_result.get("edited_from")
                            }
                            # Include slice hint if provided
                            if isinstance(rm_result.get("slice_hint"), dict):
                                tool_result_for_openai["slice_hint"] = rm_result.get("slice_hint")

                            tool_results_for_history.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": "resource_manager",
                                "content": json.dumps(tool_result_for_openai)
                            })
                
                    # Add the assistant's decision to call the tool to history
                    assistant_message = {"role": "assistant", "content": None, "tool_calls": original_tool_calls_for_history}
                    
                    # Add reasoning content if available (thinking mode)
                    if full_reasoning_content:
                        assistant_message["reasoning_content"] = full_reasoning_content
                    if thinking_blocks:
                        assistant_message["thinking_blocks"] = thinking_blocks
                    
                    conversation_messages.append(assistant_message)
                    print(f"CONVERSATION_ADD: Added assistant message with tool calls")
                    
                    # Add the results of the tool call to history
                    for tool_result in tool_results_for_history:
                        if tool_result is None:
                            print(f"CONVERSATION_ADD: ERROR - Attempting to add None tool result!")
                            continue
                        conversation_messages.append(tool_result)
                                                    # print(f"CONVERSATION_ADD: Added tool result: {tool_result.get('name', 'unknown')}")

                    # CRITICAL FIX: These lines must be INSIDE the "if backend_calls:" block!
                    # Now, loop again to get the final text response from the AI
                    print(f"CONVERSATION_LOOP: Backend tools complete. Continuing loop for AI's final text response (conversation now has {len(conversation_messages)} messages)")
                    print(f"CONVERSATION_LOOP: Last 3 message roles: {[conversation_messages[i]['role'] for i in range(max(0, len(conversation_messages)-3), len(conversation_messages))]}")
                    
                    # CRITICAL DEBUG: Show what we're sending back to AI
                    if len(conversation_messages) > 0:
                        last_msg = conversation_messages[-1]
                        if last_msg.get('role') == 'tool':
                            content_preview = str(last_msg.get('content', ''))[:200]
                            print(f"CONVERSATION_LOOP: Last tool result preview: {content_preview}...")
                    
                    continue  # MUST be inside "if backend_calls:" block!

                # --- Frontend Tool Calls & Final Text Responses ---
                
                print(f"FRONTEND_PROCESSING: Reached frontend tool processing. tool_call_aggregator has {len(tool_call_aggregator)} tools")
                
                # If we get here, it means no backend tools were called.
                # It's either a final text response or tool calls for the frontend.
                
                
                # CRITICAL FIX: Create original_tool_calls_for_history for frontend tools
                original_tool_calls_for_history = []
                print(f"FRONTEND_PROCESSING: Creating tool calls for history from {len(tool_call_aggregator)} remaining tools")
                for i, func in tool_call_aggregator.items():
                    tool_id = tool_ids[i]
                    tool_call_entry = {
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": func["name"], "arguments": _sanitize_tool_arguments(func["arguments"])},
                    }
                    original_tool_calls_for_history.append(tool_call_entry)
                    print(f"FRONTEND_TOOL_CALL: Created tool call for {func['name']} (ID: {tool_id})")
                    
                print(f"FRONTEND_PROCESSING: Created {len(original_tool_calls_for_history)} tool calls for history")
                
                # Append assistant message (will include tool calls if any)
                assistant_message = {
                    "role": "assistant",
                    "content": full_text_response if full_text_response else None,
                }
                
                # Add reasoning content if available (thinking mode)
                if full_reasoning_content:
                    assistant_message["reasoning_content"] = full_reasoning_content
                if thinking_blocks:
                    assistant_message["thinking_blocks"] = thinking_blocks

                if tool_call_aggregator:
                    print(f"FRONTEND_PROCESSING: Processing {len(tool_call_aggregator)} frontend tool calls")
                    # Prepare tool calls for both history and frontend
                    tool_calls_for_history = original_tool_calls_for_history  # Use the already created list
                    tool_calls_for_frontend = []
                    for i, func in tool_call_aggregator.items():
                        tool_id = tool_ids[i]
                        print(f"FRONTEND_PROCESSING: Processing tool {func['name']} with id {tool_id}")
                        
                        # CRITICAL: Detect argument corruption for frontend tools too
                        original_args = func["arguments"]
                        sanitized_args = _sanitize_tool_arguments(original_args)
                        
                        # Log corruption detection
                        if len(original_args) > 1000 and sanitized_args == "{}":
                            print(f"🚨 FRONTEND_TOOL_CORRUPTION: {func['name']} arguments ({len(original_args)} chars) sanitized to empty!")
                            print(f"🚨 Original args preview: {original_args[:200]}...")
                            print(f"🚨 This will cause 'missing op parameter' error in frontend tool execution!")
                        
                        tool_calls_for_frontend.append({
                            "id": tool_id,
                            "function": {
                                "name": func["name"],
                                "arguments": sanitized_args
                            }
                        })
                    
                    assistant_message["tool_calls"] = tool_calls_for_history
                    conversation_messages.append(assistant_message)
                    print(f"CONVERSATION_ADD: Added frontend assistant message with {len(tool_calls_for_history)} tool calls")
                    
                    print(f"FRONTEND_PROCESSING: Sending {len(tool_calls_for_frontend)} tool calls to frontend")
                    
                    # Log each frontend tool call
                    for tool_call in tool_calls_for_frontend:
                        try:
                            tool_name = tool_call['function']['name']
                            tool_id = tool_call['id']
                            arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}
                            log_tool_call(tool_name, tool_id, arguments)
                        except Exception as e:
                            print(f"⚠️  Error logging frontend tool call: {e}")
                    
                    # Yield tool calls to the frontend in the format it expects
                    frontend_response = {
                        "status": "executing_tools",
                        "assistant_message": {
                            "role": "assistant",
                            "content": full_text_response or None,
                            "tool_calls": tool_calls_for_frontend
                        }
                    }
                    print(f"FRONTEND_PROCESSING: ⚡ SENDING executing_tools to frontend NOW with {len(tool_calls_for_frontend)} tools")
                    
                    # SAFER: Use proper HTTP streaming without corrupting chunked encoding
                    base_response = json.dumps(frontend_response)
                    response_line = base_response + '\n'
                    
                    # Force flush using a safer method - yield empty string first to ensure transmission
                    if len(response_line) < 4096:
                        print(f"FRONTEND_PROCESSING: Small response ({len(response_line)} bytes) - forcing flush with empty yield")
                        yield ""  # Force Flask to send current buffer
                    
                    yield response_line
                    print(f"FRONTEND_PROCESSING: ✅ Yielded executing_tools ({len(response_line)} bytes) safely")
                    
                    # The assistant message was already added above, so don't add it again
                    
                    # Signal that the stream is ending but the overall task is waiting on the frontend.
                    yield json.dumps({"status": "awaiting_frontend_action"}) + '\n'
                    print(f"FRONTEND_PROCESSING: Tool calls sent, stream closing. Awaiting frontend tool execution in next request.")
                    break  # Exit loop after sending tools to frontend

                # If no tools, it's a final text response. Append and break.
                print(f"FRONTEND_PROCESSING: No tools detected, treating as final text response")
                conversation_messages.append(assistant_message)
                print(f"CONVERSATION_ADD: Added final text response message")
                try:
                    log_event('chat_completed', {
                        'final_text_len': len(full_text_response or ''),
                        'used_tools': [f.get('name') for f in tool_call_aggregator.values()] if tool_call_aggregator else [],
                    })
                except Exception:
                    pass
                
                print(f"🔍 CALLBACK_TRACE: About to check manual callback trigger, litellm_logger={litellm_logger is not None}")
                
                # CRITICAL FIX: Manually log streaming responses (bypass LiteLLM callbacks)
                # Direct logging to prevent duplicates and ensure reliability
                try:
                    if litellm_logger and full_text_response:
                        # Check if already logged to prevent duplicates
                        if not completion_params.get('_godot_callback_triggered', False):
                            completion_params['_godot_callback_triggered'] = True
                            
                            # Get timing
                            start_time = completion_params.get('_godot_start_time')
                            end_time = time.time()
                            duration_ms = int((end_time - start_time) * 1000) if start_time else 0
                            
                            # Get stored context
                            stored_context = getattr(litellm_logger, '_stored_context', {})
                            
                            # Create user input log
                            user_log = litellm_logger._create_user_input_log(
                                completion_params, 
                                completion_params.get('_godot_request_id', str(uuid.uuid4())), 
                                duration_ms, 
                                stored_context
                            )
                            
                            # Create assistant response log
                            assistant_log = litellm_logger._create_assistant_response_log(
                                completion_params, 
                                type('MockResponse', (), {'choices': [type('Choice', (), {'message': type('Message', (), {'content': full_text_response})()})()]})(),
                                full_text_response,
                                completion_params.get('_godot_request_id', str(uuid.uuid4())), 
                                duration_ms, 
                                stored_context
                            )
                            
                            print(f"🔄 STREAMING_LOG: Logging user input and assistant response ({len(full_text_response)} chars)")
                            
                            # Send both logs
                            litellm_logger._queue_log(user_log)
                            litellm_logger._queue_log(assistant_log)
                        else:
                            print(f"⚠️  DUPLICATE_PREVENTION: Callback already triggered for this request")
                except Exception as callback_err:
                    print(f"⚠️  STREAMING_LOG_ERROR: Failed to log streaming response: {callback_err}")
                
                # Track usage AFTER successful completion
                try:
                    pricing_service.track_usage(user['id'], "ai-requests", 1)
                    print(f"AUTUMN_USAGE_TRACKED: Tracked 1 AI request for user {user['id']}")
                except Exception as track_err:
                    print(f"⚠️  AUTUMN_TRACK_ERROR: Failed to track usage: {track_err}")
                    # Don't fail the request if tracking fails
                
                print(f"BACKEND_DEBUG: About to send final status: completed")
                yield json.dumps({"status": "completed"}) + '\n'
                print(f"BACKEND_DEBUG: Final status completed sent")
                break # Exit loop
        
        except Exception as e:
            print(f"ERROR: Exception in stream generation: {e}")
            
            # Send detailed error information for better frontend handling
            error_type = e.__class__.__name__
            error_str = str(e)
            
            # Categorize errors for better user experience
            if "rate limit" in error_str.lower() or "RateLimitError" in error_type:
                error_category = "rate_limit"
                user_message = "Rate limit exceeded. The system will automatically retry with a different provider."
            elif "tool_calls" in error_str.lower() and "tool_call_id" in error_str.lower():
                error_category = "tool_call_error"
                user_message = "Tool execution was interrupted. Your conversation has been recovered and you can continue."
            elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                error_category = "connection_error"
                user_message = "Connection issue detected. Please check your internet connection and try again."
            elif "overloaded" in error_str.lower() or "unavailable" in error_str.lower():
                error_category = "service_overloaded"
                user_message = "AI service is currently overloaded. Please try again in a moment."
            else:
                error_category = "unknown_error"
                user_message = "An unexpected error occurred. Please try your request again."
            
            yield json.dumps({
                "error": error_str, 
                "status": "error",
                "error_category": error_category,
                "user_message": user_message,
                "error_type": error_type,
                "recoverable": error_category in ["rate_limit", "tool_call_error", "connection_error", "service_overloaded"]
            }) + '\n'
        
        finally:
            # Clean up this request from active requests
            with stop_requests_lock:
                if request_id in ACTIVE_REQUESTS:
                    del ACTIVE_REQUESTS[request_id]
                    print(f"CLEANUP: Removed request {request_id} from active requests")
            try:
                log_event('chat_end', {'request_id': request_id})
            except Exception:
                pass

    # CRITICAL GCP FIX: Enhanced streaming response handling for Cloud Run
    # GCP Cloud Run has different buffering behavior than local Flask
    def generate_with_gcp_optimization():
        """Enhanced stream generator with GCP Cloud Run optimizations"""
        try:
            for chunk in generate_stream():
                # CRITICAL: Force immediate flushing for GCP Cloud Run
                # Without this, responses can buffer and cause apparent "hangs"
                if chunk:
                    yield chunk
                    # Force flush every few chunks to prevent buffering issues in GCP
                    if hasattr(chunk, '__len__') and len(chunk) > 100:
                        yield ""  # Force Flask to flush the buffer
        except Exception as stream_error:
            print(f"STREAM_GENERATION_ERROR: {stream_error}")
            # Send error as final stream chunk instead of raising
            error_chunk = json.dumps({
                "status": "stream_error",
                "error": f"Stream generation failed: {str(stream_error)}",
                "error_type": "stream_generation_failure",
                "recoverable": True
            }) + '\n'
            yield error_chunk

    try:
        # CRITICAL: Add explicit headers for GCP Cloud Run streaming
        response = Response(
            stream_with_context(generate_with_gcp_optimization()),
            mimetype='application/x-ndjson',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )
        return response
    except Exception as e:
        print(f"STREAMING_RESPONSE_ERROR: {e}")
        # Enhanced fallback with proper error categorization
        error_type = "streaming_setup_failure"
        if "timeout" in str(e).lower():
            error_type = "streaming_timeout"
        elif "memory" in str(e).lower():
            error_type = "streaming_memory_issue"
        elif "connection" in str(e).lower():
            error_type = "streaming_connection_error"
            
        return jsonify({
            "error": f"Streaming failed: {str(e)}",
            "suggestion": "Try starting a new conversation or using shorter responses",
            "status": "error",
            "error_type": error_type,
            "recoverable": True,
            "timestamp": int(time.time())
        }), 500

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
        openai_fallback_friendly, _ = _get_openai_preferred_model()

        while True:
            try:
                model_id = get_validated_chat_model(model_for_script)
                # Use the original model name to preserve thinking mode selection
                model_friendly = model_for_script if model_for_script in ALLOWED_CHAT_MODELS else DEFAULT_MODEL
                reasoning_params = _get_reasoning_params(model_friendly, model_id)
                
                completion_params = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": script_prompt}],
                    "timeout": 120,  # CRITICAL: 2 minute timeout for script generation
                    "request_timeout": 120  # CRITICAL: Explicit request timeout for GCP Cloud Run
                }
                completion_params.update(reasoning_params)
                
                try:
                    response = completion(**completion_params)
                except Exception as e_comp:
                    err_msg = str(e_comp).lower()
                    if reasoning_params and ("reasoning" in err_msg or "thinking" in err_msg or "unsupported" in err_msg or "invalid" in err_msg):
                        print("GENERATE_SCRIPT: Provider rejected thinking params; retrying without reasoning/thinking")
                        completion_params_no_reason = dict(completion_params)
                        completion_params_no_reason.pop("reasoning_effort", None)
                        completion_params_no_reason.pop("thinking", None)
                        completion_params_no_reason.pop("reasoning", None)
                        response = completion(**completion_params_no_reason)
                    else:
                        raise
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

                # After 5 retries, try preferred OpenAI fallback
                if attempts >= max_attempts and model_for_script != openai_fallback_friendly:
                    print(f"GENERATE_SCRIPT: Switching to {openai_fallback_friendly} after {max_attempts} failed attempts")
                    model_for_script = openai_fallback_friendly
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

def _analyze_tscn_embedded_script_context(file_content: str, edit_prompt: str) -> str:
    """
    Analyze .tscn files with embedded GDScript to provide proper escaping guidance.
    This helps the AI understand how to properly format strings for .tscn files.
    """
    if not file_content.strip():
        return ""
    
    # Check if this is a .tscn file with embedded scripts
    has_embedded_script = ("[sub_resource" in file_content and 
                          "type=\"GDScript\"" in file_content and 
                          "script/source =" in file_content)
    
    if not has_embedded_script:
        return ""
    
    # Extract embedded script content to analyze escaping patterns
    import re
    script_pattern = r'script/source = "([^"]*(?:\\.[^"]*)*)"'
    matches = re.findall(script_pattern, file_content, re.DOTALL | re.MULTILINE)
    
    rules = [
        "=== .TSCN EMBEDDED SCRIPT ESCAPING RULES ===",
        "- You are editing a .tscn scene file with embedded GDScript",
        "- Strings in the embedded script MUST be properly escaped for .tscn format",
        "- Use \\\" instead of \" for quotation marks inside print statements",
        "- Use \\\\ instead of \\ for backslashes",
        "- Newlines should be literal \\n in the script/source string",
        "- Example: print(\"Hello\") should become print(\\\"Hello\\\") in .tscn format",
        "- Example: print(\"🌬️ Smoke emission stopped\") should become print(\\\"🌬️ Smoke emission stopped\\\") in .tscn",
        "- The entire script content is stored as a single escaped string",
        ""
    ]
    
    if matches:
        # Analyze existing escaping patterns in the file
        for i, script_content in enumerate(matches[:2]):  # Only analyze first 2 scripts
            rules.append(f"- Existing embedded script {i+1} escaping pattern detected:")
            
            # Show how quotes are currently escaped
            if "\\\"" in script_content:
                rules.append("  • Contains properly escaped quotes (\\\")")
            
            # Show how backslashes are escaped
            if "\\\\" in script_content:
                rules.append("  • Contains properly escaped backslashes (\\\\)")
            
            # Show a snippet of the escaping pattern
            if len(script_content) > 0:
                preview = script_content[:100].replace('\n', '\\n')
                rules.append(f"  • Preview: {preview}...")
    
    rules.extend([
        "",
        "CRITICAL: When editing this file, maintain the EXACT same escaping pattern!",
        "If you see \\\" in the original, keep using \\\" in your edits.",
        "If you see \\\\ in the original, keep using \\\\ in your edits."
    ])
    
    return '\n'.join(rules)

def _validate_tscn_content(file_content: str, operation: str) -> dict:
    """
    Validate .tscn file content for common issues that cause tool failures.
    Returns validation results to help debug .tscn editing problems.
    """
    validation = {
        "is_tscn": file_content.strip().startswith("[gd_scene") or file_content.strip().startswith("[gd_resource"),
        "has_embedded_script": False,
        "embedded_script_count": 0,
        "escaping_issues": [],
        "recommendations": []
    }
    
    if not validation["is_tscn"]:
        return validation
    
    # Check for embedded scripts
    if "[sub_resource" in file_content and "type=\"GDScript\"" in file_content:
        validation["has_embedded_script"] = True
        
        # Count embedded scripts
        import re
        script_matches = re.findall(r'\[sub_resource[^]]*type="GDScript"[^]]*\]', file_content)
        validation["embedded_script_count"] = len(script_matches)
        
        # Extract script content to check for escaping issues
        script_content_pattern = r'script/source = "([^"]*(?:\\.[^"]*)*)"'
        script_contents = re.findall(script_content_pattern, file_content, re.DOTALL)
        
        for i, script_content in enumerate(script_contents):
            # Check for common escaping issues
            if 'print("' in script_content and 'print(\\"' not in script_content:
                validation["escaping_issues"].append(f"Script {i+1}: Unescaped quotes in print statement")
                validation["recommendations"].append(f"Script {i+1}: Change print(\"...\") to print(\\\"...\\\")")
            
            if '\\' in script_content and '\\\\' not in script_content:
                # Single backslashes that should be escaped
                validation["escaping_issues"].append(f"Script {i+1}: Single backslashes detected - may need escaping")
                validation["recommendations"].append(f"Script {i+1}: Ensure backslashes are properly escaped as \\\\")
    
    return validation

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
        
        # Analyze file-specific context FIRST (before building prompts)
        indentation_context = ""
        tscn_escaping_context = ""
        
        if path and path.endswith('.gd'):
            indentation_context = _analyze_gdscript_indentation(file_content, prompt)
            print(f"GDSCRIPT INDENTATION ANALYSIS for {path}:")
            print(indentation_context)
        elif path and path.endswith('.tscn'):
            tscn_escaping_context = _analyze_tscn_embedded_script_context(file_content, prompt)
            if tscn_escaping_context:
                print(f"TSCN ESCAPING ANALYSIS for {path}:")
                print(tscn_escaping_context)
                
                # Also run validation to catch potential issues
                validation = _validate_tscn_content(file_content, "edit")
                if validation["escaping_issues"]:
                    print(f"TSCN VALIDATION WARNINGS: {len(validation['escaping_issues'])} issues detected:")
                    for issue in validation["escaping_issues"]:
                        print(f"  - {issue}")
                    for rec in validation["recommendations"]:
                        print(f"  → {rec}")
        
        # Build a simple, clear prompt
        if is_range:
            # For range edits, provide context about the specific lines
            context_reminders = []
            if indentation_context:
                context_reminders.append(f"CRITICAL INDENTATION RULES:\n{indentation_context}")
            if tscn_escaping_context:
                context_reminders.append(f"CRITICAL ESCAPING RULES:\n{tscn_escaping_context}")
            
            combined_context = "\n\n".join(context_reminders) if context_reminders else ""
            indentation_reminder = f"\n\n{combined_context}" if combined_context else ""
            
            # Check if this is expanded context (context expansion detection)
            file_lines = file_content.split('\n')
            actual_range_size = end_line - start_line + 1
            provided_lines = len(file_lines)
            is_expanded_context = provided_lines > actual_range_size
            
            if is_expanded_context:
                # Calculate exact position of target lines within the expanded segment
                # Frontend uses: context_start = MAX(0, range_start - 1 - context_lines) with context_lines=10
                # From logs: "Expanded context from lines 44-64 (original range: 54-54)"
                # So: context_start = MAX(0, 54 - 1 - 10) = 43 (0-based), first line = 44 (1-based)
                # Target line 54 is at position: 54 - 44 + 1 = 11 in the segment
                
                # Calculate based on frontend's exact algorithm
                context_lines_used = 10  # From frontend constant
                original_context_start_0based = max(0, start_line - 1 - context_lines_used)
                first_line_in_segment_1based = original_context_start_0based + 1
                
                # Target position within the segment
                target_start_in_segment = start_line - first_line_in_segment_1based + 1
                target_end_in_segment = end_line - first_line_in_segment_1based + 1
                
                print(f"CONTEXT_CALC: first_line_in_segment={first_line_in_segment_1based}, target_in_segment={target_start_in_segment}-{target_end_in_segment}")
                
                # Validate calculation
                if target_start_in_segment <= 0 or target_end_in_segment > provided_lines:
                    print(f"CONTEXT_CALC: Invalid target calculation, falling back to simple detection")
                    # Simple fallback: assume target is in the middle
                    target_start_in_segment = (provided_lines // 2)
                    target_end_in_segment = target_start_in_segment
                
                full_prompt = (
                    f"Task: {prompt}\n\n"
                    f"IMPORTANT: You received {provided_lines} lines for context, but your edit target is ONLY lines {start_line}-{end_line} from the original file.\n\n"
                    f"The expanded segment contains:\n"
                    f"- Lines 1-{target_start_in_segment-1}: Context before target (if any)\n"
                    f"- Lines {target_start_in_segment}-{target_end_in_segment}: Your actual edit target\n"  
                    f"- Lines {target_end_in_segment+1}-{provided_lines}: Context after target (if any)\n\n"
                    f"STRICT INSTRUCTIONS:\n"
                    f"1. Apply your edit ONLY to the target lines ({target_start_in_segment}-{target_end_in_segment})\n"
                    f"2. Copy ALL other lines exactly as provided - including every space, tab, and character\n"
                    f"3. Do not add any markers, arrows, or annotations to the output\n"
                    f"4. Do not change indentation of any non-target lines{indentation_reminder}\n\n"
                    f"Code segment:\n"
                    f"{file_content}\n\n"
                    f"Return the complete {provided_lines}-line segment with your edit applied only to the target lines."
                )
            else:
                # Original logic for non-expanded ranges
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
                context_reminders = []
                if indentation_context:
                    context_reminders.append(f"CRITICAL INDENTATION RULES:\n{indentation_context}")
                if tscn_escaping_context:
                    context_reminders.append(f"CRITICAL ESCAPING RULES:\n{tscn_escaping_context}")
                
                combined_context = "\n\n".join(context_reminders) if context_reminders else ""
                context_reminder = f"\n\n{combined_context}" if combined_context else ""
                
                full_prompt = (
                    f"Task: {prompt}\n\n"
                    f"Current file content:\n"
                    f"{file_content}\n\n"
                    f"IMPORTANT: Reply with the COMPLETE edited file content. You must include ALL original lines plus your changes.{context_reminder}\n\n"
                    "Output format: Just the complete file content, no explanations, no markdown, no truncation."
                )

        # Indentation context already analyzed above
        
        # OPTIMIZATION: Add temperature and timeout settings
        # Use claude-4 by default for apply_edit as it's often faster
        model_for_edit = data.get('model', 'claude-4')
        openai_fallback_friendly, _ = _get_openai_preferred_model()
        
        # Add retry logic for apply_edit as well
        attempts = 0
        max_attempts = 5
        while True:
            try:
                # Enhanced system prompt for file-specific awareness
                system_prompt = "You are a code editor. Output only edited code, no explanations."
                
                context_requirements = []
                if indentation_context:
                    context_requirements.append(f"CRITICAL INDENTATION REQUIREMENTS:\n{indentation_context}\n\nYou MUST preserve exact indentation. Copy the whitespace characters exactly as shown. This is non-negotiable.")
                if tscn_escaping_context:
                    context_requirements.append(f"CRITICAL ESCAPING REQUIREMENTS:\n{tscn_escaping_context}\n\nYou MUST preserve exact string escaping. This is essential for .tscn file format compatibility.")
                
                if context_requirements:
                    system_prompt += "\n\n" + "\n\n".join(context_requirements)
                
                model_id = get_validated_chat_model(model_for_edit)
                # Use the original model name to preserve thinking mode selection
                model_friendly = model_for_edit if model_for_edit in ALLOWED_CHAT_MODELS else DEFAULT_MODEL
                reasoning_params = _get_reasoning_params(model_friendly, model_id)
                print(f"APPLY_EDIT: Using model '{model_for_edit}' -> friendly_name '{model_friendly}', thinking_mode: {_is_thinking_mode(model_friendly)}, reasoning_params: {reasoning_params}")
                
                completion_params = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    "max_tokens": 16000,  # Higher limit to ensure complete file generation
                    "timeout": 180,  # CRITICAL: 3 minute timeout for code editing
                    "request_timeout": 180  # CRITICAL: Explicit request timeout for GCP Cloud Run
                }
                
                # Apply reasoning params first, then set temperature if not in thinking mode
                completion_params.update(reasoning_params)
                if not reasoning_params:  # Only set lower temp if NOT in thinking mode
                    completion_params["temperature"] = 0.2  # Lower temperature for precise indentation (GPT-5.x min is 0.1)
                
                try:
                    response = completion(**completion_params)
                except Exception as e_comp:
                    err_msg = str(e_comp).lower()
                    if reasoning_params and ("reasoning" in err_msg or "thinking" in err_msg or "unsupported" in err_msg or "invalid" in err_msg):
                        print("APPLY_EDIT: Provider rejected thinking params; retrying without reasoning/thinking")
                        completion_params_no_reason = dict(completion_params)
                        completion_params_no_reason.pop("reasoning_effort", None)
                        completion_params_no_reason.pop("thinking", None)
                        completion_params_no_reason.pop("reasoning", None)
                        response = completion(**completion_params_no_reason)
                    else:
                        raise
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
                
                # After 5 retries, try preferred OpenAI fallback
                if attempts >= max_attempts and model_for_edit != openai_fallback_friendly:
                    print(f"APPLY_EDIT: Switching to {openai_fallback_friendly} after {max_attempts} failed attempts")
                    model_for_edit = openai_fallback_friendly
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

        # Preserve AI output exactly; only unwrap a top-level fenced code block if present
        edited_content = raw
        # Unwrap a single top-level code fence: ```<lang?>\n...\n```
        try:
            import re as _re
            m = _re.match(r"^```[a-zA-Z0-9_+\-]*\n([\s\S]*?)\n```\s*$", edited_content)
            if m:
                edited_content = m.group(1)
        except Exception:
            pass
        
        # Build the full edited content based on edit mode
        import difflib
        
        if is_range:
            # Check if this is context expansion (frontend sent expanded segment)
            file_lines_received = len(file_content.split('\n'))
            actual_range_size = end_line - start_line + 1
            is_context_expansion = file_lines_received > actual_range_size
            
            if is_context_expansion:
                # Context expansion case: AI edited the entire expanded segment
                # Don't splice - the AI result IS the full edited content for this section
                print(f"BACKEND: Context expansion detected - using AI result directly (received {file_lines_received} lines for {actual_range_size}-line target)")
                
                # The edited_content already contains the properly edited expanded segment
                # We still need to splice with the true pre/post that weren't included in expansion
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
                # Regular range edit: splice normally
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
            "inline_diff": "",  # Frontend renders its own display-only diff
            "inline_diff_data": [],
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

@app.route('/auth/initialize_autumn', methods=['POST'])
def initialize_autumn():
    """Initialize Autumn account after successful Supabase login"""
    try:
        # Robust JSON parse
        try:
            data = request.get_json()
        except Exception:
            raw = request.get_data(cache=False, as_text=True)
            filtered = ''.join(ch for ch in raw if ord(ch) >= 32 or ch in '\n\r\t')
            data = json.loads(filtered)
        
        user_id = data.get('user_id')
        user_email = data.get('user_email')
        
        if not user_id:
            return jsonify({"error": "user_id required", "success": False}), 400
        
        # Initialize Autumn account (creates customer + assigns Free plan if new)
        customer_data = pricing_service.initialize_customer(user_id, user_email)
        
        if "error" in customer_data:
            return jsonify({
                "success": False,
                "error": customer_data.get("error"),
                "customer_data": None
            }), 500
        
        return jsonify({
            "success": True,
            "customer_data": customer_data
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
            force_reindex = bool(data.get('force_reindex') or batch_info.get('force_reindex'))
            try:
                stats = cloud_vector_manager.index_files_with_content(
                    files, user['id'], project_id, max_workers=max_workers, force_reindex=force_reindex
                )
            except TypeError:
                stats = cloud_vector_manager.index_files_with_content(
                    files, user['id'], project_id, force_reindex=force_reindex
                )
            
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

        arguments = {
            "query": query,
            "project_root": project_root,
            "project_id": project_id,
            "max_results": data.get('max_results', 5),
            "include_graph": data.get('include_graph', False),
            "graph_preview": data.get('graph_preview', False),
            "trace_dependencies": data.get('trace_dependencies', False),
            "search_mode": data.get('search_mode', 'semantic'),
            "graph_depth": data.get('graph_depth', 1),
            "graph_edge_kinds": data.get('graph_edge_kinds') or [],
        }
        result = search_across_project_internal(arguments, current_user=user)
        return jsonify(result)

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
        
        # Rebuild MODEL_MAP with thinking variants
        base_models_with_cerebras = BASE_MODEL_MAP.copy()
        base_models_with_cerebras.update(fresh_cerebras)
        MODEL_MAP = _create_thinking_variants(base_models_with_cerebras)
        
        # Update allowed models
        global ALLOWED_CHAT_MODELS
        ALLOWED_CHAT_MODELS = set(MODEL_MAP.keys())
        
        models = []
        for friendly_name, model_id in MODEL_MAP.items():
            models.append({
                "id": friendly_name,
                "name": friendly_name,
                "provider": model_id.split('/')[0] if '/' in model_id else 'unknown',
                "model_id": model_id,
                "supports_thinking": _is_thinking_mode(friendly_name),
                "is_thinking_variant": "(thinking)" in friendly_name.lower(),
                "is_fast_variant": False  # No more (fast) variants
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

@app.route('/count_tokens', methods=['POST'])
def count_tokens():
    """Count tokens for a conversation using LiteLLM"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        messages = data.get('messages', [])
        model = data.get('model', 'openai/gpt-4o')
        
        if not messages:
            return jsonify({
                'token_count': 0,
                'model_limit': _get_model_token_limit(model),
                'percentage': 0.0
            })
        
        # Use existing token counting function
        token_count = _count_tokens_for_messages(messages, model)
        model_limit = _get_model_token_limit(model)
        percentage = (token_count / model_limit * 100) if model_limit > 0 else 0
        
        return jsonify({
            'token_count': token_count,
            'model_limit': model_limit,
            'percentage': percentage
        })
    except Exception as e:
        print(f"TOKEN_COUNT_ERROR: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/crash_report/test_supabase', methods=['GET'])
def test_crash_supabase():
    """Test Supabase crash reporting connection"""
    if not SUPABASE_CRASH_REPORTING_ENABLED:
        return jsonify({
            "success": False,
            "enabled": False,
            "message": "Supabase crash reporting not configured",
            "instructions": "Add SUPABASE_URL and SUPABASE_SERVICE_KEY to backend/.env"
        })
    
    try:
        # Test insert a dummy crash report
        test_data = {
            'report_id': f'test_{int(time.time())}',
            'platform': 'test',
            'engine_version': '4.4.dev (test)',
            'project_name': 'Connection Test',
            'user_id': 'test',
            'machine_id': 'test',
            'crash_dump': 'Test crash dump for connection verification',
            'timestamp_reported': int(time.time())
        }
        
        # Use Supabase REST API directly (same as logging_server.py)
        supabase_rest_url = f"{SUPABASE_URL}/rest/v1/{CRASH_REPORTS_TABLE}"
        headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        
        response = requests.post(supabase_rest_url, json=test_data, headers=headers, timeout=5)
        
        if response.status_code == 201:
            return jsonify({
                "success": True,
                "enabled": True,
                "message": "✅ Supabase connection working! Test crash report stored.",
                "table": CRASH_REPORTS_TABLE,
                "test_report_id": test_data['report_id'],
                "supabase_url": SUPABASE_URL
            })
        else:
            return jsonify({
                "success": False,
                "enabled": True,
                "error": f"HTTP {response.status_code}: {response.text[:500]}",
                "message": "Supabase connected but insert failed. Check table exists and RLS policies.",
                "table": CRASH_REPORTS_TABLE
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "enabled": True,
            "error": str(e),
            "message": "Failed to connect to Supabase. Check SUPABASE_URL and SUPABASE_SERVICE_KEY."
        }), 500

@app.route('/crash_report', methods=['POST'])
def crash_report():
    """Receive crash reports from Godot clients for investigation"""
    try:
        data = request.get_json() or {}
        
        # Extract crash report data
        crash_dump = data.get('crash_dump', '')
        platform = data.get('platform', 'unknown')
        engine_version = data.get('engine_version', 'unknown')
        user_id = data.get('user_id', 'anonymous')
        machine_id = data.get('machine_id', 'unknown')
        project_name = data.get('project_name', 'unknown')
        timestamp = data.get('timestamp', int(time.time()))
        
        if not crash_dump:
            return jsonify({"success": False, "error": "No crash_dump provided"}), 400
        
        # Log the crash report
        print("=" * 80)
        print("🚨 CRASH REPORT RECEIVED")
        print(f"Platform: {platform}")
        print(f"Engine Version: {engine_version}")
        print(f"Project: {project_name}")
        print(f"User: {user_id}")
        print(f"Machine: {machine_id}")
        print(f"Timestamp: {timestamp}")
        print("=" * 80)
        print(crash_dump)
        print("=" * 80)
        
        # Log to structured logging if enabled
        log_event('crash_report_received', {
            'platform': platform,
            'engine_version': engine_version,
            'project': project_name,
            'user_h': _anon(user_id),
            'machine_h': _anon(machine_id),
            'crash_dump_size': len(crash_dump)
        }, 'ERROR')
        
        # Generate report ID
        report_id = f"crash_{timestamp}_{machine_id[:8]}"
        
        # Store in Supabase if enabled (use direct REST API like logging_server.py)
        if SUPABASE_CRASH_REPORTING_ENABLED:
            try:
                crash_data = {
                    'report_id': report_id,
                    'platform': platform,
                    'engine_version': engine_version,
                    'project_name': project_name,
                    'user_id': user_id,
                    'machine_id': machine_id,
                    'crash_dump': crash_dump,
                    'timestamp_reported': timestamp,
                    'user_agent': request.headers.get('User-Agent', 'Unknown')
                }
                
                # Use Supabase REST API directly (same as logging_server.py)
                supabase_rest_url = f"{SUPABASE_URL}/rest/v1/{CRASH_REPORTS_TABLE}"
                headers = {
                    'apikey': SUPABASE_SERVICE_KEY,
                    'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=minimal'
                }
                
                response = requests.post(supabase_rest_url, json=crash_data, headers=headers, timeout=5)
                
                if response.status_code == 201:
                    print(f"SUPABASE_CRASH: ✅ Stored crash report {report_id} to database")
                    return jsonify({
                        "success": True,
                        "message": "Crash report received and stored in database",
                        "report_id": report_id,
                        "stored_in_database": True
                    })
                else:
                    print(f"SUPABASE_CRASH: ⚠️ Failed to store (HTTP {response.status_code}): {response.text[:200]}")
                    return jsonify({
                        "success": True,
                        "message": "Crash report received and logged (database insert failed)",
                        "report_id": report_id,
                        "stored_in_database": False,
                        "storage_error": f"HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                print(f"SUPABASE_CRASH_ERROR: Failed to store crash report: {e}")
                # Still return success even if DB storage fails (at least we logged it)
                return jsonify({
                    "success": True,
                    "message": "Crash report received and logged (database storage failed)",
                    "report_id": report_id,
                    "stored_in_database": False,
                    "storage_error": str(e)
                })
        else:
            # No database configured, just log it
            return jsonify({
                "success": True,
                "message": "Crash report received and logged",
                "report_id": report_id,
                "stored_in_database": False,
                "note": "Configure SUPABASE_URL and SUPABASE_SERVICE_KEY to enable database storage"
            })
        
    except Exception as e:
        print(f"CRASH_REPORT_ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with GCP hang detection"""
    version_info = version_checker.get_version_info()
    
    # CRITICAL: Validate tools array integrity to detect corruption
    tools_valid = True
    tools_error = None
    try:
        if not godot_tools or not isinstance(godot_tools, list):
            tools_valid = False
            tools_error = "godot_tools is not a list"
        elif len(godot_tools) == 0:
            tools_valid = False
            tools_error = "godot_tools is empty"
        else:
            # Check first tool has required structure
            first_tool = godot_tools[0]
            if not isinstance(first_tool, dict):
                tools_valid = False
                tools_error = "godot_tools[0] is not a dict"
            elif "type" not in first_tool:
                tools_valid = False
                tools_error = "godot_tools[0] missing 'type' field"
            elif first_tool.get("type") != "function":
                tools_valid = False
                tools_error = f"godot_tools[0].type is '{first_tool.get('type')}' not 'function'"
    except Exception as e:
        tools_valid = False
        tools_error = f"Exception validating tools: {str(e)}"
    
    # CRITICAL: GCP Cloud Run environment detection and resource monitoring
    is_gcp_cloud = bool(os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('CLOUD_RUN_JOB'))
    
    # Check for active requests that might be hanging
    active_request_count = len(ACTIVE_REQUESTS)
    oldest_request_age = 0
    if ACTIVE_REQUESTS:
        oldest_timestamp = min(data["timestamp"] for data in ACTIVE_REQUESTS.values())
        oldest_request_age = int(time.time() - oldest_timestamp)
    
    # Memory usage check (basic)
    import sys
    memory_mb = sys.getsizeof(locals()) / (1024 * 1024)  # Rough approximation
    
    response = {
        "status": "healthy" if tools_valid else "degraded", 
        "service": "orca-engine-ai-service",
        "providers": ["openai", "anthropic", "google"],
        "available_models": list(MODEL_MAP.keys()),
        "version": auto_update_manager.current_version,
        "version_info": version_info,
        "tools_valid": tools_valid,
        "tools_count": len(godot_tools) if isinstance(godot_tools, list) else 0,
        # CRITICAL: GCP hang detection metrics
        "gcp_environment": is_gcp_cloud,
        "active_requests": active_request_count,
        "oldest_request_age_seconds": oldest_request_age,
        "memory_approx_mb": memory_mb,
        "hang_protection_enabled": True,  # Indicates timeout fixes are active
        "streaming_optimized": True,  # Indicates GCP streaming fixes are active
        "timeout_fixes_version": "2025-10-28"  # Track when fixes were applied
    }
    
    if not tools_valid:
        response["tools_error"] = tools_error
        print(f"⚠️ HEALTH_CHECK: Tools array corrupted: {tools_error}")
    
    # CRITICAL: Log health check in GCP for hang monitoring
    if is_gcp_cloud:
        print(f"GCP_HEALTH_CHECK: status={response['status']}, active_requests={active_request_count}, oldest_age={oldest_request_age}s")
        if oldest_request_age > 300:  # 5+ minute old requests
            print(f"GCP_HANG_WARNING: 🚨 Oldest active request is {oldest_request_age}s old - possible hang!")
    
    return jsonify(response)

@app.route('/version', methods=['GET'])
def get_version_info():
    """Get detailed version information for compatibility checking"""
    version_info = version_checker.get_version_info()
    return jsonify({
        "success": True,
        "orca_version": auto_update_manager.current_version,
        **version_info
    })

@app.route('/version/check', methods=['POST'])
def check_version_compatibility_endpoint():
    """Explicit endpoint for checking version compatibility"""
    try:
        data = request.json or {}
        frontend_version = data.get('frontend_version') or request.headers.get('X-Frontend-Version')
        frontend_api_version = data.get('frontend_api_version') or request.headers.get('X-Frontend-API-Version')
        
        if not frontend_version or not frontend_api_version:
            return jsonify({
                "success": False,
                "error": "frontend_version and frontend_api_version are required"
            }), 400
        
        compatibility_status = version_checker.get_compatibility_status(frontend_version, frontend_api_version)
        
        return jsonify({
            "success": True,
            "compatibility_check": compatibility_status
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Version compatibility check failed: {str(e)}"
        }), 500

# --- Auto-Update Endpoints ---

@app.route('/update/check', methods=['GET', 'POST'])
def check_for_updates():
    """Check for available updates"""
    try:
        data = request.get_json() if request.method == 'POST' else {}
        platform = data.get('platform') if data else request.args.get('platform')
        force = data.get('force', False) if data else request.args.get('force', 'false').lower() == 'true'
        
        update_info = auto_update_manager.check_for_updates(force=force, platform=platform)
        
        if update_info:
            return jsonify({
                'success': True,
                'update_available': True,
                'update_info': {
                    'version': update_info.version,
                    'download_url': update_info.download_url,
                    'file_size': update_info.file_size,
                    'file_size_mb': round(update_info.file_size / (1024 * 1024), 1),
                    'release_notes': update_info.release_notes,
                    'published_at': update_info.published_at,
                    'is_critical': update_info.is_critical
                },
                'current_version': auto_update_manager.current_version
            })
        else:
            return jsonify({
                'success': True,
                'update_available': False,
                'current_version': auto_update_manager.current_version,
                'message': 'You have the latest version'
            })
            
    except Exception as e:
        print(f"UPDATE_CHECK_ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/status', methods=['GET'])
def get_update_status():
    """Get current update system status"""
    try:
        status = auto_update_manager.get_update_status()
        return jsonify({
            'success': True,
            **status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/download', methods=['POST'])
def download_update():
    """Download an available update"""
    try:
        data = request.get_json() or {}
        version = data.get('version')
        download_path = data.get('download_path')
        
        if not auto_update_manager.cached_update_info:
            return jsonify({
                'success': False,
                'error': 'No update available to download'
            }), 400
        
        update_info = auto_update_manager.cached_update_info
        
        # Verify version matches if specified
        if version and version != update_info.version:
            return jsonify({
                'success': False,
                'error': f'Version mismatch: requested {version}, available {update_info.version}'
            }), 400
        
        # Download the update
        result = auto_update_manager.download_update(update_info, download_path)
        
        # CRITICAL FIX: Mark version as installed when download completes successfully
        if result.get('success', False):
            auto_update_manager.mark_version_installed(update_info.version)
            print(f"UPDATE_DOWNLOAD: Marked version {update_info.version} as installed after successful download")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"UPDATE_DOWNLOAD_ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/install', methods=['POST'])
def install_update():
    """Install a downloaded update"""
    try:
        data = request.get_json() or {}
        download_path = data.get('download_path')
        restart_app = data.get('restart_app', True)
        version = data.get('version')  # Version being installed
        
        if not download_path or not os.path.exists(download_path):
            return jsonify({
                'success': False,
                'error': 'Download path not provided or file not found'
            }), 400
        
        # Schedule installation
        result = auto_update_manager.schedule_install(download_path, restart_app)
        
        # CRITICAL FIX: Mark version as installed when installation starts
        if result.get('success', False) and version:
            auto_update_manager.mark_version_installed(version)
            print(f"UPDATE_INSTALL: Marked version {version} as installed")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"UPDATE_INSTALL_ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/mark_installed', methods=['POST'])
def mark_version_installed():
    """Manually mark a version as installed (useful for manual updates)"""
    try:
        data = request.get_json() or {}
        version = data.get('version')
        
        if not version:
            return jsonify({
                'success': False,
                'error': 'Version is required'
            }), 400
        
        auto_update_manager.mark_version_installed(version)
        
        return jsonify({
            'success': True,
            'message': f'Version {version} marked as installed',
            'version': version
        })
        
    except Exception as e:
        print(f"UPDATE_MARK_ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/refresh_version', methods=['POST'])
def refresh_version():
    """Refresh current version detection (for testing)"""
    try:
        global auto_update_manager
        
        old_version = auto_update_manager.current_version
        new_version = auto_update_manager._get_current_version()
        auto_update_manager.current_version = new_version
        
        return jsonify({
            'success': True,
            'old_version': old_version,
            'new_version': new_version,
            'message': f'Version refreshed: {old_version} -> {new_version}'
        })
        
    except Exception as e:
        print(f"VERSION_REFRESH_ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/notes/<version>', methods=['GET'])
def get_release_notes(version):
    """Get release notes for a specific version"""
    try:
        notes = auto_update_manager.get_release_notes(version)
        return jsonify({
            'success': True,
            'version': version,
            'release_notes': notes
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update/webhook', methods=['POST'])
def update_webhook():
    """GitHub webhook endpoint to trigger immediate update checks"""
    try:
        # Verify this is a release event
        event_type = request.headers.get('X-GitHub-Event')
        if event_type != 'release':
            return jsonify({'message': 'Not a release event'}), 200
        
        data = request.get_json() or {}
        action = data.get('action')
        
        if action in ['published', 'released']:
            release = data.get('release', {})
            version = release.get('tag_name', '').lstrip('v')
            
            print(f"AUTO_UPDATE: GitHub webhook - new release v{version}")
            
            # Clear cache to force fresh check
            auto_update_manager.cached_update_info = None
            auto_update_manager.last_check = 0
            
            # Trigger immediate check
            update_info = auto_update_manager.force_check_now()
            
            return jsonify({
                'success': True,
                'message': f'Update check triggered for v{version}',
                'update_available': update_info is not None
            })
        else:
            return jsonify({'message': f'Ignored action: {action}'}), 200
            
    except Exception as e:
        print(f"UPDATE_WEBHOOK_ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

def search_across_godot_docs_internal(arguments: dict, use_enhanced: bool = True) -> dict:
    """Search the production Godot docs using enhanced search if available"""
    try:
        query = arguments.get('query', '')
        if not query:
            return {"success": False, "error": "Query parameter is required"}
        
        # Try enhanced search first if available
        if use_enhanced:
            try:
                from enhanced_docs_search import EnhancedGodotDocsSearch
                weaviate_url = os.getenv('WEAVIATE_URL')
                weaviate_key = os.getenv('WEAVIATE_API_KEY')
                
                if weaviate_url and weaviate_key:
                    searcher = EnhancedGodotDocsSearch(weaviate_url, weaviate_key)
                    
                    # Determine search mode from query
                    mode = "auto"
                    if "how" in query.lower() or "tutorial" in query.lower():
                        mode = "semantic"
                    elif any(keyword in query.lower() for keyword in ["func", "class", "signal", "property", "constant", "enum", "mode_"]):
                        mode = "keyword"
                    
                    # Boost search for specific class queries
                    class_filter = None
                    if "input" in query.lower() and ("mouse" in query.lower() or "capture" in query.lower()):
                        class_filter = "Input"
                    elif "characterbody3d" in query.lower().replace(" ", ""):
                        class_filter = "CharacterBody3D"
                    
                    # Extract filters from query
                    section_filter = None
                    if "tutorial" in query.lower():
                        section_filter = "tutorials"
                    elif "class" in query.lower() or "reference" in query.lower():
                        section_filter = "classes"
                        
                    result = searcher.search_godot_docs_enhanced(
                        query=query,
                        mode=mode,
                        section_filter=section_filter,
                        max_results=arguments.get('max_results', 5)
                    )
                    
                    if result.get("results"):
                        return {"success": True, "results": result["results"], "source": "enhanced"}
            except Exception as e:
                print(f"Enhanced search failed, falling back: {e}")
        
        max_results = int(arguments.get('max_results', 5))
        section_filter = arguments.get('section_filter')
        class_filter = arguments.get('class_filter')
        
        print(f"DOCS_SEARCH: Searching production docs for: '{query}'")
        
        # Connect to our production docs collection
        if not (WEAVIATE_URL and WEAVIATE_API_KEY and api_key):
            return {"success": False, "error": "Docs search unavailable - missing configuration"}
        
        try:
            import weaviate
            import weaviate.classes as wvc
            
            os.environ['OPENAI_API_KEY'] = api_key
            # CRITICAL: Add timeout protection for Weaviate connection in GCP
            import weaviate.connect
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
                headers={'X-OpenAI-Api-Key': api_key},
                additional_config=weaviate.connect.ConnectionConfig(
                    session_pool_connections=20,
                    session_pool_maxsize=200,
                    session_pool_max_retries=3
                )
            )
            # Set client timeout to prevent hanging
            if hasattr(client, '_timeout'):
                client._timeout = 60  # 1 minute timeout for database operations
            
            # Use our production collection
            collection_name = "GodotDocs_Production"
            if not client.collections.exists(collection_name):
                return {
                    "success": False, 
                    "error": f"Production docs not indexed yet. Run production_docs_indexer.py first."
                }
            
            collection = client.collections.get(collection_name)
            
            # Generate query embedding manually (same as production indexer)
            openai_client = openai.OpenAI(api_key=api_key)
            query_response = openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"  # Same model as indexing
            )
            query_vector = query_response.data[0].embedding
            
            # Search using manual embeddings (old client - no where filter support)
            search_results = collection.query.near_vector(
                near_vector=query_vector,
                limit=max_results * 2,  # Get more results for manual filtering
                return_metadata=["distance"]
            )
            
            # Manual filtering if needed (since old client doesn't support where in near_vector)
            filtered_objects = search_results.objects
            if section_filter or class_filter:
                filtered_objects = []
                for obj in search_results.objects:
                    props = obj.properties
                    
                    # Apply section filter
                    if section_filter and props.get('section') != section_filter:
                        continue
                    
                    # Apply class filter
                    if class_filter and props.get('class_name') != class_filter:
                        continue
                    
                    filtered_objects.append(obj)
                
                # Limit to requested results after filtering
                filtered_objects = filtered_objects[:max_results]
            else:
                # No filtering needed, just limit results
                filtered_objects = filtered_objects[:max_results]
            
            # Format results for AI tool
            formatted_results = []
            for obj in filtered_objects:
                props = obj.properties
                distance = obj.metadata.distance if obj.metadata else 1.0
                similarity = 1.0 - distance  # Convert distance to similarity
                
                formatted_results.append({
                    'title': props.get('title', 'Unknown'),
                    'snippet': props.get('content', '')[:400] + '...' if len(props.get('content', '')) > 400 else props.get('content', ''),
                    'full_content': props.get('content', ''),
                    'similarity': similarity,
                    'source': 'production_docs',
                    'file_path': f"docs/{props.get('class_name', 'unknown')}/{props.get('section', 'unknown')}",
                    'class_name': props.get('class_name', ''),
                    'section': props.get('section', ''),
                    'url': props.get('url', ''),
                    'keywords': props.get('keywords', []),
                    'search_mode': 'production_semantic'
                })
            
            client.close()
            
            print(f"DOCS_SEARCH: Found {len(formatted_results)} results with similarity scores")
            
            return {
                'success': True,
                'query': query,
                'search_mode': 'production_semantic',
                'results': formatted_results,
                'file_count': len(formatted_results),
                'collection': collection_name,
                'embedding_model': 'text-embedding-3-small'
            }
            
        except Exception as e:
            print(f"DOCS_SEARCH: Production search failed: {e}")
            
            # Fallback to old system if production search fails
            print("DOCS_SEARCH: Falling back to legacy search...")
            results = _search_godot_docs_bq(query, max_results)
            formatted = []
            for r in results:
                fp = r.get('file_path', '')
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
                    'source': 'legacy_docs',
                    'file_path': fp,
                    'search_mode': 'legacy_fallback'
                })
            
            return {
                'success': True,
                'query': query,
                'search_mode': 'legacy_fallback',
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
            
            # Some assets already include an 'addons/' root in the archive.
            # If so, extract at the project root to avoid 'addons/addons/...'.
            has_addons_root = any(
                (name.startswith('addons/') or name.startswith('addons\\')) and not name.endswith('/')
                for name in file_list
            )
            extract_base = project_path if has_addons_root else install_path
            if has_addons_root:
                print(f"ASSET_INSTALL: Detected 'addons/' root in archive; extracting to project root: {project_path}")
            else:
                print(f"ASSET_INSTALL: Extracting to install path: {install_path}")
            
            for file_info in zip_file.infolist():
                # Skip directories and hidden files
                if file_info.is_dir() or file_info.filename.startswith('.'):
                    continue
                    
                # Extract file to computed base
                extracted_path = zip_file.extract(file_info, extract_base)
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
        
        # Report the effective install directory users should look in
        final_install_root = os.path.join(project_path, 'addons') if has_addons_root else install_path
        installation_info = {
            "asset_id": asset_id,
            "asset_name": asset_name,
            "version": asset_data.get('version', '1.0'),
            "author": asset_data.get('author', 'Unknown'),
            "installed_to": final_install_root,
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
            "installation_info": installation_info,
            "installed_paths": [installation_info.get("installed_to")] if installation_info.get("installed_to") else [],
            "enabled_plugins": []
        }
        
    except Exception as e:
        import traceback
        print(f"ASSET_INSTALL_ERROR: {e}")
        print(f"ASSET_INSTALL_TRACEBACK: {traceback.format_exc()}")
        return {"success": False, "error": f"Asset installation failed: {str(e)}"}


@app.route('/search_docs', methods=['POST'])
def search_docs():
    """HTTP endpoint to search across the production Godot docs corpus.
    Uses the new production docs collection with working embeddings.
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
            'section_filter': data.get('section_filter'),
            'class_filter': data.get('class_filter')
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/docs/index', methods=['POST'])
def index_docs():
    """Trigger production Godot docs indexing"""
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    try:
        # Import and run the production indexer
        import subprocess
        import sys
        
        # Run the production indexer as a subprocess
        result = subprocess.run([
            sys.executable, 
            'production_docs_indexer.py'
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "message": "Godot docs indexed successfully",
                "output": result.stdout,
                "collection": "GodotDocs_Production"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Docs indexing failed",
                "output": result.stderr
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to trigger docs indexing: {str(e)}"
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
        
        print(f"DEBUG FORWARD: Forwarding to {url} with method {method.upper()}")
        
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

@app.route('/download/<path:filename>', methods=['GET'])
def download_3d_model_legacy(filename: str):
    """Legacy download route for compatibility with Point-E responses"""
    return _forward_to_3d_service(f'download/{filename}')

@app.route('/api/3d/models/<user_id>', methods=['GET'])
def list_user_3d_models(user_id: str):
    """List 3D models for a user"""
    print(f"DEBUG: /api/3d/models/{user_id} endpoint hit! MODEL_3D_ENABLED={MODEL_3D_ENABLED}")
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    if not MODEL_3D_ENABLED:
        return jsonify({
            'error': '3D model generation not available',
            'message': 'Service not configured'
        }), 503
    
    # Basic authorization - users can only see their own models
    if hasattr(g, 'user_id') and g.user_id != user_id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Point-E server doesn't support model listing, return mock response
    return jsonify({
        'user_id': user_id,
        'models': [],
        'message': 'Model listing not supported by Point-E service'
    })

@app.route('/api/3d/frontend-log', methods=['POST'])
def log_frontend_error():
    """
    Receive and store frontend error logs from the 3D editor plugin.
    Logs are appended to the frontend_logs column in the jobs table.
    """
    gate = verify_server_key_if_required()
    if gate is not None:
        return gate
    
    try:
        data = request.json or {}
        
        # Extract log data
        job_id = data.get('job_id', '')
        user_id = data.get('user_id', '')
        level = data.get('level', 'error')  # error, warning, info
        message = data.get('message', '')
        context = data.get('context', {})
        stack_trace = data.get('stack_trace', '')
        timestamp = data.get('timestamp', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Validate level
        if level not in ['error', 'warning', 'info', 'debug']:
            level = 'error'
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'context': context,
        }
        
        if stack_trace:
            log_entry['stack_trace'] = stack_trace
        
        # Store in Supabase if enabled
        if SUPABASE_CRASH_REPORTING_ENABLED and job_id:
            try:
                # Try both tables since job_id could be in either three_d_models or texture_jobs
                headers = {
                    'apikey': SUPABASE_SERVICE_KEY,
                    'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                }
                
                # Helper function to try updating a table
                def try_update_table(table_name, job_id_field="id"):
                    try:
                        table_url = f"{SUPABASE_URL}/rest/v1/{table_name}"
                        
                        # Get current job to retrieve existing logs
                        get_response = requests.get(
                            f"{table_url}?{job_id_field}=eq.{job_id}&select=frontend_logs",
                            headers=headers,
                            timeout=5
                        )
                        
                        if get_response.status_code != 200:
                            return False, f"HTTP {get_response.status_code}"
                        
                        jobs = get_response.json()
                        if not jobs or len(jobs) == 0:
                            return False, "Job not found"
                        
                        existing_logs = jobs[0].get('frontend_logs', [])
                        if not isinstance(existing_logs, list):
                            existing_logs = []
                        
                        # Append new log entry (keep last 100 entries to prevent bloat)
                        existing_logs.append(log_entry)
                        if len(existing_logs) > 100:
                            existing_logs = existing_logs[-100:]  # Keep last 100 entries
                        
                        # Update the job with new logs
                        update_data = {'frontend_logs': existing_logs}
                        update_response = requests.patch(
                            f"{table_url}?{job_id_field}=eq.{job_id}",
                            json=update_data,
                            headers=headers,
                            timeout=5
                        )
                        
                        if update_response.status_code in [200, 204]:
                            return True, f"Updated {table_name}"
                        else:
                            return False, f"HTTP {update_response.status_code}: {update_response.text[:200]}"
                            
                    except Exception as e:
                        return False, str(e)
                
                # Try three_d_models table first (main 3D jobs)
                success, result = try_update_table("three_d_models", "id")
                if success:
                    print(f"FRONTEND_LOG: ✅ Stored log in three_d_models for job {job_id[:8]}... ({level}): {message[:100]}")
                else:
                    # Try texture_jobs table
                    success, result = try_update_table("texture_jobs", "id")
                    if success:
                        print(f"FRONTEND_LOG: ✅ Stored log in texture_jobs for job {job_id[:8]}... ({level}): {message[:100]}")
                    else:
                        # Try with job_id field instead of id field (different table schema)
                        success, result = try_update_table("three_d_models", "job_id")
                        if success:
                            print(f"FRONTEND_LOG: ✅ Stored log in three_d_models (by job_id) for job {job_id[:8]}... ({level}): {message[:100]}")
                        else:
                            print(f"FRONTEND_LOG: ⚠️ Failed to store log in any table: {result}")
                    
            except Exception as e:
                print(f"FRONTEND_LOG_ERROR: Failed to store log in Supabase: {e}")
                # Don't fail the request if logging fails - just print error
        
        # Also print to console for immediate debugging
        log_prefix = f"[FRONTEND_LOG/{level.upper()}]"
        if job_id:
            log_prefix += f" Job:{job_id[:8]}..."
        if user_id:
            log_prefix += f" User:{user_id[:8]}..."
        
        print(f"{log_prefix} {message}")
        if context:
            print(f"  Context: {json.dumps(context, indent=2)}")
        if stack_trace:
            print(f"  Stack: {stack_trace[:500]}")  # Truncate long stack traces
        
        return jsonify({
            'success': True,
            'message': 'Log stored successfully'
        }), 200
        
    except Exception as e:
        print(f"FRONTEND_LOG_ERROR: Exception in log endpoint: {e}")
        return jsonify({
            'error': 'Failed to process log',
            'message': str(e)
        }), 500

# --- JSON RPC Router for deterministic, idempotent tools ---
@app.route('/rpc', methods=['POST'])
def json_rpc_router():
    start = time.time()
    data = request.json or {}
    tool = data.get('tool') or data.get('method') or data.get('function_name')
    args = data.get('params') or data.get('arguments') or {}

    gate = verify_server_key_if_required()
    if gate is not None:
        return gate

    def _telemetry(ok: bool, error_code: str | None):
        return {
            'tool': tool,
            'duration_ms': int((time.time() - start) * 1000),
            'ok': ok,
            'error_code': error_code
        }

    if not tool or not isinstance(args, dict):
        resp = {'ok': False, 'error_code': 'INVALID_ARGUMENT', 'error': 'Missing tool or invalid params'}
        resp['telemetry'] = _telemetry(False, 'INVALID_ARGUMENT')
        return jsonify(resp), 400

    editor_tools = {
        'resource_info', 'script_info', 'set_import_preset', 'reimport_resource', 'wait_for_import',
        'enable_plugin', 'ensure_project_settings', 'ensure_input_actions', 'ensure_autoload',
        'ensure_node', 'batch_scene_ops', 'load_and_assign_resource', 'call_node_method'
    }

    try:
        if tool in editor_tools:
            r = requests.post('http://127.0.0.1:8001', json={'function_name': tool, 'arguments': args}, timeout=30)
            r.raise_for_status()
            payload = r.json()
            ok = bool(payload.get('ok', payload.get('success', False)))
            payload['ok'] = ok
            payload['telemetry'] = _telemetry(ok, payload.get('error_code'))
            return jsonify(payload)

        if tool == 'install_godot_asset':
            result = install_godot_asset_internal(args)
            ok = result.get('success', False)
            result['ok'] = ok
            result['telemetry'] = _telemetry(ok, None if ok else 'ASSET_INSTALL_ERROR')
            return jsonify(result)

        if tool == 'recipe_install_asset_and_instance':
            asset_id = args.get('asset_id')
            instance_scene_path = args.get('instance_scene_path')
            if not asset_id or not instance_scene_path:
                resp = {'ok': False, 'error_code': 'INVALID_ARGUMENT', 'error': 'asset_id and instance_scene_path required'}
                resp['telemetry'] = _telemetry(False, 'INVALID_ARGUMENT')
                return jsonify(resp), 400
            checklist = []
            enable_plugin_flag = bool(args.get('enable_plugin', True))
            auto_configure_flag = bool(args.get('auto_configure', True))

            install_res = install_godot_asset_internal({
                'asset_id': asset_id,
                'project_path': args.get('project_path', ''),
                'install_location': args.get('install_location', 'addons/'),
                'create_backup': args.get('create_backup', True)
            })
            if not install_res.get('success'):
                resp = {'ok': False, 'error_code': 'ASSET_INSTALL_ERROR', 'error': install_res.get('error', 'install failed'), 'checklist': checklist}
                resp['telemetry'] = _telemetry(False, 'ASSET_INSTALL_ERROR')
                return jsonify(resp), 500
            checklist.append('installed')

            enabled_plugins = []
            if enable_plugin_flag and install_res.get('installation_info', {}).get('is_plugin'):
                plugin_name = os.path.basename(install_res['installation_info']['installed_to']).strip()
                try:
                    r = requests.post('http://127.0.0.1:8001', json={'function_name': 'enable_plugin', 'arguments': {'plugin_name': plugin_name}}, timeout=20)
                    if r.ok and r.json().get('ok'):
                        enabled_plugins.append(plugin_name)
                        checklist.append('plugin_enabled')
                except Exception:
                    pass

            if args.get('await_imports', True):
                try:
                    requests.post('http://127.0.0.1:8001', json={'function_name': 'wait_for_import', 'arguments': {'resource_path': instance_scene_path, 'timeout_ms': args.get('timeout_ms', 10000)}}, timeout=30)
                    checklist.append('imports_ready')
                except Exception:
                    pass

            post_config = args.get('post_config', {}) or {}
            if auto_configure_flag and post_config:
                settings = post_config.get('project_settings') or {}
                if settings:
                    requests.post('http://127.0.0.1:8001', json={'function_name': 'ensure_project_settings', 'arguments': {'settings': settings}}, timeout=20)
                    checklist.append('project_settings')
                actions = post_config.get('input_actions') or []
                if actions:
                    requests.post('http://127.0.0.1:8001', json={'function_name': 'ensure_input_actions', 'arguments': {'actions': actions}}, timeout=20)
                    checklist.append('input_actions')
                autoloads = post_config.get('autoloads') or []
                if autoloads:
                    requests.post('http://127.0.0.1:8001', json={'function_name': 'ensure_autoload', 'arguments': {'entries': autoloads}}, timeout=20)
                    checklist.append('autoloads')

            r = requests.post('http://127.0.0.1:8001', json={'function_name': 'manage_scene', 'arguments': {'operation': 'instantiate', 'scene_path': instance_scene_path, 'parent_path': args.get('target_parent', '')}}, timeout=30)
            if r.ok:
                payload = r.json()
                if payload.get('success'):
                    resp = {'ok': True, 'instance_path': payload.get('instance_path'), 'enabled_plugins': enabled_plugins, 'checklist': checklist}
                    resp['telemetry'] = _telemetry(True, None)
                    return jsonify(resp)

            resp = {'ok': True, 'enabled_plugins': enabled_plugins, 'checklist': checklist}
            resp['telemetry'] = _telemetry(True, None)
            return jsonify(resp)

        resp = {'ok': False, 'error_code': 'UNKNOWN_TOOL', 'error': f'Unknown tool {tool}'}
        resp['telemetry'] = _telemetry(False, 'UNKNOWN_TOOL')
        return jsonify(resp), 404

    except requests.exceptions.Timeout:
        resp = {'ok': False, 'error_code': 'TIMEOUT', 'error': 'Timed out calling editor'}
        resp['telemetry'] = _telemetry(False, 'TIMEOUT')
        return jsonify(resp), 504
    except requests.exceptions.RequestException as e:
        resp = {'ok': False, 'error_code': 'EDITOR_UNAVAILABLE', 'error': str(e)}
        resp['telemetry'] = _telemetry(False, 'EDITOR_UNAVAILABLE')
        return jsonify(resp), 502
    except Exception as e:
        resp = {'ok': False, 'error_code': 'INTERNAL', 'error': str(e)}
        resp['telemetry'] = _telemetry(False, 'INTERNAL')
        return jsonify(resp), 500

# ===== PRICING ENDPOINTS =====

@app.route('/pricing/customer', methods=['GET'])
def get_customer_info():
    """Get customer subscription and usage information"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        customer_data = pricing_service.get_customer_info(user['id'])
        return jsonify({"success": True, "customer": customer_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/pricing/checkout', methods=['POST'])
def create_checkout():
    """Create checkout URL for product upgrade/purchase"""
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    try:
        data = request.get_json()
        if not data or 'product_id' not in data:
            return jsonify({"success": False, "error": "product_id is required"}), 400
        
        product_id = data.get('product_id')
        checkout_data = pricing_service.get_checkout_url(user['id'], product_id)
        
        return jsonify({"success": True, "checkout": checkout_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/pricing/tiers', methods=['GET'])
def get_pricing_tiers():
    """Get available pricing tiers"""
    return jsonify({
        "success": True,
        "tiers": {
            "free": {
                "product_id": "free",
                "name": "Free",
                "price": 0,
                "requests_per_month": 200,
                "features": ["Community support"]
            },
            "pro": {
                "product_id": "pro", 
                "name": "Pro",
                "price": 20,
                "requests_per_month": 500,
                "features": ["Priority support", "Advanced features"]
            },
            "proplus": {
                "product_id": "proplus",
                "name": "Pro+", 
                "price": 60,
                "requests_per_month": 1500,
                "features": ["Priority support", "Early access", "Team features"]
            }
        }
    })

# Initialize services at module level for Gunicorn compatibility
# Print 3D service status on startup
if MODEL_3D_ENABLED:
    print(f"3D_GENERATION: Enabled, forwarding to {MODEL_3D_SERVICE_URL}")
else:
    print("3D_GENERATION: Disabled (configure MODEL_3D_* environment variables to enable)")

# Initialize auto-update system
print(f"AUTO_UPDATE: Orca Engine v{auto_update_manager.current_version} - Update system initialized")

# Print Claude configuration on startup
claude_model_id = BASE_MODEL_MAP.get("claude-4", "unknown")
if "vertex_ai" in claude_model_id:
    print(f"✅ CLAUDE_CONFIG: Using Vertex AI for Claude (leveraging your GCP credits)")
    print(f"   Model ID: {claude_model_id}")
    print(f"   Frontend shows: 'claude-4' (simplified)")
elif "anthropic" in claude_model_id:
    print(f"CLAUDE_CONFIG: Using direct Anthropic API")
    print(f"   Model ID: {claude_model_id}")
    print(f"   Frontend shows: 'claude-4' (simplified)")
else:
    print(f"CLAUDE_CONFIG: Unknown provider - Model ID: {claude_model_id}")

# Start background update checker in production
if not _dev_mode:
    auto_update_manager.start_background_checker()
    print("AUTO_UPDATE: Background checker started for production mode")
else:
    print("AUTO_UPDATE: Background checker disabled in dev mode")

# CRITICAL: Enhanced startup logging for GCP VM deployment
is_gcp_vm = bool(os.getenv('GCP_OPTIMIZED') or os.path.exists('/opt/godot-ai-backend'))
if is_gcp_vm:
    print("🏗️  GCP_VM_DEPLOYMENT: Starting Godot AI Backend on dedicated VM instance")
    print(f"🔥 GUNICORN_WORKERS: Configured for high-throughput with multiple workers")
    print(f"🛡️  HANG_PROTECTION: All timeout fixes and streaming optimizations active")

if __name__ == '__main__':
    # Local development only - production uses Gunicorn
    print("🧪 LOCAL_DEV: Starting Flask development server")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=_dev_mode)