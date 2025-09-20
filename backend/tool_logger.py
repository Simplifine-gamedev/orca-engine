#!/usr/bin/env python3
"""
Tool logging helper for capturing tool calls and results in conversations
"""

import requests
import json
import time
import os
from flask import g

def log_tool_call(tool_name: str, tool_id: str, arguments: dict):
    """Log when AI decides to call a tool"""
    try:
        # Check if logging is enabled
        logging_server_url = os.getenv('LOGGING_SERVER_URL') if os.getenv('DEV_MODE', 'false').lower() != 'true' else 'http://localhost:3031'
        if not logging_server_url:
            return
        
        # Get conversation context from Flask g
        conversation_id = getattr(g, 'conversation_id', 'no_flask_context')
        request_id = getattr(g, 'request_id', tool_id)
        
        # Create tool call log
        tool_call_log = {
            'request_id': f"{request_id}_tool_{tool_id}",
            'event_type': 'tool_call',
            'model': 'N/A',  # Tool calls don't have models
            'provider': 'godot_backend',
            
            # Basic info
            'messages_count': 0,
            'duration_ms': 0,
            'success': True,
            
            # Conversation tracking
            'conversation_id': conversation_id,
            'message_type': 'tool_call',
            'message_content': f"Tool: {tool_name}({json.dumps(arguments)[:100]}...)",
            'message_content_full': {
                'tool_name': tool_name,
                'tool_id': tool_id,
                'arguments': arguments
            },
            'parent_message_id': request_id,  # Link to the user request that triggered this
            
            # Tool-specific info
            'tools_used': [tool_name],
            'has_tools': True,
            'content_type': 'tool_execution',
            
            # System context
            'user_id': getattr(g, 'user', {}).get('id') if hasattr(g, 'user') else None,
            'project_id': getattr(g, 'project_root', None),
            'session_id': request_id,
            'godot_version': os.getenv('GODOT_VERSION', '4.3'),
            'backend_version': os.getenv('BACKEND_VERSION', '1.0.0'),
            'deployment_mode': os.getenv('DEPLOYMENT_MODE', 'development'),
        }
        
        # Send to logging server (fire and forget)
        try:
            requests.post(
                f"{logging_server_url}/webhook/litellm",
                json=tool_call_log,
                timeout=2  # Quick timeout - don't block tool execution
            )
        except Exception:
            pass  # Don't break tool execution if logging fails
            
    except Exception as e:
        print(f"⚠️  Tool call logging failed: {e}")

def log_tool_result(tool_name: str, tool_id: str, result: dict, duration_ms: int = 0):
    """Log when a tool returns a result"""
    try:
        # Check if logging is enabled
        logging_server_url = os.getenv('LOGGING_SERVER_URL') if os.getenv('DEV_MODE', 'false').lower() != 'true' else 'http://localhost:3031'
        if not logging_server_url:
            return
        
        # Get conversation context from Flask g
        conversation_id = getattr(g, 'conversation_id', 'no_flask_context')
        request_id = getattr(g, 'request_id', tool_id)
        
        # Create tool result log
        success = result.get('success', True)
        error_message = result.get('error') if not success else None
        
        # Create clean result preview
        result_preview = json.dumps(result)[:500]
        if len(result_preview) >= 500:
            result_preview += "..."
        
        tool_result_log = {
            'request_id': f"{request_id}_result_{tool_id}",
            'event_type': 'tool_result',
            'model': 'N/A',  # Tool results don't have models
            'provider': 'godot_backend',
            
            # Basic info
            'messages_count': 0,
            'duration_ms': duration_ms,
            'success': success,
            'error_message': error_message,
            
            # Conversation tracking
            'conversation_id': conversation_id,
            'message_type': 'tool_result',
            'message_content': result_preview,
            'message_content_full': {
                'tool_name': tool_name,
                'tool_id': tool_id,
                'result': result if len(json.dumps(result)) < 2000 else {'truncated': True, 'preview': result_preview}
            },
            'parent_message_id': f"{request_id}_tool_{tool_id}",  # Link to the tool call
            
            # Tool-specific info
            'tools_used': [tool_name],
            'has_tools': True,
            'content_type': 'tool_result',
            
            # System context
            'user_id': getattr(g, 'user', {}).get('id') if hasattr(g, 'user') else None,
            'project_id': getattr(g, 'project_root', None),
            'session_id': request_id,
            'godot_version': os.getenv('GODOT_VERSION', '4.3'),
            'backend_version': os.getenv('BACKEND_VERSION', '1.0.0'),
            'deployment_mode': os.getenv('DEPLOYMENT_MODE', 'development'),
        }
        
        # Send to logging server (fire and forget)
        try:
            requests.post(
                f"{logging_server_url}/webhook/litellm",
                json=tool_result_log,
                timeout=2  # Quick timeout - don't block response
            )
        except Exception:
            pass  # Don't break response if logging fails
            
    except Exception as e:
        print(f"⚠️  Tool result logging failed: {e}")
