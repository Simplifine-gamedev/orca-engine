"""
Custom LiteLLM Callback for logging to external logging server
© 2025 Simplifine Corp. LiteLLM integration for Godot AI backend.
"""

import asyncio
import aiohttp
import json
import time
import uuid
import os
from datetime import datetime, timezone
from threading import Thread
from typing import Dict, Any, Optional
from litellm.integrations.custom_logger import CustomLogger
import queue

class GodotLiteLLMLogger(CustomLogger):
    """Custom LiteLLM logger that sends logs to external logging server"""
    
    def __init__(self, logging_server_url: Optional[str] = None):
        super().__init__()
        self.logging_server_url = logging_server_url or os.getenv('LOGGING_SERVER_URL')
        self.log_queue = queue.Queue(maxsize=1000)  # Prevent memory issues
        self.session = None
        
        # Start background thread for async logging
        if self.logging_server_url:
            self._start_background_logger()
            print(f"✅ LiteLLM Logger initialized: {self.logging_server_url}")
        else:
            print("⚠️  LiteLLM Logger disabled: LOGGING_SERVER_URL not configured")
    
    def _start_background_logger(self):
        """Start background thread for async HTTP requests"""
        def run_async_logger():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_log_queue())
        
        thread = Thread(target=run_async_logger, daemon=True)
        thread.start()
    
    async def _process_log_queue(self):
        """Process log queue in background"""
        while True:
            try:
                # Get log from queue (non-blocking with timeout)
                try:
                    log_data = self.log_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Send to logging server
                success = await self._send_log_async(log_data)
                if success:
                    self.log_queue.task_done()
                else:
                    # Log locally on failure but don't break
                    print(f"⚠️  Failed to send log to server: {log_data.get('request_id', 'unknown')}")
                    self.log_queue.task_done()
                    
            except Exception as e:
                print(f"❌ Error in log processing: {e}")
                await asyncio.sleep(1)
    
    async def _send_log_async(self, log_data: dict) -> bool:
        """Send log data to logging server asynchronously"""
        if not self.logging_server_url:
            return False
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
            
            url = f"{self.logging_server_url.rstrip('/')}/webhook/litellm"
            
            async with self.session.post(url, json=log_data) as response:
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Logging server error ({response.status}): {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            print("⏱️  Logging server timeout (skipping)")
            return False
        except Exception as e:
            print(f"❌ Error sending log: {e}")
            return False
    
    def _extract_user_context(self, kwargs: dict) -> dict:
        """Extract enhanced user context from request kwargs"""
        try:
            # Try to get user context from Flask g or request headers
            from flask import g, request
            
            user_context = {}
            
            # Get user ID from Flask context
            if hasattr(g, 'user') and g.user:
                user_context['user_id'] = g.user.get('id')
                user_context['user_provider'] = g.user.get('provider')
                user_context['user_name'] = g.user.get('name')  # For project name inference
            
            # Get project info from Flask context or headers
            if hasattr(g, 'project_root'):
                project_root = g.project_root
                if project_root:
                    import hashlib
                    user_context['project_id'] = hashlib.md5(project_root.encode()).hexdigest()
                    user_context['project_root'] = project_root  # Save full path
                    
                    # Try to extract project name from path
                    try:
                        import os
                        project_name = os.path.basename(project_root.rstrip('/'))
                        user_context['project_name'] = project_name
                    except Exception:
                        pass
            
            # Get request ID from Flask context
            if hasattr(g, 'request_id'):
                user_context['session_id'] = g.request_id
            
            # Get conversation ID from Flask context for conversation tracking
            if hasattr(g, 'conversation_id') and g.conversation_id:
                user_context['conversation_id'] = g.conversation_id
                # Store for consistency in this callback instance
                self._conversation_id = g.conversation_id
            
            # CRITICAL: Store user context for later use when Flask context is gone
            if user_context.get('user_id') or user_context.get('project_id'):
                self._stored_user_context = user_context.copy()
            
            # Get user agent from request (but skip IP address for privacy)
            if request:
                try:
                    user_context['user_agent'] = request.headers.get('User-Agent')
                    # Skip IP address for privacy: user_context['ip_address'] = None
                except Exception:
                    pass
            
            return user_context
            
        except Exception:
            # Not in Flask context or other error
            return {}
    
    def _extract_model_info(self, kwargs: dict) -> dict:
        """Extract model and provider information"""
        try:
            model = kwargs.get('model', 'unknown')
            provider = 'unknown'
            
            # Extract provider from model name (format: provider/model)
            if isinstance(model, str) and '/' in model:
                provider = model.split('/')[0]
            
            return {
                'model': model,
                'provider': provider
            }
        except Exception:
            return {'model': 'unknown', 'provider': 'unknown'}
    
    def _extract_token_info(self, kwargs: dict, response_obj: Any = None) -> dict:
        """Extract token usage information"""
        try:
            token_info = {
                'tokens_prompt': 0,
                'tokens_completion': 0,
                'tokens_total': 0
            }
            
            # Try to get from response usage
            if response_obj and hasattr(response_obj, 'usage'):
                usage = response_obj.usage
                if usage:
                    token_info['tokens_prompt'] = getattr(usage, 'prompt_tokens', 0)
                    token_info['tokens_completion'] = getattr(usage, 'completion_tokens', 0)
                    token_info['tokens_total'] = getattr(usage, 'total_tokens', 0)
            
            # Estimate from messages if not available
            if token_info['tokens_total'] == 0:
                messages = kwargs.get('messages', [])
                if messages:
                    # Rough estimation: 4 chars per token
                    total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                    token_info['tokens_prompt'] = total_chars // 4
                    token_info['tokens_total'] = token_info['tokens_prompt']
            
            return token_info
            
        except Exception:
            return {'tokens_prompt': 0, 'tokens_completion': 0, 'tokens_total': 0}
    
    def _extract_features(self, kwargs: dict) -> dict:
        """Extract feature usage information"""
        try:
            # Check for streaming
            stream = kwargs.get('stream', False)
            
            # Check for tool usage
            tools_used = []
            if kwargs.get('tools'):
                tools_used = [tool.get('function', {}).get('name', 'unknown') 
                             for tool in kwargs.get('tools', [])]
            
            # Check for thinking mode (reasoning parameters)
            thinking_mode = bool(kwargs.get('reasoning_effort') or kwargs.get('reasoning_content'))
            
            # Check for cache hit
            cache_hit = kwargs.get('cache_hit', False)
            
            return {
                'stream': stream,
                'tools_used': tools_used,
                'thinking_mode': thinking_mode,
                'cache_hit': cache_hit
            }
            
        except Exception:
            return {'stream': False, 'tools_used': [], 'thinking_mode': False, 'cache_hit': False}
    
    def _extract_model_version(self, model: str) -> str:
        """Extract model version from model string"""
        try:
            if not model:
                return None
            
            # Extract version patterns
            if 'gpt-3.5-turbo' in model:
                if '0125' in model:
                    return 'gpt-3.5-turbo-0125'
                elif '1106' in model:
                    return 'gpt-3.5-turbo-1106'
                else:
                    return 'gpt-3.5-turbo-latest'
            elif 'gpt-4' in model:
                if 'vision' in model:
                    return 'gpt-4-vision'
                elif 'turbo' in model:
                    return 'gpt-4-turbo'
                elif '0125' in model:
                    return 'gpt-4-0125-preview'
                else:
                    return 'gpt-4-latest'
            elif 'claude' in model:
                if 'sonnet' in model:
                    if '20241022' in model:
                        return 'claude-3.5-sonnet-20241022'
                    elif '20250514' in model:
                        return 'claude-sonnet-4-20250514'
                    else:
                        return 'claude-sonnet-latest'
                elif 'haiku' in model:
                    return 'claude-haiku-latest'
                elif 'opus' in model:
                    return 'claude-opus-latest'
            elif 'gemini' in model:
                if '2.5' in model:
                    return 'gemini-2.5-pro'
                elif 'pro' in model:
                    return 'gemini-pro'
                else:
                    return 'gemini-latest'
            
            return model  # Return as-is if no pattern matches
            
        except Exception:
            return None
    
    def _detect_pii(self, kwargs: dict) -> bool:
        """Basic PII detection in request content"""
        try:
            messages = kwargs.get('messages', [])
            if not isinstance(messages, list):
                return False
            
            # Simple patterns for PII detection
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
            ]
            
            import re
            for msg in messages:
                if isinstance(msg, dict):
                    content = str(msg.get('content', ''))
                    for pattern in pii_patterns:
                        if re.search(pattern, content):
                            return True
            
            return False
            
        except Exception:
            return False  # Conservative: assume no PII if detection fails
    
    def _extract_conversation_info(self, kwargs: dict) -> dict:
        """Extract conversation and message information for debugging"""
        try:
            messages = kwargs.get('messages', [])
            if not isinstance(messages, list) or not messages:
                return {
                    'conversation_id': None,
                    'message_type': 'unknown',
                    'message_content': None,
                    'message_content_full': None,
                    'message_index': 0,
                    'parent_message_id': None
                }
            
            # SIMPLIFIED: Always use Flask context conversation ID
            conversation_id = None
            try:
                from flask import g
                conversation_id = getattr(g, 'conversation_id', None)
            except Exception:
                pass
            
            # If no Flask context, create a simple stable ID
            if not conversation_id:
                import hashlib
                try:
                    from flask import request
                    if request:
                        user_id = request.headers.get('X-User-ID', 'anon')
                        machine_id = request.headers.get('X-Machine-ID', 'unknown')
                        # Simple stable ID based on user+machine only
                        conv_seed = f"{user_id}_{machine_id}"
                        conversation_id = hashlib.md5(conv_seed.encode()).hexdigest()[:16]
                    else:
                        conversation_id = 'no_context'
                except Exception:
                    conversation_id = 'fallback_conv'
            
            # Store for reuse in this callback instance
            self._conversation_id = conversation_id
            
            # Get the last message (what triggered this API call)
            last_message = messages[-1] if messages else {}
            
            # Determine message type
            message_type = 'unknown'
            last_role = last_message.get('role', '')
            
            if last_role == 'user':
                message_type = 'user_input'
            elif last_role == 'assistant':
                if last_message.get('tool_calls'):
                    message_type = 'tool_call'
                else:
                    message_type = 'assistant_output'
            elif last_role == 'tool':
                message_type = 'tool_result'
            elif last_role == 'system':
                message_type = 'system'
            
            # Extract content safely
            message_content = None
            message_content_full = None
            
            if last_message:
                # Get text content (truncate for storage)
                content = last_message.get('content', '')
                if isinstance(content, str):
                    message_content = content[:2000]  # Truncate for DB storage
                    message_content_full = {'content': content[:5000]}  # Longer but still limited
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    text_content = ' '.join(text_parts)
                    message_content = text_content[:2000]
                    message_content_full = {'multimodal_content': content[:10]}  # Limit array size
                
                # Store full message structure (limited)
                if message_content_full is None:
                    message_content_full = {
                        'role': last_role,
                        'content_type': type(content).__name__,
                        'has_tool_calls': bool(last_message.get('tool_calls')),
                        'tool_call_id': last_message.get('tool_call_id'),
                        'name': last_message.get('name')
                    }
            
            # Get parent message ID for tool results
            parent_message_id = None
            if message_type == 'tool_result':
                parent_message_id = last_message.get('tool_call_id')
            
            return {
                'conversation_id': conversation_id,
                'message_type': message_type,
                'message_content': message_content,
                'message_content_full': message_content_full,
                'message_index': len(messages) - 1,  # Position of this message
                'parent_message_id': parent_message_id
            }
            
        except Exception as e:
            print(f"❌ Error extracting conversation info: {e}")
            return {
                'conversation_id': None,
                'message_type': 'error',
                'message_content': f'Error extracting: {str(e)[:100]}',
                'message_content_full': None,
                'message_index': 0,
                'parent_message_id': None
            }

    def _create_base_log(self, kwargs: dict, event_type: str, request_id: str = None) -> dict:
        """Create enhanced base log structure"""
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Extract all information
        user_context = self._extract_user_context(kwargs)
        model_info = self._extract_model_info(kwargs)
        features = self._extract_features(kwargs)
        token_info = self._extract_token_info(kwargs)
        conversation_info = self._extract_conversation_info(kwargs)
        
        # Count messages and analyze content
        messages = kwargs.get('messages', [])
        messages_count = len(messages) if isinstance(messages, list) else 0
        
        # Calculate input characters
        input_chars = 0
        has_images = False
        content_type = 'chat'  # default
        language_detected = None
        
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        input_chars += len(content)
                        # Simple language detection
                        if 'extends' in content or 'func ' in content or '@export' in content:
                            language_detected = 'gdscript'
                        elif 'def ' in content or 'import ' in content or 'class ' in content:
                            language_detected = 'python'
                        elif any(word in content.lower() for word in ['generate', 'create', 'make']):
                            content_type = 'code_generation'
                    elif isinstance(content, list):
                        # Multi-modal content
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    text_content = item.get('text', '')
                                    input_chars += len(text_content)
                                elif item.get('type') == 'image_url':
                                    has_images = True
                                    content_type = 'image_analysis'
                    
                    # Check for images in message
                    if msg.get('images'):
                        has_images = True
                        content_type = 'image_analysis'
        
        # Extract request parameters
        max_tokens = kwargs.get('max_tokens')
        temperature = kwargs.get('temperature')
        
        # Determine if this is billable (based on model and user type)
        is_billable = True
        billing_category = 'standard'
        
        # Check if user is guest (might be free tier)
        if user_context.get('user_provider') == 'guest':
            billing_category = 'free_tier'
        elif user_context.get('user_provider') in ['google', 'github']:
            billing_category = 'standard'
        
        # PII detection (basic)
        has_pii = self._detect_pii(kwargs)
        
        # System context
        godot_version = os.getenv('GODOT_VERSION', '4.3')
        backend_version = os.getenv('BACKEND_VERSION', '1.0.0')
        deployment_mode = os.getenv('DEPLOYMENT_MODE', 'development')
        
        return {
            'request_id': request_id,
            'event_type': event_type,
            
            # Model info
            **model_info,
            'model_version': self._extract_model_version(kwargs.get('model')),
            
            # Request details
            'messages_count': messages_count,
            'input_chars': input_chars,
            'max_tokens': max_tokens,
            'temperature': temperature,
            
            # Token info (will be updated by specific handlers)
            **token_info,
            
            # User context
            **user_context,
            
            # Conversation tracking
            **conversation_info,
            
            # Feature usage
            **features,
            'has_images': has_images,
            
            # Content analysis
            'content_type': content_type,
            'language_detected': language_detected,
            
            # System context
            'godot_version': godot_version,
            'backend_version': backend_version,
            'deployment_mode': deployment_mode,
            
            # Business metrics
            'is_billable': is_billable,
            'billing_category': billing_category,
            
            # Data quality
            'data_quality_score': 1.0,  # Will be adjusted based on completeness
            'has_pii': has_pii,
            
            # Will be filled by specific event handlers
            'success': True,
            'duration_ms': 0,
            'cost_usd': 0.0,
            'error_message': None,
            'error_type': None,
            'status_code': None,
            'retry_count': 0
        }
    
    def _queue_log(self, log_data: dict):
        """Add log to queue for async processing"""
        if not self.logging_server_url:
            return
        
        try:
            self.log_queue.put_nowait(log_data)
        except queue.Full:
            print("⚠️  Log queue full, dropping log entry")
    
    # LiteLLM hook implementations
    
    def log_pre_api_call(self, model, messages, kwargs):
        """Log before API call - STORE CONTEXT but don't log yet"""
        try:
            request_id = str(uuid.uuid4())
            # Store request_id in kwargs for tracking
            kwargs['_godot_request_id'] = request_id
            kwargs['_godot_start_time'] = time.time()
            
            # IMPORTANT: Store context now while Flask context is available
            user_context = self._extract_user_context(kwargs)
            conversation_info = self._extract_conversation_info(kwargs)
            
            # Store for later use when Flask context might be gone
            self._stored_context = {
                **user_context,
                **conversation_info,
                'request_id': request_id
            }
            
            # Don't log pre_call to reduce noise - we'll log on success
            
        except Exception as e:
            print(f"❌ Error in pre_api_call logging: {e}")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        """Log after API call - DON'T log to reduce noise"""
        try:
            # Just calculate duration and store it - don't create a log entry
            request_id = kwargs.get('_godot_request_id', str(uuid.uuid4()))
            
            # Safe duration calculation
            duration_ms = 0
            if start_time and end_time:
                try:
                    duration_ms = int((end_time - start_time).total_seconds() * 1000)
                except Exception:
                    duration_ms = 0
            
            # Store duration for success event
            kwargs['_godot_duration_ms'] = duration_ms
            
            # Don't log post_call to reduce noise
            
        except Exception as e:
            print(f"❌ Error in post_api_call logging: {e}")
    
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful API call - ONLY LOG 2 ENTRIES: user input + assistant response"""
        try:
            request_id = kwargs.get('_godot_request_id', str(uuid.uuid4()))
            
            # Get stored duration from post_call
            duration_ms = kwargs.get('_godot_duration_ms', 0)
            if duration_ms == 0 and start_time and end_time:
                try:
                    duration_ms = int((end_time - start_time).total_seconds() * 1000)
                except Exception:
                    duration_ms = 0
            
            # Use stored context from pre_call (when Flask context was available)
            stored_context = getattr(self, '_stored_context', {})
            
            # 1. LOG USER INPUT (only one entry for the user's message)
            user_log = self._create_user_input_log(kwargs, request_id, duration_ms, stored_context)
            self._queue_log(user_log)
            
            # 2. LOG ASSISTANT RESPONSE (if we got a response)
            if response_obj and hasattr(response_obj, 'choices') and response_obj.choices:
                try:
                    assistant_content = response_obj.choices[0].message.content
                    if assistant_content:
                        assistant_log = self._create_assistant_response_log(
                            kwargs, response_obj, assistant_content, request_id, duration_ms, stored_context
                        )
                        self._queue_log(assistant_log)
                except Exception as e:
                    print(f"⚠️  Error logging assistant response: {e}")
            
        except Exception as e:
            print(f"❌ Error in success_event logging: {e}")
    
    def _create_user_input_log(self, kwargs: dict, request_id: str, duration_ms: int, stored_context: dict) -> dict:
        """Create log entry for user input"""
        model_info = self._extract_model_info(kwargs)
        features = self._extract_features(kwargs)
        token_info = self._extract_token_info(kwargs)
        
        # Get the last user message
        messages = kwargs.get('messages', [])
        user_message = None
        user_content = ""
        
        # Find the last user message
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                user_message = msg
                user_content = str(msg.get('content', ''))
                break
        
        return {
            'request_id': request_id,
            'event_type': 'user_input',
            
            # Model info
            **model_info,
            'model_version': self._extract_model_version(kwargs.get('model')),
            
            # Request details
            'messages_count': len(messages),
            'input_chars': len(user_content),
            'max_tokens': kwargs.get('max_tokens'),
            'temperature': kwargs.get('temperature'),
            
            # Token info
            **token_info,
            
            # Performance
            'duration_ms': duration_ms,
            'cost_usd': kwargs.get('response_cost', 0.0),
            'success': True,
            
            # FIXED: Use stored context (from when Flask was available)
            **stored_context,
            
            # Features
            **features,
            'has_images': self._has_images_in_messages(messages),
            
            # Content analysis
            'content_type': 'user_input',
            'language_detected': self._detect_language_in_text(user_content),
            
            # Conversation tracking
            'message_type': 'user_input',
            'message_content': user_content[:2000],
            'message_content_full': {'content': user_content[:5000]} if user_content else None,
            'message_index': len(messages) - 1,
            'parent_message_id': None,
            
            # System context
            'godot_version': os.getenv('GODOT_VERSION', '4.3'),
            'backend_version': os.getenv('BACKEND_VERSION', '1.0.0'),
            'deployment_mode': os.getenv('DEPLOYMENT_MODE', 'development'),
            
            # Business metrics
            'is_billable': True,
            'billing_category': 'standard',
            'data_quality_score': 1.0,
            'has_pii': self._detect_pii_in_text(user_content),
        }
    
    def _has_images_in_messages(self, messages: list) -> bool:
        """Check if messages contain images"""
        try:
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get('images'):
                        return True
                    content = msg.get('content')
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                return True
            return False
        except Exception:
            return False
    
    def _create_assistant_response_log(self, kwargs: dict, response_obj, assistant_content: str, request_id: str, duration_ms: int, stored_context: dict) -> dict:
        """Create a separate log entry for the assistant response"""
        try:
            model_info = self._extract_model_info(kwargs)
            features = self._extract_features(kwargs)
            token_info = self._extract_token_info(kwargs, response_obj)
            
            # CRITICAL FIX: Use stored context from pre_call (when Flask was available)
            # This ensures assistant responses get the SAME conversation_id as user inputs
            conversation_id = stored_context.get('conversation_id', 'no_context_fix_needed')
            
            # Use stored context for all user-related fields
            user_context = stored_context
            
            # Create assistant-specific log
            assistant_log = {
                'request_id': request_id + '_response',  # Unique ID for assistant response
                'event_type': 'assistant_response',
                
                # Model info
                **model_info,
                'model_version': self._extract_model_version(kwargs.get('model')),
                
                # Request details
                'messages_count': len(kwargs.get('messages', [])),
                'input_chars': len(assistant_content),
                'max_tokens': kwargs.get('max_tokens'),
                'temperature': kwargs.get('temperature'),
                
                # Token info
                **token_info,
                
                # Performance
                'duration_ms': duration_ms,
                'cost_usd': kwargs.get('response_cost', 0.0),
                'success': True,
                
                # FIXED: Use stored context (preserves conversation_id, user_id, project_id, etc.)
                **user_context,
                
                # Conversation tracking - ASSISTANT RESPONSE
                'conversation_id': conversation_id,  # This will be the SAME as user input now
                'message_type': 'assistant_output',
                'message_content': assistant_content[:2000],
                'message_content_full': {'content': assistant_content[:5000]},
                'message_index': len(kwargs.get('messages', [])),  # Next position after input
                'parent_message_id': request_id,  # Link back to the request
                
                # Features
                **features,
                'has_images': False,  # Assistant text responses don't have images
                
                # Content analysis
                'content_type': 'assistant_response',
                'language_detected': self._detect_language_in_text(assistant_content),
                
                # System context
                'godot_version': os.getenv('GODOT_VERSION', '4.3'),
                'backend_version': os.getenv('BACKEND_VERSION', '1.0.0'),
                'deployment_mode': os.getenv('DEPLOYMENT_MODE', 'development'),
                
                # Business metrics
                'is_billable': True,
                'billing_category': user_context.get('user_provider') == 'guest' and 'free_tier' or 'standard',
                'data_quality_score': 1.0,
                'has_pii': self._detect_pii_in_text(assistant_content),
            }
            
            return assistant_log
            
        except Exception as e:
            print(f"❌ Error creating assistant response log: {e}")
            return {}
    
    def _detect_language_in_text(self, text: str) -> str:
        """Detect programming language in text"""
        if not text:
            return None
            
        text_lower = text.lower()
        if 'extends' in text_lower and ('func ' in text_lower or '@export' in text_lower):
            return 'gdscript'
        elif 'def ' in text_lower and 'import ' in text_lower:
            return 'python'
        elif any(word in text_lower for word in ['#include', 'int main', 'void ']):
            return 'cpp'
        elif any(word in text_lower for word in ['function', 'const ', 'let ', 'var ']):
            return 'javascript'
        else:
            return 'natural_language'
    
    def _detect_pii_in_text(self, text: str) -> bool:
        """Detect PII in text content"""
        if not text:
            return False
            
        import re
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed API call"""
        try:
            request_id = kwargs.get('_godot_request_id', str(uuid.uuid4()))
            
            # Safe duration calculation
            duration_ms = 0
            if start_time and end_time:
                try:
                    duration_ms = int((end_time - start_time).total_seconds() * 1000)
                except Exception:
                    duration_ms = 0
            
            log_data = self._create_base_log(kwargs, 'failure', request_id)
            log_data.update({
                'success': False,
                'duration_ms': duration_ms,
                'error_message': str(response_obj) if response_obj else 'Unknown error'
            })
            
            # Try to extract error details
            if hasattr(response_obj, 'status_code'):
                log_data['status_code'] = response_obj.status_code
            
            # Try to extract error type
            if response_obj:
                error_str = str(response_obj).lower()
                if 'rate limit' in error_str:
                    log_data['error_type'] = 'rate_limit'
                elif 'timeout' in error_str:
                    log_data['error_type'] = 'timeout'
                elif 'connection' in error_str:
                    log_data['error_type'] = 'connection_error'
                else:
                    log_data['error_type'] = 'api_error'
            
            self._queue_log(log_data)
            
        except Exception as e:
            print(f"❌ Error in failure_event logging: {e}")
    
    # Async versions for acompletion
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful async API call"""
        # For now, use sync version (the queue processing is already async)
        self.log_success_event(kwargs, response_obj, start_time, end_time)
    
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed async API call"""
        # For now, use sync version (the queue processing is already async)
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'session') and self.session:
            try:
                asyncio.create_task(self.session.close())
            except Exception:
                pass
