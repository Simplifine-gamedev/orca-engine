"""
LiteLLM Logging Server - Receives webhook calls from the main app and saves to Supabase
© 2025 Simplifine Corp. Logging service for Godot AI backend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
import uuid
from datetime import datetime, timezone
import asyncio
import aiohttp
from threading import Thread
import queue
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Use service role key for backend
SUPABASE_TABLE_NAME = os.getenv('SUPABASE_TABLE_NAME', 'llm_logs')

# Validation
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("❌ SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    raise ValueError("Missing Supabase configuration")

# Queue for processing logs asynchronously
log_queue = queue.Queue()

class SupabaseClient:
    def __init__(self, url: str, service_key: str):
        self.url = url.rstrip('/')
        self.service_key = service_key
        self.headers = {
            'apikey': service_key,
            'Authorization': f'Bearer {service_key}',
            'Content-Type': 'application/json'
        }
    
    async def insert_log(self, log_data: dict):
        """Insert log data into Supabase table"""
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.url}/rest/v1/{SUPABASE_TABLE_NAME}"
                
                async with session.post(url, json=log_data, headers=self.headers) as response:
                    if response.status == 201:
                        logger.info(f"✅ Log saved to Supabase: {log_data['request_id']}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Supabase insert failed ({response.status}): {error_text}")
                        return False
                        
            except Exception as e:
                logger.error(f"❌ Error inserting to Supabase: {e}")
                return False

# Initialize Supabase client
supabase = SupabaseClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def _limit_json_size(data: dict, max_size_kb: int = 50) -> dict:
    """Limit JSON size to prevent database bloat"""
    try:
        if not data:
            return {}
        
        json_str = json.dumps(data)
        size_kb = len(json_str) / 1024
        
        if size_kb <= max_size_kb:
            return data
        
        # If too large, return a truncated version
        return {
            "truncated": True,
            "original_size_kb": round(size_kb, 2),
            "sample": str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data)
        }
    except Exception:
        return {"error": "Failed to serialize data"}

def process_log_queue():
    """Background thread to process log queue"""
    async def process_logs():
        while True:
            try:
                # Get log from queue (non-blocking with timeout)
                try:
                    log_data = log_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the log
                success = await supabase.insert_log(log_data)
                if success:
                    log_queue.task_done()
                else:
                    logger.warning("⚠️ Failed to save log, discarding")
                    log_queue.task_done()
                    
            except Exception as e:
                logger.error(f"❌ Error processing log queue: {e}")
                await asyncio.sleep(1)
    
    # Run async processing in thread
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_logs())
    
    thread = Thread(target=run_async, daemon=True)
    thread.start()
    logger.info("🔄 Log processing thread started")

def parse_log_data(raw_data: dict) -> dict:
    """Parse and structure log data for Supabase with enhanced fields"""
    try:
        # Extract basic info
        log_entry = {
            'id': str(uuid.uuid4()),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'request_id': raw_data.get('request_id', str(uuid.uuid4())),
            'event_type': raw_data.get('event_type', 'unknown'),
            
            # Model and provider info
            'model': raw_data.get('model', 'unknown'),
            'provider': raw_data.get('provider', 'unknown'),
            'model_version': raw_data.get('model_version'),
            
            # Request details
            'messages_count': raw_data.get('messages_count', 0),
            'input_chars': raw_data.get('input_chars', 0),
            'max_tokens': raw_data.get('max_tokens'),
            'temperature': raw_data.get('temperature'),
            
            # Token usage
            'tokens_prompt': raw_data.get('tokens_prompt', 0),
            'tokens_completion': raw_data.get('tokens_completion', 0),
            'tokens_total': raw_data.get('tokens_total', 0),
            
            # Performance metrics
            'duration_ms': raw_data.get('duration_ms', 0),
            'cost_usd': raw_data.get('cost_usd', 0.0),
            
            # Success/failure tracking
            'success': raw_data.get('success', True),
            'error_message': raw_data.get('error_message'),
            'error_type': raw_data.get('error_type'),
            'status_code': raw_data.get('status_code'),
            'retry_count': raw_data.get('retry_count', 0),
            
            # User and project context
            'user_id': raw_data.get('user_id'),
            'user_provider': raw_data.get('user_provider'),
            'project_id': raw_data.get('project_id'),
            'project_name': raw_data.get('project_name'),
            'project_root': raw_data.get('project_root'),  # Full project path
            'session_id': raw_data.get('session_id'),
            # Skip IP address for privacy: 'ip_address': None,
            'user_agent': raw_data.get('user_agent'),
            
            # Feature usage
            'stream': raw_data.get('stream', False),
            'cache_hit': raw_data.get('cache_hit', False),
            'thinking_mode': raw_data.get('thinking_mode', False),
            'has_images': raw_data.get('has_images', False),
            'has_tools': raw_data.get('has_tools', False),
            
            # Tool and feature details
            'tools_used': raw_data.get('tools_used', []),
            'tool_execution_count': raw_data.get('tool_execution_count', 0),
            'tool_execution_time_ms': raw_data.get('tool_execution_time_ms', 0),
            
            # Content analysis
            'content_type': raw_data.get('content_type'),
            'language_detected': raw_data.get('language_detected'),
            
            # Conversation and message tracking (for debugging)
            'conversation_id': raw_data.get('conversation_id'),
            'message_type': raw_data.get('message_type'),
            'message_content': raw_data.get('message_content'),
            'message_content_full': raw_data.get('message_content_full'),
            'message_index': raw_data.get('message_index', 0),
            'parent_message_id': raw_data.get('parent_message_id'),
            
            # System context
            'godot_version': raw_data.get('godot_version', os.getenv('GODOT_VERSION')),
            'backend_version': raw_data.get('backend_version', os.getenv('BACKEND_VERSION')),
            'deployment_mode': raw_data.get('deployment_mode', os.getenv('DEPLOYMENT_MODE', 'unknown')),
            
            # Geographical context
            'country_code': raw_data.get('country_code', os.getenv('DEFAULT_COUNTRY_CODE')),
            'region': raw_data.get('region', os.getenv('DEFAULT_REGION')),
            
            # Business metrics
            'is_billable': raw_data.get('is_billable', True),
            'billing_category': raw_data.get('billing_category', 'standard'),
            
            # Data quality
            'data_quality_score': raw_data.get('data_quality_score', 1.0),
            'has_pii': raw_data.get('has_pii', False),
            
            # Raw data (for debugging) - limit size to prevent database bloat
            'raw_request': _limit_json_size(raw_data.get('raw_request', {})),
            'raw_response': _limit_json_size(raw_data.get('raw_response', {}))
        }
        
        return log_entry
        
    except Exception as e:
        logger.error(f"❌ Error parsing log data: {e}")
        # Return minimal log entry
        return {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'request_id': raw_data.get('request_id', str(uuid.uuid4())),
            'event_type': 'parse_error',
            'error_message': str(e),
            'success': False,
            'raw_kwargs': json.dumps(raw_data) if isinstance(raw_data, dict) else str(raw_data)
        }

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'litellm-logging-server',
        'queue_size': log_queue.qsize(),
        'supabase_configured': bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        'table_name': SUPABASE_TABLE_NAME
    })

@app.route('/webhook/litellm', methods=['POST'])
def receive_litellm_log():
    """Receive LiteLLM log data from the main app"""
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        logger.info(f"📝 Received log: {data.get('event_type', 'unknown')} for request {data.get('request_id', 'unknown')}")
        
        # Parse and structure the data
        structured_log = parse_log_data(data)
        
        # Add to queue for async processing
        log_queue.put(structured_log)
        
        return jsonify({
            'success': True,
            'message': 'Log queued for processing',
            'queue_size': log_queue.qsize()
        })
        
    except Exception as e:
        logger.error(f"❌ Error receiving log: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get logging statistics"""
    return jsonify({
        'queue_size': log_queue.qsize(),
        'table_name': SUPABASE_TABLE_NAME,
        'supabase_url': SUPABASE_URL,
        'uptime': time.time()  # Simple uptime indicator
    })

@app.route('/test', methods=['POST'])
def test_log():
    """Test endpoint to verify logging works"""
    test_data = {
        'request_id': f'test_{int(time.time())}',
        'event_type': 'test',
        'model': 'gpt-3.5-turbo',
        'provider': 'openai',
        'success': True,
        'duration_ms': 1500,
        'cost_usd': 0.002,
        'tokens_total': 150,
        'user_id': 'test_user'
    }
    
    structured_log = parse_log_data(test_data)
    log_queue.put(structured_log)
    
    return jsonify({
        'success': True,
        'message': 'Test log queued',
        'test_data': structured_log
    })

# CRITICAL: Start background processor at module level (not just in __main__)
# This ensures it runs even when loaded by Gunicorn
process_log_queue()
logger.info("🔄 Log processing thread started at module level")

if __name__ == '__main__':
    # Already started above
    # process_log_queue()
    
    # Print configuration
    logger.info("🚀 Starting LiteLLM Logging Server")
    logger.info(f"📊 Supabase URL: {SUPABASE_URL}")
    logger.info(f"📋 Table name: {SUPABASE_TABLE_NAME}")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=False)
