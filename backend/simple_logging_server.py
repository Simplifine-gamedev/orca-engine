#!/usr/bin/env python3
"""
Simple VM-based Logging Server for Godot AI
- Accepts logs via HTTP POST
- Returns immediately (fire-and-forget)
- Processes logs in background
- Simple queue-based architecture
"""

import os
import json
import time
import uuid
import threading
import queue
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Configuration
PORT = int(os.getenv('PORT', 8082))
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
SUPABASE_TABLE = os.getenv('SUPABASE_TABLE_NAME', 'llm_logs')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY required")
    exit(1)

# Simple in-memory queue
log_queue = queue.Queue(maxsize=1000)
stats = {
    'logs_received': 0,
    'logs_processed': 0,
    'logs_failed': 0,
    'queue_size': 0,
    'started_at': time.time()
}

def process_log_queue():
    """Background thread to process logs"""
    logger.info("🔄 Starting log processing thread")
    
    while True:
        try:
            # Get log from queue (blocking with timeout)
            log_data = log_queue.get(timeout=5.0)
            
            # Send to Supabase
            success = send_to_supabase(log_data)
            
            if success:
                stats['logs_processed'] += 1
                logger.info(f"✅ Processed log: {log_data.get('request_id', 'unknown')}")
            else:
                stats['logs_failed'] += 1
                logger.warning(f"❌ Failed to process log: {log_data.get('request_id', 'unknown')}")
            
            # Update stats
            stats['queue_size'] = log_queue.qsize()
            log_queue.task_done()
            
        except queue.Empty:
            # No logs to process, continue
            continue
        except Exception as e:
            logger.error(f"Error in log processing: {e}")
            stats['logs_failed'] += 1
            time.sleep(1)

def send_to_supabase(log_data: dict) -> bool:
    """Send log data to Supabase"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, json=log_data, headers=headers, timeout=10)
        
        if response.status_code == 201:
            return True
        else:
            logger.error(f"Supabase error ({response.status_code}): {response.text[:200]}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending to Supabase: {e}")
        return False

@app.route('/webhook/litellm', methods=['POST'])
def receive_log():
    """Receive log from backend - return immediately"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        # Add to queue immediately
        log_queue.put_nowait(data)
        
        # Update stats
        stats['logs_received'] += 1
        stats['queue_size'] = log_queue.qsize()
        
        # Return immediately - don't wait for processing
        return jsonify({
            'success': True,
            'message': 'Log queued',
            'queue_size': stats['queue_size']
        })
        
    except queue.Full:
        stats['logs_failed'] += 1
        logger.warning("Queue full, dropping log")
        return jsonify({'error': 'Queue full'}), 503
    except Exception as e:
        stats['logs_failed'] += 1
        logger.error(f"Error receiving log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'simple-logging-server',
        'stats': stats,
        'supabase_configured': bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)
    })

@app.route('/stats', methods=['GET']) 
def get_stats():
    """Get server statistics"""
    stats['queue_size'] = log_queue.qsize()
    return jsonify(stats)

@app.route('/test', methods=['POST'])
def test_log():
    """Test endpoint"""
    test_data = {
        'request_id': f'test_{int(time.time())}',
        'event_type': 'test',
        'message_type': 'test',
        'message_content': 'Simple VM test',
        'model': 'test-model',
        'success': True,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    log_queue.put_nowait(test_data)
    stats['logs_received'] += 1
    stats['queue_size'] = log_queue.qsize()
    
    return jsonify({
        'success': True,
        'message': 'Test log queued',
        'test_data': test_data
    })

# Start background processing thread
processing_thread = threading.Thread(target=process_log_queue, daemon=True)
processing_thread.start()

if __name__ == '__main__':
    logger.info(f"🚀 Starting Simple Logging Server on port {PORT}")
    logger.info(f"📊 Supabase URL: {SUPABASE_URL}")
    logger.info(f"📋 Table: {SUPABASE_TABLE}")
    
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)

