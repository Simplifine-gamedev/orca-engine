#!/usr/bin/env python3
"""
Local development server for LiteLLM logging
Runs on localhost:3031 for testing the complete logging flow locally
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the logging server
from logging_server import app, process_log_queue, logger

def main():
    """Run the logging server locally"""
    print("🚀 Starting LiteLLM Logging Server (Local Development)")
    print("=" * 50)
    
    # Check configuration
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    table_name = os.getenv('SUPABASE_TABLE_NAME', 'llm_logs')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase configuration!")
        print("   Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        return
    
    print(f"📊 Supabase URL: {supabase_url}")
    print(f"📋 Table: {table_name}")
    print(f"🌐 Local URL: http://localhost:3031")
    print(f"📝 Health check: http://localhost:3031/health")
    print(f"🧪 Test endpoint: http://localhost:3031/test")
    print()
    
    # Start background log processor
    process_log_queue()
    
    # Run Flask app locally
    try:
        print("🔄 Starting Flask server on port 3031...")
        app.run(
            host='127.0.0.1',  # Local only
            port=3031,
            debug=True,  # Enable debug mode for local development
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Logging server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()

