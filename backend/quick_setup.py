#!/usr/bin/env python3
"""
Quick setup for LiteLLM logging - simple and clean
"""

import os
import sys
import requests
from dotenv import load_dotenv

def main():
    """Quick setup function"""
    load_dotenv()
    
    print("🚀 Quick LiteLLM Logging Setup")
    print("=" * 30)
    
    # Check Supabase config
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Add to .env:")
        print("SUPABASE_URL=https://your-project.supabase.co")
        print("SUPABASE_SERVICE_KEY=your-service-key")
        print("SUPABASE_TABLE_NAME=llm_logs")
        return
    
    # Test connection
    print("🧪 Testing Supabase...")
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{supabase_url}/rest/v1/llm_logs?select=id&limit=1", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Supabase connected!")
        elif response.status_code == 404:
            print("⚠️  Table 'llm_logs' doesn't exist")
            print("   1. Go to your Supabase SQL Editor")
            print("   2. Run: simple_supabase_table.sql")
            print("   3. Then run: alter_existing_table.sql") 
        else:
            print(f"❌ Supabase error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Check for local/cloud mode
    if '--local' in sys.argv:
        print("\n🧪 Local mode setup:")
        print("1. Update .env: LOGGING_SERVER_URL=http://localhost:3031")
        print("2. Start server: python run_logging_server_local.py")
        print("3. Test: python test_local_logging.py")
    else:
        print("\n☁️  Cloud mode setup:")
        print("1. Deploy: ./deploy_logger.sh YOUR_PROJECT_ID") 
        print("2. Update .env with deployed URL")
        print("3. Deploy main app: ./deploy.sh YOUR_PROJECT_ID")
    
    print(f"\n📊 Check logs at: {supabase_url.replace('/rest/v1', '')}")

if __name__ == "__main__":
    main()
