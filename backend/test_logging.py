"""
Test script for LiteLLM logging system
© 2025 Simplifine Corp. Test the logging integration.
"""

import os
import requests
import json
import time
from litellm_callback import GodotLiteLLMLogger
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_logging_server_direct():
    """Test the logging server directly"""
    logging_server_url = os.getenv('LOGGING_SERVER_URL')
    if not logging_server_url:
        print("❌ LOGGING_SERVER_URL not configured")
        return False
    
    print(f"🧪 Testing logging server: {logging_server_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{logging_server_url}/health", timeout=10)
        response.raise_for_status()
        health_data = response.json()
        print(f"✅ Health check: {health_data}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test webhook endpoint with sample data
    test_log = {
        'request_id': f'test_{int(time.time())}',
        'event_type': 'test_direct',
        'model': 'gpt-3.5-turbo',
        'provider': 'openai',
        'success': True,
        'duration_ms': 1500,
        'cost_usd': 0.002,
        'tokens_total': 150,
        'user_id': 'test_user',
        'messages_count': 2
    }
    
    try:
        response = requests.post(
            f"{logging_server_url}/webhook/litellm",
            json=test_log,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Direct webhook test: {result}")
        return True
    except Exception as e:
        print(f"❌ Direct webhook test failed: {e}")
        return False

def test_litellm_integration():
    """Test LiteLLM integration with our custom logger"""
    logging_server_url = os.getenv('LOGGING_SERVER_URL')
    if not logging_server_url:
        print("❌ LOGGING_SERVER_URL not configured - skipping LiteLLM test")
        return False
    
    print(f"🧪 Testing LiteLLM integration")
    
    # Initialize custom logger
    logger = GodotLiteLLMLogger(logging_server_url)
    litellm.callbacks = [logger]
    
    # Test with a simple completion call
    try:
        # Make sure we have OpenAI API key for testing
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ OPENAI_API_KEY not configured - skipping LiteLLM completion test")
            return False
        
        print("🔄 Making test completion call...")
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": "Say 'Hello from LiteLLM logging test!'"
            }],
            max_tokens=50
        )
        
        print(f"✅ LiteLLM test completed: {response.choices[0].message.content}")
        
        # Wait a moment for async logging to complete
        print("⏳ Waiting for logs to be processed...")
        time.sleep(3)
        
        return True
        
    except Exception as e:
        print(f"❌ LiteLLM integration test failed: {e}")
        return False

def test_supabase_connection():
    """Test Supabase connection"""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase configuration missing - skipping Supabase test")
        return False
    
    print(f"🧪 Testing Supabase connection: {supabase_url}")
    
    try:
        # Test the connection by checking the table exists
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
            'Content-Type': 'application/json'
        }
        
        # Try to query the table (limit 1 to minimize impact)
        table_name = os.getenv('SUPABASE_TABLE_NAME', 'llm_logs')
        response = requests.get(
            f"{supabase_url}/rest/v1/{table_name}?select=id&limit=1",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        print(f"✅ Supabase connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        print("   Make sure you've run create_supabase_table.sql in your Supabase SQL editor")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting LiteLLM logging system tests")
    print("=" * 50)
    
    tests = [
        ("Supabase Connection", test_supabase_connection),
        ("Logging Server Direct", test_logging_server_direct),
        ("LiteLLM Integration", test_litellm_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! LiteLLM logging system is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check configuration and try again.")
    
    return all_passed

if __name__ == "__main__":
    main()

