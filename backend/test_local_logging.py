#!/usr/bin/env python3
"""
Local development testing for LiteLLM logging system
Tests the complete flow locally before deploying to cloud
"""

import os
import sys
import time
import requests
import subprocess
import signal
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()

def start_local_logging_server():
    """Start the local logging server in background"""
    print("🚀 Starting local logging server...")
    
    try:
        # Start the server as a subprocess
        process = subprocess.Popen([
            sys.executable, 'run_logging_server_local.py'
        ], cwd=os.path.dirname(__file__))
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test if server is responding
        try:
            response = requests.get('http://localhost:3031/health', timeout=5)
            if response.status_code == 200:
                print("✅ Local logging server started successfully!")
                print(f"   PID: {process.pid}")
                print(f"   URL: http://localhost:3031")
                return process
            else:
                print(f"❌ Server started but not responding correctly: {response.status_code}")
                process.terminate()
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Server started but not accepting connections")
            process.terminate()
            return None
        except Exception as e:
            print(f"❌ Error testing server: {e}")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Failed to start local logging server: {e}")
        return None

def test_local_flow():
    """Test the complete local logging flow"""
    print("\n🧪 Testing complete local logging flow...")
    
    # Test the logging server directly
    test_data = {
        'request_id': f'local_test_{int(time.time())}',
        'event_type': 'local_test',
        'model': 'gpt-3.5-turbo',
        'provider': 'openai',
        'messages_count': 1,
        'tokens_total': 50,
        'duration_ms': 1500,
        'cost_usd': 0.002,
        'success': True,
        'user_id': 'local_test_user',
        'content_type': 'test',
        'deployment_mode': 'local_development'
    }
    
    try:
        # Send test log to local server
        response = requests.post(
            'http://localhost:3031/webhook/litellm',
            json=test_data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        print("✅ Direct logging test successful!")
        print(f"   Response: {result}")
        
        # Wait for processing
        print("⏳ Waiting for log to be processed...")
        time.sleep(2)
        
        # Check server stats
        stats_response = requests.get('http://localhost:3031/stats', timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"📊 Server stats: Queue size: {stats.get('queue_size', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct logging test failed: {e}")
        return False

def test_with_litellm():
    """Test LiteLLM integration locally"""
    print("\n🔄 Testing LiteLLM integration...")
    
    # Check if we have OpenAI API key for testing
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found - skipping LiteLLM integration test")
        print("   Add OPENAI_API_KEY to .env to test the complete flow")
        return True  # Not a failure, just skipped
    
    try:
        # Set up LiteLLM with local logging
        from litellm_callback import GodotLiteLLMLogger
        import litellm
        
        # Initialize logger with local URL
        logger = GodotLiteLLMLogger('http://localhost:3031')
        litellm.callbacks = [logger]
        
        print("📝 Making test LiteLLM call...")
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": "Reply with exactly: 'Local logging test successful!'"
            }],
            max_tokens=20
        )
        
        content = response.choices[0].message.content
        print(f"🤖 LiteLLM response: {content}")
        
        # Wait for logs to process
        print("⏳ Waiting for logs to process...")
        time.sleep(3)
        
        print("✅ LiteLLM integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ LiteLLM integration test failed: {e}")
        return False

def main():
    """Main local testing function"""
    print("🧪 LOCAL LITELLM LOGGING SYSTEM TEST")
    print("="*50)
    print()
    
    # Check configuration
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase configuration!")
        print("   Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        return False
    
    print(f"📊 Supabase URL: {supabase_url}")
    print()
    
    # Start local logging server
    server_process = start_local_logging_server()
    if not server_process:
        print("❌ Cannot continue without logging server")
        return False
    
    try:
        # Run tests
        tests = [
            ("Direct Logging Test", test_local_flow),
            ("LiteLLM Integration Test", test_with_litellm),
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
        print("\n" + "="*50)
        print("📊 LOCAL TEST SUMMARY")
        print("="*50)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All local tests passed!")
            print("\n📋 Next steps:")
            print("1. Add to your .env: LOGGING_SERVER_URL=http://localhost:3031")
            print("2. Test with your main app locally")
            print("3. Deploy to cloud when ready: ./deploy_logger.sh YOUR_PROJECT_ID")
        else:
            print("\n⚠️  Some tests failed. Check the output above.")
        
        print(f"\n📊 Check your Supabase dashboard for logged data!")
        print(f"   {supabase_url.replace('/rest/v1', '')}")
        
        return all_passed
        
    finally:
        # Clean up server process
        if server_process:
            print(f"\n🛑 Stopping local logging server (PID: {server_process.pid})...")
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
                print("✅ Server stopped")
            except subprocess.TimeoutExpired:
                server_process.kill()
                print("🔥 Server killed (forced)")
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")

