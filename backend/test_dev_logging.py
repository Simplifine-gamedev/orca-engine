#!/usr/bin/env python3
"""
Test DEV_MODE automatic logging server selection
"""

import os
import time
import subprocess
import requests
from dotenv import load_dotenv

def test_dev_mode_logic():
    """Test that DEV_MODE correctly switches logging servers"""
    load_dotenv()
    
    print("🧪 Testing DEV_MODE logging server selection")
    print("=" * 45)
    
    # Test 1: DEV_MODE=true should use localhost:3031
    print("\n1️⃣ Testing DEV_MODE=true (should use localhost:3031)")
    os.environ['DEV_MODE'] = 'true'
    
    # Import app to trigger the logging setup
    import sys
    if 'app' in sys.modules:
        del sys.modules['app']  # Force reload
    
    try:
        from app import LOGGING_SERVER_URL, _dev_mode
        print(f"   DEV_MODE detected: {_dev_mode}")
        print(f"   Logging URL: {LOGGING_SERVER_URL}")
        
        if LOGGING_SERVER_URL == 'http://localhost:3031':
            print("   ✅ Correctly using local server in dev mode")
        else:
            print(f"   ❌ Expected localhost:3031, got {LOGGING_SERVER_URL}")
    except Exception as e:
        print(f"   ❌ Error testing dev mode: {e}")
    
    # Test 2: DEV_MODE=false should use env variable
    print("\n2️⃣ Testing DEV_MODE=false (should use LOGGING_SERVER_URL env)")
    os.environ['DEV_MODE'] = 'false'
    os.environ['LOGGING_SERVER_URL'] = 'https://example-cloud-server.run.app'
    
    # Force reload modules
    modules_to_reload = ['app', 'litellm_callback']
    for mod in modules_to_reload:
        if mod in sys.modules:
            del sys.modules[mod]
    
    try:
        from app import LOGGING_SERVER_URL as prod_url, _dev_mode as prod_dev_mode
        print(f"   DEV_MODE detected: {prod_dev_mode}")
        print(f"   Logging URL: {prod_url}")
        
        if prod_url == 'https://example-cloud-server.run.app':
            print("   ✅ Correctly using cloud server in production mode")
        else:
            print(f"   ❌ Expected cloud URL, got {prod_url}")
    except Exception as e:
        print(f"   ❌ Error testing production mode: {e}")
    
    print("\n" + "=" * 45)
    print("📋 Summary:")
    print("✅ DEV_MODE=true → localhost:3031 (local)")
    print("✅ DEV_MODE=false → LOGGING_SERVER_URL env (cloud)")

def test_with_local_server():
    """Test the complete flow with local server"""
    print("\n🚀 Testing complete flow with local server")
    print("=" * 40)
    
    # Set dev mode
    os.environ['DEV_MODE'] = 'true'
    
    # Start local logging server
    print("🔄 Starting local logging server...")
    try:
        server_process = subprocess.Popen([
            'python', 'run_logging_server_local.py'
        ], cwd=os.path.dirname(__file__))
        
        # Wait for server to start
        time.sleep(3)
        
        # Test server is running
        response = requests.get('http://localhost:3031/health', timeout=5)
        if response.status_code == 200:
            print("✅ Local server started successfully")
            
            # Now test that app.py connects to it
            print("🧪 Testing app.py connects to local server...")
            
            # Force reload app to pick up DEV_MODE
            import sys
            if 'app' in sys.modules:
                del sys.modules['app']
            
            from app import LOGGING_SERVER_URL, litellm_logger
            
            if LOGGING_SERVER_URL == 'http://localhost:3031':
                print("✅ app.py correctly configured for local logging")
                
                if litellm_logger:
                    print("✅ LiteLLM logger initialized")
                    
                    # Test a simple LiteLLM call
                    if os.getenv('OPENAI_API_KEY'):
                        print("🤖 Testing LiteLLM call with logging...")
                        import litellm
                        
                        response = litellm.completion(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Say 'logging test'"}],
                            max_tokens=10
                        )
                        
                        print(f"🤖 Response: {response.choices[0].message.content}")
                        
                        # Wait for all logs to be processed
                        print("⏳ Waiting for all logs to be processed...")
                        time.sleep(5)  # Give more time for async processing
                        
                        print("✅ LiteLLM call completed - check localhost:3031 logs!")
                    else:
                        print("⚠️  No OPENAI_API_KEY - skipping LiteLLM test")
                else:
                    print("❌ LiteLLM logger not initialized")
            else:
                print(f"❌ app.py using wrong URL: {LOGGING_SERVER_URL}")
        else:
            print(f"❌ Local server not responding: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing with local server: {e}")
    finally:
        # Clean up
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            print("🛑 Local server stopped")
        except:
            try:
                server_process.kill()
            except:
                pass

if __name__ == "__main__":
    test_dev_mode_logic()
    
    # Ask if user wants to test complete flow
    response = input("\n🎯 Test complete flow with local server? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        test_with_local_server()
