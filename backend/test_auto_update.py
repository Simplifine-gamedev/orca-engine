#!/usr/bin/env python3
"""
Test script for the auto-update system
"""

import requests
import json
import time

def test_update_endpoints():
    """Test all auto-update endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Orca Engine Auto-Update System")
    print("=" * 50)
    
    # Test 1: Health check (should include version)
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Health OK - Service: {health.get('service')}")
            print(f"   📱 Current Version: {health.get('version', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 2: Update status
    print("\n2. Testing update status...")
    try:
        response = requests.get(f"{base_url}/update/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Status retrieved")
            print(f"   📱 Current Version: {status.get('current_version')}")
            print(f"   🔄 Background Checker: {status.get('background_checker_running', False)}")
            print(f"   ⏱️  Check Interval: {status.get('check_interval')}s")
            print(f"   📅 Last Check: {status.get('last_check_human', 'Never')}")
        else:
            print(f"   ❌ Status failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Status error: {e}")
    
    # Test 3: Force update check
    print("\n3. Testing forced update check...")
    try:
        response = requests.post(f"{base_url}/update/check", 
                               json={"force": True, "platform": "mac"})
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Update check completed")
            print(f"   🔄 Update Available: {result.get('update_available', False)}")
            
            if result.get('update_available'):
                update_info = result.get('update_info', {})
                print(f"   📱 New Version: v{update_info.get('version')}")
                print(f"   💾 File Size: {update_info.get('file_size_mb')}MB")
                print(f"   ⚠️  Critical: {update_info.get('is_critical', False)}")
                print(f"   📝 Release Notes Preview: {update_info.get('release_notes', '')[:100]}...")
            else:
                print(f"   ✅ You have the latest version!")
        else:
            print(f"   ❌ Update check failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Update check error: {e}")
    
    # Test 4: Test via AI tool system
    print("\n4. Testing via AI tool system...")
    try:
        # Simulate what the AI would send
        messages = [
            {
                "role": "user",
                "content": "Check if there are any updates available for Orca Engine"
            }
        ]
        
        # This simulates the AI calling the tool
        tool_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "test_update_call",
                    "type": "function", 
                    "function": {
                        "name": "check_for_app_updates",
                        "arguments": json.dumps({
                            "force_check": True,
                            "show_notification": True
                        })
                    }
                }
            ]
        }
        messages.append(tool_call)
        
        response = requests.post(
            f"{base_url}/chat",
            json={
                "messages": messages,
                "model": "gpt-5"
            },
            headers={
                "Content-Type": "application/json",
                "X-Machine-ID": "test_machine_update",
                "X-Guest-Name": "Update Tester"
            },
            stream=True,
            timeout=30
        )
        
        print(f"   📡 Chat Response: {response.status_code}")
        
        if response.status_code == 200:
            # Parse streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("tool_executed") == "check_for_app_updates":
                            result = data.get("tool_result", {})
                            print(f"   ✅ Tool executed successfully")
                            print(f"   🔄 Update Available: {result.get('update_available', False)}")
                            
                            if result.get('show_popup'):
                                popup = result.get('popup_config', {})
                                print(f"   🔔 Popup: '{popup.get('title')}' - {popup.get('buttons')}")
                            
                            break
                        elif data.get("error"):
                            print(f"   ❌ Tool error: {data['error']}")
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"   ❌ Chat request failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ AI tool test error: {e}")
    
    print("\n🎯 Auto-Update System Test Complete!")
    print("\n💡 How it works:")
    print("   1. Backend checks GitHub releases every hour")
    print("   2. When update found, AI can call check_for_app_updates tool")
    print("   3. Tool returns popup configuration for frontend")
    print("   4. User sees 'Update Available' popup with 'Install Now/Later' buttons")
    print("   5. Download and install handled by platform-specific methods")

def test_github_api_directly():
    """Test GitHub API directly to see what's available"""
    print("\n🔗 Testing GitHub API directly...")
    
    try:
        response = requests.get("https://api.github.com/repos/Simplifine-gamedev/orca-engine/releases/latest")
        if response.status_code == 200:
            release = response.json()
            print(f"   ✅ Latest Release: {release['tag_name']}")
            print(f"   📅 Published: {release['published_at']}")
            print(f"   📦 Assets: {len(release.get('assets', []))}")
            
            for asset in release.get('assets', []):
                print(f"      - {asset['name']} ({round(asset['size']/(1024*1024), 1)}MB)")
        else:
            print(f"   ❌ GitHub API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ GitHub API error: {e}")

if __name__ == "__main__":
    test_github_api_directly()
    test_update_endpoints()
