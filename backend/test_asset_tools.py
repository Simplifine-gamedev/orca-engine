#!/usr/bin/env python3
"""
Test script for the Godot Asset Library integration tools.
This demonstrates Phase 1 of the asset management system.

Usage:
    python test_asset_tools.py
"""
import requests
import json
import tempfile
import os
import sys
from datetime import datetime

# Backend URL (adjust if needed)
BACKEND_URL = "http://localhost:8000"

def test_search_godot_assets():
    """Test searching the Godot Asset Library"""
    print("=" * 60)
    print("🔍 TESTING: search_godot_assets")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Search for shader assets",
            "query": "shader",
            "category": "shaders",
            "max_results": 5
        },
        {
            "name": "Search for FPS controllers",
            "query": "fps",
            "category": "scripts", 
            "max_results": 3
        },
        {
            "name": "Search for controller scripts",
            "query": "controller",
            "max_results": 5
        },
        {
            "name": "Search for 2D tools",
            "query": "tool",
            "category": "2d_tools",
            "max_results": 3
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print("-" * 40)
        
        # Prepare the chat request
        messages = [
            {
                "role": "user",
                "content": f"Search for Godot assets: {test['query']}"
            }
        ]
        
        # Simulate the tool call that the AI would make
        tool_call = {
            "role": "assistant", 
            "content": None,
            "tool_calls": [
                {
                    "id": f"test_call_{i}",
                    "type": "function",
                    "function": {
                        "name": "search_godot_assets",
                        "arguments": json.dumps(test)
                    }
                }
            ]
        }
        messages.append(tool_call)
        
        try:
            # Make request to backend
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "messages": messages,
                    "model": "gpt-5"
                },
                headers={
                    "Content-Type": "application/json",
                    "X-Machine-ID": "test_machine_123",
                    "X-Guest-Name": "Test User"
                },
                stream=True,
                timeout=60
            )
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Parse streaming response
                assets_found = []
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("tool_executed") == "search_godot_assets":
                                result = data.get("tool_result", {})
                                if result.get("success"):
                                    assets = result.get("assets", [])
                                    assets_found = assets
                                    print(f"✅ Found {len(assets)} assets")
                                    
                                    for j, asset in enumerate(assets[:3], 1):  # Show first 3
                                        print(f"   {j}. {asset.get('title', 'Unknown')} (ID: {asset.get('id', 'N/A')})")
                                        print(f"      Author: {asset.get('author', 'Unknown')}")
                                        print(f"      Category: {asset.get('category', 'Unknown')}")
                                        print(f"      Version: {asset.get('version', 'Unknown')}")
                                        print(f"      Rating: {asset.get('rating', 0)}")
                                        if asset.get('description'):
                                            desc = asset['description'][:100] + "..." if len(asset['description']) > 100 else asset['description']
                                            print(f"      Description: {desc}")
                                        print()
                                else:
                                    print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                                break
                            elif data.get("error"):
                                print(f"❌ Error: {data['error']}")
                                break
                        except json.JSONDecodeError:
                            continue
                
                # Store first asset ID for installation test
                if assets_found and not hasattr(test_search_godot_assets, 'sample_asset_id'):
                    test_search_godot_assets.sample_asset_id = assets_found[0].get('id')
                    test_search_godot_assets.sample_asset_name = assets_found[0].get('title')
                    print(f"💾 Saved asset for installation test: {test_search_godot_assets.sample_asset_name} (ID: {test_search_godot_assets.sample_asset_id})")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during test: {e}")
    
    return True

def test_install_godot_asset():
    """Test installing a Godot asset"""
    print("\n" + "=" * 60)
    print("📦 TESTING: install_godot_asset")
    print("=" * 60)
    
    # Check if we have a sample asset from the search test
    if not hasattr(test_search_godot_assets, 'sample_asset_id'):
        print("⚠️  No sample asset ID available. Running search first...")
        test_search_godot_assets()
    
    if not hasattr(test_search_godot_assets, 'sample_asset_id'):
        print("❌ Could not get a sample asset to install. Skipping installation test.")
        return False
    
    # Create a temporary project directory
    with tempfile.TemporaryDirectory() as temp_project:
        print(f"📁 Using temporary project directory: {temp_project}")
        
        asset_id = test_search_godot_assets.sample_asset_id
        asset_name = test_search_godot_assets.sample_asset_name
        
        print(f"📦 Installing asset: {asset_name} (ID: {asset_id})")
        print("-" * 40)
        
        # Prepare the chat request
        messages = [
            {
                "role": "user", 
                "content": f"Install the asset with ID {asset_id} to my project"
            }
        ]
        
        # Simulate the tool call
        tool_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "test_install_1",
                    "type": "function",
                    "function": {
                        "name": "install_godot_asset",
                        "arguments": json.dumps({
                            "asset_id": asset_id,
                            "project_path": temp_project,
                            "install_location": "addons/",
                            "create_backup": True
                        })
                    }
                }
            ]
        }
        messages.append(tool_call)
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "messages": messages,
                    "model": "gpt-5"
                },
                headers={
                    "Content-Type": "application/json",
                    "X-Machine-ID": "test_machine_123",
                    "X-Guest-Name": "Test User"
                },
                stream=True,
                timeout=120  # Longer timeout for downloads
            )
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("tool_executed") == "install_godot_asset":
                                result = data.get("tool_result", {})
                                if result.get("success"):
                                    print("✅ Asset installed successfully!")
                                    install_info = result.get("installation_info", {})
                                    print(f"   📁 Installed to: {install_info.get('installed_to', 'Unknown')}")
                                    print(f"   📄 Files extracted: {install_info.get('files_extracted', 'Unknown')}")
                                    print(f"   🔌 Is plugin: {install_info.get('is_plugin', 'Unknown')}")
                                    print(f"   📝 Version: {install_info.get('version', 'Unknown')}")
                                    print(f"   👤 Author: {install_info.get('author', 'Unknown')}")
                                    
                                    # Verify files were actually extracted
                                    install_path = install_info.get('installed_to')
                                    if install_path and os.path.exists(install_path):
                                        files = []
                                        for root, dirs, filenames in os.walk(install_path):
                                            files.extend([os.path.join(root, f) for f in filenames])
                                        print(f"   📋 Verified {len(files)} files on disk")
                                        
                                        # Show some example files
                                        if files:
                                            print("   📄 Sample files:")
                                            for f in files[:5]:  # Show first 5 files
                                                rel_path = os.path.relpath(f, install_path)
                                                print(f"      - {rel_path}")
                                            if len(files) > 5:
                                                print(f"      ... and {len(files) - 5} more files")
                                    
                                else:
                                    print(f"❌ Installation failed: {result.get('error', 'Unknown error')}")
                                break
                            elif data.get("error"):
                                print(f"❌ Error: {data['error']}")
                                break
                            elif data.get("status") == "tool_starting":
                                print("⏳ Starting asset installation...")
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during installation test: {e}")
    
    return True

def test_direct_api_calls():
    """Test the internal functions directly (without streaming)"""
    print("\n" + "=" * 60)
    print("🔧 TESTING: Direct API calls")
    print("=" * 60)
    
    # Test direct search
    from app import search_godot_assets_internal
    
    print("📋 Testing direct search_godot_assets_internal call...")
    result = search_godot_assets_internal({
        "query": "dialogue",
        "category": "plugins", 
        "max_results": 3
    })
    
    print(f"✅ Direct search result: {result.get('success', False)}")
    if result.get('success'):
        print(f"   Found {result.get('total_found', 0)} assets")
        for asset in result.get('assets', [])[:2]:
            print(f"   - {asset.get('title')} by {asset.get('author')}")
    else:
        print(f"   Error: {result.get('error')}")
    
    return result.get('success', False)

def run_all_tests():
    """Run all asset tool tests"""
    print("🚀 STARTING GODOT ASSET LIBRARY TESTS")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Backend URL: {BACKEND_URL}")
    
    tests = [
        ("Search Assets", test_search_godot_assets),
        ("Install Asset", test_install_godot_asset),
        ("Direct API", test_direct_api_calls)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"
            print(f"❌ Test '{test_name}' failed with error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results.items():
        print(f"{test_name:<20}: {result}")
    
    passed = sum(1 for r in results.values() if r.startswith("✅"))
    total = len(results)
    print(f"\n🏆 {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Asset tools are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Check if backend is running
    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Backend not healthy. Status: {health_response.status_code}")
            sys.exit(1)
        print("✅ Backend is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BACKEND_URL}: {e}")
        print("💡 Make sure the backend is running: python app.py")
        sys.exit(1)
    
    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
