#!/usr/bin/env python3
"""
Direct test of Godot Asset Library functions (Phase 1)
Tests the internal functions directly without requiring the backend server.

Usage:
    python test_asset_functions_direct.py
"""
import os
import sys
import tempfile
import json
from datetime import datetime

# Add the backend directory to path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_search_assets_direct():
    """Test the search function directly"""
    print("=" * 60)
    print("🔍 TESTING: search_godot_assets_internal (Direct)")
    print("=" * 60)
    
    # Import the function
    try:
        from app import search_godot_assets_internal
    except ImportError as e:
        print(f"❌ Cannot import search function: {e}")
        return False
    
    test_cases = [
        {
            "name": "Search for shader assets",
            "query": "shader",
            "category": "shaders", 
            "max_results": 5
        },
        {
            "name": "Search for controller scripts",
            "query": "controller",
            "category": "scripts",
            "max_results": 3
        },
        {
            "name": "General search for FPS",
            "query": "fps",
            "max_results": 3
        }
    ]
    
    all_passed = True
    sample_assets = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print("-" * 40)
        
        try:
            result = search_godot_assets_internal(test)
            
            if result.get('success'):
                assets = result.get('assets', [])
                print(f"✅ Found {len(assets)} assets")
                
                # Show details of first few assets
                for j, asset in enumerate(assets[:3], 1):
                    print(f"   {j}. {asset.get('title', 'Unknown')} (ID: {asset.get('id', 'N/A')})")
                    print(f"      📧 Author: {asset.get('author', 'Unknown')}")
                    print(f"      📂 Category: {asset.get('category', 'Unknown')}")
                    print(f"      🏷️  Version: {asset.get('version', 'Unknown')}")
                    print(f"      ⭐ Rating: {asset.get('rating', 0)}")
                    if asset.get('description'):
                        desc = asset['description'][:80] + "..." if len(asset['description']) > 80 else asset['description']
                        print(f"      📝 Description: {desc}")
                    if asset.get('download_url'):
                        print(f"      🔗 Download: {asset['download_url'][:50]}...")
                    print()
                
                # Save some assets for installation test
                if assets and len(sample_assets) < 2:
                    sample_assets.extend(assets[:2])
                    
            else:
                print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Exception during search: {e}")
            all_passed = False
    
    # Save sample assets for next test
    test_search_assets_direct.sample_assets = sample_assets
    return all_passed

def test_install_asset_direct():
    """Test the install function directly"""
    print("\n" + "=" * 60)
    print("📦 TESTING: install_godot_asset_internal (Direct)")
    print("=" * 60)
    
    # Import the function
    try:
        from app import install_godot_asset_internal
    except ImportError as e:
        print(f"❌ Cannot import install function: {e}")
        return False
    
    # Get sample assets from search test
    if not hasattr(test_search_assets_direct, 'sample_assets') or not test_search_assets_direct.sample_assets:
        print("⚠️  No sample assets available. Running search first...")
        test_search_assets_direct()
    
    sample_assets = getattr(test_search_assets_direct, 'sample_assets', [])
    if not sample_assets:
        print("❌ No assets available for installation test")
        return False
    
    # Use the first available asset
    test_asset = sample_assets[0]
    asset_id = test_asset.get('id')
    asset_name = test_asset.get('title', 'Unknown Asset')
    
    if not asset_id:
        print("❌ No valid asset ID found")
        return False
    
    # Create temporary project directory
    with tempfile.TemporaryDirectory() as temp_project:
        print(f"📁 Using temporary project: {temp_project}")
        print(f"📦 Installing: {asset_name} (ID: {asset_id})")
        print("-" * 40)
        
        try:
            result = install_godot_asset_internal({
                "asset_id": asset_id,
                "project_path": temp_project,
                "install_location": "addons/",
                "create_backup": True
            })
            
            if result.get('success'):
                print("✅ Asset installed successfully!")
                install_info = result.get('installation_info', {})
                
                print(f"   📁 Installed to: {install_info.get('installed_to', 'Unknown')}")
                print(f"   📄 Files extracted: {install_info.get('files_extracted', 'Unknown')}")
                print(f"   🔌 Is plugin: {install_info.get('is_plugin', False)}")
                print(f"   📝 Version: {install_info.get('version', 'Unknown')}")
                print(f"   👤 Author: {install_info.get('author', 'Unknown')}")
                
                if install_info.get('plugin_config'):
                    print(f"   ⚙️  Plugin config: {install_info['plugin_config']}")
                
                # Verify installation by checking files
                install_path = install_info.get('installed_to')
                if install_path and os.path.exists(install_path):
                    files = []
                    for root, dirs, filenames in os.walk(install_path):
                        files.extend([os.path.join(root, f) for f in filenames])
                    
                    print(f"   ✅ Verified {len(files)} files on disk")
                    
                    if files:
                        print("   📄 Sample installed files:")
                        for f in files[:5]:  # Show first 5 files
                            rel_path = os.path.relpath(f, install_path)
                            file_size = os.path.getsize(f)
                            print(f"      - {rel_path} ({file_size} bytes)")
                        if len(files) > 5:
                            print(f"      ... and {len(files) - 5} more files")
                    
                    return True
                else:
                    print(f"⚠️  Installation path not found: {install_path}")
                    return False
            else:
                print(f"❌ Installation failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Exception during installation: {e}")
            return False

def test_api_connectivity():
    """Test if we can connect to Godot Asset Library API"""
    print("\n" + "=" * 60)
    print("🌐 TESTING: Godot Asset Library API Connectivity")
    print("=" * 60)
    
    import requests
    
    try:
        # Test basic API connection
        print("📡 Testing API connectivity...")
        response = requests.get(
            "https://godotengine.org/asset-library/api/asset",
            params={"filter": "test", "max_results": 1},
            timeout=10
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            total_results = len(data.get('result', []))
            print(f"   ✅ API is accessible")
            print(f"   📊 Sample query returned {total_results} results")
            
            if total_results > 0:
                sample = data['result'][0]
                print(f"   📦 Sample asset: {sample.get('title', 'Unknown')}")
            
            return True
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Cannot connect to API: {e}")
        return False

def run_direct_tests():
    """Run all direct function tests"""
    print("🚀 STARTING DIRECT ASSET FUNCTION TESTS")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    tests = [
        ("API Connectivity", test_api_connectivity),
        ("Search Assets", test_search_assets_direct), 
        ("Install Asset", test_install_asset_direct)
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
            import traceback
            traceback.print_exc()
    
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
        print("🎉 All tests passed! Asset functions are working correctly.")
        print("\n💡 Next steps:")
        print("   1. Start the backend server: python app.py")
        print("   2. Test the full integration: python test_asset_tools.py")
        print("   3. Try the tools in your Godot AI chat!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_direct_tests()
    sys.exit(0 if success else 1)
