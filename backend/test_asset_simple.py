#!/usr/bin/env python3
"""
Simple standalone test of Godot Asset Library integration
Tests both search and install with a known working asset.
"""
import os
import sys
import tempfile
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple():
    """Simple test of search and install"""
    print("🚀 SIMPLE ASSET TEST")
    print("=" * 50)
    
    try:
        from app import search_godot_assets_internal, install_godot_asset_internal
        print("✅ Successfully imported functions")
    except ImportError as e:
        print(f"❌ Cannot import functions: {e}")
        return False
    
    # Test 1: Search for a common term
    print("\n🔍 Testing search...")
    search_result = search_godot_assets_internal({
        "query": "controller",
        "max_results": 3
    })
    
    if not search_result.get('success'):
        print(f"❌ Search failed: {search_result.get('error')}")
        return False
    
    assets = search_result.get('assets', [])
    print(f"✅ Found {len(assets)} assets:")
    
    if not assets:
        print("❌ No assets found")
        return False
    
    for i, asset in enumerate(assets[:2], 1):
        print(f"   {i}. {asset.get('title')} (ID: {asset.get('id')})")
        print(f"      By: {asset.get('author')}")
    
    # Test 2: Try to install the first asset
    print(f"\n📦 Testing installation of: {assets[0].get('title')}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        install_result = install_godot_asset_internal({
            "asset_id": assets[0].get('id'),
            "project_path": temp_dir,
            "install_location": "addons/",
            "create_backup": True
        })
        
        if install_result.get('success'):
            print("✅ Installation successful!")
            info = install_result.get('installation_info', {})
            print(f"   📁 Installed to: {info.get('installed_to')}")
            print(f"   📄 Files: {info.get('files_extracted')}")
            print(f"   🔌 Plugin: {info.get('is_plugin')}")
            
            # Verify files exist
            install_path = info.get('installed_to')
            if install_path and os.path.exists(install_path):
                file_count = sum(len(files) for _, _, files in os.walk(install_path))
                print(f"   ✅ Verified {file_count} files on disk")
                return True
            else:
                print("❌ Installation path not found")
                return False
        else:
            error = install_result.get('error', 'Unknown error')
            print(f"❌ Installation failed: {error}")
            
            # If it's a 404, this might be expected for old assets
            if "404" in str(error):
                print("   💡 This is likely an old asset with a broken download link")
                print("   🔄 Trying next asset...")
                
                # Try the second asset if available
                if len(assets) > 1:
                    print(f"\n📦 Trying second asset: {assets[1].get('title')}")
                    install_result2 = install_godot_asset_internal({
                        "asset_id": assets[1].get('id'),
                        "project_path": temp_dir,
                        "install_location": "addons/",
                        "create_backup": True
                    })
                    
                    if install_result2.get('success'):
                        print("✅ Second asset installation successful!")
                        return True
                    else:
                        print(f"❌ Second asset also failed: {install_result2.get('error')}")
            
            return False

def main():
    """Main test function"""
    print("Testing Godot Asset Library Integration")
    print("This test verifies both search and installation functions")
    
    success = test_simple()
    
    if success:
        print("\n🎉 SUCCESS! Asset tools are working correctly.")
        print("\n💡 What you can do now:")
        print("   1. Start your backend server: python app.py")
        print("   2. In your Godot AI chat, try:")
        print("      - 'Search for shader assets'")
        print("      - 'Install a controller script for my project'") 
        print("      - 'Find me some 2D tools'")
        print("\n📝 The AI will now be able to:")
        print("   ✅ Search the Godot Asset Library")
        print("   ✅ Browse available plugins, templates, and tools")
        print("   ✅ Install assets directly to your project")
        print("   ✅ Handle backups and plugin detection")
    else:
        print("\n❌ Some functionality may not work as expected.")
        print("   This could be due to network issues or outdated asset links.")
        print("   The search functionality appears to be working though!")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)

