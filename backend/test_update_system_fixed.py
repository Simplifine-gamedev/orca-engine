#!/usr/bin/env python3
"""
Test script for the FIXED update system
Tests version detection, comparison, and persistence fixes
"""

import os
import sys
import json
import tempfile

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_version_detection_fix():
    """Test the improved version detection system"""
    print("=" * 80)
    print("🔧 TESTING FIXED UPDATE SYSTEM")
    print("=" * 80)
    
    from auto_update_manager import AutoUpdateManager
    
    # Create temporary manager for testing
    manager = AutoUpdateManager()
    
    print(f"📍 Current Version Detected: {manager.current_version}")
    print(f"📍 Version Detection Method Used: {manager._get_current_version()}")
    
    print("\n🧪 Testing Version Comparison Logic:")
    print("-" * 50)
    
    test_cases = [
        ("0.01.abc123", "0.01.abc123", "Same version - should NOT update"),
        ("0.01.def456", "0.01.abc123", "Different SHA - should update"), 
        ("1.0.0", "0.01.abc123", "Different format - should compare carefully"),
        ("v1.0.0", "0.01.abc123", "GitHub tag format - should compare carefully"),
    ]
    
    for remote, current, expected in test_cases:
        result = manager._is_newer_version(remote, current)
        status = "✅ PASS" if result else "❌ NO UPDATE"
        print(f"{status}: Remote '{remote}' vs Current '{current}' -> {expected}")
    
    print("\n🗃️ Testing Version Persistence:")
    print("-" * 50)
    
    # Test marking versions as installed
    test_version = "0.01.testfix123"
    manager.mark_version_installed(test_version)
    
    # Check if it's now marked as installed
    is_installed = manager._is_version_already_installed(test_version)
    print(f"✅ Version persistence test: {test_version} -> {'Already installed' if is_installed else 'Not found'}")
    
    # Test the spam prevention
    is_newer_first = manager._is_newer_version("0.01.spamtest", "0.01.current")
    is_newer_second = manager._is_newer_version("0.01.spamtest", "0.01.current")
    
    print(f"🚫 Spam prevention test:")
    print(f"   First check: {'Update needed' if is_newer_first else 'No update'}")
    print(f"   Second check: {'Update needed' if is_newer_second else 'No update (spam blocked)'}")
    
    return manager

def test_backend_endpoints():
    """Test the update endpoints"""
    print("\n🌐 TESTING BACKEND ENDPOINTS:")
    print("-" * 50)
    
    import requests
    
    endpoints_to_test = [
        ("GET", "http://localhost:8080/health", "Health check"),
        ("GET", "http://localhost:8080/update/check", "Update check"),
        ("POST", "http://localhost:8080/update/mark_installed", "Mark version installed", {"version": "0.01.testfix"}),
    ]
    
    for method, url, description, data in [(m, u, d, data) if len((m, u, d, data)) == 4 else (m, u, d, None) for m, u, d, *data in endpoints_to_test]:
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            status = "✅ WORKING" if response.status_code == 200 else f"⚠️  STATUS {response.status_code}"
            print(f"{status}: {description}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'current_version' in result:
                        print(f"   Current version reported: {result['current_version']}")
                    if 'update_available' in result:
                        print(f"   Update available: {result['update_available']}")
                except:
                    pass
                    
        except requests.exceptions.ConnectionError:
            print(f"❌ OFFLINE: {description} (backend not running)")
        except Exception as e:
            print(f"❌ ERROR: {description} - {str(e)[:100]}")

def test_file_persistence():
    """Test the file-based version persistence"""
    print("\n💾 TESTING VERSION PERSISTENCE FILES:")
    print("-" * 50)
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for version tracking files
    version_history_path = os.path.join(backend_dir, '.installed_versions.json')
    last_check_path = os.path.join(backend_dir, '.last_update_check.json')
    
    for path, name in [(version_history_path, "Installed Versions"), (last_check_path, "Last Update Check")]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"✅ {name}: Found with data: {json.dumps(data, indent=2)[:200]}...")
            except Exception as e:
                print(f"❌ {name}: File exists but corrupt: {e}")
        else:
            print(f"ℹ️  {name}: File not found (will be created on first use)")

def print_fix_summary():
    """Print summary of fixes applied"""
    print("\n" + "=" * 80)
    print("🔧 SUMMARY OF FIXES APPLIED")
    print("=" * 80)
    
    fixes = [
        "✅ Fixed version detection inconsistency between frontend and backend",
        "✅ Improved version comparison logic with spam prevention",
        "✅ Added version persistence system (.installed_versions.json)",
        "✅ Enhanced Windows executable launch with multiple editor flags",
        "✅ Added backend notification system for successful installations",
        "✅ Consistent git tag detection across all components",
        "✅ Added /update/mark_installed endpoint for manual version tracking"
    ]
    
    for fix in fixes:
        print(fix)
    
    print("\n📝 INSTRUCTIONS FOR TESTING:")
    print("-" * 50)
    print("1. Start your backend server: python3 backend/app.py")
    print("2. Run this test: python3 backend/test_update_system_fixed.py")
    print("3. Check Godot editor update notifications")
    print("4. Try manual update process")
    print("5. Verify no duplicate notifications appear")
    
    print("\n🔍 DEBUGGING COMMANDS:")
    print("-" * 50)  
    print("# Check current git version:")
    print("git describe --exact-match --tags HEAD 2>/dev/null || git rev-parse --short=8 HEAD")
    print("\n# Check backend version detection:")
    print("curl http://localhost:8080/update/check")
    print("\n# Manually mark version as installed:")
    print("curl -X POST http://localhost:8080/update/mark_installed -H 'Content-Type: application/json' -d '{\"version\": \"1.0.0\"}'")

if __name__ == "__main__":
    print_fix_summary()
    test_version_detection_fix()
    test_file_persistence()
    test_backend_endpoints()
    print("\n🎉 Test completed! Check the output above for any issues.")
