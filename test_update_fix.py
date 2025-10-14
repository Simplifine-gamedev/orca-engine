#!/usr/bin/env python3
"""
Quick test script for update system fixes
Run this to test the update mechanism
"""

import os
import subprocess
import requests
import json

def test_version_detection():
    """Test version detection methods"""
    print("🔍 TESTING VERSION DETECTION")
    print("=" * 50)
    
    # Test git tag detection
    try:
        result = subprocess.run(['git', 'describe', '--exact-match', '--tags', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            tag = result.stdout.strip()
            print(f"✅ Git tag: {tag}")
        else:
            print("ℹ️  No git tag at HEAD")
    except Exception as e:
        print(f"❌ Git tag check failed: {e}")
    
    # Test git SHA detection  
    try:
        result = subprocess.run(['git', 'rev-parse', '--short=8', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            sha = result.stdout.strip()
            print(f"✅ Git SHA: 0.01.{sha}")
        else:
            print("❌ Git SHA detection failed")
    except Exception as e:
        print(f"❌ Git SHA check failed: {e}")
    
    # Check environment variable
    env_version = os.getenv('ORCA_VERSION')
    if env_version:
        print(f"✅ Environment ORCA_VERSION: {env_version}")
    else:
        print("ℹ️  ORCA_VERSION not set")
    
    return True

def test_backend_if_running():
    """Test backend update endpoints if running"""
    print("\n🌐 TESTING BACKEND (if running)")
    print("=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8080/health", timeout=3)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Backend online: {health.get('service', 'unknown')}")
            
            # Test update check
            update_response = requests.get("http://localhost:8080/update/check", timeout=5)
            if update_response.status_code == 200:
                update_data = update_response.json()
                current_version = update_data.get('current_version', 'unknown')
                update_available = update_data.get('update_available', False)
                print(f"✅ Update check: Current={current_version}, Update available={update_available}")
                
                if update_available:
                    update_info = update_data.get('update_info', {})
                    new_version = update_info.get('version', 'unknown')
                    print(f"   New version: {new_version}")
            else:
                print(f"❌ Update check failed: {update_response.status_code}")
                
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("ℹ️  Backend not running (start with: python3 backend/app.py)")
    except Exception as e:
        print(f"❌ Backend test error: {e}")

def show_manual_testing_steps():
    """Show manual testing steps"""
    print("\n🧪 MANUAL TESTING STEPS")
    print("=" * 50)
    
    steps = [
        "1. Set a test version: export ORCA_VERSION='1.0.0'",
        "2. Start backend: cd backend && python3 app.py",  
        "3. Open Godot editor - check if update notification appears",
        "4. If notification appears, check the version numbers shown",
        "5. Try the 'Download & Install' button",
        "6. After 'installation', restart and check if notification still appears",
        "7. Check backend logs for version detection messages"
    ]
    
    for step in steps:
        print(step)
    
    print(f"\n🐛 DEBUGGING WEIRD VERSION NUMBERS:")
    print("If you see nonsense versions like 'msg_0_2025-10-13T20-25-14':")
    print("- This means Engine::get_version_info() is returning junk data")
    print("- The fix forces git-based version detection instead")
    print("- Check console output for 'Version from git tag/SHA' messages")

def show_quick_fixes():
    """Show quick fixes users can apply"""
    print(f"\n⚡ QUICK FIXES YOU CAN TRY NOW:")
    print("=" * 50)
    
    print("1. SET ENVIRONMENT VERSION (temporary fix):")
    print("   export ORCA_VERSION='1.0.0'")
    print("   # Then restart Godot")
    
    print(f"\n2. CHECK YOUR CURRENT GIT VERSION:")
    print("   cd /Users/alikavoosi/Desktop/3d-design/GODOT/godot")
    print("   git rev-parse --short=8 HEAD")
    
    print(f"\n3. FORCE VERSION IN BACKEND:")
    print("   # Add this to backend/.env:")
    print("   echo 'ORCA_VERSION=1.0.0' >> backend/.env")
    
    print(f"\n4. MANUALLY MARK VERSION AS INSTALLED:")
    print("   curl -X POST http://localhost:8080/update/mark_installed \\")
    print("        -H 'Content-Type: application/json' \\") 
    print("        -d '{\"version\": \"0.01.9b45879b\"}'")
    
    print(f"\n5. CHECK BACKEND VERSION DETECTION:")
    print("   curl http://localhost:8080/update/check | jq")

if __name__ == "__main__":
    print("🔧 ORCA ENGINE UPDATE SYSTEM - QUICK TEST")
    print("=" * 60)
    
    test_version_detection()
    test_backend_if_running() 
    show_manual_testing_steps()
    show_quick_fixes()
    
    print(f"\n✨ The fixes should prevent constant update notifications!")
    print("🎯 Key improvement: Versions are now properly tracked and compared")
