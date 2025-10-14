#!/usr/bin/env python3
"""
Simple version test that doesn't require external dependencies
"""

import os
import subprocess

def test_version_detection():
    """Test the same version detection logic as the backend"""
    print("🔍 TESTING VERSION DETECTION (No External Dependencies)")
    print("=" * 60)
    
    # Test environment variable
    env_version = os.getenv('ORCA_VERSION')
    print(f"Environment ORCA_VERSION: {env_version or 'Not set'}")
    
    # Test git tag detection  
    try:
        result = subprocess.run(['git', 'describe', '--exact-match', '--tags', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            tag = result.stdout.strip()
            if tag.startswith('v'):
                tag = tag[1:]
            print(f"Git tag: {tag}")
        else:
            print("Git tag: Not on a tagged commit")
    except Exception as e:
        print(f"Git tag: Error - {e}")
    
    # Test git SHA detection
    try:
        result = subprocess.run(['git', 'rev-parse', '--short=8', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            sha = result.stdout.strip()
            print(f"Git SHA: 0.01.{sha}")
        else:
            print("Git SHA: Error getting commit hash")
    except Exception as e:
        print(f"Git SHA: Error - {e}")
    
    # Show what version the fixed backend would detect
    print(f"\n🎯 BACKEND VERSION LOGIC SIMULATION:")
    print("-" * 40)
    
    if env_version:
        detected_version = env_version
        method = "Environment variable"
    else:
        try:
            # Try tag first
            result = subprocess.run(['git', 'describe', '--exact-match', '--tags', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                tag = result.stdout.strip()
                if tag.startswith('v'):
                    tag = tag[1:]
                detected_version = tag
                method = "Git tag"
            else:
                # Fall back to SHA
                result = subprocess.run(['git', 'rev-parse', '--short=8', 'HEAD'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    sha = result.stdout.strip()
                    detected_version = f"0.01.{sha}"
                    method = "Git SHA"
                else:
                    detected_version = "1.0.0"
                    method = "Fallback"
        except:
            detected_version = "1.0.0"
            method = "Fallback"
    
    print(f"Detected Version: {detected_version}")
    print(f"Detection Method: {method}")
    
    return detected_version

def show_testing_instructions(detected_version):
    """Show step-by-step testing instructions"""
    print(f"\n🧪 STEP-BY-STEP TESTING:")
    print("=" * 60)
    
    print(f"1. SET VERSION TO STOP NOTIFICATIONS:")
    print(f"   export ORCA_VERSION='{detected_version}'")
    print(f"   # This tells the system you have the 'latest' version")
    
    print(f"\n2. START BACKEND:")
    print(f"   cd backend")  
    print(f"   python3 app.py")
    print(f"   # Look for: 'AUTO_UPDATE: Version from environment: {detected_version}'")
    
    print(f"\n3. REBUILD & TEST GODOT:")
    print(f"   # Rebuild Godot with the C++ changes")
    print(f"   # Launch editor - should show '{detected_version}' instead of nonsense")
    
    print(f"\n4. TEST UPDATE CYCLE:")
    print(f"   # Change ORCA_VERSION to something else:")
    print(f"   export ORCA_VERSION='0.9.0'")
    print(f"   # Restart - should now show update available")
    
    print(f"\n5. SIMULATE SUCCESSFUL UPDATE:")
    print(f"   # After 'installing' an update, run:")
    print(f"   curl -X POST http://localhost:8080/update/mark_installed \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"version\": \"1.0.1\"}}'")
    print(f"   # Then set: export ORCA_VERSION='1.0.1'")
    print(f"   # Restart - should show no updates")

def show_debugging_commands():
    """Show debugging commands"""
    print(f"\n🔍 DEBUGGING COMMANDS:")
    print("=" * 60)
    
    print(f"# Check what version numbers are being compared:")
    print("curl -s http://localhost:8080/update/check | python3 -c \"import sys, json; data=json.load(sys.stdin); print('Backend Current:', data.get('current_version')); print('Update Available:', data.get('update_available'));\"")
    
    print(f"\n# Check installed version history:")
    print(f"cat backend/.installed_versions.json 2>/dev/null || echo 'No version history yet'")
    
    print(f"\n# Check update check history:")  
    print(f"cat backend/.last_update_check.json 2>/dev/null || echo 'No check history yet'")
    
    print(f"\n# Clear version history (force fresh start):")
    print(f"rm -f backend/.installed_versions.json backend/.last_update_check.json")

if __name__ == "__main__":
    detected = test_version_detection()
    show_testing_instructions(detected) 
    show_debugging_commands()
    
    print(f"\n🎯 QUICK FIX FOR YOUR CURRENT ISSUE:")
    print("=" * 60)
    print(f"The nonsense version suggests git detection failed.")
    print(f"Try this immediately:")
    print(f"")
    print(f"export ORCA_VERSION='1.0.0'")
    print(f"# Then restart Godot - should show '1.0.0' instead of nonsense")
