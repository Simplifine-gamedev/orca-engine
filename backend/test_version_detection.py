#!/usr/bin/env python3
"""
Test script to verify version detection is working properly
Run this to check what version the system detects
"""

import os
import sys
import subprocess

def get_current_version():
    """Get current version matching the GitHub workflow format (0.01.{SHA})"""
    def get_repo_root():
        """Find the git repository root directory"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != '/':
            if os.path.exists(os.path.join(current, '.git')):
                return current
            current = os.path.dirname(current)
        # Fallback to parent directory of backend
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Method 1: Environment variable (for deployments) - highest priority
        env_version = os.getenv('ORCA_VERSION')
        if env_version:
            print(f"Version from environment: {env_version}")
            return env_version
        
        # Method 2: Generate version matching GitHub workflow format
        # This matches the format used in workflows: 0.01.{SHORT_SHA}
        result = subprocess.run([
            'git', 'rev-parse', '--short=8', 'HEAD'
        ], capture_output=True, text=True, cwd=get_repo_root())
        
        if result.returncode == 0:
            short_sha = result.stdout.strip()
            version = f"0.01.{short_sha}"
            print(f"Version from git SHA (matching workflow): {version}")
            return version
                    
    except Exception as e:
        print(f"Git version detection failed: {e}")
    
    # Method 3: Read from version.py as fallback
    try:
        version_file = os.path.join(os.path.dirname(__file__), '..', 'version.py')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                content = f.read()
                # Extract version from version.py
                for line in content.split('\n'):
                    if line.strip().startswith('version') and '=' in line:
                        version = line.split('=')[1].strip().strip('"\'')
                        print(f"Version from version.py: {version}")
                        return version
    except Exception as e:
        print(f"Error reading version.py: {e}")
        
    # Final fallback
    print("Using fallback version 0.01.unknown")
    return '0.01.unknown'

def format_version_for_display(version):
    """Format version string for user-friendly display"""
    if not version:
        return "Unknown"
    
    # Handle the 0.01.{SHA} format used by workflows
    if version.startswith('0.01.') and len(version) > 5:
        sha_part = version.split('0.01.')[1]
        if sha_part == 'unknown':
            return "0.01 (Unknown)"
        else:
            return f"0.01.{sha_part[:8]}"  # Show first 8 chars of SHA
        
    # Handle development builds
    if '-dev.' in version:
        # 1.0.0-dev.5+g1234567 -> "1.0.0 (Development Build)"
        base = version.split('-dev.')[0]
        return f"{base} (Development Build)"
    elif '-beta.' in version:
        # 1.0.0-beta.1 -> "1.0.0 Beta 1"
        parts = version.split('-beta.')
        if len(parts) == 2:
            return f"{parts[0]} Beta {parts[1].split('+')[0]}"
    elif '-alpha.' in version:
        # 1.0.0-alpha.1 -> "1.0.0 Alpha 1"
        parts = version.split('-alpha.')
        if len(parts) == 2:
            return f"{parts[0]} Alpha {parts[1].split('+')[0]}"
    elif '-rc.' in version:
        # 1.0.0-rc.1 -> "1.0.0 Release Candidate 1"
        parts = version.split('-rc.')
        if len(parts) == 2:
            return f"{parts[0]} RC {parts[1].split('+')[0]}"
    elif '-unknown' in version:
        return version.replace('-unknown', ' (Unknown)')
    
    # Remove build metadata if present
    if '+' in version:
        version = version.split('+')[0]
        
    return version

def test_version_detection():
    """Test the version detection system"""
    print("=" * 60)
    print("🔍 ORCA ENGINE VERSION DETECTION TEST")
    print("=" * 60)
    
    # Get current version
    current_version = get_current_version()
    
    print(f"\n📌 Current Version: {current_version}")
    print(f"📌 Display Version: {format_version_for_display(current_version)}")
    
    # Check git status
    print("\n📊 Git Information:")
    try:
        # Check for tags
        result = subprocess.run(['git', 'describe', '--tags', '--always'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Git Describe: {result.stdout.strip()}")
        
        # Check current commit
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Current Commit: {result.stdout.strip()}")
        
        # List recent tags
        result = subprocess.run(['git', 'tag', '-l', '--sort=-v:refname'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            tags = result.stdout.strip().split('\n')[:5]
            if tags and tags[0]:
                print(f"  Recent Tags: {', '.join(tags)}")
            else:
                print("  Recent Tags: No tags found")
                
    except Exception as e:
        print(f"  Git info unavailable: {e}")
    
    # Check environment
    print("\n🔧 Environment:")
    print(f"  ORCA_VERSION env: {os.getenv('ORCA_VERSION', 'Not set')}")
    
    # Check version.py
    try:
        import version
        print(f"  version.py: {version.version}")
    except:
        print("  version.py: Not found or error")
    
    # Test version comparison
    print("\n🔄 Version Comparison Tests:")
    test_versions = [
        ("1.0.0", "1.0.1", True),
        ("1.0.0", "2.0.0", True),
        ("1.0.0", "1.0.0", False),
        ("1.0.0", "0.9.9", False),
        ("1.0.0-beta.1", "1.0.0", True),
        ("1.0.0", "1.0.0-beta.1", False),
        ("1.0.0-dev.5+abc123", "1.0.1", True),
    ]
    
    # Skip version comparison tests for now (would need to import semver)
    print("  (Version comparison tests skipped - requires semver module)")
    
    # Test version formatting
    print("\n🎨 Version Display Formatting:")
    test_formats = [
        "1.0.0",
        "1.0.0-beta.1",
        "1.0.0-alpha.2",
        "1.0.0-rc.1",
        "1.0.0-dev.5+g1234567",
        "1.0.0-unknown",
    ]
    
    for ver in test_formats:
        display = format_version_for_display(ver)
        print(f"  {ver} -> {display}")
    
    # Test update type detection
    print("\n📈 Update Type Detection:")
    print("  (Update type detection tests skipped - requires semver module)")
    
    print("\n" + "=" * 60)
    print("✅ Version detection test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_version_detection()
