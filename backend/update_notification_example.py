#!/usr/bin/env python3
"""
Example of how the Orca Engine app can integrate with the auto-update system
Shows how to display the update popup and handle user responses
"""

import requests
import json
from typing import Dict, Any

class UpdateNotificationHandler:
    """Handles update notifications in the Orca Engine app"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
    
    def check_for_updates_on_startup(self) -> Dict[str, Any]:
        """Check for updates when app starts (non-blocking)"""
        try:
            response = requests.get(f"{self.backend_url}/update/check?force=false", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"UPDATE_STARTUP: Failed to check for updates: {e}")
            return {"success": False, "error": str(e)}
    
    def show_update_popup(self, update_info: Dict[str, Any]) -> str:
        """
        Show update popup to user and return their choice
        This would be implemented in your Orca Engine UI system
        """
        # This is pseudocode for how your Orca Engine app would show the popup
        
        version = update_info.get('version', 'unknown')
        current_version = update_info.get('current_version', 'unknown')
        file_size = update_info.get('file_size_mb', 0)
        is_critical = update_info.get('is_critical', False)
        release_notes = update_info.get('release_notes', '')
        
        popup_title = "Critical Update Available" if is_critical else "Update Available"
        
        popup_message = f"""Orca Engine v{version} is now available.

Current version: v{current_version}
Download size: {file_size}MB

What's new:
{release_notes[:200]}{'...' if len(release_notes) > 200 else ''}"""
        
        # In real implementation, this would show actual UI popup
        print(f"\n🔔 {popup_title}")
        print("=" * 50)
        print(popup_message)
        print("\nChoices:")
        print("1. Install Now")
        print("2. Later")
        
        # Simulate user choice (in real app, this would be actual user input)
        choice = input("\nEnter choice (1/2): ").strip()
        
        if choice == "1":
            return "install_now"
        else:
            return "later"
    
    def download_and_install_update(self, update_info: Dict[str, Any]) -> Dict[str, Any]:
        """Download and install the update"""
        try:
            version = update_info.get('version')
            
            print(f"📥 Downloading Orca Engine v{version}...")
            
            # Download the update
            download_response = requests.post(
                f"{self.backend_url}/update/download",
                json={"version": version},
                timeout=300  # 5 minutes for download
            )
            
            if download_response.status_code != 200:
                return {"success": False, "error": "Download failed"}
            
            download_result = download_response.json()
            
            if not download_result.get('success'):
                return download_result
            
            download_path = download_result.get('download_path')
            print(f"✅ Downloaded to: {download_path}")
            
            # Install the update
            print("🔧 Installing update...")
            install_response = requests.post(
                f"{self.backend_url}/update/install",
                json={
                    "download_path": download_path,
                    "restart_app": True
                },
                timeout=60
            )
            
            if install_response.status_code != 200:
                return {"success": False, "error": "Install request failed"}
            
            install_result = install_response.json()
            
            if install_result.get('success'):
                print("✅ Update installed successfully!")
                
                if install_result.get('requires_exit'):
                    print("🔄 App will restart automatically...")
                    # In real app, you would exit here
                    return {"success": True, "action": "restart_required"}
                else:
                    print("ℹ️  Manual restart may be required")
                    return {"success": True, "action": "restart_recommended"}
            else:
                return install_result
                
        except Exception as e:
            print(f"UPDATE_INSTALL: Error: {e}")
            return {"success": False, "error": str(e)}

def demo_update_flow():
    """Demonstrate the complete update flow"""
    print("🎮 Orca Engine Update Flow Demo")
    print("=" * 40)
    
    handler = UpdateNotificationHandler()
    
    # 1. Check for updates on startup
    print("1. Checking for updates on app startup...")
    update_check = handler.check_for_updates_on_startup()
    
    if not update_check.get('success'):
        print(f"   ❌ Update check failed: {update_check.get('error')}")
        return
    
    if not update_check.get('update_available'):
        print("   ✅ No updates available - you have the latest version!")
        return
    
    # 2. Show update popup
    print("\n2. Update available - showing popup...")
    update_info = update_check.get('update_info', {})
    user_choice = handler.show_update_popup(update_info)
    
    # 3. Handle user choice
    if user_choice == "install_now":
        print("\n3. User chose 'Install Now' - proceeding with update...")
        result = handler.download_and_install_update(update_info)
        
        if result.get('success'):
            print("🎉 Update completed successfully!")
        else:
            print(f"❌ Update failed: {result.get('error')}")
    else:
        print("\n3. User chose 'Later' - update postponed")
        print("   💡 User will be reminded on next app launch")

if __name__ == "__main__":
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not running. Start it with: python app.py")
            exit(1)
    except:
        print("❌ Cannot connect to backend. Make sure it's running on port 8000")
        exit(1)
    
    demo_update_flow()
