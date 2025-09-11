# 🔄 Auto-Update System for Orca Engine

Event-driven auto-update system that checks for new releases and shows "Update Available" popups to users.

## 🎯 Features

### ✅ **Event-Driven Updates (Not Polling)**
- **GitHub Webhooks**: Instant notifications when new releases are published
- **Background Checker**: Hourly fallback checks (configurable)
- **Smart Caching**: Avoids redundant API calls

### ✅ **Cross-Platform Support**
- **Mac**: DMG downloads with automatic mounting/installation
- **Windows**: EXE installers with silent installation
- **Linux**: AppImage and .deb package support

### ✅ **User Experience**
- **Non-Intrusive**: Checks happen in background
- **Smart Notifications**: Only shows popup when update is actually available
- **Critical Updates**: Special handling for security/urgent updates

## 🔧 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub        │ -> │  Auto Update     │ -> │   Orca Engine   │
│   Webhook       │    │  Manager         │    │   Popup UI      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                       │
        v                        v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  New Release    │    │  Version Check   │    │  "Update        │
│  Published      │    │  + Download      │    │   Available"    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Setup

### 1. **Backend Configuration**

Add to `backend/.env`:
```bash
# Auto-update settings
ORCA_VERSION=0.01.5                    # Current app version
UPDATE_CHECK_INTERVAL=3600             # Check every hour (3600 seconds)
APPCAST_URL=https://simplifine-gamedev.github.io/orca-engine/appcast.xml
APPCAST_URL_WINDOWS=https://simplifine-gamedev.github.io/orca-engine/appcast-windows.xml
```

### 2. **GitHub Webhook** (Optional but Recommended)

Set up webhook in your GitHub repo:
- **URL**: `https://your-backend.com/update/webhook`
- **Events**: `Releases`
- **Content Type**: `application/json`

### 3. **Version File**

Create/update `version.py` in project root:
```python
version = "0.01.5"  # Current version
```

## 📡 API Endpoints

### **GET /update/check**
Check for available updates:
```bash
curl "http://localhost:8000/update/check?force=true&platform=mac"
```

Response:
```json
{
  "success": true,
  "update_available": true,
  "update_info": {
    "version": "0.01.6",
    "download_url": "https://github.com/.../Orca%20(Mac).dmg",
    "file_size_mb": 69.5,
    "release_notes": "Bug fixes and performance improvements",
    "is_critical": false
  },
  "current_version": "0.01.5"
}
```

### **GET /update/status**
Get update system status:
```bash
curl "http://localhost:8000/update/status"
```

### **POST /update/download**
Download an update:
```bash
curl -X POST "http://localhost:8000/update/download" \
  -H "Content-Type: application/json" \
  -d '{"version": "0.01.6"}'
```

### **POST /update/install**
Install downloaded update:
```bash
curl -X POST "http://localhost:8000/update/install" \
  -H "Content-Type: application/json" \
  -d '{"download_path": "/path/to/update.dmg", "restart_app": true}'
```

## 🤖 AI Tool Integration

The system includes a `check_for_app_updates` tool that the AI can call:

```json
{
  "name": "check_for_app_updates",
  "description": "Check if a newer version of Orca Engine is available and show update notification to user",
  "parameters": {
    "force_check": false,
    "show_notification": true
  }
}
```

### **AI Usage Examples:**

**User**: "Is there an update for Orca Engine?"
**AI**: Calls `check_for_app_updates` → Shows popup if update available

**User**: "Check for updates"
**AI**: Calls `check_for_app_updates` with `force_check: true`

## 🔔 Popup Configuration

When an update is found, the tool returns popup configuration:

```json
{
  "popup_config": {
    "title": "Update Available",
    "message": "Orca Engine v0.01.6 is now available.\n\nCurrent version: v0.01.5",
    "buttons": ["Install Now", "Later"],
    "default_button": 1,
    "icon": "info"
  }
}
```

### **Critical Updates:**
```json
{
  "popup_config": {
    "title": "Critical Update Available", 
    "message": "Security update v0.01.7 is now available...",
    "buttons": ["Install Now", "Later"],
    "default_button": 0,
    "icon": "warning"
  }
}
```

## 🎮 Frontend Integration

### **In Your Orca Engine App:**

1. **On Startup**: Call `/update/check` to see if updates are available
2. **Show Popup**: Use the `popup_config` to display native dialog
3. **Handle Choice**: 
   - **"Install Now"**: Call `/update/download` then `/update/install`
   - **"Later"**: Store preference, check again later

### **Example Integration (Pseudocode):**
```gdscript
# In your main scene _ready() function
func _ready():
    check_for_updates_on_startup()

func check_for_updates_on_startup():
    var http_request = HTTPRequest.new()
    add_child(http_request)
    http_request.request_completed.connect(_on_update_check_completed)
    http_request.request("http://localhost:8000/update/check")

func _on_update_check_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
    if response_code == 200:
        var json = JSON.new()
        var parse_result = json.parse(body.get_string_from_utf8())
        
        if parse_result == OK:
            var data = json.data
            if data.get("update_available", false):
                show_update_popup(data.get("update_info", {}))

func show_update_popup(update_info: Dictionary):
    var popup = AcceptDialog.new()
    popup.title = "Update Available"
    popup.dialog_text = "Orca Engine v%s is available.\n\nInstall now?" % update_info.get("version", "")
    
    # Add Install Now button
    popup.add_button("Install Now", false, "install")
    popup.add_button("Later", true, "later")
    
    popup.custom_action.connect(_on_update_choice)
    add_child(popup)
    popup.popup_centered()

func _on_update_choice(action: String):
    if action == "install":
        download_and_install_update()
    else:
        print("Update postponed")
```

## 🔧 Testing

### **Test the System:**
```bash
cd backend
python test_auto_update.py
```

### **Test with AI:**
Start backend and ask the AI:
- "Check for Orca Engine updates"
- "Is there a new version available?"
- "Update the app"

## 🎯 What Users See

### **Update Available Popup:**
```
┌─────────────────────────────────────┐
│           Update Available          │
├─────────────────────────────────────┤
│ Orca Engine v0.01.6 is now available │
│                                     │
│ Current version: v0.01.5            │
│ Download size: 69.5MB               │
│                                     │
│ What's new:                         │
│ • Fixed Mac signing issues          │
│ • Improved Windows performance      │
│ • New AI features                   │
│                                     │
│ [Install Now]  [Later]              │
└─────────────────────────────────────┘
```

### **Critical Update:**
```
┌─────────────────────────────────────┐
│       ⚠️ Critical Update Available   │
├─────────────────────────────────────┤
│ Security update v0.01.7 is now     │
│ available and recommended.          │
│                                     │
│ [Install Now]  [Later]              │
└─────────────────────────────────────┘
```

## 🔐 Security

### **Download Verification:**
- ✅ Downloads from official GitHub releases only
- ✅ File size verification 
- ✅ HTTPS-only downloads
- ✅ Platform-specific format validation

### **Installation Safety:**
- ✅ Mac: DMG signature verification via macOS
- ✅ Windows: Code-signed executables
- ✅ Linux: Package manager integration where possible

## 📊 Configuration

### **Environment Variables:**
```bash
ORCA_VERSION=0.01.5                    # Current version
UPDATE_CHECK_INTERVAL=3600             # Background check interval (seconds)
APPCAST_URL=https://...                # Mac appcast URL
APPCAST_URL_WINDOWS=https://...        # Windows appcast URL
```

### **Disable Updates:**
```bash
UPDATE_CHECK_INTERVAL=0                # Disable background checks
# Or don't include the auto_update_manager import
```

## 🎉 Result

Users get a seamless update experience:

1. **Automatic Detection**: Updates detected within minutes of release
2. **User Choice**: Non-intrusive popup with clear options
3. **One-Click Install**: Download and install with single click
4. **Platform Native**: Uses each platform's standard update mechanisms

**Your automated build system + auto-update = Users always have the latest Orca Engine!** 🚀
