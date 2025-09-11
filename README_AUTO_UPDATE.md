# Orca Engine Auto-Update System

This document describes the comprehensive auto-update system implemented for Orca Engine.

## 🚀 Overview

The auto-update system provides seamless, user-friendly updates for Orca Engine across all platforms (Mac, Windows, Linux). It consists of both backend and frontend components that work together to deliver a smooth update experience.

## 🏗️ Architecture

### Backend Components

1. **`backend/auto_update_manager.py`** - Core Python update manager
   - GitHub API integration for release detection
   - Cross-platform download and installation handling
   - Semantic version comparison
   - Caching and background checking

2. **`backend/app.py`** - API endpoints
   - `/api/update/check` - Check for available updates
   - `/api/update/status` - Get system status
   - `/api/update/download` - Download update files
   - `/api/update/install` - Install downloaded updates
   - `/api/update/webhook` - GitHub webhook integration
   - `/api/update/notes/<version>` - Get release notes

3. **AI Tool Integration** - `check_for_app_updates` function
   - Users can ask "Check for updates" in AI chat
   - Returns formatted update information

### Frontend Components

1. **`core/auto_update_manager.h/cpp`** - C++ singleton manager
   - Core update functionality exposed to GDScript
   - HTTP client for backend communication
   - Signal-based event system

2. **`scene/gui/update_notification_dialog.h/cpp`** - Update popup UI
   - Custom AcceptDialog with "Install Now", "Install Later", "Skip Version" buttons
   - Progress indicators for download/install
   - Release notes display with rich text formatting

3. **`scene/main/auto_update_service.h/cpp`** - Integration service
   - Automatic startup checking
   - Background periodic checks
   - User preference management
   - Dialog lifecycle management

## 🎯 User Experience Flow

1. **Startup Check**: Orca Engine automatically checks for updates on startup
2. **Update Available**: Non-intrusive popup appears with update details
3. **User Choice**:
   - **Install Now**: Downloads and installs immediately, then restarts
   - **Install Later**: Dismisses popup, will check again later
   - **Skip Version**: Won't show notifications for this specific version
4. **Background Checks**: Periodic checks every 24 hours (configurable)
5. **AI Integration**: Users can manually check via AI chat

## 🔧 Configuration

### Environment Variables (Backend)

```bash
# GitHub API (optional, increases rate limits)
GITHUB_TOKEN=your_github_token

# Webhook security (optional)
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Backend URL (auto-detected in most cases)
ORCA_BACKEND_URL=http://localhost:8080
```

### User Preferences

Stored in `user://auto_update_settings.cfg`:
- `auto_check_on_startup` - Enable/disable startup checks
- `background_checking_enabled` - Enable/disable periodic checks
- `check_interval_hours` - Hours between background checks
- `skipped_version` - Version user chose to skip

### Project Settings

Configure via GDScript:
```gdscript
# Get the service
var update_service = AutoUpdateService.get_singleton()

# Configure settings
update_service.set_auto_check_on_startup(true)
update_service.set_background_checking_enabled(true)
update_service.set_check_interval_hours(24)

# Manual operations
update_service.check_for_updates_now()
update_service.show_update_dialog_if_available()
```

## 🔌 API Reference

### AutoUpdateManager (Singleton)

```gdscript
# Check for updates
AutoUpdateManager.check_for_updates()

# Get current status
var status = AutoUpdateManager.get_status()
var info = AutoUpdateManager.get_update_info()

# Configuration
AutoUpdateManager.set_backend_url("http://localhost:8080")
AutoUpdateManager.set_auto_check_enabled(true)
```

### AutoUpdateService (Singleton)

```gdscript
# Manual operations
var service = AutoUpdateService.get_singleton()
service.check_for_updates_now()
service.show_update_dialog_if_available()

# Skip a version
service.skip_version("1.2.0")
service.reset_skipped_versions()

# Status
var available = service.is_update_available()
var info = service.get_last_update_info()
```

### UpdateNotificationDialog

```gdscript
# Create and show dialog manually
var dialog = UpdateNotificationDialog.new()
dialog.set_update_info({
    "current_version": "1.0.0",
    "latest_version": "1.1.0",
    "release_notes": "Bug fixes and improvements",
    "download_url": "https://github.com/user/repo/releases/download/v1.1.0/orca.dmg"
})
get_tree().current_scene.add_child(dialog)
dialog.popup_centered()

# Connect to signals
dialog.connect("install_now_requested", _on_install_now)
dialog.connect("install_later_requested", _on_install_later)
dialog.connect("skip_version_requested", _on_skip_version)
```

## 🔄 Signals

### AutoUpdateManager
- `update_available(version: String, notes: String)`
- `update_downloaded(file_path: String)`
- `update_error(error_message: String)`
- `update_progress(progress: float)`

### AutoUpdateService
- `update_check_completed(update_info: Dictionary)`
- `update_notification_shown(version: String)`
- `update_install_started()`
- `update_install_completed()`

### UpdateNotificationDialog
- `install_now_requested()`
- `install_later_requested()`
- `skip_version_requested()`
- `update_action_selected(action: int)`

## 🧪 Testing

### Demo Script

Use `auto_update_demo.gd` to test the system:

```gdscript
# Add to your scene and run
extends Node

func _ready():
    # Demonstrates the auto-update system
    # Check console for output
```

### Manual Testing

```gdscript
# In the console or a script
var service = AutoUpdateService.get_singleton()

# Test update check
service.check_for_updates_now()

# Test dialog display
service.show_update_dialog_if_available()

# Test backend connection
var manager = AutoUpdateManager
manager.check_for_updates()
```

## 📦 Platform Support

### macOS (.dmg)
- Automatic DMG mounting and installation
- Code signing verification
- Seamless app replacement

### Windows (.exe/.msi)
- Silent installer execution
- UAC handling for system-wide installs
- Registry integration

### Linux (.AppImage/.deb/.rpm)
- AppImage replacement
- Package manager integration for .deb/.rpm
- Desktop file updates

## 🔐 Security

- **HTTPS Only**: All downloads use secure connections
- **Signature Verification**: Package integrity checking
- **Webhook Security**: GitHub webhook signature validation
- **Rate Limiting**: Prevents excessive API calls
- **User Consent**: No automatic installations without user approval

## 🚨 Error Handling

The system gracefully handles:
- Network connectivity issues
- GitHub API rate limits
- Invalid or corrupted downloads
- Installation failures
- Permission errors
- Platform incompatibility

## 🎛️ Customization

### Custom Update Sources

Replace GitHub with your own update server by modifying:
1. Backend API endpoints in `backend/auto_update_manager.py`
2. Frontend backend URL in AutoUpdateManager

### Custom UI

Extend `UpdateNotificationDialog` to customize:
- Visual appearance
- Button layout
- Progress indicators
- Release notes formatting

### Integration with CI/CD

The system works with automated build pipelines:
1. Build creates platform-specific packages
2. CI uploads to GitHub releases
3. Webhook notifies update system
4. Users get automatic notifications

## 📋 TODO / Future Enhancements

- [ ] Delta updates for smaller downloads
- [ ] Rollback functionality
- [ ] Update scheduling (install at specific time)
- [ ] Bandwidth throttling
- [ ] Update channels (stable, beta, nightly)
- [ ] Automatic restart after installation
- [ ] Update statistics and analytics

## 🐛 Troubleshooting

### Common Issues

1. **No update notifications**
   - Check backend URL configuration
   - Verify network connectivity
   - Check GitHub API rate limits

2. **Download failures**
   - Verify HTTPS connectivity
   - Check available disk space
   - Ensure write permissions in temp directory

3. **Installation failures**
   - Check file permissions
   - Verify platform compatibility
   - Review system requirements

### Debug Information

Enable debug logging:
```gdscript
# Get detailed status
var manager = AutoUpdateManager
print(manager.get_status())
print(manager.get_system_info())

var service = AutoUpdateService.get_singleton()
print(service.get_last_update_info())
```

## 📞 Support

For issues or questions about the auto-update system:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Test with the demo script
4. Check network connectivity and GitHub API status