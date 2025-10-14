# 🚀 Orca Engine Production Update System

## Overview

This is the **automated update system for distributed Orca Engine binaries**. It works for your users who download releases from GitHub - completely independent of the backend.

## 🎯 How It Works

### **For Users (Production)**
1. User downloads Orca Engine v0.01.abc123 from GitHub
2. Version `0.01.abc123` is **baked into the binary** during build
3. Periodically, Orca checks GitHub API for latest release
4. If GitHub has v0.01.xyz789 (different), shows update notification
5. User clicks "Install" → Downloads and installs new version
6. Next launch: New binary has v0.01.xyz789 baked in → No notification

### **Version Embedding (Build Time)**
- **SCons** runs `core_builders.orca_version_builder()` before compilation
- Generates `core/orca_version.gen.cpp` with `ORCA_VERSION_STRING`
- This gets compiled into the Godot binary
- No runtime git dependency needed!

### **Update Check (Runtime)**
- Frontend reads `ORCA_VERSION_STRING` from compiled binary
- Checks GitHub API: `https://api.github.com/repos/Simplifine-gamedev/orca-engine/releases/latest`
- Compares: If `tag_name` != `ORCA_VERSION_STRING` → Show notification

## 🏗️ GitHub Actions Integration

### **Add to your build workflows:**

```yaml
# .github/workflows/macos-build.yml (and windows, linux)
name: Build Orca Engine

on:
  push:
    tags:
      - 'v*'  # Triggers on version tags like v0.01.abc123

jobs:
  build:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set Orca Version from Tag
        run: |
          # Extract version from git tag
          VERSION=$(git describe --exact-match --tags HEAD 2>/dev/null || echo "0.01.$(git rev-parse --short=8 HEAD)")
          echo "ORCA_VERSION=$VERSION" >> $GITHUB_ENV
          echo "Building Orca Engine version: $VERSION"
      
      - name: Build Orca Engine
        run: |
          scons platform=macos target=editor -j8
          # The orca_version_builder runs automatically during SCons
          # It embeds $ORCA_VERSION into the binary
      
      - name: Create Release Assets
        run: |
          # Package your .app, .dmg, etc.
          # Upload to GitHub release
```

## 📝 Version Format

**GitHub Release Tags:**
- Format: `v0.01.{SHORT_SHA}` (e.g., `v0.01.9b45879b`)
- The `v` prefix is stripped for comparison
- Uses 8-character git SHA for uniqueness

**Development Builds:**
- Auto-generated from current commit
- Format: `0.01.fc3a12ee` (no `v` prefix)

## 🧪 Testing Locally

### **1. Build with embedded version:**
```bash
# Set the version you want to test
export ORCA_VERSION="0.01.testbuild"

# Build Godot - version gets baked in
scons platform=macos target=editor

# Check embedded version
./bin/godot.macos.editor.arm64 --version
```

### **2. Test update notification:**
```bash
# Build with old version
export ORCA_VERSION="0.01.oldversion"
scons platform=macos target=editor

# Launch - should show update to 0.01.9b45879b (current GitHub release)
./bin/godot.macos.editor.arm64
```

### **3. Test "no update" scenario:**
```bash
# Build with current GitHub release version
export ORCA_VERSION="0.01.9b45879b"  
scons platform=macos target=editor

# Launch - should NOT show update notification
./bin/godot.macos.editor.arm64
```

## 🔧 How Version is Embedded

### **Build Process:**
1. SCons runs → Calls `core_builders.orca_version_builder()`
2. Reads `ORCA_VERSION` env var or detects from git
3. Generates `core/orca_version.gen.cpp`:
   ```cpp
   const char *const ORCA_VERSION_STRING = "0.01.9b45879b";
   ```
4. Compiles into the Godot binary
5. Runtime code reads `ORCA_VERSION_STRING` constant

### **Version Detection Priority:**
1. **ORCA_VERSION** env var (build time)
2. **Git tag** at HEAD (e.g., v0.01.abc123)
3. **Git SHA** (development builds)
4. **Fallback**: "1.0.0-dev"

## ✅ Fixes Applied

### **Problem 1: Constant Update Notifications**
- **Before**: Used git SHA at runtime → Always different from GitHub tag
- **After**: Baked GitHub tag into binary → Matches on comparison

### **Problem 2: Nonsense Version Numbers**
- **Before**: Read from Engine::get_version_info() → Returned internal messages
- **After**: Read from `ORCA_VERSION_STRING` → Correct version

### **Problem 3: Windows Launches Game Instead of Editor**
- **Before**: Only `--editor` flag
- **After**: Multiple flags: `--editor --no-window --path <project>`

## 🎬 Production Deployment Checklist

- [ ] Add version embedding to GitHub Actions workflows
- [ ] Tag releases with `v0.01.{SHA}` format
- [ ] Build with `ORCA_VERSION` or `GITHUB_REF` set
- [ ] Distribute binaries with embedded version
- [ ] Users get automatic update notifications when new releases available
- [ ] No backend/git dependency for users

## 🔍 Debugging

**Check what version is compiled into binary:**
```bash
# On Mac/Linux:
strings ./bin/godot.macos.editor.arm64 | grep "0\.01\."

# Should show: 0.01.abc123 (the embedded version)
```

**Check update system logs:**
- Look for: `UpdateNotificationPopup: ✅ Version from compiled binary:`
- Compare with: `UpdateNotificationPopup: Comparing versions`
- Should be: `✅ Versions match - no update notification`

## 🎯 Result

Your users will experience:
- ✅ Clean version display (actual GitHub release versions)
- ✅ Updates ONLY when you push a new release to GitHub
- ✅ No spam notifications
- ✅ Works offline (version is baked in)
- ✅ No backend dependency for update checks
- ✅ Windows launches editor correctly after update

