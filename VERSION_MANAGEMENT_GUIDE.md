# 📋 Orca Engine Version Management Guide

## Overview

Orca Engine now uses **Git-based Semantic Versioning** as the single source of truth for version numbers. This eliminates version mismatches and provides a consistent, automated versioning system across all platforms.

## 🎯 Key Changes Implemented

### 1. **Version Detection Hierarchy**
The system now uses the following priority order to determine the current version:

1. **Git Tags** (Primary) - Exact tag at HEAD (e.g., `v1.0.0`)
2. **Git Describe** (Development) - Tag + commits ahead (e.g., `1.0.0-dev.5+g1234567`)
3. **Environment Variable** - `ORCA_VERSION` for deployments
4. **version.py** (Fallback) - Last resort fallback

### 2. **Files Modified**

#### Backend
- `backend/auto_update_manager.py` - Enhanced version detection with git integration
- `version.py` - Updated with documentation about fallback nature

#### GitHub Workflows
- `.github/workflows/macos-clean.yml` - Semantic versioning for Mac builds
- `.github/workflows/windows-production.yml` - Semantic versioning for Windows builds
- `.github/workflows/linux-production.yml` - Semantic versioning for Linux builds
- `.github/workflows/create-release.yml` - New workflow for creating releases

#### Testing
- `backend/test_version_detection.py` - Test script to verify version detection

## 🚀 How to Use

### Creating a New Release

1. **Use the GitHub Actions Workflow**:
   - Go to Actions → "🚀 Create Release"
   - Select version bump type (major/minor/patch/prerelease)
   - The workflow will create and push the tag automatically

2. **Manual Tag Creation**:
   ```bash
   # Create a new version tag
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

### Version Formats

- **Release**: `1.0.0`
- **Beta**: `1.0.0-beta.1`
- **Alpha**: `1.0.0-alpha.1`
- **RC**: `1.0.0-rc.1`
- **Development**: `1.0.0-dev.5+g1234567`

### Testing Version Detection

Run the test script to verify version detection:
```bash
python3 backend/test_version_detection.py
```

## 📊 Version Display

The system automatically formats versions for user-friendly display:

- `1.0.0` → "1.0.0"
- `1.0.0-beta.1` → "1.0.0 Beta 1"
- `1.0.0-dev.5+g1234567` → "1.0.0 (Development Build)"
- `1.0.0-alpha.2` → "1.0.0 Alpha 2"
- `1.0.0-rc.1` → "1.0.0 RC 1"

## 🔄 Auto-Update Behavior

### Update Detection
The system properly compares semantic versions:
- `1.0.0` to `1.0.1` = Patch update available
- `1.0.0` to `1.1.0` = Minor update available
- `1.0.0` to `2.0.0` = Major update available
- `1.0.0-beta.1` to `1.0.0` = Stable release available

### Update Types
The system categorizes updates to provide better context:
- **Major**: Significant new features and breaking changes
- **Minor**: New features and enhancements
- **Patch**: Bug fixes and stability improvements
- **Prerelease**: Beta/Alpha/RC versions

## ⚠️ Important Notes

1. **Always use tags** for production releases
2. **Development builds** automatically get version numbers like `1.0.0-dev.N+hash`
3. **The version.py file** is now only a fallback - don't manually update it
4. **GitHub releases** will automatically trigger builds with the correct version

## 🔧 Troubleshooting

### Issue: Version shows as "1.0.0-unknown"
**Solution**: You're not in a git repository or git is not available. This is the ultimate fallback.

### Issue: Version shows old number after creating tag
**Solution**: Make sure you're on the tagged commit. Run `git describe --tags` to verify.

### Issue: Auto-update not detecting new versions
**Solution**: Ensure the new version tag is higher than the current version using semantic versioning rules.

## 📝 Migration from Old System

The old system used hardcoded versions like `0.01.{SHA}`. The new system:
1. Uses proper semantic versioning (MAJOR.MINOR.PATCH)
2. Derives version from git tags automatically
3. No manual version updates needed in code

## 🎉 Benefits

1. **No Version Conflicts**: Single source of truth (git tags)
2. **Automatic Versioning**: No manual updates needed
3. **Clear Update Path**: Users understand what type of update they're getting
4. **Development Builds**: Automatic versioning for non-release builds
5. **Industry Standard**: Follows semantic versioning best practices

---

*Version management system implemented on September 2024*
*Current version baseline: v1.0.0*
