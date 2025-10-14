#!/bin/bash
# Complete test of the Orca Engine update system

echo "🧪 COMPREHENSIVE UPDATE SYSTEM TEST"
echo "====================================="

# Function to check embedded version in binary
check_embedded_version() {
    local binary_path="$1"
    if [ -f "$binary_path" ]; then
        echo "🔍 Checking embedded version in: $binary_path"
        
        # Try multiple methods to extract version
        VERSION_FOUND=$(strings "$binary_path" | grep "0\.01\." | head -1)
        if [ -n "$VERSION_FOUND" ]; then
            echo "   ✅ Found embedded version: $VERSION_FOUND"
        else
            echo "   ❌ No embedded version found"
        fi
        
        # Check for Orca version constants
        ORCA_VERSION=$(strings "$binary_path" | grep "Orca Engine v" | head -1)
        if [ -n "$ORCA_VERSION" ]; then
            echo "   ✅ Orca version string: $ORCA_VERSION"
        fi
        
        return 0
    else
        echo "   ❌ Binary not found: $binary_path"
        return 1
    fi
}

echo ""
echo "📦 STEP 1: Testing Local Build Version Embedding"
echo "================================================="

# Test current build
LOCAL_BINARY="./bin/orca.macos.editor.dev.arm64"
if [ -f "$LOCAL_BINARY" ]; then
    check_embedded_version "$LOCAL_BINARY"
    
    echo ""
    echo "🚀 Testing version detection at runtime:"
    echo "   (Launch and check console for version messages)"
    echo "   Expected: 'Version from compiled binary: 0.01.test123'"
else
    echo "❌ Local binary not found. Build first with: ./build_with_version.sh"
fi

echo ""
echo "📥 STEP 2: Download and Test Production Binary"
echo "================================================="

# Get latest release info
LATEST_RELEASE_URL="https://api.github.com/repos/Simplifine-gamedev/orca-engine/releases/latest"
echo "🌐 Fetching latest release info..."

if command -v curl >/dev/null 2>&1; then
    RELEASE_INFO=$(curl -s "$LATEST_RELEASE_URL")
    LATEST_TAG=$(echo "$RELEASE_INFO" | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    MAC_DOWNLOAD_URL=$(echo "$RELEASE_INFO" | grep '"browser_download_url".*Mac.*dmg"' | head -1 | cut -d'"' -f4)
    
    echo "   Latest release: $LATEST_TAG"
    echo "   Mac download: $MAC_DOWNLOAD_URL"
    
    if [ -n "$MAC_DOWNLOAD_URL" ]; then
        DMG_FILE="Orca-Production-Test.dmg"
        echo ""
        echo "🔽 Downloading production binary for testing..."
        curl -L -o "$DMG_FILE" "$MAC_DOWNLOAD_URL"
        
        if [ -f "$DMG_FILE" ]; then
            echo "   ✅ Downloaded: $DMG_FILE"
            
            # Mount and check version
            echo ""
            echo "🔍 Checking version in production binary..."
            MOUNT_POINT="/tmp/orca_test_mount"
            mkdir -p "$MOUNT_POINT"
            
            if hdiutil attach "$DMG_FILE" -mountpoint "$MOUNT_POINT" -nobrowse; then
                # Find .app in mounted DMG
                APP_IN_DMG=$(find "$MOUNT_POINT" -name "*.app" | head -1)
                if [ -n "$APP_IN_DMG" ]; then
                    BINARY_IN_APP="$APP_IN_DMG/Contents/MacOS/Orca"
                    check_embedded_version "$BINARY_IN_APP"
                fi
                
                # Unmount
                hdiutil detach "$MOUNT_POINT"
                rm -rf "$MOUNT_POINT"
            fi
            
            # Clean up
            rm -f "$DMG_FILE"
        fi
    fi
else
    echo "   ❌ curl not available - skipping production download test"
fi

echo ""
echo "🔄 STEP 3: Update Flow Testing Instructions"
echo "=============================================="

echo "📋 MANUAL TEST PLAN:"
echo ""
echo "A. PREPARE VERSIONS:"
echo "   1. Build OLD version: ./build_with_version.sh '0.01.oldversion'"
echo "   2. Launch it - should show update notification"
echo "   3. Open your demo project: rocket-game-v-2"
echo ""
echo "B. TEST UPDATE PROCESS:"
echo "   4. Click 'Download & Install Update'"
echo "   5. Let it download and install"
echo "   6. Watch console logs during installation"
echo ""
echo "C. VERIFY RESULTS:"
echo "   7. New version should launch automatically"
echo "   8. Should open the SAME PROJECT (rocket-game-v-2)"
echo "   9. Check version: Should show embedded version from downloaded binary"
echo "   10. Should NOT show update notification anymore"
echo ""
echo "🔍 WHAT TO CHECK:"
echo "   ✅ Project state preserved (same scenes open)"
echo "   ✅ No update notification on new version launch"
echo "   ✅ Correct version numbers displayed"
echo "   ✅ Editor opens (not game mode)"

echo ""
echo "🐛 DEBUGGING:"
echo "   • Console logs will show version detection process"  
echo "   • Look for: 'Version from compiled binary: X.X.X'"
echo "   • Look for: 'Comparing versions - Current: X vs Remote: Y'"
echo "   • Installation logs will show mounting/copying process"

echo ""
echo "⚠️  KNOWN ISSUES TO VERIFY:"
echo "   • Windows: Should launch editor (not game) after update"
echo "   • Mac: Should preserve project state"  
echo "   • All platforms: No constant update notifications"

echo ""
echo "🎯 SUCCESS CRITERIA:"
echo "   1. Update notification shows ONLY when versions differ"
echo "   2. Installation preserves project state"
echo "   3. New version has correct embedded version"
echo "   4. No spam notifications after update"
