#!/bin/bash

# Mac Notarization Fix Script for Orca Engine
# This script re-submits the Mac app for notarization and staples the ticket

set -euo pipefail

APP_PATH="bin/Orca.app"
ZIP_PATH="bin/Orca.zip"

echo "🍎 Fixing Mac app notarization for Orca Engine..."

# Check if app exists
if [ ! -d "$APP_PATH" ]; then
    echo "❌ Error: App not found at $APP_PATH"
    echo "Please ensure the Mac app is built"
    exit 1
fi

# Check if app is signed
echo "🔍 Checking code signature..."
codesign -dv --verbose=4 "$APP_PATH" 2>&1 || {
    echo "❌ Error: App is not properly signed"
    exit 1
}

echo "✅ App is properly signed"

# Create notarization zip
echo "📦 Creating notarization zip..."
rm -f "$ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"

echo "🔐 Submitting for notarization..."

# Check which credentials are available
if [ -n "${AC_API_KEY_ID:-}" ] && [ -n "${AC_API_ISSUER_ID:-}" ] && [ -n "${AC_API_PRIVATE_KEY_P8:-}" ]; then
    echo "Using App Store Connect API key for notarization..."
    
    # Write the private key to a temporary file
    echo "$AC_API_PRIVATE_KEY_P8" > /tmp/AuthKey.p8
    
    # Submit for notarization using API key
    xcrun notarytool submit "$ZIP_PATH" \
        --key "/tmp/AuthKey.p8" \
        --key-id "$AC_API_KEY_ID" \
        --issuer "$AC_API_ISSUER_ID" \
        --wait \
        --timeout 30m
    
    # Clean up temp file
    rm -f /tmp/AuthKey.p8
    
elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ] && [ -n "${TEAM_ID:-}" ]; then
    echo "Using Apple ID for notarization..."
    
    # Submit for notarization using Apple ID
    xcrun notarytool submit "$ZIP_PATH" \
        --apple-id "$APPLE_ID" \
        --password "$APPLE_APP_PASSWORD" \
        --team-id "$TEAM_ID" \
        --wait \
        --timeout 30m
        
else
    echo "❌ Error: Missing notarization credentials"
    echo ""
    echo "Please set one of the following credential sets:"
    echo ""
    echo "Option 1 - App Store Connect API Key:"
    echo "  - AC_API_KEY_ID"
    echo "  - AC_API_ISSUER_ID" 
    echo "  - AC_API_PRIVATE_KEY_P8"
    echo ""
    echo "Option 2 - Apple ID:"
    echo "  - APPLE_ID"
    echo "  - APPLE_APP_PASSWORD (app-specific password)"
    echo "  - TEAM_ID"
    echo ""
    exit 1
fi

# Check if notarization was successful
if [ $? -eq 0 ]; then
    echo "✅ Notarization successful!"
    
    # Staple the ticket
    echo "📎 Stapling notarization ticket..."
    xcrun stapler staple "$APP_PATH"
    
    # Verify the stapling
    echo "🔍 Verifying stapled ticket..."
    xcrun stapler validate "$APP_PATH"
    
    # Test Gatekeeper
    echo "🛡️ Testing Gatekeeper validation..."
    spctl -a -vvv "$APP_PATH" 2>&1 && echo "✅ Gatekeeper validation passed!" || echo "❌ Gatekeeper validation failed"
    
    echo ""
    echo "🎉 Mac app notarization process completed!"
    echo "📁 Notarized app location: $APP_PATH"
    
else
    echo "❌ Notarization failed"
    echo "Please check the error messages above"
    exit 1
fi

# Clean up
rm -f "$ZIP_PATH"
