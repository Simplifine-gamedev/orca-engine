#!/bin/bash

# Windows Code Signing Script for Orca Engine
# This script signs the Windows production binary using SSL.com CodeSignTool

set -euo pipefail

BINARY_PATH="bin/orca-windows-production.exe"
SIGNED_DIR="bin/signed"
SIGNED_BINARY="$SIGNED_DIR/orca-windows-production.exe"

echo "🔐 Signing Windows binary for Orca Engine..."

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    echo "❌ Error: Binary not found at $BINARY_PATH"
    echo "Please build the Windows binary first"
    exit 1
fi

# Create signed directory if it doesn't exist
mkdir -p "$SIGNED_DIR"

# Copy binary to signed directory
cp "$BINARY_PATH" "$SIGNED_BINARY"

# Check if CodeSignTool exists
if [ ! -f "CodeSignTool.sh" ]; then
    echo "❌ Error: CodeSignTool.sh not found"
    echo "Please ensure CodeSignTool is properly set up"
    exit 1
fi

# Check if certificate exists
if [ ! -f "eSigner_CKA.zip" ]; then
    echo "❌ Error: Certificate file eSigner_CKA.zip not found"
    echo "Please ensure the certificate is in the project root"
    exit 1
fi

# Extract certificate if needed
if [ ! -d "eSigner_CKA" ]; then
    echo "📦 Extracting certificate..."
    unzip -q "eSigner_CKA.zip"
fi

echo "🔏 Signing binary with SSL.com CodeSignTool..."

# Sign the binary using CodeSignTool
# Note: You'll need to set these environment variables:
# - SSL_COM_USERNAME
# - SSL_COM_PASSWORD  
# - SSL_COM_TOTP_SECRET (for 2FA)

if [ -z "${SSL_COM_USERNAME:-}" ] || [ -z "${SSL_COM_PASSWORD:-}" ]; then
    echo "❌ Error: SSL.com credentials not set"
    echo "Please set SSL_COM_USERNAME and SSL_COM_PASSWORD environment variables"
    echo "You may also need SSL_COM_TOTP_SECRET for 2FA"
    exit 1
fi

# Run the signing command
./CodeSignTool.sh sign \
    -username="$SSL_COM_USERNAME" \
    -password="$SSL_COM_PASSWORD" \
    -credential_id="00987e95-0523-49f1-a7a9-8c71d6cec2d9" \
    -totp_secret="$SSL_COM_TOTP_SECRET" \
    -input_file_path="$SIGNED_BINARY" \
    -output_dir_path="$SIGNED_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Windows binary signed successfully!"
    echo "📦 Signed binary available at: $SIGNED_BINARY"
else
    echo "❌ Code signing failed"
    echo "📄 Check the logs above for details"
    exit 1
fi

echo "🔍 Verifying signed binary..."
ls -la "$SIGNED_BINARY"

echo "✅ Windows code signing complete!"
    -username="$SSL_COM_USERNAME" \
    -password="$SSL_COM_PASSWORD" \
    -credential_id="kaXTRACNijSWsFdRKg_KAfD3fqrBlzMbWs6TwWHwAn8" \
    -totp_secret="${SSL_COM_TOTP_SECRET:-}" \
    -input_file_path="$SIGNED_BINARY" \
    -output_dir_path="$SIGNED_DIR"

# Verify the signature
echo "🔍 Verifying signature..."
if command -v osslsigncode &> /dev/null; then
    osslsigncode verify "$SIGNED_BINARY" && echo "✅ Windows binary successfully signed!" || echo "❌ Signature verification failed"
else
    echo "⚠️  osslsigncode not available for verification, but signing command completed"
    echo "✅ Windows binary signing process completed!"
fi

echo ""
echo "📁 Signed binary location: $SIGNED_BINARY"
echo "🎯 This signed binary will be used by the Inno Setup installer"
