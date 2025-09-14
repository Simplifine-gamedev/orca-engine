#!/bin/bash

set -euo pipefail

# Linux Binary Signing Script for Orca Engine
# This script signs Linux binaries using GPG and creates checksums

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_PATH=""
OUTPUT_DIR=""
SIGNING_KEY=""
PASSPHRASE=""

usage() {
    echo "Usage: $0 --binary <path> --output <dir> [--key <gpg-key-id>] [--passphrase <passphrase>]"
    echo ""
    echo "Options:"
    echo "  --binary      Path to the binary to sign"
    echo "  --output      Output directory for signed files and checksums"
    echo "  --key         GPG key ID for signing (optional, uses default key if not specified)"
    echo "  --passphrase  GPG key passphrase (optional, will prompt if not provided)"
    echo "  --help        Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  GPG_SIGNING_KEY       GPG key ID for signing"
    echo "  GPG_PASSPHRASE        GPG key passphrase"
    exit 0
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --binary)
            BINARY_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --key)
            SIGNING_KEY="$2"
            shift 2
            ;;
        --passphrase)
            PASSPHRASE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Check required parameters
if [[ -z "$BINARY_PATH" ]]; then
    error "Binary path is required (--binary)"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    error "Output directory is required (--output)"
fi

# Use environment variables if not provided via command line
if [[ -z "$SIGNING_KEY" && -n "${GPG_SIGNING_KEY:-}" ]]; then
    SIGNING_KEY="$GPG_SIGNING_KEY"
fi

if [[ -z "$PASSPHRASE" && -n "${GPG_PASSPHRASE:-}" ]]; then
    PASSPHRASE="$GPG_PASSPHRASE"
fi

# Validate inputs
if [[ ! -f "$BINARY_PATH" ]]; then
    error "Binary file not found: $BINARY_PATH"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get binary info
BINARY_NAME=$(basename "$BINARY_PATH")
BINARY_SIZE=$(stat -c%s "$BINARY_PATH")

log "Starting binary signing process for: $BINARY_NAME"
log "Binary size: $BINARY_SIZE bytes"

# Copy binary to output directory
cp "$BINARY_PATH" "$OUTPUT_DIR/"
SIGNED_BINARY="$OUTPUT_DIR/$BINARY_NAME"

# Generate checksums
log "Generating checksums..."

# SHA256
sha256sum "$SIGNED_BINARY" > "$OUTPUT_DIR/$BINARY_NAME.sha256"
SHA256_HASH=$(cut -d' ' -f1 "$OUTPUT_DIR/$BINARY_NAME.sha256")
log "SHA256: $SHA256_HASH"

# SHA512
sha512sum "$SIGNED_BINARY" > "$OUTPUT_DIR/$BINARY_NAME.sha512"
SHA512_HASH=$(cut -d' ' -f1 "$OUTPUT_DIR/$BINARY_NAME.sha512")
log "SHA512: $SHA512_HASH"

# MD5 (for compatibility)
md5sum "$SIGNED_BINARY" > "$OUTPUT_DIR/$BINARY_NAME.md5"
MD5_HASH=$(cut -d' ' -f1 "$OUTPUT_DIR/$BINARY_NAME.md5")
log "MD5: $MD5_HASH"

# Create combined checksum file
cat > "$OUTPUT_DIR/$BINARY_NAME.checksums" << EOF
# Checksums for $BINARY_NAME
# Generated on $(date -u '+%Y-%m-%d %H:%M:%S UTC')
# File size: $BINARY_SIZE bytes

# SHA256
$SHA256_HASH  $BINARY_NAME

# SHA512
$SHA512_HASH  $BINARY_NAME

# MD5
$MD5_HASH  $BINARY_NAME
EOF

log "Checksums generated and saved to $BINARY_NAME.checksums"

# GPG signing
if command -v gpg >/dev/null 2>&1; then
    log "GPG found, proceeding with signing..."
    
    # Set up GPG options
    GPG_OPTS=(--batch --yes --armor)
    
    if [[ -n "$SIGNING_KEY" ]]; then
        GPG_OPTS+=(--local-user "$SIGNING_KEY")
        log "Using GPG key: $SIGNING_KEY"
    else
        log "Using default GPG key"
    fi
    
    if [[ -n "$PASSPHRASE" ]]; then
        GPG_OPTS+=(--pinentry-mode loopback --passphrase "$PASSPHRASE")
    fi
    
    # Sign the binary
    log "Signing binary..."
    if gpg "${GPG_OPTS[@]}" --detach-sign --output "$OUTPUT_DIR/$BINARY_NAME.sig" "$SIGNED_BINARY"; then
        log "✅ Binary signed successfully: $BINARY_NAME.sig"
    else
        log "⚠️  Binary signing failed, but continuing..."
    fi
    
    # Sign the checksums
    log "Signing checksums..."
    if gpg "${GPG_OPTS[@]}" --clearsign --output "$OUTPUT_DIR/$BINARY_NAME.checksums.asc" "$OUTPUT_DIR/$BINARY_NAME.checksums"; then
        log "✅ Checksums signed successfully: $BINARY_NAME.checksums.asc"
    else
        log "⚠️  Checksum signing failed, but continuing..."
    fi
    
else
    log "⚠️  GPG not found, skipping digital signatures"
fi

# Create verification script
cat > "$OUTPUT_DIR/verify.sh" << 'EOF'
#!/bin/bash

set -euo pipefail

BINARY_NAME=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the binary file (assuming there's only one main binary)
for file in "$SCRIPT_DIR"/*; do
    if [[ -f "$file" && -x "$file" && ! "$file" =~ \.(sig|sha256|sha512|md5|checksums|asc)$ && "$(basename "$file")" != "verify.sh" ]]; then
        BINARY_NAME="$(basename "$file")"
        break
    fi
done

if [[ -z "$BINARY_NAME" ]]; then
    echo "❌ No binary found to verify"
    exit 1
fi

echo "🔍 Verifying $BINARY_NAME..."

# Verify checksums
if [[ -f "$SCRIPT_DIR/$BINARY_NAME.sha256" ]]; then
    echo "Checking SHA256..."
    if cd "$SCRIPT_DIR" && sha256sum -c "$BINARY_NAME.sha256"; then
        echo "✅ SHA256 checksum verified"
    else
        echo "❌ SHA256 checksum verification failed"
        exit 1
    fi
fi

if [[ -f "$SCRIPT_DIR/$BINARY_NAME.sha512" ]]; then
    echo "Checking SHA512..."
    if cd "$SCRIPT_DIR" && sha512sum -c "$BINARY_NAME.sha512"; then
        echo "✅ SHA512 checksum verified"
    else
        echo "❌ SHA512 checksum verification failed"
        exit 1
    fi
fi

# Verify GPG signatures if available
if command -v gpg >/dev/null 2>&1; then
    if [[ -f "$SCRIPT_DIR/$BINARY_NAME.sig" ]]; then
        echo "Checking GPG signature for binary..."
        if gpg --verify "$SCRIPT_DIR/$BINARY_NAME.sig" "$SCRIPT_DIR/$BINARY_NAME"; then
            echo "✅ Binary GPG signature verified"
        else
            echo "❌ Binary GPG signature verification failed"
            exit 1
        fi
    fi
    
    if [[ -f "$SCRIPT_DIR/$BINARY_NAME.checksums.asc" ]]; then
        echo "Checking GPG signature for checksums..."
        if gpg --verify "$SCRIPT_DIR/$BINARY_NAME.checksums.asc"; then
            echo "✅ Checksums GPG signature verified"
        else
            echo "❌ Checksums GPG signature verification failed"
            exit 1
        fi
    fi
else
    echo "⚠️  GPG not found, skipping signature verification"
fi

echo "🎉 All verifications passed!"
EOF

chmod +x "$OUTPUT_DIR/verify.sh"

# Create release info file
cat > "$OUTPUT_DIR/RELEASE_INFO.txt" << EOF
Orca Engine Linux Build - Release Information
============================================

Build Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Binary: $BINARY_NAME
Size: $BINARY_SIZE bytes

Checksums:
- SHA256: $SHA256_HASH
- SHA512: $SHA512_HASH
- MD5: $MD5_HASH

Files in this release:
- $BINARY_NAME                 - Main executable
- $BINARY_NAME.sha256          - SHA256 checksum
- $BINARY_NAME.sha512          - SHA512 checksum  
- $BINARY_NAME.md5             - MD5 checksum
- $BINARY_NAME.checksums       - Combined checksums file
- verify.sh                    - Verification script
- RELEASE_INFO.txt             - This file

Optional files (if GPG signing was available):
- $BINARY_NAME.sig             - GPG signature for binary
- $BINARY_NAME.checksums.asc   - GPG signed checksums

Verification:
1. Run ./verify.sh to verify all checksums and signatures
2. Or manually verify with: sha256sum -c $BINARY_NAME.sha256

For more information, visit: https://orcaengine.ai
EOF

log "✅ Binary signing and verification files created successfully!"
log "📁 Output directory: $OUTPUT_DIR"
log "📋 Files created:"
ls -la "$OUTPUT_DIR"

echo ""
echo "🎉 Binary signing completed!"
echo "📁 Signed files are in: $OUTPUT_DIR"
echo "🔍 To verify the binary, run: $OUTPUT_DIR/verify.sh"