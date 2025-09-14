#!/bin/bash

set -euo pipefail

# Linux Packaging Script for Orca Engine
# Creates AppImage, DEB, and TAR.GZ packages for distribution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_PATH=""
OUTPUT_DIR="packages"
VERSION=""
CREATE_APPIMAGE=false
CREATE_DEB=false
CREATE_TAR=true
VERBOSE=false

usage() {
    echo "Usage: $0 --binary <path> [options]"
    echo ""
    echo "Required:"
    echo "  --binary      Path to the Orca Engine binary"
    echo ""
    echo "Options:"
    echo "  --output      Output directory for packages (default: packages)"
    echo "  --version     Version string (default: auto-detect from git)"
    echo "  --appimage    Create AppImage package"
    echo "  --deb         Create DEB package"
    echo "  --tar         Create TAR.GZ package (default: true)"
    echo "  --all         Create all package types"
    echo "  --verbose     Verbose output"
    echo "  --help        Show this help message"
    exit 0
}

log() {
    if [[ "$VERBOSE" == true ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    fi
}

info() {
    echo "📦 $1"
}

error() {
    echo "❌ ERROR: $1" >&2
    exit 1
}

success() {
    echo "✅ $1"
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
        --version)
            VERSION="$2"
            shift 2
            ;;
        --appimage)
            CREATE_APPIMAGE=true
            shift
            ;;
        --deb)
            CREATE_DEB=true
            shift
            ;;
        --tar)
            CREATE_TAR=true
            shift
            ;;
        --all)
            CREATE_APPIMAGE=true
            CREATE_DEB=true
            CREATE_TAR=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate required parameters
if [[ -z "$BINARY_PATH" ]]; then
    error "Binary path is required (--binary)"
fi

if [[ ! -f "$BINARY_PATH" ]]; then
    error "Binary file not found: $BINARY_PATH"
fi

# Auto-detect version if not provided
if [[ -z "$VERSION" ]]; then
    if git describe --tags --always 2>/dev/null; then
        VERSION=$(git describe --tags --always 2>/dev/null)
    else
        VERSION="1.0.0-$(date +%Y%m%d)"
    fi
fi

info "Packaging Orca Engine v$VERSION"
info "Binary: $BINARY_PATH"
info "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get binary info
BINARY_NAME=$(basename "$BINARY_PATH")
BINARY_SIZE=$(stat -c%s "$BINARY_PATH")

log "Binary name: $BINARY_NAME"
log "Binary size: $BINARY_SIZE bytes"

# Create TAR.GZ package
if [[ "$CREATE_TAR" == true ]]; then
    info "Creating TAR.GZ package..."
    
    TAR_DIR="$OUTPUT_DIR/orca-engine-$VERSION"
    mkdir -p "$TAR_DIR"
    
    # Copy binary
    cp "$BINARY_PATH" "$TAR_DIR/orca-engine"
    chmod +x "$TAR_DIR/orca-engine"
    
    # Create launcher script
    cat > "$TAR_DIR/orca-engine.sh" << 'EOF'
#!/bin/bash
# Orca Engine Launcher Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/orca-engine"

if [ ! -f "$BINARY" ]; then
    echo "Error: Orca Engine binary not found at $BINARY"
    exit 1
fi

# Set library path for potential dependencies
export LD_LIBRARY_PATH="$SCRIPT_DIR:$LD_LIBRARY_PATH"

# Launch Orca Engine with all arguments
exec "$BINARY" "$@"
EOF
    chmod +x "$TAR_DIR/orca-engine.sh"
    
    # Add documentation
    cat > "$TAR_DIR/README.txt" << EOF
Orca Engine - Linux Distribution
===============================

Version: $VERSION
Package Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

Files:
- orca-engine: Main executable
- orca-engine.sh: Launcher script (recommended)
- README.txt: This file
- LICENSE: Software license (if available)

Installation:
1. Extract this archive to your desired location
2. Make sure the binary is executable: chmod +x orca-engine
3. Run with: ./orca-engine or ./orca-engine.sh

System Requirements:
- Linux x86_64
- OpenGL 3.3 compatible graphics card
- Audio system (ALSA/PulseAudio)
- Minimum 2GB RAM
- 500MB free disk space

For support and documentation, visit: https://orcaengine.ai

EOF
    
    # Copy license files if they exist
    for license_file in LICENSE LICENSE.txt COPYRIGHT.txt AUTHORS.md CONTRIBUTORS.md; do
        if [[ -f "$SCRIPT_DIR/../$license_file" ]]; then
            cp "$SCRIPT_DIR/../$license_file" "$TAR_DIR/"
        fi
    done
    
    # Create the archive
    cd "$OUTPUT_DIR"
    tar -czf "orca-engine-$VERSION-linux-x86_64.tar.gz" "orca-engine-$VERSION/"
    
    success "TAR.GZ package created: orca-engine-$VERSION-linux-x86_64.tar.gz"
    log "Package size: $(du -h orca-engine-$VERSION-linux-x86_64.tar.gz | cut -f1)"
fi

# Create AppImage package
if [[ "$CREATE_APPIMAGE" == true ]]; then
    info "Creating AppImage package..."
    
    # Check for required tools
    if ! command -v wget >/dev/null 2>&1; then
        error "wget is required for AppImage creation"
    fi
    
    APPIMAGE_DIR="$OUTPUT_DIR/OrcaEngine.AppDir"
    mkdir -p "$APPIMAGE_DIR/usr/bin"
    mkdir -p "$APPIMAGE_DIR/usr/share/applications"
    mkdir -p "$APPIMAGE_DIR/usr/share/icons/hicolor/256x256/apps"
    
    # Copy binary
    cp "$BINARY_PATH" "$APPIMAGE_DIR/usr/bin/orca-engine"
    chmod +x "$APPIMAGE_DIR/usr/bin/orca-engine"
    
    # Create desktop file
    cat > "$APPIMAGE_DIR/usr/share/applications/orca-engine.desktop" << EOF
[Desktop Entry]
Name=Orca Engine
Comment=Game Development Engine
Exec=orca-engine
Icon=orca-engine
Type=Application
Categories=Development;IDE;
Terminal=false
StartupNotify=true
Version=1.0
EOF
    
    # Create or copy icon
    ICON_CREATED=false
    for icon_file in "$SCRIPT_DIR/../icon.png" "$SCRIPT_DIR/../orcabranding/icon.png"; do
        if [[ -f "$icon_file" ]]; then
            cp "$icon_file" "$APPIMAGE_DIR/usr/share/icons/hicolor/256x256/apps/orca-engine.png"
            ICON_CREATED=true
            break
        fi
    done
    
    if [[ "$ICON_CREATED" == false ]]; then
        # Create a simple placeholder icon using ImageMagick if available
        if command -v convert >/dev/null 2>&1; then
            convert -size 256x256 xc:blue -fill white -gravity center -pointsize 48 \
                -annotate +0+0 "ORCA" "$APPIMAGE_DIR/usr/share/icons/hicolor/256x256/apps/orca-engine.png"
            ICON_CREATED=true
        fi
    fi
    
    # Copy icon to AppDir root
    if [[ "$ICON_CREATED" == true ]]; then
        cp "$APPIMAGE_DIR/usr/share/icons/hicolor/256x256/apps/orca-engine.png" "$APPIMAGE_DIR/"
    fi
    
    # Copy desktop file to AppDir root
    cp "$APPIMAGE_DIR/usr/share/applications/orca-engine.desktop" "$APPIMAGE_DIR/"
    
    # Create AppRun script
    cat > "$APPIMAGE_DIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/orca-engine" "$@"
EOF
    chmod +x "$APPIMAGE_DIR/AppRun"
    
    # Download AppImageTool if not present
    APPIMAGETOOL="$OUTPUT_DIR/appimagetool"
    if [[ ! -f "$APPIMAGETOOL" ]]; then
        log "Downloading AppImageTool..."
        wget -O "$APPIMAGETOOL" "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
        chmod +x "$APPIMAGETOOL"
    fi
    
    # Create AppImage
    cd "$OUTPUT_DIR"
    if "$APPIMAGETOOL" "OrcaEngine.AppDir" "OrcaEngine-$VERSION-x86_64.AppImage"; then
        success "AppImage package created: OrcaEngine-$VERSION-x86_64.AppImage"
        log "AppImage size: $(du -h OrcaEngine-$VERSION-x86_64.AppImage | cut -f1)"
    else
        error "AppImage creation failed"
    fi
fi

# Create DEB package
if [[ "$CREATE_DEB" == true ]]; then
    info "Creating DEB package..."
    
    DEB_DIR="$OUTPUT_DIR/orca-engine-deb"
    mkdir -p "$DEB_DIR/DEBIAN"
    mkdir -p "$DEB_DIR/usr/bin"
    mkdir -p "$DEB_DIR/usr/share/applications"
    mkdir -p "$DEB_DIR/usr/share/icons/hicolor/256x256/apps"
    mkdir -p "$DEB_DIR/usr/share/doc/orca-engine"
    
    # Copy binary
    cp "$BINARY_PATH" "$DEB_DIR/usr/bin/orca-engine"
    chmod +x "$DEB_DIR/usr/bin/orca-engine"
    
    # Create desktop file
    cat > "$DEB_DIR/usr/share/applications/orca-engine.desktop" << EOF
[Desktop Entry]
Name=Orca Engine
Comment=Game Development Engine
Exec=orca-engine
Icon=orca-engine
Type=Application
Categories=Development;IDE;
Terminal=false
StartupNotify=true
Version=1.0
EOF
    
    # Copy icon if available
    for icon_file in "$SCRIPT_DIR/../icon.png" "$SCRIPT_DIR/../orcabranding/icon.png"; do
        if [[ -f "$icon_file" ]]; then
            cp "$icon_file" "$DEB_DIR/usr/share/icons/hicolor/256x256/apps/orca-engine.png"
            break
        fi
    done
    
    # Create control file
    INSTALLED_SIZE=$(du -s "$DEB_DIR" | cut -f1)
    cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: orca-engine
Version: $VERSION
Section: development
Priority: optional
Architecture: amd64
Depends: libc6, libstdc++6, libgl1-mesa-glx, libasound2
Maintainer: Orca Engine Team <support@orcaengine.ai>
Installed-Size: $INSTALLED_SIZE
Description: Orca Game Development Engine
 Orca Engine is a powerful game development engine that provides
 a comprehensive set of tools for creating 2D and 3D games.
 .
 This package contains the Orca Engine editor and runtime.
Homepage: https://orcaengine.ai
EOF
    
    # Create postinst script
    cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

# Update desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q
fi

# Update icon cache
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

exit 0
EOF
    chmod +x "$DEB_DIR/DEBIAN/postinst"
    
    # Create postrm script
    cat > "$DEB_DIR/DEBIAN/postrm" << 'EOF'
#!/bin/bash
set -e

if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database -q
    fi
    
    # Update icon cache
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
    fi
fi

exit 0
EOF
    chmod +x "$DEB_DIR/DEBIAN/postrm"
    
    # Add documentation
    cat > "$DEB_DIR/usr/share/doc/orca-engine/README" << EOF
Orca Engine - Game Development Engine

Version: $VERSION
Package Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

Orca Engine is a powerful game development engine that provides
a comprehensive set of tools for creating 2D and 3D games.

For more information, visit: https://orcaengine.ai
EOF
    
    # Copy license files
    for license_file in LICENSE LICENSE.txt COPYRIGHT.txt; do
        if [[ -f "$SCRIPT_DIR/../$license_file" ]]; then
            cp "$SCRIPT_DIR/../$license_file" "$DEB_DIR/usr/share/doc/orca-engine/"
        fi
    done
    
    # Create copyright file
    cat > "$DEB_DIR/usr/share/doc/orca-engine/copyright" << EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: orca-engine
Source: https://orcaengine.ai

Files: *
Copyright: $(date +%Y) Orca Engine Team
License: MIT
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 .
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 .
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
EOF
    
    # Set proper permissions
    find "$DEB_DIR" -type f -exec chmod 644 {} \;
    find "$DEB_DIR" -type d -exec chmod 755 {} \;
    chmod +x "$DEB_DIR/usr/bin/orca-engine"
    chmod +x "$DEB_DIR/DEBIAN/postinst"
    chmod +x "$DEB_DIR/DEBIAN/postrm"
    
    # Build DEB package
    cd "$OUTPUT_DIR"
    if dpkg-deb --build orca-engine-deb "orca-engine_$VERSION-1_amd64.deb"; then
        success "DEB package created: orca-engine_$VERSION-1_amd64.deb"
        log "DEB size: $(du -h orca-engine_$VERSION-1_amd64.deb | cut -f1)"
    else
        error "DEB package creation failed"
    fi
fi

info "Packaging completed!"
info "Output directory: $OUTPUT_DIR"
echo ""
echo "📋 Created packages:"
ls -la "$OUTPUT_DIR"/*.{tar.gz,AppImage,deb} 2>/dev/null || echo "No packages found"