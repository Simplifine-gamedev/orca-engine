#!/bin/bash

# Rebuild Orca Engine for Web with proper branding
echo "=== Rebuilding Orca Engine Web Export ==="
echo ""

# Check if emcc is available
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten not found. Please ensure it's installed."
    exit 1
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf bin/web_export_new
mkdir -p bin/web_export_new

# Update version info to show Orca
echo "Setting Orca version info..."
export GODOT_VERSION_STATUS="Orca Engine"
export GODOT_VERSION_NAME="Orca 1.0"

# Build with custom branding
echo "Building Orca Engine for web (this will take 15-30 minutes)..."
scons platform=web target=editor production=yes \
    optimize=size \
    wasm_simd=yes \
    javascript_eval=yes \
    dlink_enabled=false \
    custom_name="orca" \
    -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Copy generated files
echo "Copying generated files..."
mkdir -p bin/web_export_new

# Find and copy all web files
for file in bin/*.web.editor.wasm32*; do
    if [ -f "$file" ]; then
        # Rename godot references to orca
        newname=$(basename "$file" | sed 's/godot/orca/g')
        cp "$file" "bin/web_export_new/$newname"
    fi
done

# Copy HTML template and customize
if [ -f "misc/dist/html/editor.html" ]; then
    cp misc/dist/html/editor.html bin/web_export_new/index.html
    
    # Replace all Godot references with Orca
    sed -i '' 's/Godot Engine/Orca Engine/g' bin/web_export_new/index.html
    sed -i '' 's/Godot/Orca/g' bin/web_export_new/index.html
    sed -i '' 's/godotengine\.org/orcaengine.ai/g' bin/web_export_new/index.html
    sed -i '' 's/___GODOT_VERSION___/Orca Engine v1.0/g' bin/web_export_new/index.html
    sed -i '' 's/___GODOT_THREADS_ENABLED___/true/g' bin/web_export_new/index.html
    
    # Update file references to use orca names
    sed -i '' 's/godot\.web\.editor/orca.web.editor/g' bin/web_export_new/index.html
fi

# Copy Orca branding assets
echo "Adding Orca branding..."
if [ -f "orcabranding/icon.png" ]; then
    cp orcabranding/icon.png bin/web_export_new/favicon.png
fi
if [ -f "orcabranding/orca_white_transparent.png" ]; then
    cp orcabranding/orca_white_transparent.png bin/web_export_new/logo.png
fi

# Update JavaScript references
for js in bin/web_export_new/*.js; do
    if [ -f "$js" ]; then
        sed -i '' 's/Godot Engine/Orca Engine/g' "$js" 2>/dev/null || true
        sed -i '' 's/godot/orca/g' "$js" 2>/dev/null || true
    fi
done

# Copy Vercel config
if [ -f "vercel.json" ]; then
    cp vercel.json bin/web_export_new/
fi

echo ""
echo "=== Rebuild Complete! ==="
echo "New build is in: bin/web_export_new/"
echo ""
echo "To deploy:"
echo "1. cd bin/web_export_new"
echo "2. vercel --prod"
echo ""
echo "Or replace the old build:"
echo "rm -rf bin/web_export && mv bin/web_export_new bin/web_export"
