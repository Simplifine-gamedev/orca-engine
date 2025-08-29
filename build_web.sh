#!/bin/bash

# Build script for Orca Engine web export
# Requires Emscripten to be installed and available in PATH

echo "Building Orca Engine for Web..."

# Check if emcc is available
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten (emcc) not found in PATH"
    echo "Please install Emscripten from: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Clean previous build
rm -rf bin/web_export
mkdir -p bin/web_export

# Build the web export (optimized for size, building editor for web)
echo "Running scons build..."
scons platform=web target=editor production=yes \
    optimize=size \
    wasm_simd=yes \
    javascript_eval=yes \
    dlink_enabled=false \
    -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Find and copy the generated files to the export directory
echo "Copying web export files..."

# Copy all web export files
if ls bin/godot.web.editor.wasm32* 1> /dev/null 2>&1; then
    cp bin/godot.web.editor.wasm32* bin/web_export/ 2>/dev/null || true
fi
if ls bin/godot.editor.wasm32* 1> /dev/null 2>&1; then
    cp bin/godot.editor.wasm32* bin/web_export/ 2>/dev/null || true
fi
if ls bin/*.wasm 1> /dev/null 2>&1; then
    cp bin/*.wasm bin/web_export/ 2>/dev/null || true
fi
if ls bin/*.js 1> /dev/null 2>&1; then
    cp bin/*.js bin/web_export/ 2>/dev/null || true
fi
if ls bin/*.pck 1> /dev/null 2>&1; then
    cp bin/*.pck bin/web_export/ 2>/dev/null || true
fi

# Copy HTML template
cp misc/dist/html/editor.html bin/web_export/index.html

# Replace Godot branding with Orca in the HTML
sed -i '' 's/Godot Engine/Orca Engine/g' bin/web_export/index.html
sed -i '' 's/Godot/Orca/g' bin/web_export/index.html
sed -i '' 's/godotengine\.org/orcaengine.ai/g' bin/web_export/index.html

# Copy favicon if exists
if [ -f "orcabranding/icon.png" ]; then
    cp orcabranding/icon.png bin/web_export/favicon.png
fi

echo "Web export complete! Files are in bin/web_export/"
echo "You can test locally by running: python3 platform/web/serve.py --root bin/web_export/"
