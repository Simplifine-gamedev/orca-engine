#!/bin/bash

set -euo pipefail

# Parse command line arguments
PRODUCTION_BUILD=false
BUILD_TYPE="development"

while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            PRODUCTION_BUILD=true
            BUILD_TYPE="production"
            shift
            ;;
        --help)
            echo "Usage: $0 [--production]"
            echo "  --production    Build for production with optimizations and static linking"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🔨 Building Orca Engine for Linux ($BUILD_TYPE build)..."

# Choose the appropriate Dockerfile
if [ "$PRODUCTION_BUILD" = true ]; then
    DOCKERFILE="docker/Dockerfile.orca-builder-production"
    TAG="orca-builder-production"
    echo "📋 Using production build configuration with optimizations"
else
    DOCKERFILE="docker/Dockerfile.orca-builder"
    TAG="orca-builder"
    echo "📋 Using development build configuration"
fi

# Build the Docker image with Orca compiler
echo "🐳 Building Docker image..."
docker build -f "$DOCKERFILE" -t "$TAG" .

# Create a container and copy the binary out
echo "📦 Extracting Orca binary..."
docker create --name orca-extract "$TAG"
docker cp orca-extract:/build/bin/. ./bin/
docker rm orca-extract

echo "✅ Orca Linux binary built successfully!"
ls -la bin/ | grep -E "(linux|x86_64)" || ls -la bin/

if [ "$PRODUCTION_BUILD" = true ]; then
    echo "🚀 Production Linux build ready for distribution!"
    echo "📁 Binary location: bin/godot.linuxbsd.editor.x86_64"
    echo "🔍 Binary size: $(du -h bin/godot.linuxbsd.editor.x86_64 2>/dev/null || echo 'Binary not found')"
else
    echo "🚀 Development Linux build ready for cloud deployment!"
fi

