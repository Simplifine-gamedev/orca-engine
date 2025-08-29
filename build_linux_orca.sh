#!/bin/bash

echo "🔨 Building Orca Engine for Linux..."

# Build the Docker image with Orca compiler
docker build -f docker/Dockerfile.orca-builder -t orca-builder .

# Create a container and copy the binary out
echo "📦 Extracting Orca binary..."
docker create --name orca-extract orca-builder
docker cp orca-extract:/build/bin/. ./bin/
docker rm orca-extract

echo "✅ Orca Linux binary built successfully!"
ls -la bin/ | grep linux

echo "🚀 Now you can use the Linux Orca binary in the cloud deployment"

