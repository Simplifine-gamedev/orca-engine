#!/bin/bash

# Deploy script for Orca Engine to Vercel
# This script builds the web export locally and deploys to Vercel

echo "=== Orca Engine Vercel Deployment Script ==="
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Error: Vercel CLI not found"
    echo "Installing Vercel CLI..."
    npm install -g vercel
    if [ $? -ne 0 ]; then
        echo "Failed to install Vercel CLI. Please install it manually:"
        echo "npm install -g vercel"
        exit 1
    fi
fi

# Check if emcc is available for building
if ! command -v emcc &> /dev/null; then
    echo "Warning: Emscripten (emcc) not found in PATH"
    echo "You need to build the web export first."
    echo "Install Emscripten from: https://emscripten.org/docs/getting_started/downloads.html"
    echo ""
    echo "Or if you've already built the web export, we can proceed with deployment."
    read -p "Do you want to continue with deployment? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    # Build the web export
    echo "Building Orca Engine for web..."
    ./build_web.sh
    if [ $? -ne 0 ]; then
        echo "Build failed! Aborting deployment."
        exit 1
    fi
fi

# Check if the web export exists
if [ ! -d "bin/web_export" ] || [ -z "$(ls -A bin/web_export)" ]; then
    echo "Error: Web export directory is empty or doesn't exist!"
    echo "Please run ./build_web.sh first"
    exit 1
fi

# Deploy to Vercel
echo ""
echo "Deploying to Vercel..."
echo "Note: You'll be prompted to log in if you haven't already"
echo ""

# Deploy with Vercel CLI
vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Deployment successful! ==="
    echo "Your Orca Engine is now live on Vercel!"
    echo ""
    echo "Important notes:"
    echo "1. The CORS headers are configured for web assembly support"
    echo "2. You can test locally with: python3 platform/web/serve.py --root bin/web_export/"
    echo "3. Check your Vercel dashboard for the deployment URL"
else
    echo ""
    echo "Deployment failed. Please check the error messages above."
    exit 1
fi
