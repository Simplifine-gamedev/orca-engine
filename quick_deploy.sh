#!/bin/bash

# Quick deployment script for Orca Engine to Vercel
# This can be used while the build is running or after it completes

echo "=== Quick Deploy to Vercel ==="
echo ""

# Check build status
if [ -f "build_web.log" ]; then
    echo "Build status:"
    tail -5 build_web.log
    echo ""
fi

# Check if web export directory exists
if [ -d "bin/web_export" ] && [ -n "$(ls -A bin/web_export 2>/dev/null)" ]; then
    echo "Found web export files in bin/web_export/"
    ls -lh bin/web_export/ | head -10
    echo ""
    
    # Check if files are complete
    if [ -f "bin/web_export/index.html" ] && ls bin/web_export/*.wasm >/dev/null 2>&1; then
        WASM_SIZE=$(ls -lh bin/web_export/*.wasm | awk '{print $5}')
        echo "WASM file size: $WASM_SIZE"
        
        if [ "$WASM_SIZE" = "0B" ] || [ "$WASM_SIZE" = "0" ]; then
            echo "⚠️  Warning: WASM file is empty. Build may still be in progress."
            echo "Check build progress with: tail -f build_web.log"
            echo ""
            read -p "Do you want to wait for build to complete? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Waiting for build... (Press Ctrl+C to stop waiting)"
                while [ ! -s "bin/web_export/*.wasm" ]; do
                    sleep 10
                    echo -n "."
                done
                echo " Build complete!"
            fi
        fi
    fi
else
    echo "Web export not found. The build may still be running."
    echo "Check build progress with: tail -f build_web.log"
    exit 1
fi

# Deploy options
echo ""
echo "Deployment Options:"
echo "1. Deploy to new Vercel project (editor.orcaengine.ai)"
echo "2. Deploy to subdirectory of existing site (/editor)"
echo "3. Deploy to staging URL for testing"
echo ""
read -p "Choose option (1-3): " -n 1 -r DEPLOY_OPTION
echo ""

case $DEPLOY_OPTION in
    1)
        echo "Deploying to new project for editor.orcaengine.ai..."
        vercel --prod --name=orca-editor
        echo ""
        echo "After deployment:"
        echo "1. Go to Vercel dashboard"
        echo "2. Add custom domain: editor.orcaengine.ai"
        echo "3. Configure DNS CNAME record"
        ;;
    2)
        echo "For subdirectory deployment:"
        echo "1. Copy bin/web_export/* to your orca-website repo in /public/editor/"
        echo "2. Push to your existing orca-website repo"
        echo "3. Access at: https://orcaengine.ai/editor"
        ;;
    3)
        echo "Deploying to staging URL..."
        vercel --name=orca-editor-staging
        echo ""
        echo "Use the staging URL to test before going live"
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=== Deployment Instructions Complete ==="
