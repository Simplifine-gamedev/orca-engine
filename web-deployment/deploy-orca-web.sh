#!/bin/bash

# Orca Engine Web Deployment Script
# Automated deployment to Vercel for editor.orcaengine.ai

echo "🚀 Orca Engine Web Deployment Script"
echo "======================================"
echo ""

# Check if we're in the right directory
if [[ ! -f "orca.web.editor.wasm32.wasm" ]]; then
    echo "❌ Error: Please run this script from the web-deployment directory"
    echo "   Expected files not found. Are you in the right directory?"
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Vercel CLI not found. Installing..."
    npm install -g vercel
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Vercel CLI"
        echo "   Please install manually: npm install -g vercel"
        exit 1
    fi
    echo "✅ Vercel CLI installed successfully"
fi

# Display current files
echo "📁 Current web files:"
ls -lh *.wasm *.js *.html *.png 2>/dev/null | while read line; do
    echo "   $line"
done
echo ""

# Check if user wants to continue
echo "🔍 Ready to deploy Orca Engine web editor"
echo "   Target: editor.orcaengine.ai"
echo "   Files: $(ls -1 orca.web.editor.wasm32.* | wc -l | tr -d ' ') web files found"
echo ""

read -p "Continue with deployment? [y/N]: " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 Deployment cancelled by user"
    exit 0
fi

# Deploy to Vercel
echo ""
echo "🚢 Deploying to Vercel..."
echo "⏳ This may take a few minutes due to large WASM file (87MB)..."
echo ""

vercel --prod

# Check deployment result
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Test the deployment at the URL shown above"
    echo "   2. Set up custom domain editor.orcaengine.ai in Vercel Dashboard"
    echo "   3. Configure DNS: CNAME editor.orcaengine.ai → cname.vercel-dns.com"
    echo ""
    echo "🔗 Useful links:"
    echo "   • Vercel Dashboard: https://vercel.com/dashboard"  
    echo "   • Project Settings: https://vercel.com/orca-team/godot"
    echo ""
else
    echo ""
    echo "❌ Deployment failed!"
    echo "   Check the error messages above for details"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   • Ensure you're logged into Vercel: vercel login"
    echo "   • Check network connectivity"
    echo "   • Try: vercel --debug --prod for more details"
    exit 1
fi

echo "✨ Orca Engine deployment complete!"



