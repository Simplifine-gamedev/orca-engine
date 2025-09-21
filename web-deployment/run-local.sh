#!/bin/bash

# Orca Engine Web Editor - Local Development Server
# Quick script to run the web editor locally

echo "🚀 Starting Orca Engine Web Editor (Local Development)"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    echo "💡 Please install Python 3 from: https://python.org"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "index.html" ]] || [[ ! -f "orca.web.editor.wasm32.wasm" ]]; then
    echo "❌ Error: Please run this script from the web-deployment directory"
    echo "💡 Expected files (index.html, orca.web.editor.wasm32.wasm) not found"
    exit 1
fi

# Start the local server
echo "🌐 Starting local development server..."
echo "📱 The editor will open at: http://localhost:8080"
echo ""
echo "💡 Tips:"
echo "   • Use Chrome/Edge for best performance"
echo "   • The first load may take time (87MB WebAssembly file)"
echo "   • Your projects are saved locally in the browser"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Run the Python server
python3 serve-local.py



