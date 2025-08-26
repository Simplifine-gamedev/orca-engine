# Orca Engine - Web Deployment Guide

This guide explains how to deploy the Orca Engine (Godot rebrand) to Vercel as a web application.

## Prerequisites

1. **Emscripten** - Required for building the web export
   - Install from: https://emscripten.org/docs/getting_started/downloads.html
   - Verify installation: `emcc --version`

2. **Node.js & npm** - Required for Vercel CLI
   - Install from: https://nodejs.org/

3. **Vercel Account** - Sign up at https://vercel.com

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Build Web Export
```bash
./build_web.sh
# or
npm run build
```

### 3. Test Locally
```bash
python3 platform/web/serve.py --root bin/web_export/
# or
npm run serve
```
Then open http://127.0.0.1:8000 in your browser.

### 4. Deploy to Vercel
```bash
./deploy_to_vercel.sh
# or
npm run deploy
```

## Manual Deployment

If you prefer to deploy manually:

1. Build the web export:
   ```bash
   scons platform=web target=template_release production=yes
   ```

2. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

3. Deploy:
   ```bash
   vercel --prod
   ```

## Configuration

The deployment is configured via `vercel.json` which includes:
- CORS headers for WebAssembly support
- Cross-Origin policies for SharedArrayBuffer
- Proper MIME types for WASM files
- Caching strategies

## Troubleshooting

### Emscripten Not Found
If you get an error about `emcc` not being found:
1. Install Emscripten following the official guide
2. Activate the Emscripten environment: `source ./emsdk_env.sh`
3. Verify: `emcc --version`

### Build Fails
- Ensure you have enough memory (at least 8GB RAM recommended)
- Try building with fewer threads: `scons platform=web -j2`
- Check the Godot/Orca build documentation

### CORS Issues
The `vercel.json` file includes all necessary CORS headers. If you still face issues:
- Clear browser cache
- Check browser console for specific error messages
- Ensure SharedArrayBuffer is supported in your browser

## Features

When deployed, the Orca Engine web editor includes:
- Full editor functionality in the browser
- Project creation and management
- Scene editing
- Script editing
- Asset import
- Game preview

## Browser Requirements

- Modern browser with WebAssembly support
- SharedArrayBuffer support (requires HTTPS and proper headers)
- WebGL 2.0 support
- Recommended: Chrome 91+, Firefox 89+, Edge 91+

## Performance Tips

- Use Chrome or Edge for best performance
- Enable hardware acceleration in browser settings
- Close unnecessary tabs to free up memory
- The initial load may take time due to WASM file size

## Support

For issues specific to:
- Orca Engine: Check the project repository
- Vercel deployment: https://vercel.com/docs
- Godot web export: https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_web.html
