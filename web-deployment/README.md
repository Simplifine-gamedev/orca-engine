# Orca Engine Web Deployment

This directory contains the web build of Orca Engine for deployment to Vercel at `editor.orcaengine.ai`.

## Files

- **`index.html`** - Main HTML page for the Orca web editor
- **`orca.web.editor.wasm32.wasm`** - Main engine WebAssembly file (87MB)
- **`orca.web.editor.wasm32.js`** - JavaScript glue code for WASM
- **`orca.web.editor.wasm32.*.js`** - Additional JavaScript files (workers, engine, etc.)
- **`favicon.png`** - Orca Engine favicon
- **`logo.png`** - Orca Engine logo
- **`vercel.json`** - Vercel deployment configuration with CORS headers
- **`deploy-orca-web.sh`** - Automated deployment script

## Quick Deployment

```bash
cd web-deployment
./deploy-orca-web.sh
```

## Manual Deployment

```bash
cd web-deployment
vercel --prod
```

## Features

- ✅ Full Orca Engine editor in browser
- ✅ WebAssembly with threading support (SharedArrayBuffer)
- ✅ Proper CORS headers for web assembly
- ✅ Optimized caching for static assets
- ✅ Custom domain support (editor.orcaengine.ai)

## Browser Requirements

- Chrome 91+ / Edge 91+ / Firefox 89+ / Safari 15.2+
- WebGL 2.0 support
- SharedArrayBuffer support (requires HTTPS)
- 4GB+ device RAM recommended

## Custom Domain Setup

1. Deploy using the script above
2. Go to [Vercel Dashboard](https://vercel.com/dashboard)
3. Select your project → Settings → Domains
4. Add `editor.orcaengine.ai`
5. Configure DNS: `CNAME editor.orcaengine.ai → cname.vercel-dns.com`

## Development

To rebuild the web export:
```bash
cd .. # Go back to project root
./build_web.sh
cp bin/web_export/* web-deployment/
```

## File Sizes

- Total: ~135MB
- WASM file: 87MB (largest component)
- JS files: ~1MB combined
- Assets: ~47MB

The large WASM file contains the entire Orca Engine compiled for WebAssembly.



