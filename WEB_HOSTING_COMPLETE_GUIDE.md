# Orca Engine Web Hosting - Complete Guide

## Overview

The Orca Engine (Godot fork) can run entirely in the browser as a full-featured game development environment. **No dedicated VM or backend server is required** - everything runs client-side using WebAssembly.

This guide covers everything you need to know about hosting Orca on the web, including:
- How it works technically
- Building for web
- Hosting on Vercel
- Browser requirements and limitations
- Performance optimization

## How It Works

### Architecture
```
┌─────────────────┐
│   User Browser  │
├─────────────────┤
│  Orca Engine    │ ← WebAssembly (WASM)
│  - Editor UI    │ ← JavaScript + WebGL
│  - Game Runtime │ ← SharedArrayBuffer
│  - File System  │ ← IndexedDB/LocalStorage
└─────────────────┘
         ↓
   Static Files on
   Vercel CDN (No Server!)
```

The Orca Engine web build consists of:
1. **WebAssembly Module** (`.wasm`) - The compiled engine code
2. **JavaScript Glue** (`.js`) - Handles browser integration
3. **HTML Shell** - The web page container
4. **Pack File** (`.pck`) - Editor resources and assets

### Key Technologies
- **WebAssembly (WASM)**: Runs the compiled C++ engine code at near-native speed
- **WebGL 2.0**: Handles all graphics rendering
- **SharedArrayBuffer**: Enables multi-threading (requires special headers)
- **IndexedDB**: Stores projects and user data locally in browser
- **Web Audio API**: Handles audio playback

## Requirements

### Build Requirements
- **Emscripten** 3.1.39+ (WebAssembly compiler)
- **Python 3.6+** (for SCons build system)
- **SCons** 4.0+ (build system)
- **Node.js 16+** (for Vercel deployment)
- **8GB+ RAM** (for building)

### Hosting Requirements
- **Static file hosting** with custom header support
- **HTTPS** (required for SharedArrayBuffer)
- **CORS headers** properly configured
- **MIME types** for WASM files

### Browser Requirements
- Chrome 91+ / Edge 91+ / Firefox 89+ / Safari 15.2+
- WebGL 2.0 support
- SharedArrayBuffer support
- 4GB+ device RAM recommended

## Building for Web

### 1. Install Emscripten

```bash
# Clone Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# Install latest SDK
./emsdk install latest
./emsdk activate latest

# Add to PATH (add to .bashrc/.zshrc for permanent)
source ./emsdk_env.sh

# Verify installation
emcc --version
```

### 2. Build Orca for Web

```bash
# Clean previous builds
rm -rf bin/web_export
scons platform=web target=editor --clean

# Build the web editor (full build)
scons platform=web \
    target=editor \
    production=yes \
    optimize=size \
    javascript_eval=yes \
    threads=yes \
    use_closure_compiler=yes \
    -j$(nproc)

# Or use the convenience script
./build_web.sh
```

Build options explained:
- `target=editor`: Builds the full editor (not just runtime)
- `production=yes`: Optimized production build
- `optimize=size`: Minimize file size (important for web)
- `javascript_eval=yes`: Enable JavaScript evaluation
- `threads=yes`: Enable multi-threading via SharedArrayBuffer
- `use_closure_compiler=yes`: Further optimize JavaScript

### 3. Prepare Web Export

The build creates several files:
```
bin/
├── orca.web.editor.wasm32.wasm    # ~50-70MB - Main engine
├── orca.web.editor.wasm32.js      # ~2MB - JavaScript glue
├── orca.web.editor.wasm32.audio.worklet.js  # Audio worker
├── orca.web.editor.wasm32.worker.js         # Web worker
└── orca.editor.wasm32.pck         # ~30MB - Editor resources
```

## Vercel Deployment

### Why Vercel?

Vercel is ideal for Orca because:
- **Global CDN**: Fast loading worldwide
- **Custom Headers**: Full control over CORS/COOP/COEP
- **Automatic HTTPS**: Required for SharedArrayBuffer
- **GitHub Integration**: Auto-deploy on push
- **Free Tier**: Sufficient for most projects

### Deployment Steps

#### 1. Configure vercel.json

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "bin/web_export",
  "framework": null,
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Cross-Origin-Opener-Policy",
          "value": "same-origin"
        },
        {
          "key": "Cross-Origin-Embedder-Policy", 
          "value": "require-corp"
        },
        {
          "key": "Cross-Origin-Resource-Policy",
          "value": "cross-origin"
        },
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        }
      ]
    },
    {
      "source": "/(.*).wasm",
      "headers": [
        {
          "key": "Content-Type",
          "value": "application/wasm"
        },
        {
          "key": "Content-Encoding",
          "value": "gzip"
        },
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

#### 2. Deploy via CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy (first time)
vercel

# Deploy to production
vercel --prod

# Or use the convenience script
./deploy_to_vercel.sh
```

#### 3. Deploy via GitHub

1. Connect repository to Vercel Dashboard
2. Set build command: `npm run build` or `./build_web.sh`
3. Set output directory: `bin/web_export`
4. Enable automatic deployments

### Environment Variables

Set in Vercel Dashboard or `.env`:
```bash
# Optional: Analytics
VITE_ANALYTICS_ID=your-analytics-id

# Optional: Custom domain
CUSTOM_DOMAIN=orca.yourdomain.com

# Build optimization
NODE_OPTIONS=--max-old-space-size=8192
```

## Critical: SharedArrayBuffer Requirements

SharedArrayBuffer is **required** for Orca to function properly (enables threading). This requires specific security headers:

### Required Headers (Already in vercel.json)
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Testing SharedArrayBuffer
```javascript
// Add to your HTML to verify
if (typeof SharedArrayBuffer === 'undefined') {
  alert('SharedArrayBuffer not available! Check COOP/COEP headers.');
}
```

### Troubleshooting SharedArrayBuffer

If SharedArrayBuffer is unavailable:
1. **Check HTTPS**: Must use HTTPS (Vercel handles this)
2. **Verify Headers**: Use browser DevTools Network tab
3. **Browser Support**: Some browsers disable it in certain contexts
4. **Iframe Issues**: Cannot be used in cross-origin iframes

## Performance Optimization

### 1. Enable Compression

Vercel automatically compresses files, but ensure WASM files are compressed:
```bash
# Pre-compress files before deployment
gzip -9 bin/web_export/*.wasm
gzip -9 bin/web_export/*.pck
```

### 2. Optimize Loading

Create a custom loading experience:
```html
<!DOCTYPE html>
<html>
<head>
  <title>Orca Engine</title>
  <style>
    .loader {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }
    .progress-bar {
      width: 300px;
      height: 20px;
      background: #333;
      border-radius: 10px;
    }
    .progress-fill {
      height: 100%;
      background: #4CAF50;
      border-radius: 10px;
      transition: width 0.3s;
    }
  </style>
</head>
<body>
  <div class="loader">
    <h2>Loading Orca Engine...</h2>
    <div class="progress-bar">
      <div class="progress-fill" id="progress"></div>
    </div>
    <p id="status">Initializing...</p>
  </div>
  
  <script>
    // Custom loading progress
    var Module = {
      onRuntimeInitialized: function() {
        document.querySelector('.loader').style.display = 'none';
      },
      setStatus: function(text) {
        document.getElementById('status').innerText = text;
      },
      totalDependencies: 0,
      monitorRunDependencies: function(left) {
        this.totalDependencies = Math.max(this.totalDependencies, left);
        if (left) {
          var progress = (this.totalDependencies - left) / this.totalDependencies * 100;
          document.getElementById('progress').style.width = progress + '%';
        }
      }
    };
  </script>
  <script src="orca.web.editor.wasm32.js"></script>
</body>
</html>
```

### 3. CDN and Caching

Vercel provides excellent CDN coverage, but you can enhance it:
- Use Cache-Control headers (already in vercel.json)
- Enable Edge caching for static assets
- Consider using a separate CDN for large assets

### 4. Memory Management

Help browsers manage memory:
```javascript
// Add to initialization
if (navigator.deviceMemory && navigator.deviceMemory < 4) {
  alert('Warning: Your device may have insufficient memory for optimal performance.');
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
  if (Module && Module.abort) {
    Module.abort();
  }
});
```

## Testing Locally

### Simple Python Server
```bash
# Python 3
python3 -m http.server 8000 --directory bin/web_export

# With proper headers (create serve.py)
```

Create `serve.py`:
```python
#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()

if __name__ == '__main__':
    port = 8000
    httpd = HTTPServer(('localhost', port), CORSHTTPRequestHandler)
    print(f'Server running at http://localhost:{port}/')
    httpd.serve_forever()
```

### Using Node.js
```bash
# Install serve with CORS support
npm install -g serve

# Serve with headers
serve bin/web_export -l 8000 \
  --cors \
  --ssl-cert cert.pem \
  --ssl-key key.pem
```

## Common Issues and Solutions

### 1. "SharedArrayBuffer is not defined"
**Solution**: Ensure COOP/COEP headers are set correctly. Check browser console for header warnings.

### 2. "Out of Memory" errors
**Solution**: 
- Build with `optimize=size`
- Reduce texture sizes
- Enable texture compression
- Warn users about memory requirements

### 3. Slow initial load
**Solution**:
- Enable gzip compression
- Use a loading screen
- Consider lazy-loading assets
- Implement progressive loading

### 4. Audio not working
**Solution**: 
- User must interact with page first (browser requirement)
- Add a "Click to Start" button
- Check Web Audio API support

### 5. File system issues
**Solution**:
- IndexedDB has size limits (~50% of free disk space)
- Implement cleanup for old projects
- Warn users about storage limits

## Limitations of Web Version

### What Works
✅ Full editor interface
✅ Scene editing
✅ Script editing (GDScript)
✅ Asset importing (with limitations)
✅ Project management
✅ Game preview/testing
✅ Most 2D features
✅ Most 3D features (WebGL 2.0)

### What Doesn't Work
❌ Native plugins (GDExtension)
❌ C# scripting
❌ Direct file system access
❌ Some advanced rendering features
❌ Threads (limited to SharedArrayBuffer)
❌ Large file imports (>100MB)
❌ Some platform-specific features

## Monitoring and Analytics

### Add Analytics (Optional)
```html
<!-- Add to index.html -->
<script>
  // Simple analytics
  window.addEventListener('load', function() {
    fetch('/api/analytics', {
      method: 'POST',
      body: JSON.stringify({
        event: 'orca_load',
        timestamp: Date.now(),
        memory: navigator.deviceMemory || 'unknown',
        platform: navigator.platform
      })
    });
  });
</script>
```

### Performance Monitoring
```javascript
// Monitor performance
const perfObserver = new PerformanceObserver((items) => {
  items.getEntries().forEach((entry) => {
    console.log(entry.name, entry.duration);
  });
});
perfObserver.observe({ entryTypes: ["measure"] });
```

## Next Steps

1. **Custom Domain**: Set up a custom domain in Vercel
2. **PWA Support**: Make it installable as a Progressive Web App
3. **Offline Mode**: Implement service workers for offline support
4. **Cloud Save**: Add cloud project storage (optional backend)
5. **Collaboration**: Implement real-time collaboration features

## Summary

The Orca Engine can run entirely in the browser without any backend infrastructure. Vercel provides an excellent platform for hosting with:
- Automatic HTTPS
- Global CDN distribution  
- Custom header support for SharedArrayBuffer
- GitHub integration for CI/CD
- Generous free tier

The key requirements are:
1. Build with Emscripten for WebAssembly
2. Configure COOP/COEP headers correctly
3. Ensure HTTPS is enabled
4. Use a modern browser with SharedArrayBuffer support

No VMs or backend servers needed - everything runs client-side in the browser!

## Resources

- [Godot Web Export Docs](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_web.html)
- [Vercel Documentation](https://vercel.com/docs)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [MDN SharedArrayBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
- [WebAssembly.org](https://webassembly.org/)

## Support

- **Orca Engine Issues**: Create an issue in your GitHub repo
- **Vercel Issues**: https://vercel.com/support
- **Build Issues**: Check Emscripten and SCons documentation
