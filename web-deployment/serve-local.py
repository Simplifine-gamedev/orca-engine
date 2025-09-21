#!/usr/bin/env python3
"""
Local development server for Orca Engine Web Editor
Serves files with proper CORS headers for SharedArrayBuffer support
"""

import http.server
import socketserver
import os
import sys
import socket
from contextlib import closing
from urllib.parse import urlparse

class OrcaWebHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required headers for SharedArrayBuffer (threading) support
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        
        # Additional CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept')
        
        # Content-Type headers for specific file types
        if self.path.endswith('.wasm'):
            self.send_header('Content-Type', 'application/wasm')
        elif self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        elif self.path.endswith('.html'):
            self.send_header('Content-Type', 'text/html')
        elif self.path.endswith('.png'):
            self.send_header('Content-Type', 'image/png')
        
        # Security headers
        self.send_header('X-Content-Type-Options', 'nosniff')
        
        super().end_headers()

    def do_GET(self):
        # Handle root path
        if self.path == '/':
            self.path = '/index.html'
        
        # Serve the file
        super().do_GET()

    def log_message(self, format, *args):
        # Custom log format
        print(f"🌐 {self.address_string()} - {format % args}")

def _find_free_port(preferred=8082, start=8080, end=8099):
    # Port from env or CLI
    env_port = os.environ.get('PORT')
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass
    # Try preferred
    for p in [preferred] + [x for x in range(start, end + 1) if x != preferred]:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    return preferred

def main():
    port = _find_free_port()
    
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Starting Orca Engine Web Development Server")
    print("=" * 50)
    print(f"📁 Serving directory: {os.getcwd()}")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"🔗 Open in browser: http://localhost:{port}")
    print("=" * 50)
    print("📋 Features enabled:")
    print("   ✅ SharedArrayBuffer (COOP/COEP headers)")
    print("   ✅ WebAssembly MIME types")
    print("   ✅ CORS headers")
    print("   ✅ Security headers")
    print("=" * 50)
    print("💡 Press Ctrl+C to stop the server")
    print("")
    
    try:
        with socketserver.TCPServer(("", port), OrcaWebHandler) as httpd:
            httpd.allow_reuse_address = True
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {port} is already in use")
            print("💡 Try a different port or stop the other server")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
