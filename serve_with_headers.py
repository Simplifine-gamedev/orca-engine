#!/usr/bin/env python3
"""
Serve Orca Engine with proper CORS headers for SharedArrayBuffer
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required headers for SharedArrayBuffer
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        
        # Additional headers for better compatibility
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Cache control for WASM files
        if self.path.endswith('.wasm'):
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        
        super().end_headers()

if __name__ == '__main__':
    port = 8000
    directory = 'bin/web_export'
    
    # Change to the web_export directory
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print(f"Error: Directory {directory} not found!")
        print("Run this script from the orca-engine root directory")
        sys.exit(1)
    
    httpd = HTTPServer(('localhost', port), CORSHTTPRequestHandler)
    print(f"🚀 Orca Engine Web Server")
    print(f"✅ SharedArrayBuffer headers enabled")
    print(f"🌐 Server running at http://localhost:{port}/")
    print(f"📁 Serving from: {os.getcwd()}")
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped")
        httpd.shutdown()
