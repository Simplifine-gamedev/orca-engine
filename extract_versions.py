#!/usr/bin/env python3
"""
Simple script to extract version constants from version.py for build system.
Outputs C++ preprocessor defines that can be used during compilation.
"""

import sys
import os

def main():
    # Try to find and load version.py
    version_file = os.path.join(os.path.dirname(__file__), 'version.py')
    
    if not os.path.exists(version_file):
        print("Error: version.py not found", file=sys.stderr)
        sys.exit(1)
    
    # Execute version.py to get variables
    version_vars = {}
    try:
        with open(version_file, 'r') as f:
            exec(f.read(), version_vars)
    except Exception as e:
        print(f"Error reading version.py: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract version values with fallbacks
    api_version = version_vars.get('api_version', '1.0')
    frontend_version = version_vars.get('frontend_version', '1.0.0')
    backend_version = version_vars.get('backend_version', '1.0.0')
    orca_version = version_vars.get('version', '1.0.0')
    
    # Output format depends on first argument
    output_format = sys.argv[1] if len(sys.argv) > 1 else 'cpp_defines'
    
    if output_format == 'cpp_defines':
        # Output C++ preprocessor defines
        print(f'#define GODOT_API_VERSION "{api_version}"')
        print(f'#define GODOT_FRONTEND_VERSION "{frontend_version}"')
        print(f'#define GODOT_BACKEND_VERSION "{backend_version}"')
        print(f'#define GODOT_ORCA_VERSION "{orca_version}"')
    
    elif output_format == 'env_vars':
        # Output shell environment variables for deploy.sh
        print(f'API_VERSION="{api_version}"')
        print(f'FRONTEND_VERSION="{frontend_version}"')  
        print(f'BACKEND_VERSION="{backend_version}"')
        print(f'ORCA_VERSION="{orca_version}"')
    
    elif output_format == 'json':
        # Output JSON for other tools
        import json
        versions = {
            'api_version': api_version,
            'frontend_version': frontend_version,
            'backend_version': backend_version,
            'orca_version': orca_version
        }
        print(json.dumps(versions, indent=2))
    
    else:
        print(f"Unknown output format: {output_format}", file=sys.stderr)
        print("Available formats: cpp_defines, env_vars, json", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
