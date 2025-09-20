#!/usr/bin/env python3
"""
Generate version header file for Godot frontend compilation.
This reads version.py and creates a C++ header with version constants.
"""

import os
import sys

def main():
    # Load version.py
    version_file = os.path.join(os.path.dirname(__file__), 'version.py')
    
    if not os.path.exists(version_file):
        print("Error: version.py not found", file=sys.stderr)
        sys.exit(1)
    
    version_vars = {}
    try:
        with open(version_file, 'r') as f:
            exec(f.read(), version_vars)
    except Exception as e:
        print(f"Error reading version.py: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract versions
    api_version = version_vars.get('api_version', '1.0')
    frontend_version = version_vars.get('frontend_version', '1.0.0')
    backend_version = version_vars.get('backend_version', '1.0.0')
    orca_version = version_vars.get('version', '1.0.0')
    
    # Generate header content
    header_content = f'''// Auto-generated version header from version.py
// DO NOT EDIT MANUALLY - This file is overwritten during build
#ifndef GODOT_VERSION_GENERATED_H
#define GODOT_VERSION_GENERATED_H

#define GODOT_API_VERSION "{api_version}"
#define GODOT_FRONTEND_VERSION "{frontend_version}"
#define GODOT_BACKEND_VERSION "{backend_version}"
#define GODOT_ORCA_VERSION "{orca_version}"

#endif // GODOT_VERSION_GENERATED_H
'''
    
    # Write to core/version_generated.h
    output_file = os.path.join('core', 'version_generated.h')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            f.write(header_content)
        print(f"✅ Generated {output_file} with versions:")
        print(f"   API: {api_version}")
        print(f"   Frontend: {frontend_version}")
        print(f"   Backend: {backend_version}")
        print(f"   Orca: {orca_version}")
    except Exception as e:
        print(f"Error writing header file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
