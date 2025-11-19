#!/usr/bin/env python3
"""Check what's in the enhanced graph nodes"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import cloud_vector_manager
import json

user_id = "069ae9ad-bae1-4e0f-a74b-5d2b4a770b9d"
project_id = "5695ea631a3d5da28464d01705efc939"

graph = cloud_vector_manager.get_enhanced_graph(user_id, project_id)

if graph:
    for node in graph.get('nodes', []):
        if node.get('file_path') == 'player.gd':
            print("player.gd node data:")
            print(json.dumps(node, indent=2, default=str))
            break


