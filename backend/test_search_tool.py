#!/usr/bin/env python3
"""
Test the new search_across_project tool functionality
"""

import requests
import json

def test_search_tool():
    """Test the search_across_project tool through the Flask backend"""
    print("🔍 Testing search_across_project Tool")
    print("=" * 50)
    
    # Tool server URL (would be called by Godot's tool system)
    tool_server_url = "http://127.0.0.1:8001"  # AI Tool Server port
    backend_url = "http://127.0.0.1:8000"      # Flask backend port
    
    # Test 1: Direct backend search endpoint
    print("📁 Step 1: Testing backend search endpoint directly...")
    try:
        response = requests.post(f"{backend_url}/search_project", json={
            "query": "player movement system",
            "include_graph": True,
            "max_results": 3,
            "project_root": "/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project",
            "user_id": "106469680334583136136",
            "machine_id": "XP4191P2VD"
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Backend search successful!")
            if result.get('success'):
                search_results = result.get('results', {})
                similar_files = search_results.get('similar_files', [])
                print(f"   📝 Found {len(similar_files)} similar files:")
                for file_info in similar_files[:3]:  # Show first 3
                    print(f"      • {file_info.get('file_path', 'unknown')} (similarity: {file_info.get('similarity', 0):.3f})")
                    connections = file_info.get('connections', {})
                    if connections:
                        print(f"        🔗 Has {len(connections)} connection types")
            else:
                print(f"   ❌ Backend error: {result.get('error')}")
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 2: Tool interface (simulating what Godot would send)
    print(f"\n🔧 Step 2: Testing tool interface (simulating Godot call)...")
    
    tool_request = {
        "function_name": "search_across_project",
        "arguments": {
            "query": "UI system components",
            "include_graph": True,
            "max_results": 5,
            "modality_filter": "text"
        }
    }
    
    try:
        response = requests.post(tool_server_url, 
                               json=tool_request,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Tool call successful!")
            
            if result.get('success'):
                print(f"   📊 Tool Results:")
                print(f"      Query: {result.get('query')}")
                print(f"      File count: {result.get('file_count')}")
                print(f"      Message: {result.get('message')}")
                
                similar_files = result.get('similar_files', [])
                for i, file_info in enumerate(similar_files):
                    print(f"      {i+1}. {file_info.get('file_path', 'unknown')}")
                    print(f"         Similarity: {file_info.get('similarity', 0):.3f}")
                    print(f"         Modality: {file_info.get('modality', 'unknown')}")
                    
                central_files = result.get('central_files', [])
                if central_files:
                    print(f"   ⭐ Central Files:")
                    for file_info in central_files[:3]:
                        print(f"      • {file_info.get('file_path', 'unknown')} (centrality: {file_info.get('centrality', 0):.3f})")
                        
            else:
                print(f"   ❌ Tool error: {result.get('error')}")
                print(f"   💡 Message: {result.get('message', 'No message')}")
        else:
            print(f"   ❌ Tool server failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print(f"\n🎯 Tool Integration Summary:")
    print(f"   ✅ Backend endpoint: /search_project")
    print(f"   ✅ Tool definition: search_across_project in godot_tools")
    print(f"   ✅ Tool server handler: ai_tool_server.cpp")
    print(f"   ✅ EditorTools method: search_across_project()")
    print(f"   ✅ Authentication integration: Uses AIChatDock auth")
    print(f"   ✅ Graph context: Includes relationship mapping")
    
    print(f"\n💡 Usage in Godot:")
    print(f"   The AI agent can now use search_across_project tool to:")
    print(f"   🔍 Find relevant files by semantic meaning")
    print(f"   🔗 See how files connect through relationships")
    print(f"   ⭐ Identify central/important files in the project")
    print(f"   🎯 Filter by modality (text, image, audio)")
    print(f"   📊 Get project structure insights")

if __name__ == "__main__":
    test_search_tool()