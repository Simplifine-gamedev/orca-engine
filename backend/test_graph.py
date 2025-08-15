#!/usr/bin/env python3
"""
Test the graph structure functionality
"""

import requests
import json

def test_graph_functionality():
    """Test the new graph-based features"""
    print("🌐 Testing Graph Structure Functionality")
    print("=" * 40)
    
    # Your session details
    user_id = "106469680334583136136"
    machine_id = "XP4191P2VD"
    project_root = "/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project"
    
    base_url = "http://127.0.0.1:8000"
    
    # Test 1: Index project (this will build the graph)
    print("📁 Step 1: Indexing project and building graph...")
    try:
        response = requests.post(f"{base_url}/embed", json={
            "action": "index_project",
            "user_id": user_id,
            "machine_id": machine_id,
            "project_root": project_root,
            "force_reindex": True
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Indexing and graph building successful!")
            stats = result.get('stats', {})
            for key, value in stats.items():
                print(f"      {key}: {value}")
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return
    
    # Test 2: Get project status with graph info
    print("\n📊 Step 2: Getting project status with graph information...")
    try:
        response = requests.post(f"{base_url}/embed", json={
            "action": "status",
            "user_id": user_id,
            "machine_id": machine_id,
            "project_root": project_root
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Status retrieved!")
            
            # Show embedding summary
            summary = result.get('summary', {})
            print(f"   📊 Embedding Summary:")
            for key, value in summary.items():
                print(f"      {key}: {value}")
            
            # Show graph summary
            graph = result.get('graph', {})
            if graph:
                print(f"   🌐 Graph Summary:")
                for key, value in graph.items():
                    if key == 'central_files':
                        print(f"      {key}:")
                        for file_info in value[:3]:  # Show top 3
                            print(f"         • {file_info.get('file', 'unknown')} (centrality: {file_info.get('centrality', 0):.3f})")
                    else:
                        print(f"      {key}: {value}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 3: Search with graph context
    print("\n🔍 Step 3: Testing enhanced search with graph relationships...")
    try:
        response = requests.post(f"{base_url}/embed", json={
            "action": "search",
            "query": "player movement",
            "k": 3,
            "include_graph": True,
            "user_id": user_id,
            "machine_id": machine_id,
            "project_root": project_root
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Graph-enhanced search successful!")
            
            results = result.get('results', {})
            
            # Show similarity results
            similarity_results = results.get('similarity_results', [])
            if similarity_results:
                print(f"   📝 Similar Files:")
                for path, score, modality in similarity_results:
                    print(f"      • {path} (similarity: {score:.3f}, type: {modality})")
            
            # Show connected files
            connected_files = results.get('connected_files', {})
            if connected_files:
                print(f"   🔗 Connected Files:")
                for main_file, connections in connected_files.items():
                    print(f"      {main_file}:")
                    for relation_type, files in connections.items():
                        if files:
                            print(f"         {relation_type}: {len(files)} files")
                            for f in files[:2]:  # Show first 2
                                print(f"            • {f}")
            
            # Show central files
            central_files = results.get('central_files', [])
            if central_files:
                print(f"   ⭐ Most Central Files:")
                for file_path, centrality in central_files[:3]:
                    print(f"      • {file_path} (centrality: {centrality:.3f})")
                    
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 4: Get connections for a specific file
    print("\n🔗 Step 4: Testing file connection analysis...")
    
    # Get a file to analyze (use one from previous results)
    test_file = "/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project/main.tscn"
    
    try:
        response = requests.post(f"{base_url}/embed", json={
            "action": "graph_connections",
            "file_path": test_file,
            "max_depth": 2,
            "user_id": user_id,
            "machine_id": machine_id,
            "project_root": project_root
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Connection analysis successful!")
            
            connections = result.get('connections', {})
            if connections:
                print(f"   🔗 Connections for {test_file}:")
                for relation_type, files in connections.items():
                    print(f"      {relation_type}: {len(files)} files")
                    for f in files[:3]:  # Show first 3
                        print(f"         • {f}")
            else:
                print(f"   📭 No connections found for {test_file}")
                
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 5: Get most central files
    print("\n⭐ Step 5: Testing central files analysis...")
    try:
        response = requests.post(f"{base_url}/embed", json={
            "action": "central_files",
            "top_k": 5,
            "user_id": user_id,
            "machine_id": machine_id,
            "project_root": project_root
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Central files analysis successful!")
            
            central_files = result.get('central_files', [])
            if central_files:
                print(f"   ⭐ Most Important Files:")
                for i, file_info in enumerate(central_files):
                    file_path = file_info.get('file_path', 'unknown')
                    centrality = file_info.get('centrality', 0)
                    print(f"      {i+1}. {file_path} (centrality: {centrality:.3f})")
            else:
                print(f"   📭 No central files found")
                
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print(f"\n🎉 Graph functionality test completed!")
    print(f"\n💡 Graph Features Available:")
    print(f"   🔍 Enhanced search with relationship context")
    print(f"   🔗 File connection analysis (scripts, scenes, resources)")
    print(f"   ⭐ Central file identification (most important files)")
    print(f"   📊 Project structure visualization")
    print(f"   🌐 Relationship mapping (extends, preloads, instances, etc.)")

if __name__ == "__main__":
    test_graph_functionality()