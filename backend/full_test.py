#!/usr/bin/env python3
"""
Full test with indexing first, then searching
"""

import requests
import json
import time

def full_test():
    """Test indexing then searching"""
    print("🚀 Full Embedding Test (Index + Search)")
    print("=" * 40)
    
    # Your session details
    user_id = "106469680334583136136"
    machine_id = "XP4191P2VD"
    project_root = "/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project"
    
    base_url = "http://127.0.0.1:8000"
    
    # Step 1: Index the project
    print("📁 Step 1: Indexing project...")
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
            print(f"   ✅ Indexing successful!")
            stats = result.get('stats', {})
            for key, value in stats.items():
                print(f"      {key}: {value}")
        else:
            print(f"   ❌ Indexing failed: {response.status_code}")
            print(f"      {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Indexing exception: {e}")
        return
    
    # Wait a moment for indexing to complete
    print("\n⏱️  Waiting 5 seconds for indexing to complete...")
    time.sleep(5)
    
    # Step 2: Check status
    print("\n📊 Step 2: Checking project status...")
    try:
        response = requests.post(f"{base_url}/embed", json={
            "action": "status",
            "user_id": user_id,
            "machine_id": machine_id,
            "project_root": project_root
        })
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            print(f"   ✅ Status:")
            for key, value in summary.items():
                print(f"      {key}: {value}")
        else:
            print(f"   ❌ Status failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Status exception: {e}")
    
    # Step 3: Test searches
    print("\n🔍 Step 3: Testing searches...")
    
    queries = [
        "game files",
        "project assets", 
        "godot scenes",
        "scripts code",
        "any content"
    ]
    
    for query in queries:
        print(f"\n   🔎 Searching: '{query}'")
        
        try:
            response = requests.post(f"{base_url}/embed", json={
                "action": "search",
                "query": query,
                "k": 5,
                "user_id": user_id,
                "machine_id": machine_id,
                "project_root": project_root
            })
            
            if response.status_code == 200:
                result = response.json()
                results = result.get('results', [])
                
                if results:
                    print(f"      ✅ Found {len(results)} results:")
                    for i, item in enumerate(results):
                        print(f"         {i+1}. {item['file_path']}")
                        print(f"            Similarity: {item['similarity']:.3f}, Type: {item['modality']}")
                else:
                    print("      📭 No results found")
            else:
                print(f"      ❌ Search failed: {response.status_code}")
                print(f"         {response.text}")
                
        except Exception as e:
            print(f"      ❌ Search exception: {e}")
    
    print(f"\n🎉 Test completed!")
    print(f"\n💡 If no files were found, check:")
    print(f"   1. Project path: {project_root}")
    print(f"   2. Make sure the project has .gd, .tscn, .png, etc. files")
    print(f"   3. Check server logs for any errors")

if __name__ == "__main__":
    full_test()