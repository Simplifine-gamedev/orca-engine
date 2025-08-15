#!/usr/bin/env python3
"""
Simple test script that works with your current authenticated session
"""

import requests
import json

def test_embedding_search():
    """Test the embedding search functionality"""
    print("🔍 Testing Embedding Search")
    print("=" * 30)
    
    # Your current session details
    user_id = "106469680334583136136"
    machine_id = "XP4191P2VD"
    project_root = "/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project"
    
    # Test queries
    queries = [
        "player movement code",
        "enemy sprites", 
        "UI buttons",
        "background music",
        "shader effects"
    ]
    
    base_url = "http://127.0.0.1:8000"
    
    for query in queries:
        print(f"\n🔎 Searching: '{query}'")
        
        try:
            response = requests.post(f"{base_url}/embed", json={
                "action": "search",
                "query": query,
                "k": 3,
                "user_id": user_id,
                "machine_id": machine_id,
                "project_root": project_root
            })
            
            if response.status_code == 200:
                result = response.json()
                results = result.get('results', [])
                
                if results:
                    print(f"   ✅ Found {len(results)} results:")
                    for i, item in enumerate(results):
                        print(f"      {i+1}. {item['file_path']}")
                        print(f"         Similarity: {item['similarity']:.3f}, Type: {item['modality']}")
                else:
                    print("   📭 No results found")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"      {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Test project status
    print(f"\n📊 Getting project status...")
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
            print(f"   ✅ Project Status:")
            for key, value in summary.items():
                print(f"      {key}: {value}")
        else:
            print(f"   ❌ Status Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Status Exception: {e}")

if __name__ == "__main__":
    test_embedding_search()