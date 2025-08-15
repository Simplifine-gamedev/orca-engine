#!/usr/bin/env python3
"""
Test script for multimodal embedding system
Tests both text and image embedding/querying functionality
"""

import requests
import json
import os
import base64
from PIL import Image
import io

class EmbeddingTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Test user credentials (replace with actual values)
        self.user_id = "106469680334583136136"  # Your user ID
        self.machine_id = "XP4191P2VD"  # Your machine ID
        self.project_root = "/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project"
        
        # You'd get these from the auth flow
        self.auth_token = "test_token"  # Replace with actual token
        self.api_key = os.getenv('OPENAI_API_KEY', '')
    
    def _make_request(self, action, data=None):
        """Make authenticated request to embedding endpoint"""
        if data is None:
            data = {}
        
        # Add required fields
        data.update({
            "action": action,
            "user_id": self.user_id,
            "machine_id": self.machine_id,
            "project_root": self.project_root
        })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "X-API-Key": self.api_key
        }
        
        response = self.session.post(
            f"{self.base_url}/embed",
            headers=headers,
            json=data
        )
        
        return response
    
    def test_project_indexing(self):
        """Test initial project indexing"""
        print("🔍 Testing project indexing...")
        
        response = self._make_request("index_project", {"force_reindex": True})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Project indexed successfully!")
            print(f"   Stats: {result.get('stats', {})}")
            return True
        else:
            print(f"❌ Indexing failed: {response.status_code} - {response.text}")
            return False
    
    def test_text_search(self, query="player movement code"):
        """Test text-based search"""
        print(f"\n🔍 Testing text search: '{query}'...")
        
        response = self._make_request("search", {
            "query": query,
            "k": 5,
            "modality_filter": "text"
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Text search successful!")
            
            results = result.get('results', [])
            if results:
                print(f"   Found {len(results)} results:")
                for i, item in enumerate(results):
                    print(f"   {i+1}. {item['file_path']} (similarity: {item['similarity']:.3f}, modality: {item['modality']})")
            else:
                print("   No results found.")
            return results
        else:
            print(f"❌ Text search failed: {response.status_code} - {response.text}")
            return []
    
    def test_image_search(self, query="player sprite graphics"):
        """Test image-based search"""
        print(f"\n🎨 Testing image search: '{query}'...")
        
        response = self._make_request("search", {
            "query": query,
            "k": 5,
            "modality_filter": "image"
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image search successful!")
            
            results = result.get('results', [])
            if results:
                print(f"   Found {len(results)} image results:")
                for i, item in enumerate(results):
                    print(f"   {i+1}. {item['file_path']} (similarity: {item['similarity']:.3f}, modality: {item['modality']})")
            else:
                print("   No image results found.")
            return results
        else:
            print(f"❌ Image search failed: {response.status_code} - {response.text}")
            return []
    
    def test_multimodal_search(self, query="game characters"):
        """Test search across all modalities"""
        print(f"\n🌈 Testing multimodal search: '{query}'...")
        
        response = self._make_request("search", {
            "query": query,
            "k": 10  # Get more results to see different modalities
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Multimodal search successful!")
            
            results = result.get('results', [])
            if results:
                print(f"   Found {len(results)} results across all modalities:")
                
                # Group by modality
                by_modality = {}
                for item in results:
                    modality = item['modality']
                    if modality not in by_modality:
                        by_modality[modality] = []
                    by_modality[modality].append(item)
                
                for modality, items in by_modality.items():
                    print(f"\n   📁 {modality.upper()} files:")
                    for i, item in enumerate(items):
                        print(f"      {i+1}. {item['file_path']} (similarity: {item['similarity']:.3f})")
            else:
                print("   No results found.")
            return results
        else:
            print(f"❌ Multimodal search failed: {response.status_code} - {response.text}")
            return []
    
    def test_project_status(self):
        """Test getting project embedding status"""
        print(f"\n📊 Testing project status...")
        
        response = self._make_request("status")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status retrieved successfully!")
            
            summary = result.get('summary', {})
            print(f"   Project summary:")
            for key, value in summary.items():
                print(f"   - {key}: {value}")
            return summary
        else:
            print(f"❌ Status failed: {response.status_code} - {response.text}")
            return {}
    
    def create_test_image(self, filename="test_player_sprite.png"):
        """Create a simple test image for testing"""
        print(f"\n🎨 Creating test image: {filename}...")
        
        # Create a simple test image
        img = Image.new('RGBA', (64, 64), color='blue')
        
        # Add some basic shapes to make it look like a character
        pixels = img.load()
        
        # Head (circle-ish)
        for x in range(20, 44):
            for y in range(10, 25):
                if ((x-32)**2 + (y-17)**2) < 100:
                    pixels[x, y] = (255, 200, 150, 255)  # Skin color
        
        # Body (rectangle)
        for x in range(25, 39):
            for y in range(25, 50):
                pixels[x, y] = (0, 150, 255, 255)  # Blue shirt
        
        # Save to project directory if it exists
        test_path = os.path.join(self.project_root, filename)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        try:
            img.save(test_path)
            print(f"✅ Test image saved to: {test_path}")
            return test_path
        except Exception as e:
            # Fallback to current directory
            fallback_path = filename
            img.save(fallback_path)
            print(f"✅ Test image saved to: {fallback_path} (fallback)")
            return fallback_path
    
    def test_file_indexing(self, file_path):
        """Test indexing a specific file"""
        print(f"\n📁 Testing file indexing: {file_path}...")
        
        response = self._make_request("index_file", {"file_path": file_path})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File indexed successfully!")
            print(f"   Indexed: {result.get('indexed', False)}")
            return True
        else:
            print(f"❌ File indexing failed: {response.status_code} - {response.text}")
            return False
    
    def run_full_test(self):
        """Run complete test suite"""
        print("🚀 Starting Multimodal Embedding System Test")
        print("=" * 50)
        
        # Test 1: Check if server is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print("❌ Server health check failed")
                return False
        except requests.exceptions.RequestException:
            print("❌ Server not accessible. Make sure backend is running on port 8000")
            return False
        
        print("✅ Server is accessible")
        
        # Test 2: Create a test image
        test_image_path = self.create_test_image()
        
        # Test 3: Index the test image
        self.test_file_indexing(test_image_path)
        
        # Test 4: Index the project
        self.test_project_indexing()
        
        # Test 5: Get project status
        self.test_project_status()
        
        # Test 6: Test different types of searches
        self.test_text_search("player movement code")
        self.test_text_search("enemy AI behavior")
        self.test_image_search("character sprites")
        self.test_image_search("UI buttons")
        self.test_multimodal_search("game assets")
        
        print("\n" + "=" * 50)
        print("🎉 Test suite completed!")
        print("\n💡 Usage in main agent:")
        print("   1. Call embedding endpoint with 'search' action")
        print("   2. Include query, user_id, machine_id, project_root")
        print("   3. Optional: Use modality_filter for specific content types")
        print("   4. Process returned file paths and similarity scores")

def main():
    """Main test function"""
    print("🤖 Multimodal Embedding System Tester")
    print("This script tests both text and image embedding/query functionality")
    print()
    
    # Check if required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set in environment")
    
    tester = EmbeddingTester()
    tester.run_full_test()

if __name__ == "__main__":
    main()