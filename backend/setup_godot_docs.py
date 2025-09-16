#!/usr/bin/env python3
"""
Setup script to index Godot documentation
Run this once to populate the Weaviate database with Godot docs
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

def main():
    print("🚀 Godot Documentation Setup")
    print("=" * 40)
    
    # Load environment
    
    
    # Check requirements
    required_env = ['WEAVIATE_URL', 'WEAVIATE_API_KEY', 'OPENAI_API_KEY']
    missing = [var for var in required_env if not os.getenv(var)]
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   {var}")
        print("\nCreate a .env file with:")
        print("WEAVIATE_URL=your_weaviate_cluster_url")
        print("WEAVIATE_API_KEY=your_weaviate_api_key") 
        print("OPENAI_API_KEY=your_openai_api_key")
        return
    
    # Check dependencies
    try:
        import weaviate
        import openai
        from bs4 import BeautifulSoup
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install weaviate-client openai beautifulsoup4")
        return
    
    print("\n📚 Starting Godot documentation indexing...")
    print("This will:")
    print("  1. Download latest Godot class documentation")
    print("  2. Parse XML into structured chunks")
    print("  3. Index into Weaviate with embeddings")
    print("  4. Test search functionality")
    print()
    
    # Import and run indexer
    try:
        from godot_docs_indexer import GodotDocsIndexer
        
        indexer = GodotDocsIndexer(
            os.getenv('WEAVIATE_URL'),
            os.getenv('WEAVIATE_API_KEY'),
            os.getenv('OPENAI_API_KEY')
        )
        
        # Setup schema
        if not indexer.setup_weaviate_schema():
            print("❌ Schema setup failed")
            return
        
        # Download and index docs
        docs = indexer.download_godot_class_docs()
        if not docs:
            print("❌ No documentation downloaded")
            return
        
        if not indexer.index_docs_to_weaviate(docs):
            print("❌ Indexing failed")
            return
        
        # Test search
        indexer.test_search_modes()
        
        print(f"\n🎉 Setup complete!")
        print(f"   📚 {len(docs)} documentation chunks indexed")
        print(f"   🔍 Multiple search modes available")
        print(f"   🤖 Agent can now access Godot expertise")
        
        # Test the enhanced search
        print(f"\n🧪 Testing enhanced search integration...")
        from enhanced_docs_search import EnhancedGodotDocsSearch
        
        searcher = EnhancedGodotDocsSearch(
            os.getenv('WEAVIATE_URL'),
            os.getenv('WEAVIATE_API_KEY')
        )
        
        # Test FPS-specific query
        result = searcher.search_godot_docs_enhanced(
            query="CharacterBody3D FPS movement setup",
            mode="hybrid",
            code_examples_only=True,
            max_results=3
        )
        
        if result.get('success'):
            print(f"   ✅ Enhanced search works! Found {len(result.get('results', []))} results")
            for doc in result.get('results', [])[:2]:
                print(f"      📄 {doc.get('title', 'Unknown')}")
        else:
            print(f"   ❌ Enhanced search test failed: {result.get('error')}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure godot_docs_indexer.py is in the same directory")
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()


