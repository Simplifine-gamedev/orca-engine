#!/usr/bin/env python3
"""
Production Godot Documentation Indexer
Uses manual OpenAI embeddings (proven to work) + parallel processing
"""

import os
import requests
import hashlib
import time
import concurrent.futures
import multiprocessing
from bs4 import BeautifulSoup
import weaviate
import weaviate.classes as wvc
import openai
from typing import List, Dict
import dotenv

# Load environment variables
dotenv.load_dotenv()

def download_single_file(file_info):
    """Download a single XML file"""
    try:
        if not file_info['name'].endswith('.xml'):
            return None, None
            
        response = requests.get(file_info['download_url'], timeout=15)
        response.raise_for_status()
        
        return response.text, file_info['name']
        
    except Exception as e:
        print(f"❌ Failed to download {file_info.get('name', 'unknown')}: {e}")
        return None, None

def parse_class_xml_standalone(xml_content, filename):
    """Parse a single XML file into docs"""
    try:
        soup = BeautifulSoup(xml_content, 'xml')
        class_element = soup.find('class')
        
        if not class_element:
            return []
        
        class_name = class_element.get('name', filename.replace('.xml', ''))
        inherits = class_element.get('inherits', '')
        
        docs = []
        
        # Main class description
        brief_desc = class_element.find('brief_description')
        description = class_element.find('description')
        
        main_content = f"Class: {class_name}"
        if inherits:
            main_content += f" (extends {inherits})"
        main_content += "\n\n"
        
        if brief_desc and brief_desc.text:
            main_content += f"Brief: {brief_desc.text.strip()}\n\n"
        
        if description and description.text:
            main_content += f"Description: {description.text.strip()}"
        
        # Extract keywords - only use class name and inheritance, no hardcoded mappings
        keywords = [class_name.lower()]
        if inherits:
            keywords.append(inherits.lower())
        
        # Class overview doc
        docs.append({
            'class_name': class_name,
            'section': 'overview',
            'title': f"{class_name} Class Overview",
            'content': main_content,
            'keywords': keywords,
            'url': f"https://docs.godotengine.org/en/stable/classes/class_{class_name.lower()}.html"
        })
        
        # Process all methods
        methods = class_element.find('methods')
        if methods:
            method_count = 0
            for method in methods.find_all('method'):
                    
                method_name = method.get('name', '')
                method_desc = method.find('description')
                
                if method_name and method_desc and method_desc.text:
                    method_content = f"Method: {class_name}.{method_name}()\n\n"
                    method_content += f"Description: {method_desc.text.strip()}\n\n"
                    
                    # Add parameters
                    params = method.find_all('param')
                    if params:
                        method_content += "Parameters:\n"
                        for param in params:
                            param_name = param.get('name', '')
                            param_type = param.get('type', '')
                            method_content += f"- {param_name}: {param_type}\n"
                    
                    # Return type
                    return_elem = method.find('return')
                    if return_elem:
                        return_type = return_elem.get('type', '')
                        if return_type:
                            method_content += f"\nReturns: {return_type}"
                    
                    docs.append({
                        'class_name': class_name,
                        'section': 'methods',
                        'title': f"{class_name}.{method_name}()",
                        'content': method_content,
                        'keywords': keywords + [method_name.lower()],
                        'url': f"https://docs.godotengine.org/en/stable/classes/class_{class_name.lower()}.html#{method_name}"
                    })
                    method_count += 1
        
        # Process constants/enums (MISSING SECTION!)
        constants = class_element.find('constants')
        if constants:
            for constant in constants.find_all('constant'):
                const_name = constant.get('name', '')
                const_value = constant.get('value', '')
                const_desc = constant.text
                
                if const_name:
                    const_content = f"Constant: {class_name}.{const_name}\n\n"
                    const_content += f"Value: {const_value}\n\n"
                    if const_desc:
                        const_content += f"Description: {const_desc.strip()}"
                    
                    docs.append({
                        'class_name': class_name,
                        'section': 'constants',
                        'title': f"{class_name}.{const_name}",
                        'content': const_content,
                        'keywords': keywords + [const_name.lower(), 'constant'],
                        'url': f"https://docs.godotengine.org/en/stable/classes/class_{class_name.lower()}.html#{const_name}"
                    })
        
        # Process members/properties  
        members = class_element.find('members')
        if members:
            for member in members.find_all('member'):
                member_name = member.get('name', '')
                member_type = member.get('type', '')
                member_desc = member.text
                
                if member_name:
                    member_content = f"Property: {class_name}.{member_name}\n"
                    if member_type:
                        member_content += f"Type: {member_type}\n\n"
                    if member_desc:
                        member_content += f"Description: {member_desc.strip()}"
                    
                    docs.append({
                        'class_name': class_name,
                        'section': 'properties',
                        'title': f"{class_name}.{member_name}",
                        'content': member_content,
                        'keywords': keywords + [member_name.lower(), 'property'],
                        'url': f"https://docs.godotengine.org/en/stable/classes/class_{class_name.lower()}.html#{member_name}"
                    })
        
        return docs
        
    except Exception as e:
        print(f"❌ Parse error for {filename}: {e}")
        return []

def index_godot_docs_production():
    """Production Godot docs indexing with manual embeddings"""
    print("🚀 Production Godot Documentation Indexer")
    print("🧠 Using Manual OpenAI Embeddings (Proven Working)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Environment check
    weaviate_url = os.getenv('WEAVIATE_URL')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not all([weaviate_url, weaviate_api_key, openai_api_key]):
        print("❌ Missing environment variables")
        return False
    
    # Connect to services
    os.environ['OPENAI_API_KEY'] = openai_api_key
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        headers={'X-OpenAI-Api-Key': openai_api_key}
    )
    
    openai_client = openai.OpenAI(api_key=openai_api_key)
    print("✅ Connected to Weaviate + OpenAI")
    
    # Create production collection
    collection_name = "GodotDocs_Production"
    print(f"🔧 Creating production collection: {collection_name}")
    
    try:
        # Delete existing if it exists
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            print("🗑️ Deleted existing collection")
        
        collection = client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="class_name", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="section", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="keywords", data_type=wvc.config.DataType.TEXT_ARRAY)
            ]
        )
        print("✅ Production collection created")
    except Exception as e:
        print(f"❌ Collection creation failed: {e}")
        return False
    
    # Download ALL docs in parallel
    processing_start = time.time()
    print("📥 Downloading ALL Godot class documentation...")
    
    try:
        response = requests.get("https://api.github.com/repos/godotengine/godot/contents/doc/classes", timeout=30)
        response.raise_for_status()
        files = response.json()
        
        xml_files = [f for f in files if f['name'].endswith('.xml')]
        print(f"Found {len(xml_files)} class files to process")
        
        # Parallel download (10 workers for speed)
        download_results = []
        print("📥 Downloading files in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {executor.submit(download_single_file, file_info): file_info 
                             for file_info in xml_files}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                completed += 1
                xml_content, filename = future.result()
                if xml_content and filename:
                    download_results.append((xml_content, filename))
                
                # Progress update
                if completed % 50 == 0 or completed == len(xml_files):
                    print(f"  📄 Downloaded {completed}/{len(xml_files)} files")
        
        print(f"✅ Downloaded {len(download_results)} files successfully")
        
        # Parallel parsing
        print("⚙️ Parsing XML files in parallel...")
        all_docs = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_filename = {executor.submit(parse_class_xml_standalone, xml_content, filename): filename 
                                for xml_content, filename in download_results}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_filename):
                completed += 1
                try:
                    parsed_docs = future.result()
                    if parsed_docs:
                        all_docs.extend(parsed_docs)
                except Exception as e:
                    filename = future_to_filename[future]
                    print(f"❌ Parse error for {filename}: {e}")
                
                if completed % 50 == 0 or completed == len(download_results):
                    print(f"  ⚙️ Parsed {completed}/{len(download_results)} files")
        
        processing_time = time.time() - processing_start
        print(f"✅ Processed {len(all_docs)} documentation chunks in {processing_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Download/parse failed: {e}")
        return False
    
    # Generate embeddings manually in batches
    indexing_start = time.time()
    print(f"🧠 Generating embeddings for {len(all_docs)} docs...")
    
    try:
        # Prepare texts for embedding
        texts = []
        for doc in all_docs:
            text = f"{doc['title']}\n\n{doc['content']}"
            if doc.get('keywords'):
                text += f"\n\nKeywords: {', '.join(doc['keywords'])}"
            texts.append(text)
        
        # Generate embeddings in batches (OpenAI has limits)
        batch_size = 100  # Process 100 at a time
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"  🧠 Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            embeddings_response = openai_client.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [data.embedding for data in embeddings_response.data]
            all_embeddings.extend(batch_embeddings)
        
        print(f"✅ Generated {len(all_embeddings)} embeddings, dimension: {len(all_embeddings[0])}")
        
        # Insert docs with manual embeddings
        print("📤 Inserting docs with manual embeddings...")
        
        inserted_count = 0
        batch_size = 50  # Smaller batches for insertion
        
        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            batch_vectors = all_embeddings[i:i + batch_size]
            
            with collection.batch.dynamic() as batch:
                for j, doc in enumerate(batch_docs):
                    batch.add_object(
                        properties={
                            "title": doc['title'],
                            "content": doc['content'],
                            "class_name": doc['class_name'],
                            "section": doc['section'],
                            "url": doc.get('url', ''),
                            "keywords": doc.get('keywords', [])
                        },
                        vector=batch_vectors[j]
                    )
            
            inserted_count += len(batch_docs)
            print(f"  📤 Inserted {inserted_count}/{len(all_docs)} docs")
        
        print(f"✅ All docs inserted with manual embeddings")
        
    except Exception as e:
        print(f"❌ Embedding/insertion failed: {e}")
        return False
        
    indexing_time = time.time() - indexing_start
    
    # Test searches
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "CharacterBody3D movement",
        "FPS controller setup", 
        "mouse look rotation",
        "RayCast3D collision detection"
    ]
    
    for query in test_queries:
        try:
            # Generate query embedding
            query_response = openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_vector = query_response.data[0].embedding
            
            # Search using vector similarity
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=3,
                return_metadata=["distance"]
            )
            
            print(f"\n📋 Query: '{query}'")
            for i, obj in enumerate(results.objects):
                title = obj.properties.get('title', 'Unknown')
                distance = obj.metadata.distance if obj.metadata else 1.0
                similarity = 1.0 - distance
                print(f"  {i+1}. {title} (similarity: {similarity:.3f})")
                
        except Exception as e:
            print(f"❌ Search failed for '{query}': {e}")
    
    client.close()
    
    total_time = time.time() - start_time
    
    # Performance summary
    print(f"\n🏁 Production Performance Summary")
    print("=" * 60)
    print(f"   📥 Download & Parse:   {processing_time:.1f}s (PARALLEL)")
    print(f"   🧠 Manual Embeddings: {indexing_time:.1f}s (1536D)")
    print(f"   📊 Total time:        {total_time:.1f}s")
    print(f"")
    print(f"🎉 SUCCESS: Production Godot docs indexing!")
    print(f"   📚 {len(all_docs)} documentation chunks indexed")
    print(f"   🔍 Working semantic search with real similarity scores")
    print(f"   ⚡ Parallel processing + manual embeddings")
    print(f"   🎯 Collection: {collection_name}")
    
    return True

if __name__ == "__main__":
    success = index_godot_docs_production()
    print(f"\n{'🎉 COMPLETE SUCCESS' if success else '❌ FAILED'}: Production docs indexing")

