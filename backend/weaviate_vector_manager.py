"""
Optimized Weaviate Vector Manager with parallelization and efficient querying
"""
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.config import Config, ConnectionConfig
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Filter
import hashlib
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import numpy as np
from datetime import datetime, timezone

class WeaviateVectorManager:
    """High-performance Weaviate vector manager with parallel processing"""
    
    def __init__(self, weaviate_url: str, api_key: str, openai_client):
        self.openai_client = openai_client
        self.weaviate_url = weaviate_url
        
        # Initialize Weaviate client - simpler connection first
        try:
            # Try the newer simplified connection method
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=AuthApiKey(api_key)
            )
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}")
            # Try alternative connection method
            self.client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=AuthApiKey(api_key)
            )
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.embedding_semaphore = threading.Semaphore(5)  # Limit concurrent embeddings
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize collections
        self._init_collections()
        
    def _init_collections(self):
        """Initialize Weaviate collections with optimized settings"""
        collections = self.client.collections
        
        # Main embeddings collection
        if not collections.exists("ProjectEmbedding"):
            collections.create(
                name="ProjectEmbedding",
                properties=[
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="project_id", data_type=DataType.TEXT),
                    Property(name="file_path", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="content_hash", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="chunk_start", data_type=DataType.INT),
                    Property(name="chunk_end", data_type=DataType.INT),
                    Property(name="file_type", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own embeddings
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=256,  # Higher for better recall
                    ef=128,  # Higher for better search quality
                    max_connections=32,  # More connections for better search
                ),
                inverted_index_config=Configure.inverted_index(
                    index_null_state=True,
                    index_property_length=True,
                    index_timestamps=True,
                )
            )
            
        # Graph relationships collection
        if not collections.exists("ProjectGraph"):
            collections.create(
                name="ProjectGraph",
                properties=[
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="project_id", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                    Property(name="target_file", data_type=DataType.TEXT),
                    Property(name="relationship_type", data_type=DataType.TEXT),
                    Property(name="weight", data_type=DataType.NUMBER),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in parallel batches"""
        if not texts:
            return []
        
        # OpenAI has a limit of ~8000 tokens per batch
        BATCH_SIZE = 50
        MAX_RETRIES = 3
        
        all_embeddings = []
        
        def embed_batch(batch_texts):
            """Embed a single batch with retry logic"""
            with self.embedding_semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        # Truncate long texts
                        truncated = [t[:8000] if len(t) > 8000 else t for t in batch_texts]
                        response = self.openai_client.embeddings.create(
                            model="text-embedding-3-small",
                            input=truncated
                        )
                        return [data.embedding for data in response.data]
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            print(f"Embedding batch failed after {MAX_RETRIES} attempts: {e}")
                            return [None] * len(batch_texts)
        
        # Process batches in parallel
        futures = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            future = self.executor.submit(embed_batch, batch)
            futures.append((i, future))
        
        # Collect results in order
        results = [None] * len(texts)
        for start_idx, future in futures:
            batch_embeddings = future.result()
            for j, embedding in enumerate(batch_embeddings):
                if embedding is not None:
                    results[start_idx + j] = embedding
        
        # Filter out None values
        return [e for e in results if e is not None]
    
    def index_files_with_content(self, files: List[Dict[str, Any]], user_id: str, 
                                 project_id: str, max_workers: int = 10) -> Dict[str, Any]:
        """Index multiple files in parallel with provided content"""
        start_time = time.time()
        stats = {"indexed": 0, "skipped": 0, "failed": 0, "chunks": 0}
        
        # Prepare all chunks for batch processing
        all_chunks = []
        chunk_to_file_map = []
        
        for file_data in files:
            # Handle both 'path' and 'file_path' keys for compatibility
            raw_file_path = file_data.get('path') or file_data.get('file_path', '')
            content = file_data.get('content', '')
            content_hash = file_data.get('hash', '')
            
            # NORMALIZE FILE PATHS: Always strip res:// prefix for consistency
            file_path = raw_file_path
            if file_path.startswith('res://'):
                file_path = file_path[6:]  # Remove 'res://' prefix
            
            print(f"WeaviateVector: Normalized path '{raw_file_path}' -> '{file_path}'")
            
            if not content:
                stats["skipped"] += 1
                continue
            
            # Check if file has changed (skip if unchanged)
            if content_hash:
                if self._is_file_unchanged(user_id, project_id, raw_file_path, content_hash):
                    print(f"WeaviateVector: Skipping unchanged file: {file_path}")
                    stats["skipped"] += 1
                    continue
            
            # CRITICAL: Remove existing chunks for this file to prevent duplicates!
            try:
                print(f"WeaviateVector: Cleaning old chunks for {file_path}")
                self._remove_existing_file_chunks(user_id, project_id, raw_file_path)
            except Exception as e:
                print(f"WeaviateVector: Warning - couldn't clean old chunks for {file_path}: {e}")
            
            # Generate chunks
            chunks = self._chunk_code_file(content, file_path, max_lines=150)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_file_map.append(file_path)
        
        if not all_chunks:
            return stats
        
        # Generate embeddings in parallel
        print(f"WeaviateVector: Generating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk['content'] for chunk in all_chunks]
        embeddings = self._generate_embeddings_batch(texts)
        
        # Batch insert into Weaviate
        print(f"WeaviateVector: Inserting {len(all_chunks)} chunks into Weaviate...")
        collection = self.client.collections.get("ProjectEmbedding")
        
        with collection.batch.dynamic() as batch:
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                if embedding is None:
                    stats["failed"] += 1
                    continue
                
                properties = {
                    "user_id": user_id,
                    "project_id": project_id,
                    "file_path": chunk['file_path'],
                    "content": chunk['content'],
                    "content_hash": chunk['content_hash'],
                    "chunk_index": chunk['chunk_index'],
                    "chunk_start": chunk['start_line'],
                    "chunk_end": chunk['end_line'],
                    "file_type": chunk['file_type'],
                    "timestamp": datetime.now(timezone.utc),
                }
                
                batch.add_object(
                    properties=properties,
                    vector=embedding
                )
                stats["chunks"] += 1
        
        # Update file counts
        unique_files = set(chunk_to_file_map)
        stats["indexed"] = len(unique_files)
        
        elapsed = time.time() - start_time
        print(f"WeaviateVector: Indexed {stats['chunks']} chunks from {stats['indexed']} files in {elapsed:.2f}s")
        
        # Update graph relationships in background
        self.executor.submit(self._update_graph_relationships, files, user_id, project_id)
        
        return stats
    
    def search(self, query: str, user_id: str, project_id: str, 
               max_results: int = 5) -> List[Dict[str, Any]]:
        """Fast semantic search with caching"""
        # Check cache
        cache_key = f"{user_id}:{project_id}:{query}:{max_results}"
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Generate query embedding
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = response.data[0].embedding
        
        # Search with filters
        collection = self.client.collections.get("ProjectEmbedding")
        
        try:
            # Try the new API syntax
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=max_results * 2,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["file_path", "content", "chunk_index", 
                                 "chunk_start", "chunk_end", "file_type"]
            ).where(
                Filter.by_property("user_id").equal(user_id) & 
                Filter.by_property("project_id").equal(project_id)
            ).do()
        except Exception as e:
            # Fallback to alternative syntax or fetch all and filter
            print(f"Query method failed: {e}, trying alternative approach")
            # Simple approach: fetch without filter and filter in Python
            try:
                results = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=max_results * 10,  # Get more to filter later
                    return_metadata=MetadataQuery(distance=True),
                    return_properties=["user_id", "project_id", "file_path", "content", 
                                     "chunk_index", "chunk_start", "chunk_end", "file_type"]
                )
                print(f"WeaviateSearch: Fetched {len(results.objects)} objects before filtering")
                
                # Filter results in Python
                filtered_objects = []
                for obj in results.objects:
                    if (obj.properties.get('user_id') == user_id and 
                        obj.properties.get('project_id') == project_id):
                        filtered_objects.append(obj)
                        
                print(f"WeaviateSearch: {len(filtered_objects)} objects after filtering for user {user_id}, project {project_id}")
                results.objects = filtered_objects[:max_results * 2]
                
            except Exception as e2:
                print(f"WeaviateSearch: Alternative approach also failed: {e2}")
                # Return empty results
                class EmptyResults:
                    def __init__(self):
                        self.objects = []
                results = EmptyResults()
        
        # Format results
        formatted_results = []
        seen_files = set()
        
        for obj in results.objects:
            file_path = obj.properties['file_path']
            
            # FIXED: Skip if we already have ANY result from this file (proper deduplication)
            if file_path in seen_files:
                continue
            
            seen_files.add(file_path)
            
            formatted_results.append({
                'file_path': file_path,
                'content': obj.properties['content'],
                'similarity': 1 - obj.metadata.distance,  # Convert distance to similarity
                'chunk': {
                    'chunk_index': obj.properties['chunk_index'],
                    'start_line': obj.properties['chunk_start'],
                    'end_line': obj.properties['chunk_end'],
                    'content': obj.properties['content'],
                },
                'file_type': obj.properties['file_type'],
            })
            
            if len(formatted_results) >= max_results:
                break
        
        # Cache results
        with self.cache_lock:
            self.cache[cache_key] = formatted_results
            # Limit cache size
            if len(self.cache) > 1000:
                self.cache.clear()
        
        return formatted_results
    
    def _chunk_code_file(self, content: str, file_path: str, max_lines: int) -> List[Dict]:
        """Chunk code files intelligently"""
        lines = content.split('\n')
        chunks = []
        
        # Detect file type for smart chunking
        ext = os.path.splitext(file_path)[1].lower()
        
        # For code files, try to chunk by functions/classes
        if ext in ['.py', '.gd', '.cs', '.cpp', '.js', '.ts']:
            chunks = self._smart_chunk_code(lines, max_lines)
        else:
            # Simple line-based chunking for other files
            for i in range(0, len(lines), max_lines // 2):
                chunk_lines = lines[i:i + max_lines]
                if chunk_lines:
                    chunks.append({
                        'start_line': i + 1,
                        'end_line': i + len(chunk_lines),
                        'content': '\n'.join(chunk_lines)
                    })
        
        # Add metadata to chunks
        for idx, chunk in enumerate(chunks):
            chunk['file_path'] = file_path
            chunk['chunk_index'] = idx
            chunk['content_hash'] = hashlib.md5(chunk['content'].encode()).hexdigest()
            chunk['file_type'] = ext[1:] if ext else 'unknown'
        
        return chunks
    
    def _smart_chunk_code(self, lines: List[str], max_lines: int) -> List[Dict]:
        """Smart chunking that respects code boundaries"""
        chunks = []
        current_chunk = []
        current_start = 1
        indent_level = 0
        
        for i, line in enumerate(lines):
            # Detect new top-level definitions
            stripped = line.lstrip()
            if stripped.startswith(('def ', 'class ', 'func ', 'function ', 'public ', 'private ')):
                # Start new chunk if current is getting large
                if len(current_chunk) > max_lines // 2:
                    if current_chunk:
                        chunks.append({
                            'start_line': current_start,
                            'end_line': current_start + len(current_chunk) - 1,
                            'content': '\n'.join(current_chunk)
                        })
                    current_chunk = [line]
                    current_start = i + 1
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
                
                # Force new chunk if too large
                if len(current_chunk) >= max_lines:
                    chunks.append({
                        'start_line': current_start,
                        'end_line': current_start + len(current_chunk) - 1,
                        'content': '\n'.join(current_chunk)
                    })
                    current_chunk = []
                    current_start = i + 2
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'start_line': current_start,
                'end_line': current_start + len(current_chunk) - 1,
                'content': '\n'.join(current_chunk)
            })
        
        return chunks
    
    def _update_graph_relationships(self, files: List[Dict], user_id: str, project_id: str):
        """Update graph relationships in background"""
        try:
            # Extract relationships from files
            relationships = []
            for file_data in files:
                # Handle both 'path' and 'file_path' keys
                file_path = file_data.get('path') or file_data.get('file_path', '')
                content = file_data.get('content', '')
                
                if not file_path or not content:
                    continue
                
                # Extract imports/references
                refs = self._extract_references(content, file_path)
                for ref in refs:
                    relationships.append({
                        'source_file': file_path,
                        'target_file': ref['target'],
                        'relationship_type': ref['type'],
                        'weight': ref.get('weight', 1.0)
                    })
            
            # Batch insert relationships
            if relationships:
                collection = self.client.collections.get("ProjectGraph")
                with collection.batch.dynamic() as batch:
                    for rel in relationships:
                        batch.add_object(properties={
                            'user_id': user_id,
                            'project_id': project_id,
                            'source_file': rel['source_file'],
                            'target_file': rel['target_file'],
                            'relationship_type': rel['relationship_type'],
                            'weight': rel['weight']
                        })
                print(f"WeaviateVector: Stored {len(relationships)} graph relationships")
                
        except Exception as e:
            print(f"WeaviateVector: Error updating graph: {e}")
    
    def _extract_references(self, content: str, file_path: str) -> List[Dict]:
        """Extract file references from code"""
        refs = []
        lines = content.split('\n')
        ext = os.path.splitext(file_path)[1].lower()
        
        for line in lines:
            stripped = line.strip()
            
            # Python imports
            if ext in ['.py'] and stripped.startswith(('import ', 'from ')):
                parts = stripped.split()
                if len(parts) > 1:
                    if stripped.startswith('from'):
                        # from module import ...
                        module = parts[1].split('.')[0]
                    else:
                        # import module
                        module = parts[1].split('.')[0].split(' as ')[0]
                    refs.append({
                        'target': f"{module}.py",
                        'type': 'import',
                        'weight': 1.0
                    })
            
            # GDScript patterns
            elif ext in ['.gd']:
                import re
                
                # preload() for resources
                preload_matches = re.findall(r'preload\(["\'](.*?)["\']\)', line)
                for match in preload_matches:
                    target = match.replace('res://', '')
                    refs.append({
                        'target': target,
                        'type': 'preload',
                        'weight': 1.0
                    })
                
                # load() for dynamic loading
                load_matches = re.findall(r'load\(["\'](.*?)["\']\)', line)
                for match in load_matches:
                    target = match.replace('res://', '')
                    refs.append({
                        'target': target,
                        'type': 'load',
                        'weight': 0.8
                    })
                
                # extends references
                if stripped.startswith('extends'):
                    # extends "res://path/to/script.gd"
                    string_match = re.search(r'extends\s+["\'](.*?)["\']', stripped)
                    if string_match:
                        target = string_match.group(1).replace('res://', '')
                        refs.append({
                            'target': target,
                            'type': 'extends',
                            'weight': 1.5
                        })
                
                # SIGNAL DETECTION - Critical for HP system tracing!
                
                # Signal definitions: signal health_changed(new_health) 
                signal_def_match = re.search(r'signal\s+(\w+)', stripped)
                if signal_def_match:
                    signal_name = signal_def_match.group(1)
                    refs.append({
                        'target': f"SIGNAL_DEF:{signal_name}",
                        'type': 'defines_signal',
                        'weight': 1.2,
                        'signal_name': signal_name
                    })
                
                # Signal connections: node.connect("signal_name", target, "method")
                connect_matches = re.findall(r'\.connect\(\s*["\'](\w+)["\'],\s*([^,)]+)', line)
                for signal_name, target_node in connect_matches:
                    refs.append({
                        'target': f"SIGNAL_CONNECT:{signal_name}",
                        'type': 'connects_signal',
                        'weight': 1.3,
                        'signal_name': signal_name,
                        'target_node': target_node.strip()
                    })
                
                # Signal emissions: emit_signal("signal_name", args...)
                emit_matches = re.findall(r'emit_signal\(\s*["\'](\w+)["\']', line)
                for signal_name in emit_matches:
                    refs.append({
                        'target': f"SIGNAL_EMIT:{signal_name}",
                        'type': 'emits_signal',
                        'weight': 1.1,
                        'signal_name': signal_name
                    })
                
                # get_node references: get_node("NodePath")
                get_node_matches = re.findall(r'get_node\(\s*["\']([^"\']+)["\']\s*\)', line)
                for node_path in get_node_matches:
                    refs.append({
                        'target': f"NODE_REF:{node_path}",
                        'type': 'references_node',
                        'weight': 0.9,
                        'node_path': node_path
                    })
                
                # @onready var references: @onready var player = get_node("Player")
                onready_matches = re.findall(r'@onready\s+var\s+\w+\s*=\s*get_node\(\s*["\']([^"\']+)["\']\s*\)', line)
                for node_path in onready_matches:
                    refs.append({
                        'target': f"ONREADY_REF:{node_path}",
                        'type': 'onready_reference',
                        'weight': 1.0,
                        'node_path': node_path
                    })
            
            # C# using statements
            elif ext in ['.cs'] and stripped.startswith('using '):
                # Skip system namespaces
                if not any(stripped.startswith(f'using {ns}') for ns in ['System', 'Godot']):
                    namespace = stripped.replace('using ', '').rstrip(';').strip()
                    refs.append({
                        'target': f"{namespace}.cs",
                        'type': 'using',
                        'weight': 1.0
                    })
            
            # Scene file references in any file
            if '.tscn' in line or '.scn' in line:
                import re
                scene_matches = re.findall(r'["\'](.*?\.t?scn)["\']', line)
                for match in scene_matches:
                    target = match.replace('res://', '')
                    refs.append({
                        'target': target,
                        'type': 'scene_ref',
                        'weight': 0.9
                    })
        
        # SCENE FILE (.tscn) SIGNAL DETECTION - Critical for HP system!
        if ext == '.tscn':
            import re
            
            # Parse signal connections in scene files
            # [connection signal="health_changed" from="Player" to="UI" method="_on_player_health_changed"]
            connection_matches = re.finditer(
                r'\[connection signal="([^"]+)" from="([^"]*)" to="([^"]*)" method="([^"]*)"\]', 
                content, re.MULTILINE
            )
            
            for match in connection_matches:
                signal_name, from_node, to_node, method = match.groups()
                
                # Create signal flow relationships  
                refs.append({
                    'target': f"SCENE_SIGNAL:{signal_name}:{from_node}:{to_node}",
                    'type': 'scene_signal_connection',
                    'weight': 1.4,  # High weight - signal connections are architecturally important
                    'signal_name': signal_name,
                    'from_node': from_node,
                    'to_node': to_node,
                    'method': method
                })
            
            # Parse external script attachments: script = ExtResource("id") 
            script_matches = re.finditer(r'script = ExtResource\(\s*["\']([^"\']+)["\']\s*\)', content)
            for match in script_matches:
                # Look for the corresponding ext_resource path
                ext_resource_match = re.search(
                    rf'\[ext_resource[^]]*id="?{re.escape(match.group(1))}"?[^]]*path="([^"]+)"[^]]*type="Script"',
                    content
                )
                if ext_resource_match:
                    script_path = ext_resource_match.group(1).replace('res://', '')
                    refs.append({
                        'target': script_path,
                        'type': 'scene_script_attachment',
                        'weight': 1.6,  # Very important for scene-script relationships
                    })
        
        # Deduplicate references
        seen = set()
        unique_refs = []
        for ref in refs:
            key = (ref['target'], ref['type'])
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        return unique_refs
    
    def _remove_existing_file_chunks(self, user_id: str, project_id: str, raw_file_path: str):
        """Remove all existing chunks for a file to prevent duplicates during re-indexing"""
        try:
            # Normalize path for consistency
            file_path = raw_file_path[6:] if raw_file_path.startswith('res://') else raw_file_path
            collection = self.client.collections.get("ProjectEmbedding")
            
            # Try to delete objects with filters (may not work with older Weaviate clients)
            try:
                collection.data.delete_many(
                    where=Filter.by_property("user_id").equal(user_id) & 
                          Filter.by_property("project_id").equal(project_id) &
                          Filter.by_property("file_path").equal(file_path)
                )
                print(f"WeaviateVector: Deleted old chunks for {file_path} using filter")
                return
            except Exception as filter_error:
                print(f"WeaviateVector: Filter delete failed: {filter_error}, trying fetch+delete")
            
            # Fallback: Fetch objects and delete individually
            try:
                results = collection.query.fetch_objects(
                    limit=1000,  # Should be enough for one file's chunks
                    return_properties=["user_id", "project_id", "file_path"]
                )
                
                to_delete = []
                for obj in results.objects:
                    if (obj.properties.get('user_id') == user_id and 
                        obj.properties.get('project_id') == project_id and
                        obj.properties.get('file_path') == file_path):
                        to_delete.append(obj.uuid)
                
                if to_delete:
                    for uuid in to_delete:
                        collection.data.delete_by_id(uuid)
                    print(f"WeaviateVector: Deleted {len(to_delete)} old chunks for {file_path}")
                        
            except Exception as fetch_error:
                print(f"WeaviateVector: Fetch+delete failed: {fetch_error}")
                
        except Exception as e:
            print(f"WeaviateVector: Failed to remove existing chunks for {file_path}: {e}")
            # Don't raise - we want indexing to continue even if cleanup fails
    
    def _is_file_unchanged(self, user_id: str, project_id: str, raw_file_path: str, content_hash: str) -> bool:
        """Check if file is unchanged by comparing content hash with existing chunks"""
        try:
            # Normalize path for consistency
            file_path = raw_file_path[6:] if raw_file_path.startswith('res://') else raw_file_path
            collection = self.client.collections.get("ProjectEmbedding")
            
            # Try newer API first, fallback to older API
            try:
                results = collection.query.fetch_objects(
                    limit=1,  # Just need to find one matching chunk
                    return_properties=["content_hash"],
                    where=Filter.by_property("user_id").equal(user_id) & 
                          Filter.by_property("project_id").equal(project_id) &
                          Filter.by_property("file_path").equal(file_path)
                )
            except Exception:
                # Fallback: fetch all objects and filter manually (slower but compatible)
                results = collection.query.fetch_objects(
                    limit=100,  # Get more objects for manual filtering
                    return_properties=["user_id", "project_id", "file_path", "content_hash"]
                )
            
            # If we find any chunk with the same hash, file is unchanged
            for obj in results.objects:
                # Check if this object matches our criteria
                if (obj.properties.get('user_id') == user_id and 
                    obj.properties.get('project_id') == project_id and
                    obj.properties.get('file_path') == file_path and
                    obj.properties.get('content_hash') == content_hash):
                    return True
            
            return False
            
        except Exception as e:
            print(f"WeaviateVector: Hash check disabled due to API incompatibility: {e}")
            return False  # If we can't check, assume it changed (safer for correctness)
    
    def index_project(self, project_root: str, user_id: str, project_id: str, 
                     force_reindex: bool = False, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Index all files in a project directory
        
        NOTE: This method is NOT used in cloud deployments. The frontend sends
        file content directly via the index_files action in /embed endpoint.
        This is kept for local testing only.
        """
        import os
        
        # Check if we're in a cloud environment (no local file access)
        if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('GAE_ENV', '').startswith('standard'):
            return {
                'success': False,
                'error': 'index_project not supported in cloud deployment. Use index_files with content instead.',
                'indexed': 0,
                'skipped': 0,
                'failed': 0,
                'total': 0
            }
        
        stats = {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}
        all_files = []
        
        # Walk through project directory
        for root, dirs, files in os.walk(project_root):
            # Skip hidden directories and common build folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                      {'node_modules', '__pycache__', 'build', 'dist', '.godot'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-indexable files
                if not self._should_index_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Create file data structure
                    file_data = {
                        'path': file_path,
                        'content': content,
                        'hash': hashlib.md5(content.encode()).hexdigest()
                    }
                    all_files.append(file_data)
                    stats["total"] += 1
                    
                except Exception as e:
                    print(f"WeaviateVector: Error reading {file_path}: {e}")
                    stats["failed"] += 1
        
        # Index files in batches
        if all_files:
            batch_stats = self.index_files_with_content(all_files, user_id, project_id, max_workers)
            stats.update(batch_stats)
        
        return stats
    
    def index_file(self, file_path: str, user_id: str, project_id: str, 
                   project_root: str) -> bool:
        """Index a single file
        
        NOTE: This method is NOT used in cloud deployments. The frontend sends
        file content directly via the index_files action in /embed endpoint.
        This is kept for local testing only.
        """
        import os
        
        # Check if we're in a cloud environment (no local file access)
        if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('GAE_ENV', '').startswith('standard'):
            print("WeaviateVector: index_file not supported in cloud deployment")
            return False
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_data = {
                'path': file_path,
                'content': content,
                'hash': hashlib.md5(content.encode()).hexdigest()
            }
            
            # Index using batch method
            stats = self.index_files_with_content([file_data], user_id, project_id)
            return stats.get('indexed', 0) > 0
            
        except Exception as e:
            print(f"WeaviateVector: Error indexing file {file_path}: {e}")
            return False
    
    def _should_index_file(self, file_path: str) -> bool:
        """Check if file should be indexed"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # Skip binary and system files
        skip_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.o', '.a',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.pyc', '.pyo', '.class', '.uid'
        }
        
        if ext in skip_extensions:
            return False
            
        # Skip if filename starts with dot
        filename = os.path.basename(file_path)
        if filename.startswith('.'):
            return False
            
        return True
    
    def get_stats(self, user_id: str, project_id: str) -> Dict[str, Any]:
        """Get project statistics"""
        collection = self.client.collections.get("ProjectEmbedding")
        
        try:
            # Try to get count with filter
            # Note: This might not work with current Weaviate client version
            # so we'll use a fallback approach
            results = collection.query.fetch_objects(
                limit=0,  # We just want the count
                return_properties=["user_id", "project_id"]
            )
            # Count manually - not ideal but works
            count = 0
            # Do a sample query to estimate total
            sample = collection.query.fetch_objects(
                limit=1000,
                return_properties=["user_id", "project_id"]
            )
            for obj in sample.objects:
                if (obj.properties.get('user_id') == user_id and 
                    obj.properties.get('project_id') == project_id):
                    count += 1
            
            # Rough estimate if we hit the limit
            if len(sample.objects) == 1000:
                count = f"~{count}+"
            
        except Exception as e:
            print(f"Stats query failed: {e}")
            count = "unknown"
        
        return {
            'total_chunks': count,
            'status': 'connected',
            'backend': 'weaviate'
        }
    
    def clear_project(self, user_id: str, project_id: str):
        """Clear project data"""
        try:
            # Delete from embeddings collection
            collection = self.client.collections.get("ProjectEmbedding")
            
            # Fetch and delete objects manually since filtered delete might not work
            to_delete = []
            batch_size = 100
            offset = 0
            
            while True:
                objects = collection.query.fetch_objects(
                    limit=batch_size,
                    offset=offset,
                    return_properties=["user_id", "project_id"]
                )
                
                if not objects.objects:
                    break
                    
                for obj in objects.objects:
                    if (obj.properties.get('user_id') == user_id and 
                        obj.properties.get('project_id') == project_id):
                        to_delete.append(obj.uuid)
                
                offset += batch_size
                
                # Delete in batches
                if len(to_delete) >= 50:
                    collection.data.delete_many(where=Filter.by_id().contains_any(to_delete))
                    to_delete = []
            
            # Delete remaining
            if to_delete:
                collection.data.delete_many(where=Filter.by_id().contains_any(to_delete))
            
            # Do the same for graph collection
            graph_collection = self.client.collections.get("ProjectGraph")
            # Similar deletion logic...
            
        except Exception as e:
            print(f"Error clearing project data: {e}")
        
        # Clear cache
        with self.cache_lock:
            self.cache.clear()
    
    def remove_file(self, user_id: str, project_id: str, raw_file_path: str) -> bool:
        """Remove a file from the index"""
        try:
            # Normalize path for consistency
            file_path = raw_file_path[6:] if raw_file_path.startswith('res://') else raw_file_path
            
            # Use the centralized cleanup method
            self._remove_existing_file_chunks(user_id, project_id, raw_file_path)
                
            # Clear cache
            with self.cache_lock:
                # Remove cache entries for this file
                keys_to_remove = [k for k in self.cache.keys() if file_path in k]
                for k in keys_to_remove:
                    self.cache.pop(k, None)
                    
            return True
            
        except Exception as e:
            print(f"WeaviateVector: Error removing file {file_path}: {e}")
            return False
    
    def get_graph_context_for_files(self, file_paths: List[str], user_id: str, project_id: str, 
                                   **kwargs) -> Dict[str, Any]:
        """Get graph context for files"""
        context = {}
        
        try:
            collection = self.client.collections.get("ProjectGraph")
            
            for file_path in file_paths:
                nodes = []
                edges = []
                
                # Get ALL graph edges and filter (API compatibility workaround)
                try:
                    # Fetch more objects to account for API limitations
                    all_edges = collection.query.fetch_objects(
                        limit=500,  # Increased limit to catch all relationships
                        return_properties=["user_id", "project_id", "source_file", "target_file", "relationship_type", "weight"]
                    )
                    
                    print(f"WeaviateGraph: Fetched {len(all_edges.objects)} total graph objects for file {file_path}")
                    
                    for obj in all_edges.objects:
                        props = obj.properties
                        if (props.get('user_id') == user_id and 
                            props.get('project_id') == project_id):
                            
                            source_file = props.get('source_file')
                            target_file = props.get('target_file')
                            
                            # Check if this edge involves our file (either as source or target)
                            if source_file == file_path or target_file == file_path:
                                edge = {
                                    'source': source_file,
                                    'target': target_file,
                                    'type': props.get('relationship_type', 'reference'),
                                    'weight': props.get('weight', 1.0)
                                }
                                edges.append(edge)
                                
                                # Add both source and target as nodes
                                for node_id in [source_file, target_file]:
                                    if node_id and {'id': node_id, 'type': 'file'} not in nodes:
                                        nodes.append({'id': node_id, 'type': 'file'})
                                        
                                print(f"WeaviateGraph: Found edge {source_file} -> {target_file} ({props.get('relationship_type')})")
                    
                except Exception as e:
                    print(f"WeaviateGraph: Error fetching graph edges: {e}")
                
                # Add the file itself as a node if not already present
                if {'id': file_path, 'type': 'file'} not in nodes:
                    nodes.append({'id': file_path, 'type': 'file', 'central': True})
                
                context[file_path] = {
                    'nodes': nodes,
                    'edges': edges
                }
                
                print(f"WeaviateGraph: File {file_path} final context: {len(nodes)} nodes, {len(edges)} edges")
                
        except Exception as e:
            print(f"WeaviateVector: Error getting graph context: {e}")
            # Return empty context on error
            context = {fp: {"nodes": [], "edges": []} for fp in file_paths}
        
        return context
    
    def get_graph_context_expanded(self, file_paths: List[str], user_id: str, project_id: str,
                                  depth: int = 1, kinds: List[str] = None) -> Dict[str, Any]:
        """Get expanded graph context with depth traversal"""
        if depth <= 0:
            return self.get_graph_context_for_files(file_paths, user_id, project_id)
        
        # For now, just return single-level context
        # Full depth traversal can be implemented later
        return self.get_graph_context_for_files(file_paths, user_id, project_id)
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        self.client.close()
