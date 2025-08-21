"""
© 2025 Simplifine Corp. Personal Non‑Commercial License.
See LICENSES/COMPANY-NONCOMMERCIAL.md for terms.

Optimized Weaviate Vector Manager with parallelization and efficient querying
"""
import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey
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
        
        # Initialize Weaviate client v4 with proper error handling
        try:
            # Extract cluster name from URL for v4 API
            if 'weaviate.cloud' in weaviate_url:
                # For Weaviate Cloud, use connect_to_weaviate_cloud
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=weaviate_url,
                    auth_credentials=AuthApiKey(api_key),
                    skip_init_checks=True  # Skip gRPC health check if problematic
                )
            else:
                # For local/self-hosted Weaviate
                self.client = weaviate.connect_to_local(
                    host=weaviate_url.replace('http://', '').replace('https://', ''),
                    auth_credentials=AuthApiKey(api_key) if api_key else None,
                    additional_config=wvc.init.AdditionalConfig(
                        timeout=wvc.init.Timeout(init=30, query=60, insert=120)
                    )
                )
            print(f"✅ Weaviate v4 client connected successfully")
        except Exception as e:
            print(f"❌ Weaviate connection failed: {e}")
            raise
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.embedding_semaphore = threading.Semaphore(5)  # Limit concurrent embeddings
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize collections
        self._init_collections()
        
        # Dependency types for multi-hop tracing
        self.DEPENDENCY_WEIGHTS = {
            'CALLS_FUNCTION': 1.0,      # Direct function call
            'EMITS_SIGNAL': 2.0,        # Signal emission  
            'CONNECTS_SIGNAL': 3.0,     # Signal connection
            'ACCESSES_NODE': 1.0,       # get_node() reference
            'EXTENDS_CLASS': 2.0,       # Inheritance
            'PRELOADS_RESOURCE': 1.0,   # preload() dependency
            'INSTANCES_SCENE': 2.0,     # Scene instantiation
            'USES_RESOURCE': 1.0,       # Resource reference
        }
        
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
            
        # Function-level dependencies collection for multi-hop tracing
        if not collections.exists("ProjectDependencies"):
            collections.create(
                name="ProjectDependencies",
                properties=[
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="project_id", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                    Property(name="source_function", data_type=DataType.TEXT),
                    Property(name="source_node", data_type=DataType.TEXT),
                    Property(name="target_file", data_type=DataType.TEXT),
                    Property(name="target_function", data_type=DataType.TEXT),
                    Property(name="target_node", data_type=DataType.TEXT),
                    Property(name="dependency_type", data_type=DataType.TEXT),
                    Property(name="signal_name", data_type=DataType.TEXT),
                    Property(name="line_number", data_type=DataType.INT),
                    Property(name="context", data_type=DataType.TEXT),
                    Property(name="weight", data_type=DataType.NUMBER),
                    Property(name="timestamp", data_type=DataType.DATE),
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
            
            # CRITICAL: Filter out files that shouldn't be indexed (like .import files)
            if not self._should_index_file(file_path):
                print(f"WeaviateVector: Skipping filtered file: {file_path}")
                stats["skipped"] += 1
                continue
            
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
            
            # NUCLEAR FILTER: Block .import files from EVER appearing in search results
            if file_path.endswith('.import'):
                print(f"WeaviateVector: BLOCKING old junk from search results: {file_path}")
                continue
            
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
        """Enhanced chunking for code files - chunks by semantic units (functions, classes, signals)"""
        lines = content.split('\n')
        chunks = []
        ext = os.path.splitext(file_path)[1].lower()
        
        # Use enhanced GDScript semantic chunking
        if ext == '.gd':
            chunks = self._chunk_gdscript_semantic(content, file_path, max_lines)
        elif file_path.endswith(('.tscn', '.tres')):
            chunks = self._chunk_godot_resource(content, file_path)
        elif ext in ['.py', '.cs', '.cpp', '.js', '.ts']:
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
    
    def _chunk_gdscript_semantic(self, content: str, file_path: str, max_lines: int) -> List[Dict]:
        """Semantic chunking for GDScript - each function/signal/class becomes a searchable unit"""
        chunks = []
        lines = content.split('\n')
        
        # Track current context
        current_class = None
        current_extends = None
        file_header_end = 0
        
        # Find file-level declarations (extends, class_name, tool, etc.)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('extends '):
                current_extends = stripped
                file_header_end = max(file_header_end, i + 1)
            elif stripped.startswith('class_name '):
                import re
                current_class = re.match(r'class_name\s+(\w+)', stripped)
                if current_class:
                    current_class = current_class.group(1)
                file_header_end = max(file_header_end, i + 1)
            elif stripped.startswith(('@tool', '@icon')):
                file_header_end = max(file_header_end, i + 1)
            elif stripped and not stripped.startswith('#') and i > file_header_end + 2:
                break  # End of header section
        
        # Create header chunk if meaningful
        if file_header_end > 0:
            header_content = '\n'.join(lines[:file_header_end])
            if header_content.strip():
                chunks.append({
                    'start_line': 1,
                    'end_line': file_header_end,
                    'content': f"# File: {file_path}\n{header_content}"
                })
        
        # Find all functions, signals, and other semantic units
        i = file_header_end
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Signal definition
            if stripped.startswith('signal '):
                import re
                signal_match = re.match(r'signal\s+(\w+)', stripped)
                if signal_match:
                    signal_name = signal_match.group(1)
                    # Signals are usually one line, but include any following comments
                    end_line = i + 1
                    while end_line < len(lines) and lines[end_line].strip().startswith('#'):
                        end_line += 1
                    
                    chunk_content = f"# Signal: {signal_name} in {file_path}\n"
                    chunk_content += '\n'.join(lines[i:end_line])
                    
                    chunks.append({
                        'start_line': i + 1,
                        'end_line': end_line,
                        'content': chunk_content
                    })
                    i = end_line
                    continue
            
            # Function definition
            elif stripped.startswith('func '):
                import re
                func_match = re.match(r'func\s+(\w+)\s*\((.*?)\)', stripped)
                if func_match:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2)
                    
                    # Find function end (next func, class, or dedent)
                    func_start = i
                    func_end = i + 1
                    base_indent = len(line) - len(line.lstrip())
                    
                    while func_end < len(lines):
                        next_line = lines[func_end]
                        next_stripped = next_line.strip()
                        
                        # Check for next semantic unit
                        if next_stripped.startswith(('func ', 'signal ', 'class ', 'static func ', 'const ', 'var ', 'export')):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= base_indent:
                                break
                        
                        # Check for dedent (end of function)
                        elif next_stripped and not next_stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= base_indent and func_end > func_start + 1:
                                break
                        
                        func_end += 1
                    
                    # Create searchable content with context
                    chunk_content = f"# Function: {func_name}({func_params}) in {file_path}\n"
                    if current_class:
                        chunk_content += f"# Class: {current_class}\n"
                    if current_extends:
                        chunk_content += f"# {current_extends}\n"
                    
                    func_content = '\n'.join(lines[func_start:func_end])
                    
                    # Extract key information for better searchability
                    signals_emitted = re.findall(r'emit_signal\s*\(\s*["\'](\w+)["\']', func_content)
                    if signals_emitted:
                        chunk_content += f"# Emits signals: {', '.join(set(signals_emitted))}\n"
                    
                    nodes_accessed = re.findall(r'get_node\s*\(\s*["\']([^"\']+)["\']', func_content)
                    if nodes_accessed:
                        chunk_content += f"# Accesses nodes: {', '.join(set(nodes_accessed))}\n"
                    
                    functions_called = re.findall(r'(\w+)\s*\(', func_content)
                    # Filter out keywords and this function
                    functions_called = [f for f in functions_called if f not in 
                                      ['if', 'for', 'while', 'func', 'var', 'const', 'return', 
                                       'elif', 'else', 'match', 'and', 'or', 'not', func_name]]
                    if functions_called:
                        chunk_content += f"# Calls: {', '.join(set(functions_called[:10]))}\n"
                    
                    chunk_content += func_content
                    
                    chunks.append({
                        'start_line': func_start + 1,
                        'end_line': func_end,
                        'content': chunk_content
                    })
                    i = func_end
                    continue
            
            # Export variables (important for inspector properties)
            elif stripped.startswith('@export') or stripped.startswith('export '):
                var_start = i
                var_end = i + 1
                # Include any following comments
                while var_end < len(lines) and lines[var_end].strip().startswith('#'):
                    var_end += 1
                
                chunk_content = f"# Exported variable in {file_path}\n"
                chunk_content += '\n'.join(lines[var_start:var_end])
                
                chunks.append({
                    'start_line': var_start + 1,
                    'end_line': var_end,
                    'content': chunk_content
                })
                i = var_end
                continue
            
            i += 1
        
        # If no semantic chunks were created, fall back to regular chunking
        if not chunks:
            chunks = self._smart_chunk_code(lines, max_lines)
        
        return chunks
    
    def _chunk_godot_resource(self, content: str, file_path: str) -> List[Dict]:
        """Enhanced chunking for Godot scene/resource files - each node becomes a searchable unit"""
        chunks = []
        lines = content.split('\n')
        
        # Track external resources for context
        ext_resources = {}  # id -> {path, type}
        import re
        
        # First pass: collect external resources
        for i, line in enumerate(lines):
            if line.strip().startswith('[ext_resource'):
                # Parse external resource
                res_id = self._extract_attr(line, 'id')
                res_path = self._extract_attr(line, 'path')
                res_type = self._extract_attr(line, 'type')
                if res_id:
                    ext_resources[res_id] = {'path': res_path, 'type': res_type}
        
        # Second pass: chunk by meaningful sections
        current_section = []
        current_start = 1
        current_header = None
        section_metadata = {}
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # New section starts
            if stripped.startswith('[') and stripped.endswith(']'):
                # Save previous section if it exists
                if current_section and current_header:
                    chunk_content = self._create_scene_chunk_content(
                        current_header,
                        current_section,
                        section_metadata,
                        ext_resources,
                        file_path
                    )
                    
                    chunks.append({
                        'start_line': current_start,
                        'end_line': i,
                        'content': chunk_content
                    })
                
                # Start new section
                current_header = stripped
                current_section = [line]
                current_start = i + 1
                section_metadata = self._parse_section_header(stripped)
            else:
                current_section.append(line)
                
                # Extract important metadata from content
                if current_header and current_header.startswith('[node'):
                    if 'script' in line:
                        # Extract script reference
                        script_match = re.search(r'script = ExtResource\(\s*["\']?(\d+)["\']?\s*\)', line)
                        if script_match and script_match.group(1) in ext_resources:
                            res = ext_resources[script_match.group(1)]
                            if res['type'] in ('Script', 'GDScript'):
                                section_metadata['attached_script'] = res['path']
        
        # Don't forget last section
        if current_section and current_header:
            chunk_content = self._create_scene_chunk_content(
                current_header,
                current_section,
                section_metadata,
                ext_resources,
                file_path
            )
            
            chunks.append({
                'start_line': current_start,
                'end_line': len(lines),
                'content': chunk_content
            })
        
        # If no meaningful chunks, fall back to simple chunking
        if not chunks:
            for i in range(0, len(lines), 100):
                chunk_lines = lines[i:i + 100]
                if chunk_lines:
                    chunks.append({
                        'start_line': i + 1,
                        'end_line': i + len(chunk_lines),
                        'content': '\n'.join(chunk_lines)
                    })
        
        return chunks
    
    def _parse_section_header(self, header: str) -> dict:
        """Parse section header to extract metadata"""
        metadata = {}
        
        if header.startswith('[node'):
            metadata['type'] = 'node'
            metadata['name'] = self._extract_attr(header, 'name')
            metadata['node_type'] = self._extract_attr(header, 'type')
            metadata['parent'] = self._extract_attr(header, 'parent')
            metadata['groups'] = self._extract_attr(header, 'groups')
            
        elif header.startswith('[connection'):
            metadata['type'] = 'connection'
            metadata['signal'] = self._extract_attr(header, 'signal')
            metadata['from'] = self._extract_attr(header, 'from')
            metadata['to'] = self._extract_attr(header, 'to')
            metadata['method'] = self._extract_attr(header, 'method')
            
        elif header.startswith('[ext_resource'):
            metadata['type'] = 'ext_resource'
            metadata['id'] = self._extract_attr(header, 'id')
            metadata['path'] = self._extract_attr(header, 'path')
            metadata['resource_type'] = self._extract_attr(header, 'type')
        
        return metadata
    
    def _create_scene_chunk_content(self, header: str, section_lines: list, 
                                  metadata: dict, ext_resources: dict, file_path: str) -> str:
        """Create searchable content for a scene section with rich context"""
        chunk_content = f"# Scene File: {file_path}\n"
        
        section_type = metadata.get('type', 'unknown')
        
        if section_type == 'node':
            # Node section - add rich context
            node_name = metadata.get('name', 'Unknown')
            node_type = metadata.get('node_type', 'Node')
            parent = metadata.get('parent', '')
            
            chunk_content += f"# Node: {node_name} (Type: {node_type})\n"
            
            if parent and parent != '.':
                chunk_content += f"# Parent: {parent}\n"
            
            if metadata.get('groups'):
                chunk_content += f"# Groups: {metadata['groups']}\n"
            
            if metadata.get('attached_script'):
                chunk_content += f"# Script: {metadata['attached_script']}\n"
            
            # Extract node path for better searchability
            if parent and parent != '.':
                node_path = f"{parent.rstrip('/')}/{node_name}"
            else:
                node_path = node_name
            chunk_content += f"# Path: {node_path}\n"
            
        elif section_type == 'connection':
            # Signal connection - crucial for understanding flow
            signal = metadata.get('signal', '')
            from_node = metadata.get('from', '')
            to_node = metadata.get('to', '')
            method = metadata.get('method', '')
            
            chunk_content += f"# Signal Connection: {signal}\n"
            chunk_content += f"# From: {from_node} -> To: {to_node}\n"
            chunk_content += f"# Method: {method}\n"
            chunk_content += f"# Flow: When {from_node} emits '{signal}', {to_node}.{method}() is called\n"
        
        elif section_type == 'ext_resource':
            # External resource reference
            res_path = metadata.get('path', '')
            res_type = metadata.get('resource_type', '')
            chunk_content += f"# External Resource: {res_type}\n"
            chunk_content += f"# Path: {res_path}\n"
        
        # Add the actual content
        chunk_content += '\n'.join(section_lines)
        
        return chunk_content
    
    @staticmethod
    def _extract_attr(header: str, key: str):
        """Extract attribute from header string"""
        import re
        m = re.search(key + r'\s*=\s*"([^\"]+)"', header)
        return m.group(1) if m else None
    
    def _update_graph_relationships(self, files: List[Dict], user_id: str, project_id: str):
        """Update graph relationships and detailed dependencies for multi-hop tracing"""
        try:
            # Extract relationships and detailed dependencies
            relationships = []
            dependencies = []
            
            for file_data in files:
                # Handle both 'path' and 'file_path' keys
                file_path = file_data.get('path') or file_data.get('file_path', '')
                content = file_data.get('content', '')
                
                if not file_path or not content:
                    continue
                
                # CRITICAL: Skip files that shouldn't be indexed (apply same filter as main indexing)
                if not self._should_index_file(file_path):
                    print(f"WeaviateVector: Skipping dependency extraction for filtered file: {file_path}")
                    continue
                
                # Extract basic file-level references
                refs = self._extract_references(content, file_path)
                for ref in refs:
                    relationships.append({
                        'source_file': file_path,
                        'target_file': ref['target'],
                        'relationship_type': ref['type'],
                        'weight': ref.get('weight', 1.0)
                    })
                
                # Extract detailed function-level dependencies for multi-hop tracing
                detailed_deps = self._extract_detailed_dependencies(content, file_path)
                print(f"WeaviateVector: Found {len(detailed_deps)} dependencies in {file_path}")
                dependencies.extend(detailed_deps)
            
            # Batch insert basic relationships
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
            
            # Batch insert detailed dependencies
            if dependencies:
                dep_collection = self.client.collections.get("ProjectDependencies")
                with dep_collection.batch.dynamic() as batch:
                    for dep in dependencies:
                        batch.add_object(properties={
                            'user_id': user_id,
                            'project_id': project_id,
                            'source_file': dep['source_file'],
                            'source_function': dep.get('source_function'),
                            'source_node': dep.get('source_node'),
                            'target_file': dep.get('target_file'),
                            'target_function': dep.get('target_function'),
                            'target_node': dep.get('target_node'),
                            'dependency_type': dep['dependency_type'],
                            'signal_name': dep.get('signal_name'),
                            'line_number': dep.get('line_number', 0),
                            'context': dep.get('context', ''),
                            'weight': dep.get('weight', 1.0),
                            'timestamp': datetime.now(timezone.utc),
                        })
                print(f"WeaviateVector: Stored {len(dependencies)} detailed dependencies for multi-hop tracing")
                
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
    
    def _extract_detailed_dependencies(self, content: str, file_path: str) -> List[Dict]:
        """Extract detailed function-level dependencies for multi-hop tracing"""
        dependencies = []
        lines = content.split('\n')
        ext = os.path.splitext(file_path)[1].lower()
        
        # Only extract detailed dependencies for GDScript and scene files
        if ext == '.gd':
            dependencies.extend(self._extract_gdscript_dependencies(content, file_path, lines))
        elif ext == '.tscn':
            dependencies.extend(self._extract_scene_dependencies(content, file_path, lines))
        
        return dependencies
    
    def _extract_gdscript_dependencies(self, content: str, file_path: str, lines: List[str]) -> List[Dict]:
        """Extract function-level dependencies from GDScript"""
        import re
        dependencies = []
        
        # Track functions for context
        functions = {}  # function_name -> (start_line, end_line)
        
        # First pass: identify all functions
        for i, line in enumerate(lines):
            func_match = re.match(r'^\s*func\s+(\w+)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                # Find function end (simplified)
                func_end = i + 1
                base_indent = len(line) - len(line.lstrip())
                
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() and not next_line.strip().startswith('#'):
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= base_indent and next_line.strip().startswith(('func ', 'signal ', 'class ', 'var ', 'const ')):
                            break
                    func_end = j + 1
                
                functions[func_name] = (i + 1, func_end)
        
        print(f"WeaviateVector: Found {len(functions)} functions in {file_path}: {list(functions.keys())}")
        
        # Second pass: extract dependencies within each function
        for func_name, (start_line, end_line) in functions.items():
            func_body = '\n'.join(lines[start_line-1:end_line])
            func_deps_found = 0
            
            # Function calls within this function (more comprehensive patterns)
            # Direct calls: function_name(
            for match in re.finditer(r'(\w+)\s*\(', func_body):
                called_func = match.group(1)
                # Skip keywords and built-ins
                if (called_func in functions and called_func != func_name and 
                    called_func not in ['if', 'for', 'while', 'return', 'print', 'len', 'str', 'int', 'float']):
                    dependencies.append({
                        'source_file': file_path,
                        'source_function': func_name,
                        'target_file': file_path,
                        'target_function': called_func,
                        'dependency_type': 'CALLS_FUNCTION',
                        'line_number': start_line + func_body[:match.start()].count('\n'),
                        'context': match.group(0),
                        'weight': self.DEPENDENCY_WEIGHTS.get('CALLS_FUNCTION', 1.0)
                    })
                    func_deps_found += 1
            
            # Method calls: self.method_name(
            for match in re.finditer(r'self\.(\w+)\s*\(', func_body):
                called_method = match.group(1)
                if called_method in functions and called_method != func_name:
                    dependencies.append({
                        'source_file': file_path,
                        'source_function': func_name,
                        'target_file': file_path,
                        'target_function': called_method,
                        'dependency_type': 'CALLS_FUNCTION',
                        'line_number': start_line + func_body[:match.start()].count('\n'),
                        'context': f"self.{match.group(0)}",
                        'weight': self.DEPENDENCY_WEIGHTS.get('CALLS_FUNCTION', 1.0)
                    })
                    func_deps_found += 1
            
            # Signal emissions
            for match in re.finditer(r'emit_signal\s*\(\s*["\'](\w+)["\']', func_body):
                signal_name = match.group(1)
                dependencies.append({
                    'source_file': file_path,
                    'source_function': func_name,
                    'target_file': file_path,  # Signal target will be resolved later
                    'target_function': None,
                    'dependency_type': 'EMITS_SIGNAL',
                    'signal_name': signal_name,
                    'line_number': start_line + func_body[:match.start()].count('\n'),
                    'context': match.group(0),
                    'weight': self.DEPENDENCY_WEIGHTS.get('EMITS_SIGNAL', 2.0)
                })
                func_deps_found += 1
            
            # Signal connections
            for match in re.finditer(r'(\w+)\.connect\s*\(\s*["\'](\w+)["\'],\s*([^,]+),\s*["\'](\w+)["\']', func_body):
                source_node = match.group(1)
                signal_name = match.group(2)
                target_ref = match.group(3).strip()
                target_method = match.group(4)
                
                dependencies.append({
                    'source_file': file_path,
                    'source_function': func_name,
                    'source_node': source_node,
                    'target_file': file_path,  # May need cross-file resolution
                    'target_function': target_method,
                    'target_node': target_ref,
                    'dependency_type': 'CONNECTS_SIGNAL',
                    'signal_name': signal_name,
                    'line_number': start_line + func_body[:match.start()].count('\n'),
                    'context': match.group(0),
                    'weight': self.DEPENDENCY_WEIGHTS.get('CONNECTS_SIGNAL', 3.0)
                })
                func_deps_found += 1
            
            # Node accesses (multiple patterns)
            # get_node patterns
            for match in re.finditer(r'get_node\s*\(\s*["\']([^"\']+)["\']', func_body):
                node_path = match.group(1)
                dependencies.append({
                    'source_file': file_path,
                    'source_function': func_name,
                    'target_node': node_path,
                    'dependency_type': 'ACCESSES_NODE',
                    'line_number': start_line + func_body[:match.start()].count('\n'),
                    'context': match.group(0),
                    'weight': self.DEPENDENCY_WEIGHTS.get('ACCESSES_NODE', 1.0)
                })
                func_deps_found += 1
            
            # @onready var node references
            for match in re.finditer(r'@onready\s+var\s+\w+\s*=\s*get_node\s*\(\s*["\']([^"\']+)["\']', func_body):
                node_path = match.group(1)
                dependencies.append({
                    'source_file': file_path,
                    'source_function': func_name,
                    'target_node': node_path,
                    'dependency_type': 'ONREADY_NODE_REF',
                    'line_number': start_line + func_body[:match.start()].count('\n'),
                    'context': match.group(0),
                    'weight': 1.2
                })
                func_deps_found += 1
            
            # $ node shortcuts
            for match in re.finditer(r'\$([A-Za-z_][A-Za-z0-9_]*)', func_body):
                node_name = match.group(1)
                dependencies.append({
                    'source_file': file_path,
                    'source_function': func_name,
                    'target_node': node_name,
                    'dependency_type': 'ACCESSES_NODE',
                    'line_number': start_line + func_body[:match.start()].count('\n'),
                    'context': f"${node_name}",
                    'weight': self.DEPENDENCY_WEIGHTS.get('ACCESSES_NODE', 1.0)
                })
                func_deps_found += 1
            
            # Physics/game state changes
            physics_calls = re.findall(r'(set_velocity|move_and_slide|is_on_floor|is_on_wall|apply_impulse|set_position)', func_body)
            for call in physics_calls:
                dependencies.append({
                    'source_file': file_path,
                    'source_function': func_name,
                    'target_function': call,
                    'dependency_type': 'CALLS_PHYSICS',
                    'line_number': start_line,
                    'context': f"Physics call: {call}",
                    'weight': 1.5
                })
                func_deps_found += 1
            
            print(f"WeaviateVector: Function {func_name} has {func_deps_found} dependencies")
        
        print(f"WeaviateVector: Total dependencies extracted from {file_path}: {len(dependencies)}")
        return dependencies
    
    def _extract_scene_dependencies(self, content: str, file_path: str, lines: List[str]) -> List[Dict]:
        """Extract dependencies from scene files"""
        import re
        dependencies = []
        
        # Parse signal connections in scene files
        for i, line in enumerate(lines):
            if line.strip().startswith('[connection'):
                signal = self._extract_attr(line, 'signal')
                from_node = self._extract_attr(line, 'from')
                to_node = self._extract_attr(line, 'to')
                method = self._extract_attr(line, 'method')
                
                if signal and from_node and to_node and method:
                    dependencies.append({
                        'source_file': file_path,
                        'source_node': from_node,
                        'target_file': file_path,
                        'target_function': method,
                        'target_node': to_node,
                        'dependency_type': 'CONNECTS_SIGNAL',
                        'signal_name': signal,
                        'line_number': i + 1,
                        'context': f"Scene connection: {from_node}.{signal} -> {to_node}.{method}()",
                        'weight': self.DEPENDENCY_WEIGHTS.get('CONNECTS_SIGNAL', 3.0)
                    })
        
        return dependencies
    
    def _clear_collection_by_project(self, collection, user_id: str, project_id: str):
        """Helper to clear a collection for a specific project"""
        try:
            # Fetch and delete objects manually since filtered delete might not work reliably
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
                
        except Exception as e:
            print(f"WeaviateVector: Error clearing collection: {e}")
    
    def _remove_existing_file_chunks(self, user_id: str, project_id: str, raw_file_path: str):
        """Remove all existing chunks and dependencies for a file to prevent duplicates during re-indexing"""
        try:
            # Normalize path for consistency
            file_path = raw_file_path[6:] if raw_file_path.startswith('res://') else raw_file_path
            
            # Remove from main embeddings collection
            collection = self.client.collections.get("ProjectEmbedding")
            
            # Try to delete objects with filters (may not work with older Weaviate clients)
            try:
                collection.data.delete_many(
                    where=Filter.by_property("user_id").equal(user_id) & 
                          Filter.by_property("project_id").equal(project_id) &
                          Filter.by_property("file_path").equal(file_path)
                )
                print(f"WeaviateVector: Deleted old chunks for {file_path} using filter")
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
            
            # Also remove from graph and dependencies collections
            self._remove_file_from_graph_collections(user_id, project_id, file_path)
                
        except Exception as e:
            print(f"WeaviateVector: Failed to remove existing chunks for {file_path}: {e}")
            # Don't raise - we want indexing to continue even if cleanup fails
    
    def _remove_file_from_graph_collections(self, user_id: str, project_id: str, file_path: str):
        """Remove file from graph and dependencies collections"""
        try:
            # Remove from graph relationships
            graph_collection = self.client.collections.get("ProjectGraph")
            graph_collection.data.delete_many(
                where=Filter.by_property("user_id").equal(user_id) & 
                      Filter.by_property("project_id").equal(project_id) &
                      (Filter.by_property("source_file").equal(file_path) |
                       Filter.by_property("target_file").equal(file_path))
            )
            
            # Remove from dependencies
            deps_collection = self.client.collections.get("ProjectDependencies")
            deps_collection.data.delete_many(
                where=Filter.by_property("user_id").equal(user_id) & 
                      Filter.by_property("project_id").equal(project_id) &
                      (Filter.by_property("source_file").equal(file_path) |
                       Filter.by_property("target_file").equal(file_path))
            )
        except Exception as e:
            print(f"WeaviateVector: Error removing {file_path} from graph collections: {e}")
    
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
            '.pyc', '.pyo', '.class', '.uid',
            '.import'  # Godot import metadata files - never index these!
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
        """Nuclear clear of ALL project data including old junk files"""
        print(f"WeaviateVector: NUCLEAR CLEAR starting for project {project_id}")
        
        try:
            collections_to_clear = ["ProjectEmbedding", "ProjectGraph", "ProjectDependencies"]
            
            for collection_name in collections_to_clear:
                try:
                    collection = self.client.collections.get(collection_name)
                    print(f"WeaviateVector: Clearing {collection_name}...")
                    
                    # Nuclear approach: delete ALL objects for this project
                    to_delete = []
                    batch_size = 50  # Smaller batches for reliability
                    
                    # Keep fetching until no more objects
                    while True:
                        objects = collection.query.fetch_objects(
                            limit=batch_size,
                            return_properties=["user_id", "project_id", "file_path"]
                        )
                        
                        if not objects.objects:
                            break
                        
                        found_project_objects = False
                        for obj in objects.objects:
                            if (obj.properties.get('user_id') == user_id and 
                                obj.properties.get('project_id') == project_id):
                                to_delete.append(obj.uuid)
                                found_project_objects = True
                                
                                # Log what we're deleting for transparency
                                file_path = obj.properties.get('file_path', 'unknown')
                                if file_path.endswith('.import'):
                                    print(f"WeaviateVector: Deleting old junk: {file_path}")
                        
                        # Delete current batch
                        if to_delete:
                            try:
                                collection.data.delete_many(where=Filter.by_id().contains_any(to_delete))
                                print(f"WeaviateVector: Deleted {len(to_delete)} objects from {collection_name}")
                                to_delete = []
                            except Exception as del_e:
                                print(f"WeaviateVector: Batch delete failed: {del_e}")
                        
                        # If no project objects found in this batch, we're done
                        if not found_project_objects:
                            break
                    
                    print(f"WeaviateVector: ✅ {collection_name} cleared")
                    
                except Exception as e:
                    print(f"WeaviateVector: Error clearing {collection_name}: {e}")
            
        except Exception as e:
            print(f"WeaviateVector: Nuclear clear error: {e}")
        
        # Clear cache
        with self.cache_lock:
            self.cache.clear()
            
        print(f"WeaviateVector: NUCLEAR CLEAR completed for project {project_id}")
    
    def remove_file(self, user_id: str, project_id: str, raw_file_path: str) -> bool:
        """Remove a file from the index"""
        try:
            # Normalize path for consistency
            file_path = raw_file_path[6:] if raw_file_path.startswith('res://') else raw_file_path
            
            # Remove from main embeddings collection
            self._remove_existing_file_chunks(user_id, project_id, raw_file_path)
            
            # Remove from graph relationships
            try:
                graph_collection = self.client.collections.get("ProjectGraph")
                graph_collection.data.delete_many(
                    where=Filter.by_property("user_id").equal(user_id) & 
                          Filter.by_property("project_id").equal(project_id) &
                          (Filter.by_property("source_file").equal(file_path) |
                           Filter.by_property("target_file").equal(file_path))
                )
            except Exception as e:
                print(f"WeaviateVector: Error removing graph relationships for {file_path}: {e}")
            
            # Remove from dependencies
            try:
                deps_collection = self.client.collections.get("ProjectDependencies")
                deps_collection.data.delete_many(
                    where=Filter.by_property("user_id").equal(user_id) & 
                          Filter.by_property("project_id").equal(project_id) &
                          (Filter.by_property("source_file").equal(file_path) |
                           Filter.by_property("target_file").equal(file_path))
                )
            except Exception as e:
                print(f"WeaviateVector: Error removing dependencies for {file_path}: {e}")
                
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
    
    def trace_dependencies(self, start_function: str, start_file: str, user_id: str, project_id: str,
                          direction: str = 'forward', max_hops: int = 3) -> Dict[str, Any]:
        """
        Multi-hop dependency tracing for understanding code impact and flow
        
        Args:
            start_function: Starting function name
            start_file: Starting file path
            direction: 'forward' (what this affects) or 'backward' (what affects this)
            max_hops: Maximum number of hops to trace
            
        Returns:
            Dictionary with dependency chain, affected files, and flow analysis
        """
        try:
            dependency_collection = self.client.collections.get("ProjectDependencies")
            
            visited = set()
            dependency_chain = []
            affected_files = set()
            
            def _trace_recursive(current_func, current_file, current_node, depth):
                if depth > max_hops or (current_func, current_file) in visited:
                    return
                
                visited.add((current_func, current_file))
                affected_files.add(current_file)
                
                # Query dependencies
                try:
                    if direction == 'forward':
                        # Find what this function affects
                        results = dependency_collection.query.fetch_objects(
                            limit=100,
                            return_properties=["source_file", "source_function", "source_node",
                                             "target_file", "target_function", "target_node", 
                                             "dependency_type", "signal_name", "weight", "context"],
                            where=Filter.by_property("user_id").equal(user_id) & 
                                  Filter.by_property("project_id").equal(project_id) &
                                  Filter.by_property("source_file").equal(current_file) &
                                  Filter.by_property("source_function").equal(current_func)
                        )
                    else:
                        # Find what affects this function
                        results = dependency_collection.query.fetch_objects(
                            limit=100,
                            return_properties=["source_file", "source_function", "source_node",
                                             "target_file", "target_function", "target_node", 
                                             "dependency_type", "signal_name", "weight", "context"],
                            where=Filter.by_property("user_id").equal(user_id) & 
                                  Filter.by_property("project_id").equal(project_id) &
                                  Filter.by_property("target_file").equal(current_file) &
                                  Filter.by_property("target_function").equal(current_func)
                        )
                except Exception:
                    # Fallback: fetch all and filter manually
                    results = dependency_collection.query.fetch_objects(
                        limit=500,
                        return_properties=["user_id", "project_id", "source_file", "source_function", 
                                         "target_file", "target_function", "dependency_type", "signal_name", "weight"]
                    )
                    
                    # Filter manually
                    filtered_objects = []
                    for obj in results.objects:
                        props = obj.properties
                        if (props.get('user_id') == user_id and 
                            props.get('project_id') == project_id):
                            
                            if direction == 'forward':
                                if (props.get('source_file') == current_file and 
                                    props.get('source_function') == current_func):
                                    filtered_objects.append(obj)
                            else:
                                if (props.get('target_file') == current_file and 
                                    props.get('target_function') == current_func):
                                    filtered_objects.append(obj)
                    
                    # Mock results object
                    class MockResults:
                        def __init__(self, objects):
                            self.objects = objects
                    results = MockResults(filtered_objects)
                
                # Process dependencies
                for obj in results.objects:
                    props = obj.properties
                    
                    if direction == 'forward':
                        target_func = props.get('target_function')
                        target_file = props.get('target_file')
                        target_node = props.get('target_node')
                    else:
                        target_func = props.get('source_function')
                        target_file = props.get('source_file')
                        target_node = props.get('source_node')
                    
                    if target_func and target_file:
                        dependency_chain.append({
                            'from': {'function': current_func, 'file': current_file, 'node': current_node},
                            'to': {'function': target_func, 'file': target_file, 'node': target_node},
                            'type': props.get('dependency_type'),
                            'signal': props.get('signal_name'),
                            'weight': props.get('weight', 1.0),
                            'depth': depth
                        })
                        
                        # Recurse
                        _trace_recursive(target_func, target_file, target_node, depth + 1)
            
            # Start the trace
            _trace_recursive(start_function, start_file, None, 0)
            
            return {
                'start': {'function': start_function, 'file': start_file},
                'direction': direction,
                'max_hops': max_hops,
                'chain': dependency_chain,
                'affected_files': list(affected_files),
                'total_dependencies': len(dependency_chain)
            }
            
        except Exception as e:
            print(f"WeaviateVector: Error tracing dependencies: {e}")
            return {
                'start': {'function': start_function, 'file': start_file},
                'direction': direction,
                'chain': [],
                'affected_files': [],
                'error': str(e)
            }
    
    def search_with_dependency_context(self, query: str, user_id: str, project_id: str, 
                                     max_results: int = 5, include_dependencies: bool = True) -> List[Dict]:
        """Enhanced search that includes dependency tracing for each result"""
        # Get basic search results
        results = self.search(query, user_id, project_id, max_results)
        
        if not include_dependencies:
            return results
        
        # Add dependency context to each result
        for result in results:
            content = result.get('content', '')
            file_path = result['file_path']
            
            # Extract function name if this is a function chunk
            import re
            func_match = re.search(r'# Function: (\w+)\(', content)
            if func_match:
                func_name = func_match.group(1)
                
                # Trace forward dependencies (what this function affects)
                forward_deps = self.trace_dependencies(func_name, file_path, user_id, project_id, 'forward', 2)
                
                # Trace backward dependencies (what affects this function)  
                backward_deps = self.trace_dependencies(func_name, file_path, user_id, project_id, 'backward', 2)
                
                result['dependency_context'] = {
                    'function_name': func_name,
                    'affects': forward_deps['chain'][:5],  # Limit for UI
                    'affected_by': backward_deps['chain'][:5],
                    'total_forward': len(forward_deps['chain']),
                    'total_backward': len(backward_deps['chain'])
                }
        
        return results
    
    def keyword_search(self, query: str, user_id: str, project_id: str, max_results: int = 5) -> List[Dict]:
        """Pure keyword/exact text search - no semantic similarity"""
        try:
            collection = self.client.collections.get("ProjectEmbedding")
            
            # Try BM25 search without where filter (v4 syntax issue)
            try:
                results = collection.query.bm25(
                    query=query,
                    limit=max_results * 10,  # Get more since we'll filter manually
                    return_properties=["file_path", "content", "chunk_index", 
                                     "chunk_start", "chunk_end", "file_type", 
                                     "user_id", "project_id"]
                )
            except Exception as bm25_error:
                print(f"WeaviateKeyword: BM25 failed ({bm25_error}), trying fallback search")
                # Fallback to semantic search and manually filter for exact matches
                results = collection.query.near_text(
                    query=query,
                    limit=max_results * 10,
                    return_properties=["file_path", "content", "chunk_index", 
                                     "chunk_start", "chunk_end", "file_type", 
                                     "user_id", "project_id"]
                )
            
            formatted_results = []
            seen_files = set()
            exact_matches = 0
            
            print(f"WeaviateKeyword: Found {len(results.objects)} raw BM25 matches")
            
            # Manual filtering by user_id and project_id, plus exact text matching
            for obj in results.objects:
                # Filter by user/project
                if (obj.properties.get('user_id') != user_id or 
                    obj.properties.get('project_id') != project_id):
                    continue
                    
                file_path = obj.properties['file_path']
                content = obj.properties['content']
                
                # NUCLEAR FILTER: Block .import files from search results
                if file_path.endswith('.import'):
                    print(f"WeaviateKeyword: BLOCKING old junk from keyword results: {file_path}")
                    continue
                
                # For keyword search, ensure the term actually exists in content
                if query.lower() in content.lower():
                    exact_matches += 1
                    
                    # Skip if we already have a result from this file
                    if file_path in seen_files:
                        continue
                    
                    seen_files.add(file_path)
                    
                    formatted_results.append({
                        'file_path': file_path,
                        'content': content,
                        'chunk_index': obj.properties.get('chunk_index', 0),
                        'chunk_start': obj.properties.get('chunk_start', 1),
                        'chunk_end': obj.properties.get('chunk_end', 1),
                        'similarity': 1.0,  # Keyword match = perfect relevance
                        'search_type': 'keyword'
                    })
                    
                    if len(formatted_results) >= max_results:
                        break
            
            print(f"WeaviateKeyword: {exact_matches} exact matches found, returning {len(formatted_results)} results")
            return formatted_results[:max_results]
            
        except Exception as e:
            print(f"WeaviateKeyword: Keyword search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, user_id: str, project_id: str, max_results: int = 5) -> List[Dict]:
        """Hybrid search combining semantic similarity with exact keyword matching"""
        try:
            collection = self.client.collections.get("ProjectEmbedding")
            
            # Get semantic results
            semantic_results = self.search(query, user_id, project_id, max_results)
            
            # Get keyword results using BM25/text search
            try:
                keyword_results = collection.query.bm25(
                    query=query,
                    limit=max_results,
                    return_properties=["file_path", "content", "chunk_index", 
                                     "chunk_start", "chunk_end", "file_type"],
                    where=Filter.by_property("user_id").equal(user_id) & 
                          Filter.by_property("project_id").equal(project_id)
                )
                
                # Format keyword results
                keyword_formatted = []
                for obj in keyword_results.objects:
                    keyword_formatted.append({
                        'file_path': obj.properties['file_path'],
                        'content': obj.properties['content'],
                        'similarity': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.8,  # BM25 score
                        'search_type': 'keyword',
                        'chunk': {
                            'chunk_index': obj.properties['chunk_index'],
                            'start_line': obj.properties['chunk_start'],
                            'end_line': obj.properties['chunk_end'],
                            'content': obj.properties['content'],
                        },
                        'file_type': obj.properties['file_type'],
                    })
                
            except Exception as e:
                print(f"WeaviateVector: Keyword search failed: {e}")
                keyword_formatted = []
            
            # Merge and deduplicate results
            all_results = {}  # file_path -> best_result
            
            # Add semantic results (higher priority)
            for result in semantic_results:
                file_path = result['file_path']
                # Block .import files from hybrid search too
                if file_path.endswith('.import'):
                    continue
                result['search_type'] = 'semantic'
                all_results[file_path] = result
            
            # Add keyword results if not already present or if they have higher score
            for result in keyword_formatted:
                file_path = result['file_path']
                # Block .import files from hybrid search too
                if file_path.endswith('.import'):
                    continue
                if file_path not in all_results or result['similarity'] > all_results[file_path]['similarity']:
                    all_results[file_path] = result
            
            # Sort by similarity and limit
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return final_results[:max_results]
            
        except Exception as e:
            print(f"WeaviateVector: Hybrid search error: {e}")
            # Fallback to semantic search
            return self.search(query, user_id, project_id, max_results)
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        self.client.close()
