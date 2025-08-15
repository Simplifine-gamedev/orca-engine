import os
import json
import hashlib
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import re
import threading
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    file_path: str
    chunk_index: int
    content: str
    start_line: int
    end_line: int
    embedding: Optional[List[float]] = None
    project_id: str = None
    user_id: str = None

class CloudVectorManager:
    """Cloud-based vector embedding manager using BigQuery and Vertex AI"""
    
    # File extensions to index
    TEXT_EXTENSIONS = {
        # Code files
        '.gd', '.cs', '.cpp', '.h', '.hpp', '.c',
        # Godot native project/resource/scene formats
        '.tscn', '.scn', '.tres', '.res', '.godot', '.import', '.gdns', '.gdnlib', '.gdextension',
        # Config and metadata
        '.cfg', '.ini', '.json', '.xml', '.yaml', '.yml',
        # Documentation
        '.md', '.txt', '.rst',
        # Shaders
        '.shader', '.gdshader', '.glsl'
    }
    
    # Extensions to skip entirely
    SKIP_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg',  # Images
        '.ogg', '.mp3', '.wav', '.opus',                  # Audio
        '.mp4', '.webm', '.ogv',                          # Video
        '.ttf', '.otf', '.woff', '.woff2',               # Fonts
        '.zip', '.tar', '.gz', '.rar',                   # Archives
        '.exe', '.dll', '.so', '.dylib',                 # Binaries
        '.pck', '.pak',                                   # Godot packages
        '.uid'                                           # Godot sidecar UID files (never index)
    }
    
    # Patterns for files that tend to be huge and not useful for search
    SKIP_PATTERNS = [
        r'\.import$',                    # Import files
        r'\.tmp$',                       # Temp files
        r'node_modules/',                # Node modules
        r'\.godot/',                     # Godot cache
        r'__pycache__/',                # Python cache
        r'build/',                       # Build directories
        r'dist/',                        # Distribution directories
    ]
    
    def __init__(self, gcp_project_id: str, openai_client, dataset_id: str = 'godot_embeddings'):
        self.gcp_project_id = gcp_project_id
        self.dataset_id = dataset_id
        self.table_id = 'embeddings'
        self.openai_client = openai_client
        
        if not openai_client:
            raise ValueError("OpenAI client is required")
        
        print("Using OpenAI embeddings (text-embedding-3-small)")
        # Tunables
        self.embed_batch_size = int(os.getenv('EMBED_BATCH_SIZE', '100'))
        self.max_embed_parallel = int(os.getenv('EMBED_MAX_PARALLEL', '4'))
        # Concurrency guard for outbound embedding calls
        self._embed_semaphore = threading.Semaphore(self.max_embed_parallel)
        # Default worker pool size for indexing
        default_workers = (os.cpu_count() or 4) * 2
        self.default_index_workers = int(os.getenv('INDEX_MAX_WORKERS', str(min(32, default_workers))))
        
        # Initialize BigQuery client
        self.bq_client = bigquery.Client(project=gcp_project_id)
        
        # Create dataset and table if they don't exist
        self._init_bigquery_schema()
    
    def _init_bigquery_schema(self):
        """Initialize BigQuery dataset and table with vector support"""
        # Create dataset if it doesn't exist
        dataset_ref = f"{self.gcp_project_id}.{self.dataset_id}"
        try:
            self.bq_client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = self.bq_client.create_dataset(dataset)
            print(f"Created dataset {self.dataset_id}")
        
        # Create embeddings table with vector column
        table_ref = f"{dataset_ref}.{self.table_id}"
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("project_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("file_path", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("start_line", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("end_line", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("file_hash", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("indexed_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),  # 1536-dim vector for OpenAI
        ]
        
        try:
            table = self.bq_client.get_table(table_ref)
            print(f"Table {self.table_id} already exists")
        except NotFound:
            table = bigquery.Table(table_ref, schema=schema)
            table = self.bq_client.create_table(table)
            print(f"Created table {self.table_id}")
            
            # Create index for faster queries
            self._create_indexes()

        # Create graph tables (nodes, edges)
        # Nodes (File, SceneNode, Resource, Class etc.)
        nodes_table_ref = f"{dataset_ref}.graph_nodes"
        try:
            self.bq_client.get_table(nodes_table_ref)
        except NotFound:
            node_schema = [
                bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("project_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("file_path", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("kind", "STRING", mode="REQUIRED"),  # SceneNode | File | Resource | Class
                bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("node_type", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("node_path", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("start_line", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("end_line", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ]
            self.bq_client.create_table(bigquery.Table(nodes_table_ref, schema=node_schema))

        # Edges
        edges_table_ref = f"{dataset_ref}.graph_edges"
        try:
            self.bq_client.get_table(edges_table_ref)
        except NotFound:
            edge_schema = [
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("project_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("src_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("dst_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("kind", "STRING", mode="REQUIRED"),  # CHILD_OF | ATTACHES_SCRIPT | USES_RESOURCE | ...
                bigquery.SchemaField("file_path", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("start_line", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("end_line", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ]
            self.bq_client.create_table(bigquery.Table(edges_table_ref, schema=edge_schema))
    
    def _create_indexes(self):
        """Create indexes for performance"""
        # BigQuery doesn't use traditional indexes, it uses clustering and partitioning
        # The table is automatically optimized for queries on user_id and project_id
        print("Table created with automatic optimization for user_id and project_id")
    
    def _should_index_file(self, file_path: str) -> bool:
        """Check if file should be indexed"""
        # Skip hidden files
        if os.path.basename(file_path).startswith('.'):
            return False
        
        # Check extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.SKIP_EXTENSIONS:
            return False
        if self.TEXT_EXTENSIONS and ext not in self.TEXT_EXTENSIONS:
            return False
        
        # Check skip patterns
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, file_path):
                return False
        
        # Skip very large files (>10MB)
        try:
            if os.path.getsize(file_path) > 10 * 1024 * 1024:
                return False
        except:
            return False
        
        return True
    
    def _get_file_hash(self, content: str) -> str:
        """Get hash of file content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _chunk_text(self, content: str, file_path: str, max_lines: int = 50) -> List[TextChunk]:
        """Smart chunking for code files"""
        chunks = []
        lines = content.split('\n')
        
        # For small files, return as single chunk
        if len(lines) <= max_lines:
            return [TextChunk(
                file_path=file_path,
                chunk_index=0,
                content=content,
                start_line=1,
                end_line=len(lines)
            )]
        
        # For Godot scene/resource files, chunk by sections
        if file_path.endswith(('.tscn', '.tres')):
            return self._chunk_godot_resource(content, file_path)
        
        # For code files, chunk intelligently by functions/classes
        if file_path.endswith(('.gd', '.cs', '.cpp', '.h')):
            return self._chunk_code_file(content, file_path, max_lines)
        
        # Default: chunk by lines with overlap
        chunk_size = max_lines
        overlap = 10
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            chunks.append(TextChunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content='\n'.join(chunk_lines),
                start_line=i + 1,
                end_line=min(i + len(chunk_lines), len(lines))
            ))
        
        return chunks
    
    def _chunk_godot_resource(self, content: str, file_path: str) -> List[TextChunk]:
        """Chunk Godot resource files by sections"""
        chunks = []
        lines = content.split('\n')
        current_section = []
        current_start = 1
        
        for i, line in enumerate(lines):
            # New section starts with [
            if line.strip().startswith('[') and current_section:
                # Save previous section
                chunks.append(TextChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content='\n'.join(current_section),
                    start_line=current_start,
                    end_line=i
                ))
                current_section = [line]
                current_start = i + 1
            else:
                current_section.append(line)
        
        # Don't forget last section
        if current_section:
            chunks.append(TextChunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content='\n'.join(current_section),
                start_line=current_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _chunk_code_file(self, content: str, file_path: str, max_lines: int) -> List[TextChunk]:
        """Chunk code files by functions/classes"""
        chunks = []
        lines = content.split('\n')
        
        # Pattern for function/class definitions
        if file_path.endswith('.gd'):
            # GDScript patterns
            def_pattern = r'^(func\s+\w+|class\s+\w+|signal\s+\w+|extends\s+)'
        elif file_path.endswith(('.cpp', '.h', '.cs')):
            # C++/C# patterns
            def_pattern = r'^(class\s+\w+|struct\s+\w+|\w+\s+\w+\s*\(|public\s+|private\s+|protected\s+)'
        else:
            # Default to line-based chunking
            return self._chunk_text(content, file_path, max_lines)
        
        current_chunk = []
        current_start = 1
        
        for i, line in enumerate(lines):
            if re.match(def_pattern, line.strip()) and current_chunk and len(current_chunk) > 5:
                # Save current chunk
                chunks.append(TextChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content='\n'.join(current_chunk),
                    start_line=current_start,
                    end_line=i
                ))
                current_chunk = [line]
                current_start = i + 1
            else:
                current_chunk.append(line)
                
                # Force new chunk if too large
                if len(current_chunk) >= max_lines:
                    chunks.append(TextChunk(
                        file_path=file_path,
                        chunk_index=len(chunks),
                        content='\n'.join(current_chunk),
                        start_line=current_start,
                        end_line=i + 1
                    ))
                    current_chunk = []
                    current_start = i + 2
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(TextChunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content='\n'.join(current_chunk),
                start_line=current_start,
                end_line=len(lines)
            ))
        
        return chunks
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI with batching, throttling, and retries.

        Note: If a batch fails after retries, its items are skipped and will reduce the
        number of returned embeddings. Callers should not assume 1:1 length on failure.
        """
        if not texts:
            return []
        batch_size = max(1, self.embed_batch_size)
        all_embeddings: List[List[float]] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate long texts to protect against token limits
            batch = [t[:8000] if len(t) > 8000 else t for t in batch]

            # Throttle concurrent batches
            with self._embed_semaphore:
                # Basic retry with exponential backoff and jitter
                max_attempts = 3
                attempt = 0
                while True:
                    try:
                        response = self.openai_client.embeddings.create(
                            model="text-embedding-3-small",
                            input=batch
                        )
                        for data in response.data:
                            all_embeddings.append(data.embedding)
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            print(f"Error generating embeddings (final): {e}")
                            break
                        sleep_s = (2 ** attempt) + random.uniform(0, 0.25)
                        print(f"Embedding batch failed (attempt {attempt}), retrying in {sleep_s:.2f}s: {e}")
                        time.sleep(sleep_s)
        
        return all_embeddings
    
    def index_file(self, file_path: str, user_id: str, project_id: str, project_root: str) -> bool:
        """Index a single file"""
        if not self._should_index_file(file_path):
            return False
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get relative path from project root
            rel_path = os.path.relpath(file_path, project_root)
            
            # Check if already indexed with same content
            file_hash = self._get_file_hash(content)
            
            # Query to check existing
            query = f"""
            SELECT file_hash 
            FROM `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}`
            WHERE user_id = @user_id 
              AND project_id = @project_id 
              AND file_path = @file_path
            LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                    bigquery.ScalarQueryParameter("file_path", "STRING", rel_path),
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config).result())
            if results and results[0].file_hash == file_hash:
                return False  # Already up to date
            
            # Delete old entries (may fail while rows are in the streaming buffer)
            try:
                delete_query = f"""
                DELETE FROM `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}`
                WHERE user_id = @user_id 
                  AND project_id = @project_id 
                  AND file_path = @file_path
                """
                self.bq_client.query(delete_query, job_config=job_config).result()
            except Exception as del_err:
                if "streaming buffer" in str(del_err):
                    print(f"CloudVector: Skipping delete due to streaming buffer for {rel_path}")
                else:
                    print(f"CloudVector: Delete failed for {rel_path}: {del_err}")
            
            # Chunk the content
            chunks = self._chunk_text(content, rel_path)
            
            # Generate embeddings for all chunks
            texts = [f"File: {os.path.basename(rel_path)}\n\n{chunk.content}" for chunk in chunks]
            embeddings = self._generate_embeddings_batch(texts)
            
            # Only proceed if we got embeddings
            if not embeddings:
                print(f"Failed to generate embeddings for {rel_path}")
                return False
            
            # Prepare rows for BigQuery - only for successful embeddings
            rows_to_insert = []
            for i, embedding in enumerate(embeddings):
                if i < len(chunks):  # Make sure we have a corresponding chunk
                    chunk = chunks[i]
                    row_id = f"{user_id}:{project_id}:{rel_path}:{chunk.chunk_index}"
                    rows_to_insert.append({
                        "id": hashlib.md5(row_id.encode()).hexdigest(),
                        "user_id": user_id,
                        "project_id": project_id,
                        "file_path": rel_path,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content[:10000],  # Limit content size
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "file_hash": file_hash,
                        "indexed_at": time.time(),
                        "embedding": embedding
                    })
            
            # Insert rows
            if rows_to_insert:
                table_ref = f"{self.gcp_project_id}.{self.dataset_id}.{self.table_id}"
                errors = self.bq_client.insert_rows_json(table_ref, rows_to_insert)
                if errors:
                    print(f"Error inserting rows: {errors}")
                    return False

            # Upsert File node for this file
            self._upsert_file_node(rel_path, user_id, project_id)

            # Update graph for scenes/resources
            if rel_path.endswith('.tscn') or rel_path.endswith('.tres'):
                self._upsert_scene_graph(rel_path, content, user_id, project_id)
            
            return True
            
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return False
    
    def index_project(self, project_root: str, user_id: str, project_id: str, force_reindex: bool = False, max_workers: Optional[int] = None) -> Dict[str, int]:
        """Index all files in project, using parallel workers.

        max_workers controls parallelism (defaults to INDEX_MAX_WORKERS env or CPU heuristic).
        """
        stats = {
            'total': 0,
            'indexed': 0,
            'skipped': 0,
            'failed': 0,
            'removed': 0
        }
        files_to_index: List[str] = []
        current_rel_paths: Set[str] = set()

        # First pass: enumerate files and prefilter
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                file_path = os.path.join(root, file)
                stats['total'] += 1
                if not self._should_index_file(file_path):
                    stats['skipped'] += 1
                    continue
                try:
                    rel_path = os.path.relpath(file_path, project_root)
                except Exception:
                    rel_path = file
                current_rel_paths.add(rel_path)
                if force_reindex:
                    try:
                        self._remove_existing_file_chunks(user_id, project_id, rel_path)
                    except Exception:
                        pass
                files_to_index.append(file_path)

        # Second pass: parallel indexing of files
        if files_to_index:
            workers = max_workers or self.default_index_workers
            indexed_count = 0
            skipped_or_failed = 0

            def _worker(fp: str) -> bool:
                try:
                    return self.index_file(fp, user_id, project_id, project_root)
                except Exception as e:
                    print(f"CloudVector: Exception indexing {fp}: {e}")
                    return False

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_worker, fp) for fp in files_to_index]
                for fut in as_completed(futures):
                    ok = fut.result()
                    if ok:
                        indexed_count += 1
                    else:
                        skipped_or_failed += 1

            stats['indexed'] += indexed_count
            stats['skipped'] += skipped_or_failed

        # Cleanup: remove entries for files no longer present
        try:
            removed_count = self._remove_missing_files(user_id, project_id, current_rel_paths)
            stats['removed'] = removed_count
        except Exception as e:
            print(f"CloudVector: Error during removal of missing files: {e}")

        return stats

    def _remove_missing_files(self, user_id: str, project_id: str, current_rel_paths: Set[str]) -> int:
        """Remove any indexed files that are no longer present in the project.

        Returns number of file entries (distinct paths) removed.
        """
        try:
            # Fetch currently indexed distinct file paths for this project
            table_ref = f"{self.gcp_project_id}.{self.dataset_id}.{self.table_id}"
            list_query = f"""
            SELECT DISTINCT file_path
            FROM `{table_ref}`
            WHERE user_id = @user_id AND project_id = @project_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                ]
            )
            rows = list(self.bq_client.query(list_query, job_config=job_config).result())
            indexed_paths = {row.file_path for row in rows}

            to_delete = list(indexed_paths.difference(current_rel_paths))
            if not to_delete:
                return 0

            # Delete in batches to avoid parameter limits
            batch_size = 1000
            total_removed = 0
            for i in range(0, len(to_delete), batch_size):
                batch = to_delete[i:i + batch_size]
                delete_query = f"""
                DELETE FROM `{table_ref}`
                WHERE user_id = @user_id AND project_id = @project_id
                  AND file_path IN UNNEST(@paths)
                """
                del_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                        bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                        bigquery.ArrayQueryParameter("paths", "STRING", batch),
                    ]
                )
                try:
                    self.bq_client.query(delete_query, job_config=del_config).result()
                    total_removed += len(batch)
                except Exception as del_err:
                    if "streaming buffer" in str(del_err):
                        print("CloudVector: Skipping removal batch due to streaming buffer")
                    else:
                        print(f"CloudVector: Removal batch failed: {del_err}")

            return total_removed
        except Exception as e:
            print(f"CloudVector: Failed to remove missing files: {e}")
            return 0

    def remove_file(self, user_id: str, project_id: str, file_path: str):
        """Remove all chunks for a specific file from the index."""
        try:
            table_ref = f"{self.gcp_project_id}.{self.dataset_id}.{self.table_id}"
            delete_query = f"""
            DELETE FROM `{table_ref}`
            WHERE user_id = @user_id AND project_id = @project_id AND file_path = @file_path
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                    bigquery.ScalarQueryParameter("file_path", "STRING", file_path),
                ]
            )
            self.bq_client.query(delete_query, job_config=job_config).result()
            return True
        except Exception as e:
            print(f"CloudVector: Error removing file {file_path}: {e}")
            return False
    
    def search(self, query: str, user_id: str, project_id: str, max_results: int = 10) -> List[Dict]:
        """Search for similar content using BigQuery vector search"""
        # Generate embedding for query
        query_embeddings = self._generate_embeddings_batch([query])
        if not query_embeddings:
            print("Failed to generate query embedding")
            return []
        
        query_embedding = query_embeddings[0]
        
        # Use BigQuery's vector similarity search
        # Note: BigQuery uses COSINE_DISTANCE, so smaller values = more similar
        search_query = f"""
        WITH query_embedding AS (
            SELECT {query_embedding} AS embedding
        ), ranked AS (
            SELECT 
                e.file_path,
                e.chunk_index,
                e.start_line,
                e.end_line,
                e.content,
                (1 - COSINE_DISTANCE(e.embedding, q.embedding)) AS similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY e.user_id, e.project_id, e.file_path, e.chunk_index
                    ORDER BY e.indexed_at DESC
                ) AS rn
            FROM `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}` e, query_embedding q
            WHERE e.user_id = @user_id 
              AND e.project_id = @project_id
              AND ARRAY_LENGTH(e.embedding) > 0
        )
        SELECT file_path, chunk_index, start_line, end_line, content, similarity
        FROM ranked
        WHERE rn = 1 AND NOT ENDS_WITH(file_path, '.uid')
        ORDER BY similarity DESC
        LIMIT @max_results
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                bigquery.ScalarQueryParameter("max_results", "INTEGER", max_results),
            ]
        )
        
        results = []
        try:
            for row in self.bq_client.query(search_query, job_config=job_config).result():
                results.append({
                    'file_path': row.file_path,
                    'similarity': float(row.similarity),
                    'chunk': {
                        'chunk_index': row.chunk_index,
                        'start_line': row.start_line,
                        'end_line': row.end_line
                    },
                    'content_preview': row.content[:200] + "..." if len(row.content) > 200 else row.content
                })
        except Exception as e:
            print(f"Search query failed: {e}")
            return []
        
        return results

    def get_graph_context_for_files(self, file_paths: List[str], user_id: str, project_id: str, nodes_limit: int = 20, edges_limit: int = 50) -> Dict[str, Dict]:
        """Return lightweight graph context (nodes/edges) for each file.

        Limits results per file to keep payload small.
        """
        context: Dict[str, Dict] = {}
        try:
            dataset_ref = f"{self.gcp_project_id}.{self.dataset_id}"
            nodes_table = f"{dataset_ref}.graph_nodes"
            edges_table = f"{dataset_ref}.graph_edges"

            for fp in file_paths:
                # Nodes
                q_nodes = f"""
                SELECT id, kind, name, node_type, node_path, start_line, end_line
                FROM `{nodes_table}`
                WHERE user_id = @user_id AND project_id = @project_id AND file_path = @file_path
                LIMIT @limit_nodes
                """
                cfg_nodes = bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                    bigquery.ScalarQueryParameter("file_path", "STRING", fp),
                    bigquery.ScalarQueryParameter("limit_nodes", "INTEGER", nodes_limit),
                ])
                node_rows = list(self.bq_client.query(q_nodes, job_config=cfg_nodes).result())
                nodes = []
                for r in node_rows:
                    nodes.append({
                        'id': r.id,
                        'kind': r.kind,
                        'name': r.name,
                        'node_type': r.node_type,
                        'node_path': r.node_path,
                        'start_line': r.start_line,
                        'end_line': r.end_line,
                    })

                # Edges
                q_edges = f"""
                SELECT src_id, dst_id, kind, start_line, end_line
                FROM `{edges_table}`
                WHERE user_id = @user_id AND project_id = @project_id AND file_path = @file_path
                LIMIT @limit_edges
                """
                cfg_edges = bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                    bigquery.ScalarQueryParameter("file_path", "STRING", fp),
                    bigquery.ScalarQueryParameter("limit_edges", "INTEGER", edges_limit),
                ])
                edge_rows = list(self.bq_client.query(q_edges, job_config=cfg_edges).result())
                edges = []
                for r in edge_rows:
                    edges.append({
                        'src_id': r.src_id,
                        'dst_id': r.dst_id,
                        'kind': r.kind,
                        'start_line': r.start_line,
                        'end_line': r.end_line,
                    })

                context[fp] = {'nodes': nodes, 'edges': edges}
        except Exception as e:
            print(f"CloudVector: get_graph_context_for_files error: {e}")
        return context
    
    def get_stats(self, user_id: str, project_id: str) -> Dict:
        """Get indexing statistics"""
        query = f"""
        SELECT 
            COUNT(DISTINCT file_path) as files_indexed,
            COUNT(*) as total_chunks,
            MAX(indexed_at) as last_indexed
        FROM 
            `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}`
        WHERE 
            user_id = @user_id 
            AND project_id = @project_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            ]
        )
        
        result = list(self.bq_client.query(query, job_config=job_config).result())[0]
        
        return {
            'files_indexed': result.files_indexed or 0,
            'total_chunks': result.total_chunks or 0,
            'last_indexed': result.last_indexed.timestamp() if result.last_indexed else None,
            'storage': 'BigQuery',
            'embedding_model': 'text-embedding-3-small'
        }
    
    def clear_project(self, user_id: str, project_id: str):
        """Clear all indexed data for a project"""
        query = f"""
        DELETE FROM `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}`
        WHERE user_id = @user_id AND project_id = @project_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            ]
        )
        
        self.bq_client.query(query, job_config=job_config).result()
    
    def index_files_with_content(self, files_data: List[Dict], user_id: str, project_id: str, max_workers: Optional[int] = None) -> Dict:
        """Index files using provided content, processed in parallel per file."""
        print(f"CloudVector: Indexing {len(files_data)} files with provided content")

        indexed_files = 0
        skipped_files = 0
        failed_files = 0

        # Worker to process a single file_data dict
        def _process_file(fd: Dict) -> Tuple[str, int, int, int]:
            try:
                file_path = fd.get('path', '')
                content = fd.get('content', '')
                file_hash = fd.get('hash', '')

                # Never index .uid sidecar
                if file_path.endswith('.uid'):
                    print(f"CloudVector: Skipping .uid sidecar file: {file_path}")
                    return (file_path, 0, 1, 0)

                if not file_path or not content:
                    print(f"CloudVector: Skipping file with missing path or content")
                    return (file_path or 'unknown', 0, 1, 0)

                # Skip unchanged
                if self._is_file_already_indexed(user_id, project_id, file_path, file_hash):
                    print(f"CloudVector: Skipping unchanged file: {file_path}")
                    return (file_path, 0, 1, 0)

                # Remove existing entries for this file (best effort)
                try:
                    self._remove_existing_file_chunks(user_id, project_id, file_path)
                except Exception:
                    pass

                # Chunk
                chunks = self._create_text_chunks(content, file_path)
                if not chunks:
                    print(f"CloudVector: No chunks created for {file_path}")
                    return (file_path, 0, 1, 0)

                # Embed
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = self._generate_embeddings_batch(chunk_texts)
                if len(embeddings) != len(chunks):
                    print(f"CloudVector: Embedding count mismatch for {file_path} ({len(embeddings)} vs {len(chunks)})")
                    return (file_path, 0, 0, 1)

                # Insert
                rows_to_insert = []
                current_time = time.time()
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    row = {
                        'id': f"{user_id}:{project_id}:{file_path}:{i}",
                        'user_id': user_id,
                        'project_id': project_id,
                        'file_path': file_path,
                        'chunk_index': i,
                        'content': chunk.content,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'file_hash': file_hash,
                        'indexed_at': current_time,
                        'embedding': embedding
                    }
                    rows_to_insert.append(row)
                table_ref = f"{self.gcp_project_id}.{self.dataset_id}.{self.table_id}"
                table = self.bq_client.get_table(table_ref)
                errors = self.bq_client.insert_rows_json(table, rows_to_insert)
                if errors:
                    print(f"CloudVector: Error inserting {file_path}: {errors}")
                    return (file_path, 0, 0, 1)

                # Upserts
                self._upsert_file_node(file_path, user_id, project_id)
                if file_path.endswith('.tscn') or file_path.endswith('.tres'):
                    self._upsert_scene_graph(file_path, content, user_id, project_id)

                print(f"CloudVector: ✅ Indexed {file_path} ({len(chunks)} chunks)")
                return (file_path, 1, 0, 0)
            except Exception as e:
                print(f"CloudVector: Error processing file {fd.get('path', 'unknown')}: {e}")
                return (fd.get('path', 'unknown'), 0, 0, 1)

        workers = max_workers or self.default_index_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_file, fd) for fd in files_data]
            for fut in as_completed(futures):
                _fp, idx, skp, fld = fut.result()
                indexed_files += idx
                skipped_files += skp
                failed_files += fld

        print(f"CloudVector: Batch complete - indexed: {indexed_files}, skipped: {skipped_files}, failed: {failed_files}")

        return {
            'total': len(files_data),
            'indexed': indexed_files,
            'skipped': skipped_files,
            'failed': failed_files
        }

    # Compatibility wrapper for provided-content indexing
    def _create_text_chunks(self, content: str, file_path: str) -> List[TextChunk]:
        """Create text chunks for a given in-memory file content.

        Dispatches to the same chunking logic as on-disk indexing, using
        file path to decide strategy.
        """
        try:
            return self._chunk_text(content, file_path)
        except Exception as e:
            print(f"CloudVector: _create_text_chunks error for {file_path}: {e}")
            # Fallback: one whole-chunk to avoid total failure
            return [TextChunk(
                file_path=file_path,
                chunk_index=0,
                content=content,
                start_line=1,
                end_line=len(content.split('\n'))
            )] if content else []

    def _upsert_scene_graph(self, rel_path: str, content: str, user_id: str, project_id: str):
        """Parse a .tscn/.tres file and upsert nodes/edges into graph tables.

        Minimal MVP: extract [node ...] blocks and parent-child hierarchy; attach script edges.
        """
        try:
            lines = content.split('\n')
            dataset_ref = f"{self.gcp_project_id}.{self.dataset_id}"
            nodes_table = f"{dataset_ref}.graph_nodes"
            edges_table = f"{dataset_ref}.graph_edges"
            now_ts = time.time()

            # First, clear prior graph rows for this file (best effort; ignore streaming buffer)
            for tbl in (nodes_table, edges_table):
                try:
                    q = f"""
                    DELETE FROM `{tbl}`
                    WHERE user_id = @user_id AND project_id = @project_id AND file_path = @file_path
                    """
                    cfg = bigquery.QueryJobConfig(query_parameters=[
                        bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                        bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                        bigquery.ScalarQueryParameter("file_path", "STRING", rel_path),
                    ])
                    self.bq_client.query(q, job_config=cfg).result()
                except Exception as e:
                    if "streaming buffer" in str(e):
                        pass
            # Parse ext_resources first (for PackedScene/Script references)
            ext_res: Dict[str, Dict[str, str]] = {}
            for i, line in enumerate(lines, start=1):
                s = line.strip()
                if s.startswith('[ext_resource'):
                    r_type = self._extract_attr(s, 'type') or ''
                    r_path = self._extract_attr(s, 'path') or ''
                    r_id = self._extract_attr(s, 'id') or ''
                    if r_id:
                        ext_res[r_id] = {'type': r_type, 'path': r_path}

            # Parse nodes
            nodes = []  # (id, name, type, path, start_line, end_line)
            edges = []  # (src_id, dst_id, kind, start_line, end_line)

            current_section_start = None
            current_header = None
            for i, line in enumerate(lines, start=1):
                s = line.strip()
                if s.startswith('[') and s.endswith(']'):
                    # Close previous section
                    if current_header and current_section_start:
                        # Only register node sections
                        if current_header.startswith('[node'):
                            name = self._extract_attr(current_header, 'name') or ''
                            ntype = self._extract_attr(current_header, 'type') or ''
                            parent_attr = self._extract_attr(current_header, 'parent')
                            node_attr = self._extract_attr(current_header, 'node')
                            if node_attr:
                                npath = node_attr
                            elif parent_attr and parent_attr != '.':
                                npath = parent_attr.rstrip('/') + '/' + name
                            else:
                                npath = name
                            node_id = hashlib.md5(f"{project_id}:{rel_path}:{npath}".encode()).hexdigest()
                            nodes.append((node_id, name, ntype, npath, current_section_start, i - 1))
                            if parent_attr and parent_attr != '.':
                                parent_id = hashlib.md5(f"{project_id}:{rel_path}:{parent_attr}".encode()).hexdigest()
                                edges.append((parent_id, node_id, 'CHILD_OF', current_section_start, i - 1))
                            elif '/' in npath:
                                parent_path = npath.rsplit('/', 1)[0]
                                parent_id = hashlib.md5(f"{project_id}:{rel_path}:{parent_path}".encode()).hexdigest()
                                edges.append((parent_id, node_id, 'CHILD_OF', current_section_start, i - 1))
                            inst_id = self._extract_extres_id(current_header)
                            if inst_id and inst_id in ext_res and ext_res[inst_id]['type'] == 'PackedScene':
                                target_scene = ext_res[inst_id]['path']
                                dst_id = hashlib.md5(f"{project_id}:{target_scene}".encode()).hexdigest()
                                edges.append((node_id, dst_id, 'INSTANTIATES_SCENE', current_section_start, i - 1))
                    # Open new section
                    current_header = s
                    current_section_start = i
            # Flush last section
            if current_header and current_section_start and current_header.startswith('[node'):
                name = self._extract_attr(current_header, 'name') or ''
                ntype = self._extract_attr(current_header, 'type') or ''
                parent_attr = self._extract_attr(current_header, 'parent')
                node_attr = self._extract_attr(current_header, 'node')
                if node_attr:
                    npath = node_attr
                elif parent_attr and parent_attr != '.':
                    npath = parent_attr.rstrip('/') + '/' + name
                else:
                    npath = name
                node_id = hashlib.md5(f"{project_id}:{rel_path}:{npath}".encode()).hexdigest()
                nodes.append((node_id, name, ntype, npath, current_section_start, len(lines)))
                if parent_attr and parent_attr != '.':
                    parent_id = hashlib.md5(f"{project_id}:{rel_path}:{parent_attr}".encode()).hexdigest()
                    edges.append((parent_id, node_id, 'CHILD_OF', current_section_start, len(lines)))
                elif '/' in npath:
                    parent_path = npath.rsplit('/', 1)[0]
                    parent_id = hashlib.md5(f"{project_id}:{rel_path}:{parent_path}".encode()).hexdigest()
                    edges.append((parent_id, node_id, 'CHILD_OF', current_section_start, len(lines)))
                inst_id = self._extract_extres_id(current_header)
                if inst_id and inst_id in ext_res and ext_res[inst_id]['type'] == 'PackedScene':
                    target_scene = ext_res[inst_id]['path']
                    dst_id = hashlib.md5(f"{project_id}:{target_scene}".encode()).hexdigest()
                    edges.append((node_id, dst_id, 'INSTANTIATES_SCENE', current_section_start, len(lines)))

            # Attach script edges by scanning lines within node ranges
            for node_id, name, ntype, npath, start_l, end_l in nodes:
                for ln in range(start_l - 1, min(end_l, len(lines))):
                    if 'script =' in lines[ln]:
                        m = re.search(r'"(res://[^\"]+)"', lines[ln])
                        if m:
                            script_path = m.group(1)
                            dst_id = hashlib.md5(f"{project_id}:{script_path}".encode()).hexdigest()
                            edges.append((node_id, dst_id, 'ATTACHES_SCRIPT', start_l, end_l))
                            break
                        m2 = re.search(r'ExtResource\("([^)\"]+)"\)', lines[ln])
                        if m2 and m2.group(1) in ext_res and ext_res[m2.group(1)]['type'] in ('Script', 'GDScript', 'CSharpScript'):
                            script_path = ext_res[m2.group(1)]['path']
                            dst_id = hashlib.md5(f"{project_id}:{script_path}".encode()).hexdigest()
                            edges.append((node_id, dst_id, 'ATTACHES_SCRIPT', start_l, end_l))
                            break

            # Parse connection sections
            for i, line in enumerate(lines, start=1):
                s = line.strip()
                if s.startswith('[connection'):
                    sig = self._extract_attr(s, 'signal') or ''
                    from_p = self._extract_attr(s, 'from') or ''
                    to_p = self._extract_attr(s, 'to') or ''
                    method = self._extract_attr(s, 'method') or ''
                    src_id = hashlib.md5(f"{project_id}:{rel_path}:{from_p}".encode()).hexdigest()
                    dst_id = hashlib.md5(f"{project_id}:{rel_path}:{to_p}".encode()).hexdigest()
                    kind = f"CONNECTS_SIGNAL:{sig}->{method}" if sig or method else "CONNECTS_SIGNAL"
                    edges.append((src_id, dst_id, kind, i, i))

            # Insert nodes
            if nodes:
                table = self.bq_client.get_table(nodes_table)
                node_rows = [{
                    'id': nid,
                    'user_id': user_id,
                    'project_id': project_id,
                    'file_path': rel_path,
                    'kind': 'SceneNode',
                    'name': name,
                    'node_type': ntype,
                    'node_path': npath,
                    'start_line': start_l,
                    'end_line': end_l,
                    'updated_at': now_ts,
                } for (nid, name, ntype, npath, start_l, end_l) in nodes]
                errors = self.bq_client.insert_rows_json(table, node_rows)
                if errors:
                    print(f"CloudVector(Graph): Node upsert errors for {rel_path}: {errors}")

            # Insert edges
            if edges:
                table = self.bq_client.get_table(edges_table)
                edge_rows = [{
                    'user_id': user_id,
                    'project_id': project_id,
                    'src_id': src,
                    'dst_id': dst,
                    'kind': kind,
                    'file_path': rel_path,
                    'start_line': s_l,
                    'end_line': e_l,
                    'updated_at': now_ts,
                } for (src, dst, kind, s_l, e_l) in edges]
                errors = self.bq_client.insert_rows_json(table, edge_rows)
                if errors:
                    print(f"CloudVector(Graph): Edge upsert errors for {rel_path}: {errors}")

            print(f"CloudVector(Graph): Upserted nodes={len(nodes)} edges={len(edges)} for {rel_path}")
        except Exception as e:
            print(f"CloudVector: _upsert_scene_graph error for {rel_path}: {e}")

    def _upsert_file_node(self, file_path: str, user_id: str, project_id: str):
        try:
            dataset_ref = f"{self.gcp_project_id}.{self.dataset_id}"
            nodes_table = f"{dataset_ref}.graph_nodes"
            now_ts = time.time()
            file_id = hashlib.md5(f"{project_id}:{file_path}".encode()).hexdigest()
            row = {
                'id': file_id,
                'user_id': user_id,
                'project_id': project_id,
                'file_path': file_path,
                'kind': 'File',
                'name': os.path.basename(file_path),
                'node_type': None,
                'node_path': None,
                'start_line': None,
                'end_line': None,
                'updated_at': now_ts,
            }
            table = self.bq_client.get_table(nodes_table)
            self.bq_client.insert_rows_json(table, [row])
        except Exception as e:
            print(f"CloudVector(Graph): File node upsert failed for {file_path}: {e}")

    @staticmethod
    def _extract_attr(header: str, key: str) -> Optional[str]:
        m = re.search(key + r'\s*=\s*"([^\"]+)"', header)
        return m.group(1) if m else None
    
    @staticmethod
    def _extract_extres_id(header: str) -> Optional[str]:
        m = re.search(r'instance\s*=\s*ExtResource\("([^\"]+)"\)', header)
        return m.group(1) if m else None
    
    def _is_file_already_indexed(self, user_id: str, project_id: str, file_path: str, file_hash: str) -> bool:
        """Check if file is already indexed with the same hash"""
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}`
        WHERE user_id = @user_id 
          AND project_id = @project_id 
          AND file_path = @file_path
          AND file_hash = @file_hash
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                bigquery.ScalarQueryParameter("file_path", "STRING", file_path),
                bigquery.ScalarQueryParameter("file_hash", "STRING", file_hash),
            ]
        )
        
        result = list(self.bq_client.query(query, job_config=job_config).result())[0]
        return result.count > 0
    
    def _remove_existing_file_chunks(self, user_id: str, project_id: str, file_path: str):
        """Remove existing chunks for a file before re-indexing"""
        query = f"""
        DELETE FROM `{self.gcp_project_id}.{self.dataset_id}.{self.table_id}`
        WHERE user_id = @user_id 
          AND project_id = @project_id 
          AND file_path = @file_path
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                bigquery.ScalarQueryParameter("file_path", "STRING", file_path),
            ]
        )
        try:
            self.bq_client.query(query, job_config=job_config).result()
        except Exception as del_err:
            if "streaming buffer" in str(del_err):
                print(f"CloudVector: Skipping delete for {file_path} due to streaming buffer")
            else:
                raise
