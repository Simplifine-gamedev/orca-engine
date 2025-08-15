"""
Embedding Manager for Godot Project Files
Handles file indexing, embedding generation, and storage in GCP Cloud Storage
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import faiss
from google.cloud import storage
from openai import OpenAI

@dataclass
class FileEmbedding:
    """Represents an embedded file with metadata"""
    file_path: str
    content_hash: str
    embedding: List[float]
    file_type: str
    last_modified: float
    indexed_at: float
    size: int
    
class EmbeddingManager:
    """Manages embeddings for Godot project files with GCP storage"""
    
    def __init__(self, openai_client: OpenAI, gcp_bucket_name: str, project_root: str):
        self.openai_client = openai_client
        self.gcp_bucket_name = gcp_bucket_name
        self.project_root = project_root
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcp_bucket_name)
        
        # In-memory embedding store
        self.embeddings: Dict[str, FileEmbedding] = {}
        self.embedding_dimension = 1536  # OpenAI text-embedding-3-small dimension
        
        # FAISS index for fast similarity search
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.file_paths_list: List[str] = []  # Maps FAISS indices to file paths
        
        # File types to index
        self.indexable_extensions = {
            '.gd', '.cs', '.js',  # Scripts
            '.tscn', '.scn',      # Scenes
            '.tres', '.res',      # Resources
            '.json', '.cfg',      # Config files
            '.md', '.txt'         # Documentation
        }
        
        self._load_existing_embeddings()
    
    def _get_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Tuple[str, float, int]:
        """Get file content, modification time, and size"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            stat = os.stat(file_path)
            return content, stat.st_mtime, stat.st_size
        except (UnicodeDecodeError, FileNotFoundError):
            return "", 0, 0
    
    def _should_index_file(self, file_path: str) -> bool:
        """Check if file should be indexed based on extension and other criteria"""
        if not os.path.exists(file_path):
            return False
        
        # Check extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in self.indexable_extensions:
            return False
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.split(os.sep)):
            return False
        
        # Skip build/temp directories
        skip_dirs = {'build', 'temp', '__pycache__', '.godot', '.import'}
        if any(skip_dir in file_path for skip_dir in skip_dirs):
            return False
        
        return True
    
    def _generate_embedding(self, content: str, file_path: str) -> Optional[List[float]]:
        """Generate embedding for file content using OpenAI"""
        try:
            # Prepare content for embedding
            # Include file path context and content
            embedding_text = f"File: {file_path}\n\nContent:\n{content}"
            
            # Truncate if too long (OpenAI has token limits)
            if len(embedding_text) > 8000:  # Conservative limit
                embedding_text = embedding_text[:8000] + "...[truncated]"
            
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=embedding_text
            )
            
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for {file_path}: {e}")
            return None
    
    def _save_embeddings_to_gcp(self):
        """Save embeddings index to GCP Cloud Storage"""
        try:
            # Prepare data for storage
            embeddings_data = {
                'embeddings': {path: asdict(emb) for path, emb in self.embeddings.items()},
                'timestamp': time.time(),
                'project_root': self.project_root
            }
            
            # Upload to GCP
            blob_name = f"godot_embeddings/{self._get_project_hash()}/embeddings.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(json.dumps(embeddings_data), content_type='application/json')
            
            print(f"Saved {len(self.embeddings)} embeddings to GCP: {blob_name}")
        except Exception as e:
            print(f"Error saving embeddings to GCP: {e}")
    
    def _load_existing_embeddings(self):
        """Load existing embeddings from GCP Cloud Storage"""
        try:
            blob_name = f"godot_embeddings/{self._get_project_hash()}/embeddings.json"
            blob = self.bucket.blob(blob_name)
            
            if blob.exists():
                data = json.loads(blob.download_as_text())
                
                # Reconstruct embeddings
                for path, emb_dict in data.get('embeddings', {}).items():
                    self.embeddings[path] = FileEmbedding(**emb_dict)
                
                # Rebuild FAISS index
                self._rebuild_faiss_index()
                
                print(f"Loaded {len(self.embeddings)} embeddings from GCP")
            else:
                print("No existing embeddings found in GCP")
        except Exception as e:
            print(f"Error loading embeddings from GCP: {e}")
    
    def _get_project_hash(self) -> str:
        """Generate unique hash for project (based on root path)"""
        return hashlib.sha256(self.project_root.encode('utf-8')).hexdigest()[:16]
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from current embeddings"""
        if not self.embeddings:
            return
        
        # Clear existing index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.file_paths_list = []
        
        # Add all embeddings to index
        embeddings_matrix = []
        for path, emb in self.embeddings.items():
            embeddings_matrix.append(np.array(emb.embedding, dtype=np.float32))
            self.file_paths_list.append(path)
        
        if embeddings_matrix:
            embeddings_array = np.vstack(embeddings_matrix)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.faiss_index.add(embeddings_array)
    
    def index_file(self, file_path: str) -> bool:
        """Index a single file, return True if indexed/updated"""
        if not self._should_index_file(file_path):
            return False
        
        # Get file metadata
        content, mtime, size = self._get_file_metadata(file_path)
        if not content:
            return False
        
        content_hash = self._get_content_hash(content)
        
        # Check if file needs updating
        existing = self.embeddings.get(file_path)
        if existing and existing.content_hash == content_hash:
            return False  # No changes needed
        
        # Generate embedding
        embedding = self._generate_embedding(content, file_path)
        if not embedding:
            return False
        
        # Store embedding
        file_embedding = FileEmbedding(
            file_path=file_path,
            content_hash=content_hash,
            embedding=embedding,
            file_type=os.path.splitext(file_path)[1],
            last_modified=mtime,
            indexed_at=time.time(),
            size=size
        )
        
        self.embeddings[file_path] = file_embedding
        
        # Update FAISS index
        self._rebuild_faiss_index()
        
        print(f"Indexed: {file_path}")
        return True
    
    def index_project(self, force_reindex: bool = False) -> Dict[str, int]:
        """Index all files in the project"""
        stats = {'indexed': 0, 'skipped': 0, 'errors': 0}
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'build', 'temp', '__pycache__'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if force_reindex:
                        # Remove existing embedding if force reindex
                        self.embeddings.pop(file_path, None)
                    
                    if self.index_file(file_path):
                        stats['indexed'] += 1
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")
                    stats['errors'] += 1
        
        # Save to GCP after batch indexing
        if stats['indexed'] > 0:
            self._save_embeddings_to_gcp()
        
        return stats
    
    def remove_file(self, file_path: str) -> bool:
        """Remove file from index"""
        if file_path in self.embeddings:
            del self.embeddings[file_path]
            self._rebuild_faiss_index()
            self._save_embeddings_to_gcp()
            return True
        return False
    
    def search_similar_files(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for files similar to query text"""
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query, "query")
        if not query_embedding:
            return []
        
        # Search using FAISS
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.faiss_index.search(query_vector, min(k, len(self.file_paths_list)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.file_paths_list):
                file_path = self.file_paths_list[idx]
                results.append((file_path, float(score)))
        
        return results
    
    def get_project_summary(self) -> Dict:
        """Get summary of indexed project"""
        if not self.embeddings:
            return {'total_files': 0, 'file_types': {}, 'total_size': 0}
        
        file_types = {}
        total_size = 0
        
        for emb in self.embeddings.values():
            file_type = emb.file_type
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_size += emb.size
        
        return {
            'total_files': len(self.embeddings),
            'file_types': file_types,
            'total_size': total_size,
            'last_updated': max(emb.indexed_at for emb in self.embeddings.values()) if self.embeddings else 0
        }
    
    def update_file(self, file_path: str) -> bool:
        """Update embedding for a specific file (called on file change)"""
        return self.index_file(file_path)