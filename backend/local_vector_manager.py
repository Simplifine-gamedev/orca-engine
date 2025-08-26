"""
© 2025 Simplifine Corp. Personal Non‑Commercial License.
See LICENSES/COMPANY-NONCOMMERCIAL.md for terms.
"""
import os
import hashlib
from typing import List, Dict, Optional


class LocalVectorManager:
    """Lightweight local fallback that stores a minimal in-memory index.

    This is not persistent; it's intended for development when GCP/OpenAI are
    not available. It implements the subset of methods used by the server.
    """

    def __init__(self, openai_client=None):
        # Dependencies optional – avoid hard failures if missing
        try:
            import numpy as np  # noqa: F401
        except Exception:
            pass
        self.index: Dict[str, Dict] = {}

    def index_project(self, project_root: str, user_id: str, project_id: str, force_reindex: bool = False, max_workers: Optional[int] = None) -> Dict[str, int]:
        stats = {"total": 0, "indexed": 0, "skipped": 0, "failed": 0, "removed": 0}
        for root, _, files in os.walk(project_root):
            for fn in files:
                path = os.path.join(root, fn)
                stats["total"] += 1
                if not self._should_index_file(path):
                    stats["skipped"] += 1
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    rel = os.path.relpath(path, project_root)
                    h = hashlib.md5(content.encode("utf-8")).hexdigest()
                    key = f"{user_id}:{project_id}:{rel}"
                    if not force_reindex and key in self.index and self.index[key]["hash"] == h:
                        stats["skipped"] += 1
                        continue
                    self.index[key] = {"file_path": rel, "hash": h, "content_preview": content[:5000]}
                    stats["indexed"] += 1
                except Exception:
                    stats["failed"] += 1
        return stats

    def index_file(self, file_path: str, user_id: str, project_id: str, project_root: str) -> bool:
        if not self._should_index_file(file_path):
            return False
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            rel = os.path.relpath(file_path, project_root)
            key = f"{user_id}:{project_id}:{rel}"
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            if key in self.index and self.index[key]["hash"] == h:
                return False
            self.index[key] = {"file_path": rel, "hash": h, "content_preview": content[:5000]}
            return True
        except Exception:
            return False

    def index_files_with_content(self, files: List[Dict], user_id: str, project_id: str, max_workers: Optional[int] = None) -> Dict:
        indexed = skipped = failed = 0
        for fd in files:
            try:
                rel = fd.get("path")
                content = fd.get("content", "")
                h = fd.get("hash") or hashlib.md5(content.encode("utf-8")).hexdigest()
                key = f"{user_id}:{project_id}:{rel}"
                if key in self.index and self.index[key]["hash"] == h:
                    skipped += 1
                    continue
                self.index[key] = {"file_path": rel, "hash": h, "content_preview": content[:5000]}
                indexed += 1
            except Exception:
                failed += 1
        return {"total": indexed + skipped + failed, "indexed": indexed, "skipped": skipped, "failed": failed}

    def get_project_stats(self, user_id: str, project_id: str) -> Dict[str, int]:
        """Get statistics about indexed files for a project"""
        prefix = f"{user_id}:{project_id}:"
        unique_files = set()
        total_entries = 0
        
        for key in self.index.keys():
            if key.startswith(prefix):
                total_entries += 1
                # Extract file path from key
                file_path = key[len(prefix):]
                unique_files.add(file_path)
        
        return {
            "total_files": len(unique_files),
            "total_chunks": total_entries,
            "user_id": user_id,
            "project_id": project_id
        }

    def search(self, query: str, user_id: str, project_id: str, max_results: int = 10) -> List[Dict]:
        results: List[Dict] = []
        q = query.lower()
        for key, rec in self.index.items():
            if not key.startswith(f"{user_id}:{project_id}:"):
                continue
            score = rec.get("content_preview", "").lower().count(q)
            if score > 0:
                results.append({
                    "file_path": rec["file_path"],
                    "similarity": float(min(1.0, 0.1 * score)),
                    "chunk": {"chunk_index": 0, "start_line": 1, "end_line": 1},
                    "content_preview": rec.get("content_preview", "")[:200]
                })
        return sorted(results, key=lambda r: r["similarity"], reverse=True)[:max_results]

    def get_graph_context_for_files(self, file_paths: List[str], user_id: str, project_id: str) -> Dict[str, Dict]:
        # Minimal placeholder – local mode returns empty graph
        return {fp: {"nodes": [], "edges": []} for fp in file_paths}

    def get_stats(self, user_id: str, project_id: str) -> Dict:
        files = [v for k, v in self.index.items() if k.startswith(f"{user_id}:{project_id}:")]
        return {"files_indexed": len(files), "total_chunks": len(files), "last_indexed": None, "storage": "local", "embedding_model": "n/a"}

    def clear_project(self, user_id: str, project_id: str):
        keys = [k for k in list(self.index.keys()) if k.startswith(f"{user_id}:{project_id}:")]
        for k in keys:
            self.index.pop(k, None)

    def remove_file(self, user_id: str, project_id: str, file_path: str):
        self.index.pop(f"{user_id}:{project_id}:{file_path}", None)
        return True

    def _should_index_file(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".uid"}:
            return False
        return True


