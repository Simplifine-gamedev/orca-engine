"""
Lightweight in-memory todo store for AI planning.

Stores todos per project_root so each Godot project gets its own list.
"""

from __future__ import annotations

import time
import uuid
from threading import Lock
from typing import Dict, List, Optional


class TodoStore:
    _lock: Lock = Lock()
    _todos: Dict[str, List[Dict]] = {}
    _valid_statuses = {"pending", "in_progress", "completed", "cancelled"}

    @classmethod
    def _key(cls, project_root: Optional[str]) -> str:
        if not project_root:
            return "global"
        return project_root

    @classmethod
    def list(cls, project_root: Optional[str]) -> List[Dict]:
        key = cls._key(project_root)
        with cls._lock:
            return [todo.copy() for todo in cls._todos.get(key, [])]

    @classmethod
    def clear(cls, project_root: Optional[str]) -> None:
        key = cls._key(project_root)
        with cls._lock:
            cls._todos[key] = []

    @classmethod
    def add(cls, project_root: Optional[str], content: str, status: str = "pending", created_by: Optional[str] = None) -> Dict:
        todo = {
            "id": str(uuid.uuid4()),
            "content": content,
            "status": cls._normalize_status(status),
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "created_by": created_by or "agent",
        }
        key = cls._key(project_root)
        with cls._lock:
            cls._todos.setdefault(key, [])
            cls._todos[key].append(todo)
        return todo.copy()

    @classmethod
    def add_batch(cls, project_root: Optional[str], items: List[Dict], created_by: Optional[str] = None) -> List[Dict]:
        created: List[Dict] = []
        for item in items:
            content = item.get("content", "").strip()
            status = item.get("status", "pending")
            if not content:
                continue
            created.append(cls.add(project_root, content, status, created_by=created_by))
        return created

    @classmethod
    def update(cls, project_root: Optional[str], todo_id: str, *, content: Optional[str] = None, status: Optional[str] = None) -> Optional[Dict]:
        key = cls._key(project_root)
        with cls._lock:
            todos = cls._todos.get(key, [])
            for todo in todos:
                if todo["id"] == todo_id:
                    if content is not None:
                        todo["content"] = content
                    if status is not None:
                        todo["status"] = cls._normalize_status(status)
                    todo["updated_at"] = int(time.time())
                    return todo.copy()
        return None

    @classmethod
    def remove(cls, project_root: Optional[str], todo_id: str) -> bool:
        key = cls._key(project_root)
        with cls._lock:
            todos = cls._todos.get(key, [])
            new_todos = [todo for todo in todos if todo["id"] != todo_id]
            removed = len(new_todos) != len(todos)
            cls._todos[key] = new_todos
            return removed

    @classmethod
    def _normalize_status(cls, status: Optional[str]) -> str:
        if not status:
            return "pending"
        normalized = status.strip().lower()
        if normalized not in cls._valid_statuses:
            return "pending"
        return normalized



