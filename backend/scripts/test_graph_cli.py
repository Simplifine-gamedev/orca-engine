#!/usr/bin/env python3
"""
Quick CLI to exercise graph indexing + search locally.

Usage:
    python backend/scripts/test_graph_cli.py \
        --project-root /path/to/project \
        --files backend/app.py editor/docks/ai_chat_dock.cpp \
        --query "player health" \
        --base-url http://127.0.0.1:5050
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import uuid

import requests


def read_file(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def authenticate(base_url: str, machine_id: str) -> str:
    resp = requests.post(
        f"{base_url}/auth/guest",
        json={"machine_id": machine_id, "guest_name": "GraphTester"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"Guest auth failed: {payload}")
    token = payload.get("token")
    if not token:
        raise RuntimeError("Guest auth response missing token")
    return token


def index_files(base_url: str, token: str, machine_id: str, project_root: str,
                project_id: str, files: list[dict], force: bool) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Machine-ID": machine_id,
    }
    resp = requests.post(
        f"{base_url}/embed",
        json={
            "action": "index_files",
            "project_root": project_root,
            "project_id": project_id,
        "files": files,
        "force_reindex": force,
        },
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def search_project(base_url: str, token: str, machine_id: str, project_root: str,
                   project_id: str, query: str, max_results: int, include_graph: bool,
                   graph_preview: bool) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Machine-ID": machine_id,
    }
    resp = requests.post(
        f"{base_url}/search_project",
        json={
            "project_root": project_root,
            "project_id": project_id,
            "query": query,
            "max_results": max_results,
            "include_graph": include_graph,
            "include_graph": include_graph,
            "graph_preview": graph_preview,
        },
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test graph indexing/search via backend API")
    parser.add_argument("--base-url", default="http://127.0.0.1:5050", help="Backend base URL")
    parser.add_argument("--project-root", default=str(pathlib.Path.cwd()), help="Project root (matches frontend)")
    parser.add_argument("--project-id", default=None, help="Optional project ID override")
    parser.add_argument("--machine-id", default=None, help="Machine ID (auto-generated if omitted)")
    parser.add_argument("--files", nargs="+", required=True, help="Files to index (absolute or relative paths)")
    parser.add_argument("--query", default="player", help="Search query")
    parser.add_argument("-k", "--max-results", type=int, default=5, help="Max search results")
    parser.add_argument("--graph-preview", action="store_true", help="Request trimmed graph context in response")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex even if hashes match")
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    machine_id = args.machine_id or f"cli-{uuid.uuid4().hex[:8]}"
    project_root = os.path.abspath(args.project_root)
    project_id = args.project_id or hashlib.md5(project_root.encode()).hexdigest()

    print(f"➡️  Authenticating as guest (machine_id={machine_id})")
    token = authenticate(base_url, machine_id)
    print("✅ Guest token acquired")

    files_payload = []
    for file_path_str in args.files:
        path = pathlib.Path(file_path_str).expanduser()
        if not path.is_absolute():
            path = pathlib.Path(project_root) / path
        if not path.exists():
            print(f"⚠️  Skipping missing file: {path}")
            continue
        content = read_file(path)
        relative = os.path.relpath(path, project_root)
        files_payload.append({
            "path": relative.replace("\\", "/"),
            "content": content,
            "hash": md5(content),
        })

    if not files_payload:
        print("❌ No valid files to index")
        return 1

    print(f"➡️  Indexing {len(files_payload)} files into project {project_id}")
    index_resp = index_files(base_url, token, machine_id, project_root, project_id, files_payload, args.force_reindex)
    print(f"📊 Index response: {json.dumps(index_resp, indent=2)}")

    print(f"➡️  Searching for '{args.query}' (include_graph=True, graph_preview={args.graph_preview})")
    search_resp = search_project(
        base_url, token, machine_id, project_root, project_id, args.query,
        args.max_results, True, args.graph_preview
    )
    print("🔍 Search response:")
    print(json.dumps(search_resp, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

