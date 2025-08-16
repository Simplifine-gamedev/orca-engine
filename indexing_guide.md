## Indexing and Graph Guide

This guide explains how the project indexing and graph context work end-to-end: what gets indexed, how the editor triggers real‑time updates, how the backend stores data, and how to query it.

### Components

- **Backend (Flask)**: `backend/app.py`
  - Embedding API: `POST /embed` (index/search/status/clear)
  - Project search APIs: `POST /search_project` and the tool-backed flow inside `POST /chat`
  - Uses `backend/cloud_vector_manager.py` for indexing, graph building, and semantic search
- **Vector/Graph store**: Google BigQuery tables created/managed by `CloudVectorManager`
- **Editor (Godot)**: `editor/docks/ai_chat_dock.cpp`
  - Initializes indexing, watches file save events, batches changes, and calls the backend

### What gets indexed

- **Supported text/code extensions** (subset): `.gd, .cs, .cpp, .h, .tscn, .tres, .json, .md, .glsl, .shader` (see `CloudVectorManager.TEXT_EXTENSIONS`)
- **Skipped**: binary/media (`.png, .jpg, .ogg, .wav, .mp4`), caches, large files (>10MB), Godot `.uid` sidecars, etc. (`SKIP_EXTENSIONS`, `SKIP_PATTERNS`)
- **Chunking**
  - General: smart line chunks (default ~50 lines with 10‑line overlap)
  - Scenes/resources (`.tscn/.tres`): section‑based chunks
  - Code (`.gd/.cs/.cpp/.h`): function/class‑aware chunks
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dims). Query embedding computed the same way.

### Where data is stored (schema)

Created automatically in BigQuery dataset `godot_embeddings` (configurable):

- `embeddings`
  - id, user_id, project_id, file_path (relative to project root), chunk_index, content, start_line, end_line, file_hash, indexed_at, embedding (FLOAT64 REPEATED)
- `graph_nodes`
  - id, user_id, project_id, file_path, kind, name, node_type, node_path, start_line, end_line, updated_at
- `graph_edges`
  - user_id, project_id, src_id, dst_id, kind, file_path, start_line, end_line, updated_at

IDs are stable hashes built from `project_id`, file path, and symbol identity.

### How indexing is initiated

Backend entrypoint: `POST /embed` in `backend/app.py`.

- **index_project**
  - Server walks `project_root`, filters files, chunk → embed → insert rows, and builds/updates graph for scenes, scripts, and resources.
  - Parallelized with a worker pool (tunable via `INDEX_MAX_WORKERS`).
- **index_file / update_file**
  - Index a single on‑disk file by path (server must see the filesystem).
- **index_files**
  - Client sends an array of `{ path, content, hash }`. Server chunks/embeds/inserts using the provided content (works when server can’t access the project filesystem).
- **status / clear / remove_file / search** supported as well.

Project/user scoping:

- `project_root` is required by `/embed`. If `project_id` is omitted, the backend uses `md5(project_root)`.
- Auth is verified, but the backend supports automatic guest fallback using `X-Machine-ID`. Editor sets headers for both.

### Real‑time indexing from the editor

In `editor/docks/ai_chat_dock.cpp`:

- On startup, the dock calls `_ensure_project_indexing()` which initializes the embedding system and kicks off indexing.
- `_initialize_embedding_system()`:
  - Hooks save signals (`EditorNode::resource_saved`, `EditorNode::scene_saved`) to detect changed files.
  - Starts a poll timer `_on_embedding_poll_tick()` to batch and flush pending changes.
  - Creates an `HTTPRequest` node to call the backend and a status timer to update UI.
- Initial full index: `_perform_initial_indexing()` prefers server‑side `action: "index_project"` for speed. If unavailable, it falls back to client batch upload:
  - `_scan_and_index_project_files()` reads project files, computes a per‑content hash, and sends batched `action: "index_files"` requests via `_send_file_batch()`.
- Real‑time updates: on each save, files are added to a pending set; the poll tick composes a small `index_files` payload with `{ path, content, hash }` and sends it.
- All embedding calls include headers: `X-User-ID`, `X-Machine-ID`, and `X-Project-Root` (absolute path of `res://`).

### How the graph is created

Implemented in `backend/cloud_vector_manager.py` during indexing:

- A `File` node is upserted for every indexed file.
- Scenes/resources (`.tscn/.tres`):
  - Parse `[node ...]` blocks → `SceneNode` nodes
  - Edges: `CHILD_OF` (hierarchy), `INSTANTIATES_SCENE`, `ATTACHES_SCRIPT`, `GROUP_MEMBER`, `CONNECTS_SIGNAL:<signal>-><method>`, and `REFERENCES_RESOURCE`
- GDScript (`.gd`):
  - Nodes: `ScriptClass` (class_name), `ScriptFunction` (func), `Signal`
  - Edges: `DEFINES_CLASS`, `SCRIPT_EXTENDS`, `DEFINES_FUNCTION`, `DEFINES_SIGNAL`, cross‑refs `CALLS_FUNCTION`, `EMITS_SIGNAL`

Two graph retrieval helpers:

- `get_graph_context_for_files(files, user_id, project_id)` → per‑file limited nodes/edges
- `get_graph_context_expanded(files, user_id, project_id, depth=1..2, kinds=[...])` → light multi‑hop expansion with cache

Note: `backend/project_graph_manager.py` contains an in‑memory NetworkX graph builder used for offline analysis; the live graph used by the editor comes from BigQuery via the helpers above.

### How semantic search works

`CloudVectorManager.search(query, user_id, project_id, max_results)`:

- Embeds the query with the same OpenAI model.
- Runs a BigQuery vector similarity query using `1 - COSINE_DISTANCE(embedding, query_embedding)` as similarity.
- Returns top results with file path, chunk index and line range, similarity score, and a short content preview.

### How it’s used by the backend APIs

- `POST /embed` (app.py)
  - `action: "search"` returns raw results and optional expanded graph when `include_graph=true` with `graph_depth` and `graph_edge_kinds`.
  - Other actions manage indexing lifecycle.
- `POST /search_project` (app.py)
  - Thin wrapper that formats results for editor tools.
- Tool flow inside `POST /chat` (app.py)
  - The assistant can call the tool `search_across_project`, which executes `search_across_project_internal(...)` using `CloudVectorManager.search(...)` and `get_graph_context_for_files(...)`, and streams results back to the editor UI.

### Minimal API examples

Index entire project (server‑side):

```bash
curl -X POST "$BASE/embed" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: $USER_ID" \
  -H "X-Machine-ID: $MACHINE_ID" \
  -d '{
    "action": "index_project",
    "project_root": "/absolute/path/to/your/project"
  }'
```

Batch index specific files (client‑provided content):

```bash
curl -X POST "$BASE/embed" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: $USER_ID" \
  -H "X-Machine-ID: $MACHINE_ID" \
  -d '{
    "action": "index_files",
    "project_root": "/absolute/path/to/your/project",
    "files": [
      {"path": "scripts/player.gd", "content": "...", "hash": "abcd1234"}
    ]
  }'
```

Semantic search with graph context (shared embed API):

```bash
curl -X POST "$BASE/embed" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: $USER_ID" \
  -H "X-Machine-ID: $MACHINE_ID" \
  -d '{
    "action": "search",
    "project_root": "/absolute/path/to/your/project",
    "project_id": "optional-stable-id",
    "k": 5,
    "include_graph": true,
    "graph_depth": 2,
    "graph_edge_kinds": ["ATTACHES_SCRIPT", "CONNECTS_SIGNAL"]
  }'
```

Editor search tool flow (via chat): the assistant uses the `search_across_project` tool; the backend executes `search_across_project_internal(...)` and responds with a formatted list of similar files and a small per‑file graph context.

### Configuration (env vars)

- **Core**
  - `GCP_PROJECT_ID` (enables cloud vector store); if missing, the backend may fall back to a local JSON vector index when available
  - `OPENAI_API_KEY` (required for embeddings)
  - `EMBED_DATASET` (default `godot_embeddings`), `EMBED_TABLE` (default `embeddings`)
- **Performance & behavior**
  - `INDEX_MAX_WORKERS` (default CPU‑based), `EMBED_BATCH_SIZE` (default 100)
  - `SEARCH_CACHE_TTL`, `GRAPH_CACHE_TTL` (seconds)
  - `CHUNK_MAX_LINES` (override default chunk line count)

### Practical notes & troubleshooting

- If the backend cannot access the project filesystem (e.g., remote server), the editor’s client‑side fallback uploads file content using `index_files`.
- `.uid` files are intentionally ignored in both indexing and search responses.
- Similarity is cosine‑based; higher is more similar.
- For graph context in chat responses, `get_graph_context_for_files` is used (compact). For deeper navigation (via `/embed`), prefer `get_graph_context_expanded` with `depth` and `kinds`.


