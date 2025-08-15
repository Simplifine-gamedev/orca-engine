# Orca Engine – Backend (AI Service)

Backend service for Orca Engine’s AI assistant. It handles OpenAI API calls, streams responses, and provides cloud vector search over project files.

## Architecture

1. Godot plugin handles UI and executes editor-only tools locally.
2. Backend (Flask) exposes HTTP endpoints, calls OpenAI, and executes server-side tools:
   - image_operation (image gen/edit)
   - search_across_project (semantic search over indexed files)
3. Cloud Vector Search stores embeddings and project graph in BigQuery when enabled.

Notes:
- Godot runs a local HTTP tool server on port 8001 (started by `AIChatDock`), used to execute editor-affecting tools with editor access. The backend does not call 8001 directly; instead it streams tool_calls, and the editor invokes the tool server.

## Key components

Backend service (default PORT 8000):
- /chat streaming endpoint with OpenAI function calling
- /embed indexing/search/status/clear for cloud vector search
- /search_project convenience API for semantic search
- OAuth endpoints for cloud mode

Cloud Vector Manager (BigQuery + OpenAI embeddings):
- Creates dataset `godot_embeddings` with tables: `embeddings`, `graph_nodes`, `graph_edges`
- Uses OpenAI model `text-embedding-3-small` (1536 dims)

## Setup (local)

1) Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

2) Environment
Create a `.env` in backend:
```
OPENAI_API_KEY=your-openai-key
# Optional for local dev
DEV_MODE=true
```

For production, set `FLASK_SECRET_KEY` and do not set `DEV_MODE=true`.

3) Run
```bash
python app.py           # dev (binds 0.0.0.0:8000)
# or production
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 8 --timeout 120 app:app
```

## Environment variables

- OPENAI_API_KEY (required)
- FLASK_SECRET_KEY (required in production; optional in DEV_MODE)
- DEV_MODE=true|false (default false)
- DEPLOYMENT_MODE=oss|cloud (default oss)
- REQUIRE_SERVER_API_KEY=true|false and SERVER_API_KEY (optional API gate)
- GCP_PROJECT_ID (enables cloud vector features; if unset, vector APIs return 501)
- ALLOW_GUESTS=true|false (default true in OSS, false in cloud unless set)
- PROJECT_ROOT (optional default for search/index requests)

## API

### Chat
```http
POST /chat
```
- Streams NDJSON lines. First line includes request_id. May emit tool_calls for frontend or execute backend tools.

### Stop
```http
POST /stop {"request_id": "..."}
```

### Generate script
```http
POST /generate_script
{ "script_type": "...", "node_type": "Node", "description": "..." }
```

### Predict code edit
```http
POST /predict_code_edit
{ "file_content": "...", "prompt": "..." }
```

### Auth (cloud mode)
- GET /auth/login?machine_id=...&provider=google|github|microsoft|guest
- GET /auth/callback and /api/auth/callback
- POST /auth/status
- GET /auth/providers
- POST /auth/guest
- POST /auth/logout

### Embedding/indexing
```http
POST /embed
{ "action": "index_project", "project_root": "/path", "force_reindex": false }
{ "action": "index_file", "project_root": "/path", "file_path": "/path/file.gd" }
{ "action": "index_files", "project_root": "/path", "files": [{"file_path":"...","content":"..."}] }
{ "action": "search", "project_root": "/path", "query": "player movement", "k": 5, "include_graph": false }
{ "action": "status", "project_root": "/path" }
{ "action": "clear", "project_root": "/path" }
```
Requires auth; set `X-Project-Root` header as fallback when needed.

### Project search (convenience)
```http
POST /search_project
{ "query": "...", "project_root": "/path", "max_results": 5 }
```

### Health
```http
GET /health
```

## Cloud vector search

Embeddings: OpenAI `text-embedding-3-small` (1536 dims). Storage: BigQuery.

Table `godot_embeddings.embeddings`:
- id (STRING), user_id (STRING), project_id (STRING), file_path (STRING)
- chunk_index (INTEGER), content (STRING), start_line (INTEGER), end_line (INTEGER)
- file_hash (STRING), indexed_at (TIMESTAMP)
- embedding (FLOAT64 REPEATED)

Graph tables:
- graph_nodes(id, user_id, project_id, file_path, kind, name, node_type, node_path, start_line, end_line, updated_at)
- graph_edges(user_id, project_id, src_id, dst_id, kind, file_path, start_line, end_line, updated_at)

### Graph semantics

- graph_nodes.kind:
  - "File": a file node for each indexed file
  - "SceneNode": a node declared in a `.tscn` scene; fields: `name`, `node_type`, `node_path`, and source line range

- graph_edges.kind (examples):
  - "CHILD_OF": parent-child scene hierarchy within a `.tscn`
  - "ATTACHES_SCRIPT": scene node attaches a script resource
  - "INSTANTIATES_SCENE": scene node instantiates an external `PackedScene`
  - "CONNECTS_SIGNAL[:signal->method]": connection lines found in `[connection]` sections

How it is built:
- During indexing of `.tscn` files, headers like `[node name="Player" type="CharacterBody2D" parent="."]` are parsed to create `SceneNode` entries and `CHILD_OF` edges. `ExtResource` references to `PackedScene` create `INSTANTIATES_SCENE` edges. `script = ...` lines and `ExtResource("...")` script references create `ATTACHES_SCRIPT`. `[connection ...]` sections create `CONNECTS_SIGNAL` edges. For all files, a `File` node is also upserted.

Indexing scope:
- Indexes text-like Godot files: .gd, .cs, .cpp, .h, .tscn, .tres, .res, .godot, .gdextension, .json, .cfg, .md, .txt, .shader, .gdshader, .glsl
- Skips binaries: images/audio/video/fonts/archives/binaries/.uid/.import/.godot caches, etc. (we will be making indexing multimodal soon :) )

## Deployment (Cloud Run)

```bash
cd backend
./deploy.sh your-gcp-project-id
```
The script builds and deploys to Cloud Run, enables required APIs (Cloud Build, Run, BigQuery, Secret Manager), and uploads `.env` keys as secrets. Vertex AI is not required (embeddings use OpenAI).

## Security

- OAuth (Google/GitHub/Microsoft) supported via AuthManager in cloud mode
- Guest sessions allowed by default in OSS mode; can be disabled
- Optional server-side API key gate for sensitive endpoints
- TLS provided by Cloud Run in production

## Troubleshooting

- 501 errors on /embed or /search_project: set `GCP_PROJECT_ID`
- Auth required: provide machine_id and valid session (or enable DEV_MODE for local)
- Large prompts: logs warn if total message size is very large

## License

Same as project root. See root `NOTICE` for licensing of Simplifine additions and upstream Godot.