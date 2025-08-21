# Orca Engine – Backend (AI Service)

Backend service for Orca Engine's AI assistant. It handles OpenAI API calls, streams responses, and provides **advanced vector search** with **function-level intelligence** over project files.

## Architecture

1. Godot plugin handles UI and executes editor-only tools locally.
2. Backend (Flask) exposes HTTP endpoints, calls OpenAI, and executes server-side tools:
   - image_operation (image gen/edit)
   - search_across_project (semantic/keyword/hybrid search with dependency tracing)
3. **Enhanced Indexing System** stores function-level chunks, signal flows, and dependency graphs in Weaviate.

> 📖 **For detailed indexing documentation, see [indexing.md](./indexing.md)**

Notes:
- Godot runs a local HTTP tool server on port 8001 (started by `AIChatDock`), used to execute editor-affecting tools with editor access. The backend does not call 8001 directly; instead it streams tool_calls, and the editor invokes the tool server.

## Key components

Backend service (default PORT 8000):
- /chat streaming endpoint with OpenAI function calling + advanced search modes
- /embed indexing/search/status/clear for enhanced vector search with dependency tracing
- /search_project convenience API for semantic/keyword/hybrid search
- OAuth endpoints for cloud mode

**Enhanced Vector Manager (Weaviate + OpenAI embeddings)**:
- **Function-level chunking**: GDScript functions, signals, exports as separate searchable units
- **Signal flow tracking**: Complete signal emission → connection → handler chains  
- **Multi-hop dependency tracing**: Function call chains across files
- **Smart search modes**: Semantic (AI), keyword (exact), hybrid (both)
- Uses OpenAI model `text-embedding-3-small` (1536 dims) with Weaviate vector database

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

# Weaviate Configuration (for enhanced vector search)
WEAVIATE_URL=https://your-cluster.weaviate.cloud  
WEAVIATE_API_KEY=your-weaviate-api-key

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

- OPENAI_API_KEY (required for embeddings)
- WEAVIATE_URL (required for enhanced vector search)  
- WEAVIATE_API_KEY (required for Weaviate access)
- FLASK_SECRET_KEY (required in production; optional in DEV_MODE)
- DEV_MODE=true|false (default false)
- DEPLOYMENT_MODE=oss|cloud (default oss)
- REQUIRE_SERVER_API_KEY=true|false and SERVER_API_KEY (optional API gate)
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

### Enhanced Embedding/Indexing
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

### Advanced Project Search (with AI intelligence)
```http  
POST /search_across_project
{
  "query": "player collision physics",
  "search_mode": "semantic|keyword|hybrid",
  "trace_dependencies": true,
  "include_graph": true,
  "max_results": 5
}
```

### Project search (simple convenience)
```http
POST /search_project  
{ "query": "...", "project_root": "/path", "max_results": 5 }
```

### Health
```http
GET /health
```

## Enhanced Vector Search with Weaviate

**Embeddings**: OpenAI `text-embedding-3-small` (1536 dims)  
**Storage**: Weaviate vector database with advanced indexing

### Weaviate Collections

**ProjectEmbedding** (function-level chunks):
- file_path, content, chunk_index, chunk_start, chunk_end
- **chunk_type**: "function", "signal", "export", "node", "header"
- **function_name**: Extracted function name (if applicable)
- **signals_emitted**: List of signals emitted by this chunk
- **functions_called**: List of functions called by this chunk  
- **nodes_accessed**: List of node paths accessed
- user_id, project_id, file_hash, indexed_at
- embedding (1536-dim vector)

**ProjectGraph** (file relationships):
- source_file, target_file, relationship_type, weight
- user_id, project_id, updated_at

**ProjectDependencies** (function-level dependencies):
- source_file, source_function, target_file, target_function
- dependency_type, line_number, context, weight
- user_id, project_id, updated_at

### Enhanced Graph Semantics

**File-Level Relationships** (ProjectGraph):
- "EXTENDS": Class inheritance (`extends CharacterBody2D`)
- "PRELOADS": Resource preloading (`preload("res://...")`)  
- "INSTANTIATES_SCENE": Scene instantiation in `.tscn` files
- "ATTACHES_SCRIPT": Script attachment in scenes
- "CONNECTS_SIGNAL": Signal connections in scenes

**Function-Level Dependencies** (ProjectDependencies):
- "CALLS_FUNCTION": Direct function calls (`function_name()`, `self.method()`)
- "EMITS_SIGNAL": Signal emissions (`emit_signal("signal_name")`)
- "CONNECTS_SIGNAL": Signal connections (`signal.connect(method)`)
- "ACCESSES_NODE": Node references (`get_node()`, `$NodePath`)
- "USES_PHYSICS_API": Physics calls (`move_and_slide()`, `is_on_floor()`)
- "ACCESSES_INPUT": Input system (`Input.is_action_pressed()`)

**How Enhanced Indexing Works**:
1. **GDScript files**: Functions parsed individually, each `func`, `signal`, `@export` becomes a chunk with dependency metadata
2. **Scene files**: Each `[node ...]` section becomes a searchable node chunk with signal connections extracted
3. **Dependency extraction**: Pattern matching finds function calls, signal emissions, node accesses within function bodies
4. **Multi-hop tracing**: Dependency chains followed across files (e.g., Input → Controller → Physics → Animation)

Indexing scope:
- Indexes text-like Godot files: .gd, .cs, .cpp, .h, .tscn, .tres, .res, .godot, .gdextension, .json, .cfg, .md, .txt, .shader, .gdshader, .glsl
- Skips binaries: images/audio/video/fonts/archives/binaries/.uid/.import/.godot caches, etc. (we will be making indexing multimodal soon :) )

## Deployment (Cloud Run)

```bash
cd backend
./deploy.sh your-gcp-project-id
```
The script builds and deploys to Cloud Run, enables required APIs (Cloud Build, Run, Secret Manager), and uploads `.env` keys as secrets. Configure your Weaviate cluster separately and set `WEAVIATE_URL` and `WEAVIATE_API_KEY` environment variables.

## Security

- OAuth (Google/GitHub/Microsoft) supported via AuthManager in cloud mode
- Guest sessions allowed by default in OSS mode; can be disabled
- Optional server-side API key gate for sensitive endpoints
- TLS provided by Cloud Run in production

## Troubleshooting

### Vector Search Issues
- **No search results**: Check Weaviate connection (`WEAVIATE_URL` and `WEAVIATE_API_KEY`)
- **Slow indexing**: Large projects with many functions take longer; check Weaviate cluster performance  
- **Function not found**: Ensure file is indexed and not filtered (check for `.import` files)
- **Poor dependency tracking**: Verify function extraction in logs (`Found X functions in file.gd`)

### Authentication & General
- **Auth required**: provide machine_id and valid session (or enable `DEV_MODE=true` for local)
- **Large prompts**: logs warn if total message size is very large  
- **Search mode issues**: Use `search_mode: "keyword"` for exact text matching, `"semantic"` for AI understanding

### Debug Commands
```bash
# Check Weaviate connection
grep "Weaviate v4 client connected" logs

# Verify function-level chunking  
grep "Found [0-9]+ functions" logs

# Check dependency extraction
grep "dependencies extracted" logs
```

## License

Same as project root. See root `NOTICE` for licensing of Simplifine additions and upstream Godot.