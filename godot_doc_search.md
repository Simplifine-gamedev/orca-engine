## Godot Docs Search: Indexing and Querying

This document explains how the Godot documentation corpus is indexed into the BigQuery vector store and how the editor queries it via the `search_across_godot_docs` tool.

### What gets indexed
- Tutorials and guides from the official `godot-docs` repository (RST files)
- Class reference from the Godot engine repository (XML files)

Each document is parsed to plain text, chunked, embedded with OpenAI `text-embedding-3-small`, and upserted into the vector table compatible with the backend `CloudVectorManager`.

### Corpus identifiers
- `user_id`: `public_docs`
- `project_id`: `godot_docs_latest`

These can be overridden by CLI flags.

### Requirements
- Environment variables:
  - `GCP_PROJECT_ID`: Your Google Cloud project with BigQuery enabled
  - `OPENAI_API_KEY`: API key for embeddings
- Optional environment overrides:
  - `EMBED_DATASET` (default `godot_embeddings`)
  - `EMBED_TABLE` (default `embeddings`)
  - `DOCS_BRANCH` (default `latest`, falls back to `master`, etc.)
  - `GODOT_ENGINE_BRANCH` (default `master`)

### One-liners (recommended)

Run from anywhere; the script resolves paths itself.

1) Quick smoke test to JSONL (does not touch BigQuery):
```bash
backend/scripts/index_godot_docs.sh \
  --max-files-rst 25 --max-files-xml 10 --limit-chunks 100 \
  --out-jsonl build_temp/godot_docs.jsonl
```

2) Full run into BigQuery (creates dataset/table if missing):
```bash
GCP_PROJECT_ID=your-gcp-project OPENAI_API_KEY=sk-... \
backend/scripts/index_godot_docs.sh --force
```

3) Pin branches:
```bash
backend/scripts/index_godot_docs.sh --branch 4.2 --engine-branch 4.2
```

For a usage banner:
```bash
backend/scripts/index_godot_docs.sh --usage
```

### Under the hood
- Script: `backend/scripts/index_godot_docs.py`
  - Downloads `godot-docs` (RST) and the engine repo (XML) with robust branch fallbacks
  - Converts RST to plain text, parses XML for classes, methods, signals, and properties
  - Chunks by characters with overlap (defaults tunable via env)
  - Embeds in batches with retry and backoff
  - Inserts to BigQuery in batches (or writes JSONL shards if `--out-jsonl` is used)

- Shell wrapper: `backend/scripts/index_godot_docs.sh`
  - Reads `GCP_PROJECT_ID` and `OPENAI_API_KEY` from environment or `backend/.env`
  - Executes the Python module from repo root for stable imports
  - Forwards all CLI flags to the Python script

### Querying from the editor
The editor tool `search_across_godot_docs` calls the backend endpoint:
- Python handler: `backend/app.py` → `search_across_godot_docs_internal`
- HTTP endpoint (optional direct access): `POST /search_docs` with JSON:
```json
{
  "query": "How do I connect a signal between nodes in Godot 4?",
  "max_results": 5
}
```

Results are shown in the AI Chat dock as expandable cards with snippet and full content. The backend relies on the shared corpus identifiers (`public_docs/godot_docs_latest`) to fetch from BigQuery.

### Tuning performance
- Embedding:
  - `EMBED_BATCH_SIZE` (default 128)
  - `EMBED_MAX_PARALLEL` (default 8)
  - `DOCS_CHUNK_MAX_CHARS` (default 2000)
  - `DOCS_CHUNK_OVERLAP` (default 200)
- JSONL sharding: `DOCS_JSONL_SHARD_SIZE` (default 10000)

### Troubleshooting
- 404 on docs download: script automatically falls back through branches (`master`, `stable`, `4.3`, `4.2`).
- BigQuery permission errors: ensure your `gcloud auth application-default login` is configured or ADC is available to the environment running the script.
- Empty snippets in results: the backend now prefers `content` and falls back to `content_preview` or nested `chunk.content`.

### Example cURL to test the backend endpoint directly
```bash
curl -sS -X POST http://127.0.0.1:8001/search_docs \
  -H "Content-Type: application/json" \
  -H "X-Machine-ID: local-dev" \
  --data-raw '{"query":"How do I connect a signal between nodes in Godot 4?","max_results":5}' | jq
```

This should return a JSON with `success: true` and a `results` array containing titles, snippets, full content, similarity, and file paths.


