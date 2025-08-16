#!/usr/bin/env bash
set -euo pipefail

# Godot Docs Indexer Runner
# - Reads GCP project and API key from env or backend/.env
# - Invokes the Python indexer with sane defaults
#
# Examples (run from repo root or any dir):
#   # Quick smoke test to JSONL only
#   backend/scripts/index_godot_docs.sh --max-files-rst 25 --max-files-xml 10 --limit-chunks 100 --out-jsonl build_temp/godot_docs.jsonl
#
#   # Full run into BigQuery (uses env GCP_PROJECT_ID and OPENAI_API_KEY)
#   backend/scripts/index_godot_docs.sh --force
#
#   # Pin branches
#   backend/scripts/index_godot_docs.sh --branch 4.2 --engine-branch 4.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"

# Load from backend/.env if present (without exporting everything)
read_env_var() {
  local key="$1"; local def_val="${2:-}"
  if [[ -n "${!key:-}" ]]; then
    echo "${!key}"
    return 0
  fi
  if [[ -f "$BACKEND_DIR/.env" ]]; then
    local raw
    raw=$(grep -m1 "^${key}=" "$BACKEND_DIR/.env" | cut -d= -f2- | sed 's/^"//; s/"$//') || true
    if [[ -n "$raw" ]]; then
      echo "$raw"
      return 0
    fi
  fi
  echo "$def_val"
}

GCP_PROJECT_ID="$(read_env_var GCP_PROJECT_ID)"
OPENAI_API_KEY_VAL="$(read_env_var OPENAI_API_KEY)"

if [[ -z "$GCP_PROJECT_ID" ]]; then
  echo "❌ GCP_PROJECT_ID is not set (env or backend/.env)" >&2
  exit 1
fi
if [[ -z "$OPENAI_API_KEY_VAL" ]]; then
  echo "❌ OPENAI_API_KEY is not set (env or backend/.env)" >&2
  exit 1
fi

# Optional usage banner
if [[ "${1:-}" == "--usage" || "${1:-}" == "--examples" ]]; then
  cat <<'USAGE'
Godot Docs Indexer – usage examples

Quick smoke test (JSONL output only):
  backend/scripts/index_godot_docs.sh \
    --max-files-rst 25 --max-files-xml 10 --limit-chunks 100 \
    --out-jsonl build_temp/godot_docs.jsonl

Full run to BigQuery (requires GCP_PROJECT_ID and OPENAI_API_KEY):
  backend/scripts/index_godot_docs.sh --force

Pin specific branches:
  backend/scripts/index_godot_docs.sh --branch 4.2 --engine-branch 4.2

Pass any Python flags after the script name; they are forwarded as-is.
USAGE
  exit 0
fi

echo "🚀 Indexing Godot docs into BigQuery project: $GCP_PROJECT_ID"
echo "📦 Dataset: ${EMBED_DATASET:-godot_embeddings}, Table: ${EMBED_TABLE:-embeddings}"

export OPENAI_API_KEY="$OPENAI_API_KEY_VAL"
export PYTHONUNBUFFERED=1

# Ensure module import works no matter where we run from: execute at repo root
pushd "$REPO_ROOT" >/dev/null
# Pass through any additional flags (e.g., --force, --branch, --engine-branch)
python -u -m backend.scripts.index_godot_docs \
  --gcp-project "$GCP_PROJECT_ID" \
  --openai-key "$OPENAI_API_KEY" \
  "$@"
popd >/dev/null


