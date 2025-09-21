#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<EOF
Orca Web build runner

Usage:
  bash build_web_all.sh [--serve] [--deploy] [--no-build]

Default behavior: build + copy artifacts to web-deployment + serve locally.

Flags:
  --serve      Run local dev server after build (default if no flags given)
  --deploy     Deploy using web-deployment/deploy-orca-web.sh (requires Vercel)
  --no-build   Skip compilation step, only copy and then serve/deploy
  -h, --help   Show this help
EOF
}

SERVE=false
DEPLOY=false
DO_BUILD=true

if [[ $# -eq 0 ]]; then
  SERVE=true
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve) SERVE=true; shift ;;
    --deploy) DEPLOY=true; shift ;;
    --no-build) DO_BUILD=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

ensure_emscripten() {
  if command -v emcc >/dev/null 2>&1; then
    return
  fi

  # Prefer bundled emsdk if present
  if [[ -d "$ROOT_DIR/emsdk" ]]; then
    echo "📦 Installing Emscripten via bundled emsdk..."
    pushd "$ROOT_DIR/emsdk" >/dev/null
    ./emsdk install latest
    ./emsdk activate latest
    # shellcheck disable=SC1091
    source ./emsdk_env.sh
    popd >/dev/null
    if command -v emcc >/dev/null 2>&1; then
      return
    fi
  fi

  # Fallback to Homebrew
  if command -v brew >/dev/null 2>&1; then
    echo "📦 Installing Emscripten via Homebrew..."
    brew install emscripten
  fi

  if ! command -v emcc >/dev/null 2>&1; then
    echo "❌ Emscripten (emcc) not found and could not be installed automatically."
    echo "   Please install emsdk or emscripten, then re-run this script."
    exit 1
  fi
}

copy_artifacts_into_web_deployment() {
  echo "📥 Copying build artifacts into web-deployment..."
  local SRC="$ROOT_DIR/bin/web_export"
  local DST="$ROOT_DIR/web-deployment"

  if [[ ! -d "$SRC" ]]; then
    echo "❌ Build output not found at $SRC"
    exit 1
  fi

  cp -f "$SRC"/orca.web.editor.wasm32.* "$DST"/ 2>/dev/null || true
  cp -f "$SRC"/vercel.json "$DST"/ 2>/dev/null || true

  echo "✅ Copied artifacts to $DST"
}

if [[ "$DO_BUILD" == true ]]; then
  ensure_emscripten
  echo "🚧 Building web editor..."
  bash "$ROOT_DIR/build_web.sh"
fi

copy_artifacts_into_web_deployment

if [[ "$SERVE" == true ]]; then
  echo "🌐 Starting local server..."
  pushd "$ROOT_DIR/web-deployment" >/dev/null
  bash ./run-local.sh
  popd >/dev/null
fi

if [[ "$DEPLOY" == true ]]; then
  echo "🚢 Deploying to Vercel..."
  pushd "$ROOT_DIR/web-deployment" >/dev/null
  bash ./deploy-orca-web.sh
  popd >/dev/null
fi

echo "✨ Done."


