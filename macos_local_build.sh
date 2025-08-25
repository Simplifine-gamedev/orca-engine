#!/usr/bin/env bash
set -euo pipefail

# Local macOS build and .app packaging for the editor
# Usage:
#   bash macos_local_build.sh
# Env (optionally set in ./.env.macos.local):
#   APP_NAME   (default: Orca)
#   BUNDLE_ID  (default: com.simplifine.orca)
#   VERSION    (default: 1.0.0)
#   UNIVERSAL  (default: 0, set to 1 to build x86_64+arm64 and lipo)

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load local env if present
if [ -f "$REPO_ROOT/.env.macos.local" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env.macos.local"
fi

APP_NAME="${APP_NAME:-Orca}"
BUNDLE_ID="${BUNDLE_ID:-com.simplifine.orca}"
VERSION="${VERSION:-1.0.0}"
UNIVERSAL="${UNIVERSAL:-0}"

cd "$REPO_ROOT"

echo "[1/4] Building editor binary..."
BUILD_FLAGS_COMMON=(platform=macos target=editor dev_build=no)

if [ "$UNIVERSAL" = "1" ]; then
  if scons "${BUILD_FLAGS_COMMON[@]}" arch=x86_64 -j"$(sysctl -n hw.ncpu)"; then
    :
  else
    echo "Vulkan SDK missing or build failed; retrying without Vulkan (vulkan=no)."
    scons "${BUILD_FLAGS_COMMON[@]}" arch=x86_64 vulkan=no -j"$(sysctl -n hw.ncpu)"
  fi
  if scons "${BUILD_FLAGS_COMMON[@]}" arch=arm64 -j"$(sysctl -n hw.ncpu)"; then
    :
  else
    echo "Vulkan SDK missing or build failed; retrying without Vulkan (vulkan=no)."
    scons "${BUILD_FLAGS_COMMON[@]}" arch=arm64 vulkan=no -j"$(sysctl -n hw.ncpu)"
  fi
  lipo -create ./bin/godot.macos.editor.x86_64 ./bin/godot.macos.editor.arm64 -output ./bin/godot.macos.editor.universal
  BIN="./bin/godot.macos.editor.universal"
else
  if scons "${BUILD_FLAGS_COMMON[@]}" arch=arm64 -j"$(sysctl -n hw.ncpu)"; then
    :
  else
    echo "Vulkan SDK missing or build failed; retrying without Vulkan (vulkan=no)."
    scons "${BUILD_FLAGS_COMMON[@]}" arch=arm64 vulkan=no -j"$(sysctl -n hw.ncpu)"
  fi
  BIN="./bin/godot.macos.editor.arm64"
  [ -f "$BIN" ] || BIN="./bin/godot.macos.editor.dev.arm64"
fi

if [ ! -f "$BIN" ]; then
  echo "Failed to locate built editor binary." >&2
  ls -la ./bin || true
  exit 1
fi

echo "[2/4] Creating ${APP_NAME}.app from template..."
rm -rf "${APP_NAME}.app"
cp -R misc/dist/macos_tools.app "${APP_NAME}.app"
mkdir -p "${APP_NAME}.app/Contents/MacOS"
install -m 755 "$BIN" "${APP_NAME}.app/Contents/MacOS/${APP_NAME}"

echo "[3/4] Updating Info.plist..."
PLIST_PATH="${APP_NAME}.app/Contents/Info.plist"
plist_set() {
  local key="$1"; shift
  local type="$1"; shift
  local value="$1"; shift
  /usr/libexec/PlistBuddy -c "Set :${key} ${value}" "$PLIST_PATH" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :${key} ${type} ${value}" "$PLIST_PATH"
}
plist_set CFBundleIdentifier string "${BUNDLE_ID}"
plist_set CFBundleName string "${APP_NAME}"
plist_set CFBundleDisplayName string "${APP_NAME}"
plist_set CFBundleExecutable string "${APP_NAME}"
plist_set CFBundleShortVersionString string "${VERSION}"
plist_set CFBundleVersion string "${VERSION}"

echo "[4/4] Done: ${APP_NAME}.app"
open -R "${APP_NAME}.app" || true




