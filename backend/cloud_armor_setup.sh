#!/usr/bin/env bash

set -euo pipefail

# Cloud Armor setup for the AI backend behind an HTTPS Load Balancer.
# Idempotently creates/updates a policy and safe rate-limit rules, and optionally attaches to a backend service.
#
# Usage:
#   ./cloud_armor_setup.sh [--policy godot-ai-armor] [--backend-service BACKEND_SERVICE_NAME] [--enforce]
#
# Notes:
# - Requires: gcloud authenticated and project set (gcloud config set project ...)
# - The backend service is the global HTTPS LB backend (serverless NEG to Cloud Run). Use:
#     gcloud compute backend-services list --global
#   to find the correct NAME to pass to --backend-service.

POLICY_NAME="godot-ai-armor"
BACKEND_SERVICE=""
ENFORCE_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      POLICY_NAME="$2"; shift 2 ;;
    --backend-service)
      BACKEND_SERVICE="$2"; shift 2 ;;
    --enforce)
      ENFORCE_MODE=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--policy godot-ai-armor] [--backend-service BACKEND_SERVICE_NAME] [--enforce]"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if ! command -v gcloud >/dev/null 2>&1; then
  echo "❌ gcloud not found. Install Google Cloud SDK and authenticate." >&2
  exit 1
fi

echo "🔐 Using project: $(gcloud config get-value project 2>/dev/null)"

if ! gcloud compute security-policies describe "$POLICY_NAME" >/dev/null 2>&1; then
  echo "🛡️  Creating Cloud Armor policy: $POLICY_NAME"
  gcloud compute security-policies create "$POLICY_NAME" --description "Armor for AI backend"
else
  echo "🛡️  Policy exists: $POLICY_NAME (will update rules)"
fi

# Helper: create or update a rule idempotently
create_or_update_rule() {
  local priority="$1"; shift
  if gcloud compute security-policies rules describe "$priority" --security-policy "$POLICY_NAME" >/dev/null 2>&1; then
    gcloud compute security-policies rules update "$priority" --security-policy "$POLICY_NAME" "$@"
  else
    gcloud compute security-policies rules create "$priority" --security-policy "$POLICY_NAME" "$@"
  fi
}

# Preview flag toggling
PREVIEW_FLAG=(--preview)
if [[ "$ENFORCE_MODE" == true ]]; then
  PREVIEW_FLAG=(--no-preview)
fi

echo "⚙️  Configuring high-rate rule for /embed"
create_or_update_rule 200 \
  --expression 'request.path.startsWith("/embed")' \
  --action throttle \
  --rate-limit-threshold-count 600 \
  --rate-limit-threshold-interval-sec 60 \
  --conform-action allow \
  --exceed-action deny-429 \
  --enforce-on-key IP \
  "${PREVIEW_FLAG[@]}" \
  --description "High-rate for embed; $( [[ "$ENFORCE_MODE" == true ]] && echo enforce || echo preview first )"

echo "⚙️  Configuring moderate-rate rule for /chat and /search_project"
create_or_update_rule 210 \
  --expression 'request.path.startsWith("/chat") || request.path.startsWith("/search_project")' \
  --action throttle \
  --rate-limit-threshold-count 240 \
  --rate-limit-threshold-interval-sec 60 \
  --conform-action allow \
  --exceed-action deny-429 \
  --enforce-on-key IP \
  "${PREVIEW_FLAG[@]}" \
  --description "Moderate rate for chat/search; $( [[ "$ENFORCE_MODE" == true ]] && echo enforce || echo preview first )"

echo "⚙️  Configuring default safety rule"
create_or_update_rule 900 \
  --expression 'true' \
  --action throttle \
  --rate-limit-threshold-count 120 \
  --rate-limit-threshold-interval-sec 60 \
  --conform-action allow \
  --exceed-action deny-429 \
  --enforce-on-key IP \
  "${PREVIEW_FLAG[@]}" \
  --description "Default rate limit; $( [[ "$ENFORCE_MODE" == true ]] && echo enforce || echo preview first )"

if [[ -n "$BACKEND_SERVICE" ]]; then
  echo "🔗 Attaching policy to backend service: $BACKEND_SERVICE"
  gcloud compute backend-services update "$BACKEND_SERVICE" --security-policy "$POLICY_NAME" --global
else
  echo "ℹ️  Skipping attach. To attach later: gcloud compute backend-services update <BACKEND_SERVICE> --security-policy $POLICY_NAME --global"
fi

echo "✅ Cloud Armor policy '$POLICY_NAME' configured."


