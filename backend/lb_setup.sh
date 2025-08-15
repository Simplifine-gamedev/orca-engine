#!/usr/bin/env bash

set -euo pipefail

# HTTPS Load Balancer for Cloud Run + Cloud Armor attachment
#
# Usage:
#   ./lb_setup.sh --domain gamechat.simplifine.com --service godot-ai-backend --region us-central1 [--policy godot-ai-armor]
#
# Requires: gcloud authenticated; project set

DOMAIN=""
SERVICE=""
REGION=""
POLICY="godot-ai-armor"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --domain <FQDN> --service <cloud-run-service> --region <region> [--policy name]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DOMAIN" || -z "$SERVICE" || -z "$REGION" ]]; then
  echo "❌ Missing required args. Use --domain, --service, --region." >&2
  exit 1
fi

echo "🔐 Project: $(gcloud config get-value project 2>/dev/null)"

NEG="neg-${SERVICE}-${REGION}"
BACKEND="bs-${SERVICE}"
URLMAP="um-${SERVICE}"
PROXY="thp-${SERVICE}"
CERT="cert-${DOMAIN//./-}"
IPNAME="ip-${SERVICE}"
FWRULE="fr-${SERVICE}"

echo "🧩 Creating serverless NEG: $NEG"
gcloud compute network-endpoint-groups create "$NEG" \
  --region="$REGION" \
  --network-endpoint-type=SERVERLESS \
  --cloud-run-service="$SERVICE" \
  || true

echo "🧩 Creating backend service: $BACKEND"
gcloud compute backend-services create "$BACKEND" --global --load-balancing-scheme=EXTERNAL --protocol=HTTPS || true

echo "🧩 Adding NEG to backend: $BACKEND"
gcloud compute backend-services add-backend "$BACKEND" --global \
  --network-endpoint-group="$NEG" --network-endpoint-group-region="$REGION" || true

echo "🛡️  Attaching Cloud Armor policy: $POLICY"
gcloud compute backend-services update "$BACKEND" --security-policy "$POLICY" --global || true

echo "🧩 Creating URL map: $URLMAP"
gcloud compute url-maps create "$URLMAP" --default-service "$BACKEND" --global || true

echo "🔐 Creating managed cert: $CERT for $DOMAIN"
gcloud compute ssl-certificates create "$CERT" --domains="$DOMAIN" --global || true

echo "🧩 Creating target HTTPS proxy: $PROXY"
gcloud compute target-https-proxies create "$PROXY" --url-map="$URLMAP" --ssl-certificates="$CERT" --global || true

echo "🌐 Reserving global static IP: $IPNAME"
gcloud compute addresses create "$IPNAME" --global || true
IP=$(gcloud compute addresses describe "$IPNAME" --global --format='value(address)')
echo "📌 Reserved IP: $IP"

echo "🧩 Creating forwarding rule: $FWRULE"
gcloud compute forwarding-rules create "$FWRULE" --address="$IPNAME" --global --target-https-proxy="$PROXY" --ports=443 || true

echo "✅ HTTPS LB configured. Point $DOMAIN to $IP via DNS A record and wait for SSL cert to provision."


