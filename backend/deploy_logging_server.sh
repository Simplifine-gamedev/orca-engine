#!/bin/bash

# Godot AI Logging Service - Standalone Cloud Run Deployment

# Configuration
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else
    echo "❌ Error: GCP project id not provided."
    echo "Usage: ./deploy_logging_server.sh <GCP_PROJECT_ID> [SERVICE_NAME]"
    echo "   OR: Set GCP_PROJECT_ID environment variable"
    exit 1
fi

# Service name
SERVICE_NAME="${2:-godot-ai-logging-service}"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Godot AI Logging Service"
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo ""

# Run from script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "$SCRIPT_DIR" >/dev/null

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: No active gcloud authentication."
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project ${PROJECT_ID}

# Enable APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create logging-specific Dockerfile
cat > Dockerfile <<EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for logging service
RUN pip install --no-cache-dir \\
    Flask==3.0.0 \\
    flask-cors==4.0.0 \\
    python-dotenv==1.0.0 \\
    aiohttp==3.9.1 \\
    requests==2.31.0 \\
    gunicorn==21.2.0

# Copy only logging service files
COPY logging_server.py .
COPY litellm_callback.py .

# Copy .env if it exists (optional - secrets will be provided via Cloud Run)
# Note: .env files are handled via GCP Secret Manager, not copied into container

# Expose port (Cloud Run will override this)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:\$PORT/health || exit 1

# Run with gunicorn for production
CMD gunicorn --bind 0.0.0.0:\$PORT --workers 2 --timeout 120 logging_server:app
EOF

echo "🏗️  Building logging service container..."
gcloud builds submit --tag ${IMAGE_NAME}

# Handle secrets from .env
SECRET_REFS=""
if [ -f ".env" ]; then
    echo "🔐 Processing Supabase secrets..."
    
    # Get service account for permissions
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
    # Grant Secret Manager access
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
    
    # Process only Supabase-related secrets
    SECRET_NAMES=()
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Only Supabase secrets for logging service
        if [[ $key =~ ^SUPABASE_.*$ ]]; then
            value=$(echo "$value" | sed 's/^"//;s/"$//')
            
            echo "🔐 Uploading secret: $key"
            if echo -n "$value" | gcloud secrets create "$key" --data-file=- --replication-policy="automatic" --quiet 2>/dev/null || \
               echo -n "$value" | gcloud secrets versions add "$key" --data-file=- --quiet 2>/dev/null; then
                SECRET_NAMES+=("$key")
            fi
        fi
    done < .env
    
    # Build secret references
    for secret in "${SECRET_NAMES[@]}"; do
        if [ -z "$SECRET_REFS" ]; then
            SECRET_REFS="${secret}=${secret}:latest"
        else
            SECRET_REFS="${SECRET_REFS},${secret}=${secret}:latest"
        fi
    done
    
    if [ -n "$SECRET_REFS" ]; then
        echo "✅ Secrets configured: ${#SECRET_NAMES[@]} Supabase secrets"
    fi
fi

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."

DEPLOY_ARGS=(
    --image "${IMAGE_NAME}"
    --platform managed
    --region "${REGION}"
    --allow-unauthenticated
    --port 8080
    --memory 1Gi
    --cpu 1
    --concurrency 100
    --min-instances 0
    --max-instances 10
    --timeout 300
    --set-env-vars "FLASK_ENV=production"
)

if [ -n "$SECRET_REFS" ]; then
    DEPLOY_ARGS+=(--set-secrets "$SECRET_REFS")
fi

gcloud run deploy "${SERVICE_NAME}" "${DEPLOY_ARGS[@]}"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

# Clean up
rm -f Dockerfile

echo ""
echo "✅ Logging Service Deployed Successfully!"
echo "=================================="
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "🔧 Add this to your main backend .env:"
echo "LOGGING_SERVER_URL=${SERVICE_URL}"
echo ""
echo "🧪 Test the service:"
echo "curl ${SERVICE_URL}/health"
echo "curl ${SERVICE_URL}/stats"
echo ""
echo "📋 View logs:"
echo "gcloud logs tail --follow --project=${PROJECT_ID} --resource-names=${SERVICE_NAME}"

popd >/dev/null
