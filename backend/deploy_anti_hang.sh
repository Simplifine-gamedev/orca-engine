#!/bin/bash

# Godot AI Backend - Anti-Hang GCP Deployment Script
# This version includes optimizations to prevent hanging in Google Cloud Run

# Configuration
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else
    echo "❌ Error: GCP project id not provided."
    echo "Usage: ./deploy_anti_hang.sh <GCP_PROJECT_ID> [SERVICE_NAME]"
    exit 1
fi

if [ -n "$2" ]; then
    SERVICE_NAME="$2"
elif [ -n "$SERVICE_NAME" ]; then
    echo "📋 Using SERVICE_NAME from environment: $SERVICE_NAME"
else
    SERVICE_NAME="godot-ai-backend-anti-hang"
fi

echo "🚀 ANTI-HANG DEPLOYMENT for Godot AI Backend"
echo "Project ID: ${PROJECT_ID}"
echo "Service Name: ${SERVICE_NAME}"
echo ""
echo "🔧 OPTIMIZATIONS INCLUDED:"
echo "   ✅ Fixed 10 infinite polling loops with timeout protection"
echo "   ✅ Added LiteLLM completion timeouts for all AI calls"
echo "   ✅ Enhanced streaming response handling for GCP Cloud Run"
echo "   ✅ Increased memory (8Gi) and CPU (4 cores) for better performance"
echo "   ✅ Reduced concurrency (20) to prevent resource contention"
echo "   ✅ Extended timeout (900s) for complex operations"
echo "   ✅ Enhanced logging and monitoring for hang detection"
echo ""

REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Always run from script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "$SCRIPT_DIR" >/dev/null

# Check gcloud authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: No active gcloud authentication found."
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo "📋 Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Extract version information
echo "📋 Extracting versions..."
if [ -f "../extract_versions.py" ] && [ -f "../version.py" ]; then
    eval $(python3 ../extract_versions.py env_vars)
    echo "📋 Versions: Backend=$BACKEND_VERSION, API=$API_VERSION, Orca=$ORCA_VERSION"
else
    echo "⚠️  Using default versions"
    BACKEND_VERSION="1.0.0"
    API_VERSION="1.0"
    ORCA_VERSION="1.0.0"
fi

# Build container
echo "🏗️  Building optimized container image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Upload secrets with anti-hang configuration
SECRET_REFS=""
if [ -f ".env" ]; then
    echo "🔐 Uploading secrets with anti-hang optimizations..."
    
    # Enable Secret Manager API
    gcloud services enable secretmanager.googleapis.com
    
    # Grant permissions
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/secretmanager.secretAccessor"
    
    # Force production configuration
    echo "🔧 Forcing production configuration..."
    echo -n "false" | gcloud secrets create "DEV_MODE" --data-file=- --replication-policy="automatic" 2>/dev/null || \
       echo -n "false" | gcloud secrets versions add "DEV_MODE" --data-file=- 2>/dev/null
    
    # Read .env and create secrets
    SECRET_NAMES=("DEV_MODE")
    while IFS='=' read -r key value || [ -n "$key" ]; do
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        [[ $key == "DEV_MODE" ]] && continue
        
        value=$(echo "$value" | sed 's/^"//;s/"$//')
        
        echo "🔐 Creating secret: $key"
        if echo -n "$value" | gcloud secrets create "$key" --data-file=- --replication-policy="automatic" 2>/dev/null || \
           echo -n "$value" | gcloud secrets versions add "$key" --data-file=- 2>/dev/null; then
            SECRET_NAMES+=("$key")
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
fi

# Deploy with anti-hang optimizations
echo "🚀 Deploying with anti-hang optimizations..."
if [ -n "$SECRET_REFS" ]; then
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 8Gi \
        --cpu 4 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 50 \
        --timeout 900 \
        --execution-environment gen2 \
        --set-env-vars "FLASK_ENV=production,BACKEND_VERSION=${BACKEND_VERSION},API_VERSION=${API_VERSION},ORCA_VERSION=${ORCA_VERSION},DETAILED_LOGGING=auto,GCP_OPTIMIZED=true" \
        --set-secrets "$SECRET_REFS"
else
    echo "⚠️  No secrets found, deploying without secrets"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 8Gi \
        --cpu 4 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 50 \
        --timeout 900 \
        --execution-environment gen2 \
        --set-env-vars "FLASK_ENV=production,BACKEND_VERSION=${BACKEND_VERSION},API_VERSION=${API_VERSION},ORCA_VERSION=${ORCA_VERSION},DETAILED_LOGGING=auto,GCP_OPTIMIZED=true"
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "✅ ANTI-HANG DEPLOYMENT COMPLETE!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "🔧 ANTI-HANG OPTIMIZATIONS DEPLOYED:"
echo "   ✅ Memory increased to 8Gi (from 4Gi)"
echo "   ✅ CPU increased to 4 cores (from 2)"
echo "   ✅ Concurrency reduced to 20 (from 40) to prevent resource contention"
echo "   ✅ Timeout increased to 900s (from 600s)"
echo "   ✅ Using gen2 execution environment for better performance"
echo "   ✅ Enhanced request logging and monitoring active"
echo ""
echo "🔍 MONITORING:"
echo "   Health check: ${SERVICE_URL}/health"
echo "   Logs: gcloud logs tail --follow --project=${PROJECT_ID} --resource-names=${SERVICE_NAME}"
echo ""
echo "💡 The service now has comprehensive hang protection and should be more stable!"

popd >/dev/null
