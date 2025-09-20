#!/bin/bash

# LiteLLM Logging Server - Cloud Run Deployment Script

# Configuration
# PROJECT_ID must be provided as arg or via GCP_PROJECT_ID env var.
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else
    echo "❌ Error: GCP project id not provided."
    echo "Usage: ./deploy_logger.sh <GCP_PROJECT_ID> (or set GCP_PROJECT_ID env var)"
    exit 1
fi
SERVICE_NAME="litellm-logging-server"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying LiteLLM Logging Server to Google Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Service Name: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Always run from this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "$SCRIPT_DIR" >/dev/null

# Check if gcloud is authenticated
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

# Create Dockerfile for logging server
echo "📝 Creating Dockerfile for logging server..."
cat > Dockerfile.logger << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add aiohttp for async Supabase calls
RUN pip install --no-cache-dir aiohttp

# Copy logging server
COPY logging_server.py .
COPY .env* ./

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8081

# Run the logging server
CMD ["python", "logging_server.py"]
EOF

# Build and push the container image
echo "🏗️  Building logging server container image..."
gcloud builds submit --tag ${IMAGE_NAME} --file Dockerfile.logger

# Upload Supabase secrets to GCP Secret Manager if .env exists
SECRET_REFS=""
if [ -f ".env" ]; then
    echo "📋 Found .env file, uploading Supabase secrets to GCP Secret Manager..."
    
    # Enable Secret Manager API
    gcloud services enable secretmanager.googleapis.com
    
    # Get project number for service account
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
    echo "🔧 Granting Secret Manager access to Cloud Run service account..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/secretmanager.secretAccessor"
    
    # Create secrets for Supabase configuration
    SUPABASE_SECRETS=("SUPABASE_URL" "SUPABASE_SERVICE_KEY" "SUPABASE_TABLE_NAME")
    SECRET_NAMES=()
    
    for secret_name in "${SUPABASE_SECRETS[@]}"; do
        # Check if secret exists in .env
        if grep -q "^${secret_name}=" .env 2>/dev/null; then
            secret_value=$(grep "^${secret_name}=" .env | head -n1 | cut -d'=' -f2- | sed 's/^"//;s/"$//')
            
            if [ -n "$secret_value" ]; then
                echo "🔐 Creating/updating secret: $secret_name"
                if echo -n "$secret_value" | gcloud secrets create "$secret_name" --data-file=- --replication-policy="automatic" 2>/dev/null || \
                   echo -n "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=- 2>/dev/null; then
                    SECRET_NAMES+=("$secret_name")
                fi
            fi
        fi
    done
    
    # Build secret references dynamically
    for secret in "${SECRET_NAMES[@]}"; do
        if [ -z "$SECRET_REFS" ]; then
            SECRET_REFS="${secret}=${secret}:latest"
        else
            SECRET_REFS="${SECRET_REFS},${secret}=${secret}:latest"
        fi
    done
    
    echo "✅ Supabase secrets uploaded to GCP Secret Manager"
    echo "🔗 Secret references: $SECRET_REFS"
fi

# Deploy to Cloud Run with secret references
echo "🚀 Deploying logging server to Cloud Run..."
if [ -n "$SECRET_REFS" ]; then
    echo "🔐 Using secrets: $SECRET_REFS"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8081 \
        --memory 1Gi \
        --cpu 1 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production" \
        --set-secrets "$SECRET_REFS"
else
    echo "⚠️  No Supabase secrets found, deploying without secrets"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8081 \
        --memory 1Gi \
        --cpu 1 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production"
fi

# Get the service URL
LOGGER_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo "✅ Logging server deployment complete!"
echo "🌐 Logging Server URL: ${LOGGER_URL}"
echo ""
echo "📝 Add this to your main app's .env file:"
echo "LOGGING_SERVER_URL=${LOGGER_URL}"
echo ""
echo "🧪 Test the logging server:"
echo "curl -X POST ${LOGGER_URL}/test"
echo "curl ${LOGGER_URL}/health"
echo ""
echo "📋 To view logs:"
echo "gcloud logs tail --follow --project=${PROJECT_ID} --resource-names=${SERVICE_NAME}"
echo ""
echo "🗃️  Remember to create the Supabase table with this SQL:"
echo "Run: create_supabase_table.sql"

# Clean up temporary files
rm -f Dockerfile.logger

popd >/dev/null

