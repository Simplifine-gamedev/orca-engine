#!/bin/bash

# Godot Image Generation Service - Cloud Run Deployment Script

# Configuration
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else:
    echo "❌ Error: GCP project id not provided."
    echo "Usage: ./deploy_imagen.sh <GCP_PROJECT_ID> [SERVICE_NAME]"
    echo "       ./deploy_imagen.sh my-project-id                    # Uses default godot-imagen-service"
    echo "       ./deploy_imagen.sh my-project-id my-custom-service  # Uses custom service name"
    echo "   OR: Set GCP_PROJECT_ID and/or SERVICE_NAME environment variables"
    exit 1
fi

# Handle optional service name parameter or environment variable
if [ -n "$2" ]; then
    SERVICE_NAME="$2"
elif [ -n "$SERVICE_NAME" ]; then
    echo "📋 Using SERVICE_NAME from environment: $SERVICE_NAME"
else:
    SERVICE_NAME="godot-imagen-service"
fi

echo "📋 DEPLOYMENT CONFIGURATION:"
echo "   Service: $SERVICE_NAME (Image Generation Microservice)"
echo "   Purpose: Dedicated GPT Image generation for 2D Design Studio"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🎨 Deploying Godot Image Generation Service to Google Cloud Run"
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

# Create Dockerfile specifically for imagen service
echo "📝 Creating Dockerfile for imagen service..."
cat > Dockerfile.imagen << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only required files for imagen service
COPY requirements.txt .
COPY imagen_app.py .
COPY .env* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run the imagen service
CMD exec gunicorn --bind :8080 --workers 2 --threads 4 --timeout 300 imagen_app:app
EOF

echo "✅ Dockerfile.imagen created"

# Build and push the container image
echo "🏗️  Building container image..."
gcloud builds submit --tag ${IMAGE_NAME} -f Dockerfile.imagen .

# Upload secrets to GCP Secret Manager if .env exists
SECRET_REFS=""
if [ -f ".env" ]; then
    echo "📋 Found .env file, uploading secrets to GCP Secret Manager..."
    
    # Enable Secret Manager API
    gcloud services enable secretmanager.googleapis.com
    
    # Get project number for service account
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
    echo "🔧 Granting Secret Manager access to Cloud Run service account..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/secretmanager.secretAccessor"
    
    # Force DEV_MODE=false for cloud deployments
    echo "🔧 Overriding DEV_MODE=false for cloud deployment..."
    if echo -n "false" | gcloud secrets create "IMAGEN_DEV_MODE" --data-file=- --replication-policy="automatic" 2>/dev/null || \
       echo -n "false" | gcloud secrets versions add "IMAGEN_DEV_MODE" --data-file=- 2>/dev/null; then
        echo "✅ IMAGEN_DEV_MODE set to false for production"
    fi
    
    # Read .env and create secrets (with IMAGEN_ prefix to avoid conflicts)
    SECRET_NAMES=("IMAGEN_DEV_MODE")
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Skip DEV_MODE - we force it via IMAGEN_DEV_MODE
        [[ $key == "DEV_MODE" ]] && continue
        
        # Remove any quotes from value
        value=$(echo "$value" | sed 's/^"//;s/"$//')
        
        # Create secret with IMAGEN_ prefix for service isolation
        secret_name="IMAGEN_${key}"
        echo "🔐 Creating/updating secret: $secret_name"
        if echo -n "$value" | gcloud secrets create "$secret_name" --data-file=- --replication-policy="automatic" 2>/dev/null || \
           echo -n "$value" | gcloud secrets versions add "$secret_name" --data-file=- 2>/dev/null; then
            SECRET_NAMES+=("$secret_name")
        fi
    done < .env
    
    # Build secret references for deployment
    for secret in "${SECRET_NAMES[@]}"; do
        # Map IMAGEN_OPENAI_API_KEY back to OPENAI_API_KEY env var in container
        env_var_name="${secret#IMAGEN_}"
        if [ -z "$SECRET_REFS" ]; then
            SECRET_REFS="${env_var_name}=${secret}:latest"
        else:
            SECRET_REFS="${SECRET_REFS},${env_var_name}=${secret}:latest"
        fi
    done
    
    echo "✅ Secrets uploaded to GCP Secret Manager"
    echo "🔗 Secret references: $SECRET_REFS"
fi

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
if [ -n "$SECRET_REFS" ]; then
    echo "🔐 Using secrets: $SECRET_REFS"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 1 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 50 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production" \
        --set-secrets "$SECRET_REFS"
else
    echo "⚠️  No secrets found, deploying without secrets"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 1 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 50 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production"
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "📋 Endpoints available:"
echo "   POST ${SERVICE_URL}/api/image/generate - Generate image from text"
echo "   POST ${SERVICE_URL}/api/image/edit - Edit/combine images"
echo "   POST ${SERVICE_URL}/api/spritesheet/generate - Generate sprite sheets"
echo "   POST ${SERVICE_URL}/api/image/mask/generate - Generate masks"
echo "   GET  ${SERVICE_URL}/health - Health check"
echo ""
echo "💡 Update your Godot editor to use this URL for image generation"
echo ""
echo "📋 To view logs:"
echo "gcloud logs tail --follow --project=${PROJECT_ID} --resource-names=${SERVICE_NAME}"

# Clean up temporary Dockerfile
rm -f Dockerfile.imagen

popd >/dev/null


