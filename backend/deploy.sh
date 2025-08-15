#!/bin/bash

# Godot AI Backend - Cloud Run Deployment Script

# Configuration
# PROJECT_ID must be provided as arg or via GCP_PROJECT_ID env var.
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else
    echo "❌ Error: GCP project id not provided."
    echo "Usage: ./deploy.sh <GCP_PROJECT_ID> (or set GCP_PROJECT_ID env var)"
    exit 1
fi
SERVICE_NAME="godot-ai-backend"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Godot AI Backend to Google Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Service Name: ${SERVICE_NAME}"
echo "Region: ${REGION}"

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
gcloud services enable bigquery.googleapis.com
# gcloud services enable aiplatform.googleapis.com  # Not needed for OpenAI embeddings

# Build and push the container image
echo "🏗️  Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}

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

    echo "🔧 Granting BigQuery access to Cloud Run service account..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/bigquery.user"
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/bigquery.dataEditor"
    
    # echo "🔧 Granting Vertex AI access to Cloud Run service account..."
    # gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    #     --member="serviceAccount:${COMPUTE_SA}" \
    #     --role="roles/aiplatform.user"
    
    # Optionally set OAuth redirect URI from env or .env; otherwise skip
    if grep -q '^OAUTH_REDIRECT_URI=' .env 2>/dev/null || [ -n "$OAUTH_REDIRECT_URI" ]; then
        echo "🔗 Setting OAuth redirect URI from provided configuration..."
        REDIRECT_VAL="${OAUTH_REDIRECT_URI}"
        if [ -z "$REDIRECT_VAL" ]; then
            REDIRECT_VAL=$(grep '^OAUTH_REDIRECT_URI=' .env | head -n1 | cut -d'=' -f2- | sed 's/^"//;s/"$//')
        fi
        if [ -n "$REDIRECT_VAL" ]; then
            if echo -n "$REDIRECT_VAL" | gcloud secrets create "OAUTH_REDIRECT_URI" --data-file=- --replication-policy="automatic" 2>/dev/null || \
               echo -n "$REDIRECT_VAL" | gcloud secrets versions add "OAUTH_REDIRECT_URI" --data-file=- 2>/dev/null; then
                echo "✅ OAUTH_REDIRECT_URI set"
            fi
        fi
    else
        echo "ℹ️  Skipping OAUTH_REDIRECT_URI auto-set (provide in .env or OAUTH_REDIRECT_URI env to configure)"
    fi
    
    # Read .env and create secrets
    SECRET_NAMES=("OAUTH_REDIRECT_URI")
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove any quotes from value
        value=$(echo "$value" | sed 's/^"//;s/"$//')
        
        echo "🔐 Creating/updating secret: $key"
        if echo -n "$value" | gcloud secrets create "$key" --data-file=- --replication-policy="automatic" 2>/dev/null || \
           echo -n "$value" | gcloud secrets versions add "$key" --data-file=- 2>/dev/null; then
            SECRET_NAMES+=("$key")
        fi
    done < .env
    
    # Build secret references dynamically
    for secret in "${SECRET_NAMES[@]}"; do
        if [ -z "$SECRET_REFS" ]; then
            SECRET_REFS="${secret}=${secret}:latest"
        else
            SECRET_REFS="${SECRET_REFS},${secret}=${secret}:latest"
        fi
    done
    
    echo "✅ Secrets uploaded to GCP Secret Manager with proper permissions"
    echo "🔗 Secret references: $SECRET_REFS"
fi

# Deploy to Cloud Run with secret references
echo "🚀 Deploying to Cloud Run..."
if [ -n "$SECRET_REFS" ]; then
    echo "🔐 Using secrets: $SECRET_REFS"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 4Gi \
        --cpu 2 \
        --concurrency 40 \
        --min-instances 2 \
        --max-instances 100 \
        --timeout 600 \
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
        --memory 4Gi \
        --cpu 2 \
        --concurrency 40 \
        --min-instances 2 \
        --max-instances 100 \
        --timeout 600 \
        --set-env-vars "FLASK_ENV=production"
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "🔐 Secrets are securely managed via GCP Secret Manager"
echo "💡 Your .env file values have been uploaded as secrets"
echo ""
echo "📋 To view logs:"
echo "gcloud logs tail --follow --project=${PROJECT_ID} --resource-names=${SERVICE_NAME}"
echo ""
echo "ℹ️  If using a custom domain, ensure DNS and redirect URIs are configured."