#!/bin/bash

# Godot AI Backend - TEST Cloud Run Deployment Script
# 🧪 This deploys to a separate test environment for validation before production

# Configuration
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else
    echo "❌ Error: GCP project id not provided."
    echo "Usage: ./deploy_for_test.sh <GCP_PROJECT_ID> [TEST_SERVICE_NAME]"
    echo "       ./deploy_for_test.sh my-project-id                    # Uses default test service name"
    echo "       ./deploy_for_test.sh my-project-id my-test-backend    # Uses custom test service name"
    echo "   OR: Set GCP_PROJECT_ID environment variable"
    exit 1
fi

# Handle optional test service name
if [ -n "$2" ]; then
    TEST_SERVICE_NAME="$2"
elif [ -n "$TEST_SERVICE_NAME" ]; then
    echo "📋 Using TEST_SERVICE_NAME from environment: $TEST_SERVICE_NAME"
else
    TEST_SERVICE_NAME="godot-ai-backend-test"  # Default test service name
fi

echo "🧪 TEST DEPLOYMENT STRATEGY:"
echo "   🧪 Test service: $TEST_SERVICE_NAME (for network resilience testing)"
echo "   🚀 Prod service: godot-ai-backend-v2 (deployed separately via deploy.sh)"
echo "   🔬 Purpose: Test network resilience, stream resumption, recovery APIs"
echo "   💡 Smaller resources, shorter timeouts, test-friendly config"

REGION="us-central1"
TEST_IMAGE_NAME="gcr.io/${PROJECT_ID}/${TEST_SERVICE_NAME}"

echo ""
echo "🧪 Deploying Godot AI Backend to TEST Cloud Run Environment"
echo "Project ID: ${PROJECT_ID}"
echo "Test Service: ${TEST_SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Purpose: Network Resilience Testing"

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
echo "📋 Setting GCP project for testing..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs for test environment..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Extract version information
echo "📋 Extracting versions for test deployment..."
if [ -f "../extract_versions.py" ] && [ -f "../version.py" ]; then
    eval $(python3 ../extract_versions.py env_vars)
    echo "📋 Test versions: Backend=$BACKEND_VERSION, API=$API_VERSION, Frontend=$FRONTEND_VERSION, Orca=$ORCA_VERSION"
else
    echo "⚠️  Using default test versions"
    BACKEND_VERSION="1.0.0-test"
    API_VERSION="1.0-test"
    FRONTEND_VERSION="1.0.0-test"
    ORCA_VERSION="1.0.0-test"
fi

# Build and push test container image
echo "🏗️  Building TEST container image..."
echo "📋 Test versions will be injected during deployment"
gcloud builds submit --tag ${TEST_IMAGE_NAME}

# Upload test secrets to GCP Secret Manager
TEST_SECRET_REFS=""
if [ -f ".env" ]; then
    echo "📋 Setting up test secrets (prefixed with 'TEST_')..."
    
    # Enable Secret Manager API
    gcloud services enable secretmanager.googleapis.com
    
    # Get service account for permissions
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
    echo "🔧 Granting test environment permissions..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${COMPUTE_SA}" \
        --role="roles/secretmanager.secretAccessor"
    
    # Force DEV_MODE=false for test cloud deployment (but allow test-specific features)
    echo "🔧 Setting TEST_DEV_MODE=false for cloud test..."
    if echo -n "false" | gcloud secrets create "TEST_DEV_MODE" --data-file=- --replication-policy="automatic" 2>/dev/null || \
       echo -n "false" | gcloud secrets versions add "TEST_DEV_MODE" --data-file=- 2>/dev/null; then
        echo "✅ TEST_DEV_MODE set to false"
    fi
    
    # CRITICAL FIX: Ensure FLASK_SECRET_KEY exists for test deployment
    echo "🔧 Ensuring TEST_FLASK_SECRET_KEY exists..."
    TEST_SECRET_KEY=$(openssl rand -hex 32)
    if echo -n "$TEST_SECRET_KEY" | gcloud secrets create "TEST_FLASK_SECRET_KEY" --data-file=- --replication-policy="automatic" 2>/dev/null || \
       echo -n "$TEST_SECRET_KEY" | gcloud secrets versions add "TEST_FLASK_SECRET_KEY" --data-file=- 2>/dev/null; then
        echo "✅ Generated TEST_FLASK_SECRET_KEY for test deployment"
    fi
    
    # Add test-specific configuration for easy testing
    echo "🧪 Adding test-friendly configuration..."
    
    # Enable guest sessions for easy testing
    if echo -n "true" | gcloud secrets create "TEST_ALLOW_GUESTS" --data-file=- --replication-policy="automatic" 2>/dev/null || \
       echo -n "true" | gcloud secrets versions add "TEST_ALLOW_GUESTS" --data-file=- 2>/dev/null; then
        echo "✅ TEST_ALLOW_GUESTS enabled for test environment"
    fi
    
    # Enable network resilience debugging
    if echo -n "true" | gcloud secrets create "TEST_NETWORK_RESILIENCE_DEBUG" --data-file=- --replication-policy="automatic" 2>/dev/null || \
       echo -n "true" | gcloud secrets versions add "TEST_NETWORK_RESILIENCE_DEBUG" --data-file=- 2>/dev/null; then
        echo "✅ TEST_NETWORK_RESILIENCE_DEBUG enabled for testing"
    fi
    
    # Create test secrets with TEST_ prefix to avoid conflicts with prod
    TEST_SECRET_NAMES=("TEST_DEV_MODE" "TEST_FLASK_SECRET_KEY" "TEST_ALLOW_GUESTS" "TEST_NETWORK_RESILIENCE_DEBUG")
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Skip DEV_MODE - we handle it above
        [[ $key == "DEV_MODE" ]] && continue
        
        # Prefix with TEST_ for isolation
        TEST_KEY="TEST_${key}"
        
        # Remove quotes from value
        value=$(echo "$value" | sed 's/^"//;s/"$//')
        
        echo "🔐 Creating/updating test secret: $TEST_KEY"
        if echo -n "$value" | gcloud secrets create "$TEST_KEY" --data-file=- --replication-policy="automatic" 2>/dev/null || \
           echo -n "$value" | gcloud secrets versions add "$TEST_KEY" --data-file=- 2>/dev/null; then
            TEST_SECRET_NAMES+=("$TEST_KEY")
        fi
    done < .env
    
    # Build test secret references
    for secret in "${TEST_SECRET_NAMES[@]}"; do
        if [ -z "$TEST_SECRET_REFS" ]; then
            TEST_SECRET_REFS="${secret}=${secret}:latest"
        else
            TEST_SECRET_REFS="${TEST_SECRET_REFS},${secret}=${secret}:latest"
        fi
    done
    
    echo "✅ Test secrets configured with TEST_ prefix for isolation"
    echo "🔗 Test secret references: $TEST_SECRET_REFS"
fi

# Deploy to TEST Cloud Run with test-optimized configuration
echo "🧪 Deploying to TEST Cloud Run environment..."
if [ -n "$TEST_SECRET_REFS" ]; then
    echo "🔐 Using test secrets: $TEST_SECRET_REFS"
    gcloud run deploy ${TEST_SERVICE_NAME} \
        --image ${TEST_IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 1 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=test,BACKEND_VERSION=${BACKEND_VERSION}-test,API_VERSION=${API_VERSION}-test,FRONTEND_VERSION=${FRONTEND_VERSION},ORCA_VERSION=${ORCA_VERSION},DETAILED_LOGGING=true,NETWORK_RESILIENCE_TEST=true" \
        --set-secrets "$TEST_SECRET_REFS"
else
    echo "⚠️  No test secrets found, deploying test environment without secrets"
    gcloud run deploy ${TEST_SERVICE_NAME} \
        --image ${TEST_IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 1 \
        --concurrency 20 \
        --min-instances 1 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=test,BACKEND_VERSION=${BACKEND_VERSION}-test,API_VERSION=${API_VERSION}-test,FRONTEND_VERSION=${FRONTEND_VERSION},ORCA_VERSION=${ORCA_VERSION},DETAILED_LOGGING=true,NETWORK_RESILIENCE_TEST=true"
fi

# Get the test service URL
TEST_SERVICE_URL=$(gcloud run services describe ${TEST_SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "🧪 TEST DEPLOYMENT COMPLETE!"
echo "🌐 Test Service URL: ${TEST_SERVICE_URL}"
echo "🔬 Test Environment Features:"
echo "   ✅ Network resilience system enabled"
echo "   ✅ Stream resumption testing"
echo "   ✅ Recovery API endpoints"
echo "   ✅ Tool completion verification"
echo "   ✅ Message receipt tracking"
echo "   ✅ Cloud-compatible state persistence"
echo ""
echo "🧪 TESTING INSTRUCTIONS:"
echo "1. Launch Godot in test mode:"
echo "   IS_DEV=true TEST_MODE=cloud ./bin/godot.macos.editor.arm64"
echo ""
echo "2. Test network interruption scenarios:"
echo "   • Disconnect internet mid-tool execution"
echo "   • Disconnect during AI streaming response"  
echo "   • Test with long operations (image generation)"
echo ""
echo "3. Monitor test logs:"
echo "   gcloud logs tail --follow --project=${PROJECT_ID} --resource-names=${TEST_SERVICE_NAME}"
echo ""
echo "🎯 SIMPLE MODE SWITCHING:"
echo "   🏠 Local:      IS_DEV=true ./bin/godot..."
echo "   🧪 Test:       IS_DEV=true TEST_MODE=cloud ./bin/godot..."
echo "   🚀 Production: ./bin/godot..."
echo ""
echo "✅ When testing passes, deploy to production with: ./deploy.sh ${PROJECT_ID}"
echo ""
echo "📊 Test vs Production Comparison:"
echo "   🧪 TEST:  ${TEST_SERVICE_NAME} - 2Gi RAM, 1 CPU, 1-10 instances"
echo "   🚀 PROD:  godot-ai-backend-v2 - 4Gi RAM, 2 CPU, 2-100 instances"

# Return to original directory
popd >/dev/null
