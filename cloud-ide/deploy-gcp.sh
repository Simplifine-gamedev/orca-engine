#!/bin/bash

# Deploy Orca Cloud IDE to Google Cloud Platform

PROJECT_ID="new-website-85890"
REGION="us-central1"
IMAGE_NAME="orca-cloud"
SERVICE_NAME="orca-cloud-backend"

echo "🚀 Deploying Orca Cloud IDE to GCP..."

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs if not already enabled
echo "📦 Enabling required APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com \
  cloudbuild.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "📦 Creating Artifact Registry..."
gcloud artifacts repositories create orca-cloud \
  --repository-format=docker \
  --location=$REGION \
  --description="Orca Cloud IDE Docker images" 2>/dev/null || true

# Configure Docker auth
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push backend Docker image
echo "🏗️ Building backend Docker image..."
docker build -f docker/Dockerfile.orca-cloud -t $IMAGE_NAME .

# Tag for Artifact Registry
docker tag $IMAGE_NAME ${REGION}-docker.pkg.dev/${PROJECT_ID}/orca-cloud/${IMAGE_NAME}:latest

# Push to Artifact Registry
echo "⬆️ Pushing image to Artifact Registry..."
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/orca-cloud/${IMAGE_NAME}:latest

# Deploy backend to Cloud Run (with GPU support if available)
echo "🚀 Deploying backend to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/orca-cloud/${IMAGE_NAME}:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --memory=4Gi \
  --cpu=2 \
  --max-instances=10 \
  --min-instances=1

# For GPU support, we need to use GCE instead of Cloud Run
# Uncomment below for GPU instances:

# echo "🖥️ Creating GPU instance template..."
# gcloud compute instance-templates create orca-gpu-template \
#   --machine-type=n1-standard-4 \
#   --accelerator=type=nvidia-tesla-t4,count=1 \
#   --maintenance-policy=TERMINATE \
#   --image-family=ubuntu-2204-lts \
#   --image-project=ubuntu-os-cloud \
#   --boot-disk-size=50GB \
#   --metadata-from-file startup-script=startup-gpu.sh

# Deploy frontend to Cloud Run
echo "🎨 Building frontend..."
cd cloud-ide/frontend
npm install
npm run build

# Create frontend Dockerfile
cat > Dockerfile <<EOF
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY .next ./.next
COPY public ./public
COPY next.config.js ./
EXPOSE 3000
CMD ["npm", "start"]
EOF

# Build and deploy frontend
docker build -t orca-frontend .
docker tag orca-frontend ${REGION}-docker.pkg.dev/${PROJECT_ID}/orca-cloud/orca-frontend:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/orca-cloud/orca-frontend:latest

gcloud run deploy orca-cloud-frontend \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/orca-cloud/orca-frontend:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --port=3000 \
  --memory=512Mi \
  --cpu=1 \
  --max-instances=100 \
  --min-instances=0 \
  --set-env-vars="NEXT_PUBLIC_API_URL=https://${SERVICE_NAME}-$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)' | sed 's|https://||')"

# Get URLs
BACKEND_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
FRONTEND_URL=$(gcloud run services describe orca-cloud-frontend --region=${REGION} --format='value(status.url)')

echo "✅ Deployment complete!"
echo "Backend URL: $BACKEND_URL"
echo "Frontend URL: $FRONTEND_URL"
echo ""
echo "Visit $FRONTEND_URL to access Orca Cloud IDE"
