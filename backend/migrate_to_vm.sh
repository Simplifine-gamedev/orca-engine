#!/bin/bash

# Complete migration from Cloud Run to dedicated VM
# This script handles everything: VM creation + code deployment + service start

set -e

PROJECT_ID="${1:-eastern-rider-436701-f4}"
INSTANCE_NAME="${2:-godot-ai-backend-vm}"
ZONE="${3:-us-central1-c}"

echo "🔄 COMPLETE MIGRATION FROM CLOUD RUN TO DEDICATED VM"
echo "This will eliminate ALL streaming connection issues!"
echo ""
echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Instance: $INSTANCE_NAME"
echo "   Zone: $ZONE"
echo "   Workers: 24 (3x CPU cores for maximum throughput)"
echo ""

read -p "Continue with migration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Migration cancelled"
    exit 0
fi

# Step 1: Create the VM
echo "🏗️  Step 1/4: Creating VM instance..."
./deploy_vm_instance.sh "$PROJECT_ID" "$INSTANCE_NAME" "$ZONE"

# Step 2: Wait for VM to be ready
echo "⏳ Step 2/4: Waiting for VM to fully initialize..."
sleep 60

# Step 3: Deploy the backend code
echo "📦 Step 3/4: Deploying backend code to VM..."

# Get VM IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo "🔗 VM IP: $EXTERNAL_IP"

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf backend_package.tar.gz *.py *.txt *.md requirements.txt 2>/dev/null || true

# Copy files to VM
echo "📤 Uploading backend code..."
gcloud compute scp backend_package.tar.gz $INSTANCE_NAME:/tmp/ --zone=$ZONE --project=$PROJECT_ID

# Copy environment file if it exists
if [ -f ".env" ]; then
    echo "🔐 Uploading environment configuration..."
    gcloud compute scp .env $INSTANCE_NAME:/tmp/backend.env --zone=$ZONE --project=$PROJECT_ID
fi

# SSH in and set up the application
echo "⚙️  Setting up application on VM..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e
echo '🐧 VM SETUP: Extracting and configuring backend...'

# Extract the code
cd /opt/godot-ai-backend
sudo tar -xzf /tmp/backend_package.tar.gz
sudo chown -R www-data:www-data /opt/godot-ai-backend

# Set up environment variables
if [ -f '/tmp/backend.env' ]; then
    sudo cp /tmp/backend.env /opt/godot-ai-backend/.env
    sudo chown www-data:www-data /opt/godot-ai-backend/.env
    echo '🔐 Environment variables configured'
fi

# Install Python dependencies
cd /opt/godot-ai-backend
source venv/bin/activate
pip install -r requirements.txt

echo '✅ Backend application ready'
"

# Step 4: Start the service
echo "🚀 Step 4/4: Starting the Godot AI Backend service..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
sudo systemctl start godot-ai-backend
sleep 5
sudo systemctl status godot-ai-backend --no-pager
echo ''
echo '📊 Service Status:'
sudo systemctl is-active godot-ai-backend && echo '✅ Service is running' || echo '❌ Service failed to start'
"

# Clean up local files
rm -f backend_package.tar.gz

echo ""
echo "✅ MIGRATION COMPLETE!"
echo ""
echo "🌐 NEW BACKEND URL: http://$EXTERNAL_IP:8080"  
echo "🔥 24 Gunicorn workers handling concurrent requests"
echo "🛡️  No more Cloud Run streaming issues!"
echo ""
echo "🔧 MANAGEMENT COMMANDS:"
echo "   Monitor logs:    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command=\"sudo journalctl -u godot-ai-backend -f\""
echo "   Restart service: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command=\"sudo systemctl restart godot-ai-backend\""
echo "   Check status:    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command=\"sudo systemctl status godot-ai-backend\""
echo "   Update code:     ./update_vm_backend.sh $PROJECT_ID $INSTANCE_NAME $ZONE"
echo ""
echo "🌍 UPDATE YOUR CLIENT:"
echo "   Change backend URL from Cloud Run to: http://$EXTERNAL_IP:8080"
echo "   Or set up a custom domain pointing to this IP"
echo ""
echo "💰 COST OPTIMIZATION:"
echo "   This VM costs ~$200/month but eliminates ALL streaming issues"
echo "   Much more reliable than Cloud Run for persistent AI workloads"
