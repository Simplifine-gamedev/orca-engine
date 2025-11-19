#!/bin/bash

# Quick backend code update script for the VM instance

PROJECT_ID="${1:-eastern-rider-436701-f4}"
INSTANCE_NAME="${2:-godot-ai-backend-vm}"
ZONE="${3:-us-central1-c}"

echo "🔄 UPDATING BACKEND CODE ON VM"
echo "Instance: $INSTANCE_NAME"

# Create deployment package
echo "📦 Packaging updated code..."
tar -czf backend_update.tar.gz *.py *.txt *.md requirements.txt 2>/dev/null || true

# Upload to VM
echo "📤 Uploading to VM..."
gcloud compute scp backend_update.tar.gz $INSTANCE_NAME:/tmp/ --zone=$ZONE --project=$PROJECT_ID

# Update environment if it exists
if [ -f ".env" ]; then
    echo "🔐 Uploading updated environment..."
    gcloud compute scp .env $INSTANCE_NAME:/tmp/backend_update.env --zone=$ZONE --project=$PROJECT_ID
fi

# Deploy on VM
echo "⚙️  Deploying on VM..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e
echo '🔄 UPDATING BACKEND APPLICATION...'

# Stop service (ignore errors if already stopped)
sudo systemctl stop godot-ai-backend || true

# Extract new code
cd /opt/godot-ai-backend
sudo tar -xzf /tmp/backend_update.tar.gz
sudo chown -R www-data:www-data /opt/godot-ai-backend

# Update environment if provided
if [ -f '/tmp/backend_update.env' ]; then
    sudo cp /tmp/backend_update.env /opt/godot-ai-backend/.env
    sudo chown www-data:www-data /opt/godot-ai-backend/.env
    echo '🔐 Environment updated'
fi

# Update dependencies (continue even if this fails)
set +e
source venv/bin/activate
pip install -r requirements.txt
PIP_EXIT_CODE=\$?
set -e
if [ \$PIP_EXIT_CODE -ne 0 ]; then
    echo '⚠️  Warning: pip install had errors, but continuing with restart...'
fi

# Always restart service, even if pip install failed
echo '🔄 Restarting service...'
sudo systemctl restart godot-ai-backend || sudo systemctl start godot-ai-backend
sleep 5

# Check status
echo ''
echo '📊 Service Status:'
sudo systemctl status godot-ai-backend --no-pager || true
echo ''

# Verify service is running
if sudo systemctl is-active --quiet godot-ai-backend; then
    echo '✅ Backend updated and running successfully'
    # Test health endpoint
    sleep 2
    if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
        echo '✅ Health check passed'
    else
        echo '⚠️  Warning: Service is running but health check failed'
    fi
else
    echo '❌ Backend failed to start after update'
    echo '📋 Recent logs:'
    sudo journalctl -u godot-ai-backend --lines=20 --no-pager
    exit 1
fi
"

# Clean up
rm -f backend_update.tar.gz

# Get VM IP for reference
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "✅ BACKEND UPDATE COMPLETE!"
echo "🌐 Backend URL: http://$EXTERNAL_IP:8080"
echo "📊 Monitor: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command=\"sudo journalctl -u godot-ai-backend -f\""
