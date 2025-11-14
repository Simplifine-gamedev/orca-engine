#!/bin/bash

# Deploy logging server code to VM
PROJECT_ID="${1:-eastern-rider-436701-f4}"
VM_NAME="${2:-godot-logging-server-vm}"
ZONE="${3:-us-central1-c}"

echo "📦 Deploying logging server code..."

# Create deployment package
tar -czf logging_server.tar.gz simple_logging_server.py .env 2>/dev/null

# Upload to VM
echo "📤 Uploading to VM..."
gcloud compute scp logging_server.tar.gz $VM_NAME:/tmp/ --zone=$ZONE --project=$PROJECT_ID

# Deploy on VM
echo "⚙️  Deploying on VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e
echo '🔄 Deploying logging server...'

# Stop service if running
sudo systemctl stop logging-server 2>/dev/null || true

# Extract code
cd /opt/logging-server
sudo tar -xzf /tmp/logging_server.tar.gz
sudo chown -R www-data:www-data /opt/logging-server

# Start service
sudo systemctl daemon-reload
sudo systemctl enable logging-server
sudo systemctl start logging-server

# Check status
sleep 2
sudo systemctl status logging-server --no-pager

if sudo systemctl is-active logging-server > /dev/null; then
    echo '✅ Logging server deployed and running'
else
    echo '❌ Logging server failed to start'
    sudo journalctl -u logging-server --lines=10 --no-pager
fi
"

# Clean up
rm -f logging_server.tar.gz

# Get VM IP
EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT_ID --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "✅ LOGGING SERVER DEPLOYED!"
echo "🌐 Server URL: http://$EXTERNAL_IP:8082"
echo "🩺 Health check: curl http://$EXTERNAL_IP:8082/health"
echo "📊 Stats: curl http://$EXTERNAL_IP:8082/stats"
echo ""
echo "🔧 Update your main backend .env:"
echo "LOGGING_SERVER_URL=http://$EXTERNAL_IP:8082"

