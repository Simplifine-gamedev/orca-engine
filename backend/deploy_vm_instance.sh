#!/bin/bash

# Godot AI Backend - Dedicated VM Instance Deployment
# This creates a dedicated GCP VM with proper CPU/memory and 20+ Gunicorn workers

set -e  # Exit on any error

# Configuration
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$GCP_PROJECT_ID" ]; then
    PROJECT_ID="$GCP_PROJECT_ID"
else
    echo "❌ Error: GCP project id required"
    echo "Usage: ./deploy_vm_instance.sh <PROJECT_ID> [INSTANCE_NAME] [ZONE]"
    exit 1
fi

INSTANCE_NAME="${2:-godot-ai-backend-vm}"
ZONE="${3:-us-central1-c}"
MACHINE_TYPE="n1-standard-8"  # 8 vCPUs, 30GB RAM - perfect for 20+ workers

echo "🚀 CREATING DEDICATED VM FOR GODOT AI BACKEND"
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"  
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE (8 vCPUs, 30GB RAM)"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create startup script for the VM
cat > startup_script.sh << 'EOF'
#!/bin/bash
set -e

echo "🐧 VM STARTUP: Installing dependencies..."

# Update system
apt-get update
apt-get install -y python3 python3-pip python3-venv git nginx supervisor

# Create app directory
mkdir -p /opt/godot-ai-backend
cd /opt/godot-ai-backend

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Clone/copy the backend code (you'll need to modify this for your repo)
# For now, create placeholder structure
mkdir -p app
cat > requirements.txt << 'REQ_EOF'
flask==3.0.0
flask-cors==4.0.0
openai==1.12.0
litellm==1.54.0
gunicorn==21.2.0
requests==2.31.0
python-dotenv==1.0.0
pillow==10.2.0
REQ_EOF

# Install dependencies
pip install -r requirements.txt

# Create Gunicorn configuration
cat > gunicorn.conf.py << 'GUNICORN_EOF'
# Gunicorn configuration for Godot AI Backend
bind = "0.0.0.0:8080"
workers = 24  # 3x CPU cores (8 cores * 3 = 24 workers)
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 300  # 5 minutes
keepalive = 10
preload_app = True

# Performance optimizations
worker_tmp_dir = "/dev/shm"  # Use RAM for worker temp files
tmp_upload_dir = "/dev/shm"

# Logging
accesslog = "/var/log/godot-ai/access.log"
errorlog = "/var/log/godot-ai/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "godot-ai-backend"

# Security
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 8192
GUNICORN_EOF

# Create log directory
mkdir -p /var/log/godot-ai
chmod 755 /var/log/godot-ai

# Create systemd service
cat > /etc/systemd/system/godot-ai-backend.service << 'SERVICE_EOF'
[Unit]
Description=Godot AI Backend Flask Application
After=network.target
Wants=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/godot-ai-backend
Environment=PATH=/opt/godot-ai-backend/venv/bin
Environment=PYTHONPATH=/opt/godot-ai-backend
Environment=FLASK_ENV=production
ExecStart=/opt/godot-ai-backend/venv/bin/gunicorn -c gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
PrivateTmp=true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=godot-ai-backend

# Security settings
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/godot-ai /tmp /dev/shm

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Set up permissions
chown -R www-data:www-data /opt/godot-ai-backend
chown -R www-data:www-data /var/log/godot-ai

# Enable and start the service (will fail initially until app.py is deployed)
systemctl daemon-reload
systemctl enable godot-ai-backend

echo "✅ VM STARTUP COMPLETE - Ready for app deployment"
EOF

echo "🏗️  Creating VM instance with startup script..."

# Create the VM instance
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=default \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server,godot-ai-backend \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20251023,mode=rw,size=50,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-standard \
    --metadata-from-file startup-script=startup_script.sh \
    --reservation-affinity=any

echo "⏳ Waiting for VM to start up and run initialization script..."
sleep 30

# Create firewall rules
echo "🔧 Creating firewall rules..."
gcloud compute firewall-rules create allow-godot-ai-backend \
    --project=$PROJECT_ID \
    --allow tcp:8080,tcp:80,tcp:443 \
    --source-ranges 0.0.0.0/0 \
    --target-tags godot-ai-backend \
    --description "Allow HTTP/HTTPS traffic to Godot AI Backend VM" \
    2>/dev/null || echo "Firewall rule already exists"

# Get the VM's external IP
echo "🌐 Getting VM external IP..."
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "✅ VM INSTANCE CREATED SUCCESSFULLY!"
echo "🌐 External IP: $EXTERNAL_IP"
echo "🔗 Backend URL: http://$EXTERNAL_IP:8080"
echo ""
echo "📋 INSTANCE SPECS:"
echo "   🖥️  Machine: $MACHINE_TYPE (8 vCPUs, 30GB RAM)"
echo "   ⚡ Workers: 24 Gunicorn workers configured"
echo "   🔄 Auto-restart: systemd service enabled"
echo "   📂 Logs: /var/log/godot-ai/"
echo ""
echo "🔧 NEXT STEPS:"
echo "1. Upload your backend code:"
echo "   scp -r . $USER@$EXTERNAL_IP:/opt/godot-ai-backend/app/"
echo ""
echo "2. Upload environment variables:"
echo "   scp .env $USER@$EXTERNAL_IP:/opt/godot-ai-backend/"
echo ""
echo "3. SSH in and start the service:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "   sudo systemctl start godot-ai-backend"
echo "   sudo systemctl status godot-ai-backend"
echo ""
echo "4. Monitor logs:"
echo "   sudo journalctl -u godot-ai-backend -f"
echo ""
echo "💡 This eliminates ALL Cloud Run streaming issues!"

# Clean up
rm -f startup_script.sh
