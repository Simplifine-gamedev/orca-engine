#!/bin/bash

# Setup the logging server VM with Python environment
PROJECT_ID="${1:-eastern-rider-436701-f4}"
VM_NAME="${2:-godot-logging-server-vm}"
ZONE="${3:-us-central1-c}"

echo "⚙️  Setting up logging server VM..."

# Setup commands to run on the VM
SETUP_COMMANDS="
set -e
echo '🔄 Setting up Python environment...'

# Update system
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv nginx ufw

# Create app directory
sudo mkdir -p /opt/logging-server
cd /opt/logging-server

# Create virtual environment  
sudo python3 -m venv venv
sudo ./venv/bin/pip install --upgrade pip

# Install dependencies
sudo ./venv/bin/pip install flask flask-cors python-dotenv requests

# Create systemd service
sudo tee /etc/systemd/system/logging-server.service > /dev/null <<EOF
[Unit]
Description=Godot AI Simple Logging Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/logging-server
Environment=PATH=/opt/logging-server/venv/bin
ExecStart=/opt/logging-server/venv/bin/python simple_logging_server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Configure firewall
sudo ufw allow 8082
sudo ufw allow ssh
sudo ufw --force enable

# Set permissions
sudo chown -R www-data:www-data /opt/logging-server

echo '✅ VM setup complete!'
"

# Execute setup on VM
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command="$SETUP_COMMANDS"

echo "✅ Logging VM setup complete!"
echo "🔧 Next: ./deploy_logging_vm.sh $PROJECT_ID $VM_NAME $ZONE"

