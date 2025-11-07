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
