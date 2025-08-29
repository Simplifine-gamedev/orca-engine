#!/bin/bash

# Start supervisor to manage all services
echo "Starting Orca Engine Cloud IDE..."
echo "Resolution: ${RESOLUTION}"
echo "Display: ${DISPLAY}"

# Create necessary directories
mkdir -p /workspace/projects
mkdir -p /var/log/supervisor

# Start all services via supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
