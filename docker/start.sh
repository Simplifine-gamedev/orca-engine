#!/bin/bash

# Generate self-signed certificate for websockify
openssl req -new -x509 -days 365 -nodes \
    -out /tmp/self.pem -keyout /tmp/self.pem \
    -subj "/C=US/ST=CA/L=SF/O=Orca/CN=localhost" 2>/dev/null

# Create VNC password (optional, remove -nopw from x11vnc if you want to use it)
# echo "orca" | vncpasswd -f > /root/.vnc/passwd
# chmod 600 /root/.vnc/passwd

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
