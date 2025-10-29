#!/bin/bash

# Map api.orcaengine.ai domain to the new VM backend
# This replaces the Cloud Run domain mapping

PROJECT_ID="eastern-rider-436701-f4"
VM_IP="34.72.233.19"
DOMAIN="api.orcaengine.ai"

echo "🌐 MAPPING DOMAIN TO VM BACKEND"
echo "Domain: $DOMAIN"
echo "VM IP: $VM_IP"
echo "Project: $PROJECT_ID"
echo ""

# Step 1: Create HTTP(S) Load Balancer pointing to VM
echo "🔧 Step 1: Creating Load Balancer..."

# Create backend service
gcloud compute backend-services create godot-ai-backend-service \
    --protocol HTTP \
    --port-name http \
    --health-checks godot-ai-health-check \
    --global \
    --project=$PROJECT_ID

# Create health check
gcloud compute health-checks create http godot-ai-health-check \
    --port 8080 \
    --request-path /health \
    --project=$PROJECT_ID

# Create instance group for the VM
gcloud compute instance-groups unmanaged create godot-ai-instance-group \
    --zone us-central1-c \
    --project=$PROJECT_ID

gcloud compute instance-groups unmanaged add-instances godot-ai-instance-group \
    --instances godot-ai-backend-vm \
    --zone us-central1-c \
    --project=$PROJECT_ID

# Set named port for the instance group
gcloud compute instance-groups unmanaged set-named-ports godot-ai-instance-group \
    --named-ports http:8080 \
    --zone us-central1-c \
    --project=$PROJECT_ID

# Add instance group to backend service
gcloud compute backend-services add-backend godot-ai-backend-service \
    --instance-group godot-ai-instance-group \
    --instance-group-zone us-central1-c \
    --global \
    --project=$PROJECT_ID

# Create URL map
gcloud compute url-maps create godot-ai-url-map \
    --default-service godot-ai-backend-service \
    --global \
    --project=$PROJECT_ID

# Create HTTP(S) proxy
gcloud compute target-https-proxies create godot-ai-https-proxy \
    --url-map godot-ai-url-map \
    --ssl-certificates godot-ai-ssl-cert \
    --global \
    --project=$PROJECT_ID

# Get SSL certificate (you'll need to create this manually or use Let's Encrypt)
echo "🔐 Creating SSL certificate..."
gcloud compute ssl-certificates create godot-ai-ssl-cert \
    --domains=$DOMAIN \
    --global \
    --project=$PROJECT_ID

# Create global forwarding rule
gcloud compute forwarding-rules create godot-ai-https-rule \
    --global \
    --target-https-proxy godot-ai-https-proxy \
    --ports 443 \
    --project=$PROJECT_ID

# Get the load balancer IP
LB_IP=$(gcloud compute forwarding-rules describe godot-ai-https-rule --global --project=$PROJECT_ID --format="get(IPAddress)")

echo ""
echo "✅ LOAD BALANCER CREATED!"
echo "🌐 Load Balancer IP: $LB_IP"
echo ""
echo "🔧 FINAL STEP - UPDATE DNS:"
echo "   1. Go to your domain registrar (where you manage orcaengine.ai)"
echo "   2. Update the A record for 'api' subdomain:"
echo "      OLD: Points to Cloud Run (ghs.googlehosted.com)"
echo "      NEW: Points to $LB_IP"
echo ""
echo "🚀 After DNS update, $DOMAIN will route to your VM with 24 workers!"
echo "🛡️  No more Cloud Run streaming issues!"
