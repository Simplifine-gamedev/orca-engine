#!/bin/bash

# Switch api.orcaengine.ai from Cloud Run to VM backend
# This preserves your domain so Godot clients need no changes

PROJECT_ID="eastern-rider-436701-f4"
DOMAIN="api.orcaengine.ai"
VM_IP="34.72.233.19"

echo "🔄 SWITCHING DOMAIN FROM CLOUD RUN TO VM"
echo "Domain: $DOMAIN"  
echo "From: Cloud Run godot-ai-backend-v2"
echo "To: VM at $VM_IP"
echo ""

read -p "This will redirect ALL traffic from Cloud Run to VM. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 0
fi

# Step 1: Remove existing Cloud Run domain mapping
echo "🗑️  Step 1: Removing Cloud Run domain mapping..."
gcloud beta run domain-mappings delete $DOMAIN --project=$PROJECT_ID --region=us-central1 --quiet

echo "✅ Cloud Run domain mapping removed"

# Step 2: Set up Load Balancer pointing to VM
echo "🔧 Step 2: Creating Load Balancer for VM..."

# Create health check
gcloud compute health-checks create http godot-ai-health-check \
    --port 8080 \
    --request-path /health \
    --check-interval 30s \
    --timeout 10s \
    --project=$PROJECT_ID \
    --global 2>/dev/null || echo "Health check already exists"

# Create instance group
gcloud compute instance-groups unmanaged create godot-ai-instance-group \
    --zone us-central1-c \
    --project=$PROJECT_ID 2>/dev/null || echo "Instance group already exists"

gcloud compute instance-groups unmanaged add-instances godot-ai-instance-group \
    --instances godot-ai-backend-vm \
    --zone us-central1-c \
    --project=$PROJECT_ID 2>/dev/null || echo "Instance already in group"

# Set named port
gcloud compute instance-groups unmanaged set-named-ports godot-ai-instance-group \
    --named-ports http:8080 \
    --zone us-central1-c \
    --project=$PROJECT_ID

# Create backend service
gcloud compute backend-services create godot-ai-backend-service \
    --protocol HTTP \
    --port-name http \
    --health-checks godot-ai-health-check \
    --timeout 300s \
    --global \
    --project=$PROJECT_ID 2>/dev/null || echo "Backend service already exists"

# Add backend
gcloud compute backend-services add-backend godot-ai-backend-service \
    --instance-group godot-ai-instance-group \
    --instance-group-zone us-central1-c \
    --max-rate 1000 \
    --global \
    --project=$PROJECT_ID 2>/dev/null || echo "Backend already added"

# Create URL map
gcloud compute url-maps create godot-ai-url-map \
    --default-service godot-ai-backend-service \
    --global \
    --project=$PROJECT_ID 2>/dev/null || echo "URL map already exists"

# Create SSL certificate for HTTPS
gcloud compute ssl-certificates create godot-ai-ssl-cert \
    --domains=$DOMAIN \
    --global \
    --project=$PROJECT_ID 2>/dev/null || echo "SSL cert already exists"

# Create HTTPS proxy  
gcloud compute target-https-proxies create godot-ai-https-proxy \
    --url-map godot-ai-url-map \
    --ssl-certificates godot-ai-ssl-cert \
    --global \
    --project=$PROJECT_ID 2>/dev/null || echo "HTTPS proxy already exists"

# Create forwarding rule
gcloud compute forwarding-rules create godot-ai-https-rule \
    --global \
    --target-https-proxy godot-ai-https-proxy \
    --ports 443 \
    --project=$PROJECT_ID 2>/dev/null || echo "Forwarding rule already exists"

# Get load balancer IP
LB_IP=$(gcloud compute forwarding-rules describe godot-ai-https-rule --global --project=$PROJECT_ID --format="get(IPAddress)")

echo ""
echo "✅ LOAD BALANCER CREATED!"
echo "🌐 Load Balancer IP: $LB_IP"
echo ""
echo "🔧 FINAL DNS UPDATE NEEDED:"
echo "   Go to your DNS provider and update:"
echo "   Record: api.orcaengine.ai"
echo "   From: ghs.googlehosted.com (Cloud Run)"  
echo "   To: $LB_IP (Load Balancer → VM)"
echo ""
echo "⏱️  DNS propagation takes 5-60 minutes"
echo "🧪 Test: curl https://api.orcaengine.ai/health"
echo ""
echo "🎯 RESULT: Your Godot clients will seamlessly switch to VM backend!"
echo "   NO code changes needed in ai_chat_dock.cpp"
