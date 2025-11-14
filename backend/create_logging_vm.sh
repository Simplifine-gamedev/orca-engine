#!/bin/bash

# Create a simple VM for logging server
PROJECT_ID="${1:-eastern-rider-436701-f4}"
VM_NAME="${2:-godot-logging-server-vm}"
ZONE="${3:-us-central1-c}"

echo "🚀 Creating simple logging server VM..."
echo "Project: $PROJECT_ID"
echo "VM: $VM_NAME"
echo "Zone: $ZONE"

# Create VM instance
gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=e2-micro \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --metadata=enable-oslogin=true \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=976792908107-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=logging-disk,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20241115,mode=rw,size=20,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-standard \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=purpose=logging-server \
    --reservation-affinity=any

# Wait for VM to start
echo "⏳ Waiting for VM to start..."
gcloud compute instances wait-until-running $VM_NAME --zone=$ZONE --project=$PROJECT_ID

# Get VM IP
EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT_ID --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo "✅ VM created successfully!"
echo "🌐 External IP: $EXTERNAL_IP"
echo ""
echo "Next steps:"
echo "1. Setup VM: ./setup_logging_vm.sh $PROJECT_ID $VM_NAME $ZONE"
echo "2. Deploy code: ./deploy_logging_vm.sh $PROJECT_ID $VM_NAME $ZONE"
echo "3. Update backend: LOGGING_SERVER_URL=http://$EXTERNAL_IP:8082"

