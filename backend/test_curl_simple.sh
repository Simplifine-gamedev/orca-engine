#!/bin/bash
# Simple curl test for image generation - triggers AI to call the tool

BACKEND_URL="http://localhost:5050"
USER_ID="5ecc1fdb-8f4d-4710-b3ef-a354938679c9"
EMAIL="a.kavoosi1999@gmail.com"
MACHINE_ID=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")

echo "Testing Nano Banana via chat endpoint"
echo "====================================="
echo "This will ask the AI to generate an image, which should trigger image_operation tool"
echo ""

# Simple request - just ask for an image
PAYLOAD=$(cat <<EOF
{
  "supabase_user_id": "$USER_ID",
  "user_id": "$USER_ID",
  "machine_id": "$MACHINE_ID",
  "messages": [
    {
      "role": "user",
      "content": "Please generate an image of a cute nano banana with a friendly smile in cartoon style, size 1024x1024"
    }
  ],
  "model": "gpt-4o",
  "mode": "agent"
}
EOF
)

echo "Sending request to generate image..."
echo ""

# Make the curl request and filter for tool results
curl -X POST "$BACKEND_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: $USER_ID" \
  -H "X-Machine-ID: $MACHINE_ID" \
  -H "X-Supabase-User-ID: $USER_ID" \
  -H "X-Supabase-Email: $EMAIL" \
  -d "$PAYLOAD" \
  --no-buffer \
  2>/dev/null | while IFS= read -r line; do
    if echo "$line" | grep -q "tool_result\|tool_executed\|image_id\|success"; then
      echo ">>> $line"
    fi
  done

echo ""
echo "Test complete. Check the output above for tool_result containing image data."


