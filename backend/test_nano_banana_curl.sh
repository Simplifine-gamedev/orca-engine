#!/bin/bash
# Test Nano Banana image generation via curl

BACKEND_URL="http://localhost:5050"
USER_ID="5ecc1fdb-8f4d-4710-b3ef-a354938679c9"
EMAIL="a.kavoosi1999@gmail.com"
MACHINE_ID=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")

echo "Testing Nano Banana image generation"
echo "===================================="
echo "Backend URL: $BACKEND_URL"
echo "User ID: $USER_ID"
echo "Machine ID: $MACHINE_ID"
echo ""

# Create the JSON payload
PAYLOAD=$(cat <<EOF
{
  "supabase_user_id": "$USER_ID",
  "user_id": "$USER_ID",
  "machine_id": "$MACHINE_ID",
  "messages": [
    {
      "role": "user",
      "content": "Generate an image of a cute nano banana"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "test_tool_call_123",
          "type": "function",
          "function": {
            "name": "image_operation",
            "arguments": "{\"description\": \"A cute nano banana with a friendly smile, cartoon style\", \"style\": \"cartoon\", \"size\": \"1024x1024\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "test_tool_call_123",
      "name": "image_operation",
      "content": ""
    }
  ],
  "model": "gpt-4o",
  "mode": "agent"
}
EOF
)

echo "Sending request..."
echo ""

# Make the curl request
curl -X POST "$BACKEND_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: $USER_ID" \
  -H "X-Machine-ID: $MACHINE_ID" \
  -H "X-Supabase-User-ID: $USER_ID" \
  -H "X-Supabase-Email: $EMAIL" \
  -d "$PAYLOAD" \
  --no-buffer \
  -v 2>&1 | head -100

echo ""
echo ""
echo "Note: The response is NDJSON (newline-delimited JSON) stream."
echo "Look for lines containing 'tool_result' to see the image generation result."


