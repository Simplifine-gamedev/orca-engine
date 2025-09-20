#!/bin/bash

# Quick test script for the logging service

if [ -z "$1" ]; then
    echo "Usage: ./test_logging.sh <LOGGING_SERVICE_URL>"
    echo "Example: ./test_logging.sh https://godot-ai-logging-service-xxx.run.app"
    exit 1
fi

LOGGING_URL="$1"

echo "🧪 Testing Logging Service: $LOGGING_URL"
echo "=========================================="

# Test health endpoint
echo "🔍 Health Check:"
curl -s "$LOGGING_URL/health" | python3 -m json.tool 2>/dev/null || curl -s "$LOGGING_URL/health"
echo -e "\n"

# Test stats endpoint  
echo "🔍 Stats Check:"
curl -s "$LOGGING_URL/stats" | python3 -m json.tool 2>/dev/null || curl -s "$LOGGING_URL/stats"
echo -e "\n"

# Test log submission
echo "🔍 Test Log Submission:"
curl -X POST -H "Content-Type: application/json" \
    -d '{"request_id":"test_123","event_type":"test","model":"gpt-4","success":true}' \
    "$LOGGING_URL/webhook/litellm" | python3 -m json.tool 2>/dev/null
echo -e "\n"

echo "✅ Test completed!"
