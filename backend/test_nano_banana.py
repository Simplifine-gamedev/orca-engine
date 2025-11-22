#!/usr/bin/env python3
"""
Test script for Nano Banana image generation
"""
import requests
import json

# Test configuration
BACKEND_URL = "http://localhost:5050"
USER_ID = "5ecc1fdb-8f4d-4710-b3ef-a354938679c9"
EMAIL = "a.kavoosi1999@gmail.com"

def test_image_generation():
    """Test text-to-image generation"""
    url = f"{BACKEND_URL}/chat"
    
    # Generate a test machine_id
    import uuid
    machine_id = str(uuid.uuid4())
    
    headers = {
        "Content-Type": "application/json",
        "X-User-ID": USER_ID,
        "X-Machine-ID": machine_id,
        "X-Supabase-User-ID": USER_ID,
        "X-Supabase-Email": EMAIL
    }
    
    # Create a chat request with image_operation tool call
    payload = {
        "supabase_user_id": USER_ID,
        "user_id": USER_ID,
        "machine_id": machine_id,
        "messages": [
            {
                "role": "user",
                "content": "Generate an image of a cute nano banana"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "test_tool_call_123",
                        "type": "function",
                        "function": {
                            "name": "image_operation",
                            "arguments": json.dumps({
                                "description": "A cute nano banana with a friendly smile, cartoon style",
                                "style": "cartoon",
                                "size": "1024x1024"
                            })
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "test_tool_call_123",
                "name": "image_operation",
                "content": ""  # Empty, will be filled by backend
            }
        ],
        "model": "gpt-4o",
        "mode": "agent"
    }
    
    print(f"Testing image generation at {url}")
    print(f"User ID: {USER_ID}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nSending request...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("\nResponse (first 1000 chars):")
            content = response.text[:1000]
            print(content)
            
            # Try to parse NDJSON if it's streaming
            if '\n' in response.text:
                print("\n\nParsing NDJSON lines...")
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            data = json.loads(line)
                            if 'tool_result' in data:
                                result = data['tool_result']
                                if result.get('success'):
                                    print(f"\n✅ Image generation successful!")
                                    print(f"   Image ID: {result.get('image_id')}")
                                    print(f"   Size: {result.get('width')}x{result.get('height')}")
                                    print(f"   Image data length: {len(result.get('image_data', ''))}")
                                else:
                                    print(f"\n❌ Image generation failed: {result.get('error')}")
                        except json.JSONDecodeError:
                            pass
        else:
            print(f"\nError Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_generation()

