#!/usr/bin/env python3
"""
Test script for the image generation functionality in the AI backend.
This script tests the backend's ability to handle the responses API and image generation.
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_image_generation():
    """Test image generation through the backend"""
    
    # Backend URL
    backend_url = "http://localhost:8000/chat"
    
    # Test message requesting image generation
    test_message = {
        "messages": [
            {
                "role": "user",
                "content": "Please generate an image of a cute cartoon cat wearing a wizard hat"
            }
        ]
    }
    
    print("Testing image generation...")
    print(f"Sending request to: {backend_url}")
    print(f"Message: {test_message['messages'][0]['content']}")
    print("\nResponse stream:")
    print("-" * 50)
    
    try:
        # Send request with streaming
        response = requests.post(
            backend_url,
            json=test_message,
            stream=True,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            return
            
        # Process streaming response
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    if data.get("status") == "streaming":
                        print(f"[TEXT] {data.get('content_delta', '')}", end="", flush=True)
                    
                    elif data.get("status") == "tool_completed":
                        tool_result = data.get("tool_result", {})
                        if data.get("tool_executed") == "generate_image" and tool_result.get("success"):
                            base64_data = tool_result.get("image_data", "")
                            prompt = tool_result.get("prompt", "unknown")
                            
                            print(f"\n[IMAGE] Generated image for prompt: {prompt} ({len(base64_data)} chars base64)")
                            
                            # Save the image
                            if base64_data:
                                import base64
                                try:
                                    with open(f"test_generated_image.png", "wb") as f:
                                        f.write(base64.b64decode(base64_data))
                                    print(f"[SAVED] Image saved as test_generated_image.png")
                                except Exception as e:
                                    print(f"[ERROR] Failed to save image: {e}")
                        else:
                            tool_name = data.get("tool_executed", "unknown")
                            print(f"\n[TOOL] {tool_name} completed: {tool_result.get('message', 'no message')}")
                    
                    elif data.get("status") == "image_generated":
                        # Handle the old image_generated format (if it still exists)
                        image_data = data.get("image_generated", {})
                        base64_data = image_data.get("base64_data", "")
                        image_id = image_data.get("id", "unknown")
                        
                        print(f"\n[IMAGE] Generated image {image_id} ({len(base64_data)} chars base64)")
                        
                        # Optionally save the image
                        if base64_data:
                            import base64
                            try:
                                with open(f"test_generated_{image_id}.png", "wb") as f:
                                    f.write(base64.b64decode(base64_data))
                                print(f"[SAVED] Image saved as test_generated_{image_id}.png")
                            except Exception as e:
                                print(f"[ERROR] Failed to save image: {e}")
                    
                    elif data.get("status") == "completed":
                        print(f"\n[COMPLETED] Conversation finished")
                        break
                    
                    else:
                        print(f"\n[OTHER] {data}")
                        
                except json.JSONDecodeError as e:
                    print(f"\n[JSON ERROR] {e}: {line}")
                    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")

if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file in the backend directory with your OpenAI API key")
        exit(1)
    
    print("🎨 AI Image Generation Backend Test")
    print("=" * 40)
    test_image_generation()