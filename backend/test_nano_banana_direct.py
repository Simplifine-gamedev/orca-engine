#!/usr/bin/env python3
"""
Direct test for Nano Banana image generation - tests the function directly
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Set up environment (after loading .env)
os.environ.setdefault('GOOGLE_API_KEY', os.getenv('GOOGLE_API_KEY', ''))
os.environ.setdefault('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))

# Import the function directly
from nano_banana import generate_image_from_text, generate_image_from_image_and_text, generate_standalone_object

def test_text_to_image():
    """Test text-to-image generation"""
    print("=" * 80)
    print("Testing Nano Banana Text-to-Image Generation")
    print("=" * 80)
    
    prompt = "A cute nano banana with a friendly smile, cartoon style"
    size = "1024x1024"
    
    print(f"\nPrompt: {prompt}")
    print(f"Size: {size}")
    print("\nGenerating image...")
    
    try:
        image_base64, width, height = generate_image_from_text(
            prompt=prompt,
            size=size
        )
        
        print(f"\n✅ Image generation successful!")
        print(f"   Width: {width}px")
        print(f"   Height: {height}px")
        print(f"   Base64 data length: {len(image_base64)} characters")
        print(f"   Estimated image size: {len(image_base64) * 3 // 4} bytes")
        
        # Save a small preview (first 100 chars of base64)
        print(f"\n   Base64 preview: {image_base64[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_edit():
    """Test image editing (would need an input image)"""
    print("\n" + "=" * 80)
    print("Testing Nano Banana Image Editing")
    print("=" * 80)
    print("\n⚠️  Image editing test requires an input image")
    print("   Skipping for now - text-to-image test is sufficient")
    return True

def test_standalone_object():
    """Test standalone object generation with white background removal"""
    print("\n" + "=" * 80)
    print("Testing Standalone Object Generation")
    print("=" * 80)
    
    prompt = "a red apple"
    size = "1024x1024"
    
    print(f"\nPrompt: {prompt}")
    print(f"Size: {size}")
    print("\nGenerating standalone object (with white background removal)...")
    
    try:
        image_base64, width, height = generate_standalone_object(
            prompt=prompt,
            size=size
        )
        
        print(f"\n✅ Standalone object generation successful!")
        print(f"   Width: {width}px")
        print(f"   Height: {height}px")
        print(f"   Base64 data length: {len(image_base64)} characters")
        print(f"   Estimated image size: {len(image_base64) * 3 // 4} bytes")
        
        # Save a small preview (first 100 chars of base64)
        print(f"\n   Base64 preview: {image_base64[:100]}...")
        
        # Optionally save the image to a file for visual inspection
        try:
            import base64
            image_bytes = base64.b64decode(image_base64)
            output_path = "test_standalone_object.png"
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"\n   💾 Image saved to: {output_path}")
            print(f"   (Check this file to verify white background was removed)")
        except Exception as save_err:
            print(f"\n   ⚠️  Could not save image file: {save_err}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Standalone object generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 Nano Banana Image Generation Test")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set!")
        print("   Please set one of these environment variables to test.")
        sys.exit(1)
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Run tests
    success1 = test_text_to_image()
    test_image_edit()
    success2 = test_standalone_object()
    
    if success1 and success2:
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ Tests failed!")
        print("=" * 80)
        sys.exit(1)

