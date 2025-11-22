#!/usr/bin/env python3
"""
Direct test of image_operation_internal function
"""
import os
import sys
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Flask app context
from app import app, image_operation_internal

def test_image_generation():
    """Test image generation directly"""
    print("=" * 80)
    print("Testing Nano Banana Image Generation (Direct)")
    print("=" * 80)
    
    # Test arguments
    arguments = {
        "description": "A cute nano banana with a friendly smile, cartoon style",
        "style": "cartoon",
        "size": "1024x1024"
    }
    
    print(f"\nArguments:")
    print(json.dumps(arguments, indent=2))
    print("\nCalling image_operation_internal...")
    
    with app.app_context():
        try:
            result = image_operation_internal(arguments, conversation_messages=[])
            
            print(f"\n✅ Image operation completed!")
            print(f"   Success: {result.get('success')}")
            
            if result.get('success'):
                print(f"   Image ID: {result.get('image_id')}")
                print(f"   Width: {result.get('width')}px")
                print(f"   Height: {result.get('height')}px")
                print(f"   Format: {result.get('format')}")
                image_data_len = len(result.get('image_data', ''))
                print(f"   Image data length: {image_data_len} characters")
                print(f"   Estimated image size: {image_data_len * 3 // 4} bytes")
                
                if image_data_len > 0:
                    print(f"\n   ✅ Image data present! First 100 chars: {result.get('image_data', '')[:100]}...")
                else:
                    print(f"\n   ⚠️  No image data in result")
            else:
                print(f"   Error: {result.get('error')}")
            
            return result.get('success', False)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("\n🧪 Nano Banana Direct Image Operation Test")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠️  WARNING: GOOGLE_API_KEY or GEMINI_API_KEY not set!")
        print("   The test will fail if the API key is required.")
        print("   Continuing anyway...\n")
    
    # Run test
    success = test_image_generation()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
    print("=" * 80)
    
    sys.exit(0 if success else 1)

