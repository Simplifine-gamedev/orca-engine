#!/usr/bin/env python3
"""
Test the new image.create_isolated_object tool integration
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import the function
from app import create_isolated_object_internal

def test_isolated_object_creation():
    """Test creating a new isolated object"""
    print("=" * 70)
    print("Testing Isolated Object Creation (New Object)")
    print("=" * 70)
    
    arguments = {
        'object_description': 'a red apple icon',
        'size': '512x512',
        'white_threshold': 240,
        'target_resolution': 128,  # Test default downsampling
        'path_to_save': 'res://icons/apple_icon.png'
    }
    
    print(f"\nArguments: {json.dumps(arguments, indent=2)}")
    print("\nCreating isolated object...")
    
    try:
        result = create_isolated_object_internal(arguments, None)
        
        if result.get('success'):
            print(f"\n✅ Success!")
            print(f"   Image ID: {result.get('image_id')}")
            print(f"   Dimensions: {result.get('width')}x{result.get('height')}")
            if result.get('original_size'):
                print(f"   Original size: {result.get('original_size')}")
            print(f"   Target resolution: {result.get('target_resolution')}")
            print(f"   Format: {result.get('format')}")
            print(f"   Has transparent background: {result.get('has_transparent_background')}")
            print(f"   Path to save: {result.get('path_to_save')}")
            print(f"   Base64 length: {len(result.get('image_data', ''))}")
            return True
        else:
            print(f"\n❌ Failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_isolated_object_from_existing():
    """Test creating isolated object from existing image"""
    print("\n" + "=" * 70)
    print("Testing Isolated Object Creation (From Existing Image)")
    print("=" * 70)
    
    # Test with a file path (this will test the transparent_bg_image_gen function)
    image_path = '/Users/alikavoosi/Desktop/3d-design/GODOT/godot/backend/draw_sv.jpeg'
    if not os.path.exists(image_path):
        print(f"\n⚠️  Test image not found: {image_path}")
        print("   Skipping this test")
        return True
    
    arguments = {
        'object_description': 'convert this drawing into a polished 2D platformer level icon',
        'input_image_path': image_path,
        'size': '512x512',
        'white_threshold': 230,
        'target_resolution': 64,  # Test smaller resolution for level icon
        'path_to_save': 'res://icons/level_icon.png'
    }
    
    print(f"\nArguments: {json.dumps(arguments, indent=2)}")
    print("\nCreating isolated object from existing image...")
    
    try:
        result = create_isolated_object_internal(arguments, None)
        
        if result.get('success'):
            print(f"\n✅ Success!")
            print(f"   Image ID: {result.get('image_id')}")
            print(f"   Dimensions: {result.get('width')}x{result.get('height')}")
            if result.get('original_size'):
                print(f"   Original size: {result.get('original_size')}")
            print(f"   Target resolution: {result.get('target_resolution')}")
            print(f"   Input image: {result.get('input_image')}")
            print(f"   Has transparent background: {result.get('has_transparent_background')}")
            print(f"   Path to save: {result.get('path_to_save')}")
            print(f"   Base64 length: {len(result.get('image_data', ''))}")
            return True
        else:
            print(f"\n❌ Failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 Isolated Object Tool Integration Test")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set!")
        print("   Please set one of these environment variables to test.")
        sys.exit(1)
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Run tests
    success1 = test_isolated_object_creation()
    success2 = test_isolated_object_from_existing()
    
    if success1 and success2:
        print("\n" + "=" * 70)
        print("✅ All integration tests passed!")
        print("✅ The image.create_isolated_object tool is ready to use!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed!")
        print("=" * 70)
        sys.exit(1)
