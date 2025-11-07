#!/usr/bin/env python3
"""
Test script for the video generation functionality using Google Veo 3.1.
This script tests the backend's ability to generate videos using Google's Veo 3.1 model.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the video generation module
try:
    from video_gen import generate_video, generate_video_from_image, check_api_key
except ImportError as e:
    print(f"Error importing video_gen module: {e}")
    print("Make sure video_gen.py is in the same directory as this test script.")
    sys.exit(1)

def test_video_generation_from_image():
    """Test video generation from a custom image"""
    
    # Check if API key is available
    if not check_api_key():
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file in the backend directory with:")
        print("  GOOGLE_API_KEY=your_google_api_key")
        return False
    
    print("✅ Google API key found")
    print("\n" + "=" * 60)
    print("🎬 AI Video Generation from Image Test (Google Veo 3.1)")
    print("=" * 60)
    
    # Image path - pixelated Robin Hood character
    image_path = "/Users/alikavoosi/Desktop/3d-design/GODOT/godot/backend/pix_robinhood.png"
    
    # Test prompt for sprite walking animation
    test_prompt = """This will be a sprite fashion animation for a 2d game, JUST make a simple walking animation for the 2d character, showing it walking non stop in a smooth animation, NO BACKGROUND only the white background in the original image, just the isolated walking animation of the pixalated robinhood character"""
    
    # Generate a unique output filename
    test_output = f"test_video_{int(time.time())}.mp4"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image file not found: {image_path}")
        return False
    
    print(f"\n🖼️  Input image: {image_path}")
    print(f"📝 Prompt: {test_prompt}")
    print(f"💾 Output file: {test_output}")
    print("\n⚠️  Note: Video generation can take several minutes to complete.")
    print("Starting generation...\n")
    
    try:
        # Generate the video from image
        result = generate_video_from_image(
            image_path=image_path,
            prompt=test_prompt,
            output_filename=test_output,
            poll_interval=10,  # Check every 10 seconds
            timeout=900  # 15 minute timeout for testing (video gen can take longer)
        )
        
        # Check results
        if result.get("success"):
            output_path = result.get("output_path")
            operation_id = result.get("operation_id")
            
            print("\n" + "=" * 60)
            print("✅ Video Generation Successful!")
            print("=" * 60)
            print(f"📁 Output file: {output_path}")
            if operation_id:
                print(f"🆔 Operation ID: {operation_id}")
            
            # Check if file exists
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
                print(f"\n✅ Test passed: Video file created successfully at {output_path}")
                
                # Ask if user wants to keep the file
                print("\n💡 The test video file has been saved.")
                print("   You can delete it manually if you don't need it.")
                
                return True
            else:
                print(f"\n⚠️  Warning: Video generation reported success but file not found at {output_path}")
                return False
        else:
            error = result.get("error", "Unknown error")
            operation_id = result.get("operation_id")
            
            print("\n" + "=" * 60)
            print("❌ Video Generation Failed")
            print("=" * 60)
            print(f"Error: {error}")
            if operation_id:
                print(f"Operation ID: {operation_id}")
            
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_key_check():
    """Test the API key check function"""
    print("\n" + "=" * 60)
    print("🔑 Testing API Key Check")
    print("=" * 60)
    
    has_key = check_api_key()
    if has_key:
        print("✅ API key is configured")
        return True
    else:
        print("❌ API key is not configured")
        return False

if __name__ == "__main__":
    print("\n🧪 Running Video Generation Tests\n")
    
    # First test: API key check
    api_key_ok = test_api_key_check()
    
    if not api_key_ok:
        print("\n❌ Cannot proceed without API key. Please configure GOOGLE_API_KEY in your .env file.")
        sys.exit(1)
    
    # Second test: Video generation from image
    print("\n" + "=" * 60)
    video_gen_ok = test_video_generation_from_image()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"API Key Check: {'✅ PASS' if api_key_ok else '❌ FAIL'}")
    print(f"Video Generation: {'✅ PASS' if video_gen_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if api_key_ok and video_gen_ok:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

