import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('test_nano_banana_direct.py')))
from dotenv import load_dotenv
load_dotenv()
from nano_banana import generate_standalone_object
import base64

print('Testing standalone object generation: Side-scrolling 2D map (Mario style)')
print('=' * 70)

try:
    image_base64, width, height = generate_standalone_object(
        prompt='a healthpoint heart, but a realistic looking heart',
        size='1024x1024'
    )
    
    print(f'\n✅ Success!')
    print(f'   Dimensions: {width}x{height}')
    print(f'   Base64 length: {len(image_base64)} chars')
    
    # Save the image
    image_bytes = base64.b64decode(image_base64)
    output_path = 'test_2d_map_side.png'
    with open(output_path, 'wb') as f:
        f.write(image_bytes)
    print(f'\n💾 Saved to: {output_path}')
    
except Exception as e:
    print(f'\n❌ Failed: {e}')
    import traceback
    traceback.print_exc()