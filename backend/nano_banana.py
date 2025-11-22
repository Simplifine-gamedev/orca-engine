"""
Nano Banana (Google Gemini Image Generation) Integration
© 2025 Simplifine Corp. Original backend contribution for this Godot fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""
import os
import base64
import io
from typing import Optional, Tuple
from PIL import Image
from google import genai
from google.genai import types


def get_gemini_client():
    """Initialize and return Gemini client"""
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def _try_open_image_from_bytes(image_bytes: bytes) -> tuple[Image.Image, bytes]:
    """
    Try to open image from bytes. If the bytes are actually base64-ASCII, decode first.
    Returns (PIL.Image, decoded_binary_bytes).
    """
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # First attempt: treat as binary image
    try:
        stream = io.BytesIO(image_bytes)
        img = Image.open(stream)
        img.load()  # Force load to catch truncated issues early
        return img, image_bytes
    except Exception:
        pass
    # Second attempt: treat as base64-ASCII and decode
    try:
        # Try direct base64 decode on bytes
        decoded = base64.b64decode(image_bytes, validate=False)
        stream2 = io.BytesIO(decoded)
        img2 = Image.open(stream2)
        img2.load()
        return img2, decoded
    except Exception:
        pass
    # Third attempt: decode to ASCII string then base64-decode
    try:
        as_str = image_bytes.decode('ascii', errors='ignore').strip()
        decoded2 = base64.b64decode(as_str, validate=False)
        stream3 = io.BytesIO(decoded2)
        img3 = Image.open(stream3)
        img3.load()
        return img3, decoded2
    except Exception as final_err:
        raise ValueError(f"Unable to open image from provided bytes: {final_err}")


def _aspect_ratio_from_size(size_str: str) -> Optional[str]:
    """Convert size string (e.g., '1024x1024') to Gemini aspect ratio format"""
    try:
        if not size_str:
            return None
        parts = str(size_str).lower().replace(' ', '').split('x')
        if len(parts) != 2:
            return None
        w, h = int(float(parts[0])), int(float(parts[1]))
        
        # Calculate aspect ratio
        ratio = w / h if h > 0 else 1.0
        
        # Map to Gemini supported aspect ratios
        # Supported: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
        if abs(ratio - 1.0) < 0.1:
            return "1:1"
        elif abs(ratio - 2/3) < 0.1:
            return "2:3"
        elif abs(ratio - 3/2) < 0.1:
            return "3:2"
        elif abs(ratio - 3/4) < 0.1:
            return "3:4"
        elif abs(ratio - 4/3) < 0.1:
            return "4:3"
        elif abs(ratio - 4/5) < 0.1:
            return "4:5"
        elif abs(ratio - 5/4) < 0.1:
            return "5:4"
        elif abs(ratio - 9/16) < 0.1:
            return "9:16"
        elif abs(ratio - 16/9) < 0.1:
            return "16:9"
        elif abs(ratio - 21/9) < 0.1:
            return "21:9"
        else:
            # Default to 1:1 for unsupported ratios
            return "1:1"
    except Exception:
        return None


def _remove_white_background(image: Image.Image, threshold: int = 240) -> Image.Image:
    """
    Remove white background from an image by making white pixels transparent.
    
    Args:
        image: PIL Image to process
        threshold: RGB threshold (0-255) - pixels with R, G, B all above this value are considered white
    
    Returns:
        PIL Image with RGBA mode and transparent white background
    """
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Get image data as array
    data = image.getdata()
    
    # Create new image data with transparency
    new_data = []
    for item in data:
        r, g, b, a = item
        # If pixel is close to white (all channels above threshold), make it transparent
        if r >= threshold and g >= threshold and b >= threshold:
            new_data.append((255, 255, 255, 0))  # Transparent white
        else:
            new_data.append(item)  # Keep original pixel
    
    # Create new image with modified data
    new_image = Image.new('RGBA', image.size)
    new_image.putdata(new_data)
    
    return new_image


def generate_image_from_text(
    prompt: str,
    size: str = "1024x1024"
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Generate an image from text using Gemini 2.5 Flash Image model.
    
    Args:
        prompt: Text description for image generation
        size: Target size string (e.g., "1024x1024") - used to determine aspect ratio
    
    Returns:
        Tuple of (base64_image_data, width, height)
    """
    try:
        client = get_gemini_client()
        
        # Build config - aspect ratio is handled automatically by the model
        # based on the prompt or can be specified via mediaResolution if needed
        config = types.GenerateContentConfig(
            response_modalities=['Image']
        )
        
        # Generate image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            config=config
        )
        
        # Extract image from response
        # Response structure: response.candidates[0].content.parts[0].inline_data
        image_base64 = None
        width = None
        height = None
        
        if not response.candidates or len(response.candidates) == 0:
            raise ValueError("No candidates returned from Gemini")
        
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            raise ValueError("No content parts in response")
        
        for part in candidate.content.parts:
            if part.inline_data is not None:
                # part.inline_data is a Blob with data (bytes) and mime_type
                blob = part.inline_data
                if blob.data is None:
                    print(f"NANO_BANANA_DEBUG: blob.data is None, mime_type={blob.mime_type}")
                    continue
                
                image_bytes = blob.data
                if not isinstance(image_bytes, bytes):
                    raise ValueError(f"Expected bytes, got {type(image_bytes)}")
                
                if len(image_bytes) == 0:
                    print(f"NANO_BANANA_DEBUG: blob.data is empty, mime_type={blob.mime_type}")
                    continue
                
                print(f"NANO_BANANA_DEBUG: Got image data: {len(image_bytes)} bytes, mime_type={blob.mime_type}")
                # Convert bytes to PIL Image (handle both binary and base64-ASCII)
                pil_image, decoded_binary = _try_open_image_from_bytes(image_bytes)
                width, height = pil_image.size
                print(f"NANO_BANANA_DEBUG: Image opened successfully: {width}x{height}")
                
                # Convert PIL Image to base64 PNG
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                buffer.seek(0)  # Reset buffer position
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                break
        
        if not image_base64:
            raise ValueError("No image data returned from Gemini")
        
        return image_base64, width, height
        
    except Exception as e:
        raise Exception(f"Gemini image generation failed: {str(e)}")


def generate_image_from_image_and_text(
    image_base64: str,
    prompt: str,
    size: str = "1024x1024"
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Generate/Edit an image from an existing image and text prompt using Gemini 2.5 Flash Image model.
    
    Args:
        image_base64: Base64 encoded input image
        prompt: Text description for image editing/generation
        size: Target size string (e.g., "1024x1024") - used to determine aspect ratio
    
    Returns:
        Tuple of (base64_image_data, width, height)
    """
    try:
        client = get_gemini_client()
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Build config - aspect ratio is handled automatically by the model
        config = types.GenerateContentConfig(
            response_modalities=['Image']
        )
        
        # Generate image with image + text input
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, pil_image],
            config=config
        )
        
        # Extract image from response
        # Response structure: response.candidates[0].content.parts[0].inline_data
        image_base64_result = None
        width = None
        height = None
        
        if not response.candidates or len(response.candidates) == 0:
            raise ValueError("No candidates returned from Gemini")
        
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            raise ValueError("No content parts in response")
        
        for part in candidate.content.parts:
            if part.inline_data is not None:
                # part.inline_data is a Blob with data (bytes) and mime_type
                blob = part.inline_data
                if blob.data is None:
                    print(f"NANO_BANANA_DEBUG: blob.data is None, mime_type={blob.mime_type}")
                    continue
                
                image_bytes = blob.data
                if not isinstance(image_bytes, bytes):
                    raise ValueError(f"Expected bytes, got {type(image_bytes)}")
                
                if len(image_bytes) == 0:
                    print(f"NANO_BANANA_DEBUG: blob.data is empty, mime_type={blob.mime_type}")
                    continue
                
                print(f"NANO_BANANA_DEBUG: Got image data: {len(image_bytes)} bytes, mime_type={blob.mime_type}")
                # Convert bytes to PIL Image (handle both binary and base64-ASCII)
                pil_image, decoded_binary = _try_open_image_from_bytes(image_bytes)
                width, height = pil_image.size
                print(f"NANO_BANANA_DEBUG: Image opened successfully: {width}x{height}")
                
                # Convert PIL Image to base64 PNG
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                buffer.seek(0)  # Reset buffer position
                image_base64_result = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                break
        
        if not image_base64_result:
            raise ValueError("No image data returned from Gemini")
        
        return image_base64_result, width, height
        
    except Exception as e:
        raise Exception(f"Gemini image editing failed: {str(e)}")


def generate_standalone_object(
    prompt: str,
    size: str = "1024x1024",
    white_threshold: int = 240
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Generate a standalone object image with white background removed.
    
    This function is designed to create isolated objects (icons, symbols, characters, etc.)
    on a white background, then automatically remove the background to extract just the object.
    
    Args:
        prompt: Text description for the object to generate
        size: Target size string (e.g., "1024x1024") - used to determine aspect ratio
        white_threshold: RGB threshold (0-255) for white detection. Pixels with R, G, B all
                        above this value will be made transparent. Default is 240.
    
    Returns:
        Tuple of (base64_image_data, width, height) with transparent background
    """
    try:
        # Add the special instruction prompt
        standalone_instruction = """### Instructions for this generation

this image generation is intended to create a stand alone object, this could be for a variety of things, from icons and symbols to characters. this requires making what the user is asking for on a clear white background with no shadows, nothing, NOTHING but the isolated object."""
        
        # Combine user prompt with instruction
        enhanced_prompt = f"{prompt}\n\n{standalone_instruction}"
        
        print(f"NANO_BANANA_DEBUG: Generating standalone object with enhanced prompt")
        
        # Generate image using the existing function
        image_base64, width, height = generate_image_from_text(
            prompt=enhanced_prompt,
            size=size
        )
        
        print(f"NANO_BANANA_DEBUG: Image generated, removing white background...")
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Remove white background
        pil_image_transparent = _remove_white_background(pil_image, threshold=white_threshold)
        
        print(f"NANO_BANANA_DEBUG: White background removed, converting to base64...")
        
        # Convert back to base64 PNG (PNG supports transparency)
        buffer = io.BytesIO()
        pil_image_transparent.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64_transparent = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"NANO_BANANA_DEBUG: Standalone object generation complete: {width}x{height}")
        
        return image_base64_transparent, width, height
        
    except Exception as e:
        raise Exception(f"Standalone object generation failed: {str(e)}")


def transparent_bg_image_gen(
    image_input: str,
    prompt: str,
    size: str = "1024x1024",
    white_threshold: int = 240
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Generate/edit an image with transparent background from an input image and prompt.
    
    This function takes an existing image and a prompt, enhances it with instructions
    to create a standalone object on white background, then removes the background.
    
    Args:
        image_input: Either a base64 encoded image string OR a file path to an image
        prompt: Text description for image editing/generation (user's normal prompt)
        size: Target size string (e.g., "1024x1024") - used to determine aspect ratio
        white_threshold: RGB threshold (0-255) for white detection. Pixels with R, G, B all
                        above this value will be made transparent. Default is 240.
    
    Returns:
        Tuple of (base64_image_data, width, height) with transparent background
    """
    try:
        # Handle input: could be base64 string or file path
        if os.path.exists(image_input):
            # It's a file path - read and encode to base64
            print(f"NANO_BANANA_DEBUG: Reading image from file: {image_input}")
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        else:
            # Assume it's already base64 encoded
            image_base64 = image_input
        
        # Add the special instruction prompt
        standalone_instruction = """### Instructions for this generation

this image generation is intended to create a stand alone object, this could be for a variety of things, from icons and symbols to characters. this requires making what the user is asking for on a clear white background with no shadows, nothing, NOTHING but the isolated object."""
        
        # Combine user prompt with instruction
        enhanced_prompt = f"{prompt}\n\n{standalone_instruction}"
        
        print(f"NANO_BANANA_DEBUG: Generating transparent bg image with enhanced prompt")
        
        # Generate image using the existing image+text function
        image_base64_result, width, height = generate_image_from_image_and_text(
            image_base64=image_base64,
            prompt=enhanced_prompt,
            size=size
        )
        
        print(f"NANO_BANANA_DEBUG: Image generated, removing white background...")
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64_result)
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Remove white background
        pil_image_transparent = _remove_white_background(pil_image, threshold=white_threshold)
        
        print(f"NANO_BANANA_DEBUG: White background removed, converting to base64...")
        
        # Convert back to base64 PNG (PNG supports transparency)
        buffer = io.BytesIO()
        pil_image_transparent.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64_transparent = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"NANO_BANANA_DEBUG: Transparent bg image generation complete: {width}x{height}")
        
        return image_base64_transparent, width, height
        
    except Exception as e:
        raise Exception(f"Transparent background image generation failed: {str(e)}")

