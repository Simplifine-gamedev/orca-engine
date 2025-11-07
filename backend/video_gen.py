"""
Video Generation Module for Google Veo 3.1
© 2025 Simplifine Corp. Original backend contribution for this Godot fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Retrieve the Google API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize the Google GenAI client (will be None if API key is missing)
_client = None

def get_client():
    """Get or initialize the Google GenAI client"""
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client

def generate_video(
    prompt: str,
    output_filename: Optional[str] = None,
    model: str = "veo-3.1-generate-preview",
    poll_interval: int = 10,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates a video based on the provided prompt using Google's Veo 3.1.

    Args:
        prompt (str): The prompt describing the desired video content.
        output_filename (str, optional): The filename to save the generated video.
            If not provided, a default name will be generated.
        model (str): The model to use for generation. Defaults to "veo-3.1-generate-preview".
        poll_interval (int): Seconds to wait between polling for completion. Defaults to 10.
        timeout (int, optional): Maximum time in seconds to wait for video generation.
            If None, will wait indefinitely.

    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the generation was successful
            - output_path (str, optional): Path to the generated video file
            - error (str, optional): Error message if generation failed
            - operation_id (str, optional): The operation ID for tracking
    """
    try:
        client = get_client()
        
        # Generate default filename if not provided
        if not output_filename:
            import uuid
            output_filename = f"generated_video_{uuid.uuid4().hex[:8]}.mp4"
        
        # Initiate the video generation process
        print(f"Starting video generation with model: {model}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
        )
        
        operation_id = operation.name if hasattr(operation, 'name') else None
        print(f"Video generation started. Operation ID: {operation_id}")
        
        # Poll the operation status until the video is ready
        start_time = time.time()
        while not operation.done:
            elapsed = time.time() - start_time
            if timeout and elapsed > timeout:
                return {
                    "success": False,
                    "error": f"Video generation timed out after {timeout} seconds",
                    "operation_id": operation_id
                }
            
            print(f"Waiting for video generation to complete... (elapsed: {int(elapsed)}s)")
            time.sleep(poll_interval)
            operation = client.operations.get(operation)
        
        # Check if operation completed successfully
        if hasattr(operation, 'response') and operation.response:
            # Download the generated video
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(output_filename)
            
            print(f"Generated video saved to {output_filename}")
            
            return {
                "success": True,
                "output_path": output_filename,
                "operation_id": operation_id
            }
        else:
            return {
                "success": False,
                "error": "Video generation completed but no video was returned",
                "operation_id": operation_id
            }
            
    except ValueError as e:
        # API key missing or configuration error
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        error_msg = f"An error occurred during video generation: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

def generate_video_from_image(
    image_path: str,
    prompt: str,
    output_filename: Optional[str] = None,
    model: str = "veo-3.1-generate-preview",
    poll_interval: int = 10,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates a video from a custom image using Google's Veo 3.1.

    Args:
        image_path (str): Path to the input image file.
        prompt (str): The prompt describing the desired video animation.
        output_filename (str, optional): The filename to save the generated video.
            If not provided, a default name will be generated.
        model (str): The model to use for generation. Defaults to "veo-3.1-generate-preview".
        poll_interval (int): Seconds to wait between polling for completion. Defaults to 10.
        timeout (int, optional): Maximum time in seconds to wait for video generation.
            If None, will wait indefinitely.

    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the generation was successful
            - output_path (str, optional): Path to the generated video file
            - error (str, optional): Error message if generation failed
            - operation_id (str, optional): The operation ID for tracking
    """
    try:
        client = get_client()
        
        # Validate image file exists
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }
        
        # Load and upload the image
        print(f"Loading image from: {image_path}")
        try:
            # Open image with PIL to ensure it's valid
            pil_image = Image.open(image_path)
            print(f"Image loaded: {pil_image.size[0]}x{pil_image.size[1]} pixels, mode: {pil_image.mode}")
            
            # Upload the image file to Google GenAI
            # The API expects a file to be uploaded first, then we use the uploaded file object
            uploaded_file = client.files.upload(path=image_path)
            print(f"Image uploaded successfully. File ID: {uploaded_file.name}")
            
            # Use the uploaded file as the image input
            # The API expects the file object itself, not the path
            image_input = uploaded_file
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load/upload image: {str(e)}"
            }
        
        # Generate default filename if not provided
        if not output_filename:
            import uuid
            base_name = Path(image_path).stem
            output_filename = f"video_from_{base_name}_{uuid.uuid4().hex[:8]}.mp4"
        
        # Initiate the video generation process with image
        print(f"Starting video generation with model: {model}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image_input,
        )
        
        operation_id = operation.name if hasattr(operation, 'name') else None
        print(f"Video generation started. Operation ID: {operation_id}")
        
        # Poll the operation status until the video is ready
        start_time = time.time()
        while not operation.done:
            elapsed = time.time() - start_time
            if timeout and elapsed > timeout:
                return {
                    "success": False,
                    "error": f"Video generation timed out after {timeout} seconds",
                    "operation_id": operation_id
                }
            
            print(f"Waiting for video generation to complete... (elapsed: {int(elapsed)}s)")
            time.sleep(poll_interval)
            operation = client.operations.get(operation)
        
        # Check if operation completed successfully
        if hasattr(operation, 'response') and operation.response:
            # Download the generated video
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(output_filename)
            
            print(f"Generated video saved to {output_filename}")
            
            return {
                "success": True,
                "output_path": output_filename,
                "operation_id": operation_id
            }
        else:
            return {
                "success": False,
                "error": "Video generation completed but no video was returned",
                "operation_id": operation_id
            }
            
    except ValueError as e:
        # API key missing or configuration error
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        error_msg = f"An error occurred during video generation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg
        }

def check_api_key() -> bool:
    """Check if Google API key is configured"""
    return bool(GOOGLE_API_KEY)

