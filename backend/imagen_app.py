"""
© 2025 Simplifine Corp. Image Generation Service for Godot AI Fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os
import base64
import uuid
import time
from PIL import Image
import io
from dotenv import load_dotenv
import hashlib
import asyncio
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["*"])

# OpenAI client for GPT Image
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY not set - image generation will fail!")
    client = None
else:
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized for GPT Image")

# Development mode detection
DEV_MODE = os.getenv('DEV_MODE', 'false').lower() == 'true'

# Import sprite sheet generator
from spritesheet_generator import SpriteSheetGenerator
sprite_gen = SpriteSheetGenerator(client) if client else None

# Image storage for referencing (in-memory for now, could be Redis/DB later)
# Structure: {image_id: {name, base64_data, mime_type, width, height, created_at, prompt}}
IMAGE_REGISTRY = {}

def generate_image_id(prefix="img"):
    """Generate unique image ID"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def register_image(image_data: dict) -> str:
    """Register an image in the registry and return its ID"""
    image_id = generate_image_id()
    image_data['id'] = image_id
    image_data['created_at'] = time.time()
    IMAGE_REGISTRY[image_id] = image_data
    print(f"IMAGE_REGISTRY: Registered {image_id} ({image_data.get('width')}x{image_data.get('height')})")
    return image_id

def get_image_by_id(image_id: str) -> dict:
    """Retrieve image from registry"""
    return IMAGE_REGISTRY.get(image_id)

def cleanup_old_images(max_age_seconds=3600):
    """Clean up images older than specified age (default 1 hour)"""
    current_time = time.time()
    to_remove = []
    for img_id, img_data in IMAGE_REGISTRY.items():
        if current_time - img_data.get('created_at', 0) > max_age_seconds:
            to_remove.append(img_id)
    
    for img_id in to_remove:
        del IMAGE_REGISTRY[img_id]
    
    if to_remove:
        print(f"CLEANUP: Removed {len(to_remove)} old images from registry")

@app.before_request
def log_request():
    """Log incoming requests"""
    if DEV_MODE:
        print(f"REQUEST: {request.method} {request.path} from {request.environ.get('REMOTE_ADDR')}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "godot-imagen-service",
        "openai_configured": client is not None,
        "registered_images": len(IMAGE_REGISTRY)
    })

@app.route('/api/image/generate', methods=['POST'])
def generate_image():
    """
    Legacy generate endpoint - forwards to unified process endpoint.
    Kept for backward compatibility.
    """
    data = request.get_json() or {}
    
    # Convert to unified format (no images = generation mode)
    unified_data = {
        "prompt": data.get('prompt', ''),
        "style": data.get('style', ''),
        "size": data.get('size', '1024x1024'),
        "quality": data.get('quality', 'low'),
        "output_format": data.get('output_format', 'png'),
        "output_compression": data.get('output_compression', 85),
        "background": data.get('background', 'opaque')
    }
    
    # Forward to unified endpoint
    request._cached_json = (unified_data, unified_data)
    return process_image()

@app.route('/api/image/process', methods=['POST'])
def process_image():
    """
    UNIFIED DYNAMIC IMAGE PROCESSING ENDPOINT
    
    Intelligently handles both generation and editing based on input:
    - No images provided → Generate new image from text
    - Images provided → Edit/combine images with text prompt
    
    Request body:
    {
        "prompt": "description or edit instruction",
        "images": [
            {
                "id": "img_abc123" (optional - retrieve from registry),
                "data": "base64_png_data" (optional - provide directly),
                "format": "png" | "jpeg" | "webp" (auto-detected if not provided)
            }
        ],
        "mask": {
            "data": "base64_mask_data",
            "format": "png"
        } (optional),
        "size": "1024x1024" | "1024x1536" | "1536x1024" | "auto",
        "quality": "low" | "medium" | "high" | "auto",
        "output_format": "png" | "jpeg" | "webp",
        "style": "pixel art" | "photorealistic" | etc.
    }
    
    Response:
    {
        "success": true,
        "image_id": "img_xyz789",
        "image_data": "base64_encoded_image",
        "width": 1024,
        "height": 1024,
        "format": "png",
        "operation": "generate" | "edit",
        "input_images": 0 | 1 | 2+
    }
    """
    if not client:
        return jsonify({
            "success": False,
            "error": "Image generation not configured - OPENAI_API_KEY missing"
        }), 503
    
    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '')
        style = data.get('style', '')
        row_descriptions = data.get('row_descriptions', [])
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"}), 400
        
        # Build full prompt
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style"
        
        # Collect input images
        images_input = data.get('images', [])
        image_files = []
        
        for img_spec in images_input:
            if isinstance(img_spec, dict):
                # Check if we have an ID to retrieve from registry
                img_id = img_spec.get('id')
                img_data_b64 = img_spec.get('data')
                img_format = img_spec.get('format', 'png')
                
                if img_id:
                    # Retrieve from registry
                    registered_img = get_image_by_id(img_id)
                    if registered_img:
                        img_data_b64 = registered_img['base64_data']
                        img_format = registered_img.get('format', 'png')
                        print(f"IMAGE_PROCESS: Retrieved {img_id} from registry")
                    else:
                        print(f"IMAGE_PROCESS: Warning - {img_id} not found in registry")
                        continue
                
                if img_data_b64:
                    try:
                        image_bytes = base64.b64decode(img_data_b64)
                        
                        # Create properly formatted file-like object with correct mime type
                        # OpenAI requires the file to have a name attribute with extension
                        class NamedBytesIO(io.BytesIO):
                            def __init__(self, data, name):
                                super().__init__(data)
                                self.name = name
                        
                        # Use proper extension for mime type detection
                        ext = img_format if img_format in ['png', 'jpg', 'jpeg', 'webp'] else 'png'
                        named_file = NamedBytesIO(image_bytes, f"image.{ext}")
                        
                        image_files.append(named_file)
                        print(f"IMAGE_PROCESS: Added image file with extension .{ext} ({len(image_bytes)} bytes)")
                    except Exception as e:
                        print(f"IMAGE_PROCESS: Failed to decode image: {e}")
                        continue
        
        # Get optional mask
        mask_spec = data.get('mask')
        mask_file = None
        if mask_spec and isinstance(mask_spec, dict):
            mask_data = mask_spec.get('data')
            if mask_data:
                try:
                    mask_bytes = base64.b64decode(mask_data)
                    class NamedBytesIO(io.BytesIO):
                        def __init__(self, data, name):
                            super().__init__(data)
                            self.name = name
                    mask_file = NamedBytesIO(mask_bytes, "mask.png")
                    print(f"IMAGE_PROCESS: Using mask for selective editing")
                except Exception as e:
                    print(f"MASK_ERROR: {e}")
        
        # Get generation parameters
        size = data.get('size', '1024x1024')
        quality = data.get('quality', 'low')
        output_format = data.get('output_format', 'png')
        output_compression = data.get('output_compression', 85)
        background = data.get('background', 'opaque')
        
        # Determine operation mode
        operation = "edit" if image_files else "generate"
        
        print(f"IMAGE_PROCESS: Operation: {operation}, Images: {len(image_files)}, Prompt: '{full_prompt[:100]}...'")
        
        start_time = time.time()
        
        # Build API parameters
        api_params = {
            "model": "gpt-image-1",
            "prompt": full_prompt,
            "size": size,
            "quality": quality,
            "output_format": output_format
        }
        
        # Add format-specific parameters
        if output_format in ['jpeg', 'webp']:
            api_params["output_compression"] = output_compression
        
        if output_format in ['png', 'webp'] and background == 'transparent':
            api_params["background"] = "transparent"
        
        # Call appropriate API endpoint
        if operation == "edit" and image_files:
            api_params["image"] = image_files
            if mask_file:
                api_params["mask"] = mask_file
            result = client.images.edit(**api_params)
        else:
            # Remove image-specific params for generation
            api_params.pop('image', None)
            api_params.pop('mask', None)
            result = client.images.generate(**api_params)
        
        generation_time = time.time() - start_time
        
        if not result.data or not hasattr(result.data[0], 'b64_json'):
            return jsonify({
                "success": False,
                "error": f"Image {operation} returned no data"
            }), 500
        
        image_base64 = result.data[0].b64_json
        
        # Get actual image dimensions
        try:
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
        except Exception as e:
            print(f"WARNING: Could not determine image size: {e}")
            width, height = None, None
        
        # Register the image
        image_data = {
            'base64_data': image_base64,
            'mime_type': f'image/{output_format}',
            'width': width,
            'height': height,
            'prompt': prompt,
            'style': style,
            'format': output_format,
            'operation': operation,
            'input_image_count': len(image_files)
        }
        
        image_id = register_image(image_data)
        
        print(f"IMAGE_PROCESS: {operation.upper()} complete - {image_id} ({width}x{height}) in {generation_time:.2f}s")
        
        return jsonify({
            "success": True,
            "image_id": image_id,
            "image_data": image_base64,
            "width": width,
            "height": height,
            "format": output_format,
            "prompt": prompt,
            "style": style,
            "generation_time": generation_time,
            "operation": operation,
            "input_images": len(image_files),
            "quality": quality,
            "size_requested": size
        })
        
    except Exception as e:
        print(f"IMAGE_PROCESS_ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Image processing failed: {str(e)}"
        }), 500

# Keep legacy endpoints for backward compatibility
@app.route('/api/image/edit', methods=['POST'])
def edit_image():
    """Legacy edit endpoint - redirects to unified process endpoint"""
    data = request.get_json() or {}
    
    # Convert legacy format to new unified format
    images = []
    
    # Handle image_ids
    for img_id in data.get('image_ids', []):
        images.append({"id": img_id})
    
    # Handle images_base64
    for img_b64 in data.get('images_base64', []):
        images.append({"data": img_b64, "format": "png"})
    
    # Convert to unified format
        unified_data = {
        "prompt": data.get('prompt', ''),
        "images": images,
        "size": data.get('size', '1024x1024'),
            "quality": data.get('quality', 'low'),
        "output_format": data.get('output_format', 'png'),
        "output_compression": data.get('output_compression', 85)
    }
    
    if data.get('mask_base64'):
        unified_data["mask"] = {"data": data.get('mask_base64'), "format": "png"}
    
    # Forward to unified endpoint
    request._cached_json = (unified_data, unified_data)
    return process_image()

@app.route('/api/image/get/<image_id>', methods=['GET'])
def get_image(image_id):
    """Retrieve an image by ID from the registry"""
    image_data = get_image_by_id(image_id)
    
    if not image_data:
        return jsonify({
            "success": False,
            "error": f"Image {image_id} not found"
        }), 404
    
    return jsonify({
        "success": True,
        "image_id": image_id,
        "image_data": image_data.get('base64_data'),
        "width": image_data.get('width'),
        "height": image_data.get('height'),
        "format": image_data.get('format'),
        "prompt": image_data.get('prompt'),
        "created_at": image_data.get('created_at')
    })

@app.route('/api/image/registry/stats', methods=['GET'])
def registry_stats():
    """Get statistics about the image registry"""
    total_size = sum(
        len(img.get('base64_data', '')) for img in IMAGE_REGISTRY.values()
    )
    
    return jsonify({
        "success": True,
        "total_images": len(IMAGE_REGISTRY),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "image_ids": list(IMAGE_REGISTRY.keys())
    })

@app.route('/api/image/cleanup', methods=['POST'])
def cleanup_images():
    """Manually trigger cleanup of old images"""
    data = request.get_json() or {}
    max_age = data.get('max_age_seconds', 3600)
    
    before_count = len(IMAGE_REGISTRY)
    cleanup_old_images(max_age)
    after_count = len(IMAGE_REGISTRY)
    removed = before_count - after_count
    
    return jsonify({
        "success": True,
        "removed": removed,
        "remaining": after_count
    })

@app.route('/api/spritesheet/generate_progressive', methods=['POST'])
def generate_spritesheet_progressive():
    """
    Generate sprite sheet progressively with streaming updates.
    Generates first column in parallel, then each row in parallel with sequential columns.
    
    Request body:
    {
        "prompt": "sprite sheet description",
        "seed_image_id": "img_abc123",
        "grid_width": 4,
        "grid_height": 3,
        "style": "pixel art"
    }
    
    Response: NDJSON stream with progress updates
    """
    if not client or not sprite_gen:
        return jsonify({
            "success": False,
            "error": "Image generation not configured"
        }), 503
    
    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '')
        seed_image_id = data.get('seed_image_id', '')
        grid_width = int(data.get('grid_width', 4))
        grid_height = int(data.get('grid_height', 3))
        style = data.get('style', '')
        row_descriptions = data.get('row_descriptions', [])
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"}), 400
        
        if not seed_image_id:
            return jsonify({"success": False, "error": "Seed image ID is required"}), 400
        
        # Get seed image from registry
        seed_data = get_image_by_id(seed_image_id)
        if not seed_data:
            return jsonify({
                "success": False,
                "error": f"Seed image {seed_image_id} not found in registry"
            }), 404
        
        seed_image_b64 = seed_data['base64_data']
        
        print(f"PROGRESSIVE_SPRITESHEET: Starting {grid_width}x{grid_height} generation")
        print(f"PROGRESSIVE_SPRITESHEET: Seed: {seed_image_id}, Style: {style}")
        
        def generate_stream():
            """Stream progress updates as cells are generated"""
            import queue
            import threading
            
            # Send initial status
            yield json.dumps({
                "status": "started",
                "grid_width": grid_width,
                "grid_height": grid_height,
                "total_cells": grid_width * grid_height
            }) + '\n'
            
            # Thread-safe queue for progress updates
            progress_queue = queue.Queue()
            generation_complete = threading.Event()
            
            async def progress_callback(update):
                progress_data = {
                    "status": "progress",
                    "phase": update['phase'],
                    "row": update['row'],
                    "col": update['col'],
                    "completed": update['completed'],
                    "total": update['total'],
                    "progress_percent": round((update['completed'] / update['total']) * 100, 1)
                }
                
                # Include cell data
                cell_data = update.get('cell_data', {})
                if cell_data:
                    progress_data["cell"] = {
                        "row": cell_data.get('row'),
                        "col": cell_data.get('col'),
                        "cell_number": cell_data.get('cell_number'),
                        "width": cell_data.get('width'),
                        "height": cell_data.get('height'),
                        "generation_time": cell_data.get('generation_time'),
                        "image_data": cell_data.get('image_data')
                    }
                
                # Put update in queue immediately
                progress_queue.put(progress_data)
            
            # Run the async generation in a background thread
            def run_generation():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        sprite_gen.generate_sprite_sheet_progressive(
                            prompt=prompt,
                            seed_image_b64=seed_image_b64,
                            grid_width=grid_width,
                            grid_height=grid_height,
                            style=style,
                            size="1024x1024",
                            row_descriptions=row_descriptions,
                            progress_callback=progress_callback
                        )
                    )
                    progress_queue.put({"_complete": True, "result": result})
                except Exception as e:
                    progress_queue.put({"_error": True, "error": str(e)})
                finally:
                    loop.close()
                    generation_complete.set()
            
            # Start generation in background thread
            gen_thread = threading.Thread(target=run_generation, daemon=True)
            gen_thread.start()
            
            try:
                # Stream progress updates as they come in
                while not generation_complete.is_set() or not progress_queue.empty():
                    try:
                        # Get update with short timeout to check if generation is done
                        update = progress_queue.get(timeout=0.1)
                        
                        # Check for completion/error markers
                        if update.get("_complete"):
                            result = update.get("result", {})
                            # Send completion WITHOUT cell data (to avoid >16MB chunk)
                            if result.get('success'):
                                yield json.dumps({
                                    "status": "completed",
                                    "success": True,
                                    "grid_width": grid_width,
                                    "grid_height": grid_height,
                                    "completed_cells": result['completed_cells'],
                                    "total_cells": result['total_cells'],
                                    "total_time": result['total_time']
                                }) + '\n'
                            else:
                                yield json.dumps({
                                    "status": "error",
                                    "error": "Sprite sheet generation failed"
                                }) + '\n'
                            break
                        elif update.get("_error"):
                            yield json.dumps({
                                "status": "error",
                                "error": update.get("error", "Unknown error")
                            }) + '\n'
                            break
                        else:
                            # Regular progress update - yield immediately!
                            yield json.dumps(update) + '\n'
                            print(f"STREAM_SENT: Progress update for cell [{update.get('row')},{update.get('col')}]")
                    except queue.Empty:
                        # No update available yet, continue polling
                        continue
                        
            except Exception as e:
                print(f"PROGRESSIVE_ERROR: {e}")
                import traceback
                traceback.print_exc()
                yield json.dumps({
                    "status": "error",
                    "error": str(e)
                }) + '\n'
        
        return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')
        
    except Exception as e:
        print(f"PROGRESSIVE_SPRITESHEET_ERROR: {e}")
        return jsonify({
            "success": False,
            "error": f"Progressive sprite sheet generation failed: {str(e)}"
        }), 500

@app.route('/api/spritesheet/generate', methods=['POST'])
def generate_spritesheet():
    """
    Generate a sprite sheet with specific grid layout.
    Uses the unified image processing endpoint internally.
    
    Request body:
    {
        "prompt": "description of sprite sheet",
        "grid_width": 4,
        "grid_height": 3,
        "tile_size": 128,
        "images": [{"id": "img_123"} or {"data": "base64"}] (optional),
        "style": "pixel art" (optional)
    }
    """
    if not client:
        return jsonify({
            "success": False,
            "error": "Image generation not configured"
        }), 503
    
    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '')
        grid_width = int(data.get('grid_width', 4))
        grid_height = int(data.get('grid_height', 3))
        tile_size = int(data.get('tile_size', 128))
        style = data.get('style', '')
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"}), 400
        
        # Calculate total image size
        total_width = grid_width * tile_size
        total_height = grid_height * tile_size
        
        # Choose closest supported size
        if total_width == total_height:
            size = "1024x1024"
        elif total_width > total_height:
            size = "1536x1024"
        else:
            size = "1024x1536"
        
        # Build sprite sheet prompt
        full_prompt = f"{prompt}\n\nSprite sheet layout: {grid_width}x{grid_height} grid. Each cell should contain one frame of the animation or variation. Arrange frames left-to-right, top-to-bottom."
        
        # Forward to unified process endpoint
        unified_data = {
            "prompt": full_prompt,
            "style": style,
            "images": data.get('images', []),  # Pass through any seed images
            "size": size,
            "quality": "high",
            "output_format": "png"
        }
        
        # Call unified endpoint internally
        request._cached_json = (unified_data, unified_data)
        response = process_image()
        
        # If successful, add sprite sheet metadata to response
        if response.status_code == 200:
            response_data = response.get_json()
            if response_data.get('success'):
                response_data['sprite_sheet'] = {
                    "grid_width": grid_width,
                    "grid_height": grid_height,
                    "tile_size": tile_size,
                    "total_frames": grid_width * grid_height
                }
                return jsonify(response_data)
        
        return response
        
    except Exception as e:
        print(f"SPRITE_SHEET_ERROR: {e}")
        return jsonify({
            "success": False,
            "error": f"Sprite sheet generation failed: {str(e)}"
        }), 500

@app.route('/api/image/mask/generate', methods=['POST'])
def generate_mask():
    """
    Generate a mask for an image using AI.
    Useful for selective editing.
    
    Request body:
    {
        "image_id": "img_abc123" (or)
        "image_base64": "base64_data",
        "mask_prompt": "describe what to mask"
    }
    """
    if not client:
        return jsonify({
            "success": False,
            "error": "Image generation not configured"
        }), 503
    
    try:
        data = request.get_json() or {}
        mask_prompt = data.get('mask_prompt', 'generate a mask delimiting the entire character in the picture, using white where the character is and black for the background')
        
        # Get source image
        image_id = data.get('image_id')
        image_base64 = data.get('image_base64')
        
        if image_id:
            img_data = get_image_by_id(image_id)
            if img_data:
                image_base64 = img_data['base64_data']
            else:
                return jsonify({
                    "success": False,
                    "error": f"Image {image_id} not found"
                }), 404
        
        if not image_base64:
            return jsonify({
                "success": False,
                "error": "Image is required (provide image_id or image_base64)"
            }), 400
        
        # Convert to file-like object
        image_bytes = base64.b64decode(image_base64)
        image_file = io.BytesIO(image_bytes)
        
        # Generate mask
        result = client.images.edit(
            model="gpt-image-1",
            image=image_file,
            prompt=mask_prompt,
            output_format="png"
        )
        
        if not result.data or not hasattr(result.data[0], 'b64_json'):
            return jsonify({
                "success": False,
                "error": "Mask generation returned no data"
            }), 500
        
        mask_base64 = result.data[0].b64_json
        
        # Convert to alpha channel mask
        mask_bytes = base64.b64decode(mask_base64)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_rgba = mask_img.convert("RGBA")
        mask_rgba.putalpha(mask_img)
        
        # Convert back to base64
        buf = io.BytesIO()
        mask_rgba.save(buf, format="PNG")
        final_mask_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        mask_id = register_image({
            'base64_data': final_mask_base64,
            'mime_type': 'image/png',
            'width': mask_rgba.size[0],
            'height': mask_rgba.size[1],
            'prompt': mask_prompt,
            'format': 'png',
            'is_mask': True,
            'source_image_id': image_id
        })
        
        print(f"MASK_GEN: Generated mask {mask_id}")
        
        return jsonify({
            "success": True,
            "mask_id": mask_id,
            "mask_data": final_mask_base64,
            "width": mask_rgba.size[0],
            "height": mask_rgba.size[1],
            "format": "png",
            "source_image_id": image_id
        })
        
    except Exception as e:
        print(f"MASK_GEN_ERROR: {e}")
        return jsonify({
            "success": False,
            "error": f"Mask generation failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("🎨 Godot Image Generation Service")
    print(f"   DEV_MODE: {DEV_MODE}")
    print(f"   OpenAI API: {'Configured ✅' if client else 'Not configured ❌'}")
    
    # Cleanup task runs periodically
    import threading
    def periodic_cleanup():
        while True:
            time.sleep(600)  # Every 10 minutes
            cleanup_old_images(3600)  # Remove images older than 1 hour
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    print("   Cleanup task: Started (1 hour TTL)")
    
    port = int(os.environ.get('PORT', 3031))
    app.run(host='0.0.0.0', port=port, debug=DEV_MODE)

