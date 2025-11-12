#!/usr/bin/env python3
"""
Godot AI Database Server - 3D Model Retrieval Service

This server provides endpoints to fetch 3D models from Supabase storage.
It connects to the same Supabase instance used by the GPU shape generation server
and serves as a proxy/cache layer for Godot clients.

Endpoints:
- GET /models/{user_id} - List all models for a user
- GET /models/{user_id}/{model_id} - Get specific model details
- GET /download/{user_id}/{model_id}/obj - Download OBJ file
- GET /download/{user_id}/{model_id}/image - Download reference image
- GET /health - Health check
- GET /stats - Get server statistics
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import traceback
import requests
from urllib.parse import unquote
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import tempfile
import hashlib

# Supabase
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app, origins="*")

# Environment variables
DEV_MODE = os.getenv('DEV_MODE', 'true').lower() == 'true'
PORT = int(os.getenv('PORT', 8080))

# Supabase configuration - read from .env file
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Service key for backend access
SUPABASE_PROJECT_ID = os.getenv('SUPABASE_PROJECT_ID')
# Get table names and strip any whitespace/newlines
MODELS_TABLE = os.getenv('MODELS_TABLE', 'three_d_models')
TEXTURE_JOBS_TABLE = os.getenv('TEXTURE_JOBS_TABLE', 'texture_jobs')

# Clean up table names - remove all whitespace, newlines, quotes
if MODELS_TABLE:
    MODELS_TABLE = MODELS_TABLE.strip().strip('\'"').replace(' ', '').replace('\n', '').replace('\r', '')
if TEXTURE_JOBS_TABLE:
    TEXTURE_JOBS_TABLE = TEXTURE_JOBS_TABLE.strip().strip('\'"').replace(' ', '').replace('\n', '').replace('\r', '')

# Set defaults if empty after cleaning
MODELS_TABLE = MODELS_TABLE or 'three_d_models'
TEXTURE_JOBS_TABLE = TEXTURE_JOBS_TABLE or 'texture_jobs'

# Debug logging for table names
logger.info(f"📋 Using table names: MODELS_TABLE='{MODELS_TABLE}', TEXTURE_JOBS_TABLE='{TEXTURE_JOBS_TABLE}'")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
    if DEV_MODE:
        logger.warning("Running in DEV_MODE - some features may not work without Supabase")
    else:
        sys.exit(1)

# Initialize Supabase client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {e}")
        if not DEV_MODE:
            sys.exit(1)

# Cache for downloaded files (in-memory for now, could be Redis in production)
file_cache: Dict[str, bytes] = {}
CACHE_MAX_SIZE = 100  # Maximum number of files to cache
CACHE_TTL = 3600  # 1 hour TTL


class DatabaseServerError(Exception):
    """Custom exception for database server errors"""
    pass


def get_cache_key(user_id: str, model_id: str, file_type: str) -> str:
    """Generate cache key for file caching"""
    return f"{user_id}_{model_id}_{file_type}"


def cache_file(key: str, data: bytes) -> None:
    """Cache file data with LRU eviction"""
    if len(file_cache) >= CACHE_MAX_SIZE:
        # Simple LRU - remove oldest entry
        oldest_key = next(iter(file_cache))
        del file_cache[oldest_key]
    
    file_cache[key] = data
    logger.debug(f"Cached file with key: {key} ({len(data)} bytes)")


def get_cached_file(key: str) -> Optional[bytes]:
    """Get cached file data"""
    return file_cache.get(key)


def download_file_from_url(url: str, timeout: int = 30) -> bytes:
    """Download file from URL with error handling and retry"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading file from: {url} (attempt {attempt + 1})")
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Download in chunks
            file_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_data += chunk
            
            logger.info(f"Successfully downloaded {len(file_data)} bytes")
            return file_data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise DatabaseServerError(f"Failed to download file after {max_retries} attempts: {e}")


def validate_user_id(user_id: str) -> bool:
    """Validate user_id format"""
    if not user_id or len(user_id) < 3 or len(user_id) > 100:
        return False
    # Allow alphanumeric, underscore, hyphen
    return all(c.isalnum() or c in '_-' for c in user_id)


def validate_model_id(model_id: str) -> bool:
    """Validate model_id format (UUID)"""
    if not model_id or len(model_id) != 36:
        return False
    # Basic UUID format check
    try:
        parts = model_id.split('-')
        if len(parts) != 5:
            return False
        lengths = [8, 4, 4, 4, 12]
        return all(len(part) == length and all(c in '0123456789abcdef-' for c in part) 
                  for part, length in zip(parts, lengths))
    except:
        return False


def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to prevent information leakage"""
    error_str = str(error_msg).lower()
    
    # List of sensitive terms to redact
    sensitive_terms = [
        'three_d_models', 'texture_jobs', 'table', 'schema', 'database', 
        'supabase', 'postgresql', 'postgres', 'pgrst', 'column', 'relation'
    ]
    
    # If any sensitive term is found, return generic message
    for term in sensitive_terms:
        if term in error_str:
            return 'Database operation failed'
    
    # Return sanitized message
    return 'Service temporarily unavailable'


@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler with security sanitization"""
    logger.error(f"Unhandled error: {error}")
    logger.error(traceback.format_exc())
    
    if isinstance(error, DatabaseServerError):
        # Even custom errors should be sanitized
        sanitized_msg = sanitize_error_message(str(error))
        return jsonify({'error': sanitized_msg}), 500
    
    # Never expose internal error details in production
    if DEV_MODE:
        return jsonify({
            'error': 'Internal server error',
            'debug_message': str(error)  # Only in dev mode
        }), 500
    else:
        return jsonify({
            'error': 'Internal server error'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Supabase connection
        supabase_status = "disconnected"
        if supabase:
            try:
                # Simple query to test connection
                result = supabase.table(MODELS_TABLE).select('id').limit(1).execute()
                supabase_status = "connected"
            except Exception as e:
                # Sanitize error message for security
                logger.error(f"Supabase connection test failed: {e}")
                supabase_status = "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'supabase_status': supabase_status,
            'cache_size': len(file_cache),
            'dev_mode': DEV_MODE,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics"""
    try:
        # Get model counts by status
        stats = {
            'cache_size': len(file_cache),
            'cache_max_size': CACHE_MAX_SIZE,
            'dev_mode': DEV_MODE,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if supabase:
            try:
                # Count models by status
                result = supabase.table(MODELS_TABLE).select('status', count='exact').execute()
                stats['total_models'] = result.count if hasattr(result, 'count') else 0
                
                # Get status breakdown
                status_result = supabase.rpc('get_model_stats').execute()
                if status_result.data:
                    stats['model_stats'] = status_result.data
                    
            except Exception as e:
                logger.warning(f"Could not fetch model stats: {e}")
                stats['model_stats_error'] = sanitize_error_message(str(e))
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/texture-jobs/<user_id>', methods=['GET'])
def list_user_texture_jobs(user_id: str):
    """List all texture jobs for a specific user"""
    try:
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Query parameters
        status = request.args.get('status')  # Filter by status
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100
        offset = int(request.args.get('offset', 0))
        
        # Build query
        query = supabase.table(TEXTURE_JOBS_TABLE).select('*').eq('user_id', user_id)
        
        if status:
            query = query.eq('status', status)
        
        # Order by created_at desc and apply pagination
        query = query.order('created_at', desc=True).range(offset, offset + limit - 1)
        
        result = query.execute()
        
        return jsonify({
            'texture_jobs': result.data,
            'count': len(result.data),
            'offset': offset,
            'limit': limit,
            'has_more': len(result.data) == limit
        })
        
    except Exception as e:
        logger.error(f"Error listing texture jobs for user {user_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/texture-jobs/<user_id>/<texture_job_id>', methods=['GET'])
def get_texture_job_details(user_id: str, texture_job_id: str):
    """Get detailed information about a specific texture job"""
    try:
        # Validate inputs
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not validate_model_id(texture_job_id):
            return jsonify({'error': 'Invalid texture_job_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Query specific texture job
        result = supabase.table(TEXTURE_JOBS_TABLE).select('*').eq('user_id', user_id).eq('id', texture_job_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Texture job not found'}), 404
        
        texture_job = result.data[0]
        
        # Add download endpoints
        texture_job['download_endpoints'] = {
            'textured_model': f"/download-texture/{user_id}/{texture_job_id}/model"
        }
        
        return jsonify(texture_job)
        
    except Exception as e:
        logger.error(f"Error getting texture job {texture_job_id} for user {user_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/download-texture/<user_id>/<texture_job_id>/<file_type>', methods=['GET'])
def download_textured_file(user_id: str, texture_job_id: str, file_type: str):
    """Download individual textured model files (obj, mtl, textures, etc.)"""
    try:
        # Validate inputs
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not validate_model_id(texture_job_id):
            return jsonify({'error': 'Invalid texture_job_id format'}), 400
            
        if file_type not in ['obj', 'mtl', 'glb', 'albedo', 'metallic', 'roughness', 'combined', 'reference']:
            return jsonify({'error': 'Invalid file_type. Must be: obj, mtl, glb, albedo, metallic, roughness, combined, reference'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Check cache first
        cache_key = get_cache_key(user_id, texture_job_id, file_type)
        cached_data = get_cached_file(cache_key)
        
        if cached_data:
            logger.info(f"Serving cached textured file: {cache_key}")
            content_type, filename = _get_texture_file_info(file_type, texture_job_id)
            return Response(
                cached_data,
                content_type=content_type,
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        # Get texture job details from database
        result = supabase.table(TEXTURE_JOBS_TABLE).select('*').eq('user_id', user_id).eq('id', texture_job_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Texture job not found'}), 404
        
        texture_job = result.data[0]
        
        # Check if texture job is completed
        if texture_job['status'] != 'completed':
            return jsonify({'error': f'Texture job is not ready. Status: {texture_job["status"]}'}), 400
        
        # Build file URL based on GLB path (replace filename)
        base_glb_url = texture_job.get('textured_glb_path')
        if not base_glb_url:
            return jsonify({'error': 'Texture job base path not available'}), 404
        
        # Construct URLs for different file types based on the directory structure
        file_url = _build_texture_file_url(base_glb_url, file_type)
        
        if not file_url:
            return jsonify({'error': f'Could not construct URL for {file_type} file'}), 404
        
        # Download file from Supabase storage
        try:
            file_data = download_file_from_url(file_url)
            
            # Cache the downloaded file
            cache_file(cache_key, file_data)
            
            # Return file
            content_type, filename = _get_texture_file_info(file_type, texture_job_id)
            return Response(
                file_data,
                content_type=content_type,
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
            
        except DatabaseServerError as e:
            return jsonify({'error': sanitize_error_message(str(e))}), 500
        
    except Exception as e:
        logger.error(f"Error downloading textured {file_type} for texture job {texture_job_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


def _build_texture_file_url(base_glb_url: str, file_type: str) -> str:
    """Build URL for specific texture file type based on GLB URL"""
    if not base_glb_url:
        return ""
    
    # Extract base directory from GLB URL
    base_url = base_glb_url.replace('/textured_model.glb', '')
    
    # Map file types to actual filenames
    file_mapping = {
        'obj': 'obj_model.obj',
        'mtl': 'mtl_material.mtl', 
        'glb': 'textured_model.glb',
        'albedo': 'albedo_texture.jpg',
        'metallic': 'metallic_texture.jpg',
        'roughness': 'roughness_texture.jpg',
        'combined': 'metallic_roughness_combined.png',
        'reference': 'ai_reference.png'
    }
    
    filename = file_mapping.get(file_type)
    if not filename:
        return ""
    
    return f"{base_url}/{filename}"


def _get_texture_file_info(file_type: str, texture_job_id: str) -> tuple:
    """Get content type and filename for texture file type"""
    file_info = {
        'obj': ('application/octet-stream', f'{texture_job_id}_textured.obj'),
        'mtl': ('text/plain', f'{texture_job_id}_material.mtl'),
        'glb': ('application/octet-stream', f'{texture_job_id}_textured.glb'),
        'albedo': ('image/jpeg', f'{texture_job_id}_albedo.jpg'),
        'metallic': ('image/jpeg', f'{texture_job_id}_metallic.jpg'), 
        'roughness': ('image/jpeg', f'{texture_job_id}_roughness.jpg'),
        'combined': ('image/png', f'{texture_job_id}_pbr.png'),
        'reference': ('image/png', f'{texture_job_id}_reference.png')
    }
    
    return file_info.get(file_type, ('application/octet-stream', f'{texture_job_id}_{file_type}'))


@app.route('/download-texture/<user_id>/<texture_job_id>/complete', methods=['GET'])
def download_complete_textured_model(user_id: str, texture_job_id: str):
    """Download complete textured model package (OBJ + MTL + all textures as ZIP)"""
    try:
        # Validate inputs
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not validate_model_id(texture_job_id):
            return jsonify({'error': 'Invalid texture_job_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Get texture job details
        result = supabase.table(TEXTURE_JOBS_TABLE).select('*').eq('user_id', user_id).eq('id', texture_job_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Texture job not found'}), 404
        
        texture_job = result.data[0]
        
        if texture_job['status'] != 'completed':
            return jsonify({'error': f'Texture job not ready. Status: {texture_job["status"]}'}), 400
        
        base_glb_url = texture_job.get('textured_glb_path')
        if not base_glb_url:
            return jsonify({'error': 'Texture job base path not available'}), 404
        
        # Download all required files for PBR workflow
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Required files for complete PBR material
            required_files = [
                ('obj', 'obj_model.obj'),
                ('mtl', 'mtl_material.mtl'),
                ('albedo', 'albedo_texture.jpg'),
                ('metallic', 'metallic_texture.jpg'),
                ('roughness', 'roughness_texture.jpg'),
                ('combined', 'metallic_roughness_combined.png'),
                ('reference', 'ai_reference.png')
            ]
            
            successful_files = []
            
            for file_type, filename in required_files:
                try:
                    file_url = _build_texture_file_url(base_glb_url, file_type)
                    if file_url:
                        file_data = download_file_from_url(file_url, timeout=15)
                        zip_file.writestr(filename, file_data)
                        successful_files.append(filename)
                        logger.info(f"Added {filename} to textured model package ({len(file_data)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to download {filename}: {e}")
                    # Continue with other files - some might be missing
            
            # Add a README file explaining the contents
            readme_content = f"""AI Textured Model Package
Generated by Orca Engine

Contents:
- obj_model.obj: 3D geometry
- mtl_material.mtl: Material definition file
- albedo_texture.jpg: Base color/diffuse texture
- metallic_texture.jpg: Metallic map
- roughness_texture.jpg: Roughness map  
- metallic_roughness_combined.png: Combined PBR texture
- ai_reference.png: Reference image used for generation

Usage in Godot:
1. Import the OBJ file
2. Godot will automatically detect and import the MTL file
3. Assign the imported textures to create PBR materials

Files included: {', '.join(successful_files)}
"""
            zip_file.writestr('README.txt', readme_content)
        
        zip_data = zip_buffer.getvalue()
        
        if len(zip_data) == 0:
            return jsonify({'error': 'Failed to create textured model package'}), 500
        
        return Response(
            zip_data,
            content_type='application/zip',
            headers={
                'Content-Disposition': f'attachment; filename=textured_model_{texture_job_id}.zip',
                'X-Files-Included': str(len(successful_files))
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating complete textured model package for {texture_job_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/download-texture/<user_id>/<texture_job_id>/model', methods=['GET'])  
def download_textured_model(user_id: str, texture_job_id: str):
    """Backward compatibility - redirects to GLB download"""
    return download_textured_file(user_id, texture_job_id, 'glb')


@app.route('/models/<user_id>', methods=['GET'])
def list_user_models(user_id: str):
    """List all 3D models for a specific user with texture job information"""
    try:
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Query parameters
        status = request.args.get('status')  # Filter by status
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100
        offset = int(request.args.get('offset', 0))
        include_textured = request.args.get('include_textured', 'true').lower() == 'true'
        
        # Get models first
        query = supabase.table(MODELS_TABLE).select('*').eq('user_id', user_id)
        
        if status:
            query = query.eq('status', status)
        
        # Order by created_at desc and apply pagination
        query = query.order('created_at', desc=True).range(offset, offset + limit - 1)
        result = query.execute()
        
        models = result.data if hasattr(result, 'data') else []
        
        # If including texture information, fetch texture jobs for each model
        if include_textured and models:
            model_ids = [model['id'] for model in models]
            
            # Get texture jobs for these models
            texture_query = supabase.table(TEXTURE_JOBS_TABLE).select('*').in_('base_model_id', model_ids)
            texture_result = texture_query.execute()
            
            # Group texture jobs by base_model_id
            texture_jobs_by_model = {}
            for texture_job in texture_result.data:
                base_model_id = texture_job['base_model_id']
                if base_model_id not in texture_jobs_by_model:
                    texture_jobs_by_model[base_model_id] = []
                texture_jobs_by_model[base_model_id].append({
                    'texture_job_id': texture_job['id'],
                    'texture_status': texture_job['status'],
                    'texture_type': texture_job['texture_type'],
                    'texture_resolution': texture_job['texture_resolution'],
                    'textured_mesh_url': texture_job['textured_mesh_url'],
                    'textured_glb_path': texture_job['textured_glb_path'],
                    'textured_obj_path': texture_job['textured_obj_path'],
                    'texture_created_at': texture_job['created_at']
                })
            
            # Add texture job information to each model
            for model in models:
                model_id = model['id']
                texture_jobs = texture_jobs_by_model.get(model_id, [])
                # Sort by created_at desc
                texture_jobs.sort(key=lambda x: x['texture_created_at'] if x['texture_created_at'] else '', reverse=True)
                
                model['texture_jobs'] = texture_jobs
                model['texture_job_count'] = len(texture_jobs)
                model['has_completed_textures'] = any(tj.get('texture_status') == 'completed' for tj in texture_jobs)
                model['latest_texture_status'] = texture_jobs[0].get('texture_status') if texture_jobs else None
        
        return jsonify({
            'models': models,
            'count': len(models),
            'offset': offset,
            'limit': limit,
            'has_more': len(models) == limit,
            'include_textured': include_textured
        })
        
    except Exception as e:
        logger.error(f"Error listing models for user {user_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/models/<user_id>/<model_id>', methods=['GET'])
def get_model_details(user_id: str, model_id: str):
    """Get detailed information about a specific 3D model"""
    try:
        # Validate inputs
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not validate_model_id(model_id):
            return jsonify({'error': 'Invalid model_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Query specific model
        result = supabase.table(MODELS_TABLE).select('*').eq('user_id', user_id).eq('id', model_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Model not found'}), 404
        
        model = result.data[0]
        
        # Add download endpoints
        model['download_endpoints'] = {
            'obj_file': f"/download/{user_id}/{model_id}/obj",
            'reference_image': f"/download/{user_id}/{model_id}/image"
        }
        
        return jsonify(model)
        
    except Exception as e:
        logger.error(f"Error getting model {model_id} for user {user_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/download/<user_id>/<model_id>/<file_type>', methods=['GET'])
def download_model_file(user_id: str, model_id: str, file_type: str):
    """Download 3D model file (obj) or reference image"""
    try:
        # Validate inputs
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not validate_model_id(model_id):
            return jsonify({'error': 'Invalid model_id format'}), 400
            
        if file_type not in ['obj', 'image']:
            return jsonify({'error': 'Invalid file_type. Must be "obj" or "image"'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Check cache first
        cache_key = get_cache_key(user_id, model_id, file_type)
        cached_data = get_cached_file(cache_key)
        
        if cached_data:
            logger.info(f"Serving cached file: {cache_key}")
            # Determine content type and filename
            if file_type == 'obj':
                content_type = 'application/octet-stream'
                filename = f'{model_id}.obj'
            else:  # image
                content_type = 'image/png'
                filename = f'{model_id}_reference.png'
            
            return Response(
                cached_data,
                content_type=content_type,
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        
        # Get model details from database
        result = supabase.table(MODELS_TABLE).select('*').eq('user_id', user_id).eq('id', model_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Model not found'}), 404
        
        model = result.data[0]
        
        # Check if model is completed
        if model['status'] != 'completed':
            return jsonify({'error': f'Model is not ready. Status: {model["status"]}'}), 400
        
        # Get the appropriate URL
        if file_type == 'obj':
            file_url = model.get('output_file_url')
            content_type = 'application/octet-stream'
            filename = f'{model_id}.obj'
        else:  # image
            file_url = model.get('reference_image_url')
            content_type = 'image/png'
            filename = f'{model_id}_reference.png'
        
        if not file_url:
            return jsonify({'error': f'{file_type} file not available for this model'}), 404
        
        # Download file from Supabase storage
        try:
            file_data = download_file_from_url(file_url)
            
            # Cache the downloaded file
            cache_file(cache_key, file_data)
            
            # Return file
            return Response(
                file_data,
                content_type=content_type,
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
            
        except DatabaseServerError as e:
            return jsonify({'error': sanitize_error_message(str(e))}), 500
        
    except Exception as e:
        logger.error(f"Error downloading {file_type} for model {model_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/models/<user_id>/search', methods=['GET'])
def search_user_models(user_id: str):
    """Search models by prompt or other criteria"""
    try:
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Search parameters
        query_text = request.args.get('q', '').strip()
        model_type = request.args.get('type')
        status = request.args.get('status', 'completed')
        limit = min(int(request.args.get('limit', 20)), 100)
        
        if not query_text:
            return jsonify({'error': 'Search query (q) parameter is required'}), 400
        
        # Build search query
        query = supabase.table(MODELS_TABLE).select('*').eq('user_id', user_id)
        
        # Add filters
        if status:
            query = query.eq('status', status)
        if model_type:
            query = query.eq('model_type', model_type)
        
        # Text search in prompt (case-insensitive)
        query = query.ilike('prompt', f'%{query_text}%')
        
        # Order by relevance (created_at desc for now)
        query = query.order('created_at', desc=True).limit(limit)
        
        result = query.execute()
        
        return jsonify({
            'models': result.data,
            'count': len(result.data),
            'query': query_text,
            'filters': {
                'type': model_type,
                'status': status
            }
        })
        
    except Exception as e:
        logger.error(f"Error searching models for user {user_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/models/<user_id>/textured', methods=['GET'])
def list_user_textured_models(user_id: str):
    """List all models that have completed texture jobs for a specific user"""
    try:
        # Validate user_id
        if not validate_user_id(user_id):
            return jsonify({'error': 'Invalid user_id format'}), 400
            
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        # Query parameters
        texture_type = request.args.get('texture_type')  # Filter by texture type
        limit = min(int(request.args.get('limit', 20)), 100)  # Max 100
        offset = int(request.args.get('offset', 0))
        
        # Get models that have completed texture jobs
        texture_query = supabase.table(TEXTURE_JOBS_TABLE).select('base_model_id, *').eq('user_id', user_id).eq('status', 'completed')
        
        if texture_type:
            texture_query = texture_query.eq('texture_type', texture_type)
        
        texture_query = texture_query.order('created_at', desc=True).range(offset, offset + limit - 1)
        texture_result = texture_query.execute()
        
        if not texture_result.data:
            return jsonify({
                'textured_models': [],
                'count': 0,
                'offset': offset,
                'limit': limit,
                'has_more': False
            })
        
        # Get the base model IDs
        base_model_ids = list(set([tj['base_model_id'] for tj in texture_result.data if tj['base_model_id']]))
        
        # Fetch the base models
        models_query = supabase.table(MODELS_TABLE).select('*').in_('id', base_model_ids)
        models_result = models_query.execute()
        
        # Create a map of model_id -> model data
        models_map = {model['id']: model for model in models_result.data}
        
        # Combine texture jobs with their base models
        textured_models = []
        for texture_job in texture_result.data:
            base_model_id = texture_job['base_model_id']
            if base_model_id in models_map:
                base_model = models_map[base_model_id].copy()
                # Add texture job information
                base_model['texture_job'] = {
                    'texture_job_id': texture_job['id'],
                    'texture_status': texture_job['status'],
                    'texture_type': texture_job['texture_type'],
                    'texture_resolution': texture_job['texture_resolution'],
                    'textured_mesh_url': texture_job['textured_mesh_url'],
                    'textured_glb_path': texture_job['textured_glb_path'],
                    'textured_obj_path': texture_job['textured_obj_path'],
                    'texture_created_at': texture_job['created_at'],
                    'texture_prompt': texture_job['prompt']
                }
                # Add download endpoints
                base_model['download_endpoints'] = {
                    'base_obj_file': f"/download/{user_id}/{base_model_id}/obj",
                    'base_reference_image': f"/download/{user_id}/{base_model_id}/image",
                    'textured_model': f"/download-texture/{user_id}/{texture_job['id']}/model"
                }
                textured_models.append(base_model)
        
        return jsonify({
            'textured_models': textured_models,
            'count': len(textured_models),
            'offset': offset,
            'limit': limit,
            'has_more': len(texture_result.data) == limit,
            'texture_type_filter': texture_type
        })
        
    except Exception as e:
        logger.error(f"Error listing textured models for user {user_id}: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/models/recent', methods=['GET'])
def get_recent_models():
    """Get recently created models across all users (public endpoint)"""
    try:
        if not supabase:
            return jsonify({'error': 'Supabase client not available'}), 503
        
        limit = min(int(request.args.get('limit', 10)), 50)
        status = request.args.get('status', 'completed')
        
        # Get recent completed models
        query = supabase.table(MODELS_TABLE).select('id,user_id,prompt,model_type,created_at,reference_image_url')
        
        if status:
            query = query.eq('status', status)
        
        query = query.order('created_at', desc=True).limit(limit)
        
        result = query.execute()
        
        # Anonymize user_ids for privacy
        for model in result.data:
            user_id = model.get('user_id', '')
            if len(user_id) > 8:
                # Show first 4 and last 4 characters
                model['user_id'] = f"{user_id[:4]}...{user_id[-4:]}"
        
        return jsonify({
            'models': result.data,
            'count': len(result.data)
        })
        
    except Exception as e:
        logger.error(f"Error getting recent models: {e}")
        return jsonify({'error': sanitize_error_message(str(e))}), 500


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Godot AI Database Server',
        'description': '3D Model Retrieval Service for Orca Engine',
        'version': '1.1.0',
        'endpoints': {
            '/health': 'Health check',
            '/stats': 'Server statistics',
            '/models/{user_id}': 'List user models (with texture job info)',
            '/models/{user_id}/{model_id}': 'Get model details',
            '/models/{user_id}/textured': 'List models with completed textures',
            '/models/{user_id}/search?q={query}': 'Search user models',
            '/models/recent': 'Get recent public models',
            '/download/{user_id}/{model_id}/obj': 'Download OBJ file',
            '/download/{user_id}/{model_id}/image': 'Download reference image',
            '/texture-jobs/{user_id}': 'List user texture jobs',
            '/texture-jobs/{user_id}/{texture_job_id}': 'Get texture job details',
            '/download-texture/{user_id}/{texture_job_id}/model': 'Download textured GLB file',
            '/download-texture/{user_id}/{texture_job_id}/complete': 'Download complete textured package (ZIP with OBJ+MTL+textures)',
            '/download-texture/{user_id}/{texture_job_id}/obj': 'Download textured OBJ file',
            '/download-texture/{user_id}/{texture_job_id}/mtl': 'Download material file',
            '/download-texture/{user_id}/{texture_job_id}/albedo': 'Download albedo texture',
            '/download-texture/{user_id}/{texture_job_id}/metallic': 'Download metallic texture',
            '/download-texture/{user_id}/{texture_job_id}/roughness': 'Download roughness texture',
            '/download-texture/{user_id}/{texture_job_id}/combined': 'Download PBR combined texture'
        },
        'features': {
            'texture_integration': 'Models linked to texture jobs',
            'textured_model_listing': 'Dedicated endpoint for textured models',
            'caching': 'File caching for performance',
            'security': 'Input validation and error sanitization'
        },
        'gpu_server': 'https://shapegen.orcaengine.ai',
        'texture_server': 'https://texture.orcaengine.ai',
        'dev_mode': DEV_MODE
    })


if __name__ == '__main__':
    logger.info("🚀 Starting Godot AI Database Server")
    logger.info(f"   Port: {PORT}")
    logger.info(f"   Dev Mode: {DEV_MODE}")
    logger.info(f"   Supabase URL: {SUPABASE_URL}")
    logger.info(f"   Cache Max Size: {CACHE_MAX_SIZE}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEV_MODE,
        threaded=True
    )
