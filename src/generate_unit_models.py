#!/usr/bin/env python3
"""
Script to generate 3D models for RTS units using the Orca backend's 3D generation API.
This script reads the character definitions and generates the required 3D models.
"""

import json
import os
import sys
import requests
from pathlib import Path

# Configuration
BACKEND_URL = os.getenv('ORCA_BACKEND_URL', 'http://localhost:8000')
OUTPUT_DIR = 'models'

def load_character_definitions():
    """Load character definitions from JSON file"""
    json_path = Path(__file__).parent.parent / 'generated_factions' / 'all_factions_characters.json'
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_3d_model(prompt: str, output_filename: str):
    """Generate a 3D model using the backend API"""
    url = f"{BACKEND_URL}/api/3d/generate/text"
    
    print(f"Generating 3D model: {output_filename}")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(
            url,
            json={"prompt": prompt},
            timeout=300  # 5 minute timeout for 3D generation
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get('success'):
            model_url = result.get('model_url')
            print(f"✓ Model generated: {model_url}")
            return model_url
        else:
            print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return None

def download_model(model_url: str, output_path: str):
    """Download the generated model file"""
    try:
        response = requests.get(model_url, timeout=60)
        response.raise_for_status()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def generate_all_models():
    """Generate all required 3D models"""
    data = load_character_definitions()
    
    # Check if 3D generation is available
    try:
        health_response = requests.get(f"{BACKEND_URL}/api/3d/health", timeout=5)
        if health_response.status_code != 200:
            print("⚠ 3D generation service not available. Skipping model generation.")
            print("Models will need to be created manually or service needs to be enabled.")
            return
    except:
        print("⚠ Cannot connect to backend. Skipping model generation.")
        print("Ensure backend is running with 3D generation enabled.")
        return
    
    models_to_generate = []
    
    # Collect character models
    for char_id, char_data in data.get('characters', {}).items():
        model_info = char_data.get('model', {})
        if 'generationPrompt' in model_info:
            models_to_generate.append({
                'type': 'unit',
                'id': char_id,
                'prompt': model_info['generationPrompt'],
                'output': model_info['path'].replace('res://', '')
            })
    
    # Collect building models
    for building_id, building_data in data.get('buildings', {}).items():
        model_info = building_data.get('model', {})
        if 'generationPrompt' in model_info:
            models_to_generate.append({
                'type': 'building',
                'id': building_id,
                'prompt': model_info['generationPrompt'],
                'output': model_info['path'].replace('res://', '')
            })
    
    # Collect projectile models
    for proj_id, proj_data in data.get('projectiles', {}).items():
        model_info = proj_data.get('model', {})
        if 'generationPrompt' in model_info:
            models_to_generate.append({
                'type': 'projectile',
                'id': proj_id,
                'prompt': model_info['generationPrompt'],
                'output': model_info['path'].replace('res://', '')
            })
    
    print(f"\n{'='*60}")
    print(f"Generating {len(models_to_generate)} 3D models")
    print(f"{'='*60}\n")
    
    success_count = 0
    for i, model in enumerate(models_to_generate, 1):
        print(f"\n[{i}/{len(models_to_generate)}] {model['type'].upper()}: {model['id']}")
        print("-" * 60)
        
        model_url = generate_3d_model(model['prompt'], model['output'])
        if model_url:
            if download_model(model_url, model['output']):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Generation complete: {success_count}/{len(models_to_generate)} models generated")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    generate_all_models()
