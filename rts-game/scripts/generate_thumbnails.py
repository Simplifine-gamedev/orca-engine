#!/usr/bin/env python3
"""
Generate 2D PNG thumbnail images from GLB 3D models for RTS game factions.

This script renders GLB models from a fixed camera angle and saves them as PNG thumbnails.
Suitable for generating faction building and unit thumbnails.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import trimesh
    import numpy as np
    from PIL import Image
    import pyrender
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Install with: pip install trimesh pillow pyrender")
    sys.exit(1)


def render_glb_to_thumbnail(
    glb_path: str,
    output_path: str,
    resolution: Tuple[int, int] = (256, 256),
    camera_distance: float = 3.0,
    camera_angle: Tuple[float, float] = (45, 45)
) -> bool:
    """
    Render a GLB model to a PNG thumbnail.
    
    Args:
        glb_path: Path to the GLB model file
        output_path: Path to save the PNG thumbnail
        resolution: Output image resolution (width, height)
        camera_distance: Distance of camera from model
        camera_angle: Camera angle (elevation, azimuth) in degrees
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the GLB model
        mesh = trimesh.load(glb_path)
        
        # Convert to pyrender scene
        if isinstance(mesh, trimesh.Scene):
            scene = pyrender.Scene.from_trimesh_scene(mesh)
        else:
            scene = pyrender.Scene()
            mesh_node = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh_node)
        
        # Center the model
        bounds = mesh.bounds if hasattr(mesh, 'bounds') else np.array([[-1, -1, -1], [1, 1, 1]])
        center = (bounds[0] + bounds[1]) / 2
        scale = np.linalg.norm(bounds[1] - bounds[0])
        
        # Set up camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        
        # Calculate camera position based on angle
        elevation_rad = np.radians(camera_angle[0])
        azimuth_rad = np.radians(camera_angle[1])
        
        camera_pos = center + camera_distance * scale * np.array([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.sin(elevation_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad)
        ])
        
        # Point camera at center
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_pos
        
        # Look at center
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)
        
        # Add ambient light
        ambient = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        scene.add(ambient)
        
        # Render
        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, depth = renderer.render(scene)
        
        # Convert to PIL Image and save
        image = Image.fromarray(color)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with transparency if possible
        image.save(output_path, 'PNG')
        
        print(f"✓ Generated thumbnail: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to render {glb_path}: {e}")
        return False


def generate_faction_thumbnails(
    faction_data: Dict,
    models_dir: str,
    output_dir: str,
    resolution: Tuple[int, int] = (256, 256)
) -> Dict[str, str]:
    """
    Generate thumbnails for all units and buildings in a faction.
    
    Args:
        faction_data: Faction configuration data
        models_dir: Directory containing GLB model files
        output_dir: Directory to save thumbnails
        resolution: Thumbnail resolution
        
    Returns:
        Dictionary mapping model IDs to thumbnail paths
    """
    thumbnail_urls = {}
    faction_id = faction_data['id']
    
    # Generate unit thumbnails
    print(f"\nGenerating unit thumbnails for {faction_data['name']}...")
    for unit_id, unit_data in faction_data.get('units', {}).items():
        model_url = unit_data.get('model_url', '')
        if not model_url:
            continue
            
        # Construct paths
        model_filename = f"{unit_id}.glb"
        model_path = os.path.join(models_dir, faction_id, 'units', model_filename)
        thumbnail_path = os.path.join(output_dir, faction_id, 'units', f"{unit_id}_thumbnail.png")
        
        # Generate thumbnail if model exists
        if os.path.exists(model_path):
            if render_glb_to_thumbnail(model_path, thumbnail_path, resolution):
                thumbnail_urls[f"units/{unit_id}"] = f"/assets/units/{faction_id}/{unit_id}_thumbnail.png"
        else:
            print(f"⚠ Model not found: {model_path}")
    
    # Generate building thumbnails
    print(f"\nGenerating building thumbnails for {faction_data['name']}...")
    for building_id, building_data in faction_data.get('buildings', {}).items():
        model_url = building_data.get('model_url', '')
        if not model_url:
            continue
            
        # Construct paths
        model_filename = f"{building_id}.glb"
        model_path = os.path.join(models_dir, faction_id, 'buildings', model_filename)
        thumbnail_path = os.path.join(output_dir, faction_id, 'buildings', f"{building_id}_thumbnail.png")
        
        # Generate thumbnail if model exists
        if os.path.exists(model_path):
            if render_glb_to_thumbnail(model_path, thumbnail_path, resolution, camera_distance=4.0):
                thumbnail_urls[f"buildings/{building_id}"] = f"/assets/buildings/{faction_id}/{building_id}_thumbnail.png"
        else:
            print(f"⚠ Model not found: {model_path}")
    
    return thumbnail_urls


def main():
    parser = argparse.ArgumentParser(description='Generate thumbnails from GLB models')
    parser.add_argument('--faction', type=str, help='Faction ID (e.g., undead)')
    parser.add_argument('--models-dir', type=str, default='./models', help='Directory containing GLB models')
    parser.add_argument('--output-dir', type=str, default='./public/assets', help='Output directory for thumbnails')
    parser.add_argument('--resolution', type=int, default=256, help='Thumbnail resolution (square)')
    parser.add_argument('--faction-config', type=str, help='Path to faction configuration JSON')
    
    args = parser.parse_args()
    
    # Load faction configuration if provided
    if args.faction_config:
        with open(args.faction_config, 'r') as f:
            all_factions = json.load(f)
        
        if args.faction:
            faction_data = all_factions.get(args.faction)
            if not faction_data:
                print(f"Error: Faction '{args.faction}' not found in config")
                sys.exit(1)
            factions_to_process = {args.faction: faction_data}
        else:
            factions_to_process = all_factions
            
        # Generate thumbnails for each faction
        for faction_id, faction_data in factions_to_process.items():
            print(f"\n{'='*60}")
            print(f"Processing faction: {faction_data['name']}")
            print(f"{'='*60}")
            
            thumbnail_urls = generate_faction_thumbnails(
                faction_data,
                args.models_dir,
                args.output_dir,
                (args.resolution, args.resolution)
            )
            
            print(f"\nGenerated {len(thumbnail_urls)} thumbnails for {faction_data['name']}")
    else:
        # Single file mode
        if not args.faction:
            print("Error: Must specify --faction or --faction-config")
            sys.exit(1)
            
        print("Single file mode not implemented. Use --faction-config for batch processing.")
        sys.exit(1)


if __name__ == '__main__':
    main()
