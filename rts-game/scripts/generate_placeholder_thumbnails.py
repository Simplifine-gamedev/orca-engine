#!/usr/bin/env python3
"""
Generate placeholder thumbnail images for RTS game factions.

This script creates simple colored placeholder thumbnails when actual 3D models
are not available yet. Useful for development and testing.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL (Pillow) is required")
    print("Install with: pip install pillow")
    sys.exit(1)


def create_placeholder_thumbnail(
    output_path: str,
    text: str,
    color: Tuple[int, int, int] = (100, 100, 100),
    resolution: Tuple[int, int] = (256, 256)
) -> bool:
    """
    Create a simple placeholder thumbnail with text.
    
    Args:
        output_path: Path to save the PNG thumbnail
        text: Text to display on thumbnail
        color: Background color (R, G, B)
        resolution: Output image resolution (width, height)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create image with colored background
        image = Image.new('RGB', resolution, color)
        draw = ImageDraw.Draw(image)
        
        # Add text
        # Use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (resolution[0] - text_width) // 2
        y = (resolution[1] - text_height) // 2
        
        # Draw text with shadow for readability
        shadow_offset = 2
        draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        image.save(output_path, 'PNG')
        print(f"✓ Generated placeholder: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create placeholder {output_path}: {e}")
        return False


def generate_faction_placeholders(output_dir: str = "./rts-game/public/assets"):
    """
    Generate placeholder thumbnails for all factions.
    """
    
    # Faction color schemes
    colors = {
        'undead': {
            'units': (80, 60, 100),      # Dark purple
            'buildings': (60, 50, 80)    # Darker purple
        },
        'human': {
            'units': (100, 120, 150),    # Blue
            'buildings': (80, 100, 130)  # Darker blue
        },
        'dwarf': {
            'units': (150, 100, 80),     # Brown/orange
            'buildings': (130, 80, 60)   # Darker brown
        }
    }
    
    # Undead faction
    print("\n=== Generating Undead faction placeholders ===")
    undead_units = ['worker', 'soldier_light', 'soldier_medium', 'soldier_heavy']
    undead_buildings = ['city_center', 'barracks', 'farm', 'bank', 'mill', 'warehouse', 'tower']
    
    for unit_id in undead_units:
        path = os.path.join(output_dir, 'units', 'undead', f'{unit_id}_thumbnail.png')
        create_placeholder_thumbnail(
            path,
            unit_id.replace('_', '\n'),
            colors['undead']['units']
        )
    
    for building_id in undead_buildings:
        path = os.path.join(output_dir, 'buildings', 'undead', f'{building_id}_thumbnail.png')
        create_placeholder_thumbnail(
            path,
            building_id.replace('_', '\n'),
            colors['undead']['buildings']
        )
    
    # Human faction
    print("\n=== Generating Human faction placeholders ===")
    human_units = ['footman', 'archer', 'knight']
    human_buildings = ['barracks']
    
    for unit_id in human_units:
        path = os.path.join(output_dir, 'units', 'human', f'{unit_id}_preview.png')
        if not os.path.exists(path):  # Only create if doesn't exist
            create_placeholder_thumbnail(
                path,
                unit_id,
                colors['human']['units']
            )
    
    for building_id in human_buildings:
        path = os.path.join(output_dir, 'buildings', 'human', f'{building_id}_thumbnail.png')
        create_placeholder_thumbnail(
            path,
            building_id,
            colors['human']['buildings']
        )
    
    # Dwarf faction
    print("\n=== Generating Dwarf faction placeholders ===")
    dwarf_units = ['warrior', 'rifleman', 'hammerer']
    dwarf_buildings = ['barracks']
    
    for unit_id in dwarf_units:
        path = os.path.join(output_dir, 'units', 'dwarf', f'{unit_id}_preview.png')
        if not os.path.exists(path):  # Only create if doesn't exist
            create_placeholder_thumbnail(
                path,
                unit_id,
                colors['dwarf']['units']
            )
    
    for building_id in dwarf_buildings:
        path = os.path.join(output_dir, 'buildings', 'dwarf', f'{building_id}_thumbnail.png')
        create_placeholder_thumbnail(
            path,
            building_id,
            colors['dwarf']['buildings']
        )
    
    print("\n✓ All placeholder thumbnails generated!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate placeholder thumbnails')
    parser.add_argument('--output-dir', type=str, 
                       default='./rts-game/public/assets',
                       help='Output directory for thumbnails')
    
    args = parser.parse_args()
    generate_faction_placeholders(args.output_dir)
