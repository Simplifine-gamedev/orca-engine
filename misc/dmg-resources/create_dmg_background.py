#!/usr/bin/env python3
"""
Create a DMG background image with drag arrow for Orca Engine
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_dmg_background():
    # DMG window size (width, height)
    width, height = 640, 400
    
    # Create image with dark background
    bg = Image.new('RGBA', (width, height), (45, 45, 45, 255))
    draw = ImageDraw.Draw(bg)
    
    # Draw drag arrow from left (app) to right (Applications)
    # App position: ~160, 200 (center-left)
    # Applications position: ~480, 200 (center-right)
    
    app_x, app_y = 160, 200
    apps_x, apps_y = 480, 200
    
    # Draw curved arrow
    arrow_color = (100, 149, 237, 200)  # Semi-transparent blue
    arrow_width = 4
    
    # Draw arrow shaft (curved line)
    # Control points for bezier curve
    start_x, start_y = app_x + 60, app_y
    end_x, end_y = apps_x - 60, apps_y
    
    # Simple curved arrow using multiple line segments
    steps = 20
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        
        # Bezier curve calculation
        x1 = start_x + t * (end_x - start_x) + t * (1 - t) * 40 * (0.5 - abs(t - 0.5))
        y1 = start_y - 20 * t * (1 - t)  # Arc upward
        
        x2 = start_x + t_next * (end_x - start_x) + t_next * (1 - t_next) * 40 * (0.5 - abs(t_next - 0.5))
        y2 = start_y - 20 * t_next * (1 - t_next)
        
        draw.line([(x1, y1), (x2, y2)], fill=arrow_color, width=arrow_width)
    
    # Draw arrowhead
    arrow_tip_x, arrow_tip_y = end_x, end_y
    arrow_points = [
        (arrow_tip_x, arrow_tip_y),
        (arrow_tip_x - 20, arrow_tip_y - 10),
        (arrow_tip_x - 20, arrow_tip_y + 10)
    ]
    draw.polygon(arrow_points, fill=arrow_color)
    
    # Add subtle text
    try:
        # Try to use system font
        font_size = 16
        font = ImageFont.load_default()
        text = "Drag to install"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = start_y - 60
        
        # Draw text with shadow
        shadow_color = (0, 0, 0, 100)
        text_color = (200, 200, 200, 180)
        
        draw.text((text_x + 1, text_y + 1), text, font=font, fill=shadow_color)
        draw.text((text_x, text_y), text, font=font, fill=text_color)
    except:
        pass  # Skip text if font loading fails
    
    return bg

if __name__ == "__main__":
    # Create the background image
    bg_image = create_dmg_background()
    
    # Save as PNG
    output_path = "misc/dmg-resources/dmg-background.png"
    bg_image.save(output_path, "PNG")
    print(f"DMG background created: {output_path}")
    
    # Also save a @2x version for retina displays
    bg_2x = bg_image.resize((1280, 800), Image.LANCZOS)
    bg_2x.save("misc/dmg-resources/dmg-background@2x.png", "PNG")
    print("Retina version created: dmg-background@2x.png")
