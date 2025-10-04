"""
Parallel Sprite Sheet Generator
Generates sprite sheets progressively using GPT Image with intelligent context passing
"""
import asyncio
import base64
import io
from PIL import Image
from typing import List, Dict, Optional
import time


class SpriteSheetGenerator:
    def __init__(self, openai_client):
        self.client = openai_client
    
    def _create_named_bytes_io(self, data: bytes, name: str):
        """Create file-like object with proper name for MIME type detection"""
        class NamedBytesIO(io.BytesIO):
            def __init__(self, data, name):
                super().__init__(data)
                self.name = name
        return NamedBytesIO(data, name)
    
    async def _generate_single_cell(
        self,
        row: int,
        col: int,
        prompt: str,
        seed_image_b64: str,
        previous_images_b64: List[str],
        grid_width: int,
        grid_height: int,
        style: str,
        size: str = "1024x1024",
        row_description: Optional[str] = None
    ) -> Dict:
        """
        Generate a single cell with full context.
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            prompt: User's overall sprite sheet description
            seed_image_b64: Base64 of the seed/reference image
            previous_images_b64: Base64 of previous frames in this row
            grid_width: Total columns
            grid_height: Total rows
            style: Art style
            size: Output size
        
        Returns:
            Dict with image_data, width, height, etc.
        """
        try:
            cell_number = row * grid_width + col + 1
            total_cells = grid_width * grid_height
            
            # Build context-rich prompt
            context_prompt = f"""Create frame {cell_number} of {total_cells} for this sprite sheet.

SPRITE SHEET CONTEXT:
- Grid: {grid_width} columns × {grid_height} rows
- Current position: Row {row + 1}, Column {col + 1}
- Overall description: {prompt}
- Style: {style}
{f'- Row description: ' + row_description if row_description else ''}

FRAME CONTEXT:
"""
            
            if col == 0:
                # First column - establish the visual baseline that ALL other frames must match
                context_prompt += f"- This is the FIRST frame (frame 1 of {grid_width}) of row {row + 1}\n"
                context_prompt += f"- Show the STARTING POSE of this animation sequence\n"
                context_prompt += f"- This should be the INITIAL position/state before the action begins\n"
                context_prompt += f"- IMPORTANT: This frame establishes the character design that MUST be copied exactly in all subsequent frames\n"
                context_prompt += f"- Use the seed image as your ONLY reference for character appearance\n"
                context_prompt += f"- Match the seed image's style, colors, proportions, and level of detail PRECISELY\n"
            else:
                # Subsequent columns - be VERY specific about animation progression
                progress_percent = (col / (grid_width - 1)) * 100
                context_prompt += f"- This is frame {col + 1} of {grid_width} in row {row + 1} ({progress_percent:.0f}% through the animation)\n"
                context_prompt += f"- {len(previous_images_b64)} previous frame(s) in this row show the progression so far\n"
                context_prompt += f"- Create the NEXT LOGICAL STEP in this {grid_width}-frame animation sequence\n"
                context_prompt += f"- Frame {col + 1} timing: "
                
                if col == 1:
                    context_prompt += f"Show the action 25% complete (early phase of movement)\n"
                elif col == 2:
                    context_prompt += f"Show the action 50% complete (peak/middle of movement)\n"  
                elif col == grid_width - 1:
                    context_prompt += f"Show the action 100% complete (ending pose that loops back to frame 1)\n"
                else:
                    context_prompt += f"Show the action {progress_percent:.0f}% complete\n"
                
                context_prompt += f"- Study the previous frames carefully and continue the motion smoothly\n"
                context_prompt += f"- Maintain EXACT same character size, style, and colors\n"
            
            context_prompt += f"""
CRITICAL REQUIREMENTS FOR CONSISTENCY:
- EXACT SAME CHARACTER from the reference images - same colors, same details, same proportions
- IDENTICAL art style - copy the exact rendering technique from previous frames
- SAME SIZE AND SCALE - the character must be the same pixel dimensions
- CLEAN WHITE or SOLID COLOR background - NO transparency artifacts
- Character completely isolated on background, no shadows or ground elements
- ONLY THE POSE CHANGES - everything else (colors, details, style, size) stays IDENTICAL
- NO creative interpretation - copy the character design exactly, only adjust limb positions
- NO variations in line weight, shading style, or color palette between frames
- Character centered in frame at same position and scale
- If multiple previous frames exist, COPY their visual style exactly

ANIMATION REQUIREMENT:
- This is a sprite sheet for animation - consistency is MORE important than creativity
- Only the limb/body positions should change frame-to-frame
- The character's appearance, style, colors, and proportions must remain COMPLETELY UNCHANGED
- Think of this as tracing the same character in different poses, not drawing new interpretations

Generate frame {cell_number} with the EXACT same character, only the pose changed."""
            
            # Prepare input images
            input_images = []
            
            # DEBUGGING: Log exactly what we're sending
            print(f"\n{'='*80}")
            print(f"CELL [{row},{col}] GENERATION REQUEST DEBUG")
            print(f"{'='*80}")
            print(f"PROMPT LENGTH: {len(context_prompt)} chars")
            print(f"PROMPT PREVIEW (first 500 chars):")
            print(context_prompt[:500])
            print(f"...")
            if row_description:
                print(f"ROW DESCRIPTION: {row_description[:200]}")
            print(f"\nIMAGES BEING SENT:")
            
            # Always include seed image as primary reference
            if seed_image_b64:
                seed_bytes = base64.b64decode(seed_image_b64)
                input_images.append(self._create_named_bytes_io(seed_bytes, "seed.png"))
                print(f"  1. SEED IMAGE: {len(seed_bytes)} bytes (name: seed.png)")
            else:
                print(f"  WARNING: No seed image!")
            
            # Add previous frames in this row for context
            for i, prev_b64 in enumerate(previous_images_b64[-3:]):  # Last 3 frames max
                prev_bytes = base64.b64decode(prev_b64)
                input_images.append(self._create_named_bytes_io(prev_bytes, f"prev_{i}.png"))
                print(f"  {i+2}. PREVIOUS FRAME {i+1}: {len(prev_bytes)} bytes (name: prev_{i}.png)")
            
            print(f"\nTOTAL INPUT IMAGES: {len(input_images)}")
            print(f"GENERATION PARAMS:")
            print(f"  - size: {size}")
            print(f"  - quality: high")
            print(f"  - output_format: png")
            print(f"  - background: opaque (for consistency)")
            print(f"{'='*80}\n")
            
            start_time = time.time()
            
            # Generate the cell
            if input_images:
                # Use edit mode with context images
                # Using opaque background for better consistency
                result = await asyncio.to_thread(
                    self.client.images.edit,
                    model="gpt-image-1",
                    image=input_images,
                    prompt=context_prompt,
                    size=size,
                    quality="high",
                    output_format="png"
                    # Removed background="transparent" - defaults to opaque for better consistency
                )
            else:
                # Fallback to generation if no seed image
                result = await asyncio.to_thread(
                    self.client.images.generate,
                    model="gpt-image-1",
                    prompt=context_prompt,
                    size=size,
                    quality="high",
                    output_format="png"
                    # Opaque background for consistency
                )
            
            generation_time = time.time() - start_time
            
            if not result.data or not hasattr(result.data[0], 'b64_json'):
                raise Exception(f"Cell {row},{col} generation returned no data")
            
            image_b64 = result.data[0].b64_json
            
            # Get dimensions
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            
            print(f"CELL_GEN: Row {row + 1}, Col {col + 1} complete ({width}x{height}) in {generation_time:.2f}s")
            
            return {
                "success": True,
                "row": row,
                "col": col,
                "cell_number": cell_number,
                "image_data": image_b64,
                "width": width,
                "height": height,
                "generation_time": generation_time
            }
            
        except Exception as e:
            print(f"CELL_GEN_ERROR: Row {row + 1}, Col {col + 1} failed: {e}")
            return {
                "success": False,
                "row": row,
                "col": col,
                "error": str(e)
            }
    
    async def generate_sprite_sheet_progressive(
        self,
        prompt: str,
        seed_image_b64: str,
        grid_width: int,
        grid_height: int,
        style: str = "",
        size: str = "1024x1024",
        progress_callback = None,
        row_descriptions: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate sprite sheet with parallel processing and progressive updates.
        
        Process:
        1. Generate first column (all rows) in parallel
        2. For each row in parallel, generate remaining columns sequentially
        3. Each cell gets full context of seed image + previous frames in row
        
        Args:
            prompt: Overall sprite sheet description
            seed_image_b64: Base64 of seed/reference image
            grid_width: Number of columns
            grid_height: Number of rows
            style: Art style
            size: Cell size
            progress_callback: Async function to call with progress updates
        
        Returns:
            Dict with complete sprite sheet data and metadata
        """
        print(f"SPRITE_SHEET_PROGRESSIVE: Starting {grid_width}x{grid_height} grid generation")
        print(f"SPRITE_SHEET_PROGRESSIVE: Prompt: {prompt[:100]}...")
        
        # Storage for all generated cells
        cells = [[None for _ in range(grid_width)] for _ in range(grid_height)]
        
        start_time = time.time()
        
        # PHASE 1: Generate first column (all rows) in PARALLEL
        print(f"PHASE 1: Generating first column ({grid_height} cells) in parallel...")
        
        first_column_tasks = []
        for row in range(grid_height):
            task = self._generate_single_cell(
                row=row,
                col=0,
                prompt=prompt,
                seed_image_b64=seed_image_b64,
                previous_images_b64=[],  # No previous frames for first column
                grid_width=grid_width,
                grid_height=grid_height,
                style=style,
                size=size,
                row_description=(row_descriptions[row] if row_descriptions and row < len(row_descriptions) else None)
            )
            first_column_tasks.append(task)
        
        # Execute all first column cells in parallel
        first_column_results = await asyncio.gather(*first_column_tasks, return_exceptions=True)
        
        # Store results and notify progress
        completed_cells = 0
        for result in first_column_results:
            if isinstance(result, Exception):
                print(f"PHASE 1 ERROR: {result}")
                continue
            
            if result.get('success'):
                row = result['row']
                col = result['col']
                cells[row][col] = result
                completed_cells += 1
                
                if progress_callback:
                    await progress_callback({
                        "phase": 1,
                        "row": row,
                        "col": col,
                        "completed": completed_cells,
                        "total": grid_width * grid_height,
                        "cell_data": result
                    })
        
        print(f"PHASE 1 COMPLETE: {completed_cells}/{grid_height} first column cells generated")
        
        # PHASE 2: Generate remaining columns for each row in PARALLEL
        if grid_width > 1:
            print(f"PHASE 2: Generating remaining columns for all rows in parallel...")
            
            async def generate_row_remainder(row: int):
                """Generate remaining columns for a single row"""
                row_cells_generated = 0
                
                for col in range(1, grid_width):
                    # Collect previous frames in this row for context
                    previous_frames = []
                    for prev_col in range(col):
                        if cells[row][prev_col] and cells[row][prev_col].get('success'):
                            previous_frames.append(cells[row][prev_col]['image_data'])
                    
                    # Generate this cell with full context
                    result = await self._generate_single_cell(
                        row=row,
                        col=col,
                        prompt=prompt,
                        seed_image_b64=seed_image_b64,
                        previous_images_b64=previous_frames,
                        grid_width=grid_width,
                        grid_height=grid_height,
                        style=style,
                        size=size,
                        row_description=(row_descriptions[row] if row_descriptions and row < len(row_descriptions) else None)
                    )
                    
                    if result.get('success'):
                        cells[row][col] = result
                        row_cells_generated += 1
                        
                        nonlocal completed_cells
                        completed_cells += 1
                        
                        if progress_callback:
                            await progress_callback({
                                "phase": 2,
                                "row": row,
                                "col": col,
                                "completed": completed_cells,
                                "total": grid_width * grid_height,
                                "cell_data": result
                            })
                
                print(f"ROW {row + 1} COMPLETE: {row_cells_generated}/{grid_width - 1} remaining cells generated")
                return row_cells_generated
            
            # Process all rows in parallel
            row_tasks = [generate_row_remainder(row) for row in range(grid_height)]
            await asyncio.gather(*row_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Count successful cells
        successful_cells = sum(
            1 for row in cells for cell in row if cell and cell.get('success')
        )
        
        print(f"SPRITE_SHEET_COMPLETE: {successful_cells}/{grid_width * grid_height} cells in {total_time:.2f}s")
        
        return {
            "success": True,
            "cells": cells,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "completed_cells": successful_cells,
            "total_cells": grid_width * grid_height,
            "total_time": total_time,
            "prompt": prompt,
            "style": style
        }

