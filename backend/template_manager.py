"""
Template Manager - Download and manage base game templates for Orca Engine
Handles downloading curated open-source Godot 4 game frameworks from GitHub
"""

import json
import os
import requests
import zipfile
import tempfile
import shutil
from typing import List, Dict, Optional
from pathlib import Path

# Path to the catalog file
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "game_templates_catalog.json")


class TemplateManager:
    """Manages base game template catalog and installations"""
    
    def __init__(self, catalog_path: str = CATALOG_PATH):
        self.catalog_path = catalog_path
        self.templates = self._load_catalog()
    
    def _load_catalog(self) -> List[Dict]:
        """Load the template catalog from JSON file"""
        try:
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Template catalog not found at {self.catalog_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing template catalog: {e}")
            return []
    
    def list_templates(self, category: Optional[str] = None) -> List[Dict]:
        """
        List available templates, optionally filtered by category
        
        Args:
            category: Optional category filter (fps, 2d-platformer, 3d-platformer, city-building, slasher)
        
        Returns:
            List of template dictionaries
        """
        if category:
            return [t for t in self.templates if t.get('category') == category]
        return self.templates
    
    def get_template(self, template_id: str) -> Optional[Dict]:
        """Get a specific template by ID"""
        for template in self.templates:
            if template.get('id') == template_id:
                return template
        return None
    
    def download_template(self, template_id: str, target_path: str) -> Dict:
        """
        Download and extract a template to the target path
        
        Args:
            template_id: ID of the template to download
            target_path: Directory where the template should be extracted
        
        Returns:
            Dict with status, message, and path information
        """
        template = self.get_template(template_id)
        if not template:
            return {
                "success": False,
                "error": f"Template '{template_id}' not found in catalog",
                "available_templates": [t['id'] for t in self.templates]
            }
        
        # Get download URL
        source = template.get('source', {})
        download_url = source.get('url')
        if not download_url:
            return {
                "success": False,
                "error": f"No download URL configured for template '{template_id}'"
            }
        
        # Ensure target path exists
        try:
            os.makedirs(target_path, exist_ok=True)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create target directory: {str(e)}"
            }
        
        # Download the template
        try:
            print(f"Downloading template '{template_id}' from {download_url}...")
            response = requests.get(download_url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Create temporary file for the zip
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_path = tmp_file.name
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                
                print(f"Downloaded {downloaded} bytes")
            
            # Extract the zip file
            print(f"Extracting to {target_path}...")
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                # Get the root directory name from the zip
                zip_contents = zip_ref.namelist()
                if not zip_contents:
                    return {
                        "success": False,
                        "error": "Downloaded archive is empty"
                    }
                
                # GitHub archives typically have a root folder
                root_folder = zip_contents[0].split('/')[0] if '/' in zip_contents[0] else None
                
                # Extract to a temporary location first
                temp_extract = tempfile.mkdtemp()
                zip_ref.extractall(temp_extract)
                
                # Move contents to target path
                if root_folder:
                    source_dir = os.path.join(temp_extract, root_folder)
                else:
                    source_dir = temp_extract
                
                # Copy all files from extracted directory to target
                for item in os.listdir(source_dir):
                    s = os.path.join(source_dir, item)
                    d = os.path.join(target_path, item)
                    if os.path.isdir(s):
                        if os.path.exists(d):
                            shutil.rmtree(d)
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
                
                # Cleanup temp directories
                shutil.rmtree(temp_extract)
            
            # Cleanup downloaded zip
            os.unlink(tmp_path)
            
            # Verify project.godot exists
            project_file = os.path.join(target_path, "project.godot")
            if not os.path.exists(project_file):
                return {
                    "success": False,
                    "error": "Downloaded template does not contain a valid project.godot file",
                    "path": target_path
                }
            
            print(f"Template '{template_id}' successfully installed to {target_path}")
            return {
                "success": True,
                "message": f"Template '{template['name']}' installed successfully",
                "template_id": template_id,
                "template_name": template['name'],
                "path": target_path,
                "entry_scene": template.get('entry_scene'),
                "license": template.get('license')
            }
            
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to download template: {str(e)}"
            }
        except zipfile.BadZipFile:
            return {
                "success": False,
                "error": "Downloaded file is not a valid zip archive"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during installation: {str(e)}"
            }
    
    def get_categories(self) -> List[str]:
        """Get list of all available categories"""
        categories = set()
        for template in self.templates:
            cat = template.get('category')
            if cat:
                categories.add(cat)
        return sorted(list(categories))


# Global instance
template_manager = TemplateManager()


def list_templates(category: Optional[str] = None) -> List[Dict]:
    """Convenience function to list templates"""
    return template_manager.list_templates(category)


def install_template(template_id: str, target_path: str) -> Dict:
    """Convenience function to install a template"""
    return template_manager.download_template(template_id, target_path)


def get_template_info(template_id: str) -> Optional[Dict]:
    """Convenience function to get template info"""
    return template_manager.get_template(template_id)


def get_categories() -> List[str]:
    """Convenience function to get available categories"""
    return template_manager.get_categories()

