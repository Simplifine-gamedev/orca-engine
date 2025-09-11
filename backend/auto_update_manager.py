"""
© 2025 Simplifine Corp. Auto-update system for Orca Engine.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""

import requests
import json
import os
import platform
import tempfile
import hashlib
import threading
import time
from datetime import datetime, timedelta
from packaging import version
from typing import Dict, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoUpdateManager:
    """
    Comprehensive auto-update system for Orca Engine that automatically detects 
    new releases and handles cross-platform updates.
    """
    
    def __init__(self):
        self.github_api_url = "https://api.github.com/repos/simplifine-llc/orca"
        self.current_version = self._get_current_version()
        self.platform_info = self._get_platform_info()
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        self.last_check_time = None
        self.background_thread = None
        self.is_checking = False
        self._stop_background_check = False
        
    def _get_current_version(self) -> str:
        """Get the current version of Orca Engine"""
        try:
            # Try to read version from version.txt or similar file
            version_files = ['version.txt', 'VERSION', '.version']
            for version_file in version_files:
                if os.path.exists(version_file):
                    with open(version_file, 'r') as f:
                        return f.read().strip()
            
            # Fallback to environment variable or default
            return os.getenv('ORCA_VERSION', '1.0.0')
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return '1.0.0'
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform-specific information for downloads"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        platform_map = {
            'darwin': 'mac',
            'windows': 'windows', 
            'linux': 'linux'
        }
        
        arch_map = {
            'x86_64': 'x64',
            'amd64': 'x64',
            'arm64': 'arm64',
            'aarch64': 'arm64'
        }
        
        return {
            'os': platform_map.get(system, system),
            'arch': arch_map.get(machine, machine),
            'system': system,
            'machine': machine
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp', 0)
        return time.time() - cached_time < self.cache_timeout
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        return None
    
    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for available updates from GitHub releases
        Returns: Dict with update information
        """
        cache_key = 'latest_release'
        
        # Return cached data if valid
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self.is_checking = True
            logger.info("Checking for updates...")
            
            # Get latest release from GitHub API
            headers = {'Accept': 'application/vnd.github.v3+json'}
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(f"{self.github_api_url}/releases/latest", 
                                  headers=headers, timeout=10)
            
            if response.status_code == 404:
                # No releases found
                result = {
                    'update_available': False,
                    'current_version': self.current_version,
                    'latest_version': None,
                    'message': 'No releases available',
                    'error': None
                }
                self._cache_data(cache_key, result)
                return result
            
            response.raise_for_status()
            release_data = response.json()
            
            latest_version = release_data['tag_name'].lstrip('v')
            release_notes = release_data.get('body', '')
            release_url = release_data.get('html_url', '')
            is_prerelease = release_data.get('prerelease', False)
            
            # Compare versions
            update_available = self._compare_versions(self.current_version, latest_version)
            
            # Find appropriate download asset
            download_info = self._find_download_asset(release_data.get('assets', []))
            
            result = {
                'update_available': update_available,
                'current_version': self.current_version,
                'latest_version': latest_version,
                'release_notes': release_notes,
                'release_url': release_url,
                'is_prerelease': is_prerelease,
                'download_url': download_info.get('url'),
                'download_size': download_info.get('size'),
                'download_name': download_info.get('name'),
                'platform_supported': download_info.get('supported', False),
                'error': None,
                'checked_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self._cache_data(cache_key, result)
            self.last_check_time = datetime.now()
            
            logger.info(f"Update check complete. Update available: {update_available}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking for updates: {e}")
            return {
                'update_available': False,
                'current_version': self.current_version,
                'latest_version': None,
                'error': f'Network error: {str(e)}',
                'checked_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return {
                'update_available': False,
                'current_version': self.current_version,
                'latest_version': None,
                'error': f'Unexpected error: {str(e)}',
                'checked_at': datetime.now().isoformat()
            }
        finally:
            self.is_checking = False
    
    def _compare_versions(self, current: str, latest: str) -> bool:
        """Compare version strings using semantic versioning"""
        try:
            return version.parse(latest) > version.parse(current)
        except Exception:
            # Fallback to string comparison if version parsing fails
            return latest != current
    
    def _find_download_asset(self, assets: list) -> Dict[str, Any]:
        """Find the appropriate download asset for current platform"""
        platform_os = self.platform_info['os']
        platform_arch = self.platform_info['arch']
        
        # Platform-specific file patterns
        patterns = {
            'mac': ['.dmg', '-mac', '-macos', '-darwin'],
            'windows': ['.exe', '.msi', '-win', '-windows'],
            'linux': ['.AppImage', '.tar.gz', '.deb', '.rpm', '-linux']
        }
        
        platform_patterns = patterns.get(platform_os, [])
        
        for asset in assets:
            asset_name = asset['name'].lower()
            
            # Check if asset matches platform
            if any(pattern in asset_name for pattern in platform_patterns):
                return {
                    'url': asset['browser_download_url'],
                    'name': asset['name'],
                    'size': asset['size'],
                    'supported': True
                }
        
        # No matching asset found
        return {'supported': False}
    
    def download_update(self, download_url: str, progress_callback=None) -> Dict[str, Any]:
        """
        Download update file
        Args:
            download_url: URL to download the update from
            progress_callback: Optional callback for progress updates
        Returns: Dict with download result
        """
        try:
            logger.info(f"Starting download from: {download_url}")
            
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            filename = download_url.split('/')[-1]
            temp_path = os.path.join(temp_dir, f"orca_update_{filename}")
            
            # Download with progress tracking
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            progress_callback(progress)
            
            # Verify download
            if os.path.getsize(temp_path) != total_size and total_size > 0:
                raise Exception("Download size mismatch")
            
            logger.info(f"Download completed: {temp_path}")
            
            return {
                'success': True,
                'file_path': temp_path,
                'size': downloaded_size,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {
                'success': False,
                'file_path': None,
                'error': str(e)
            }
    
    def install_update(self, file_path: str) -> Dict[str, Any]:
        """
        Install downloaded update file
        Args:
            file_path: Path to the downloaded update file
        Returns: Dict with installation result
        """
        try:
            logger.info(f"Installing update from: {file_path}")
            
            if not os.path.exists(file_path):
                raise Exception("Update file not found")
            
            platform_os = self.platform_info['os']
            
            if platform_os == 'mac':
                return self._install_mac_update(file_path)
            elif platform_os == 'windows':
                return self._install_windows_update(file_path)
            elif platform_os == 'linux':
                return self._install_linux_update(file_path)
            else:
                raise Exception(f"Unsupported platform: {platform_os}")
                
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'requires_restart': False
            }
    
    def _install_mac_update(self, file_path: str) -> Dict[str, Any]:
        """Install update on macOS"""
        try:
            # For DMG files, we need to mount and copy
            if file_path.endswith('.dmg'):
                # This would typically involve mounting the DMG and copying the app
                # For now, we'll return instructions for manual installation
                return {
                    'success': True,
                    'message': 'Please mount the DMG and drag Orca to Applications folder',
                    'requires_restart': True,
                    'manual_install': True
                }
            else:
                raise Exception("Unsupported Mac installer format")
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'requires_restart': False}
    
    def _install_windows_update(self, file_path: str) -> Dict[str, Any]:
        """Install update on Windows"""
        try:
            # For EXE/MSI files, run the installer
            if file_path.endswith('.exe') or file_path.endswith('.msi'):
                import subprocess
                # Run installer in silent mode
                result = subprocess.run([file_path, '/S'], capture_output=True)
                
                return {
                    'success': result.returncode == 0,
                    'message': 'Update installed successfully' if result.returncode == 0 else 'Installation failed',
                    'requires_restart': True
                }
            else:
                raise Exception("Unsupported Windows installer format")
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'requires_restart': False}
    
    def _install_linux_update(self, file_path: str) -> Dict[str, Any]:
        """Install update on Linux"""
        try:
            if file_path.endswith('.AppImage'):
                # For AppImage, just make it executable and replace current binary
                import subprocess
                import shutil
                
                # Make executable
                os.chmod(file_path, 0o755)
                
                # This would typically involve replacing the current AppImage
                # For now, return instructions
                return {
                    'success': True,
                    'message': 'Please replace your current Orca AppImage with the downloaded file',
                    'requires_restart': True,
                    'manual_install': True
                }
            else:
                raise Exception("Unsupported Linux installer format")
                
        except Exception as e:
            return {'success': False, 'error': str(e), 'requires_restart': False}
    
    def start_background_checking(self, interval_hours: int = 24):
        """Start background thread for periodic update checking"""
        if self.background_thread and self.background_thread.is_alive():
            return  # Already running
        
        self._stop_background_check = False
        self.background_thread = threading.Thread(
            target=self._background_check_loop,
            args=(interval_hours,),
            daemon=True
        )
        self.background_thread.start()
        logger.info(f"Started background update checking (every {interval_hours} hours)")
    
    def stop_background_checking(self):
        """Stop background update checking"""
        self._stop_background_check = True
        if self.background_thread:
            self.background_thread.join(timeout=5)
        logger.info("Stopped background update checking")
    
    def _background_check_loop(self, interval_hours: int):
        """Background loop for checking updates"""
        interval_seconds = interval_hours * 3600
        
        while not self._stop_background_check:
            try:
                # Check for updates
                result = self.check_for_updates()
                
                # Log if update is available
                if result.get('update_available'):
                    logger.info(f"Update available: {result.get('latest_version')}")
                
                # Wait for next check
                for _ in range(interval_seconds):
                    if self._stop_background_check:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Background update check error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the update system"""
        return {
            'current_version': self.current_version,
            'platform': self.platform_info,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'is_checking': self.is_checking,
            'background_checking': self.background_thread and self.background_thread.is_alive(),
            'cache_entries': len(self.cache)
        }

# Global instance
update_manager = AutoUpdateManager()