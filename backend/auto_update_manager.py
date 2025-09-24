"""
© 2025 Simplifine Corp. Personal Non‑Commercial License.
See LICENSES/COMPANY-NONCOMMERCIAL.md for terms.

Auto-Update Manager for Orca Engine
Provides event-driven update checking and notification system
"""

import os
import json
import time
import hashlib
import threading
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import requests
from flask import jsonify
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import semantic_version as semver

@dataclass
class UpdateInfo:
    """Information about an available update"""
    version: str
    download_url: str
    release_notes: str
    file_size: int
    published_at: str
    is_critical: bool = False
    minimum_version: str = None

class AutoUpdateManager:
    """Manages auto-update functionality for Orca Engine"""
    
    def __init__(self):
        # Configuration
        self.github_repo = "Simplifine-gamedev/orca-engine"
        self.current_version = self._get_current_version()
        self.check_interval = int(os.getenv('UPDATE_CHECK_INTERVAL', '3600'))  # 1 hour default
        self.appcast_url = os.getenv('APPCAST_URL', 'https://simplifine-gamedev.github.io/orca-engine/appcast.xml')
        self.appcast_url_windows = os.getenv('APPCAST_URL_WINDOWS', 'https://simplifine-gamedev.github.io/orca-engine/appcast-windows.xml')
        
        # Update state
        self.last_check = 0
        self.cached_update_info: Optional[UpdateInfo] = None
        self.update_callbacks: List[Callable] = []
        self.check_lock = threading.Lock()
        
        # Background checker thread
        self.checker_thread = None
        self.should_stop = threading.Event()
        
        print(f"AUTO_UPDATE: Initialized for Orca Engine v{self.current_version}")
    
    def _get_current_version(self) -> str:
        """Get current version matching the GitHub workflow format (0.01.{SHA})"""
        import subprocess
        
        try:
            # Method 1: Environment variable (for deployments) - highest priority
            env_version = os.getenv('ORCA_VERSION')
            if env_version:
                print(f"AUTO_UPDATE: Version from environment: {env_version}")
                return env_version
            
            # Method 2: Generate version matching GitHub workflow format
            # This matches the format used in workflows: 0.01.{SHORT_SHA}
            result = subprocess.run([
                'git', 'rev-parse', '--short=8', 'HEAD'
            ], capture_output=True, text=True, cwd=self._get_repo_root())
            
            if result.returncode == 0:
                short_sha = result.stdout.strip()
                version = f"0.01.{short_sha}"
                print(f"AUTO_UPDATE: Version from git SHA (matching workflow): {version}")
                return version
                        
        except Exception as e:
            print(f"AUTO_UPDATE: Git version detection failed: {e}")
        
        # Method 3: Read from version.py as fallback
        try:
            version_file = os.path.join(os.path.dirname(__file__), '..', 'version.py')
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    content = f.read()
                    # Extract version from version.py
                    for line in content.split('\n'):
                        if line.strip().startswith('version') and '=' in line:
                            version = line.split('=')[1].strip().strip('"\'')
                            print(f"AUTO_UPDATE: Version from version.py: {version}")
                            return version
        except Exception as e:
            print(f"AUTO_UPDATE: Error reading version.py: {e}")
            
        # Final fallback
        print("AUTO_UPDATE: Using fallback version 0.01.unknown")
        return '0.01.unknown'
    
    def _get_repo_root(self) -> str:
        """Find the git repository root directory"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != '/':
            if os.path.exists(os.path.join(current, '.git')):
                return current
            current = os.path.dirname(current)
        # Fallback to parent directory of backend
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def register_update_callback(self, callback: Callable[[UpdateInfo], None]):
        """Register a callback to be called when an update is available"""
        self.update_callbacks.append(callback)
    
    def start_background_checker(self):
        """Start background thread that periodically checks for updates"""
        if self.checker_thread and self.checker_thread.is_alive():
            return
        
        self.should_stop.clear()
        self.checker_thread = threading.Thread(target=self._background_check_loop, daemon=True)
        self.checker_thread.start()
        print("AUTO_UPDATE: Background checker started")
    
    def stop_background_checker(self):
        """Stop background update checker"""
        self.should_stop.set()
        if self.checker_thread:
            self.checker_thread.join(timeout=5)
    
    def _background_check_loop(self):
        """Background loop that checks for updates"""
        while not self.should_stop.is_set():
            try:
                # Check for updates
                update_info = self.check_for_updates(force=False)
                
                if update_info:
                    print(f"AUTO_UPDATE: New version available: {update_info.version}")
                    
                    # Notify all registered callbacks
                    for callback in self.update_callbacks:
                        try:
                            callback(update_info)
                        except Exception as e:
                            print(f"AUTO_UPDATE: Callback error: {e}")
                
                # Wait for next check
                self.should_stop.wait(self.check_interval)
                
            except Exception as e:
                print(f"AUTO_UPDATE: Background check error: {e}")
                # Wait a bit before retrying on error
                self.should_stop.wait(300)  # 5 minutes on error
    
    def check_for_updates(self, force: bool = False, platform: str = None) -> Optional[UpdateInfo]:
        """Check for available updates"""
        with self.check_lock:
            current_time = time.time()
            
            # Skip if recently checked and not forced
            if not force and (current_time - self.last_check) < 300:  # 5 minutes minimum
                return self.cached_update_info
            
            try:
                # Determine platform
                if platform is None:
                    platform = self._detect_platform()
                
                print(f"AUTO_UPDATE: Checking for updates (platform: {platform}, current: v{self.current_version})")
                
                # Check GitHub releases first (most reliable)
                update_info = self._check_github_releases(platform)
                
                # Fallback to appcast if GitHub fails
                if not update_info:
                    update_info = self._check_appcast(platform)
                
                self.last_check = current_time
                self.cached_update_info = update_info
                
                if update_info:
                    print(f"AUTO_UPDATE: Update available - v{update_info.version}")
                else:
                    print(f"AUTO_UPDATE: No updates available")
                
                return update_info
                
            except Exception as e:
                print(f"AUTO_UPDATE: Check failed: {e}")
                return None
    
    def _detect_platform(self) -> str:
        """Detect current platform"""
        import platform
        system = platform.system().lower()
        
        if system == 'darwin':
            return 'mac'
        elif system == 'windows':
            return 'windows'
        elif system == 'linux':
            return 'linux'
        else:
            return 'unknown'
    
    def _check_github_releases(self, platform: str) -> Optional[UpdateInfo]:
        """Check GitHub releases for updates"""
        try:
            # Get latest release from GitHub API
            url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')
            
            # Compare versions
            if not self._is_newer_version(latest_version, self.current_version):
                return None
            
            # Find appropriate download asset
            download_url = None
            file_size = 0
            
            for asset in release_data.get('assets', []):
                asset_name = asset['name'].lower()
                
                if platform == 'mac' and asset_name.endswith('.dmg'):
                    download_url = asset['browser_download_url']
                    file_size = asset['size']
                    break
                elif platform == 'windows' and asset_name.endswith('.exe'):
                    download_url = asset['browser_download_url']
                    file_size = asset['size']
                    break
                elif platform == 'linux' and any(ext in asset_name for ext in ['.appimage', '.tar.gz', '.deb']):
                    download_url = asset['browser_download_url']
                    file_size = asset['size']
                    break
            
            if not download_url:
                print(f"AUTO_UPDATE: No suitable download found for platform {platform}")
                return None
            
            # Check if this is a critical update
            is_critical = self._is_critical_update(release_data.get('body', ''))
            
            return UpdateInfo(
                version=latest_version,
                download_url=download_url,
                release_notes=release_data.get('body', ''),
                file_size=file_size,
                published_at=release_data['published_at'],
                is_critical=is_critical
            )
            
        except Exception as e:
            print(f"AUTO_UPDATE: GitHub check failed: {e}")
            return None
    
    def _check_appcast(self, platform: str) -> Optional[UpdateInfo]:
        """Check appcast XML for updates (fallback method)"""
        try:
            appcast_url = self.appcast_url_windows if platform == 'windows' else self.appcast_url
            
            response = requests.get(appcast_url, timeout=10)
            response.raise_for_status()
            
            # Parse appcast XML
            root = ET.fromstring(response.content)
            
            # Find the latest item
            items = root.findall('.//item')
            if not items:
                return None
            
            latest_item = items[0]  # Assume first item is latest
            
            # Extract version from title or enclosure
            title = latest_item.findtext('title', '')
            version_match = None
            
            import re
            version_pattern = r'v?(\d+\.\d+\.\d+)'
            version_match = re.search(version_pattern, title)
            
            if not version_match:
                return None
            
            latest_version = version_match.group(1)
            
            # Compare versions
            if not self._is_newer_version(latest_version, self.current_version):
                return None
            
            # Get download URL
            enclosure = latest_item.find('enclosure')
            if enclosure is None:
                return None
            
            download_url = enclosure.get('url')
            file_size = int(enclosure.get('length', 0))
            
            return UpdateInfo(
                version=latest_version,
                download_url=download_url,
                release_notes=latest_item.findtext('description', ''),
                file_size=file_size,
                published_at=latest_item.findtext('pubDate', '')
            )
            
        except Exception as e:
            print(f"AUTO_UPDATE: Appcast check failed: {e}")
            return None
    
    def _is_newer_version(self, remote_version: str, current_version: str) -> bool:
        """Compare version strings to see if remote is newer"""
        try:
            # Handle the 0.01.{SHA} format used by GitHub workflows
            if remote_version.startswith('0.01.') and current_version.startswith('0.01.'):
                # Extract SHA parts for comparison
                remote_sha = remote_version.split('0.01.')[1]
                current_sha = current_version.split('0.01.')[1]
                
                # If they're different SHAs, consider remote as newer
                # (This assumes any new commit is a potential update)
                return remote_sha != current_sha
            
            # Try semantic versioning comparison for standard versions
            try:
                remote_ver = semver.Version(remote_version)
                current_ver = semver.Version(current_version)
                return remote_ver > current_ver
            except Exception:
                pass
            
            # Fallback to simple string comparison
            try:
                remote_parts = [int(x) for x in remote_version.split('.')]
                current_parts = [int(x) for x in current_version.split('.')]
                
                # Pad shorter version with zeros
                max_len = max(len(remote_parts), len(current_parts))
                remote_parts.extend([0] * (max_len - len(remote_parts)))
                current_parts.extend([0] * (max_len - len(current_parts)))
                
                return remote_parts > current_parts
            except Exception:
                # If all else fails, assume different versions mean update available
                return remote_version != current_version
                
        except Exception as e:
            print(f"AUTO_UPDATE: Version comparison error: {e}")
            return False
    
    def _is_critical_update(self, release_notes: str) -> bool:
        """Determine if an update is critical based on release notes"""
        critical_keywords = [
            'critical', 'security', 'urgent', 'hotfix', 
            'vulnerability', 'patch', 'important'
        ]
        
        notes_lower = release_notes.lower()
        return any(keyword in notes_lower for keyword in critical_keywords)
    
    def _format_version_for_display(self, version: str) -> str:
        """Format version string for user-friendly display"""
        if not version:
            return "Unknown"
        
        # Handle the 0.01.{SHA} format used by workflows
        if version.startswith('0.01.') and len(version) > 5:
            sha_part = version.split('0.01.')[1]
            if sha_part == 'unknown':
                return "0.01 (Unknown)"
            else:
                return f"0.01.{sha_part[:8]}"  # Show first 8 chars of SHA
            
        # Handle development builds
        if '-dev.' in version:
            # 1.0.0-dev.5+g1234567 -> "1.0.0 (Development Build)"
            base = version.split('-dev.')[0]
            return f"{base} (Development Build)"
        elif '-beta.' in version:
            # 1.0.0-beta.1 -> "1.0.0 Beta 1"
            parts = version.split('-beta.')
            if len(parts) == 2:
                return f"{parts[0]} Beta {parts[1].split('+')[0]}"
        elif '-alpha.' in version:
            # 1.0.0-alpha.1 -> "1.0.0 Alpha 1"
            parts = version.split('-alpha.')
            if len(parts) == 2:
                return f"{parts[0]} Alpha {parts[1].split('+')[0]}"
        elif '-rc.' in version:
            # 1.0.0-rc.1 -> "1.0.0 Release Candidate 1"
            parts = version.split('-rc.')
            if len(parts) == 2:
                return f"{parts[0]} RC {parts[1].split('+')[0]}"
        elif '-unknown' in version:
            return version.replace('-unknown', ' (Unknown)')
        
        # Remove build metadata if present
        if '+' in version:
            version = version.split('+')[0]
            
        return version
    
    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable time ago"""
        if not timestamp:
            return "Never"
        
        now = time.time()
        diff = now - timestamp
        
        if diff < 60:
            return "Just now"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff < 604800:
            days = int(diff / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
    
    def _determine_update_type(self, current: str, new: str) -> str:
        """Determine the type of update (major, minor, patch, prerelease)"""
        try:
            # Clean versions for comparison
            curr_clean = current.split('-')[0].split('+')[0]
            new_clean = new.split('-')[0].split('+')[0]
            
            curr_ver = semver.Version(curr_clean)
            new_ver = semver.Version(new_clean)
            
            if new_ver.major > curr_ver.major:
                return "major"
            elif new_ver.minor > curr_ver.minor:
                return "minor"
            elif new_ver.patch > curr_ver.patch:
                return "patch"
            else:
                # Check for prerelease differences
                if '-' in new and '-' not in current:
                    return "downgrade"  # Stable to prerelease
                elif '-' not in new and '-' in current:
                    return "stable"  # Prerelease to stable
                else:
                    return "prerelease"
        except Exception as e:
            print(f"AUTO_UPDATE: Error determining update type: {e}")
            return "unknown"
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status with enhanced display information"""
        current_display = self._format_version_for_display(self.current_version)
        
        status = {
            'current_version': self.current_version,
            'current_version_display': current_display,
            'is_development_build': '-dev.' in self.current_version,
            'is_prerelease': any(pre in self.current_version for pre in ['-alpha.', '-beta.', '-rc.']),
            'last_check': self.last_check,
            'last_check_human': datetime.fromtimestamp(self.last_check).isoformat() if self.last_check else None,
            'update_available': self.cached_update_info is not None,
            'background_checker_running': self.checker_thread and self.checker_thread.is_alive(),
            'check_interval': self.check_interval
        }
        
        if self.cached_update_info:
            update_display = self._format_version_for_display(self.cached_update_info.version)
            status['update_info'] = {
                'version': self.cached_update_info.version,
                'version_display': update_display,
                'download_url': self.cached_update_info.download_url,
                'file_size': self.cached_update_info.file_size,
                'is_critical': self.cached_update_info.is_critical,
                'published_at': self.cached_update_info.published_at,
                'update_type': self._determine_update_type(self.current_version, self.cached_update_info.version)
            }
        else:
            status['update_info'] = None
            
        return status
    
    def download_update(self, update_info: UpdateInfo, download_path: str = None) -> Dict[str, Any]:
        """Download an update file"""
        try:
            if not download_path:
                # Create downloads directory
                downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads', 'OrcaEngine')
                os.makedirs(downloads_dir, exist_ok=True)
                
                # Generate filename from URL
                filename = update_info.download_url.split('/')[-1]
                download_path = os.path.join(downloads_dir, filename)
            
            print(f"AUTO_UPDATE: Downloading {update_info.version} to {download_path}")
            
            # Download with progress
            response = requests.get(update_info.download_url, stream=True, timeout=300)  # 5 min timeout
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Verify download
            if os.path.exists(download_path):
                actual_size = os.path.getsize(download_path)
                if total_size > 0 and actual_size != total_size:
                    raise Exception(f"Download size mismatch: {actual_size} != {total_size}")
                
                print(f"AUTO_UPDATE: Download complete - {actual_size} bytes")
                
                return {
                    'success': True,
                    'download_path': download_path,
                    'file_size': actual_size,
                    'version': update_info.version
                }
            else:
                raise Exception("Downloaded file not found")
                
        except Exception as e:
            print(f"AUTO_UPDATE: Download failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def schedule_install(self, download_path: str, restart_app: bool = True) -> Dict[str, Any]:
        """Schedule installation of downloaded update"""
        try:
            platform = self._detect_platform()
            
            if platform == 'mac':
                return self._schedule_mac_install(download_path, restart_app)
            elif platform == 'windows':
                return self._schedule_windows_install(download_path, restart_app)
            elif platform == 'linux':
                return self._schedule_linux_install(download_path, restart_app)
            else:
                return {'success': False, 'error': 'Unsupported platform'}
                
        except Exception as e:
            print(f"AUTO_UPDATE: Install scheduling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _schedule_mac_install(self, download_path: str, restart_app: bool) -> Dict[str, Any]:
        """Schedule macOS installation"""
        try:
            if download_path.endswith('.dmg'):
                # Mount DMG and copy app
                install_script = f"""
                #!/bin/bash
                set -e
                
                echo "Mounting DMG..."
                MOUNT_POINT=$(hdiutil mount "{download_path}" | grep -o '/Volumes/.*')
                
                echo "Installing Orca Engine..."
                cp -R "$MOUNT_POINT/Orca.app" "/Applications/"
                
                echo "Unmounting DMG..."
                hdiutil unmount "$MOUNT_POINT"
                
                echo "Cleaning up..."
                rm "{download_path}"
                
                {"echo 'Restarting Orca Engine...' && open '/Applications/Orca.app'" if restart_app else ""}
                """
                
                # Save install script
                script_path = os.path.join(os.path.dirname(download_path), 'install_update.sh')
                with open(script_path, 'w') as f:
                    f.write(install_script)
                os.chmod(script_path, 0o755)
                
                return {
                    'success': True,
                    'install_script': script_path,
                    'message': 'Update ready to install. Run install script to complete.'
                }
            else:
                return {'success': False, 'error': 'Unsupported Mac update format'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _schedule_windows_install(self, download_path: str, restart_app: bool) -> Dict[str, Any]:
        """Schedule Windows installation"""
        try:
            if download_path.endswith('.exe'):
                # For .exe files, just run them
                import subprocess
                
                if restart_app:
                    # Run installer and exit current app
                    subprocess.Popen([download_path, '/SILENT'], shell=False)
                    return {
                        'success': True,
                        'message': 'Update installer launched. Current app will exit.',
                        'requires_exit': True
                    }
                else:
                    return {
                        'success': True,
                        'installer_path': download_path,
                        'message': 'Update downloaded. Run installer manually when ready.'
                    }
            else:
                return {'success': False, 'error': 'Unsupported Windows update format'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _schedule_linux_install(self, download_path: str, restart_app: bool) -> Dict[str, Any]:
        """Schedule Linux installation"""
        try:
            if download_path.endswith('.AppImage'):
                # Make executable and optionally replace current
                os.chmod(download_path, 0o755)
                
                return {
                    'success': True,
                    'executable_path': download_path,
                    'message': 'Update downloaded as AppImage. Replace your current Orca Engine executable.'
                }
            elif download_path.endswith('.deb'):
                # Install via dpkg
                install_cmd = f"sudo dpkg -i '{download_path}'"
                return {
                    'success': True,
                    'install_command': install_cmd,
                    'message': f'Run: {install_cmd}'
                }
            else:
                return {'success': False, 'error': 'Unsupported Linux update format'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_release_notes(self, version: str = None) -> str:
        """Get release notes for a specific version or latest"""
        try:
            if version:
                url = f"https://api.github.com/repos/{self.github_repo}/releases/tags/v{version}"
            else:
                url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            return release_data.get('body', 'No release notes available.')
            
        except Exception as e:
            print(f"AUTO_UPDATE: Failed to get release notes: {e}")
            return 'Release notes unavailable.'
    
    def force_check_now(self, platform: str = None) -> Optional[UpdateInfo]:
        """Force an immediate update check"""
        return self.check_for_updates(force=True, platform=platform)

# Global instance
auto_update_manager = AutoUpdateManager()
