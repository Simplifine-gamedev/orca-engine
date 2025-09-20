"""
© 2025 Simplifine Corp. Version compatibility checking system for Godot fork.
Personal Non-Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""

import os
import sys
from typing import Dict, Optional, Tuple
import semantic_version

# Add parent directory to path to import version.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import version
except ImportError:
    print("WARNING: Could not import version.py, using fallback versions")
    # Fallback version definitions
    class FallbackVersion:
        api_version = "1.0"
        backend_version = "1.0.0"
        frontend_version = "1.0.0"
        min_api_version = "1.0"
        min_backend_version = "1.0.0"
        min_frontend_version = "1.0.0"
    version = FallbackVersion()


class VersionChecker:
    """Handles version compatibility checking between frontend and backend"""
    
    def __init__(self):
        # Prioritize environment variables (set during deployment) over version.py
        self.api_version = os.getenv('API_VERSION') or getattr(version, 'api_version', '1.0')
        self.backend_version = os.getenv('BACKEND_VERSION') or getattr(version, 'backend_version', '1.0.0')
        self.frontend_version = getattr(version, 'frontend_version', '1.0.0')
        self.min_api_version = getattr(version, 'min_api_version', '1.0')
        self.min_backend_version = getattr(version, 'min_backend_version', '1.0.0')
        self.min_frontend_version = getattr(version, 'min_frontend_version', '1.0.0')
        
        print(f"VERSION_CHECKER: Initialized with API={self.api_version}, Backend={self.backend_version} (env: API_VERSION={os.getenv('API_VERSION')}, BACKEND_VERSION={os.getenv('BACKEND_VERSION')})")
    
    def get_version_info(self) -> Dict[str, str]:
        """Get complete version information for the backend"""
        return {
            'api_version': self.api_version,
            'backend_version': self.backend_version,
            'frontend_version': self.frontend_version,
            'min_api_version': self.min_api_version,
            'min_backend_version': self.min_backend_version,
            'min_frontend_version': self.min_frontend_version
        }
    
    def check_compatibility(self, frontend_version: str, frontend_api_version: str) -> Tuple[bool, str]:
        """
        Check if frontend version is compatible with backend
        
        Args:
            frontend_version: The frontend's version string (e.g., "1.0.0")
            frontend_api_version: The frontend's API version (e.g., "1.0")
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        try:
            # Check API version compatibility (must match exactly for now)
            if frontend_api_version != self.api_version:
                return False, f"API version mismatch: Frontend {frontend_api_version}, Backend {self.api_version}. Please update your Godot editor."
            
            # Check minimum frontend version requirement
            if not self._is_version_compatible(frontend_version, self.min_frontend_version):
                return False, f"Frontend version {frontend_version} is below minimum required {self.min_frontend_version}. Please update your Godot editor."
            
            return True, ""
            
        except Exception as e:
            return False, f"Version compatibility check failed: {str(e)}"
    
    def _is_version_compatible(self, version1: str, min_version: str) -> bool:
        """Check if version1 >= min_version using semantic versioning"""
        try:
            # Try semantic version first
            v1 = semantic_version.Version(version1)
            min_v = semantic_version.Version(min_version)
            return v1 >= min_v
        except ValueError:
            # Fallback to simple string comparison for non-semantic versions
            try:
                v1_parts = [int(x) for x in version1.split('.')]
                min_parts = [int(x) for x in min_version.split('.')]
                
                # Pad shorter version with zeros
                max_len = max(len(v1_parts), len(min_parts))
                v1_parts += [0] * (max_len - len(v1_parts))
                min_parts += [0] * (max_len - len(min_parts))
                
                return v1_parts >= min_parts
            except (ValueError, IndexError):
                print(f"Warning: Could not compare versions {version1} and {min_version}")
                return True  # Allow by default if we can't parse
    
    def get_compatibility_status(self, frontend_version: str, frontend_api_version: str) -> Dict:
        """
        Get detailed compatibility status information
        
        Returns a dictionary with compatibility status, versions, and recommendations
        """
        compatible, error = self.check_compatibility(frontend_version, frontend_api_version)
        
        status = {
            'compatible': compatible,
            'backend_version': self.backend_version,
            'backend_api_version': self.api_version,
            'frontend_version': frontend_version,
            'frontend_api_version': frontend_api_version,
            'min_frontend_version': self.min_frontend_version,
            'min_api_version': self.min_api_version
        }
        
        if not compatible:
            status['error'] = error
            
            # Provide specific recommendations
            if 'API version mismatch' in error:
                status['recommendation'] = "API versions must match exactly. Please ensure you're using a compatible version of the Godot editor."
            elif 'below minimum' in error:
                status['recommendation'] = f"Please update your Godot editor to version {self.min_frontend_version} or higher."
            else:
                status['recommendation'] = "Please check that you're using compatible versions of the frontend and backend."
        
        return status


# Global instance
version_checker = VersionChecker()
