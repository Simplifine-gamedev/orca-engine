"""
© 2025 Simplifine Corp. Personal Non‑Commercial License.
See LICENSES/COMPANY-NONCOMMERCIAL.md for terms.

Autumn Pricing Integration for Orca Engine
Handles subscription management, usage tracking, and billing through Autumn API
"""

import os
import asyncio
from typing import Dict, Tuple, Optional
import requests
import json
import logging

logger = logging.getLogger(__name__)

class AutumnPricingService:
    """Service for integrating with Autumn pricing and billing system"""
    
    def __init__(self):
        self.api_key = os.getenv('AUTUMN_SECRET_KEY')
        if not self.api_key:
            logger.warning("AUTUMN_SECRET_KEY not set - pricing features will be disabled")
        
        self.base_url = "https://api.useautumn.com/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _is_enabled(self) -> bool:
        """Check if Autumn integration is properly configured"""
        return bool(self.api_key)
    
    def check_usage(self, user_id: str, feature_id: str = "ai-requests") -> Tuple[bool, Dict]:
        """
        Check if user has access to make a request
        Returns (allowed, usage_info)
        """
        if not self._is_enabled():
            # If Autumn is not configured, allow unlimited requests (fallback)
            return True, {"fallback_mode": True}
        
        try:
            response = requests.post(
                f"{self.base_url}/check",
                headers=self.headers,
                json={
                    "customer_id": user_id,
                    "feature_id": feature_id
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("allowed", False), data
            else:
                logger.error(f"Autumn check failed: {response.status_code} - {response.text}")
                # Fallback: allow request if API fails
                return True, {"error": "API check failed", "fallback": True}
                
        except Exception as e:
            logger.error(f"Autumn check exception: {e}")
            # Fallback: allow request if API fails
            return True, {"error": str(e), "fallback": True}
    
    def track_usage(self, user_id: str, feature_id: str = "ai-requests", value: int = 1) -> bool:
        """
        Track usage for a user
        Returns True if successful, False otherwise
        """
        if not self._is_enabled():
            return True  # Skip tracking if not configured
        
        try:
            response = requests.post(
                f"{self.base_url}/track",
                headers=self.headers,
                json={
                    "customer_id": user_id,
                    "feature_id": feature_id,
                    "value": value
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Autumn track failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Autumn track exception: {e}")
            return False
    
    def attach_free_tier(self, user_id: str) -> bool:
        """
        Attach the free tier to a new user
        Returns True if successful
        """
        if not self._is_enabled():
            return True
        
        try:
            response = requests.post(
                f"{self.base_url}/attach",
                headers=self.headers,
                json={
                    "customer_id": user_id,
                    "product_id": "free"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully attached free tier to user {user_id}")
                return True
            else:
                logger.error(f"Failed to attach free tier: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Autumn attach exception: {e}")
            return False
    
    def check_and_track_usage(self, user_id: str, feature_id: str = "ai-requests") -> Tuple[bool, Dict]:
        """
        Check if user can make request and track usage atomically
        Auto-assigns free tier to new users
        Returns (allowed, info_dict)
        """
        if not self._is_enabled():
            return True, {"fallback_mode": True}
        
        try:
            # First check access
            allowed, check_info = self.check_usage(user_id, feature_id)
            
            # Auto-assign free tier to brand new users with no product
            # Note: If balance is None, user has no product attached yet
            has_no_balance = check_info.get("balance") is None
            
            if has_no_balance and check_info.get("code") == "feature_found":
                logger.info(f"New user {user_id} detected - assigning free tier")
                self.attach_free_tier(user_id)
                # Re-check after assignment
                allowed, check_info = self.check_usage(user_id, feature_id)
            
            if not allowed:
                # Extract useful information for the client
                return False, {
                    "error": "Request limit exceeded",
                    "limits": check_info.get("balances", []),
                    "upgrade_available": True,
                    "check_data": check_info
                }
            
            # If allowed, track the usage
            track_success = self.track_usage(user_id, feature_id)
            
            return True, {
                "usage_tracked": track_success,
                "check_data": check_info
            }
            
        except Exception as e:
            logger.error(f"Autumn check_and_track exception: {e}")
            # Fallback: allow request
            return True, {"error": str(e), "fallback": True}
    
    def get_checkout_url(self, user_id: str, product_id: str) -> Dict:
        """
        Get checkout URL for product upgrade/purchase
        Returns checkout data or error
        """
        if not self._is_enabled():
            return {"error": "Autumn not configured"}
        
        try:
            response = requests.post(
                f"{self.base_url}/checkout",
                headers=self.headers,
                json={
                    "customer_id": user_id,
                    "product_id": product_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Autumn checkout failed: {response.status_code} - {response.text}")
                return {"error": f"Checkout failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Autumn checkout exception: {e}")
            return {"error": str(e)}
    
    def get_customer_info(self, user_id: str) -> Dict:
        """
        Get customer subscription and usage information
        Returns customer data or error
        """
        if not self._is_enabled():
            return {"error": "Autumn not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/customers/{user_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Autumn customer info failed: {response.status_code} - {response.text}")
                return {"error": f"Customer info failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Autumn customer info exception: {e}")
            return {"error": str(e)}
