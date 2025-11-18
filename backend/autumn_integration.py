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
    """Service for integrating with Autumn pricing and billing system
    
    Uses website API proxy endpoints (https://orcaengine.ai/api/autumn/*) instead of
    direct Autumn API calls for better security and centralized management.
    """
    
    def __init__(self):
        # Use website API proxy instead of direct Autumn API
        self.base_url = "https://orcaengine.ai/api/autumn"
        self.headers = {
            'Content-Type': 'application/json'
        }
        # Note: Authentication is done via customer_id in Authorization header
        # The website API handles adding the Autumn secret key server-side
    
    def _is_enabled(self) -> bool:
        """Check if Autumn integration is properly configured"""
        # Always enabled when using website API proxy
        return True
    
    def check_usage(self, user_id: str, feature_id: str = "ai-requests", required_quantity: int = 1) -> Tuple[bool, Dict]:
        """
        Check if user has access to make a request
        Uses website API proxy: https://orcaengine.ai/api/autumn/check
        Returns (allowed, usage_info)
        """
        try:
            response = requests.post(
                f"{self.base_url}/check",
                headers={
                    **self.headers,
                    'Authorization': f'Bearer {user_id}'  # Send user_id as Bearer token
                },
                json={
                    "feature_id": feature_id,
                    "required_quantity": required_quantity
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
        Track usage for a user after successful AI request
        Uses website API proxy: https://orcaengine.ai/api/autumn/track
        Returns True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/track",
                headers={
                    **self.headers,
                    'Authorization': f'Bearer {user_id}'  # Send user_id as Bearer token
                },
                json={
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
    
    def initialize_customer(self, user_id: str, user_email: str = None) -> Dict:
        """
        Initialize Autumn account for a user (creates customer + assigns Free plan if new)
        Uses website API proxy: https://orcaengine.ai/api/autumn/customer
        This is called after successful Supabase login
        Returns customer data or error dict
        """
        try:
            headers = {
                **self.headers,
                'Authorization': f'Bearer {user_id}'  # Send user_id as Bearer token
            }
            if user_email:
                headers['X-User-Email'] = user_email
            
            response = requests.get(
                f"{self.base_url}/customer",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                customer_data = response.json()
                logger.info(f"Successfully initialized Autumn account for user {user_id}")
                return customer_data
            else:
                logger.error(f"Failed to initialize customer: {response.status_code} - {response.text}")
                return {"error": f"Failed to initialize: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Autumn initialize exception: {e}")
            return {"error": str(e)}
    
    def check_quota(self, user_id: str, feature_id: str = "ai-requests", required_quantity: int = 1) -> Tuple[bool, Dict]:
        """
        Check if user has quota BEFORE making AI request
        Does NOT track usage - that should be done AFTER successful completion
        Returns (allowed, usage_info)
        """
        return self.check_usage(user_id, feature_id, required_quantity)
    
    def get_checkout_url(self, user_id: str, product_id: str) -> Dict:
        """
        Get checkout URL for product upgrade/purchase
        Uses website API proxy: https://orcaengine.ai/api/autumn/attach
        Returns checkout data with checkout_url or error
        """
        try:
            response = requests.post(
                f"{self.base_url}/attach",
                headers={
                    **self.headers,
                    'Authorization': f'Bearer {user_id}'  # Send user_id as Bearer token
                },
                json={
                    "product_id": product_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Autumn attach failed: {response.status_code} - {response.text}")
                return {"error": f"Attach failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Autumn attach exception: {e}")
            return {"error": str(e)}
    
    def get_customer_info(self, user_id: str) -> Dict:
        """
        Get customer subscription and usage information
        Uses website API proxy: https://orcaengine.ai/api/autumn/customer
        Returns customer data or error
        """
        try:
            response = requests.get(
                f"{self.base_url}/customer",
                headers={
                    **self.headers,
                    'Authorization': f'Bearer {user_id}'  # Send user_id as Bearer token
                },
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
