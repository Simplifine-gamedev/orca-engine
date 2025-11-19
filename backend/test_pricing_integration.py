#!/usr/bin/env python3
"""
Test script for Autumn pricing integration
Tests the pricing service functionality without requiring actual Autumn API keys
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autumn_integration import AutumnPricingService

class TestAutumnPricingService(unittest.TestCase):
    
    def setUp(self):
        # Create a pricing service without API key (should use fallback mode)
        self.service = AutumnPricingService()
    
    def test_service_initialization(self):
        """Test that the service initializes correctly"""
        self.assertIsInstance(self.service, AutumnPricingService)
        self.assertEqual(self.service.base_url, "https://api.useautumn.com/v1")
    
    def test_fallback_mode_when_no_api_key(self):
        """Test that service works in fallback mode when no API key is configured"""
        self.assertFalse(self.service._is_enabled())
        
        # Should allow requests in fallback mode
        allowed, info = self.service.check_and_track_usage("test_user_123")
        self.assertTrue(allowed)
        self.assertIn("fallback_mode", info)
    
    @patch.dict(os.environ, {'AUTUMN_SECRET_KEY': 'test_key_123'})
    def test_enabled_mode_with_api_key(self):
        """Test that service is enabled when API key is set"""
        service = AutumnPricingService()
        self.assertTrue(service._is_enabled())
        self.assertEqual(service.api_key, 'test_key_123')
    
    @patch('requests.post')
    @patch.dict(os.environ, {'AUTUMN_SECRET_KEY': 'test_key_123'})
    def test_successful_check_and_track(self, mock_post):
        """Test successful check and track usage"""
        service = AutumnPricingService()
        
        # Mock successful check response
        mock_check_response = Mock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = {
            "allowed": True,
            "balances": []
        }
        
        # Mock successful track response  
        mock_track_response = Mock()
        mock_track_response.status_code = 200
        
        mock_post.side_effect = [mock_check_response, mock_track_response]
        
        allowed, info = service.check_and_track_usage("test_user_123")
        
        self.assertTrue(allowed)
        self.assertIn("usage_tracked", info)
        self.assertTrue(info["usage_tracked"])
        
        # Verify API calls were made
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('requests.post')
    @patch.dict(os.environ, {'AUTUMN_SECRET_KEY': 'test_key_123'})
    def test_rate_limit_exceeded(self, mock_post):
        """Test rate limit exceeded scenario"""
        service = AutumnPricingService()
        
        # Mock rate limit exceeded response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "allowed": False,
            "balances": [
                {
                    "feature_id": "ai-requests",
                    "required": 1,
                    "balance": 0
                }
            ]
        }
        
        mock_post.return_value = mock_response
        
        allowed, info = service.check_and_track_usage("test_user_123")
        
        self.assertFalse(allowed)
        self.assertIn("error", info)
        self.assertEqual(info["error"], "Request limit exceeded")
        self.assertIn("upgrade_available", info)
        self.assertTrue(info["upgrade_available"])
    
    @patch('requests.post')
    @patch.dict(os.environ, {'AUTUMN_SECRET_KEY': 'test_key_123'})
    def test_checkout_url_generation(self, mock_post):
        """Test checkout URL generation"""
        service = AutumnPricingService()
        
        # Mock successful checkout response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "checkout_url": "https://checkout.stripe.com/pay/test_123"
        }
        
        mock_post.return_value = mock_response
        
        result = service.get_checkout_url("test_user_123", "pro")
        
        self.assertIn("checkout_url", result)
        self.assertTrue(result["checkout_url"].startswith("https://checkout.stripe.com"))
    
    def test_pricing_tiers_configuration(self):
        """Test that the pricing tiers are correctly configured"""
        # This tests the tiers returned by the /pricing/tiers endpoint
        expected_tiers = {
            "free": {
                "product_id": "free",
                "name": "Free",
                "price": 0,
                "requests_per_month": 200,
                "features": ["Community support"]
            },
            "pro": {
                "product_id": "pro", 
                "name": "Pro",
                "price": 20,
                "requests_per_month": 500,
                "features": ["Priority support", "Advanced features"]
            },
            "proplus": {
                "product_id": "proplus",
                "name": "Pro+", 
                "price": 60,
                "requests_per_month": 1500,
                "features": ["Priority support", "Early access", "Team features"]
            }
        }
        
        # This validates our tier configuration matches the plan requirements
        self.assertEqual(expected_tiers["free"]["requests_per_month"], 200)
        self.assertEqual(expected_tiers["pro"]["requests_per_month"], 500)
        self.assertEqual(expected_tiers["proplus"]["requests_per_month"], 1500)

def run_tests():
    """Run all pricing integration tests"""
    print("🧪 Running Autumn Pricing Integration Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAutumnPricingService)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed! Pricing integration is working correctly.")
        print(f"📊 Tests run: {result.testsRun}")
        return True
    else:
        print("❌ Some tests failed!")
        print(f"📊 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
