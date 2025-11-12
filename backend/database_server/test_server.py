#!/usr/bin/env python3
"""
Test script for Godot AI Database Server

This script tests the database server without requiring Supabase credentials.
It validates the server structure and basic functionality.
"""

import os
import sys
import json
import tempfile
import logging
from unittest.mock import Mock, patch
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required dependencies can be imported"""
    try:
        import flask
        import flask_cors
        import requests
        logger.info("✅ All required dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def test_app_structure():
    """Test that the app.py file has the correct structure"""
    try:
        # Set mock environment variables
        os.environ['DEV_MODE'] = 'true'
        os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
        os.environ['SUPABASE_KEY'] = 'test-key'
        
        # Mock Supabase client creation to avoid requiring real credentials
        with patch('supabase.create_client') as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client
            
            # Import the app (this will execute the module)
            sys.path.insert(0, os.path.dirname(__file__))
            import app
            
            # Check that Flask app was created
            assert hasattr(app, 'app'), "Flask app not created"
            assert app.app is not None, "Flask app is None"
            
            logger.info("✅ App structure is correct")
            return True
            
    except Exception as e:
        logger.error(f"❌ App structure test failed: {e}")
        return False

def test_endpoints_registered():
    """Test that all expected endpoints are registered"""
    try:
        # Set mock environment variables
        os.environ['DEV_MODE'] = 'true'
        os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
        os.environ['SUPABASE_KEY'] = 'test-key'
        
        with patch('supabase.create_client') as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client
            
            # Import the app
            sys.path.insert(0, os.path.dirname(__file__))
            import app
            
            # Get all registered routes
            routes = []
            for rule in app.app.url_map.iter_rules():
                routes.append({
                    'endpoint': rule.endpoint,
                    'methods': list(rule.methods - {'HEAD', 'OPTIONS'}),
                    'rule': rule.rule
                })
            
            # Expected endpoints
            expected_endpoints = [
                '/',
                '/health',
                '/stats', 
                '/models/<user_id>',
                '/models/<user_id>/<model_id>',
                '/models/<user_id>/search',
                '/models/recent',
                '/download/<user_id>/<model_id>/<file_type>'
            ]
            
            registered_rules = [route['rule'] for route in routes]
            
            missing_endpoints = []
            for endpoint in expected_endpoints:
                if endpoint not in registered_rules:
                    missing_endpoints.append(endpoint)
            
            if missing_endpoints:
                logger.error(f"❌ Missing endpoints: {missing_endpoints}")
                return False
            
            logger.info("✅ All expected endpoints are registered")
            logger.info(f"   Registered routes: {len(routes)}")
            for route in routes:
                logger.info(f"   {route['methods']} {route['rule']}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Endpoint registration test failed: {e}")
        return False

def test_validation_functions():
    """Test input validation functions"""
    try:
        # Set mock environment variables
        os.environ['DEV_MODE'] = 'true'
        os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
        os.environ['SUPABASE_KEY'] = 'test-key'
        
        with patch('supabase.create_client'):
            sys.path.insert(0, os.path.dirname(__file__))
            import app
            
            # Test user_id validation
            assert app.validate_user_id('godot_user123') == True
            assert app.validate_user_id('test-user_456') == True
            assert app.validate_user_id('') == False
            assert app.validate_user_id('a') == False  # too short
            assert app.validate_user_id('a' * 101) == False  # too long
            assert app.validate_user_id('user@domain.com') == False  # invalid chars
            
            # Test model_id validation (UUID format)
            assert app.validate_model_id('e88c0a41-f202-4f5f-a21c-6c191b27a837') == True
            assert app.validate_model_id('invalid-uuid') == False
            assert app.validate_model_id('') == False
            assert app.validate_model_id('e88c0a41-f202-4f5f-a21c-6c191b27a837x') == False  # too long
            
            logger.info("✅ Validation functions work correctly")
            return True
            
    except Exception as e:
        logger.error(f"❌ Validation function test failed: {e}")
        return False

def test_cache_functions():
    """Test file caching functions"""
    try:
        # Set mock environment variables
        os.environ['DEV_MODE'] = 'true'
        os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
        os.environ['SUPABASE_KEY'] = 'test-key'
        
        with patch('supabase.create_client'):
            sys.path.insert(0, os.path.dirname(__file__))
            import app
            
            # Clear cache first
            app.file_cache.clear()
            
            # Test cache operations
            test_data = b'test file data'
            cache_key = app.get_cache_key('user123', 'model456', 'obj')
            
            # Cache should be empty initially
            assert app.get_cached_file(cache_key) is None
            
            # Cache the data
            app.cache_file(cache_key, test_data)
            
            # Should be able to retrieve it
            cached_data = app.get_cached_file(cache_key)
            assert cached_data == test_data
            
            logger.info("✅ Cache functions work correctly")
            return True
            
    except Exception as e:
        logger.error(f"❌ Cache function test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and return overall result"""
    logger.info("🧪 Starting Godot AI Database Server tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("App Structure Test", test_app_structure), 
        ("Endpoint Registration Test", test_endpoints_registered),
        ("Validation Function Test", test_validation_functions),
        ("Cache Function Test", test_cache_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📝 Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Database server is ready for deployment.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
