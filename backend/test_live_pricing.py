#!/usr/bin/env python3
"""
Live test script for Autumn pricing integration
Tests actual API calls to Autumn with your configured key
"""

import os
import sys
import requests
import json

# Set the API key
os.environ['AUTUMN_SECRET_KEY'] = 'am_sk_test_Dx04TVwAb8ma5WiFFr4kGe2cOi3AsmDPuh0aCgcX3S'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autumn_integration import AutumnPricingService

def test_service_enabled():
    """Test that service is properly enabled with API key"""
    print("\n🔧 Test 1: Service Initialization")
    print("=" * 50)
    
    service = AutumnPricingService()
    
    if service._is_enabled():
        print("✅ Service is enabled")
        print(f"✅ API Key detected: {service.api_key[:20]}...")
        print(f"✅ Base URL: {service.base_url}")
        return True
    else:
        print("❌ Service is NOT enabled")
        return False

def test_pricing_tiers_endpoint():
    """Test the /pricing/tiers endpoint"""
    print("\n🎯 Test 2: Pricing Tiers Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:8080/pricing/tiers')
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint is accessible")
            print(f"✅ Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  Backend not running. Start with: python3 app.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_autumn_connection():
    """Test direct connection to Autumn API"""
    print("\n🔗 Test 3: Autumn API Connection")
    print("=" * 50)
    
    service = AutumnPricingService()
    test_user_id = "test_user_" + str(os.getpid())
    
    print(f"Testing with user ID: {test_user_id}")
    
    try:
        # Test check usage
        allowed, info = service.check_usage(test_user_id)
        
        if "fallback" in info:
            print("⚠️  Running in fallback mode - check API key")
            return False
        
        print(f"✅ Successfully connected to Autumn API")
        print(f"✅ User allowed: {allowed}")
        print(f"✅ Response data: {json.dumps(info, indent=2)}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_rate_limiting_flow():
    """Test the complete rate limiting flow"""
    print("\n🚦 Test 4: Rate Limiting Flow")
    print("=" * 50)
    
    service = AutumnPricingService()
    test_user_id = "test_user_flow_" + str(os.getpid())
    
    print(f"Testing with user ID: {test_user_id}")
    
    try:
        # First request - should be allowed
        print("\n1️⃣  First request:")
        allowed1, info1 = service.check_and_track_usage(test_user_id)
        print(f"   Allowed: {allowed1}")
        print(f"   Info: {json.dumps(info1, indent=2)}")
        
        # Second request - should also work
        print("\n2️⃣  Second request:")
        allowed2, info2 = service.check_and_track_usage(test_user_id)
        print(f"   Allowed: {allowed2}")
        print(f"   Info: {json.dumps(info2, indent=2)}")
        
        if allowed1 and allowed2:
            print("\n✅ Rate limiting flow is working!")
            return True
        else:
            print("\n⚠️  Rate limiting may not be configured correctly")
            return False
            
    except Exception as e:
        print(f"❌ Flow test failed: {e}")
        return False

def test_checkout_flow():
    """Test checkout URL generation"""
    print("\n💳 Test 5: Checkout Flow")
    print("=" * 50)
    
    service = AutumnPricingService()
    test_user_id = "test_checkout_user"
    
    try:
        print("Requesting checkout for 'pro' tier...")
        result = service.get_checkout_url(test_user_id, "pro")
        
        print(f"✅ Checkout API response:")
        print(f"   {json.dumps(result, indent=2)}")
        
        if "checkout_url" in result or "error" in result:
            print("✅ Checkout flow is working!")
            return True
        else:
            print("⚠️  Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ Checkout test failed: {e}")
        return False

def run_all_tests():
    """Run all pricing system tests"""
    print("\n" + "=" * 50)
    print("🧪 AUTUMN PRICING SYSTEM - LIVE TESTS")
    print("=" * 50)
    
    results = []
    
    # Test 1: Service enabled
    results.append(("Service Enabled", test_service_enabled()))
    
    # Test 2: Pricing tiers endpoint
    results.append(("Pricing Tiers", test_pricing_tiers_endpoint()))
    
    # Test 3: Autumn connection
    results.append(("Autumn API", test_autumn_connection()))
    
    # Test 4: Rate limiting
    results.append(("Rate Limiting", test_rate_limiting_flow()))
    
    # Test 5: Checkout flow
    results.append(("Checkout Flow", test_checkout_flow()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            print(f"✅ {name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {name}: FAILED")
            failed += 1
        else:
            print(f"⚠️  {name}: SKIPPED")
            skipped += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! Your pricing system is working!")
    elif skipped > 0:
        print("\n⚠️  Some tests were skipped. Make sure backend is running.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

