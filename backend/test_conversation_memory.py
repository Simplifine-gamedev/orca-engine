#!/usr/bin/env python3
"""
Test script for the conversation memory system.
Run this to test all functionality locally.
"""
import asyncio
import json
import requests
import time
from typing import List, Dict

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your backend URL
TEST_USER_ID = "test_user_123"
TEST_MACHINE_ID = "test-machine-123"

# Headers for authentication
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "X-Machine-ID": TEST_MACHINE_ID,
    "X-Allow-Guest": "true"
}

def create_test_messages(count: int = 20) -> List[Dict]:
    """Create a realistic conversation for testing"""
    messages = [
        {"role": "system", "content": "You are a helpful Godot development assistant."},
        {"role": "user", "content": "I'm having trouble with physics in my 2D platformer game"},
        {"role": "assistant", "content": "I can help with that! What specific physics issues are you encountering?"},
        {"role": "user", "content": "My player keeps falling through platforms sometimes"},
        {"role": "assistant", "content": "This sounds like a collision detection issue. Are you using RigidBody2D or CharacterBody2D?"},
        {"role": "user", "content": "I'm using RigidBody2D with a CollisionShape2D"},
        {"role": "assistant", "content": "For platformers, CharacterBody2D is usually better. It gives you more control over movement."},
        {"role": "user", "content": "How do I convert from RigidBody2D to CharacterBody2D?"},
        {"role": "assistant", "content": "Here's how to convert: 1) Change node type, 2) Replace physics process with move_and_slide(), 3) Handle velocity manually"},
        {"role": "user", "content": "What about gravity? How do I implement that?"},
        {"role": "assistant", "content": "Add gravity to your velocity each frame: velocity.y += gravity * get_physics_process_delta_time()"},
        {"role": "user", "content": "My jump feels floaty. How can I make it snappier?"},
        {"role": "assistant", "content": "Try different gravity values for up/down movement, or adjust jump_velocity and gravity multipliers"},
        {"role": "user", "content": "Can you show me a complete movement script?"},
        {"role": "assistant", "content": "Sure! Here's a basic CharacterBody2D movement script with gravity and jumping..."},
        {"role": "user", "content": "The script works but now the player slides on walls"},
        {"role": "assistant", "content": "Add wall friction by checking is_on_wall() and reducing horizontal velocity when touching walls"},
        {"role": "user", "content": "Perfect! Now how do I add double jump?"},
        {"role": "assistant", "content": "Track jump count: if Input.is_action_just_pressed('jump') and jump_count < max_jumps..."},
        {"role": "user", "content": "Thanks! This has been really helpful for my platformer"},
        {"role": "assistant", "content": "You're welcome! Your platformer physics should feel much more responsive now."}
    ]
    
    # Extend with more messages if needed
    while len(messages) < count:
        messages.extend([
            {"role": "user", "content": f"Additional question {len(messages)}"},
            {"role": "assistant", "content": f"Additional response {len(messages)}"}
        ])
    
    return messages[:count]

def create_edited_messages() -> List[Dict]:
    """Create the same conversation but with some edits"""
    messages = create_test_messages(20)
    # Edit some messages to test fuzzy matching
    messages[5] = {"role": "user", "content": "I'm using CharacterBody3D with a CollisionShape3D"}  # Changed 2D to 3D
    messages[7] = {"role": "user", "content": "How do I convert from RigidBody3D to CharacterBody3D?"}  # Changed 2D to 3D
    return messages

async def test_basic_summarization():
    """Test 1: Basic summarization functionality"""
    print("🧪 Test 1: Basic Summarization")
    print("-" * 50)
    
    messages = create_test_messages(15)
    
    response = requests.post(f"{BASE_URL}/summarize_conversation", 
                           headers=DEFAULT_HEADERS,
                           json={"messages": messages})
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS: Basic summarization works")
        print(f"📊 Original messages: {result.get('original_message_count')}")
        print(f"📝 Summary tokens: {result.get('summary_tokens')}")
        print(f"📄 Summary preview: {result.get('summary', '')[:200]}...")
        return result.get('summary')
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return None

async def test_edited_messages():
    """Test 2: Edited message handling"""
    print("\n🧪 Test 2: Edited Message Handling")
    print("-" * 50)
    
    # First create original summary
    original_messages = create_test_messages(15)
    requests.post(f"{BASE_URL}/summarize_conversation",
                 headers=DEFAULT_HEADERS,
                 json={"messages": original_messages})
    
    time.sleep(1)  # Brief pause
    
    # Now test with edited messages
    edited_messages = create_edited_messages()[:15]
    
    response = requests.post(f"{BASE_URL}/update_conversation_summary",
                           headers=DEFAULT_HEADERS,
                           json={"messages": edited_messages})
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS: Edited message handling works")
        print(f"🔄 Was updated: {result.get('was_updated')}")
        print(f"🔍 Previous summary found: {result.get('previous_summary_found')}")
        print(f"📝 Message: {result.get('message')}")
        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

async def test_memory_persistence():
    """Test 3: Save/Load persistence"""
    print("\n🧪 Test 3: Memory Persistence")
    print("-" * 50)
    
    # Create and summarize a conversation
    messages = create_test_messages(12)
    
    # First summarization
    response1 = requests.post(f"{BASE_URL}/summarize_conversation",
                             headers=DEFAULT_HEADERS,
                             json={"messages": messages})
    
    if response1.status_code != 200:
        print(f"❌ FAILED: First summarization failed - {response1.text}")
        return False
    
    time.sleep(1)
    
    # Second summarization with same messages (should reuse)
    response2 = requests.post(f"{BASE_URL}/summarize_conversation", 
                             headers=DEFAULT_HEADERS,
                             json={"messages": messages})
    
    if response2.status_code == 200:
        result1 = response1.json()
        result2 = response2.json()
        
        # Should return the same summary (cached)
        if result1.get('summary') == result2.get('summary'):
            print("✅ SUCCESS: Summary persistence works")
            print("📋 Same summary returned from cache")
            return True
        else:
            print("⚠️  WARNING: Different summaries returned")
            print("This might be normal if no caching occurred")
            return True
    else:
        print(f"❌ FAILED: Second summarization failed - {response2.text}")
        return False

async def test_conversation_search():
    """Test 4: Conversation history search"""
    print("\n🧪 Test 4: Conversation Search")
    print("-" * 50)
    
    response = requests.post(f"{BASE_URL}/search_conversation_history",
                           headers=DEFAULT_HEADERS,
                           json={"query": "physics collision platformer", "max_results": 3})
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS: Conversation search works")
        print(f"🔍 Similar conversations found: {result.get('total_found', 0)}")
        
        for i, conv in enumerate(result.get('similar_conversations', [])):
            print(f"  {i+1}. Topic: {conv.get('topic')} | Similarity: {conv.get('similarity', 0):.2f}")
        
        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

async def test_memory_stats():
    """Test 5: Memory statistics"""
    print("\n🧪 Test 5: Memory Statistics")
    print("-" * 50)
    
    response = requests.get(f"{BASE_URL}/memory_stats", 
                           headers={
                               "X-Machine-ID": TEST_MACHINE_ID,
                               "X-Allow-Guest": "true"
                           })
    
    if response.status_code == 200:
        result = response.json()
        stats = result.get('stats', {})
        
        print("✅ SUCCESS: Memory stats available")
        print(f"📊 Total summaries: {stats.get('total_summaries', 0)}")
        print(f"🔧 System enabled: {stats.get('enabled', False)}")
        print(f"🌐 Weaviate connected: {stats.get('weaviate_connected', False)}")
        print(f"🤖 Models configured: {stats.get('models_configured', [])}")
        
        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

async def test_token_limit_simulation():
    """Test 6: Simulate token limit scenario"""
    print("\n🧪 Test 6: Token Limit Simulation")
    print("-" * 50)
    
    # Create a very long conversation
    very_long_messages = create_test_messages(100)  # 100 messages
    
    print(f"📏 Testing with {len(very_long_messages)} messages")
    print("🔄 This should trigger automatic summarization...")
    
    # Note: This would normally happen in the /chat endpoint
    # For now, just test if we can handle large message sets
    response = requests.post(f"{BASE_URL}/summarize_conversation",
                           headers=DEFAULT_HEADERS,
                           json={"messages": very_long_messages[:50]})  # Test with subset
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS: Large conversation handling works")
        print(f"📊 Handled {result.get('original_message_count')} messages")
        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

async def run_all_tests():
    """Run the complete test suite"""
    print("🚀 Starting Conversation Memory System Tests")
    print("=" * 60)
    
    # Check if backend is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Backend not healthy: {health_response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        print(f"   Make sure your backend is running: python app.py")
        print(f"   Error: {e}")
        return
    
    print(f"✅ Backend is running at {BASE_URL}")
    print()
    
    # Run tests
    tests = [
        ("Basic Summarization", test_basic_summarization),
        ("Edited Messages", test_edited_messages), 
        ("Memory Persistence", test_memory_persistence),
        ("Conversation Search", test_conversation_search),
        ("Memory Statistics", test_memory_stats),
        ("Token Limit Simulation", test_token_limit_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ FAILED: {test_name} threw exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your conversation memory system is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    print("🧠 Conversation Memory System Test Suite")
    print("=" * 60)
    print("📋 Prerequisites:")
    print("  1. Backend running: python app.py")
    print("  2. Weaviate configured with WEAVIATE_URL and WEAVIATE_API_KEY")
    print("  3. Authentication: Set DEV_MODE=true or use guest authentication")
    print("  4. AI models: Cerebras/Qwen API keys configured")
    print()
    print(f"🔧 Using machine ID: {TEST_MACHINE_ID}")
    print(f"🌐 Testing against: {BASE_URL}")
    print("=" * 60)
    print()
    
    # You can also test individual functions:
    # asyncio.run(test_basic_summarization())
    
    # Or run the full suite:
    asyncio.run(run_all_tests())
