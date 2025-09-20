#!/usr/bin/env python3
"""
Simple test to verify conversation ID consistency 
Shows you exactly what conversation IDs you should expect
"""

import hashlib

def predict_conversation_id(user_id: str, machine_id: str) -> str:
    """Predict what conversation ID should be generated"""
    conv_seed = f"conv_{user_id}_{machine_id}"
    conversation_id = hashlib.md5(conv_seed.encode()).hexdigest()[:16]
    return conversation_id

def main():
    """Show expected conversation IDs for your setup"""
    print("🔍 Conversation ID Predictor")
    print("=" * 30)
    
    # Based on your logs, your Godot setup uses:
    user_id = "guest:XP4191P2VD"  # From your logs
    machine_id = "XP4191P2VD"    # From your logs
    
    expected_conv_id = predict_conversation_id(user_id, machine_id)
    
    print(f"🆔 Your User ID: {user_id}")
    print(f"🖥️  Your Machine ID: {machine_id}")
    print(f"💬 Expected Conversation ID: {expected_conv_id}")
    print()
    print("📋 In your Supabase, ALL logs should have this conversation_id:")
    print(f"   conversation_id = '{expected_conv_id}'")
    print()
    print("🔍 Query to check YOUR conversation:")
    print("```sql")
    print(f"SELECT event_type, message_type, message_content, created_at")
    print(f"FROM llm_logs")
    print(f"WHERE conversation_id = '{expected_conv_id}'")
    print(f"ORDER BY created_at DESC;")
    print("```")
    print()
    print("✅ If all your new logs have this same conversation_id, it's working!")
    print("❌ If you see 'no_context' or other IDs, restart both servers and try again")

if __name__ == "__main__":
    main()
