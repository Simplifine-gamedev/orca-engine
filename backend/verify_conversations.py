#!/usr/bin/env python3
"""
Verify conversation ID consistency and provide debugging queries
"""

import os
import requests
import json
from dotenv import load_dotenv

def test_conversation_consistency():
    """Test that conversation IDs are consistent within conversations"""
    load_dotenv()
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase config")
        return
    
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Content-Type': 'application/json'
    }
    
    print("🔍 Checking conversation ID consistency...")
    
    try:
        # Get recent logs grouped by request_id
        response = requests.get(
            f"{supabase_url}/rest/v1/llm_logs?select=request_id,conversation_id,event_type,message_type,message_content&order=created_at.desc&limit=50",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        logs = response.json()
        
        if not logs:
            print("📭 No logs found")
            return
        
        # Group by request_id to check consistency
        requests_data = {}
        for log in logs:
            req_id = log['request_id']
            if req_id not in requests_data:
                requests_data[req_id] = []
            requests_data[req_id].append(log)
        
        print(f"\n📊 Found {len(logs)} logs across {len(requests_data)} requests")
        
        # Check each request for consistency
        inconsistent_requests = []
        for req_id, req_logs in requests_data.items():
            conversation_ids = set(log['conversation_id'] for log in req_logs if log['conversation_id'])
            
            if len(conversation_ids) > 1:
                inconsistent_requests.append((req_id, conversation_ids, req_logs))
                print(f"❌ Request {req_id}: {len(conversation_ids)} different conversation IDs: {conversation_ids}")
            elif len(conversation_ids) == 1:
                conv_id = list(conversation_ids)[0]
                print(f"✅ Request {req_id}: Consistent conversation ID: {conv_id}")
            else:
                print(f"⚠️  Request {req_id}: No conversation ID set")
        
        if not inconsistent_requests:
            print("\n🎉 All requests have consistent conversation IDs!")
        else:
            print(f"\n⚠️  Found {len(inconsistent_requests)} requests with inconsistent conversation IDs")
        
        # Show how to query complete conversations
        print("\n📋 To get complete conversations, use these queries:")
        print()
        
        unique_conversations = set()
        for log in logs:
            if log['conversation_id']:
                unique_conversations.add(log['conversation_id'])
        
        if unique_conversations:
            sample_conv = list(unique_conversations)[0]
            print(f"-- Get complete conversation (example: {sample_conv})")
            print(f"SELECT conversation_id, message_type, message_content, created_at, event_type")
            print(f"FROM llm_logs ")
            print(f"WHERE conversation_id = '{sample_conv}'")
            print(f"ORDER BY created_at;")
            print()
            
            print("-- Get all recent conversations:")
            print("SELECT DISTINCT conversation_id, COUNT(*) as log_count, MIN(created_at) as started")
            print("FROM llm_logs ")
            print("WHERE conversation_id IS NOT NULL")
            print("GROUP BY conversation_id")
            print("ORDER BY started DESC;")
        
        return len(inconsistent_requests) == 0
        
    except Exception as e:
        print(f"❌ Error checking conversations: {e}")
        return False

def show_recent_conversations():
    """Show recent conversations in a readable format"""
    load_dotenv()
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Get conversations with summary
        response = requests.get(
            f"{supabase_url}/rest/v1/rpc/conversation_summary", 
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 404:
            # Function doesn't exist, create a simple query
            print("📋 Recent conversations:")
            response = requests.get(
                f"{supabase_url}/rest/v1/llm_logs?select=conversation_id,message_type,message_content,created_at&order=created_at.desc&limit=20",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            logs = response.json()
            
            current_conv = None
            for log in logs:
                conv_id = log['conversation_id']
                if conv_id != current_conv:
                    print(f"\n🗨️  Conversation: {conv_id}")
                    current_conv = conv_id
                
                msg_type = log['message_type'] or 'unknown'
                content = log['message_content'] or ''
                time_str = log['created_at'][:19]  # Remove timezone for readability
                
                emoji = {'user_input': '👤', 'assistant_output': '🤖', 'tool_call': '🔧', 'tool_result': '📋'}.get(msg_type, '❓')
                print(f"  {emoji} {time_str} | {msg_type}: {content[:60]}...")
        
    except Exception as e:
        print(f"❌ Error showing conversations: {e}")

if __name__ == "__main__":
    print("🔍 Conversation ID Verification Tool")
    print("=" * 35)
    
    # Test consistency
    consistent = test_conversation_consistency()
    
    if not consistent:
        print("\n🔧 FIXING: Restart your app.py and logging server with the updated code")
        print("   Then send a new message to test the fix")
    
    # Show conversations
    print("\n" + "=" * 35)
    show_recent_conversations()
