#!/usr/bin/env python3
"""
Test conversation ID consistency after fixes
"""

import os
import requests
import time
from dotenv import load_dotenv

def test_conversation_id_stability():
    """Test that conversation IDs are now stable"""
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
    
    print("🔍 Testing conversation ID after fixes...")
    print("=" * 40)
    
    try:
        # Get the most recent logs to see if consistency improved
        response = requests.get(
            f"{supabase_url}/rest/v1/llm_logs?select=request_id,conversation_id,event_type,message_type,created_at&order=created_at.desc&limit=20",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        logs = response.json()
        
        if not logs:
            print("📭 No recent logs found")
            return
        
        print(f"📊 Checking last {len(logs)} logs...")
        
        # Group by request to check consistency
        request_groups = {}
        for log in logs:
            req_id = log['request_id']
            if req_id not in request_groups:
                request_groups[req_id] = []
            request_groups[req_id].append(log)
        
        consistent_requests = 0
        total_requests = len(request_groups)
        
        for req_id, req_logs in list(request_groups.items())[:5]:  # Check first 5
            conv_ids = set(log['conversation_id'] for log in req_logs if log['conversation_id'])
            
            if len(conv_ids) == 1:
                conv_id = list(conv_ids)[0]
                print(f"✅ Request {req_id[:8]}...: Consistent ID {conv_id}")
                consistent_requests += 1
            elif len(conv_ids) > 1:
                print(f"❌ Request {req_id[:8]}...: {len(conv_ids)} different IDs: {list(conv_ids)}")
            else:
                print(f"⚠️  Request {req_id[:8]}...: No conversation ID")
        
        print(f"\n📊 Consistency: {consistent_requests}/{total_requests} requests have consistent conversation IDs")
        
        if consistent_requests == total_requests:
            print("🎉 Perfect! All requests now have consistent conversation IDs!")
        else:
            print("⚠️  Some inconsistency remains - may need to restart both servers")
        
        # Show sample query for getting complete conversations
        if logs and logs[0]['conversation_id']:
            sample_conv = logs[0]['conversation_id']
            print(f"\n📋 To get complete conversation {sample_conv}:")
            print(f"```sql")
            print(f"SELECT event_type, message_type, message_content, created_at")
            print(f"FROM llm_logs")
            print(f"WHERE conversation_id = '{sample_conv}'")
            print(f"ORDER BY created_at;")
            print(f"```")
        
        return consistent_requests == total_requests
        
    except Exception as e:
        print(f"❌ Error checking conversations: {e}")
        return False

if __name__ == "__main__":
    test_conversation_id_stability()
