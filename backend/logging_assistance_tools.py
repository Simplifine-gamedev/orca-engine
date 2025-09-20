#!/usr/bin/env python3
"""
Logging Assistance Tools for LiteLLM Supabase Logs
© 2025 Simplifine Corp. Tools for analyzing and viewing logged conversations.
"""

import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

class LoggingAssistant:
    def __init__(self):
        load_dotenv()
        
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.table_name = os.getenv('SUPABASE_TABLE_NAME', 'llm_logs')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("❌ Missing Supabase configuration in .env")
        
        self.headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json'
        }
        
        print(f"✅ Connected to Supabase: {self.supabase_url}")
    
    def get_conversation_messages(self, conversation_id: str, show_details: bool = False):
        """Get and display all messages for a conversation ID"""
        print(f"🗨️  Fetching conversation: {conversation_id}")
        print("=" * 50)
        
        try:
            # Query logs for this conversation
            response = requests.get(
                f"{self.supabase_url}/rest/v1/{self.table_name}",
                params={
                    'conversation_id': f'eq.{conversation_id}',
                    'select': 'created_at,event_type,message_type,message_content,model,duration_ms,tokens_total,cost_usd,success,error_message,session_id,user_agent',
                    'order': 'created_at.asc'
                },
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            logs = response.json()
            
            if not logs:
                print(f"📭 No messages found for conversation ID: {conversation_id}")
                return
            
            print(f"📊 Found {len(logs)} log entries")
            print()
            
            # Group by message type for cleaner display
            conversation_flow = []
            current_exchange = None
            
            for log in logs:
                timestamp = datetime.fromisoformat(log['created_at'].replace('Z', '+00:00'))
                time_str = timestamp.strftime("%H:%M:%S")
                
                msg_type = log['message_type']
                content = log['message_content'] or ''
                event_type = log['event_type']
                
                # Skip technical events unless showing details
                if not show_details and event_type in ['pre_call', 'post_call']:
                    continue
                
                # Format message display
                if msg_type == 'user_input':
                    emoji = "👤"
                    label = "USER"
                    text_color = "\033[94m"  # Blue
                elif msg_type == 'assistant_output':
                    emoji = "🤖"
                    label = "ASSISTANT"
                    text_color = "\033[92m"  # Green
                elif msg_type == 'tool_call':
                    emoji = "🔧"
                    label = "TOOL CALL"
                    text_color = "\033[93m"  # Yellow
                elif msg_type == 'tool_result':
                    emoji = "📋"
                    label = "TOOL RESULT"
                    text_color = "\033[95m"  # Magenta
                elif event_type == 'tool_call':
                    emoji = "🔧"
                    label = "TOOL CALL"
                    text_color = "\033[93m"  # Yellow
                elif event_type == 'tool_result':
                    emoji = "📋"
                    label = "TOOL RESULT"
                    text_color = "\033[95m"  # Magenta
                else:
                    emoji = "❓"
                    label = event_type.upper()
                    text_color = "\033[90m"  # Gray
                
                reset_color = "\033[0m"
                
                # Print message
                print(f"{emoji} {text_color}{label}{reset_color} [{time_str}]")
                
                # Print content with proper formatting
                if content:
                    # Clean and format content
                    display_content = content.strip()
                    if len(display_content) > 200:
                        display_content = display_content[:200] + "..."
                    
                    # Indent content
                    lines = display_content.split('\n')
                    for line in lines:
                        print(f"   {line}")
                else:
                    print(f"   [No content]")
                
                # Show performance metrics if available
                if show_details:
                    metrics = []
                    if log['duration_ms']:
                        metrics.append(f"{log['duration_ms']}ms")
                    if log['tokens_total']:
                        metrics.append(f"{log['tokens_total']} tokens")
                    if log['cost_usd'] and float(log['cost_usd']) > 0:
                        metrics.append(f"${float(log['cost_usd']):.4f}")
                    if log['model']:
                        metrics.append(f"Model: {log['model']}")
                    
                    if metrics:
                        print(f"   \033[90m({' | '.join(metrics)}){reset_color}")
                    
                    if not log['success'] and log['error_message']:
                        print(f"   \033[91m❌ Error: {log['error_message'][:100]}{reset_color}")
                
                print()  # Empty line between messages
            
            # Summary
            print("=" * 50)
            total_cost = sum(float(log['cost_usd'] or 0) for log in logs)
            total_tokens = sum(int(log['tokens_total'] or 0) for log in logs)
            
            user_msgs = sum(1 for log in logs if log['message_type'] == 'user_input')
            assistant_msgs = sum(1 for log in logs if log['message_type'] == 'assistant_output')
            tool_calls = sum(1 for log in logs if log['message_type'] == 'tool_call' or log['event_type'] == 'tool_call')
            tool_results = sum(1 for log in logs if log['message_type'] == 'tool_result' or log['event_type'] == 'tool_result')
            
            print(f"📊 Conversation Summary:")
            print(f"   💬 Messages: {user_msgs} user → {assistant_msgs} assistant")
            if tool_calls > 0 or tool_results > 0:
                print(f"   🔧 Tools: {tool_calls} calls → {tool_results} results")
            print(f"   🔢 Total tokens: {total_tokens}")
            print(f"   💰 Total cost: ${total_cost:.4f}")
            print(f"   🆔 Conversation ID: {conversation_id}")
            
        except Exception as e:
            print(f"❌ Error fetching conversation: {e}")
    
    def list_recent_conversations(self, limit: int = 10):
        """List recent conversations with summary"""
        print(f"📋 Recent Conversations (last {limit})")
        print("=" * 40)
        
        try:
            # Get distinct conversation IDs with summary info
            response = requests.get(
                f"{self.supabase_url}/rest/v1/{self.table_name}",
                params={
                    'select': 'conversation_id,created_at,message_type,message_content,cost_usd,tokens_total',
                    'conversation_id': 'not.is.null',
                    'order': 'created_at.desc',
                    'limit': limit * 10  # Get more to group properly
                },
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            logs = response.json()
            
            if not logs:
                print("📭 No conversations found")
                return []
            
            # Group by conversation_id
            conversations = {}
            for log in logs:
                conv_id = log['conversation_id']
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        'id': conv_id,
                        'messages': [],
                        'first_seen': log['created_at'],
                        'last_seen': log['created_at'],
                        'total_cost': 0,
                        'total_tokens': 0
                    }
                
                conversations[conv_id]['messages'].append(log)
                conversations[conv_id]['last_seen'] = max(conversations[conv_id]['last_seen'], log['created_at'])
                conversations[conv_id]['total_cost'] += float(log['cost_usd'] or 0)
                conversations[conv_id]['total_tokens'] += int(log['tokens_total'] or 0)
            
            # Sort by most recent activity
            sorted_conversations = sorted(conversations.values(), key=lambda x: x['last_seen'], reverse=True)
            
            # Display conversations
            for i, conv in enumerate(sorted_conversations[:limit]):
                conv_id = conv['id']
                last_time = datetime.fromisoformat(conv['last_seen'].replace('Z', '+00:00'))
                time_str = last_time.strftime("%Y-%m-%d %H:%M")
                
                # Get first user message for preview
                first_user_msg = None
                last_assistant_msg = None
                
                for msg in conv['messages']:
                    if msg['message_type'] == 'user_input' and not first_user_msg:
                        first_user_msg = msg['message_content']
                    elif msg['message_type'] == 'assistant_output':
                        last_assistant_msg = msg['message_content']
                
                user_preview = (first_user_msg or "No user message")[:50] + "..." if first_user_msg and len(first_user_msg) > 50 else (first_user_msg or "No user message")
                assistant_preview = (last_assistant_msg or "No response")[:50] + "..." if last_assistant_msg and len(last_assistant_msg) > 50 else (last_assistant_msg or "No response")
                
                print(f"{i+1:2d}. 🗨️  {conv_id} [{time_str}]")
                print(f"    👤 {user_preview}")
                print(f"    🤖 {assistant_preview}")
                print(f"    💰 ${conv['total_cost']:.4f} | 🔢 {conv['total_tokens']} tokens | 📨 {len(conv['messages'])} logs")
                print()
            
            return [conv['id'] for conv in sorted_conversations[:limit]]
            
        except Exception as e:
            print(f"❌ Error listing conversations: {e}")
            return []

def main():
    """Interactive conversation viewer"""
    print("🔍 LiteLLM Conversation Viewer")
    print("=" * 35)
    
    try:
        assistant = LoggingAssistant()
        
        while True:
            print("\n📋 Options:")
            print("1. View specific conversation")
            print("2. List recent conversations") 
            print("3. Exit")
            
            choice = input("\n🎯 Choose (1-3): ").strip()
            
            if choice == '1':
                conv_id = input("🆔 Enter conversation ID: ").strip()
                if conv_id:
                    show_details = input("📊 Show detailed metrics? (y/n): ").lower().strip() == 'y'
                    assistant.get_conversation_messages(conv_id, show_details)
            
            elif choice == '2':
                try:
                    limit = int(input("📋 How many conversations? (default 10): ").strip() or "10")
                except ValueError:
                    limit = 10
                
                conversations = assistant.list_recent_conversations(limit)
                
                if conversations:
                    view_choice = input("\n🎯 Enter number to view conversation (or Enter to continue): ").strip()
                    try:
                        if view_choice and view_choice.isdigit():
                            idx = int(view_choice) - 1
                            if 0 <= idx < len(conversations):
                                conv_id = conversations[idx]
                                show_details = input("📊 Show detailed metrics? (y/n): ").lower().strip() == 'y'
                                assistant.get_conversation_messages(conv_id, show_details)
                    except (ValueError, IndexError):
                        print("❌ Invalid selection")
            
            elif choice == '3':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Viewer interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
