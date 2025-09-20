#!/usr/bin/env python3
"""
Quick conversation viewer - shows your conversation right away
"""

from logging_assistance_tools import LoggingAssistant

def main():
    """Quick view of your known conversation"""
    try:
        assistant = LoggingAssistant()
        
        # Your known conversation ID from the test
        conversation_id = "f650ea4a9c1a0676"
        
        print("🔍 Viewing your conversation:")
        assistant.get_conversation_messages(conversation_id, show_details=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
