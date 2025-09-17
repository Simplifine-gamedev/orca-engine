#!/usr/bin/env python3
"""Quick Langfuse integration test - run this before deployment"""
from dotenv import load_dotenv
load_dotenv()

def test_langfuse_integration():
    try:
        from langfuse import Langfuse
        import os
        
        client = Langfuse(
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'), 
            host=os.getenv('LANGFUSE_HOST', 'http://localhost:3000')
        )
        
        # Test event creation (matches backend code)
        event = client.create_event(
            name='deployment_test',
            input={'test': 'message'},
            output='test response'
        )
        client.flush()
        
        print('✅ Langfuse integration test PASSED')
        print(f'✅ Event created: {type(event)}')
        return True
        
    except Exception as e:
        print(f'❌ Langfuse integration test FAILED: {e}')
        return False

if __name__ == '__main__':
    test_langfuse_integration()
