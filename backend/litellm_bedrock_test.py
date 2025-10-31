from litellm import completion
import os
import dotenv

dotenv.load_dotenv()

# Test different Claude model formats for AWS Bedrock
models_to_test = [
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",  # Latest stable Sonnet 3.5
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",  # Previous Sonnet 3.5
    "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # With region prefix
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",    # Sonnet 3.0 fallback
]

print("Testing AWS Bedrock Claude models with LiteLLM...")
print(f"AWS Region: {os.getenv('AWS_DEFAULT_REGION', 'not set')}")
print(f"AWS Access Key: {'set' if os.getenv('AWS_ACCESS_KEY_ID') else 'not set'}")

for model in models_to_test:
    print(f"\nTesting model: {model}")
    try:
        response = completion(
            model=model,
            messages=[{"content": "Hello, what Claude model are you? Response in 10 words max.", "role": "user"}],
            max_tokens=50,
            # LiteLLM will use standard AWS env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        )
        print(f"✅ SUCCESS: {model}")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        break  # Exit on first success
    except Exception as e:
        print(f"❌ FAILED: {model}")
        print(f"Error: {str(e)[:200]}...")
        continue

print("\nTest completed!")