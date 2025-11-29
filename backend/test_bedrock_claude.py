"""
Test script for AWS Bedrock Claude models using LiteLLM.
This script tests Bedrock integration before updating app.py to use it as default.
"""
from litellm import completion
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Debug: Print what we're reading from env
print("🔍 Debugging environment variables:")
aws_access_key_raw = os.getenv('AWS_ACCESS_KEY')
aws_access_key_id_raw = os.getenv('AWS_ACCESS_KEY_ID')
aws_bedrock_api_key_raw = os.getenv('AWS_BEDROCK_API_KEY')

print(f"   AWS_ACCESS_KEY: {aws_access_key_raw[:20] + '...' if aws_access_key_raw and len(aws_access_key_raw) > 20 else (aws_access_key_raw or 'NOT SET')}")
print(f"   AWS_ACCESS_KEY_ID: {aws_access_key_id_raw[:20] + '...' if aws_access_key_id_raw and len(aws_access_key_id_raw) > 20 else (aws_access_key_id_raw or 'NOT SET')}")
print(f"   AWS_BEDROCK_API_KEY: {'SET (length: ' + str(len(aws_bedrock_api_key_raw)) + ')' if aws_bedrock_api_key_raw else 'NOT SET'}")

# AWS Bedrock configuration
# Option 1: Bearer token authentication (AWS_BEDROCK_API_KEY) - preferred
AWS_BEDROCK_API_KEY = os.getenv('AWS_BEDROCK_API_KEY')

# Option 2: Standard AWS credentials (AWS_ACCESS_KEY or AWS_ACCESS_KEY_ID)
AWS_ACCESS_KEY_ID = (
    os.getenv('AWS_ACCESS_KEY') or 
    os.getenv('AWS_ACCESS_KEY_ID')
)
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION') or 'us-east-1'

# Debug: Show which authentication method will be used
if AWS_BEDROCK_API_KEY:
    print(f"   ✅ Will use: AWS_BEDROCK_API_KEY (bearer token authentication)")
elif os.getenv('AWS_ACCESS_KEY'):
    print(f"   ✅ Will use: AWS_ACCESS_KEY (boto3 authentication)")
elif os.getenv('AWS_ACCESS_KEY_ID'):
    print(f"   ✅ Will use: AWS_ACCESS_KEY_ID (boto3 authentication)")
else:
    print(f"   ❌ No AWS credentials found!")

print("=" * 60)
print("AWS BEDROCK CLAUDE TEST")
print("=" * 60)

# Check if we have either bearer token OR AWS credentials
if not AWS_BEDROCK_API_KEY and not AWS_ACCESS_KEY_ID:
    print("❌ ERROR: No AWS credentials found in .env file")
    print("   Please set one of the following in your .env file:")
    print("   - AWS_BEDROCK_API_KEY=... (bearer token - preferred)")
    print("   - AWS_ACCESS_KEY=... (AWS access key ID)")
    print("   - AWS_ACCESS_KEY_ID=... (AWS access key ID)")
    exit(1)

if not AWS_SECRET_ACCESS_KEY:
    print("⚠️  WARNING: AWS_SECRET_ACCESS_KEY not set")
    print("   LiteLLM will try to use default AWS credentials (e.g., ~/.aws/credentials)")
    print("   If this fails, set AWS_SECRET_ACCESS_KEY in your .env file")
else:
    # Clean secret key (remove quotes, whitespace)
    AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY.strip().strip('"\'')
    if len(AWS_SECRET_ACCESS_KEY) != 40:
        print(f"\n⚠️  WARNING: Secret key length is {len(AWS_SECRET_ACCESS_KEY)}, expected 40 characters")
        print(f"   Secret key preview: {AWS_SECRET_ACCESS_KEY[:4]}...")

# Set AWS credentials for LiteLLM
# Option 1: Use bearer token authentication (AWS_BEARER_TOKEN_BEDROCK) - preferred for API key
if AWS_BEDROCK_API_KEY:
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = AWS_BEDROCK_API_KEY.strip().strip('"\'')
    print(f"   ✅ Using AWS_BEARER_TOKEN_BEDROCK for authentication (API key method)")

# Option 2: Use standard AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
if AWS_ACCESS_KEY_ID:
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    if AWS_SECRET_ACCESS_KEY:
        os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    if not AWS_BEDROCK_API_KEY:
        print(f"   ✅ Using AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY for authentication (boto3 method)")

os.environ['AWS_DEFAULT_REGION'] = AWS_DEFAULT_REGION
os.environ['AWS_REGION_NAME'] = AWS_DEFAULT_REGION  # LiteLLM also checks this

print(f"\n✅ Configuration loaded:")
print(f"   AWS_ACCESS_KEY_ID: {'set' if AWS_ACCESS_KEY_ID else 'not set'}")
if AWS_ACCESS_KEY_ID:
    # Show first 8 chars for verification (AWS access keys start with specific prefixes)
    key_preview = AWS_ACCESS_KEY_ID[:8] + "..." if len(AWS_ACCESS_KEY_ID) > 8 else AWS_ACCESS_KEY_ID
    print(f"   Access Key Preview: {key_preview} (length: {len(AWS_ACCESS_KEY_ID)})")
print(f"   AWS_SECRET_ACCESS_KEY: {'set' if AWS_SECRET_ACCESS_KEY else 'not set (using default AWS credentials)'}")
if AWS_SECRET_ACCESS_KEY:
    secret_preview = AWS_SECRET_ACCESS_KEY[:4] + "..." if len(AWS_SECRET_ACCESS_KEY) > 4 else "***"
    print(f"   Secret Key Preview: {secret_preview} (length: {len(AWS_SECRET_ACCESS_KEY)})")
print(f"   AWS_DEFAULT_REGION: {AWS_DEFAULT_REGION}")

# Verify credentials format
if AWS_ACCESS_KEY_ID:
    # Check for common issues
    key_clean = AWS_ACCESS_KEY_ID.strip()
    if len(key_clean) != len(AWS_ACCESS_KEY_ID):
        print(f"\n⚠️  WARNING: Access key has leading/trailing whitespace!")
        AWS_ACCESS_KEY_ID = key_clean
    
    # Check if it's wrapped in quotes
    if (AWS_ACCESS_KEY_ID.startswith('"') and AWS_ACCESS_KEY_ID.endswith('"')) or \
       (AWS_ACCESS_KEY_ID.startswith("'") and AWS_ACCESS_KEY_ID.endswith("'")):
        print(f"\n⚠️  WARNING: Access key appears to be wrapped in quotes!")
        AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID.strip('"\'')
    
    if not AWS_ACCESS_KEY_ID.startswith(('AKIA', 'ASIA')) or len(AWS_ACCESS_KEY_ID) != 20:
        print(f"\n❌ ERROR: Access key format is invalid!")
        print(f"   AWS access keys must:")
        print(f"   - Start with 'AKIA' or 'ASIA'")
        print(f"   - Be exactly 20 characters long")
        print(f"   Your key: '{AWS_ACCESS_KEY_ID[:8]}...' (length: {len(AWS_ACCESS_KEY_ID)})")
        print(f"\n   Please check your .env file:")
        print(f"   1. Make sure AWS_ACCESS_KEY=AKIA... (no quotes)")
        print(f"   2. Make sure there are no spaces around the = sign")
        print(f"   3. Make sure the value is exactly 20 characters")
        print(f"   4. Check if there are multiple AWS_ACCESS_KEY entries (dotenv uses the last one)")
        exit(1)

# Test Claude Sonnet 4 model - use direct model ID format from LiteLLM docs
models_to_test = [
    "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",  # From LiteLLM docs example
    "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",  # From playground (your account)
]

print("\n" + "=" * 60)
print("Testing AWS Bedrock Claude models with LiteLLM...")
print("=" * 60)

successful_model = None
for model in models_to_test:
    print(f"\n🧪 Testing model: {model}")
    try:
        # Use standard completion format (no extra headers needed for basic usage)
        completion_params = {
            "model": model,
            "messages": [{
                "role": "user", 
                "content": "Hello! What Claude model are you? Please respond in exactly 10 words or less."
            }],
            "max_tokens": 50,
            "temperature": 0.0,
        }
        
        # Add API key if available (bearer token authentication)
        if AWS_BEDROCK_API_KEY:
            completion_params["api_key"] = AWS_BEDROCK_API_KEY.strip().strip('"\'')
        
        response = completion(**completion_params)
        print(f"✅ SUCCESS: {model}")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Model used: {response.model}")
        successful_model = model
        break  # Exit on first success
    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {model}")
        print(f"   Error: {error_msg[:300]}...")
        # Don't break - try next model
        continue

print("\n" + "=" * 60)
if successful_model:
    print(f"✅ TEST PASSED: Successfully connected to Bedrock!")
    print(f"   Working model: {successful_model}")
    print(f"\n💡 Next step: Update app.py to use Bedrock as default")
else:
    print("❌ TEST FAILED: Could not connect to any Bedrock model")
    print("\nTroubleshooting:")
    print("1. Verify AWS_BEDROCK_API_KEY is set (bearer token) or AWS_ACCESS_KEY/AWS_SECRET_ACCESS_KEY")
    print("2. Verify AWS_DEFAULT_REGION/AWS_REGION_NAME is correct (default: us-east-1)")
    print("3. Check that Bedrock is enabled and Claude Sonnet 4 access is granted in AWS Console")
    print("4. Verify your AWS credentials have Bedrock access permissions")
    print("5. Make sure the IAM user/role has 'bedrock:InvokeModel' permission")
    print("6. Check AWS Bedrock Console > Model Access > Ensure Claude Sonnet 4 is enabled")
    print("7. Try the model ID from your playground: anthropic.claude-sonnet-4-20250514-v1:0")
print("=" * 60)

