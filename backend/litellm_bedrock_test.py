from litellm import completion
import os
import dotenv

dotenv.load_dotenv()


response = completion(
  model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
  messages=[{ "content": "Hello, how are you? what claude model are u?","role": "user"}],
  api_key=os.getenv("AWS_BEDROCK_API_KEY")
)

print(response)