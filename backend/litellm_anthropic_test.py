from litellm import completion
import os
import dotenv

dotenv.load_dotenv()
# Use default GCP authentication (gcloud CLI credentials)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "vertex-ai-key.json")

model = "claude-sonnet-4@20250514"
# model = "claude-4-sonnet"

vertex_ai_project = "eastern-rider-436701-f4" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "global" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = completion(
    model="vertex_ai/" + model,
    messages=[{"role": "user", "content": "what claude model are u?"}],
    temperature=0.0,
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
)
print("\nModel Response", response)