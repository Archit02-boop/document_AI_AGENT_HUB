import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found in .env")
    exit()

client = genai.Client(api_key=api_key)

print("Available Gemini models that support generate_content:\n")

for model in client.models.list():
    supported_actions = getattr(model, "supported_actions", [])

    if "generateContent" in supported_actions:
        print(model.name)