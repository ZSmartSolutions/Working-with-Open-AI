
import os
import base64
import mimetypes

from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Retrieve OpenRouter API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# Initialize OpenRouter client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# Local image path
image_path = "images.jpeg"

# Check whether image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")


# Detect image format
mime_type, _ = mimetypes.guess_type(image_path)

if not mime_type or not mime_type.startswith("image/"):
    raise ValueError("Unsupported image format")


# Convert image into Base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode("utf-8")


# Create Base64 data URL
image_data_url = f"data:{mime_type};base64,{encoded_image}"


# Prepare user input
user_input = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url
                }
            }
        ]
    }
]


# Send request to OpenRouter
response = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=user_input
)


# Display response
print(response.choices[0].message.content)
