
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


def initialize_client():
    """Initialize the OpenRouter API client."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )



def run_experiment(client, config):
    """Execute one LLM experiment with conversation history."""

    parameters = config.get("parameters") or {}

    # Initialize messages with system prompt
    messages = [
        {
            "role": "system",
            "content": config["system_prompt"]
        }
    ]

    user_input = config["user_input"]

    # Handle single user input
    if isinstance(user_input, str):
        messages.append({
            "role": "user",
            "content": user_input
        })

    # Handle conversation history
    elif isinstance(user_input, list):
        messages.extend(user_input)

    else:
        raise ValueError(
            "user_input must be a string or a list of messages."
        )

    # Send messages to LLM
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        **parameters
    )

    choice = response.choices[0]

    return {
        "assistant_response": choice.message.content,
        "finish_reason": choice.finish_reason,
        "input_tokens": response.usage.prompt_tokens if response.usage else None,
        "output_tokens": response.usage.completion_tokens if response.usage else None
    }
