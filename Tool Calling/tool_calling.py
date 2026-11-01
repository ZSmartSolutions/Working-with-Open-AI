
# Import required libraries
import os
import time
import threading

from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Retrieve OpenRouter API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# Initialize OpenRouter client using OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# Shared flag to control the progress animation
done = False


# Function to display a spinning progress animation
def show_progress():

    chars = "|/-\\"
    i = 0

    while not done:

        print(
            f"\rSearching the web {chars[i % len(chars)]}",
            end="",
            flush=True
        )

        time.sleep(0.1)
        i += 1


# Display starting message
print("Starting web search...")


# Create and start background thread
thread = threading.Thread(target=show_progress)
thread.start()


# Record request start time
start_time = time.time()

try:

    # Send request through OpenRouter
    response = client.responses.create(

        # OpenRouter model identifier
        model="openai/gpt-5.2",

        # Enable OpenRouter web search
        tools=[
            {"type": "openrouter:web_search"}
        ],

        # Limit maximum generated output tokens
        max_output_tokens=1500,

        # User question
        input=(
            "Search the web for the officially confirmed release date "
            "of the next One Piece manga chapter. "
            "Consider today's date and cite your sources. "
            "If the release date is not confirmed, say so."
        )
    )

except Exception as e:

    # Display API errors
    print(f"\nRequest failed: {type(e).__name__}: {e}")

else:

    # Calculate execution time
    elapsed_time = time.time() - start_time

finally:

    # Stop progress animation
    done = True

    # Wait for thread to finish
    thread.join()


# Display response details if request succeeded
if "response" in locals():

    print(f"\nSearch completed in {elapsed_time:.2f} seconds")

    print("-" * 60)

    # Display response status
    print("\nResponse Status:", response.status)

    # Display model used
    print("Model:", response.model)

    # Display number of output items
    print("Output Items:", len(response.output))

    # Display extracted text
    print("\nGenerated Answer:")

    if response.output_text:
        print(response.output_text)
    else:
        print("No output_text returned.")

    # Display complete response for debugging
    print("\n" + "=" * 60)
    print("COMPLETE API RESPONSE")
    print("=" * 60)

    print(response.model_dump_json(indent=2))
