
from openai import OpenAI, APIError
from dotenv import load_dotenv
import json
import os


# 1. Initialize OpenRouter client
def initialize_client():
    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )


# 2. Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_user_preferences",
            "description": "Get a user's preferences for personalized recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The unique identifier for the user"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of preferences (food, movies, books)"
                    }
                },
                "required": ["user_id", "category"],
                "additionalProperties": False
            }
        }
    }
]


# 3. Actual Python function implementation
def get_user_preferences(user_id, category):

    preferences = {
        "user123": {
            "food": ["Italian", "Mexican", "vegetarian options"],
            "movies": ["sci-fi", "documentaries", "comedy"],
            "books": ["non-fiction", "history", "technology"]
        }
    }

    user_prefs = preferences.get(user_id, {})
    return user_prefs.get(category, ["No preferences found"])


# 4. Main execution
def main():

    client = initialize_client()

    input_messages = [
        {
            "role": "user",
            "content": "I'm user123. Can you recommend some restaurants based on my food preferences?"
        }
    ]

    try:

        # STEP 1: Initial API request
        print("\nStep 1: Making initial request...")

        response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=input_messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "get_user_preferences"
            }
    }
        )

        # Check whether choices were returned
        if not response.choices:
            print("No choices returned by AI.")
            print("Full response:", response)
            return

        assistant_message = response.choices[0].message

        print("\nStep 2: AI's response:")
        print(assistant_message)

        # STEP 2: Add AI response to conversation history
        input_messages.append(assistant_message.model_dump(exclude_none=True))

        # STEP 3: Process function calls
        print("\nStep 3: Processing function calls...")

        if assistant_message.tool_calls:

            for tool_call in assistant_message.tool_calls:

                if tool_call.function.name == "get_user_preferences":

                    args = json.loads(tool_call.function.arguments)

                    print(f"  -> Calling function with: {args}")

                    preferences = get_user_preferences(
                        args["user_id"],
                        args["category"]
                    )

                    print(f"  -> Function returned: {preferences}")

                    # Send function result back to AI
                    input_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({
                            "preferences": preferences
                        })
                    })

        else:
            print("AI did not request a function call.")
            print("AI response:", assistant_message.content)
            return

        # STEP 4: Send function result back to AI
        print("\nStep 4: Getting final response...")

        final_response = client.chat.completions.create(
            model="qwen2.5:1.5b",
            messages=input_messages,
            tools=tools
        )

        if not final_response.choices:
            print("No final response choices returned.")
            print("Full response:", final_response)
            return

        # STEP 5: Display final answer
        print("\n" + "-" * 100)
        print("FINAL ANSWER:")
        print("-" * 100)

        print(final_response.choices[0].message.content)

    except APIError as e:
        print("\nAPI ERROR:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")

    except json.JSONDecodeError as e:
        print("\nJSON ERROR:")
        print(f"Could not parse function arguments: {e}")

    except Exception as e:
        print("\nUNEXPECTED ERROR:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")


if __name__ == "__main__":
    main()