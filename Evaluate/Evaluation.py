import json
import re
import time

from openai import OpenAI


# ==========================================================
# STEP 1: CONNECT TO LOCAL OLLAMA
# ==========================================================

# Ollama exposes an API compatible with the OpenAI SDK.
# The request stays on your local machine.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Placeholder; Ollama does not require a real API key.
)

# The model must already be downloaded in Ollama.
MODEL_NAME = "qwen2.5:1.5b"

# These are the only valid classification labels.
VALID_LABELS = ["Hardware", "Software", "Other"]


# ==========================================================
# STEP 2: DEFINE THE CLASSIFICATION INSTRUCTIONS
# ==========================================================

SYSTEM_PROMPT = """
You are an expert in categorizing IT support tickets.

Categorize the given ticket into exactly one of these categories:

- Hardware: Physical devices and equipment.
- Software: Applications, operating systems, software errors,
  login problems and software configuration.
- Other: General questions, requests, access requests and anything
  that does not fit Hardware or Software.

Respond with ONLY one word:
Hardware, Software, or Other.

Do not explain your answer.
"""


# ==========================================================
# STEP 3: CLASSIFY A SINGLE SUPPORT TICKET
# ==========================================================

def categorize_support_ticket(ticket_text):
    """
    Send a support ticket to the local Qwen model.

    Returns:
        The predicted category: Hardware, Software, Other,
        or Unknown if the model produces an invalid response.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            # System message defines the model's role and rules.
            {"role": "system", "content": SYSTEM_PROMPT},

            # User message contains the actual ticket.
            {"role": "user", "content": ticket_text}
        ],

        # Temperature 0 makes the output more consistent.
        temperature=0
    )

    # Extract the model's response.
    model_output = response.choices[0].message.content

    # Handle an empty response.
    if not model_output:
        return "Unknown"

    # Remove unnecessary whitespace.
    model_output = model_output.strip()

    # Check whether the response contains a valid category.
    # This also handles responses such as "Category: Hardware".
    match = re.search(
        r"\b(Hardware|Software|Other)\b",
        model_output,
        re.IGNORECASE
    )

    if match:
        # Normalize capitalization.
        return match.group(1).capitalize()

    # The model did not return a recognizable category.
    return "Unknown"


# ==========================================================
# STEP 4: CREATE THE TEST DATASET
# ==========================================================

def create_test_dataset():
    """
    Create test examples containing:
        ticket_text: The input ticket.
        correct_label: The human-provided expected answer.
    """

    test_cases = [

        # Hardware issues
        {
            "ticket_text": "My monitor won't turn on - help!",
            "correct_label": "Hardware"
        },
        {
            "ticket_text": "Keyboard keys are sticking and not responding",
            "correct_label": "Hardware"
        },
        {
            "ticket_text": "Laptop battery drains very quickly",
            "correct_label": "Hardware"
        },
        {
            "ticket_text": "Printer is making weird noises and not printing",
            "correct_label": "Hardware"
        },
        {
            "ticket_text": "Mouse cursor jumps around randomly",
            "correct_label": "Hardware"
        },

        # Software issues
        {
            "ticket_text": "Excel crashes every time I try to open it",
            "correct_label": "Software"
        },
        {
            "ticket_text": "Can't install the new antivirus update",
            "correct_label": "Software"
        },
        {
            "ticket_text": "Email client won't sync with server",
            "correct_label": "Software"
        },
        {
            "ticket_text": "Getting blue screen errors on Windows",
            "correct_label": "Software"
        },
        {
            "ticket_text": "Chrome browser running very slowly",
            "correct_label": "Software"
        },

        # Other issues
        {
            "ticket_text": "How do I request vacation time?",
            "correct_label": "Other"
        },
        {
            "ticket_text": "What's the WiFi password for the guest network?",
            "correct_label": "Other"
        },
        {
            "ticket_text": "Need access to the accounting shared folder",
            "correct_label": "Other"
        },
        {
            "ticket_text": "Where can I find the office supply ordering form?",
            "correct_label": "Other"
        },
        {
            "ticket_text": "Best restaurants near the office?",
            "correct_label": "Other"
        },

        # Edge cases
        {
            "ticket_text": "My laptop won't connect to WiFi",
            "correct_label": "Hardware"
        },
        {
            "ticket_text": "Can't log into the company portal",
            "correct_label": "Software"
        },
        {
            "ticket_text": "Desk phone not working",
            "correct_label": "Hardware"
        },
        {
            "ticket_text": "Forgot my password",
            "correct_label": "Software"
        },
        {
            "ticket_text": "Need new headphones",
            "correct_label": "Other"
        }
    ]

    print(f"Created dataset with {len(test_cases)} test cases.")

    return test_cases


# ==========================================================
# STEP 5: EVALUATE THE MODEL
# ==========================================================

def run_evaluation(test_cases):
    """
    Run every test case through Ollama.

    Compare:
        Predicted label = Model's answer
        Correct label   = Human-provided answer

    Count how many predictions are correct.
    """

    results = []

    passed = 0
    failed = 0

    print("\nStarting model evaluation...")
    print("=" * 60)

    for index, case in enumerate(test_cases, start=1):

        ticket = case["ticket_text"]
        expected = case["correct_label"]

        # Ask the local model to classify the ticket.
        predicted = categorize_support_ticket(ticket)

        # Compare model prediction against the expected answer.
        is_correct = predicted == expected

        if is_correct:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        # Store the result for later reporting.
        result = {
            "ticket_text": ticket,
            "correct_label": expected,
            "predicted_label": predicted,
            "passed": is_correct
        }

        results.append(result)

        # Print individual test results.
        print(f"\nTest Case: {index}")
        print(f"Ticket: {ticket}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Result: {status}")

    return results, passed, failed


# ==========================================================
# STEP 6: GENERATE THE EVALUATION REPORT
# ==========================================================

def generate_report(results, passed, failed):
    """
    Calculate accuracy and display evaluation statistics.
    """

    total = len(results)

    # Calculate accuracy as a percentage.
    accuracy = (passed / total) * 100 if total else 0

    print("\n")
    print("=" * 60)
    print("             EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Total test cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Count results for each category.
    print("\nCATEGORY-WISE RESULTS")

    for label in VALID_LABELS:

        # Select results belonging to this expected category.
        category_results = [
            item for item in results
            if item["correct_label"] == label
        ]

        category_passed = sum(
            item["passed"] for item in category_results
        )

        category_total = len(category_results)

        category_accuracy = (
            category_passed / category_total
        ) * 100 if category_total else 0

        print(
            f"{label}: {category_passed}/{category_total} "
            f"({category_accuracy:.2f}%)"
        )

    # Display incorrect predictions for debugging.
    print("\nINCORRECT PREDICTIONS")

    incorrect_results = [
        item for item in results if not item["passed"]
    ]

    if not incorrect_results:
        print("No incorrect predictions!")
    else:
        for item in incorrect_results:
            print(f"\nTicket: {item['ticket_text']}")
            print(f"Expected: {item['correct_label']}")
            print(f"Predicted: {item['predicted_label']}")

    # Save detailed results to a JSON file.
    with open("evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nDetailed results saved to evaluation_results.json")


# ==========================================================
# STEP 7: MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    # Create our test dataset.
    test_dataset = create_test_dataset()

    # Run all test cases against the local Qwen model.
    results, passed, failed = run_evaluation(test_dataset)

    # Generate the final evaluation report.
    generate_report(results, passed, failed)