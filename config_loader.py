
import json
from pathlib import Path


CONFIG_FILE = Path("test_configs.json")


def load_experiments():
    """Load and validate experiment configurations from JSON."""

    # Check whether configuration file exists
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    # Load JSON configurations
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        experiments = json.load(file)

    # Validate top-level structure
    if not isinstance(experiments, list):
        raise ValueError("Configurations must be a JSON list.")

    if not experiments:
        raise ValueError("No experiment configurations found.")

    # Required configuration fields
    required_fields = [
        "test_name",
        "model",
        "system_prompt",
        "user_input"
    ]

    for index, config in enumerate(experiments, start=1):

        # Validate experiment structure
        if not isinstance(config, dict):
            raise ValueError(
                f"Experiment {index} must be an object."
            )

        # Check required fields
        missing = [
            field for field in required_fields
            if field not in config or config[field] is None
        ]

        if missing:
            raise ValueError(
                f"Experiment {index} is missing: {missing}"
            )

        # Validate optional parameters
        parameters = config.get("parameters") or {}

        if not isinstance(parameters, dict):
            raise ValueError(
                f"Experiment {index}: parameters must be an object."
            )

        # Normalize missing or null parameters
        config["parameters"] = parameters

        # Validate user input
        user_input = config["user_input"]

        if isinstance(user_input, str):
            if not user_input.strip():
                raise ValueError(
                    f"Experiment {index}: user_input cannot be empty."
                )

        elif isinstance(user_input, list):

            if not user_input:
                raise ValueError(
                    f"Experiment {index}: conversation cannot be empty."
                )

            for message_index, message in enumerate(user_input, start=1):

                if not isinstance(message, dict):
                    raise ValueError(
                        f"Experiment {index}, message {message_index}: "
                        "message must be an object."
                    )

                if "role" not in message or "content" not in message:
                    raise ValueError(
                        f"Experiment {index}, message {message_index}: "
                        "role and content are required."
                    )

                if message["role"] not in ("user", "assistant"):
                    raise ValueError(
                        f"Experiment {index}, message {message_index}: "
                        "role must be 'user' or 'assistant'."
                    )

                if not isinstance(message["content"], str):
                    raise ValueError(
                        f"Experiment {index}, message {message_index}: "
                        "content must be a string."
                    )

        else:
            raise ValueError(
                f"Experiment {index}: user_input must be "
                "a string or a list of messages."
            )

    return experiments
