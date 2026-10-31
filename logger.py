
import json
from pathlib import Path
from datetime import datetime, timezone


LOG_FILE = Path("api_logs.json")


def save_experiment_log(config, results):
    """Append experiment results to the log file."""

    if LOG_FILE.exists():
        with LOG_FILE.open("r", encoding="utf-8") as file:
            logs = json.load(file)
    else:
        logs = []

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_name": config["test_name"],
        "model": config["model"],
        "parameters": config["parameters"],
        "system_prompt": config["system_prompt"],
        "user_input": config["user_input"],
        **results
    }

    logs.append(log_entry)

    with LOG_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            logs,
            file,
            indent=4,
            ensure_ascii=False
        )

    return log_entry
