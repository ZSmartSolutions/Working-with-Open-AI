
import time
import traceback

from config_loader import load_experiments
from experiment_runner import initialize_client, run_experiment
from logger import save_experiment_log


def main():
    """Run all configured LLM experiments."""

    # Step 1: Load configurations and initialize client
    try:
        experiments = load_experiments()
        client = initialize_client()

    except Exception as error:
        print("\nAPPLICATION INITIALIZATION FAILED")
        print("Error type:", type(error).__name__)
        print("Error message:", str(error))
        traceback.print_exc()
        return

    total = len(experiments)
    successful = 0
    failed = 0

    print(f"\nFound {total} experiment configurations.\n")

    # Step 2: Execute experiments
    for index, config in enumerate(experiments, start=1):

        test_name = config["test_name"]

        print(f"Running [{index}/{total}]: {test_name}")

        try:
            # Execute API request
            print("  -> Sending request to OpenRouter...")

            results = run_experiment(client, config)

            print("  -> API response received.")

            # Save experiment results
            print("  -> Saving experiment log...")

            save_experiment_log(config, results)

            successful += 1

            print("  -> Experiment completed successfully.")
            print(f"  -> Input tokens: {results['input_tokens']}")
            print(f"  -> Output tokens: {results['output_tokens']}")

        except Exception as error:
            failed += 1

            print("\n  -> EXPERIMENT FAILED")
            print("  -> Error type:", type(error).__name__)
            print("  -> Error message:", str(error))

            print("\n  -> COMPLETE TRACEBACK:")
            traceback.print_exc()

        print()

        # Small delay between API requests
        if index < total:
            time.sleep(1)

    # Step 3: Execution summary
    print("\nExperiment execution completed.")
    print("--------------------------------")
    print(f"Total experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
