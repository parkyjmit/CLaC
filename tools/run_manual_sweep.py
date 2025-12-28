import yaml
import subprocess
import itertools
import argparse
import os

def main():
    """Runs a manual sweep of experiments based on a YAML configuration file."""
    parser = argparse.ArgumentParser(description="Run a manual sweep of experiments.")
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated list of GPUs to use (e.g., '0,1').")
    parser.add_argument("--config", type=str, default="sweep.yaml", help="Path to the sweep configuration YAML file.")
    parser.add_argument("--start-from", type=int, default=0, help="Start from this experiment index (0-based).")
    args = parser.parse_args()

    # --- Environment Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Load Sweep Configuration ---
    try:
        with open(args.config, 'r') as f:
            sweep_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Sweep configuration file not found at {args.config}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    program_to_run = sweep_config.get("program", "train.py")
    parameters = sweep_config.get("parameters", {})

    if not parameters:
        print("No parameters to sweep found in the configuration file.")
        return

    # --- Generate Parameter Combinations ---
    param_keys = parameters.keys()
    param_values = [p['values'] for p in parameters.values()]
    
    combinations = list(itertools.product(*param_values))
    total_runs = len(combinations)

    print(f"Total combinations: {total_runs}")
    print(f"Starting from index: {args.start_from}")

    # Skip to the starting index
    combinations_to_run = combinations[args.start_from:]
    print(f"Running {len(combinations_to_run)} experiments.")

    # --- Run Experiments ---
    for i, combo in enumerate(combinations_to_run, start=args.start_from):
        command = ["python", program_to_run]
        print("-" * 50)
        print(f"Running experiment {i + 1}/{total_runs}")

        current_params = {}
        param_list_for_display = []
        for key, value in zip(param_keys, combo):
            param_str = f"{key}={value}"
            command.append(param_str)
            param_list_for_display.append(param_str)
            current_params[key] = value

        print(f"Combination: {', '.join(param_list_for_display)}")
        print(f"Full Command: {' '.join(command)}")

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment {i + 1}. Process returned with error code {e.returncode}.")
            print("Stopping sweep.")
            break
        except KeyboardInterrupt:
            print("\nSweep interrupted by user.")
            break

    print("-" * 50)
    print("Manual sweep finished.")

if __name__ == "__main__":
    main()
