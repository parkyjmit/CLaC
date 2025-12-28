import wandb
import yaml


def main():
    with open("sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="CLaC-revision-sweep")
    print(f"Sweep created. To run the sweep, use the following command:")
    print(f"python run_sweep.py {sweep_id} --gpus <gpu_ids>")


if __name__ == "__main__":
    main()
