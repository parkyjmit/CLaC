import wandb
import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id", type=str, help="Wandb sweep ID.")
    parser.add_argument("--gpus", type=str, help="Comma separated list of GPUs to use.", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["WANDB_MODE"] = "online"
    
    def train():
        # Initialize wandb to get the config, but don't create a full run context here.
        run = wandb.init(
            mode="online",
            settings=wandb.Settings(init_timeout=3000),
        )
        config = run.config
        command = ["python", "train.py"]
        for key, value in config.items():
            command.append(f"{key}={value}")
        
        # The subprocess will have its own complete wandb run.
        subprocess.run(command)
        run.finish()

    wandb.agent(args.sweep_id, function=train, project="CLaC-revision-sweep")

if __name__ == "__main__":
    main()
