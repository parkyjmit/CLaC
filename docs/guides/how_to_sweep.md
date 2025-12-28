# How to Run a Wandb Sweep

This guide explains how to run hyperparameter sweeps using Weights & Biases (wandb).

**Note:** All commands should be run from the project root directory.

## Prerequisites

- Active wandb account
- wandb Python package installed
- Configured `sweep.yaml` file

## Instructions

### 1. Login to Wandb

First, log in to your wandb account:

```bash
wandb login
```

Enter your API key when prompted. You can find your API key at https://wandb.ai/authorize

### 2. Create the Sweep

Create a new sweep using the `create_sweep.py` script. This script reads the `sweep.yaml` configuration file and creates a new sweep in your wandb project.

```bash
python tools/create_sweep.py
```

**Output:**
```
Created sweep with ID: abc123xyz
```

**Save this sweep ID** - you will need it for the next step.

### 3. Run the Sweep Agents

To run the experiments, start one or more wandb agents. Each agent will ask the wandb server for a set of hyperparameters to run. You can run multiple agents in parallel on different machines or different GPUs.

Use the `run_sweep.py` script with the sweep ID and the `--gpus` argument to specify which GPUs to use.

**Examples:**

Run an agent on GPUs 0, 1, 2, 3:
```bash
python tools/run_sweep.py <sweep_id> --gpus 0,1,2,3
```

Run another agent on GPUs 4, 5, 6, 7:
```bash
python tools/run_sweep.py <sweep_id> --gpus 4,5,6,7
```

**Important:** Replace `<sweep_id>` with the ID from step 2.

### 4. Running Agents in the Background (with `nohup`)

To run agents in the background and prevent them from stopping when you close the terminal, use `nohup`.

The `nohup` command runs a command and ignores the HUP (hangup) signal. Redirect the output to a file to check for errors later.

**Examples:**

Run agent 1 on GPUs 0-3 in the background:
```bash
nohup python tools/run_sweep.py <sweep_id> --gpus 0,1,2,3 > logs/sweep_agent_1.log 2>&1 &
```

Run agent 2 on GPUs 4-7 in the background:
```bash
nohup python tools/run_sweep.py <sweep_id> --gpus 4,5,6,7 > logs/sweep_agent_2.log 2>&1 &
```

This will:
- Run agents in the background
- Write output to `logs/sweep_agent_1.log` and `logs/sweep_agent_2.log`
- Continue running even if you close the terminal

**Check agent status:**
```bash
# View running processes
ps aux | grep run_sweep.py

# Monitor log files
tail -f logs/sweep_agent_1.log

# Check all sweep logs
ls -lh logs/sweep_agent_*.log
```

## Sweep Configuration

The sweep configuration is defined in `sweep.yaml`. Example:

```yaml
program: train.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  model.optimizer.lr:
    min: 0.00001
    max: 0.001
  hyperparams.batch_size:
    values: [16, 32, 64]
  model.graph_encoder.hidden_dim:
    values: [128, 256, 512]
```

See `sweep.yaml` for the full configuration.

## Monitoring Results

### View in Wandb Dashboard

1. Go to https://wandb.ai
2. Navigate to your project
3. Click on "Sweeps" tab
4. Select your sweep ID
5. View real-time results, charts, and parallel coordinates plots

### Stop a Sweep

**Stop all agents:**
```bash
# Find agent processes
ps aux | grep run_sweep.py

# Kill specific process
kill <process_id>

# Or kill all sweep agents
pkill -f run_sweep.py
```

**Pause sweep in wandb:**
- Go to wandb dashboard
- Click on your sweep
- Click "Pause" button

## Best Practices

1. **Start small**: Run 1-2 agents first to verify configuration
2. **Monitor logs**: Check log files regularly for errors
3. **Use nohup**: For long-running sweeps, always use nohup
4. **GPU allocation**: Distribute GPUs evenly across agents
5. **Sweep size**: Set a reasonable max run count in `sweep.yaml`
6. **Resource limits**: Don't overload your system with too many parallel agents

## Troubleshooting

### Problem: "sweep_id not found"
**Solution:** Verify the sweep ID is correct. List all sweeps:
```bash
wandb sweep --list
```

### Problem: Agent stops unexpectedly
**Solution:**
- Check log files for errors
- Verify GPU availability: `nvidia-smi`
- Ensure sufficient disk space: `df -h`

### Problem: "No GPU available"
**Solution:**
- Check specified GPUs exist: `nvidia-smi`
- Verify GPU IDs are correct (0-indexed)
- Check if GPUs are already in use

## Advanced Usage

### Manual Sweep (Without Wandb)

For running a manual parameter sweep without wandb:

```bash
python tools/run_manual_sweep.py --config sweep_config.yaml
```

See [run_manual_sweep.py](../../tools/run_manual_sweep.py) for details.

### Multiple Sweeps

Run multiple independent sweeps:

```bash
# Create sweep 1
python tools/create_sweep.py --config sweep_config_1.yaml
# Run agents for sweep 1

# Create sweep 2
python tools/create_sweep.py --config sweep_config_2.yaml
# Run agents for sweep 2
```

## Related Documentation

- [Wandb Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [sweep.yaml](../../sweep.yaml) - Sweep configuration file
- [tools/create_sweep.py](../../tools/create_sweep.py) - Sweep creation script
- [tools/run_sweep.py](../../tools/run_sweep.py) - Agent execution script
