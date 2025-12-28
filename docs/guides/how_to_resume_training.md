# Checkpoint Resume Training Guide

## Purpose
Resume training from a previously saved checkpoint. Model weights, optimizer state, learning rate, epoch, and more are all restored.

## Setup

### 1. Checkpoint Saving (Automatic)

Checkpoints are automatically saved during training:

```yaml
# config/trainer/default.yaml
model_checkpoints:
  save_top_k: 3        # Save top 3 checkpoints
  save_last: true      # Also save last checkpoint
  verbose: true
```

Save location: `outputs/{exp_name}/epoch=XX-step=XXXX.ckpt`

### 2. Resume from Checkpoint ⭐

**Method 1: Config Setting (Recommended)**

```yaml
# config/config.yaml
resume_from_checkpoint: outputs/my-experiment/epoch=49-step=5000.ckpt
```

```bash
python train.py
```

**Method 2: Command-line Override**

```bash
python train.py resume_from_checkpoint=outputs/my-experiment/epoch=49-step=5000.ckpt
```

**Method 3: Using a Different Config File**

```yaml
# config/resume_config.yaml
defaults:
  - config

resume_from_checkpoint: outputs/my-experiment/epoch=49-step=5000.ckpt

trainer:
  pl_trainer:
    max_epochs: 100  # Continue from 50 to 100
```

```bash
python train.py --config-name resume_config
```

## What Gets Restored

When resuming from a checkpoint, the following are restored:

✅ **Model Weights** - All layer parameters
✅ **Optimizer State** - Adam's momentum, variance, etc.
✅ **Learning Rate** - LR scheduler state
✅ **Current Epoch** - Continue from where training stopped
✅ **Global Step** - Total training step count
✅ **Random State** - Random seed for reproducibility

## Usage Examples

### Example 1: Resume After Interruption

```bash
# Initial training
python train.py

# Interrupted... last checkpoint: outputs/.../last.ckpt

# Resume with same settings
python train.py resume_from_checkpoint=outputs/.../last.ckpt
```

### Example 2: Extend Training Duration

```yaml
# Original config: max_epochs=50
# Checkpoint saved: epoch=49-step=5000.ckpt

# New config
resume_from_checkpoint: outputs/.../epoch=49-step=5000.ckpt
trainer:
  pl_trainer:
    max_epochs: 100  # Train 50 more epochs
```

### Example 3: Fine-tuning with Different Settings

```yaml
# Stage 1: Pre-training (50 epochs)
hyperparams:
  batch_size: 32
  replace_formula_prob: 0.0
trainer:
  pl_trainer:
    max_epochs: 50

# Result: epoch=49-step=5000.ckpt
```

```yaml
# Stage 2: Fine-tuning with formula replacement
resume_from_checkpoint: outputs/stage1/epoch=49-step=5000.ckpt

hyperparams:
  batch_size: 16          # Smaller batch
  replace_formula_prob: 0.5
  replace_formula_start_epoch: 0  # Activate immediately

trainer:
  pl_trainer:
    max_epochs: 70  # 20 more epochs
```

```bash
python train.py --config-name stage2_config
```

### Example 4: Resume with Changed Learning Rate

```yaml
# When training has converged but you want to continue with lower LR
resume_from_checkpoint: outputs/.../epoch=99-step=10000.ckpt

model:
  optimizer:
    lr: 0.00001  # Lower learning rate

trainer:
  pl_trainer:
    max_epochs: 120  # 20 more epochs
```

## Important Notes

### ⚠️ Model Architecture Changes
Changing the model architecture when loading a checkpoint may cause errors. The following changes are safe:

**Safe Changes:**
- Hyperparameters (batch_size, learning rate, etc.)
- Data augmentation settings
- Adding/removing callbacks
- Changing max epochs

**Risky Changes (May Cause Errors):**
- Model architecture (adding/removing layers)
- Changing embedding dimensions
- Changing optimizer type (Adam → SGD)

### ⚠️ strict=False
The code uses `strict=False`, so it will attempt to load even if some keys are missing. However, it's best to maintain the same model structure when possible.

### ⚠️ Absolute vs Relative Paths
- Absolute path: `/home/user/outputs/exp/epoch=10.ckpt` (recommended)
- Relative path: `outputs/exp/epoch=10.ckpt` (depends on execution location)

## Troubleshooting

### Problem 1: Checkpoint File Not Found
```
FileNotFoundError: Checkpoint not found: outputs/.../epoch=10.ckpt
```

**Solution:**
- Verify the path (absolute path recommended)
- Check checkpoint files with `ls outputs/*/`

### Problem 2: Key Mismatch Error
```
RuntimeError: Error(s) in loading state_dict for CLaCLite
```

**Solution:**
- Verify model architecture hasn't changed
- Ensure config file matches original training
- Check that `strict=False` is applied (already in train.py)

### Problem 3: Epoch Starts from 0 Again
```
resume_from_checkpoint not applied properly
```

**Solution:**
- Check if config has `resume_from_checkpoint: null`
- Verify command-line override is correct

## Quick Reference

| Scenario | Config Setting | Command |
|---------|---------------|---------|
| Initial training | `resume_from_checkpoint: null` | `python train.py` |
| Resume with same settings | `resume_from_checkpoint: path/to/ckpt` | `python train.py` |
| Resume via CLI | (no config needed) | `python train.py resume_from_checkpoint=path/to/ckpt` |
| Extend training | `max_epochs: 100` + checkpoint | `python train.py` |
| Fine-tuning | Change LR/augmentation + checkpoint | `python train.py --config-name finetune` |

## Config Example

```yaml
# config/config.yaml

# Checkpoint resume
resume_from_checkpoint: null  # Initial start (null or commented out)
# resume_from_checkpoint: outputs/exp/epoch=49-step=5000.ckpt  # Resume

hyperparams:
  replace_formula_prob: 0.5
  replace_formula_start_epoch: 20

trainer:
  pl_trainer:
    max_epochs: 50
```

## Related Documentation

- [Chemical Formula Replacement Guide](how_to_use_formula_replacement.md) - Formula replacement feature
- [Sweep Guide](how_to_sweep.md) - Hyperparameter sweep
