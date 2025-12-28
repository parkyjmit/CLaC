# Chemical Formula Replacement Guide

## Purpose

To reduce the model's excessive dependency on chemical formulas (e.g., Ca3V3(AsO4)4, Zn(AgO2)2), this feature replaces formulas with generic words like "material" during training.

## Configuration

### 1. Config Settings (config/config.yaml)

```yaml
hyperparams:
  # Chemical formula replacement (for reducing formula dependency)
  replace_formula_prob: 0.5            # 50% probability to replace formulas (0.0~1.0)
  replace_formula_start_epoch: 20      # Start replacement from epoch 20
```

**Parameter Descriptions:**
- `replace_formula_prob: 0.0` → Disabled
- `replace_formula_prob: 0.5` → Replace with 50% probability
- `replace_formula_prob: 1.0` → Always replace
- `replace_formula_start_epoch: 0` → Activate from the beginning
- `replace_formula_start_epoch: 20` → Activate from epoch 20

### 2. Automatic Activation (Config-Based) ⭐ **Recommended**

**No code modification required - control via config only!**

```yaml
# config/config.yaml
hyperparams:
  replace_formula_prob: 0.5
  replace_formula_start_epoch: 20  # Auto-activate from epoch 20
```

When training starts, it automatically:
- Epochs 0-19: Formula replacement disabled
- Epoch 20+: Formula replacement enabled (50% probability)

### 3. Manual Control (Advanced Users)

#### Method 1: Manual Toggle in Jupyter/IPython

```python
dm.enable_formula_replacement = True   # Enable
dm.enable_formula_replacement = False  # Disable
```

#### Method 2: Custom Callback

```python
# For more complex logic
class CustomCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Conditional activation logic
        if trainer.current_epoch >= 20 and trainer.global_step % 1000 == 0:
            trainer.datamodule.enable_formula_replacement = True
```

## How It Works

### Replacement Patterns

The following chemical formulas are replaced with "material":

| Original Text | After Replacement |
|---------------|-------------------|
| `The formula is Ca3V3(AsO4)4` | `The formula is material` |
| `Zn(AgO2)2 contains zinc` | `material contains zinc` |
| `BiO2 is an oxide` | `material is an oxide` |
| `NaSr(PO3)3 compound` | `material compound` |

### When Applied

- **During Training**: Applied when `enable_formula_replacement=True` with `replace_formula_prob` probability
- **During Validation/Test**: Always disabled (uses original text)

## Example Workflows

### Scenario 1: Normal Training for 20 Epochs, Then Formula Replacement ⭐

**Modify config only (simplest approach)**

```yaml
# config/config.yaml
hyperparams:
  replace_formula_prob: 0.5
  replace_formula_start_epoch: 20

trainer:
  pl_trainer:
    max_epochs: 50
```

```bash
python train.py
```

### Scenario 2: Enable Formula Replacement from the Start

```yaml
# config/config.yaml
hyperparams:
  replace_formula_prob: 0.5
  replace_formula_start_epoch: 0  # From the beginning
```

### Scenario 3: Two-Stage Training (Using Checkpoints)

```yaml
# Stage 1 config (config_stage1.yaml)
hyperparams:
  replace_formula_prob: 0.0  # Disabled
trainer:
  pl_trainer:
    max_epochs: 30
```

```bash
# Stage 1: Normal training
python train.py --config-name config_stage1
# Saved checkpoint: outputs/.../epoch=29-step=XXX.ckpt
```

```yaml
# Stage 2 config (config_stage2.yaml)
hyperparams:
  replace_formula_prob: 1.0  # Always replace
  replace_formula_start_epoch: 0

resume_from_checkpoint: outputs/.../epoch=29-step=XXX.ckpt

trainer:
  pl_trainer:
    max_epochs: 50  # From 30 to 50 (20 more epochs)
```

```bash
# Stage 2: Formula replacement training (resume from checkpoint)
python train.py --config-name config_stage2
```

## Configuration Summary

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `replace_formula_prob` | `config/config.yaml` | `0.5` | Formula replacement probability (0.0~1.0) |
| `replace_formula_start_epoch` | `config/config.yaml` | `20` | Epoch to start replacement |
| `enable_formula_replacement` | DataModule | `False` | Formula replacement activation flag (manual control) |

## Validation Method

Check the text in training logs, or run a simple test:

```python
from data.datamodule import replace_chemical_formula

text = "The formula is Ca3V3(AsO4)4 and it's an oxide"
print(replace_chemical_formula(text))
# Output: "The formula is material and it's an oxide"
```

## Technical Implementation

### Regex Patterns

The function uses two regex patterns to detect chemical formulas:

1. **Formulas with parentheses**: `Ca3V3(AsO4)4`, `Zn(AgO2)2`
2. **Simple multi-element formulas**: `BiO2`, `NaSr`, `PO3`

**Exclusions:**
- Very short strings (≤2 characters)
- Pure digits

### Training vs Validation Behavior

**Training phase** (`augment=True`):
```python
if self.enable_formula_replacement and random.random() < self.replace_formula_prob:
    text = replace_chemical_formula(text)
```

**Validation/Test phase** (`augment=False`):
- Formula replacement is always skipped
- Ensures validation loss is calculated on the real data distribution

## Best Practices

1. **Start with normal training first**: Train for 10-20 epochs with original formulas before enabling replacement
2. **Use moderate probability**: `0.3-0.5` works well - allows model to see both versions
3. **Monitor validation loss**: Ensure val_loss continues to improve after activation
4. **Two-stage approach**: Consider training Stage 1 without replacement, Stage 2 with replacement

## Related Documentation

- [How to Resume Training](how_to_resume_training.md) - Checkpoint resume training
- [How to Run Sweeps](how_to_sweep.md) - Hyperparameter sweeps
- `data/datamodule.py:30-67` - Implementation of `replace_chemical_formula()`
- `train.py:26-46` - `FormulaReplacementCallback` implementation
