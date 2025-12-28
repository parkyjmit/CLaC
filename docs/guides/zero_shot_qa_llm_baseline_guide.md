# Zero-Shot QA LLM Baseline Evaluation Guide

## Overview

Evaluate pure LLM baselines (LLaMA, Galactica, etc.) on zero-shot materials QA tasks using CIF structure descriptions as input.

**Key Difference from CLaC:**
- **CLaC**: Graph encoder + Text encoder (contrastive learning)
- **LLM Baseline**: Text-only model (CIF text → question answering)

**Note:** All commands should be run from the project root directory.

## Quick Start

### Basic Usage

```bash
# Galactica-125M (fast - batch processing)
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label structure_question_list \
    --batch-size 16 \
    --device cuda:0

# With custom data path
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label structure_question_list \
    --data-path /path/to/your/dataset \
    --device cuda:0

# Galactica-1.3B (2048 tokens)
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-1.3b \
    --label structure_question_list \
    --batch-size 4 \
    --device cuda:0

# LLaMA-2-7B (4096 tokens - better for long prompts)
python evaluation/compute_metrics_baseline.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --label structure_question_list \
    --batch-size 2 \
    --device cuda:0

# Mistral-7B (8192 tokens - handles long structures)
python evaluation/compute_metrics_baseline.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --label structure_question_list \
    --batch-size 2 \
    --device cuda:0

# Custom max length (if needed)
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label structure_question_list \
    --max-length 4096 \
    --device cuda:0
```

### All Question Types

```bash
# Structure questions
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label structure_question_list \
    --device cuda:0

# Composition questions
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label composition_question_list \
    --device cuda:0

# Oxide questions
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label oxide_question_list \
    --device cuda:0
```

## How It Works

### 1. Input Format

The script provides materials structure as CIF text to the LLM:

```
Given the following crystal structure in CIF format:

data_structure
_cell_length_a    5.123
_cell_length_b    5.123
...
loop_
_atom_site_label
_atom_site_fract_x
...

Question: Which of the following best describes this material's crystal structure?

Choices:
A) Face-centered cubic (FCC)
B) Body-centered cubic (BCC)
C) Hexagonal close-packed (HCP)
D) Simple cubic

Answer with only the letter (A, B, C, D):
```

### 2. Answer Parsing

The script:
1. Generates LLM response (max 10 tokens)
2. Extracts first letter (A/B/C/D)
3. If parsing fails → random guess (marked as parsing failure)

### 3. Evaluation Metrics

- **Accuracy**: Percentage of correct answers
- **Bootstrap CI**: 95% confidence interval (1000 iterations)
- **Random Baseline**: Expected performance from random guessing
- **Parsing Failures**: Percentage of samples where LLM didn't output A/B/C/D

## Output

### Results Directory

```
outputs/zero_shot_qa_baseline/
└── facebook_galactica-125m/
    ├── qa_structure_question_list.json
    ├── qa_composition_question_list.json
    └── qa_oxide_question_list.json
```

### JSON Format

```json
{
  "evaluation_method": "zero-shot QA (LLM baseline)",
  "label": "structure_question_list",
  "model_name": "facebook/galactica-125m",
  "num_samples": 12634,
  "parsing_failures": 125,
  "bootstrap_params": {
    "n_bootstrap": 1000,
    "confidence_level": 0.95,
    "random_seed": 42
  },
  "accuracy": {
    "mean": 0.4523,
    "std": 0.0044,
    "ci_lower": 0.4437,
    "ci_upper": 0.4609,
    "n_bootstrap": 1000
  },
  "random_baseline": {
    "mean": 0.2501,
    "std": 0.0039,
    "ci_lower": 0.2425,
    "ci_upper": 0.2577,
    "n_bootstrap": 1000
  }
}
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | **required** | HuggingFace model name |
| `--data-path` | `/home/lucky/Projects/CLaC-revision/datafiles/mp_3d_2020_papers` | Path to dataset (without `_test.parquet` suffix) |
| `--label` | `structure_question_list` | Question type |
| `--batch-size` | `8` | **Batch size for parallel generation** - Higher = faster (use 16-32 for speed) |
| `--num-workers` | `4` | Number of workers (currently unused) |
| `--device` | `cuda:0` | Device to use |
| `--max-new-tokens` | `10` | Max tokens to generate for answer |
| `--max-length` | `None` (auto-detect) | Max input sequence length (use model's max if not specified) |
| `--output-dir` | `outputs/zero_shot_qa_baseline` | Output directory |

## Comparison with CLaC

### Use `evaluation/plot_zero_shot_comparison.py` to compare:

```bash
# Generate LLM baseline results
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --label structure_question_list \
    --device cuda:0

# Move results to zero_shot_qa directory (for plotting)
mkdir -p outputs/zero_shot_qa/galactica-125m_baseline
cp outputs/zero_shot_qa_baseline/facebook_galactica-125m/qa_structure_question_list.json \
   outputs/zero_shot_qa/galactica-125m_baseline/

# Plot comparison
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label structure_question_list \
    --models "clac_model_last" "galactica-125m_baseline" \
    --labels "CLaC" "Galactica-125M" \
    --output figures/qa_comparison.svg
```

## Model Recommendations

### Context Length by Model

| Model | Context Length | Memory | Speed | Best For |
|-------|---------------|---------|-------|----------|
| Galactica-125M | 2048 | ~1GB | Fast | Quick tests, short prompts |
| Galactica-1.3B | 2048 | ~3GB | Medium | Better accuracy, short prompts |
| LLaMA-2-7B | 4096 | ~14GB | Slow | **Recommended** - Long prompts, good accuracy |
| Mistral-7B | 8192 | ~14GB | Slow | Very long structures, best accuracy |

**Recommendation:** Use **LLaMA-2-7B** or **Mistral-7B** for atoms dict (very long prompts).

## Expected Performance

### Galactica Models (2048 tokens)

| Model | Structure QA | Composition QA | Oxide QA |
|-------|-------------|----------------|----------|
| Galactica-125M | ~30-50% | ~35-55% | ~40-60% |
| Galactica-1.3B | ~35-60% | ~40-65% | ~45-70% |
| Random Baseline | ~25% | ~25% | ~25% |

**Note:** Actual performance depends on:
- Model's pretraining on scientific/materials data
- CIF format understanding
- Instruction following capability
- Question difficulty

### LLaMA Models

LLaMA models typically perform **worse** than Galactica on materials science tasks because:
- Not pretrained on scientific papers
- Limited exposure to chemical/materials terminology
- Poor CIF format understanding

## Performance Tips

### Batch Size Optimization

**Speed vs Memory Trade-off:**

| Batch Size | Speed | Memory (7B model) | Recommended For |
|-----------|-------|-------------------|-----------------|
| 1 | Slowest | ~14GB | Memory-constrained GPUs |
| 8 | Medium | ~16GB | Default/balanced |
| 16 | Fast | ~20GB | Good speed, moderate memory |
| 32 | **Fastest** | ~28GB | High-end GPUs (A100, RTX 4090) |

```bash
# Maximum speed (if you have memory)
python evaluation/compute_metrics_baseline.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --batch-size 32 \
    --device cuda:0

# Memory-efficient
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --batch-size 64 \
    --device cuda:0
```

**Expected Speed (12,634 samples):**
- Batch 1: ~2 hours
- Batch 8: ~20 minutes
- Batch 16: ~10 minutes ✓
- Batch 32: ~5 minutes

## Limitations

1. **Context Length**: Long structure descriptions may exceed max_length
2. **Prompt Engineering**: Fixed prompt format may not be optimal
3. **Parsing Failures**: Some models don't follow "Answer: A" format

## Future Improvements

1. **Prompt Optimization**: Test different prompt templates
2. **Few-Shot Learning**: Add example QA pairs to prompt
3. **Fine-tuning**: Fine-tune LLMs on materials QA data
4. **Structure Summarization**: Compress atoms dict for efficiency
5. **Ensemble Methods**: Combine multiple models for better accuracy

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python evaluation/compute_metrics_baseline.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --batch-size 4 \
    --device cuda:0

# Or use smaller model with larger batch
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --batch-size 32 \
    --device cuda:0

# Last resort: Use CPU (very slow)
python evaluation/compute_metrics_baseline.py \
    --model-name facebook/galactica-125m \
    --batch-size 1 \
    --device cpu
```

### Model Access

Some models require HuggingFace access:

```bash
# Login first
huggingface-cli login

# Then run evaluation
python evaluation/compute_metrics_baseline.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --device cuda:0
```

### High Parsing Failure Rate

If parsing failures > 50%:
- Model doesn't understand instruction format
- Try different model (Galactica usually better)
- Check generated outputs manually for debugging

## References

- CIF Format: https://www.iucr.org/resources/cif
- Galactica Paper: https://arxiv.org/abs/2211.09085
- Materials Project: https://materialsproject.org/
