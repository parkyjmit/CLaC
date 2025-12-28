# Evaluation Results Management Guide

This guide explains how to run evaluations and manage results for CLaC models.

## Overview

All evaluation methods now save results to JSON files organized by checkpoint name in the `outputs/` directory:

```
outputs/
├── zero_shot_retrieval/
│   └── {checkpoint_name}/
│       └── retrieval_{label}.json
├── zero_shot_qa/
│   └── {checkpoint_name}/
│       └── qa_{label}.json
├── inverse_design/
│   └── {checkpoint_name}/
│       └── inverse_design_{property}.json
└── inverse_design_dual/
    └── {checkpoint_name}/
        └── dual_inverse_design_{property1}_{property2}.json
```

## Running Evaluations

### 1. Zero-Shot Retrieval

Evaluates material-text retrieval accuracy (Top-1, Top-3, Top-10).

```bash
python evaluation/compute_metrics.py \
  --evaluation-method "zero-shot retrieval" \
  --label "text" \
  --model-ckpt path/to/checkpoint.ckpt \
  --device cuda:0 \
  --batch-size 64 \
  --graphdatatype auto
```

**Result Structure:**
```json
{
  "evaluation_method": "zero-shot retrieval",
  "label": "text",
  "overall_metrics": {
    "top1_accuracy": 0.9441,
    "top3_accuracy": 0.9836,
    "top10_accuracy": 0.9939
  },
  "per_batch_metrics": [...]
}
```

**Batch Evaluation:**
```bash
bash scripts/evaluation/zero_shot_retrieval_evaluation.sh
```

### 2. Zero-Shot QA

Evaluates question-answering accuracy.

```bash
python evaluation/compute_metrics.py \
  --evaluation-method "zero-shot QA" \
  --label "structure_question_list" \
  --model-ckpt path/to/checkpoint.ckpt \
  --device cuda:0 \
  --batch-size 64 \
  --graphdatatype auto
```

**Labels available:**
- `structure_question_list`
- `composition_question_list`
- `oxide_question_list`

**Result Structure:**
```json
{
  "evaluation_method": "zero-shot QA",
  "label": "structure_question_list",
  "accuracy": 0.85,
  "correct_answers": 850,
  "total_questions": 1000
}
```

**Batch Evaluation:**
```bash
bash scripts/evaluation/zero_shot_qa_evaluation.sh
```

### 3. Inverse Design

Evaluates single-property inverse design (query → retrieve materials with target property).

```bash
python evaluation/compute_metrics.py \
  --evaluation-method "inverse-design" \
  --property "band gap" \
  --k-values 3 10 100 \
  --model-ckpt path/to/checkpoint.ckpt \
  --device cuda:0 \
  --batch-size 64 \
  --graphdatatype auto \
  --plot
```

**Properties available:**
- `band gap`
- `density`
- `scintillation attenuation length`
- `total magnetization`
- `energy above hull`
- `formation energy per atom`

**Result Structure:**
```json
{
  "property": "band gap",
  "k_values": [3, 10, 100],
  "k3_overall_mae": 0.523,
  "k3_correlation": 0.987,
  "random_k3_overall_mae": 1.234,
  "per_query_results": [...]
}
```

**Batch Evaluation:**
```bash
bash scripts/evaluation/inverse_design_evaluation.sh
```

### 4. Dual-Property Inverse Design

Evaluates two-property inverse design (query → retrieve materials with two target properties).

```bash
python evaluation/compute_metrics_dual.py \
  --property1 "band gap" \
  --property2 "density" \
  --k-values 3 10 \
  --model-ckpt path/to/checkpoint.ckpt \
  --device cuda:0 \
  --batch-size 128 \
  --use-candidate-pool \
  --plot
```

**Using Candidate Pool:**
The candidate pool (828 samples) is optimized for dual-property evaluation:

```bash
# Create candidate pool (already done)
python tools/create_dual_candidate_pool.py

# Use in evaluation
python evaluation/compute_metrics_dual.py \
  --property1 "band gap" \
  --property2 "density" \
  --k-values 3 10 \
  --model-ckpt path/to/checkpoint.ckpt \
  --use-candidate-pool  # Enable candidate pool
```

**Result Structure:**
```json
{
  "property1": "band gap",
  "property2": "density",
  "k_values": [3, 10],
  "k3_overall_mae1": 1.259,
  "k3_overall_mae2": 0.650,
  "k3_overall_mae_combined": 0.291,
  "per_query_results": [...]
}
```

**Batch Evaluation:**
```bash
bash scripts/evaluation/run_dual_inverse_design.sh
```

## Viewing Results

### Quick Summary

View all results in a formatted table:

```bash
# All evaluations
python evaluation/summarize_results.py --eval-type all

# Specific evaluation
python evaluation/summarize_results.py --eval-type retrieval
python evaluation/summarize_results.py --eval-type qa
python evaluation/summarize_results.py --eval-type inverse
python evaluation/summarize_results.py --eval-type dual
```

### Export to CSV

Save summaries to CSV files:

```bash
python evaluation/summarize_results.py --eval-type all --output outputs/results/summary.csv
```

This creates:
- `outputs/results/summary_retrieval.csv`
- `outputs/results/summary_qa.csv`
- `outputs/results/summary_inverse.csv`
- `outputs/results/summary_dual.csv`

### Manual Inspection

```bash
# View specific result
cat outputs/zero_shot_retrieval/{checkpoint_name}/retrieval_text.json | jq '.'

# Count results
ls -1 outputs/zero_shot_retrieval/*/retrieval_*.json | wc -l

# Find best model
python evaluation/summarize_results.py --eval-type retrieval | sort -t'%' -k3 -nr
```

## Result Organization

### Directory Structure

```
CLaC-revision/
└── outputs/
    ├── zero_shot_retrieval/
    │   └── {checkpoint_name}/
    │       └── retrieval_{label}.json
    ├── zero_shot_qa/
    │   └── {checkpoint_name}/
    │       └── qa_{label}.json
    ├── inverse_design/
    │   └── {checkpoint_name}/
    │       ├── inverse_design_{property}.json
    │       └── inverse_design_{property}.svg (plot)
    ├── inverse_design_dual/
    │   └── {checkpoint_name}/
    │       ├── dual_inverse_design_{property1}_{property2}.json
    │       └── dual_inverse_design_{property1}_{property2}.svg (plot)
    └── figures/
        └── (publication-ready figures)
```

### Checkpoint Naming

Checkpoint names are automatically extracted from paths:
- `/outputs/model-name/epoch=10-step=1000.ckpt` → `model-name_epoch=10-step=1000`
- `/outputs/model-name/last.ckpt` → `model-name_last`

## Common Workflows

### Evaluate Multiple Models

```bash
# Run evaluation sweep
bash scripts/evaluation/zero_shot_retrieval_evaluation.sh
bash scripts/evaluation/zero_shot_qa_evaluation.sh
bash scripts/evaluation/inverse_design_evaluation.sh
bash scripts/evaluation/run_dual_inverse_design.sh

# Summarize all results
python evaluation/summarize_results.py --eval-type all --output outputs/results/full_summary.csv
```

### Compare Models

```python
import pandas as pd
import json
from pathlib import Path

# Load results
results = []
for json_file in Path('outputs/zero_shot_retrieval').rglob('*.json'):
    with open(json_file) as f:
        data = json.load(f)
        results.append({
            'model': json_file.parent.name,
            'top1': data['overall_metrics']['top1_accuracy'],
            'top3': data['overall_metrics']['top3_accuracy'],
            'top10': data['overall_metrics']['top10_accuracy']
        })

df = pd.DataFrame(results).sort_values('top1', ascending=False)
print(df)
```

### Track Progress

```bash
# Watch evaluation progress
watch -n 10 'python evaluation/summarize_results.py --eval-type all | tail -20'

# Count completed evaluations
echo "Retrieval: $(ls outputs/zero_shot_retrieval/*/*.json 2>/dev/null | wc -l)"
echo "QA: $(ls outputs/zero_shot_qa/*/*.json 2>/dev/null | wc -l)"
echo "Inverse: $(ls outputs/inverse_design/*/*.json 2>/dev/null | wc -l)"
echo "Dual: $(ls outputs/inverse_design_dual/*/*.json 2>/dev/null | wc -l)"
```

## Troubleshooting

### Missing Results

If results are not saved:
1. Check write permissions in current directory
2. Verify checkpoint path exists
3. Check for errors in evaluation output

### Incomplete Evaluations

If evaluation crashes:
1. Results up to the crash point are saved
2. Re-run with same arguments to continue (will overwrite)
3. Check GPU memory if running multiple evaluations

### Comparing Old Results

Old results (before JSON format) need manual migration:
```bash
# Old format: printed to stdout
# New format: saved to JSON files
# Run evaluations again to generate JSON files
```

## Tips

1. **Use `--graphdatatype auto`**: Automatically detects graph encoder type
2. **Batch evaluations**: Use provided shell scripts for multiple models
3. **Monitor progress**: Use `summarize_results.py` during long sweeps
4. **Save visualizations**: Use `--plot` flag for inverse design evaluations
5. **Candidate pool**: Use `--use-candidate-pool` for dual inverse design (6.6% of full test set, balanced sampling)

## Example: Full Evaluation Pipeline

```bash
# 1. Train model
python train.py

# 2. Run all evaluations
bash scripts/evaluation/zero_shot_retrieval_evaluation.sh
bash scripts/evaluation/zero_shot_qa_evaluation.sh
bash scripts/evaluation/inverse_design_evaluation.sh
bash scripts/evaluation/run_dual_inverse_design.sh

# 3. Summarize results
python evaluation/summarize_results.py --eval-type all --output outputs/results/model_evaluation.csv

# 4. View summary
cat outputs/results/model_evaluation_*.csv
```

## References

- Inverse Design: `evaluation/compute_metrics.py` + `scripts/evaluation/inverse_design_evaluation.sh`
- Dual Inverse Design: `evaluation/compute_metrics_dual.py` + `scripts/evaluation/run_dual_inverse_design.sh`
- Zero-Shot Retrieval: `evaluation/compute_metrics.py` + `scripts/evaluation/zero_shot_retrieval_evaluation.sh`
- Zero-Shot QA: `evaluation/compute_metrics.py` + `scripts/evaluation/zero_shot_qa_evaluation.sh`
- Results Summary: `evaluation/summarize_results.py`
