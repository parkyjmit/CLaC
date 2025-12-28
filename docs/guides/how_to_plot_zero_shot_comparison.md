# How to Plot Zero-Shot Evaluation Comparisons

## Overview

`evaluation/plot_zero_shot_comparison.py` creates publication-quality comparison plots for zero-shot evaluation results with bootstrap confidence intervals.

**Note:** All commands should be run from the project root directory.

## Features

- ✅ Compare multiple models side-by-side
- ✅ Automatic model discovery
- ✅ Bootstrap confidence intervals as error bars
- ✅ Color-coded baselines (Trained/Untrained/Random)
- ✅ Professional formatting for papers
- ✅ Supports both retrieval and QA tasks

## Quick Start

### 1. Compare All Available Results (Auto-discover)

**Zero-Shot Retrieval:**
```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --output outputs/figures/retrieval_comparison.png
```

**Zero-Shot QA:**
```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label structure_question_list \
    --auto-discover \
    --output outputs/figures/qa_structure_comparison.png
```

### 2. Compare Specific Models

**Select models manually:**
```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --models \
        "PaperData-painn-infonce-galactica-125m-vTrue-tTrue-5epoch_last" \
        "SynData-painn-infonce-galactica-125m-vTrue-tTrue-5epoch_last" \
        "untrained_baseline" \
    --output outputs/figures/ablation_study.svg
```

### 3. Exclude Random Baseline

```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --no-random \
    --output outputs/figures/models_only.png
```

## Usage Examples

### Example 1: Intramodal Loss Ablation Study

Compare models with different intramodal loss configurations:

```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --models \
        "SynData-painn-infonce-galactica-125m-vTrue-tTrue-5epoch_last" \
        "SynData-painn-infonce-galactica-125m-vFalse-tFalse-5epoch_last" \
        "SynData-painn-infonce-galactica-125m-vFalse-tTrue-5epoch_last" \
        "SynData-painn-infonce-galactica-125m-vTrue-tFalse-5epoch_last" \
        "untrained_baseline" \
    --output outputs/figures/intramodal_ablation.png
```

### Example 2: Compare All QA Types

```bash
# Structure questions
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label structure_question_list \
    --auto-discover \
    --output outputs/figures/qa_structure.png

# Composition questions
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label composition_question_list \
    --auto-discover \
    --output outputs/figures/qa_composition.png

# Oxide questions
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label oxide_question_list \
    --auto-discover \
    --output outputs/figures/qa_oxide.png
```
### Example 3: Custom Figure Size

For paper figures with specific dimensions:

```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --figsize 12 5 \
    --output outputs/figures/wide_comparison.pdf
```

### Example 4: Batch Processing Script

Create a script to generate all comparison plots:

```bash
#!/bin/bash
# generate_all_plots.sh

OUTPUT_DIR="outputs/figures/zero_shot_comparisons"
mkdir -p $OUTPUT_DIR

# Retrieval comparison
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --output $OUTPUT_DIR/retrieval_all.png

# QA comparisons (all labels)
for label in structure_question_list composition_question_list oxide_question_list; do
    python evaluation/plot_zero_shot_comparison.py \
        --evaluation-type qa \
        --label $label \
        --auto-discover \
        --output $OUTPUT_DIR/qa_${label}.png
done

echo "All plots generated in $OUTPUT_DIR/"
```

## Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--evaluation-type` | **Required** | `retrieval` or `qa` |
| `--models` | List | Model names to compare (directory names) |
| `--auto-discover` | Flag | Auto-discover all available results |
| `--label` | String | QA label (for `--evaluation-type qa`) |
| `--output` | String | Output file path (`.png`, `.pdf`, `.svg`) |
| `--no-random` | Flag | Exclude random baseline from plot |
| `--figsize` | Float Float | Figure size (width height) in inches |

## Output Format

### Zero-Shot Retrieval Plot

Creates a **3-panel plot** with:
- **Left panel**: Top-1 accuracy
- **Middle panel**: Top-3 accuracy
- **Right panel**: Top-10 accuracy

Each bar shows:
- Mean accuracy (bar height)
- 95% confidence interval (error bars)
- Percentage label on top

### Zero-Shot QA Plot

Creates a **single bar plot** with:
- Accuracy for each model
- 95% confidence interval (error bars)
- Percentage label on top

### Color Scheme

- **Teal**: Trained models
- **Red**: Untrained baseline
- **Gray**: Random baseline

## Expected Directory Structure

### For Retrieval

```
outputs/zero_shot_retrieval/
├── model1_last/
│   └── retrieval_text.json
├── model2_last/
│   └── retrieval_text.json
└── untrained_baseline/
    └── retrieval_text.json
```

### For QA

```
outputs/zero_shot_qa/
├── model1_last/
│   ├── qa_structure_question_list.json
│   ├── qa_composition_question_list.json
│   └── qa_oxide_question_list.json
└── untrained_baseline/
    ├── qa_structure_question_list.json
    ├── qa_composition_question_list.json
    └── qa_oxide_question_list.json
```

## JSON Format Requirements

The script expects JSON files with the following structure:

### Retrieval Results

```json
{
  "overall_metrics": {
    "top1_accuracy": {
      "mean": 0.8523,
      "std": 0.0045,
      "ci_lower": 0.8435,
      "ci_upper": 0.8609
    },
    "top3_accuracy": { ... },
    "top10_accuracy": { ... }
  }
}
```

### QA Results

```json
{
  "accuracy": {
    "mean": 0.7234,
    "std": 0.0052,
    "ci_lower": 0.7132,
    "ci_upper": 0.7336
  }
}
```

## Troubleshooting

### "Result file not found"

**Problem:** Model directory or JSON file doesn't exist.

**Solution:**
```bash
# Check available results
ls outputs/zero_shot_retrieval/
ls outputs/zero_shot_qa/

# Run evaluation first if results are missing
bash scripts/evaluation/zero_shot_retrieval_evaluation.sh
bash scripts/evaluation/zero_shot_qa_evaluation.sh
```

### "No valid results found"

**Problem:** All specified models have missing result files.

**Solution:** Use `--auto-discover` to see what's available:
```bash
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --output test.png
```

### Model names are too long

**Problem:** Long model names overlap on x-axis.

**Solutions:**
1. Use `--figsize` to make plot wider:
   ```bash
   --figsize 14 5
   ```

2. The script automatically shortens names > 40 characters

3. Rename model directories to shorter names

## Integration with Paper Workflow

### Step 1: Run All Evaluations

```bash
bash scripts/evaluation/zero_shot_retrieval_evaluation.sh
bash scripts/evaluation/zero_shot_qa_evaluation.sh
```

### Step 2: Generate Comparison Plots

```bash
# Main results figure (for paper)
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --models \
        "best_model_last" \
        "untrained_baseline" \
    --output outputs/figures/main_results.pdf \
    --figsize 8 3

# Ablation study figure (for paper)
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --models \
        "full_model_last" \
        "no_visual_intra_last" \
        "no_text_intra_last" \
        "no_intra_last" \
    --output outputs/figures/ablation.pdf \
    --figsize 10 4
```

### Step 3: Export for LaTeX

Use PDF or SVG format for LaTeX:

```bash
--output outputs/figures/results.pdf  # For LaTeX with pdflatex
--output outputs/figures/results.svg  # For conversion to EPS
```

### Step 4: Add to Paper

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\textwidth]{outputs/figures/main_results.pdf}
\caption{Zero-shot retrieval performance comparison. Error bars represent 95\% bootstrap confidence intervals (B=1,000).}
\label{fig:main_results}
\end{figure}
```

## Tips for Paper Figures

1. **Use PDF format** for LaTeX papers:
   ```bash
   --output figure.pdf
   ```

2. **Adjust figure size** for column width:
   - Single column: `--figsize 6 4`
   - Double column: `--figsize 12 4`

3. **Remove random baseline** for cleaner comparison:
   ```bash
   --no-random
   ```

4. **Use consistent color scheme** across all figures (automatic)

5. **Font size** is optimized for 300 DPI output (automatic)

## Advanced: Customize Plot Appearance

Edit the script to customize:

- **Colors**: Modify `colors` dictionary (line ~165)
- **Font sizes**: Modify `mpl.rc('font', size=X)` (line ~26)
- **Bar width**: Modify `width` variable (line ~180)
- **Legend position**: Modify `loc` parameter (line ~260)

## Related Scripts

- `scripts/evaluation/zero_shot_retrieval_evaluation.sh` - Run retrieval evaluations
- `scripts/evaluation/zero_shot_qa_evaluation.sh` - Run QA evaluations
- `evaluation/compute_metrics.py` - Core evaluation script
- `docs/evaluation/evaluation_methodology_for_paper.md` - Paper writing guide

## Questions?

For issues or questions, check:
1. This guide
2. Script help: `python evaluation/plot_zero_shot_comparison.py --help`
3. Example outputs in `outputs/figures/` directory
