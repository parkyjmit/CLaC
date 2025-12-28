# Threshold-Based Text Embedding Similarity Analysis

## Overview

This guide explains how to analyze text embedding similarity for **range-based queries** (e.g., "band gap greater than 1 eV") using the CLaC model. This helps evaluate whether the model understands inequality constraints and can distinguish between values that satisfy vs. don't satisfy a threshold condition.

**Note:** All commands should be run from the project root directory.

## Key Concept

Instead of comparing specific values (e.g., "1.0 eV" vs "2.0 eV"), threshold analysis compares:
- **Threshold query**: "material with band gap greater than 1 eV"
- **Specific values**: "material with band gap of 0.5 eV", "material with band gap of 1.5 eV", etc.

### Expected Behavior

For a **well-trained model**:
- Values **satisfying** the condition (e.g., > 1 eV) should have **higher similarity** to the query
- Values **not satisfying** the condition (e.g., < 1 eV) should have **lower similarity**
- Clear **separation** in similarity distributions between the two groups

For an **untrained model**:
- Minimal or no separation between groups
- Random or semantically-based similarity (from pretrained LLM only)

## Quick Start

### Run All Threshold Tests

```bash
# Trained model
bash scripts/visualization/run_threshold_embedding_analysis.sh outputs/my-model/checkpoint.ckpt cuda:0

# Untrained model (baseline)
bash scripts/visualization/run_threshold_embedding_analysis.sh untrained cuda:0
```

This runs threshold tests for all properties with predefined thresholds.

### Run Single Threshold Test

```bash
# Test: "band gap > 1 eV"
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/my-model/checkpoint.ckpt \
    --property "band gap" \
    --min-value 0.0 \
    --max-value 4.0 \
    --threshold-mode greater_than \
    --threshold-value 1.0 \
    --device cuda:0 \
    --output-dir outputs/text_embedding_similarity/my-model/threshold
```

## Threshold Modes

Four threshold modes are supported:

| Mode | Query Template | Example | Condition |
|------|----------------|---------|-----------|
| `greater_than` | "greater than X" | "band gap greater than 1.0 eV" | values > X |
| `less_than` | "less than X" | "band gap less than 1.0 eV" | values < X |
| `greater_equal` | "at least X" | "band gap at least 1.0 eV" | values >= X |
| `less_equal` | "at most X" | "band gap at most 1.0 eV" | values <= X |

## Output Visualizations

The analysis generates a 3-panel figure:

### Panel 1: Similarity vs Value
- **X-axis**: Property value
- **Y-axis**: Similarity to threshold query
- **Orange points**: Values satisfying condition
- **Blue points**: Values not satisfying condition
- **Red dashed line**: Threshold value

**What to look for**: Clear increase/decrease in similarity at the threshold

### Panel 2: Distribution Comparison
- Histogram of similarities for both groups
- **Orange**: Values satisfying condition
- **Blue**: Values not satisfying condition
- Dashed lines show mean similarity for each group

**What to look for**: Separation between distributions (orange shifted right for "greater than")

### Panel 3: Box Plot Comparison
- Side-by-side box plots
- Red diamonds indicate mean values
- Black line shows median

**What to look for**:
- Higher mean/median for satisfying group
- Minimal overlap between boxes

## Example Use Cases

### 1. Band Gap Semiconductor Testing

```bash
# Test: Find semiconductors with band gap > 1.0 eV (excludes metals/near-metals)
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/model.ckpt \
    --property "band gap" \
    --min-value 0.0 --max-value 4.0 \
    --threshold-mode greater_than \
    --threshold-value 1.0 \
    --device cuda:0
```

**Expected**: Higher similarity for values > 1.0 eV (semiconductors/insulators)

### 2. Low-Density Materials

```bash
# Test: Lightweight materials with density < 5.0 g/cm³
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/model.ckpt \
    --property "density" \
    --min-value 2.0 --max-value 10.0 \
    --threshold-mode less_than \
    --threshold-value 5.0 \
    --device cuda:0
```

**Expected**: Higher similarity for values < 5.0 g/cm³

### 3. Thermodynamically Stable Materials

```bash
# Test: Stable materials with energy above hull < 0.1 eV/atom
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/model.ckpt \
    --property "energy above hull" \
    --min-value 0.0 --max-value 1.0 \
    --threshold-mode less_than \
    --threshold-value 0.1 \
    --device cuda:0
```

**Expected**: Higher similarity for values < 0.1 eV/atom

## Interpreting Results

### Quantitative Metrics

The script prints:
```
[Results] Values satisfying condition (25): mean similarity = 0.8542 ± 0.0234
[Results] Values not satisfying condition (25): mean similarity = 0.7123 ± 0.0412
[Results] Difference in mean similarity: 0.1419
```

**Key metric**: Difference in mean similarity
- **Trained model**: Typically 0.05-0.20 (larger is better)
- **Untrained model**: Typically -0.02 to 0.02 (random)

### Visual Patterns

#### Well-Trained Model
- **Panel 1**: Clear step/slope at threshold
- **Panel 2**: Separated distributions (orange peak right of blue peak for "greater than")
- **Panel 3**: Minimal box overlap, higher mean for satisfying group

#### Poorly-Trained or Untrained Model
- **Panel 1**: No clear pattern at threshold
- **Panel 2**: Overlapping distributions
- **Panel 3**: Similar means, large overlap

## Comparison: Threshold vs Standard Analysis

| Analysis Type | Query | Use Case |
|---------------|-------|----------|
| **Standard** | "material with band gap of 2.0 eV" | Exact value matching, property space smoothness |
| **Threshold** | "material with band gap > 2.0 eV" | Range understanding, inequality reasoning |

Both are complementary:
- **Standard analysis** shows if the model learns smooth property representations
- **Threshold analysis** shows if the model understands range constraints

## Batch Analysis with Script

The `scripts/visualization/run_threshold_embedding_analysis.sh` script runs predefined tests:

```bash
# Trained model
bash scripts/visualization/run_threshold_embedding_analysis.sh outputs/model.ckpt cuda:0
```

**Default tests**:
1. Band gap: > 1.0 eV, < 1.0 eV, > 2.0 eV
2. Formation energy: > -1.0 eV/atom, < -1.0 eV/atom
3. Density: > 5.0 g/cm³, < 5.0 g/cm³
4. Scintillation: > 5.0 cm, > 10.0 cm
5. Magnetization: > 1.0 μB, > 10.0 μB
6. Energy above hull: > 0.1 eV/atom, < 0.1 eV/atom

**Output**: `outputs/text_embedding_similarity/<model_name>/threshold_analysis/`

## Advanced Options

### Untrained Model Baseline

```bash
# Use untrained model with raw LLM features
python evaluation/analyze_text_embedding_similarity.py \
    --config config/model/clac-lite.yaml \
    --property "band gap" \
    --min-value 0.0 --max-value 4.0 \
    --threshold-mode greater_than \
    --threshold-value 1.0 \
    --device cuda:0 \
    --no-projection
```

This uses:
- Random CLaC projection weights (not trained)
- Raw Galactica LLM features (pretrained)
- Useful baseline to measure training impact

### Custom Property Range

```bash
# Focus on narrow band gap range (1.0-2.0 eV) with threshold at 1.5 eV
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/model.ckpt \
    --property "band gap" \
    --min-value 1.0 \
    --max-value 2.0 \
    --threshold-mode greater_than \
    --threshold-value 1.5 \
    --device cuda:0
```

## Technical Details

### How It Works

1. **Generate threshold query embedding**:
   ```python
   query = "material with band gap greater than 1.0 eV"
   query_embedding = model.encode_text(query)
   ```

2. **Generate value embeddings**:
   ```python
   for value in [0.0, 0.1, 0.2, ..., 4.0]:
       text = f"material with band gap of {value} eV"
       value_embeddings.append(model.encode_text(text))
   ```

3. **Compute similarity**:
   ```python
   similarities = cosine_similarity(query_embedding, value_embeddings)
   ```

4. **Split by condition**:
   ```python
   satisfies = values > threshold  # for "greater_than"
   does_not = values <= threshold
   ```

5. **Analyze distributions**:
   ```python
   mean_diff = similarities[satisfies].mean() - similarities[does_not].mean()
   ```

### Template Variations

All supported properties have threshold templates:

```python
# Band gap
"material with band gap greater than {value} eV"
"material with band gap less than {value} eV"
"material with band gap at least {value} eV"
"material with band gap at most {value} eV"

# Formation energy
"material with formation energy greater than {value} eV/atom"
# ... (similar variations)

# Density, scintillation, magnetization, energy above hull
# ... (all have similar 4 template variations)
```

## Troubleshooting

### Issue: Small or negative similarity difference

**Possible causes**:
1. Model not trained sufficiently
2. Property not well-represented in training data
3. Threshold too extreme (e.g., testing 0.01 eV when training data has 0.5-4.0 eV)

**Solutions**:
- Train longer or with better hyperparameters
- Use threshold values within training data range
- Compare with untrained baseline to verify it's a training issue

### Issue: Untrained model shows separation

**Possible causes**:
1. Pretrained LLM (Galactica) has semantic understanding of numbers
2. Threshold happens to align with semantic boundaries

**This is expected**: Galactica may have weak numerical reasoning from pretraining. The key is that **trained models should show much larger separation**.

## Related Documentation

- [Text Embedding Analysis Guide](text_embedding_analysis_guide.md): Standard pairwise similarity analysis
- [Inverse Design Usage](inverse_design_usage.md): Using CLaC for materials retrieval
- `evaluation/analyze_text_embedding_similarity.py`: Main analysis script
- `scripts/visualization/run_threshold_embedding_analysis.sh`: Batch script for threshold analysis

## Citation

If you use threshold-based analysis in your research, please cite:

```
(Add citation once paper is published)
```
