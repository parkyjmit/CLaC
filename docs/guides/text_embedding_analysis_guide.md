# Text Embedding Similarity Analysis Guide

## Overview

This guide describes how to analyze text embedding similarity across different property values using trained CLaC models.

**Note:** All commands should be run from the project root directory.

## Scripts

### 1. `evaluation/analyze_text_embedding_similarity.py`

Analyzes text embedding similarity for a single property.

**Features:**
- Generates text embeddings for different property values
- Computes pairwise cosine similarity
- Creates comprehensive visualization with 3 subplots:
  - (A) Similarity matrix heatmap
  - (B) Similarity to 3 reference values
  - (C) Similarity vs value difference
- **Supports both trained and untrained models** for comparison
- **Two embedding modes**: projection layer (trained) or raw LLM features (untrained)
- **NEW: Threshold-based analysis** for range queries (e.g., "band gap > 1 eV")

**Usage:**

```bash
# Trained model (uses projection layer)
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt <checkpoint_path> \
    --property "band gap" \
    --min-value 0.0 \
    --max-value 4.0 \
    --reference-value 2.0 \
    --device cuda:0 \
    --output-dir outputs/text_embedding_similarity

# Untrained model (uses raw LLM features)
python evaluation/analyze_text_embedding_similarity.py \
    --config config/model/clac-lite.yaml \
    --property "band gap" \
    --min-value 0.0 \
    --max-value 4.0 \
    --reference-value 2.0 \
    --device cuda:0 \
    --output-dir outputs/text_embedding_similarity/untrained \
    --no-projection
```

**Arguments:**
- `--model-ckpt`: Path to trained model checkpoint (optional, if not provided uses untrained model)
- `--config`: Path to model config file (default: config/model/clac-lite.yaml, used only when --model-ckpt is not provided)
- `--no-projection`: Use raw LLM features instead of projection layer (automatically enabled for untrained models)
- `--property`: Property to analyze (choices: band gap, formation energy per atom, density, scintillation attenuation length, total magnetization, energy above hull)
- `--min-value`: Minimum property value (default: 0.0)
- `--max-value`: Maximum property value (default: 4.0)
- `--reference-value`: Central reference value (default: middle of range)
- `--threshold-mode`: Enable threshold analysis (choices: greater_than, less_than, greater_equal, less_equal)
- `--threshold-value`: Threshold value for range-based queries (required with --threshold-mode)
- `--device`: Device to use (default: cuda)
- `--output-dir`: Output directory (default: outputs/text_embedding_similarity)

**Note:** Grid spacing is automatically determined per property to ensure clean reference values (e.g., 1.0, 2.0, 3.0) are exactly on grid points. The `--n-values` parameter has been removed in favor of this automatic spacing.

**Output:**
- Standard mode:
  - Plots: `similarity_analysis_<property>.png/svg`
  - Data: `similarity_analysis_<property>.json`
- Threshold mode:
  - Plots: `threshold_analysis_<property>_<mode>_<threshold>.png/svg`
  - Data: `threshold_similarity_<property>.json`

**JSON Output Contents:**
- Standard mode: Contains `values`, `embeddings`, `similarity_matrix`, `property_name`, `unit`
- Threshold mode: Contains `values`, `similarities`, `threshold_value`, `threshold_mode`, `threshold_text`, `above_mask`, `below_mask`, `above_similarities`, `below_similarities`

### 2. `scripts/evaluation/run_text_embedding_analysis.sh`

Batch script to analyze all 6 properties automatically.

**Usage:**

```bash
# Trained model (default usage)
bash scripts/evaluation/run_text_embedding_analysis.sh

# Trained model with custom checkpoint
bash scripts/evaluation/run_text_embedding_analysis.sh outputs/your-model/checkpoint.ckpt

# Trained model with custom checkpoint and device
bash scripts/evaluation/run_text_embedding_analysis.sh outputs/your-model/checkpoint.ckpt cuda:7

# Untrained model (baseline comparison)
bash scripts/evaluation/run_text_embedding_analysis.sh untrained cuda:0

# Untrained model with custom config
bash scripts/evaluation/run_text_embedding_analysis.sh untrained cuda:0 config/model/clac-lite.yaml
```

**Output Directory Structure:**
- Trained model: `outputs/text_embedding_similarity/<model_name>/`
- Untrained model: `outputs/text_embedding_similarity/untrained/`

**Properties Analyzed:**

1. **Band gap**: 0.0-4.0 eV (spacing=0.1, 41 values, ref=2.0)
2. **Formation energy per atom**: -3.0-1.0 eV/atom (spacing=0.1, 41 values, ref=-1.0)
3. **Density**: 2.0-10.0 g/cm³ (spacing=0.2, 41 values, ref=5.0)
4. **Scintillation attenuation length**: 0.5-20.0 cm (spacing=0.5, 40 values, ref=5.0)
5. **Total magnetization**: 0.0-50.0 μB (spacing=1.0, 51 values, ref=10.0)
6. **Energy above hull**: 0.0-1.0 eV/atom (spacing=0.02, 51 values, ref=0.2)

## Example Workflow

### Single Property Analysis

```bash
# Analyze band gap with trained model
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/your-model/checkpoint.ckpt \
    --property "band gap" \
    --min-value 0.0 \
    --max-value 4.0 \
    --reference-value 2.0 \
    --device cuda:0
# Note: Spacing is auto-set to 0.1 for band gap, generating 41 values

# Analyze band gap with untrained model (baseline)
python evaluation/analyze_text_embedding_similarity.py \
    --property "band gap" \
    --min-value 0.0 \
    --max-value 4.0 \
    --reference-value 2.0 \
    --device cuda:0 \
    --output-dir outputs/text_embedding_similarity/untrained \
    --no-projection
```

### Batch Analysis (All Properties)

```bash
# Run all properties with trained model
bash scripts/evaluation/run_text_embedding_analysis.sh \
    outputs/your-model/checkpoint.ckpt \
    cuda:7

# Run all properties with untrained model (baseline)
bash scripts/evaluation/run_text_embedding_analysis.sh untrained cuda:0
```

This will generate 6 plots (one for each property) in the respective directories:
- Trained: `outputs/text_embedding_similarity/<model_name>/`
- Untrained: `outputs/text_embedding_similarity/untrained/`

### Threshold-Based Analysis (Range Queries)

For analyzing range-based queries like "band gap > 1 eV", see the dedicated guide:

**Documentation**: `docs/guides/threshold_embedding_analysis_guide.md`

**Quick start**:
```bash
# Single threshold test
python evaluation/analyze_text_embedding_similarity.py \
    --model-ckpt outputs/model.ckpt \
    --property "band gap" \
    --min-value 0.0 --max-value 4.0 \
    --threshold-mode greater_than \
    --threshold-value 1.0 \
    --device cuda:0

# Run all predefined threshold tests
bash scripts/evaluation/run_threshold_embedding_analysis.sh outputs/model.ckpt cuda:0
```

This analyzes whether the model understands inequality constraints and can distinguish values satisfying vs. not satisfying a condition.

## Interpreting Results

### Subplot (A): Similarity Matrix

- **High values (yellow)**: Property values that produce similar text embeddings
- **Low values (purple)**: Property values with more distinct embeddings
- **Diagonal**: Always 1.0 (perfect similarity with itself)

**Key Insights:**
- Smooth gradients indicate continuous semantic representations
- Sharp transitions suggest categorical boundaries learned by the model

### Subplot (B): Similarity to Reference Values

- **Three colored lines**: Show how similarity decays from 3 reference points
- **Stars**: Mark the reference values (similarity = 1.0, always at the reference point)
- **Vertical dashed lines**: Indicate reference positions
- **Reference values**: Always clean numbers (e.g., 1.0, 2.0, 3.0) that are exactly on grid points

**Key Insights:**
- Smooth decay suggests gradual semantic transitions
- Steep decay indicates the model strongly differentiates nearby values
- Asymmetric decay reveals directional biases in representation space
- Stars are always at y=1.0 since similarity with itself is always perfect

### Subplot (C): Similarity vs Value Difference

- **Scatter points**: Individual pairwise similarities colored by magnitude
- **Red line**: Mean similarity trend
- **Red shaded area**: ±1 standard deviation

**Key Insights:**
- Linear decay: Uniform semantic spacing across property range
- Non-linear decay: Model treats certain value ranges more distinctly
- High minimum similarity (>0.5): Model recognizes shared context across all values

## Example Results

For band gap analysis (0.0-4.0 eV) with Galactica-125M:

```
Mean similarity: 0.8450
Min similarity:  0.5083 (between 0.0 and 4.0 eV)
Max similarity:  1.0000 (self-similarity)
```

**Interpretation:**
- Very high mean similarity (0.845) indicates the model represents all band gap values as related materials properties
- Even extreme values (0.0 vs 4.0 eV) maintain >50% similarity, suggesting shared semantic context
- Smooth similarity decay indicates continuous rather than categorical representations

## Publication-Ready Figures

The generated figures are publication-ready (300 DPI, vector graphics available).

**Suggested Caption:**

> **Figure X: Text embedding similarity analysis for band gap property values.** (A) Similarity matrix showing pairwise cosine similarities between text embeddings of materials with different band gap values (0.0-4.0 eV). The high similarity values (mean=0.845) indicate that the model learns smooth continuous representations across the property space. (B) Similarity profiles relative to three reference values (1.0, 2.0, and 3.0 eV, shown in blue, orange, and green). Each curve demonstrates that text embeddings maintain high similarity to nearby property values and gradually decrease with distance. (C) Relationship between absolute band gap difference and embedding similarity. The red line shows the mean trend with ±1 standard deviation, revealing a gradual decay in similarity as the property value difference increases.

## Customization

### Custom Property Ranges

Edit the property configuration in `scripts/evaluation/run_text_embedding_analysis.sh`:

```bash
PROPERTIES=(
    "band gap|0.0|6.0|3.0"  # Extended range
    "your custom property|min|max|reference"
)
```

### Custom Grid Spacing

Edit the `property_spacing` dictionary in `evaluation/analyze_text_embedding_similarity.py`:

```python
property_spacing = {
    'band gap': 0.1,  # 0.0, 0.1, 0.2, ..., 1.0, ..., 4.0
    'your property': 0.05,  # Finer spacing for more detailed analysis
}
```

**Design principle:** Choose spacing so that clean reference values (integers or 0.5 increments) fall exactly on grid points. This ensures:
- Reference similarity is exactly 1.0 (no interpolation needed)
- Clean legend labels (e.g., "Ref: 1.0" instead of "Ref: 0.98")
- Accurate visual alignment of markers and curves

### Trained vs Untrained Comparison

Compare trained model against untrained baseline:

```bash
# Step 1: Analyze untrained model (baseline)
bash scripts/evaluation/run_text_embedding_analysis.sh untrained cuda:0
# Results saved to: outputs/text_embedding_similarity/untrained/

# Step 2: Analyze trained model
bash scripts/evaluation/run_text_embedding_analysis.sh outputs/your-model/checkpoint.ckpt cuda:0
# Results saved to: outputs/text_embedding_similarity/your-model/

# Step 3: Compare results visually
# Open both directories and compare the similarity matrices
```

**Expected Differences:**
- **Untrained model**: Very high similarity (>0.99) across all property values
  - Raw LLM embeddings are nearly identical due to similar sentence structure
  - Model hasn't learned to differentiate property values
- **Trained model**: Lower, more varied similarity (~0.5-0.9)
  - Contrastive learning creates distinct embeddings for different values
  - Smooth gradients indicate learned continuous property representations

### Multiple Models Comparison

Run the analysis for different models and compare results:

```bash
# Model 1
bash scripts/evaluation/run_text_embedding_analysis.sh outputs/model1/checkpoint.ckpt cuda:0
# Results in: outputs/text_embedding_similarity/model1/

# Model 2
bash scripts/evaluation/run_text_embedding_analysis.sh outputs/model2/checkpoint.ckpt cuda:0
# Results in: outputs/text_embedding_similarity/model2/

# Note: Each model's results are automatically saved to separate directories
```

### Using JSON Output for Custom Plotting

All analysis results are automatically saved to JSON files for programmatic access:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load standard similarity analysis
with open('outputs/text_embedding_similarity/my-model/similarity_analysis_band_gap.json', 'r') as f:
    data = json.load(f)

values = np.array(data['values'])
similarity_matrix = np.array(data['similarity_matrix'])
property_name = data['property_name']
unit = data['unit']

# Custom plotting example: Extract diagonal similarities
diagonal_sims = np.diag(similarity_matrix, k=1)  # Similarity to next value
plt.plot(values[:-1], diagonal_sims)
plt.xlabel(f'{property_name} ({unit})')
plt.ylabel('Similarity to next value')
plt.title('Sequential Value Similarity')
plt.savefig('custom_analysis.png')
```

**Threshold analysis JSON:**
```python
# Load threshold similarity analysis
with open('outputs/text_embedding_similarity/my-model/threshold_similarity_band_gap.json', 'r') as f:
    data = json.load(f)

values = np.array(data['values'])
similarities = np.array(data['similarities'])
above_mask = np.array(data['above_mask'])
threshold_value = data['threshold_value']

# Custom analysis: Compare distributions
satisfying = similarities[above_mask]
not_satisfying = similarities[~above_mask]
print(f"Mean similarity for values > {threshold_value}: {satisfying.mean():.3f}")
print(f"Mean similarity for values <= {threshold_value}: {not_satisfying.mean():.3f}")
```

**JSON Fields Reference:**

Standard mode (`similarity_analysis_<property>.json`):
- `values`: Array of property values tested
- `embeddings`: Text embeddings (N × embedding_dim)
- `similarity_matrix`: Pairwise cosine similarities (N × N)
- `property_name`: Property name string
- `unit`: Property unit string

Threshold mode (`threshold_similarity_<property>.json`):
- `values`: Array of property values tested
- `similarities`: Similarity to threshold query (length N)
- `threshold_value`: Threshold value used
- `threshold_mode`: Mode (e.g., "greater_than")
- `threshold_text`: Human-readable query text
- `above_mask`: Boolean mask for values satisfying condition
- `below_mask`: Boolean mask for values not satisfying condition
- `above_similarities`: Similarities for satisfying values
- `below_similarities`: Similarities for non-satisfying values

## Troubleshooting

### Out of Memory

Use coarser grid spacing in `evaluation/analyze_text_embedding_similarity.py`:
```python
property_spacing = {
    'band gap': 0.2,  # Coarser: 21 values instead of 41
}
```

### Model Loading Errors

Ensure the checkpoint is compatible with CLaCLite architecture. Check for warnings about missing keys.

### Slow Generation

- Use GPU: `--device cuda:0`
- Use coarser grid spacing (fewer values)
- Use smaller text encoder models

## Technical Details

### Text Embedding Generation

**Trained Model (default):**
1. Template text generated: `"material with band gap of {value:.2f} eV"`
2. Tokenized using model's tokenizer (max length: 128)
3. Encoded through text encoder
4. Projected through `model.loss.global_d.text_block` (learned projection layer)
5. L2-normalized for cosine similarity

**Untrained Model (with `--no-projection`):**
1. Template text generated: `"material with band gap of {value:.2f} eV"`
2. Tokenized using model's tokenizer (max length: 128)
3. Encoded through text encoder (pretrained LLM only)
4. L2-normalized for cosine similarity (no projection layer)

**Why untrained models show high similarity:**
- Similar sentence structure: "material with [property] of [value] [unit]"
- Most tokens are identical across different values (only 1-2 digits differ)
- Mean pooling over all tokens dilutes the small differences
- Pretrained LLM hasn't been trained to differentiate these specific property values
- Contrastive training is needed to learn distinct representations

### Similarity Computation

- **Metric**: Cosine similarity
- **Formula**: `similarity = (u · v) / (||u|| × ||v||)`
- **Range**: [-1, 1], but typically [0.5, 1.0] for related properties

### Performance

- **Time per property**: ~5-10 seconds (40-50 values, GPU)
- **Memory**: ~2GB GPU memory
- **Total time (6 properties)**: ~1 minute
- **Grid design**: Property-specific spacing ensures clean reference values on grid points

### Grid Spacing Details

| Property | Spacing | # Values | Clean References |
|----------|---------|----------|------------------|
| Band gap | 0.1 eV | 41 | 1.0, 2.0, 3.0 eV |
| Formation energy | 0.1 eV/atom | 41 | -2.0, -1.0, 0.0 eV/atom |
| Density | 0.2 g/cm³ | 41 | 3.0, 5.0, 8.0 g/cm³ |
| Scintillation | 0.5 cm | 40 | 5.0, 10.0, 15.0 cm |
| Magnetization | 1.0 μB | 51 | 10.0, 25.0, 40.0 μB |
| Energy above hull | 0.02 eV/atom | 51 | 0.2, 0.5, 0.8 eV/atom |

## Related Scripts

- `evaluation/plot_pareto_v2.py`: Pareto front analysis for retrieval vs inverse design
- `evaluation/compute_metrics.py`: Comprehensive evaluation metrics
- `scripts/evaluation/run_threshold_embedding_analysis.sh`: Batch threshold analysis
- `docs/guides/threshold_embedding_analysis_guide.md`: Threshold analysis guide
