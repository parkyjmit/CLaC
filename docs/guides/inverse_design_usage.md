# Inverse Materials Design Evaluation

## Overview

This evaluation measures CLaC's ability to retrieve materials with specific property values based on natural language queries.

**Task**: Given a query like "material with bandgap around 2.0 eV", retrieve materials and measure how close their actual bandgap values are to the target (2.0 eV).

**Note:** All commands should be run from the project root directory.

## Usage

### Basic Command

```bash
python evaluation/compute_metrics.py \
  --evaluation-method inverse-design \
  --property "band gap" \
  --k-values 10 50 100 \
  --model-ckpt path/to/checkpoint.ckpt \
  --device cuda \
  --batch-size 64
```

### Arguments

**Required:**
- `--evaluation-method inverse-design`: Enable inverse design evaluation
- `--model-ckpt`: Path to CLaC checkpoint

**Inverse Design Specific:**
- `--property`: Property to evaluate (choices: `band gap`, `formation energy per atom`, `density`)
- `--k-values`: List of K values for top-K retrieval (default: `10 50 100`)
- `--output-dir`: Directory to save results (default: `outputs/inverse_design`)
- `--plot`: Generate visualization plots (default: True)

**General:**
- `--data-path`: Path to data files without suffix (default: `datafiles/mp_3d_2020_gpt_narratives`)
- `--device`: Device to use (`cpu` or `cuda`)
- `--batch-size`: Batch size for encoding (default: 64)
- `--llm`: Tokenizer model (default: `facebook/galactica-125m`)
- `--graphdatatype`: Graph data type (`torch_geometric` or `orb`)

## Examples

### 1. Bandgap Inverse Design

```bash
python evaluation/compute_metrics.py \
  --evaluation-method inverse-design \
  --property "band gap" \
  --k-values 10 50 100 \
  --model-ckpt outputs/best_model.ckpt \
  --device cuda \
  --output-dir outputs/inverse_design/bandgap
```

**Query examples:**
- "material with bandgap around 0.5 eV"
- "material with bandgap around 1.0 eV"
- ...
- "material with bandgap around 4.0 eV"

### 2. Formation Energy Inverse Design

```bash
python evaluation/compute_metrics.py \
  --evaluation-method inverse-design \
  --property "formation energy per atom" \
  --k-values 50 100 \
  --model-ckpt outputs/best_model.ckpt \
  --device cuda
```

### 3. Density Inverse Design

```bash
python evaluation/compute_metrics.py \
  --evaluation-method inverse-design \
  --property "density" \
  --k-values 10 50 \
  --model-ckpt outputs/best_model.ckpt \
  --device cuda
```

## Output

### 1. JSON Results

Saved to: `{output_dir}/inverse_design_{property}.json`

```json
{
  "query_targets": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
  "k_values": [10, 50, 100],
  "property": "band gap",
  "per_query_results": [
    {
      "target": 1.5,
      "query": "material with bandgap around 1.5 eV",
      "k10_mean": 1.62,
      "k10_std": 0.35,
      "k10_mae": 0.12,
      "k50_mean": 1.58,
      "k50_std": 0.42,
      "k50_mae": 0.08,
      ...
    }
  ],
  "k10_overall_mae": 0.15,
  "k50_overall_mae": 0.12,
  "k100_overall_mae": 0.10,
  "k10_correlation": 0.95,
  "k50_correlation": 0.97,
  "k100_correlation": 0.98,
  "random_baseline": [...],
  "random_k10_overall_mae": 1.25,
  ...
}
```

### 2. Visualization Plot

Saved to: `{output_dir}/inverse_design_{property}.png`

**Features:**
- X-axis: Query target value
- Y-axis: Retrieved mean value
- Diagonal line (y=x): Perfect alignment
- Blue points: CLaC results (with error bars)
- Gray crosses: Random baseline
- Subplots for each K value

### 3. Console Output

```
================================================================================
INVERSE DESIGN RESULTS - band gap
================================================================================

K=10:
  CLaC MAE:    0.1523 eV
  Random MAE:  1.2456 eV
  Correlation: 0.9512

K=50:
  CLaC MAE:    0.1201 eV
  Random MAE:  1.1834 eV
  Correlation: 0.9687

K=100:
  CLaC MAE:    0.1045 eV
  Random MAE:  1.1523 eV
  Correlation: 0.9754
================================================================================
```

## Evaluation Metrics

### 1. Mean Absolute Error (MAE)
- **Definition**: Average |retrieved_mean - query_target|
- **Lower is better**
- Measures alignment between query and retrieved materials

### 2. Correlation
- **Definition**: Pearson correlation between query targets and retrieved means
- **Range**: -1 to 1 (higher is better)
- **Ideal**: 1.0 (perfect linear relationship)

### 3. Standard Deviation
- **Definition**: Spread of retrieved property values
- Indicates consistency of retrieval
- Lower std means tighter clustering around the target

## Baselines

### Random Baseline
- Randomly selects K materials from test set
- Computes mean property value
- Shows lower bound performance

### Future: Text-Only Baseline
- Use pretrained text encoder (MatSciBERT) without graph pretraining
- Compare with CLaC to show benefit of joint training

## Implementation Details

### Query Templates

Each property has a natural language template:

```python
property_config = {
    'band gap': {
        'targets': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'template': 'material with bandgap around {value} eV',
        'unit': 'eV'
    },
    'formation energy per atom': {
        'targets': [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        'template': 'material with formation energy around {value} eV per atom',
        'unit': 'eV/atom'
    },
    'density': {
        'targets': [2.0, 4.0, 6.0, 8.0, 10.0],
        'template': 'material with density around {value} g/cm³',
        'unit': 'g/cm³'
    }
}
```

### Workflow

1. **Encode all test materials** (graph embeddings)
2. **For each query target:**
   - Generate query text from template
   - Encode query text
   - Compute similarity with all materials
   - Retrieve top-K materials
   - Calculate mean/std of retrieved property values
3. **Compute metrics** (MAE, correlation)
4. **Compare with baselines**
5. **Generate visualizations**

## Tips

### Performance Optimization
- Use larger `--batch-size` (64-128) for faster encoding
- Use `--device cuda` if available
- Materials are encoded once and cached in memory

### Customization
- Modify `property_config` in `evaluation/compute_metrics.py` to add new properties
- Change query templates to test robustness
- Add more target values for finer-grained evaluation

### Troubleshooting
- If OOM: Reduce `--batch-size`
- If slow: Increase `--batch-size` or use GPU
- Check data path matches your dataset location
