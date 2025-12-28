# Text-Only Inverse Design Evaluation

## Overview

**Purpose**: Evaluate whether the CLaC model learned meaningful application-property relationships from synthetic narratives by querying with **application descriptions only** (no explicit property values).

**Research Question**: Can the model retrieve materials with appropriate property ranges when given only application-specific text like *"material suitable for solar cell applications"*?

This is a **synthetic narrative credibility test** - if the model performs well, it validates that:
1. Synthetic narratives accurately encode application-property relationships
2. The model learned semantic understanding, not just formula memorization
3. Application-aware templates are scientifically grounded

**Note:** All commands should be run from the project root directory.

---

## Quick Start

### Single Property Evaluation

```bash
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --k-values 10 50 100 \
    --model-ckpt outputs/your_model/last.ckpt \
    --device cuda:0 \
    --plot
```

### All Properties

```bash
bash scripts/evaluation/text_only_inverse_design_evaluation.sh [MODEL_CKPT] [DATA_PATH] [DEVICE]

# Example:
bash scripts/evaluation/text_only_inverse_design_evaluation.sh \
    outputs/my_experiment/epoch=49.ckpt \
    datafiles/mp_3d_2020_gpt_narratives \
    cuda:0
```

---

## Methodology

### 1. Application Query Templates

Unlike standard inverse design where queries specify exact values:
- **Standard**: `"material with bandgap around 1.5 eV suitable for solar cells"`
- **Text-Only**: `"material suitable for solar cell and photovoltaic applications"` ✓

### 2. Expected Property Ranges

Each application has a scientifically-defined expected range:

| Application | Expected Band Gap (eV) | Basis |
|-------------|------------------------|-------|
| Electrical conductors | 0.0 - 0.1 | Metallic/semimetallic |
| Infrared detectors | 0.1 - 1.2 | Photon wavelength λ = 1-10 μm |
| Solar cells | 1.1 - 1.8 | Shockley-Queisser limit |
| LEDs | 1.8 - 3.0 | Visible light (400-700 nm) |
| UV detectors | 3.0+ | UV wavelength < 400 nm |

### 3. Evaluation Metrics

**Precision**: What fraction of retrieved materials fall within the expected range?
```
Precision = (Retrieved ∩ Expected Range) / Retrieved
```

**Recall**: What fraction of all materials in the expected range were retrieved?
```
Recall = (Retrieved ∩ Expected Range) / Total in Range
```

**Distribution Statistics**:
- Mean, median, std of retrieved property values
- Comparison to expected range center

---

## Interpreting Results

### High Precision (> 70%)
✅ **Good**: Model successfully learned application → property mapping
✅ Synthetic narratives are credible
✅ Model relies on semantic understanding

### Low Precision (< 30%)
❌ **Poor**: Model struggles to connect applications to properties
❌ Possible causes:
- Formula memorization (not semantic learning)
- Weak synthetic narrative training
- Template mismatch with actual data

### Precision ≈ Random Baseline
⚠️ **Warning**: Model not learning meaningful relationships
⚠️ Consider:
- Checking template definitions
- Increasing synthetic data quality
- Using formula replacement during training

### Precision vs. K Trend

| Trend | Interpretation |
|-------|----------------|
| Precision stable or increases with K | Good generalization, consistent understanding |
| Precision drops sharply with K | Only top results accurate, poor diversity |
| Precision < Random for large K | Model biased toward specific property ranges |

---

## Example Output

```
TEXT-ONLY INVERSE DESIGN RESULTS - band gap
================================================================================

Application: solar_cells
Query: 'material ideal for solar cell and photovoltaic applications'
Expected range: 1.1 - 1.8 eV
Materials in range: 1245
  K= 10: Precision=0.800, Recall=0.006, Mean=1.52 eV
  K= 50: Precision=0.720, Recall=0.029, Mean=1.48 eV
  K=100: Precision=0.650, Recall=0.052, Mean=1.56 eV

Application: uv_detectors
Query: 'wide bandgap material for UV detectors and high-power electronics'
Expected range: 3.0 - 8.7 eV
Materials in range: 892
  K= 10: Precision=0.900, Recall=0.010, Mean=4.21 eV
  K= 50: Precision=0.840, Recall=0.047, Mean=4.35 eV
  K=100: Precision=0.780, Recall=0.087, Mean=4.18 eV
```

**Interpretation**:
- Solar cells: 80% precision @ K=10 → Strong learning! ✓
- UV detectors: 90% precision @ K=10 → Excellent! ✓
- Precision decreases with K → Normal behavior (retrieving more diverse results)

---

## Visualization

The evaluation generates two plots per property:

### 1. Box Plot with Expected Ranges
- **X-axis**: Application categories
- **Y-axis**: Property values (e.g., band gap in eV)
- **Box**: Distribution of retrieved materials (quartiles + median)
- **Shaded region**: Expected property range
- **Horizontal dashed lines**: Expected range boundaries

**Good Result**: Box overlaps well with shaded region
**Bad Result**: Box entirely outside shaded region

### 2. Precision/Recall Bar Chart
- **Blue bars**: Precision (CLaC model)
- **Gray bars**: Precision (Random baseline)
- **Orange bars**: Recall (CLaC model)

**Good Result**: Blue bars >> Gray bars
**Bad Result**: Blue bars ≈ Gray bars

---

## Supported Properties

| Property | Applications | Expected Ranges |
|----------|-------------|-----------------|
| `band gap` | 5 categories | 0.0 - 8.7 eV |
| `density` | 4 categories | 0.1 - 22.0 g/cm³ |
| `scintillation attenuation length` | 5 categories | 0.0 - 24.0 cm |
| `total magnetization` | 4 categories | 0.0 - 150.0 μB/f.u. |
| `energy above hull` | 4 categories | 0.0 - 6.0 eV/atom |

---

## Output Files

```
outputs/text_only_inverse_design/
└── <checkpoint_name>/
    ├── text_only_inverse_design_band_gap.json
    ├── text_only_inverse_design_band_gap.svg
    ├── text_only_inverse_design_density.json
    ├── text_only_inverse_design_density.svg
    └── ...
```

### JSON Structure

```json
{
  "property": "band gap",
  "unit": "eV",
  "k_values": [10, 50, 100],
  "application_results": [
    {
      "name": "solar_cells",
      "query": "material ideal for solar cell and photovoltaic applications",
      "expected_range": [1.1, 1.8],
      "num_materials_in_range": 1245,
      "k10": {
        "mean": 1.52,
        "std": 0.23,
        "median": 1.50,
        "precision": 0.800,
        "recall": 0.006,
        "values": [1.5, 1.6, 1.4, ...]
      }
    }
  ],
  "random_baseline": [...]
}
```

---

## Comparison with Standard Inverse Design

| Aspect | Standard Inverse Design | Text-Only Inverse Design |
|--------|------------------------|--------------------------|
| Query | "material with band gap **1.5 eV** for solar cells" | "material for solar cells" |
| Metric | MAE (Mean Absolute Error) | Precision, Recall |
| Tests | Retrieval accuracy | Semantic understanding |
| Baseline | Random retrieval | Random retrieval |
| Goal | Find materials with specific value | Find materials for application |

**Use Cases**:
- **Standard**: Property optimization, targeted material search
- **Text-Only**: Validate training data quality, test semantic learning

---

## Advanced Usage

### Custom K Values

```bash
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --k-values 5 10 20 50 100 500 1000 \
    --model-ckpt outputs/model.ckpt \
    --device cuda:0
```

### Compare Multiple Models

```bash
# Model A: Trained with synthetic narratives
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --model-ckpt outputs/synthetic_model/last.ckpt \
    --device cuda:0

# Model B: Trained with real papers only
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --model-ckpt outputs/real_papers_model/last.ckpt \
    --device cuda:0

# Compare precision values in JSON results
```

### Analyze Formula Replacement Impact

```bash
# Model trained with replace_formula_prob=0.0
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --model-ckpt outputs/no_replacement/last.ckpt

# Model trained with replace_formula_prob=1.0
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --model-ckpt outputs/full_replacement/last.ckpt
```

**Expected**: Full replacement model should have **higher precision** (less formula dependence)

---

## Troubleshooting

### Error: "Property not supported"
```
ValueError: Property 'foo' not supported for text-only inverse design
```

**Solution**: Check available properties in `get_text_only_application_queries()` function
Supported: `band gap`, `density`, `scintillation attenuation length`, `total magnetization`, `energy above hull`

### All Precision = 0.0

**Possible Causes**:
1. Model checkpoint incompatible (wrong architecture)
2. Data path incorrect (no test.parquet file)
3. Expected ranges too narrow (check template definitions)

**Debug**:
```python
# Check retrieved values distribution
import json
with open('outputs/text_only_inverse_design/.../text_only_inverse_design_band_gap.json') as f:
    results = json.load(f)
    print(results['application_results'][0]['k10']['values'])
```

### GPU Out of Memory

**Solution**: Reduce batch size or use CPU
```bash
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --batch-size 32 \  # default: 64
    --device cpu       # fallback to CPU
```

---


## Related Documentation

- [Standard Inverse Design Evaluation](inverse_design_usage.md)
- [Synthetic Data Generation](synthetic_data_generation.md)
- [Formula Replacement Feature](how_to_use_formula_replacement.md)
- [Evaluation Results Guide](../evaluation/evaluation_results_guide.md)
- `evaluation/compute_metrics.py` - Main evaluation script
- `scripts/evaluation/text_only_inverse_design_evaluation.sh` - Batch evaluation script

---

**Last Updated**: 2025-10-17
**Author**: Claude (AI Assistant)
