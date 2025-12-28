# Material Query Tool Guide

## Overview

`tools/query_materials.py` allows you to search for materials using natural language text descriptions. Unlike the inverse design tool which focuses on specific numerical property values, this tool handles more flexible, descriptive queries.

**Note:** All commands should be run from the project root directory.

## Features

- **Free-form text queries**: Describe what you're looking for in natural language
- **Top-K retrieval**: Get the most similar materials ranked by relevance
- **Property display**: See key properties (band gap, density, etc.) for each result
- **Multiple output formats**: Table (console), JSON, or CSV
- **Similarity filtering**: Set minimum similarity thresholds
- **Embedding cache**: Save/load embeddings for faster repeated queries

## Installation

No additional dependencies needed beyond the main CLaC environment.

## Basic Usage

### 1. Simple Query (Top-10 materials)

```bash
python tools/query_materials.py \
  --query "semiconductor with wide band gap for UV detection" \
  --checkpoint outputs/best_model.ckpt \
  --device cuda:0
```

**Output:**
```
========================================================================================================================
Query: semiconductor with wide band gap for UV detection
Found 10 materials
========================================================================================================================

Rank   Formula              Similarity    Band Gap     Form. Energy Density      E_hull       Magnet.
                                          (eV)         (eV/atom)    (g/cm³)      (eV/atom)    (μB)
------------------------------------------------------------------------------------------------------------------------
1      Ga2O3               0.8234        4.712        -2.123       5.880        0.000        0.000
2      ZnGa2O4             0.8156        4.234        -2.456       5.670        0.000        0.000
3      AlN                 0.8089        5.967        -2.890       3.230        0.000        0.000
...
========================================================================================================================
```

### 2. Query with More Results

```bash
python tools/query_materials.py \
  --query "lightweight magnetic material for spintronic applications" \
  --checkpoint outputs/best_model.ckpt \
  --k 20
```

### 3. Query with Similarity Threshold

```bash
python tools/query_materials.py \
  --query "stable oxide with band gap around 3 eV" \
  --checkpoint outputs/best_model.ckpt \
  --min-similarity 0.7 \
  --k 50
```

This will return up to 50 materials, but only those with similarity ≥ 0.7.

## Output Formats

### Table Format (Default)

Best for quick inspection in terminal:

```bash
python tools/query_materials.py --query "..." --format table
```

### JSON Format

Best for programmatic processing:

```bash
python tools/query_materials.py \
  --query "high-density material for radiation shielding" \
  --format json \
  --output my_results.json
```

**JSON structure:**
```json
{
  "query": "high-density material for radiation shielding",
  "num_results": 10,
  "results": [
    {
      "rank": 1,
      "formula": "PbWO4",
      "similarity": 0.8456,
      "properties": {
        "band_gap": 3.12,
        "formation_energy": -1.234,
        "density": 8.28,
        "e_above_hull": 0.0
      }
    },
    ...
  ]
}
```

### CSV Format

Best for Excel/spreadsheet analysis:

```bash
python tools/query_materials.py \
  --query "thermoelectric material with low thermal conductivity" \
  --format csv \
  --output thermoelectrics.csv
```

## Performance Optimization

### Embedding Cache

Encoding the entire candidate pool (~50k materials) takes 1-2 minutes. Use caching for faster repeated queries:

**First run (creates cache):**
```bash
python tools/query_materials.py \
  --query "first query" \
  --checkpoint outputs/best_model.ckpt \
  --cache-embeddings embeddings_cache.pt
```

**Subsequent runs (loads cache in seconds):**
```bash
python tools/query_materials.py \
  --query "second query" \
  --checkpoint outputs/best_model.ckpt \
  --cache-embeddings embeddings_cache.pt
```

### Batch Size

Adjust batch size based on your GPU memory:

```bash
python tools/query_materials.py \
  --query "..." \
  --batch-size 512  # Default: 256, larger = faster but more memory
```

## Query Examples

### Property-Based Queries

```bash
# Band gap
python tools/query_materials.py --query "wide band gap semiconductor for power electronics"
python tools/query_materials.py --query "narrow band gap material for infrared detection"

# Density
python tools/query_materials.py --query "lightweight material for aerospace applications"
python tools/query_materials.py --query "high-density material for counterweights"

# Stability
python tools/query_materials.py --query "thermodynamically stable oxide for catalysis"
python tools/query_materials.py --query "metastable material for thin film deposition"

# Magnetism
python tools/query_materials.py --query "ferromagnetic material for data storage"
python tools/query_materials.py --query "non-magnetic semiconductor for optical devices"
```

### Application-Based Queries

```bash
# Solar energy
python tools/query_materials.py --query "photovoltaic material with optimal band gap for solar cells"

# LEDs and displays
python tools/query_materials.py --query "light-emitting material for blue LED applications"

# Batteries
python tools/query_materials.py --query "stable lithium-containing compound for battery electrodes"

# Thermoelectrics
python tools/query_materials.py --query "heavy element compound with low thermal conductivity"

# Spintronics
python tools/query_materials.py --query "half-metallic ferromagnet for spin injection"
```

### Multi-Property Queries

```bash
python tools/query_materials.py --query "lightweight, stable oxide with wide band gap"
python tools/query_materials.py --query "magnetic semiconductor with intermediate band gap"
python tools/query_materials.py --query "dense, non-magnetic material for radiation shielding"
```

## Understanding Results

### Similarity Score

- **0.9-1.0**: Extremely high relevance (rare, very close semantic match)
- **0.8-0.9**: High relevance (strong match)
- **0.7-0.8**: Good relevance (reasonable match)
- **0.6-0.7**: Moderate relevance (may be useful)
- **<0.6**: Low relevance (consider refining query)

### Property Values

| Property | Typical Range | Notes |
|----------|--------------|-------|
| **Band Gap** | 0.0 - 6.0 eV | 0 = metallic, >3 = wide-gap semiconductor |
| **Formation Energy** | -5.0 - 0.0 eV/atom | More negative = more stable |
| **Density** | 1.0 - 12.0 g/cm³ | Higher = heavier |
| **E_hull** | 0.0 - 1.0 eV/atom | 0 = stable, >0.2 = metastable |
| **Magnetization** | 0.0 - 50.0 μB | 0 = non-magnetic, >1 = magnetic |

### N/A Values

If a property shows "N/A", it means the data is not available in the dataset for that material.

## Tips for Better Queries

### 1. Be Specific About Applications

❌ Bad: "good material"
✅ Good: "material for high-temperature supercapacitor electrodes"

### 2. Include Property Descriptors

❌ Bad: "semiconductor"
✅ Good: "semiconductor with band gap suitable for visible light absorption"

### 3. Mention Element Preferences (if any)

❌ Bad: "magnetic material"
✅ Good: "transition metal-based magnetic material for room temperature applications"

### 4. Combine Multiple Characteristics

❌ Bad: "stable material"
✅ Good: "stable, earth-abundant oxide for photocatalytic water splitting"

## Batch Query Script

For multiple queries, create a script:

```bash
#!/bin/bash

CHECKPOINT="outputs/best_model.ckpt"
CACHE="embeddings_cache.pt"

# Create cache on first run
python tools/query_materials.py \
  --query "semiconductor for solar cells" \
  --checkpoint $CHECKPOINT \
  --cache-embeddings $CACHE \
  --format json --output solar_materials.json

# Subsequent queries use cache
python tools/query_materials.py \
  --query "thermoelectric material" \
  --checkpoint $CHECKPOINT \
  --cache-embeddings $CACHE \
  --format json --output thermoelectric_materials.json

python tools/query_materials.py \
  --query "magnetic material for spintronics" \
  --checkpoint $CHECKPOINT \
  --cache-embeddings $CACHE \
  --format json --output magnetic_materials.json
```

## Integration with Inverse Design

These tools complement each other:

1. **Use query tool** to explore what materials exist:
   ```bash
   python tools/query_materials.py --query "stable oxide semiconductor"
   ```

2. **Use inverse design** to find materials with specific target values:
   ```bash
   python evaluation/compute_metrics.py \
     --evaluation-method inverse-design \
     --property "band gap" \
     --targets 2.0 3.0
   ```

## Troubleshooting

### Issue: "No checkpoint provided, using randomly initialized model"

**Solution**: Always provide a trained checkpoint:
```bash
--checkpoint path/to/your/checkpoint.ckpt
```

### Issue: "No materials found with similarity >= X"

**Solutions:**
- Lower the `--min-similarity` threshold
- Refine your query to be less restrictive
- Check the max similarity printed in the warning

### Issue: Out of memory

**Solutions:**
- Reduce `--batch-size` (try 128 or 64)
- Use CPU: `--device cpu` (slower but no memory limit)

### Issue: Slow performance

**Solutions:**
- Use embedding cache: `--cache-embeddings cache.pt`
- Increase `--batch-size` if you have GPU memory
- Reduce dataset size in config (for testing)

## API Usage (Python)

You can also use the tool programmatically:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tools.query_materials import load_model_and_data, encode_candidate_pool, query_materials

# Load model
model, datamodule, cfg = load_model_and_data(
    checkpoint_path="outputs/best_model.ckpt",
    device="cuda:0"
)

# Encode candidate pool (once)
graph_embeddings, formulas, properties_list = encode_candidate_pool(
    model=model,
    datamodule=datamodule,
    device="cuda:0"
)

# Query multiple times
queries = [
    "wide band gap semiconductor",
    "magnetic material",
    "lightweight oxide"
]

for query_text in queries:
    results = query_materials(
        query_text=query_text,
        model=model,
        graph_embeddings=graph_embeddings,
        formulas=formulas,
        properties_list=properties_list,
        device="cuda:0",
        k=10
    )

    print(f"\nQuery: {query_text}")
    for result in results[:3]:
        print(f"  {result['rank']}. {result['formula']} (sim={result['similarity']:.3f})")
```

## Related Tools

- `evaluation/compute_metrics.py --evaluation-method inverse-design`: Property-specific inverse design
- `evaluation/compute_metrics.py --evaluation-method zero-shot-retrieval`: Graph-to-text retrieval evaluation
- `evaluation/compute_metrics.py --evaluation-method zero-shot-qa`: Question-answering evaluation

## References

- Based on CLaC (Contrastive Learning for Chemistry) model
- Text encoder: MatBERT/MatSciBERT
- Graph encoder: CGCNN/PaiNN/ORB
- Similarity metric: Cosine similarity in projection space
