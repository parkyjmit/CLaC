# How to Compute Structure-Text Similarity

## Overview

`tools/compute_similarity.py` computes the similarity between a specific material (by mp-id) and a text prompt using a trained CLaC model.

This is useful for:
- Testing model understanding of specific materials
- Validating retrieval performance on known materials
- Interactive exploration of structure-text embeddings
- Debugging and analysis

## Quick Start

### Single Material (Local Dataset)

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id mp-19990 \
    --text "Materials for solid-state batteries applications"
```

### Multiple Materials

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-ids mp-19990 mp-1234 mp-5678 \
    --text "High energy density cathode materials"
```

### Using Materials Project API

If a material is not in your local dataset, the tool automatically falls back to Materials Project API:

```bash
# Set API key in .env file (recommended)
echo "MP_API_KEY=your_api_key_here" >> .env

python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id mp-99999 \
    --text "Solid-state electrolytes"
```

Get your API key from: https://next-gen.materialsproject.org/api

## Usage Examples

### Example 1: Test Specific Query

```bash
python tools/compute_similarity.py \
    --model outputs/SynData-painn-infonce-galactica-125m-vTrue-tTrue-5epoch/last.ckpt \
    --mp-id mp-19990 \
    --text "Solid-state electrolytes for lithium batteries" \
    --device cuda:0
```

**Output:**
```
================================================================================
Structure-Text Similarity Computation
================================================================================
Material IDs: mp-19990
Text prompt: "Solid-state electrolytes for lithium batteries"
Model: outputs/SynData-painn-infonce-galactica-125m-vTrue-tTrue-5epoch/last.ckpt
================================================================================

Loading model from: outputs/SynData-painn-infonce-galactica-125m-vTrue-tTrue-5epoch/last.ckpt
✓ Auto-detected graph encoder: torch_geometric
✓ Using model's tokenizer: facebook/galactica-125m
✓ Model loaded successfully

Encoding text prompt...
✓ Text embedding shape: torch.Size([1, 768])

Processing materials: 100%|████████████| 1/1
✓ Found mp-19990 in test split
  ✓ mp-19990: similarity = 0.7854

================================================================================
Results Summary
================================================================================
mp-19990        Li7La3Zr2O12         similarity = 0.7854
================================================================================
```

### Example 2: Compare Multiple Materials

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-ids mp-19990 mp-1234 mp-5678 mp-9999 \
    --text "Wide bandgap semiconductor for UV detection" \
    --output results/uv_materials.json
```

### Example 3: Interactive Testing

Create a script to test multiple queries:

```bash
#!/bin/bash
# test_queries.sh

MODEL="outputs/my-model/last.ckpt"
MP_ID="mp-19990"

queries=(
    "Solid-state electrolytes"
    "High ionic conductivity materials"
    "Lithium-ion battery cathodes"
    "Magnetic semiconductors"
    "Transparent conducting oxides"
)

for query in "${queries[@]}"; do
    echo "Testing: $query"
    python tools/compute_similarity.py \
        --model "$MODEL" \
        --mp-id "$MP_ID" \
        --text "$query"
    echo ""
done
```

### Example 4: Save Results to JSON

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-ids mp-19990 mp-1234 \
    --text "Battery materials" \
    --output results/battery_similarities.json
```

**Output JSON:**
```json
{
  "text_prompt": "Battery materials",
  "model": "outputs/my-model/last.ckpt",
  "results": [
    {
      "material_id": "mp-19990",
      "found": true,
      "similarity": 0.7854,
      "properties": {
        "formula": "Li7La3Zr2O12",
        "reduced_formula": "Li7La3Zr2O12",
        "space_group": "I4_1/acd:2",
        "crystal_system": "tetragonal",
        "band_gap_eV": 4.5,
        "density_g_cm3": 5.12,
        ...
      },
      "text_prompt": "Battery materials",
      "source": "local_dataset",
      "split": "test"
    },
    ...
  ]
}
```

### Example 5: Using Materials Project API

**For materials not in your local dataset:**

```bash
# Setup API key in .env file first
echo "MP_API_KEY=your_api_key_here" >> .env

# Query material not in local dataset (automatically fetches from API)
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id mp-99999 \
    --text "Novel cathode materials for lithium batteries"
```

**Output:**
```
✓ Loaded environment variables from /home/user/CLaC-revision/.env
Loading model...
✓ Model loaded successfully

Encoding text prompt...
✓ Text embedding shape: torch.Size([1, 768])

Processing materials: 100%|██| 1/1
  Material mp-99999 not found in local dataset
  Fetching mp-99999 from Materials Project API...
  ✓ Fetched mp-99999 from Materials Project API
  ✓ mp-99999: similarity = 0.8123
```

**Force Materials Project API (skip local dataset):**

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id mp-19990 \
    --text "Solid-state electrolytes" \
    --use-mp-api
```

This is useful to ensure you get the latest data from Materials Project.

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Path to model checkpoint (e.g., `outputs/my-model/last.ckpt`) |
| `--mp-id` | Single material ID to query (e.g., `mp-19990`) |
| `--mp-ids` | Multiple material IDs (e.g., `mp-1 mp-2 mp-3`) |
| `--text` | Text prompt to compute similarity with |

**Note:** Either `--mp-id` or `--mp-ids` is required (mutually exclusive).

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cuda:0` | Device to use (`cuda:0`, `cuda:1`, `cpu`) |
| `--data-path` | `datafiles/mp_3d_2020_gpt_narratives` | Path to dataset (without suffix) |
| `--id-column` | `material-id` | Column name for material ID in parquet |
| `--graphdatatype` | `auto` | Graph encoder type (`auto`, `torch_geometric`, `orb`) |
| `--llm` | `facebook/galactica-125m` | LLM tokenizer (if not auto-detected) |
| `--mp-api-key` | None | Materials Project API key (reads from `.env` or `MP_API_KEY` env var) |
| `--use-mp-api` | False | Always use Materials Project API (skip local dataset) |
| `--output` | None | Output JSON file path |
| `--verbose` | False | Print detailed information |

### Materials Project API Setup

The tool can automatically fetch materials from Materials Project API if they're not in your local dataset.

**Prerequisites:**
```bash
pip install mp-api
```

**Setup API Key (choose one method):**

1. **In `.env` file (recommended):**
   ```bash
   echo "MP_API_KEY=your_api_key_here" >> .env
   ```

2. **Environment variable:**
   ```bash
   export MP_API_KEY="your_api_key_here"
   ```

3. **Command-line argument:**
   ```bash
   python tools/compute_similarity.py ... --mp-api-key "your_api_key"
   ```

**Get API Key:**
Visit https://next-gen.materialsproject.org/api and sign in to get your free API key.

## How It Works

### 1. Model Loading

```python
# Load model from checkpoint
model = CLaCLite.load_from_checkpoint(model_ckpt)
model.eval()

# Auto-detect graph encoder type (ORB or torch_geometric)
if hasattr(model.graph_encoder, 'orb_model'):
    graphdatatype = 'orb'
else:
    graphdatatype = 'torch_geometric'
```

### 2. Material Lookup

```python
# Search for material ID in all splits
for split in ['test', 'val', 'train']:
    df = pd.read_parquet(f"{data_path}_{split}.parquet")
    matches = df[df['material-id'] == mp_id]
    if len(matches) > 0:
        return matches.iloc[0]
```

### 3. Encoding

```python
# Encode structure (graph)
atoms = get_atoms_from_data(material_data)
graph = atom_to_torch_graph_data(atoms)
g_feat = model.graph_encoder(graph)
g_emb = model.loss.global_d.img_block(g_feat)
g_emb = F.normalize(g_emb, p=2, dim=-1)

# Encode text
inputs = model.tokenizer([text], ...)
_, t_feat = model.encode_text(inputs)
t_emb = model.loss.global_d.text_block(t_feat)
t_emb = F.normalize(t_emb, p=2, dim=-1)
```

### 4. Similarity Computation

```python
# Cosine similarity (embeddings are already normalized)
similarity = (g_emb * t_emb).sum().item()
```

## Interpreting Results

### Similarity Scores

- **0.8 - 1.0**: Very high similarity (strong match)
- **0.6 - 0.8**: High similarity (good match)
- **0.4 - 0.6**: Moderate similarity (potential match)
- **0.2 - 0.4**: Low similarity (weak match)
- **0.0 - 0.2**: Very low similarity (no match)

**Note:** Absolute values depend on model training and text specificity.

### Expected Behavior

**Well-trained model:**
- High similarity for relevant queries
- Low similarity for irrelevant queries
- Smooth gradation between related concepts

**Untrained/poorly-trained model:**
- Random similarity scores
- No correlation with semantic relevance
- Similar scores for all queries

### Example Interpretations

**Good model performance:**
```
mp-19990 (Li7La3Zr2O12):
  "Solid-state electrolytes"           → 0.85 (excellent)
  "Lithium-ion conductors"             → 0.78 (good)
  "Wide bandgap semiconductors"        → 0.42 (low, correct)
  "Magnetic materials"                 → 0.15 (very low, correct)
```

**Poor model performance:**
```
mp-19990 (Li7La3Zr2O12):
  "Solid-state electrolytes"           → 0.51
  "Lithium-ion conductors"             → 0.49
  "Wide bandgap semiconductors"        → 0.52
  "Magnetic materials"                 → 0.48
  (All similar, no discrimination)
```

## Troubleshooting

### "Material not found in dataset"

**Problem:** The specified mp-id doesn't exist in the parquet files.

**Solutions:**
1. Check the mp-id is correct
2. Verify the dataset path with `--data-path`
3. Check if the material is in the dataset:
   ```bash
   python -c "import pandas as pd; df = pd.read_parquet('datafiles/mp_3d_2020_gpt_narratives_test.parquet'); print('mp-19990' in df['material-id'].values)"
   ```

### "Column 'material-id' not found"

**Problem:** The parquet file uses a different column name for material IDs.

**Solution:** Use `--id-column` to specify the correct column:
```bash
--id-column "mp-id"  # or "mpid", "id", etc.
```

### "CUDA out of memory"

**Problem:** GPU memory insufficient for large models.

**Solutions:**
1. Use CPU: `--device cpu`
2. Process materials one at a time instead of batch
3. Use a smaller model

### Different tokenizer warning

**Problem:** Model's tokenizer differs from config.

**This is usually fine** - the script auto-detects the model's tokenizer. Only specify `--llm` if auto-detection fails.

## Advanced Usage

### Custom Dataset Path

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id jarvis-1234 \
    --text "High thermal conductivity" \
    --data-path datafiles/jarvis_dft3d \
    --id-column "jid"
```

### Verbose Output

```bash
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id mp-19990 \
    --text "Battery materials" \
    --verbose
```

This prints:
- Detailed model loading info
- Embedding shapes
- Per-material encoding progress
- Full property information

### Batch Processing with Python

```python
import subprocess
import json

mp_ids = ['mp-19990', 'mp-1234', 'mp-5678']
queries = [
    'Solid-state electrolytes',
    'Magnetic semiconductors',
    'Transparent conductors'
]

for query in queries:
    result = subprocess.run([
        'python', 'tools/compute_similarity.py',
        '--model', 'outputs/my-model/last.ckpt',
        '--mp-ids'] + mp_ids + [
        '--text', query,
        '--output', f'results/{query.replace(" ", "_")}.json'
    ])
```

## Integration with Other Tools

### Use with query_materials.py

```bash
# Step 1: Find top materials for a query
python tools/query_materials.py \
    --model outputs/my-model/last.ckpt \
    --query "Solid-state electrolytes" \
    --k 10 \
    --format json \
    --output top10.json

# Step 2: Verify specific material
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-id mp-19990 \
    --text "Solid-state electrolytes"
```

### Use with Inverse Design Evaluation

```bash
# After inverse design evaluation, test specific materials
python tools/compute_similarity.py \
    --model outputs/my-model/last.ckpt \
    --mp-ids mp-1 mp-2 mp-3 \
    --text "material with band gap of 2.5 eV suitable for solar cells"
```

## Tips

1. **Start with known materials**: Test mp-ids you understand to calibrate expectations
2. **Use multiple queries**: Test various phrasings to understand model robustness
3. **Compare models**: Run same queries on trained vs untrained models
4. **Save results**: Use `--output` to track experiments over time
5. **Batch similar queries**: Process multiple materials with one text for efficiency

## Related Documentation

- `tools/query_materials.py` - Text-based material search
- `evaluation/compute_metrics.py` - Full evaluation suite
- `docs/guides/how_to_plot_zero_shot_comparison.md` - Visualization guide
- `CLAUDE.md` - Development log with inverse design details
