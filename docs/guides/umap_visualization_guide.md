# UMAP Embedding Visualization Guide

## Overview

The `evaluation/visualize_embeddings.py` script visualizes the learned embedding space of trained CLaC models using UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction.

**Note:** All commands should be run from the project root directory.

## Features

- **Flexible embedding extraction**: Graph, text, or both modalities
- **Multiple coloring schemes**:
  - Modality (graph vs text)
  - Material properties (band gap, formation energy, density, etc.)
  - Keyword presence in text descriptions
  - Text query similarity (semantic search visualization)
- **Automatic model detection**: Detects graph encoder type (ORB/torch_geometric) and tokenizer
- **High-quality plots**: Publication-ready visualizations with customizable parameters

## Installation

First, install UMAP:

```bash
pip install umap-learn
```

## Basic Usage

### 1. Visualize Graph vs Text Embeddings (Modality)

**With trained model:**
```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by modality \
  --embed-type both
```

**With untrained (random initialized) model:**
```bash
python evaluation/visualize_embeddings.py \
  --config config/config.yaml \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by modality \
  --embed-type both
```

**Output**: Shows how well graph and text embeddings align in the learned space.
**Note**: Using `--config` without `--checkpoint` will use a randomly initialized model, useful for comparing before/after training.

### 2. Color by Material Property (Band Gap)

```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by property \
  --property-name bandgap \
  --embed-type both \
  --cmap viridis
```

**Output**:
- Combined plot (graph + text colored by band gap)
- Separate graph-only plot
- Separate text-only plot

**Supported properties**:
- `bandgap` or `band_gap`
- `formation_energy` or `formation_energy_per_atom`
- `density`
- `scintillation` or `scintillation_attenuation_length`
- `magnetization` or `total_magnetization`
- `ehull` or `energy_above_hull`

### 3. Color by Keyword Presence

```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by keyword \
  --keywords magnetic semiconductor \
  --embed-type both
```

**Output**: Shows samples with/without the specified keywords, useful for understanding semantic clustering.

### 4. Color by Text Query Similarity (Semantic Search Visualization)

```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "magnetic semiconductor with high conductivity" \
  --embed-type both \
  --cmap coolwarm
```

**How it works**:
1. The text query is encoded using the model's text encoder
2. Cosine similarity is computed between the query embedding and all graph/text embeddings
3. Similarity values (0-1) are visualized using a color gradient
   - Red: High similarity (materials matching the query)
   - Blue: Low similarity (materials not matching the query)

**Output**:
- Combined plot (graph + text colored by similarity)
- Separate graph-only plot
- Separate text-only plot
- Similarity statistics (mean, std, min, max)

**Use cases**:
- **Property-based queries**: "high band gap material", "dense oxide", "low formation energy"
- **Application-based queries**: "photovoltaic material", "battery electrode", "thermoelectric"
- **Composition queries**: "lithium-containing compound", "perovskite structure"
- **Multi-attribute queries**: "lightweight magnetic semiconductor"

**Example queries**:
```bash
# Find materials similar to "high band gap insulator"
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "high band gap insulator" \
  --embed-type both

# Find materials similar to "ferromagnetic metal"
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "ferromagnetic metal with high magnetization" \
  --embed-type graph

# Find materials for specific applications
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "transparent conducting oxide for solar cells" \
  --embed-type both
```

## Advanced Options

### UMAP Parameters

Control the structure of the UMAP embedding:

```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by property \
  --property-name density \
  --n-neighbors 30 \
  --min-dist 0.05 \
  --metric euclidean
```

**Parameters**:
- `--n-neighbors`: Number of neighbors (default: 15). Higher = more global structure.
- `--min-dist`: Minimum distance (default: 0.1). Lower = tighter clusters.
- `--metric`: Distance metric (default: cosine). Options: cosine, euclidean, manhattan.

### Visualization Parameters

```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by property \
  --property-name bandgap \
  --alpha 0.4 \
  --cmap plasma
```

**Parameters**:
- `--alpha`: Point transparency (0-1, default: 0.6)
- `--cmap`: Colormap for continuous values (viridis, plasma, inferno, magma, coolwarm, etc.)

### Dataset Split

```bash
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --split train \
  --color-by property \
  --property-name density
```

**Options**: `train`, `val`, `test` (default: test)

### Only Graph or Text Embeddings

```bash
# Only graph embeddings
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --embed-type graph \
  --color-by property \
  --property-name bandgap

# Only text embeddings
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --embed-type text \
  --color-by property \
  --property-name bandgap
```

## Example Workflows

### 0. Compare Before/After Training

```bash
DATA=datafiles/mp_3d_2020_gpt_narratives
OUTPUT_BEFORE=outputs/umap_visualizations/umap_before_training
OUTPUT_AFTER=outputs/umap_visualizations/umap_after_training

# Before training (random initialization)
python evaluation/visualize_embeddings.py \
  --config config/config.yaml \
  --data-path $DATA \
  --color-by modality \
  --output-dir $OUTPUT_BEFORE

# After training
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/last.ckpt \
  --data-path $DATA \
  --color-by modality \
  --output-dir $OUTPUT_AFTER

# Compare the two outputs to see training effect
```

### 1. Full Analysis of a Model

```bash
MODEL=outputs/orb-infonce-galactica-125m/last.ckpt
DATA=datafiles/mp_3d_2020_gpt_narratives
OUTPUT=outputs/umap_visualizations/umap_analysis

# Modality alignment
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by modality --output-dir $OUTPUT

# Band gap distribution
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by property --property-name bandgap \
  --output-dir $OUTPUT

# Formation energy distribution
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by property --property-name formation_energy \
  --output-dir $OUTPUT

# Magnetic materials
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by keyword --keywords magnetic magnetization \
  --output-dir $OUTPUT

# Semiconductor materials
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by keyword --keywords semiconductor semiconducting \
  --output-dir $OUTPUT
```

### 2. Compare Multiple Properties

```bash
MODEL=outputs/your-model/last.ckpt
DATA=datafiles/mp_3d_2020_gpt_narratives
OUTPUT=outputs/umap_visualizations/property_comparison

for prop in bandgap density formation_energy magnetization; do
  python evaluation/visualize_embeddings.py \
    --checkpoint $MODEL --data-path $DATA \
    --color-by property --property-name $prop \
    --output-dir $OUTPUT
done
```

### 3. Keyword-Based Semantic Analysis

```bash
MODEL=outputs/your-model/last.ckpt
DATA=datafiles/mp_3d_2020_gpt_narratives
OUTPUT=outputs/umap_visualizations/semantic_analysis

# Check different material classes
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by keyword --keywords metal metallic conductor \
  --output-dir $OUTPUT

python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by keyword --keywords insulator insulating dielectric \
  --output-dir $OUTPUT

python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by keyword --keywords perovskite oxide \
  --output-dir $OUTPUT
```

### 4. Text Query Similarity Analysis

```bash
MODEL=outputs/your-model/last.ckpt
DATA=datafiles/mp_3d_2020_gpt_narratives
OUTPUT=outputs/umap_visualizations/query_similarity

# Find materials for different applications
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by text-similarity \
  --text-query "photovoltaic material for solar cells" \
  --output-dir $OUTPUT

python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by text-similarity \
  --text-query "thermoelectric material with high figure of merit" \
  --output-dir $OUTPUT

python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by text-similarity \
  --text-query "lightweight structural material" \
  --output-dir $OUTPUT

# Property-specific queries
python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by text-similarity \
  --text-query "material with band gap around 2 eV" \
  --output-dir $OUTPUT

python evaluation/visualize_embeddings.py \
  --checkpoint $MODEL --data-path $DATA \
  --color-by text-similarity \
  --text-query "stable magnetic material" \
  --output-dir $OUTPUT
```

## Output Files

The script generates:

1. **PNG images**: High-resolution (300 DPI) plots
   - `umap_test_modality.png` (if `--color-by modality`)
   - `umap_test_property_bandgap.png` (if `--color-by property`)
   - `umap_test_graph_property_bandgap.png` (graph only, if `--embed-type both`)
   - `umap_test_text_property_bandgap.png` (text only, if `--embed-type both`)
   - `umap_test_keyword_magnetic.png` (if `--color-by keyword`)
   - `umap_test_text_similarity_<query>.png` (if `--color-by text-similarity`)
   - `umap_test_graph_text_similarity_<query>.png` (graph only, if `--embed-type both`)
   - `umap_test_text_text_similarity_<query>.png` (text only, if `--embed-type both`)

2. **JSON metadata**: `umap_test_data.json`
   - UMAP coordinates
   - UMAP parameters
   - Model checkpoint path
   - Modality labels (if applicable)

## Interpretation Guide

### Good Alignment Indicators

1. **Modality overlap**: Graph and text embeddings of the same material should be close together
2. **Property gradients**: Continuous properties (band gap, density) should show smooth color gradients
3. **Semantic clusters**: Keywords should create distinct clusters (e.g., "magnetic" materials grouped together)
4. **Query-relevant clustering**: Text similarity should show clear red regions (high similarity) concentrated in specific areas

### Warning Signs

1. **Modality separation**: Graph and text embeddings form separate clusters → poor alignment
2. **Random property distribution**: No visible pattern → model didn't learn property relationships
3. **Scattered keywords**: No clustering by keywords → weak semantic understanding
4. **Uniform similarity**: All materials have similar cosine similarity to query → poor discriminative power

### Text Query Similarity Interpretation

**High-quality results**:
- Clear red regions (high similarity > 0.7) for query-relevant materials
- Smooth gradients transitioning from red to blue
- Graph and text embeddings of same material have similar colors
- Multiple distinct clusters for multi-attribute queries

**Poor results**:
- All similarities in narrow range (e.g., 0.45-0.55) → model not capturing query semantics
- Random color distribution → no learned relationship
- Large difference between graph and text similarities for same material → poor modality alignment

## Tips

1. **Start with modality**: Always check graph-text alignment first
2. **Try multiple properties**: Different properties may reveal different aspects of the learned space
3. **Use relevant keywords**: Choose keywords that appear frequently in your dataset
4. **Adjust UMAP parameters**: If structure is unclear, try different `n_neighbors` and `min_dist`
5. **Compare models**: Run the same analysis on different checkpoints to track training progress
6. **Test text queries**: Start with simple queries (e.g., "magnetic"), then try more complex ones (e.g., "stable magnetic semiconductor")
7. **Validate query results**: Compare text-similarity visualization with property-based visualization to verify model understanding
8. **Use appropriate colormaps**: `coolwarm` for similarity (diverging), `viridis` for properties (sequential)

## Troubleshooting

### Error: Property not found
```
ValueError: Property 'your_property' not found
```
**Solution**: Check available properties in your dataset. Use exact column names from the parquet file.

### Poor separation/Too much overlap
**Solution**: Adjust UMAP parameters:
- Increase `--n-neighbors` for more global structure
- Decrease `--min-dist` for tighter clusters
- Try different `--metric` (euclidean vs cosine)

### Out of memory
**Solution**:
- Reduce `--batch-size`
- Use smaller dataset split or subset
- Use `--embed-type graph` or `--embed-type text` instead of `both`

## Examples with Real Data

```bash
# Example 1: Analyze band gap learning
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by property --property-name bandgap \
  --n-neighbors 20 --min-dist 0.05 \
  --cmap coolwarm --alpha 0.5

# Example 2: Check for magnetic materials clustering
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by keyword --keywords magnetic ferromagnetic \
  --embed-type both

# Example 3: Dense materials
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by property --property-name density \
  --cmap YlOrRd
 
# Example 4: Text query similarity - find semiconductors
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "semiconductor with moderate band gap" \
  --embed-type both --cmap coolwarm

# Example 5: Text query similarity - find battery materials
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "lithium-containing material for battery electrodes" \
  --embed-type both --cmap coolwarm

# Example 6: Compare property vs query similarity
# First: Color by actual band gap values
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by property --property-name bandgap \
  --output-dir outputs/umap_visualizations/comparison

# Then: Color by similarity to "high band gap material"
python evaluation/visualize_embeddings.py \
  --checkpoint outputs/your-model/checkpoint.ckpt \
  --data-path datafiles/mp_3d_2020_gpt_narratives \
  --color-by text-similarity \
  --text-query "material with high band gap" \
  --output-dir outputs/umap_visualizations/comparison
# Compare the two plots to see if model correlates query with property!
```
