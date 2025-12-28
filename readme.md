# Contrastive Language-Crystal Pretraining

This is the official implementation of Contrastive Language-Crystal Pretraining (CLaC). In this work, we introduced a contrastive pretraining framework for multimodal representation learning between materials science knowledge and crystals (~126k materials-narratives pair dataset). CLaC enables multimodal tasks which was unattainable before even with latest large language models(LLMs) while improves downstream performances. Just as we interpret images and express them in words, CLaC could be a part of AI which can be utilized as a fundamental model to analyze crystal structures and identify suitable application fields, or conversely, to search for crystal structures with characteristics that align with specific applications.

![](asset/overall_architecture.png)

## Repository Structure

```
CLaC-revision/
├── asset/                    # Images and architecture diagrams
├── config/                   # Configuration files (model, training, data)
├── data/                     # Data loading and augmentation modules
├── model/                    # Model definitions (CLaC, graph encoders, text encoders)
├── evaluation/               # Evaluation and analysis scripts
│   ├── compute_metrics.py              # Main evaluation (zero-shot, inverse design)
│   ├── compute_metrics_baseline.py     # LLM baseline evaluation
│   ├── compute_metrics_dual.py         # Dual-property evaluation
│   ├── analyze_text_embedding_similarity.py  # Text embedding analysis
│   ├── visualize_embeddings.py         # UMAP visualization
│   └── plot_*.py                       # Various plotting utilities
├── scripts/                  # Executable bash scripts
│   ├── evaluation/                     # Evaluation scripts
│   ├── visualization/                  # Visualization scripts
│   └── training/                       # Training scripts
├── tools/                    # Utility scripts and data preparation
│   ├── run_manual_sweep.py
│   └── query_materials.py
├── docs/                     # Documentation
│   ├── guides/                         # Usage guides
│   └── evaluation/                     # Evaluation methodology
├── outputs/                  # All experimental results
│   ├── figures/                        # Publication-ready figures
│   ├── component_comparison/
│   ├── inverse_design_dual/
│   ├── text_embedding_similarity/
│   └── umap_visualizations/
├── train.py                  # Main training script
├── sweep.yaml               # Hyperparameter sweep configuration
└── readme.md                # This file
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.6 (or adjust installation commands for your CUDA version)
- Conda or Miniconda

### Step-by-Step Installation

**1. Create and activate conda environment:**
```bash
conda create -n clac python=3.10
conda activate clac
```

**2. Install PyTorch with CUDA support:**
```bash
# For CUDA 12.6 (recommended)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.1
# pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu121
```

**3. Install PyTorch Geometric and extensions:**
```bash
# For CUDA 12.6
pip install torch-geometric==2.6.1
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# For CUDA 12.1, use: torch-2.8.0+cu121.html
```

**4. Install materials science packages:**
```bash
pip install orb-models==0.5.5 jarvis-tools
```

**5. Install remaining dependencies:**
```bash
pip install transformers datasets spacy hydra-core lightning wandb umap-learn
python -m spacy download en_core_web_sm
```

**6. Install additional scientific packages:**
```bash
pip install matplotlib scikit-learn scipy pandas python-dotenv
```

### Alternative Installation Methods

**Option A: Using environment.yml (Quick Setup)**
```bash
conda env create -f environment.yml
conda activate clac
python -m spacy download en_core_web_sm
```

**Option B: Using requirements.txt**

After installing PyTorch and PyTorch Geometric (steps 2-3 above), you can install other dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Note:** PyTorch and PyTorch Geometric must be installed separately with the correct CUDA version before using requirements.txt.

### Verify Installation

Test your installation:
```bash
python -c "import torch; import torch_geometric; import orb_models; print('✓ Installation successful')"
```

### Troubleshooting

**CUDA version mismatch:**
- Check your CUDA version: `nvcc --version` or `nvidia-smi`
- Install matching PyTorch version from [PyTorch website](https://pytorch.org/get-started/locally/)

**PyTorch Geometric installation issues:**
- Ensure PyTorch is installed first
- Use the correct wheel URL for your CUDA version
- See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## Dataset
### Data configuration
We have proposed the following three data configurations for training CLaC:

1. **Academic paper dataset**: Text extracted from real research papers using materials as queries.
2. **Synthesized Text**: Text generated by leveraging the inference capabilities of LLMs based on material property values extracted from open materials databases.
3. **Mixed Dataset**: A dataset combining data from (1) and (2).

We appreciate your understanding that we are unable to fully share all data and the corresponding pretrained model weights due to potential copyright concerns regarding academic literature. However, the synthesized data is publicly available at [GPT-Narratives-for-Materials](https://huggingface.co/datasets/yjeong/GPT-Narratives-for-Materials).

### Download Dataset

Download the synthesized GPT narratives dataset:

```bash
# Create data directory
mkdir -p datafiles

# Download dataset from HuggingFace
wget https://huggingface.co/datasets/yjeong/GPT-Narratives-for-Materials/resolve/main/mp_3d_2020_gpt_narratives.parquet \
    -O datafiles/mp_3d_2020_gpt_narratives.parquet

# Alternative: using curl
curl -L https://huggingface.co/datasets/yjeong/GPT-Narratives-for-Materials/resolve/main/mp_3d_2020_gpt_narratives.parquet \
    -o datafiles/mp_3d_2020_gpt_narratives.parquet
```

**Split into train/val/test sets:**

```bash
python data/get_splits.py --input datafiles/mp_3d_2020_gpt_narratives.parquet --train-ratio 0.8 --val-ratio 0.1
```

This creates:
- `datafiles/mp_3d_2020_gpt_narratives_train.parquet` (80%)
- `datafiles/mp_3d_2020_gpt_narratives_val.parquet` (10%)
- `datafiles/mp_3d_2020_gpt_narratives_test.parquet` (10%)

## Pre-training

To train the CLaC model, specify the base dataset path (e.g., `datafiles/mp_3d_2020_gpt_narratives`) in the configuration. The training code will automatically recognize and use the split files (`_train.parquet`, `_val.parquet`, `_test.parquet`) for training, validation, and testing.

### Set configuration of training
See `yaml` files in `config` section. If you want to change training configuration, modify them before running.

### Run pretraining
Training code is based on pytorch lightning and hydra configuration.
To pretrain the CLaC model, run `python train.py` with appropriate configuration.

### Advanced Features

#### Resume Training from Checkpoint
To continue training from a saved checkpoint:

```yaml
# config/config.yaml
resume_from_checkpoint: outputs/experiment/epoch=49-step=5000.ckpt
```

Or via command line:
```bash
python train.py resume_from_checkpoint=outputs/experiment/epoch=49-step=5000.ckpt
```

See [docs/guides/how_to_resume_training.md](docs/guides/how_to_resume_training.md) for details.

## Evaluation for zero-shot tasks
In this work, we measured zero-shot ability of our CLaC.
For zero-shot retrieval, we measured the model's performance in finding corresponding crystals when given texts, and vice versa.
For zero-shot multimodal understanding, we utilized multiple-choice questions (included in [GPT-Narratives-for-Materials](https://huggingface.co/datasets/yjeong/GPT-Narratives-for-Materials)) regarding crystals, such as their composition and structure.

### Running Evaluations

**Zero-shot retrieval and QA:**
```bash
python evaluation/compute_metrics.py \
  --evaluation-method 'zero-shot retrieval' \
  --label text \
  --model-ckpt outputs/your-model/checkpoint.ckpt \
  --device cuda:0
```

**Inverse design:**
```bash
python evaluation/compute_metrics.py \
  --evaluation-method inverse-design \
  --property "band gap" \
  --k-values 3 10 100 \
  --model-ckpt outputs/your-model/checkpoint.ckpt \
  --device cuda:0
```

**Batch evaluation scripts:**
```bash
# Run all zero-shot evaluations
bash scripts/evaluation/zero_shot_qa_evaluation.sh

# Run inverse design evaluation
bash scripts/evaluation/inverse_design_evaluation.sh
```

For detailed evaluation methodology and usage, see [docs/evaluation/](docs/evaluation/).


## Additional Documentation

For comprehensive guides on various features and evaluations:

**Usage Guides** (`docs/guides/`):
- [How to use formula replacement](docs/guides/how_to_use_formula_replacement.md)
- [How to resume training](docs/guides/how_to_resume_training.md)
- [How to run ablation studies](docs/guides/how_to_ablation_study.md)
- [How to run hyperparameter sweeps](docs/guides/how_to_sweep.md)
- [Inverse design usage guide](docs/guides/inverse_design_usage.md)
- [Text-only inverse design guide](docs/guides/text_only_inverse_design_guide.md)
- [Text embedding similarity analysis](docs/guides/text_embedding_analysis_guide.md)
- [Threshold embedding analysis](docs/guides/threshold_embedding_analysis_guide.md)
- [UMAP visualization guide](docs/guides/umap_visualization_guide.md)
- [Zero-shot QA with LLM baselines](docs/guides/zero_shot_qa_llm_baseline_guide.md)
- [Query materials database](docs/guides/query_materials_guide.md)

**Evaluation Documentation** (`docs/evaluation/`):
- [Evaluation results guide](docs/evaluation/evaluation_results_guide.md)

## Acknowledgement
- Pytorch implementation of [PaiNN](https://github.com/MaxH1996/PaiNN-in-PyG/blob/main/)
- Self-supervised learning strategies from [CLIP-Lite](https://github.com/4m4n5/CLIP-Lite)