#!/usr/bin/env bash
set -euo pipefail

# ======================================================================================
# Ablation Study Script for CLaC Revision
#
# This script runs a prioritized sequence of ablation studies based on reviewer feedback.
# It's designed to test the most fundamental claims first.
#
# Usage:
#   bash scripts/run_priority_ablation.sh [extra hydra overrides]
#
# Example:
#   bash scripts/run_priority_ablation.sh trainer.pl_trainer.max_epochs=10
#
# Note: All runs inherit your overrides (e.g., exp_name, data path) via "$@".
# ======================================================================================

TS=$(date +%Y%m%d_%H%M%S)
EXP_PREFIX="ablation_${TS}"

# Common flags for reproducibility. Can be overridden from the command line.
COMMON_FLAGS=("trainer.deterministic=true" "trainer.random_seed=42")

echo "======================================================================"
echo "PHASE 1: Foundational Experiments (Highest Priority)"
echo "======================================================================"

# # --- 1.1: Modality Ablation (Multimodal vs. Unimodal) ---
# # Goal: Prove that the multimodal approach is better than using a single modality. (Addresses R4-Q2)
# echo "[PHASE 1.1] Running Modality Ablation..."

# # Multimodal (Baseline)
# echo "  > Running: Multimodal (Baseline)"
# python train.py \
#   exp_name="${EXP_PREFIX}_P1_multimodal_baseline" \
#   "${COMMON_FLAGS[@]}" \
#   "$@"

# # Graph-only
# # NOTE: Assumes 'graph_supervised' is a model config for a unimodal, graph-only setup.
# # You may need to adjust the model config name.
# echo "  > Running: Graph-only"
# python train.py \
#   exp_name="${EXP_PREFIX}_P1_graph_only" \
#   model=graph_supervised \
#   "${COMMON_FLAGS[@]}" \
#   "$@"

# # Text-only
# # NOTE: Assumes 'text_supervised' is a model config for a unimodal, text-only setup.
# # This likely requires creating a new config if it doesn't exist.
# echo "  > Running: Text-only (ASSUMES 'text_supervised' CONFIG EXISTS)"
# python train.py \
#   exp_name="${EXP_PREFIX}_P1_text_only" \
#   model=text_supervised \
#   "${COMMON_FLAGS[@]}" \
#   "$@"

# # --- 1.2: Data Source Ablation ---
# # Goal: Prove the quality of synthetic GPT data vs. academic paper data. (Addresses R1-Q3, R2-Q2)
# echo "[PHASE 1.2] Running Data Source Ablation..."

# # GPT Narratives (Baseline, already run above)
# echo "  > Using: Multimodal (Baseline) with GPT data (already run in P1.1)"

# # Academic Papers
# # NOTE: Assumes switching the text data type is done via 'model.datamodule.textdatatype'.
# echo "  > Running: Multimodal with Academic Papers"
# python train.py \
#   exp_name="${EXP_PREFIX}_P1_data_papers" \
#   model.datamodule.textdatatype=papers \
#   "${COMMON_FLAGS[@]}" \
#   "$@"


echo "======================================================================"
echo "PHASE 2: Novelty & Performance Validation (Medium Priority)"
echo "======================================================================"

# --- 2.1: Pre-training Ablation ---
# Goal: Show the benefit of initializing from a pre-trained graph encoder. (Addresses R1-A2)
echo "[PHASE 2.1] Running Pre-training Ablation..."

# From Scratch (Baseline, already run above)
echo "  > Using: Multimodal (Baseline) trained from scratch (already run in P1.1)"

# With Pre-trained Graph Encoder
# NOTE: Assumes you can switch to a pre-trained model by changing the model config.
# You may need to create/adjust a config like 'clac_mlip_init.yaml'.
echo "  > Running: Multimodal with Pre-trained Encoder (ASSUMES 'clac_mlip_init' CONFIG EXISTS)"
python train.py \
  exp_name="${EXP_PREFIX}_P2_pretrained_init" \
  model=clac_mlip_init \
  "${COMMON_FLAGS[@]}" \
  "$@"

# # --- 2.2: Loss Function Ablation ---
# # Goal: Validate the effectiveness of JSD loss against a standard baseline. (Addresses R1-Q2, R1-Q3)
# echo "[PHASE 2.2] Running Loss Function Ablation..."

# # JSD Loss (Baseline, already run above)
# echo "  > Using: Multimodal (Baseline) with JSD loss (already run in P1.1)"

# # InfoNCE Loss
# # NOTE: Assumes a config 'model/loss=infonce.yaml' exists.
# echo "  > Running: Multimodal with InfoNCE Loss (ASSUMES 'infonce' CONFIG EXISTS)"
# python train.py \
#   exp_name="${EXP_PREFIX}_P2_loss_infonce" \
#   model/loss=infonce \
#   "${COMMON_FLAGS[@]}" \
#   "$@"



# echo "======================================================================"
# echo "PHASE 3: Deeper Analysis (Lower Priority)"
# echo "======================================================================"

# # --- 3.1: Graph Augmentation Ablation ---
# # Goal: Understand the impact of graph augmentations. (Addresses R2-Q5)
# echo "[PHASE 3.1] Running Graph Augmentation Ablation..."

# # With Augmentations (Baseline, already run above)
# echo "  > Using: Multimodal (Baseline) with augmentations (already run in P1.1)"

# # No Augmentations
# # NOTE: Assumes augmentations can be turned off by setting the config to null.
# echo "  > Running: Multimodal with NO Graph Augmentations"
# python train.py \
#   exp_name="${EXP_PREFIX}_P3_no_augmentation" \
#   model.graph_augmentation=null \
#   "${COMMON_FLAGS[@]}" \
#   "$@"



echo "======================================================================"
echo "All prioritized ablation studies submitted."
echo "To aggregate results, you may need to adapt 'tools/collect_ablation.py'."
echo "======================================================================"
