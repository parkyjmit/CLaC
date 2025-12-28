#!/usr/bin/env bash
set -euo pipefail

# Simple driver to run common ablation sweeps via Hydra multi-run.
# Usage:
#   bash scripts/run_ablation.sh [extra hydra overrides]
# Notes:
# - All runs inherit your overrides (e.g., exp_name, data path) via "$@".
# - Each sweep sets its own exp_name suffix for easier grouping.

TS=$(date +%Y%m%d_%H%M%S)
EXP_PREFIX="ablation_${TS}"

# Common reproducibility flags (can be overridden via CLI)
COMMON_FLAGS=("trainer.deterministic=true" "trainer.random_seed=42")

echo "[run] baseline"
python train.py \
  exp_name="${EXP_PREFIX}_baseline" \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "[run] augmentation on/off"
python train.py -m \
  exp_name="${EXP_PREFIX}_augmentation" \
  model.augmentation=true,false \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "[run] lm_weight sweep (0.1 vs 0.0)"
python train.py -m \
  exp_name="${EXP_PREFIX}_lm" \
  model.lm_weight=0.1,0.0 \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "[run] priors (image/text) on/off (2x2)"
python train.py -m \
  exp_name="${EXP_PREFIX}_priors" \
  model.loss.image_prior=true,false \
  model.loss.text_prior=true,false \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "[run] self-supervised auxiliaries on/off (2x2)"
python train.py -m \
  exp_name="${EXP_PREFIX}_ssl" \
  model.loss.visual_self_supervised=true,false \
  model.loss.textual_self_supervised=true,false \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "[run] text encoder family swap"
python train.py -m \
  exp_name="${EXP_PREFIX}_text_encoder" \
  model.datamodule.tokenizer_model=m3rg-iitd/matscibert,facebook/galactica-125m \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "[run] loss type (dot vs concat)"
python train.py -m \
  exp_name="${EXP_PREFIX}_loss_type" \
  model.loss=jsd_dot,jsd_concat \
  "${COMMON_FLAGS[@]}" \
  "$@"

echo "All sweeps submitted. To aggregate results:"
echo "  python tools/collect_ablation.py"

