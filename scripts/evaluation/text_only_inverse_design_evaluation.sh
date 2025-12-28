#!/bin/bash

# Text-Only Inverse Design Evaluation Script
# This script evaluates the model's ability to retrieve materials based on
# application descriptions WITHOUT explicit property values.

# Configuration
MODEL_CKPT="${1:-outputs/your-model.ckpt}"
DATA_PATH="${2:-datafiles/mp_3d_2020_gpt_narratives}"
DEVICE="${3:-cuda:0}"

echo "=========================================="
echo "Text-Only Inverse Design Evaluation"
echo "=========================================="
echo "Model: $MODEL_CKPT"
echo "Data:  $DATA_PATH"
echo "Device: $DEVICE"
echo ""

# Band Gap
echo "[1/5] Evaluating Band Gap..."
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "band gap" \
    --k-values 3 10 30 \
    --model-ckpt "$MODEL_CKPT" \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --plot

echo ""

# Density
echo "[2/5] Evaluating Density..."
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "density" \
    --k-values 3 10 30 \
    --model-ckpt "$MODEL_CKPT" \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --plot

echo ""

# Scintillation Attenuation Length
echo "[3/5] Evaluating Scintillation Attenuation Length..."
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "scintillation attenuation length" \
    --k-values 3 10 30 \
    --model-ckpt "$MODEL_CKPT" \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --plot

echo ""

# Total Magnetization
echo "[4/5] Evaluating Total Magnetization..."
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "total magnetization" \
    --k-values 3 10 30 \
    --model-ckpt "$MODEL_CKPT" \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --plot

echo ""

# Energy Above Hull
echo "[5/5] Evaluating Energy Above Hull..."
python evaluation/compute_metrics.py \
    --evaluation-method text-only-inverse-design \
    --property "energy above hull" \
    --k-values 3 10 30 \
    --model-ckpt "$MODEL_CKPT" \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --plot

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: text_only_inverse_design/<checkpoint_name>/"
echo "=========================================="
