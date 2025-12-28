#!/bin/bash

# Inverse Design Evaluation Sweep
# This script runs inverse design evaluation for all model checkpoints and properties
# Based on sweep.yaml configuration

# Configuration
BASE_DIR="/home/lucky/Projects/CLaC-revision/outputs"
CHECKPOINT_NAME="last.ckpt"
DEVICE="cuda:3"
BATCH_SIZE=64
K_VALUES="3 10 30"

# Array of model configurations (from sweep.yaml)
MODELS=(
    "painn-infonce-galactica-125m-intra_True"
    "painn-infonce-galactica-125m-intra_False"
    "painn-infonce-matscibert-intra_True"
    "painn-infonce-matscibert-intra_False"
    "painn-infonce-opt-125m-intra_True"
    "painn-infonce-opt-125m-intra_False"
    "painn-infonce-scibert-intra_True"
    "painn-infonce-scibert-intra_False"
    "painn-jsd_dot-galactica-125m-intra_True"
    "painn-jsd_dot-galactica-125m-intra_False"
    "painn-jsd_dot-matscibert-intra_True"
    "painn-jsd_dot-matscibert-intra_False"
    "painn-jsd_dot-opt-125m-intra_True"
    "painn-jsd_dot-opt-125m-intra_False"
    "painn-jsd_dot-scibert-intra_True"
    "painn-jsd_dot-scibert-intra_False"
    "orb-infonce-galactica-125m-intra_True"
    "orb-infonce-galactica-125m-intra_False"
    "orb-infonce-matscibert-intra_True"
    "orb-infonce-matscibert-intra_False"
    "orb-infonce-opt-125m-intra_True"
    "orb-infonce-opt-125m-intra_False"
    "orb-infonce-scibert-intra_True"
    "orb-infonce-scibert-intra_False"
    "orb-jsd_dot-galactica-125m-intra_True"
    "orb-jsd_dot-galactica-125m-intra_False"
    "orb-jsd_dot-matscibert-intra_True"
    "orb-jsd_dot-matscibert-intra_False"
    "orb-jsd_dot-opt-125m-intra_True"
    "orb-jsd_dot-opt-125m-intra_False"
    "orb-jsd_dot-scibert-intra_True"
    "orb-jsd_dot-scibert-intra_False"
    "cgcnn-infonce-galactica-125m-intra_True"
    "cgcnn-infonce-galactica-125m-intra_False"
    "cgcnn-infonce-matscibert-intra_True"
    "cgcnn-infonce-matscibert-intra_False"
    "cgcnn-infonce-opt-125m-intra_True"
    "cgcnn-infonce-opt-125m-intra_False"
    "cgcnn-infonce-scibert-intra_True"
    "cgcnn-infonce-scibert-intra_False"
)

# Array of all available properties
PROPERTIES=(
    "band gap"
    "density"
    "scintillation attenuation length"
    "total magnetization"
    "energy above hull"
)

echo "========================================"
echo "Inverse Design Evaluation Sweep"
echo "========================================"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "K values: $K_VALUES"
echo "Models to evaluate: ${#MODELS[@]}"
echo "Properties per model: ${#PROPERTIES[@]}"
echo "Total evaluations: $((${#MODELS[@]} * ${#PROPERTIES[@]}))"
echo "========================================"
echo ""

# Track statistics
TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_EVALUATIONS=$((${#MODELS[@]} * ${#PROPERTIES[@]}))

# Run evaluation for each model and property
for model_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$model_idx]}"
    MODEL_CKPT="${BASE_DIR}/${MODEL}/${CHECKPOINT_NAME}"

    echo ""
    echo "========================================"
    echo "MODEL [$((model_idx+1))/${#MODELS[@]}]: $MODEL"
    echo "========================================"

    # Check if checkpoint exists
    if [ ! -f "$MODEL_CKPT" ]; then
        echo "WARNING: Checkpoint not found: $MODEL_CKPT"
        echo "Skipping this model..."
        TOTAL_FAILED=$((TOTAL_FAILED + ${#PROPERTIES[@]}))
        continue
    fi

    echo "Checkpoint: $MODEL_CKPT"
    echo ""

    # Run evaluation for each property
    for prop_idx in "${!PROPERTIES[@]}"; do
        PROPERTY="${PROPERTIES[$prop_idx]}"
        EVAL_NUM=$((model_idx * ${#PROPERTIES[@]} + prop_idx + 1))

        echo "  [$EVAL_NUM/$TOTAL_EVALUATIONS] Property: $PROPERTY"
        echo "  ----------------------------------------"

        python evaluation/compute_metrics.py \
            --evaluation-method inverse-design \
            --property "$PROPERTY" \
            --k-values $K_VALUES \
            --model-ckpt "$MODEL_CKPT" \
            --device "$DEVICE" \
            --batch-size $BATCH_SIZE

        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  ERROR: Evaluation failed (exit code: $EXIT_CODE)"
            echo "  Continuing with next property..."
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        else
            echo "  SUCCESS: Completed evaluation"
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        fi

        echo ""
    done
done

echo ""
echo "========================================"
echo "Inverse Design Evaluation Sweep Complete!"
echo "========================================"
echo "Total evaluations: $TOTAL_EVALUATIONS"
echo "Successful: $TOTAL_SUCCESS"
echo "Failed: $TOTAL_FAILED"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($TOTAL_SUCCESS/$TOTAL_EVALUATIONS)*100}")%"
echo "========================================"
