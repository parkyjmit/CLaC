#!/bin/bash

# Zero-Shot Retrieval Evaluation Sweep
# This script runs zero-shot retrieval evaluation for all model checkpoints
# Based on sweep.yaml configuration

# Configuration
BASE_DIR="outputs"
CHECKPOINT_NAME="last.ckpt"
DEVICE="cuda:1"
BATCH_SIZE=128

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

echo "========================================"
echo "Zero-Shot Retrieval Evaluation Sweep"
echo "========================================"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Models to evaluate: ${#MODELS[@]}"
echo "Total evaluations: ${#MODELS[@]}"
echo "========================================"
echo ""

# Track statistics
TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_EVALUATIONS=${#MODELS[@]}

# Run evaluation for each model (using general text descriptions, not questions)
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
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        continue
    fi

    echo "Checkpoint: $MODEL_CKPT"
    echo ""

    # Run zero-shot retrieval evaluation (uses general text descriptions)
    python evaluation/compute_metrics.py \
        --evaluation-method "zero-shot retrieval" \
        --label "text" \
        --model-ckpt "$MODEL_CKPT" \
        --device "$DEVICE" \
        --batch-size $BATCH_SIZE \
        --graphdatatype auto

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Evaluation failed (exit code: $EXIT_CODE)"
        echo "Continuing with next model..."
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    else
        echo "SUCCESS: Completed evaluation"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    fi

    echo ""
done

echo ""
echo "========================================"
echo "Zero-Shot Retrieval Evaluation Complete!"
echo "========================================"
echo "Total evaluations: $TOTAL_EVALUATIONS"
echo "Successful: $TOTAL_SUCCESS"
echo "Failed: $TOTAL_FAILED"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($TOTAL_SUCCESS/$TOTAL_EVALUATIONS)*100}")%"
echo "========================================"

echo ""
echo ""
echo "========================================"
echo "UNTRAINED BASELINE EVALUATION"
echo "========================================"
echo "Evaluating untrained (randomly initialized) model as baseline..."
echo ""

# Run untrained baseline evaluation
python evaluation/compute_metrics.py \
    --evaluation-method "zero-shot retrieval" \
    --label "text" \
    --use-untrained \
    --config config/model/clac-lite.yaml \
    --device "$DEVICE" \
    --batch-size $BATCH_SIZE \
    --graphdatatype auto

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Untrained baseline evaluation failed (exit code: $EXIT_CODE)"
else
    echo "SUCCESS: Untrained baseline evaluation completed"
    echo "Results saved to: zero_shot_retrieval/untrained_baseline/retrieval_text.json"
fi

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================"
echo "Trained models: $TOTAL_SUCCESS successful, $TOTAL_FAILED failed"
echo "Untrained baseline: $([ $EXIT_CODE -eq 0 ] && echo 'Success' || echo 'Failed')"
echo "========================================"
