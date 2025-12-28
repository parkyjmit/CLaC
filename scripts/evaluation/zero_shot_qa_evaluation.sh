#!/bin/bash

# Zero-Shot QA Evaluation Sweep
# This script runs zero-shot QA evaluation for all model checkpoints
# Based on sweep.yaml configuration

# Configuration
BASE_DIR="outputs"
CHECKPOINT_NAME="last.ckpt"
DEVICE="cuda:0"
BATCH_SIZE=64

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

# Array of question types to evaluate
LABELS=(
    "structure_question_list"
    "composition_question_list"
    "oxide_question_list"
)

echo "========================================"
echo "Zero-Shot QA Evaluation Sweep"
echo "========================================"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Models to evaluate: ${#MODELS[@]}"
echo "Labels per model: ${#LABELS[@]}"
echo "Total evaluations: $((${#MODELS[@]} * ${#LABELS[@]}))"
echo "========================================"
echo ""

# Track statistics
TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_EVALUATIONS=$((${#MODELS[@]} * ${#LABELS[@]}))

# Run evaluation for each model and label
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
        TOTAL_FAILED=$((TOTAL_FAILED + ${#LABELS[@]}))
        continue
    fi

    echo "Checkpoint: $MODEL_CKPT"
    echo ""

    # Run evaluation for each label
    for label_idx in "${!LABELS[@]}"; do
        LABEL="${LABELS[$label_idx]}"
        EVAL_NUM=$((model_idx * ${#LABELS[@]} + label_idx + 1))

        echo "  [$EVAL_NUM/$TOTAL_EVALUATIONS] Label: $LABEL"
        echo "  ----------------------------------------"

        python evaluation/compute_metrics.py \
            --evaluation-method "zero-shot QA" \
            --label "$LABEL" \
            --model-ckpt "$MODEL_CKPT" \
            --device "$DEVICE" \
            --batch-size $BATCH_SIZE \
            --graphdatatype auto

        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  ERROR: Evaluation failed (exit code: $EXIT_CODE)"
            echo "  Continuing with next label..."
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
echo "Zero-Shot QA Evaluation Complete!"
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
echo "Labels: ${LABELS[@]}"
echo ""

# Track untrained baseline results
UNTRAINED_SUCCESS=0
UNTRAINED_FAILED=0

# Run untrained baseline evaluation for each label
for label_idx in "${!LABELS[@]}"; do
    LABEL="${LABELS[$label_idx]}"
    EVAL_NUM=$((label_idx + 1))

    echo "  [$EVAL_NUM/${#LABELS[@]}] Label: $LABEL"
    echo "  ----------------------------------------"

    python evaluation/compute_metrics.py \
        --evaluation-method "zero-shot QA" \
        --label "$LABEL" \
        --use-untrained \
        --config config/model/clac-lite.yaml \
        --device "$DEVICE" \
        --batch-size $BATCH_SIZE \
        --graphdatatype auto

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ERROR: Untrained baseline evaluation failed (exit code: $EXIT_CODE)"
        UNTRAINED_FAILED=$((UNTRAINED_FAILED + 1))
    else
        echo "  SUCCESS: Untrained baseline evaluation completed"
        echo "  Results saved to: zero_shot_qa/untrained_baseline/qa_${LABEL}.json"
        UNTRAINED_SUCCESS=$((UNTRAINED_SUCCESS + 1))
    fi

    echo ""
done

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================"
echo "Trained models: $TOTAL_SUCCESS successful, $TOTAL_FAILED failed"
echo "Untrained baseline: $UNTRAINED_SUCCESS successful, $UNTRAINED_FAILED failed"
echo "========================================"
