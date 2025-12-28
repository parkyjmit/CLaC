#!/bin/bash
# Run zero-shot QA evaluation for LLM baselines
#
# Usage:
#   bash zero_shot_qa_baseline_evaluation.sh facebook/galactica-125m cuda:0
#   bash zero_shot_qa_baseline_evaluation.sh facebook/galactica-1.3b cuda:0

MODEL_NAME=${1:-"facebook/galactica-125m"}
DEVICE=${2:-"cuda:0"}

echo "=================================="
echo "Zero-Shot QA LLM Baseline Evaluation"
echo "=================================="
echo "Model: $MODEL_NAME"
echo "Device: $DEVICE"
echo "=================================="

# Evaluate all question types
for LABEL in structure_question_list composition_question_list oxide_question_list; do
    echo ""
    echo "Evaluating: $LABEL"
    echo "----------------------------------"

    python evaluation/compute_metrics_baseline.py \
        --model-name "$MODEL_NAME" \
        --label "$LABEL" \
        --device "$DEVICE" \
        --batch-size 32 \
        --max-new-tokens 10

    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for $LABEL"
        exit 1
    fi
done

echo ""
echo "=================================="
echo "All evaluations completed!"
echo "Results saved to: zero_shot_qa_baseline/"
echo "=================================="
