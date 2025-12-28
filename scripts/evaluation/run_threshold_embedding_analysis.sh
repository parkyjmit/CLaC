#!/bin/bash

# Script to run threshold-based text embedding similarity analysis
# This analyzes how well the model understands range-based queries like "band gap > 1 eV"
#
# Usage:
#   bash run_threshold_embedding_analysis.sh [model_checkpoint] [device]
#   bash run_threshold_embedding_analysis.sh untrained [device] [config_file]

# Parse arguments
if [ "$1" == "untrained" ]; then
    USE_UNTRAINED=true
    MODEL_CKPT=""
    DEVICE="${2:-cuda:0}"
    CONFIG="${3:-config/model/clac-lite.yaml}"
    MODEL_NAME="untrained"
else
    USE_UNTRAINED=false
    MODEL_CKPT="${1:-outputs/painn-infonce-galactica-125m-intra_True/epoch=4-step=495.ckpt}"
    DEVICE="${2:-cuda:0}"
    CONFIG=""
    # Extract model name from checkpoint path
    MODEL_NAME=$(echo "$MODEL_CKPT" | sed 's|.*/outputs/||' | sed 's|/.*||')
fi

OUTPUT_DIR="outputs/text_embedding_similarity/${MODEL_NAME}/threshold_analysis"

echo "================================================================================"
echo "Threshold-Based Text Embedding Similarity Analysis"
echo "================================================================================"
if [ "$USE_UNTRAINED" = true ]; then
    echo "Model: Untrained (random initialization)"
    echo "Config file: $CONFIG"
else
    echo "Model checkpoint: $MODEL_CKPT"
fi
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This analysis tests range-based queries like:"
echo "  - 'material with band gap greater than 1 eV'"
echo "  - 'material with density less than 5 g/cm³'"
echo ""
echo "Expected behavior for trained models:"
echo "  - Higher similarity for values satisfying the condition"
echo "  - Lower similarity for values not satisfying the condition"
echo "  - Clear separation in similarity distributions"
echo "================================================================================"
echo ""

# Check if model checkpoint exists (only for trained model)
if [ "$USE_UNTRAINED" = false ] && [ ! -f "$MODEL_CKPT" ]; then
    echo "[ERROR] Model checkpoint not found: $MODEL_CKPT"
    exit 1
fi

# Check if config exists (only for untrained model)
if [ "$USE_UNTRAINED" = true ] && [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Threshold configurations: "property_name|min|max|threshold|mode"
# We test both "greater_than" and "less_than" for key properties
THRESHOLD_TESTS=(
    # Band gap tests
    "band gap|0.0|4.0|1.0|greater_than"
    "band gap|0.0|4.0|1.0|less_than"
    "band gap|0.0|4.0|2.0|greater_than"

    # Formation energy tests
    "formation energy per atom|-3.0|1.0|-1.0|greater_than"
    "formation energy per atom|-3.0|1.0|-1.0|less_than"

    # Density tests
    "density|2.0|10.0|5.0|greater_than"
    "density|2.0|10.0|5.0|less_than"

    # Scintillation attenuation length tests
    "scintillation attenuation length|0.5|20.0|5.0|greater_than"
    "scintillation attenuation length|0.5|20.0|10.0|greater_than"

    # Total magnetization tests
    "total magnetization|0.0|50.0|1.0|greater_than"
    "total magnetization|0.0|50.0|10.0|greater_than"

    # Energy above hull tests
    "energy above hull|0.0|1.0|0.1|greater_than"
    "energy above hull|0.0|1.0|0.1|less_than"
)

# Counter for progress
TOTAL=${#THRESHOLD_TESTS[@]}
CURRENT=0

# Loop through all threshold tests
for test_config in "${THRESHOLD_TESTS[@]}"; do
    ((CURRENT++))

    # Parse configuration
    IFS='|' read -r PROPERTY MIN MAX THRESHOLD MODE <<< "$test_config"

    echo ""
    echo "================================================================================"
    echo "[$CURRENT/$TOTAL] Testing: $PROPERTY"
    echo "================================================================================"
    echo "Range: $MIN to $MAX"
    echo "Threshold: $THRESHOLD ($MODE)"

    # Create readable description for the query
    if [ "$MODE" == "greater_than" ]; then
        CONDITION=">"
        DESCRIPTION="greater than $THRESHOLD"
    elif [ "$MODE" == "less_than" ]; then
        CONDITION="<"
        DESCRIPTION="less than $THRESHOLD"
    elif [ "$MODE" == "greater_equal" ]; then
        CONDITION=">="
        DESCRIPTION="at least $THRESHOLD"
    else  # less_equal
        CONDITION="<="
        DESCRIPTION="at most $THRESHOLD"
    fi

    echo "Query: values $CONDITION $THRESHOLD"
    echo ""

    # Run analysis
    if [ "$USE_UNTRAINED" = true ]; then
        # Run with untrained model (use raw LLM features, no projection)
        python evaluation/analyze_text_embedding_similarity.py \
            --config "$CONFIG" \
            --property "$PROPERTY" \
            --min-value "$MIN" \
            --max-value "$MAX" \
            --threshold-mode "$MODE" \
            --threshold-value "$THRESHOLD" \
            --device "$DEVICE" \
            --output-dir "$OUTPUT_DIR" \
            --no-projection
    else
        # Run with checkpoint (use projection layer)
        python evaluation/analyze_text_embedding_similarity.py \
            --model-ckpt "$MODEL_CKPT" \
            --property "$PROPERTY" \
            --min-value "$MIN" \
            --max-value "$MAX" \
            --threshold-mode "$MODE" \
            --threshold-value "$THRESHOLD" \
            --device "$DEVICE" \
            --output-dir "$OUTPUT_DIR"
    fi

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "[SUCCESS] Completed threshold test: $PROPERTY ($DESCRIPTION)"
    else
        echo ""
        echo "[ERROR] Failed threshold test: $PROPERTY ($DESCRIPTION)"
        exit 1
    fi
done

echo ""
echo "================================================================================"
echo "ALL THRESHOLD ANALYSES COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/threshold_analysis_*.png 2>/dev/null || echo "  (No PNG files found)"
echo ""
echo "Summary:"
echo "  Total threshold tests: $TOTAL"
echo ""
echo "Interpretation guide:"
echo "  - Orange points: Values satisfying the condition (should have higher similarity)"
echo "  - Blue points: Values not satisfying the condition (should have lower similarity)"
echo "  - For trained models: expect clear separation in similarity distributions"
echo "  - For untrained models: expect minimal or no separation"
echo "================================================================================"
