#!/bin/bash

# Script to run text embedding similarity analysis for all properties
# Usage:
#   bash run_text_embedding_analysis.sh [model_checkpoint] [device]
#   bash run_text_embedding_analysis.sh untrained [device] [config_file]

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
    # Extract model name from checkpoint path (e.g., outputs/model_name/checkpoint.ckpt -> model_name)
    # This works for both relative and absolute paths
    MODEL_NAME=$(echo "$MODEL_CKPT" | sed 's|.*/outputs/||' | sed 's|/.*||')
fi

OUTPUT_DIR="outputs/text_embedding_similarity/${MODEL_NAME}"

echo "================================================================================"
echo "Text Embedding Similarity Analysis - All Properties"
echo "================================================================================"
if [ "$USE_UNTRAINED" = true ]; then
    echo "Model: Untrained (random initialization)"
    echo "Config file: $CONFIG"
else
    echo "Model checkpoint: $MODEL_CKPT"
fi
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_DIR"
echo "Note: Grid spacing is automatically set per property to ensure clean reference values"
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

# Property configurations: "property_name|min|max|reference"
PROPERTIES=(
    "band gap|0.0|4.0|2.0"
    "formation energy per atom|-3.0|1.0|-1.0"
    "density|2.0|10.0|5.0"
    "scintillation attenuation length|0.5|20.0|5.0"
    "total magnetization|0.0|50.0|10.0"
    "energy above hull|0.0|1.0|0.2"
)

# Counter for progress
TOTAL=${#PROPERTIES[@]}
CURRENT=0

# Loop through all properties
for prop_config in "${PROPERTIES[@]}"; do
    ((CURRENT++))

    # Parse configuration
    IFS='|' read -r PROPERTY MIN MAX REF <<< "$prop_config"

    echo ""
    echo "================================================================================"
    echo "[$CURRENT/$TOTAL] Analyzing: $PROPERTY"
    echo "================================================================================"
    echo "Range: $MIN to $MAX"
    echo "Reference value: $REF"
    echo ""

    # Run analysis (n-values is auto-determined by property spacing)
    if [ "$USE_UNTRAINED" = true ]; then
        # Run with untrained model (use raw LLM features, no projection)
        python evaluation/analyze_text_embedding_similarity.py \
            --config "$CONFIG" \
            --property "$PROPERTY" \
            --min-value "$MIN" \
            --max-value "$MAX" \
            --reference-value "$REF" \
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
            --reference-value "$REF" \
            --device "$DEVICE" \
            --output-dir "$OUTPUT_DIR"
    fi

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "[SUCCESS] Completed analysis for: $PROPERTY"
    else
        echo ""
        echo "[ERROR] Failed to analyze: $PROPERTY"
        exit 1
    fi
done

echo ""
echo "================================================================================"
echo "ALL ANALYSES COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/similarity_analysis_*.png 2>/dev/null || echo "  (No PNG files found)"
echo ""
echo "Summary:"
echo "  Total properties analyzed: $TOTAL"
echo "  Grid spacing: Auto-determined per property for clean reference values"
echo "================================================================================"
