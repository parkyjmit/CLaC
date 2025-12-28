#!/bin/bash
# Navigate to project root directory
cd "$(dirname "$0")/../.." || exit 1

# Generate All Zero-Shot Comparison Plots
# This script creates comparison plots for all evaluation results

OUTPUT_DIR="outputs/figures/zero_shot_comparisons"
mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Generating Zero-Shot Comparison Plots"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Zero-Shot Retrieval Plots
# ============================================================================

echo "1. Zero-Shot Retrieval - All Models"
echo "------------------------------------"
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --output $OUTPUT_DIR/retrieval_all_models.png \
    --figsize 12 4

if [ $? -eq 0 ]; then
    echo "✓ Saved: $OUTPUT_DIR/retrieval_all_models.png"
else
    echo "✗ Failed to generate retrieval comparison"
fi
echo ""

# PDF version for papers
echo "1b. Zero-Shot Retrieval - PDF version for paper"
echo "------------------------------------------------"
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --output $OUTPUT_DIR/retrieval_all_models.pdf \
    --figsize 10 3.5

if [ $? -eq 0 ]; then
    echo "✓ Saved: $OUTPUT_DIR/retrieval_all_models.pdf"
fi
echo ""

# ============================================================================
# Zero-Shot QA Plots (All Labels)
# ============================================================================

echo "2. Zero-Shot QA - Structure Questions"
echo "--------------------------------------"
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label structure_question_list \
    --auto-discover \
    --output $OUTPUT_DIR/qa_structure.png \
    --figsize 10 5

if [ $? -eq 0 ]; then
    echo "✓ Saved: $OUTPUT_DIR/qa_structure.png"
else
    echo "✗ Failed to generate QA structure comparison"
fi
echo ""

echo "3. Zero-Shot QA - Composition Questions"
echo "----------------------------------------"
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label composition_question_list \
    --auto-discover \
    --output $OUTPUT_DIR/qa_composition.png \
    --figsize 10 5

if [ $? -eq 0 ]; then
    echo "✓ Saved: $OUTPUT_DIR/qa_composition.png"
else
    echo "✗ Failed to generate QA composition comparison"
fi
echo ""

echo "4. Zero-Shot QA - Oxide Questions"
echo "----------------------------------"
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type qa \
    --label oxide_question_list \
    --auto-discover \
    --output $OUTPUT_DIR/qa_oxide.png \
    --figsize 10 5

if [ $? -eq 0 ]; then
    echo "✓ Saved: $OUTPUT_DIR/qa_oxide.png"
else
    echo "✗ Failed to generate QA oxide comparison"
fi
echo ""

# ============================================================================
# Optional: Models without random baseline (cleaner for papers)
# ============================================================================

echo "5. Clean Comparison - Without Random Baseline"
echo "----------------------------------------------"
python evaluation/plot_zero_shot_comparison.py \
    --evaluation-type retrieval \
    --auto-discover \
    --no-random \
    --output $OUTPUT_DIR/retrieval_clean.pdf \
    --figsize 10 3.5

if [ $? -eq 0 ]; then
    echo "✓ Saved: $OUTPUT_DIR/retrieval_clean.pdf"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "Plot Generation Complete!"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh $OUTPUT_DIR/ | grep -E "\.(png|pdf|svg)$" | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "To view results:"
echo "  cd $OUTPUT_DIR && ls -la"
echo "========================================"
