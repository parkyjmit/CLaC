#!/bin/bash

# Material Query Tool - Example Usage Script
# This script demonstrates various query patterns

# Configuration
CHECKPOINT="outputs/your-model/last.ckpt"  # Update with your checkpoint path
CACHE="embeddings_cache.pt"
DEVICE="cuda:0"

echo "========================================="
echo "CLaC Material Query Tool - Examples"
echo "========================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script"
    exit 1
fi
# Example 0: Li-containing materials for solid-state electrolyte
echo "Example 0: Li-containing materials for solid-state electrolyte"
echo "---------------------------------------------------------------"
python tools/query_materials.py \
  --query "Materials containing Li ions with high bandgap for solid-state electrolyte candidate" \
  --model-ckpt "$CHECKPOINT" \
  --device "$DEVICE" \
  --cache-embeddings "$CACHE" \
  --k 30

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 1: UV Detection Materials
# echo "Example 1: UV Detection Materials"
# echo "---------------------------------"
# python tools/query_materials.py \
#   --query "wide band gap semiconductor for UV detection and photodetectors" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 10

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 2: Solar Cell Materials
# echo "Example 2: Solar Cell Materials"
# echo "--------------------------------"
# python tools/query_materials.py \
#   --query "semiconductor with optimal band gap for photovoltaic solar cells" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 10

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 3: Magnetic Materials
# echo "Example 3: Magnetic Materials"
# echo "-----------------------------"
# python tools/query_materials.py \
#   --query "ferromagnetic material for spintronic and data storage applications" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 10

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 4: Lightweight Materials
# echo "Example 4: Lightweight Materials"
# echo "---------------------------------"
# python tools/query_materials.py \
#   --query "lightweight stable material for aerospace structural applications" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 10

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 5: High-Density Materials
# echo "Example 5: High-Density Materials"
# echo "----------------------------------"
# python tools/query_materials.py \
#   --query "high-density material for radiation shielding and gamma ray protection" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 10

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 6: With Similarity Filtering (JSON output)
# echo "Example 6: High-Quality Matches Only (JSON output)"
# echo "---------------------------------------------------"
# python tools/query_materials.py \
#   --query "stable oxide semiconductor for catalysis and photocatalysis" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --min-similarity 0.75 \
#   --k 20 \
#   --format json \
#   --output stable_oxide_catalysts.json

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 7: Thermoelectric Materials (CSV output)
# echo "Example 7: Thermoelectric Materials (CSV output)"
# echo "-------------------------------------------------"
# python tools/query_materials.py \
#   --query "heavy element compound with low thermal conductivity for thermoelectric applications" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 15 \
#   --format csv \
#   --output thermoelectric_candidates.csv

# echo ""
# echo "Press Enter to continue to next example..."
# read

# # Example 8: Multi-Property Query
# echo "Example 8: Multi-Property Query"
# echo "--------------------------------"
# python tools/query_materials.py \
#   --query "lightweight, thermodynamically stable oxide with wide band gap for transparent conductive applications" \
#   --model-ckpt "$CHECKPOINT" \
#   --device "$DEVICE" \
#   --cache-embeddings "$CACHE" \
#   --k 10

echo ""
echo "========================================="
echo "All examples completed!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - embeddings_cache.pt (reusable embedding cache)"
echo "  - stable_oxide_catalysts.json"
echo "  - thermoelectric_candidates.csv"
echo ""
echo "Tip: The embedding cache is now created and can be reused"
echo "for instant queries! Try your own:"
echo ""
echo "python tools/query_materials.py \\"
echo "  --query \"YOUR QUERY HERE\" \\"
echo "  --model-ckpt \"$CHECKPOINT\" \\"
echo "  --cache-embeddings \"$CACHE\""
