"""
Replot text-only inverse design results from existing JSON files.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_metrics import plot_text_only_inverse_design_results

def main():
    # Path to results
    result_dir = Path("outputs/text_only_inverse_design/evaluation_results")

    # Properties to replot
    properties = [
        "band_gap",
        "density",
        "energy_above_hull",
        "scintillation_attenuation_length",
        "total_magnetization"
    ]

    for prop in properties:
        json_file = result_dir / f"text_only_inverse_design_{prop}.json"

        if not json_file.exists():
            print(f"[SKIP] {json_file} not found")
            continue

        print(f"\n[Replotting] {prop}...")

        # Load JSON
        with open(json_file, 'r') as f:
            results = json.load(f)

        # Create config namespace
        cfg = SimpleNamespace(
            property=results['property'],
            k_values=results['k_values'],
            plot=True
        )

        # Replot
        plot_text_only_inverse_design_results(results, cfg, result_dir)
        print(f"[DONE] {prop}")

    print("\n[SUCCESS] All plots regenerated!")

if __name__ == "__main__":
    main()
