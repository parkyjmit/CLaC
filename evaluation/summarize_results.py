"""
Summarize evaluation results from JSON files.

This script collects and displays results from:
- Zero-shot retrieval evaluations
- Zero-shot QA evaluations
- Inverse design evaluations
- Dual-property inverse design evaluations
"""

import json
from pathlib import Path
import pandas as pd
import argparse


def load_json_safe(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def summarize_zero_shot_retrieval(results_dir='zero_shot_retrieval'):
    """Summarize zero-shot retrieval results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return None

    print("\n" + "="*80)
    print("ZERO-SHOT RETRIEVAL RESULTS")
    print("="*80)

    data = []
    for ckpt_dir in sorted(results_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        model_name = ckpt_dir.name
        for json_file in sorted(ckpt_dir.glob('*.json')):
            result = load_json_safe(json_file)
            if result:
                label = result.get('label', 'unknown')
                metrics = result.get('overall_metrics', {})
                data.append({
                    'Model': model_name,
                    'Label': label,
                    'Top-1': f"{metrics.get('top1_accuracy', 0)*100:.2f}%",
                    'Top-3': f"{metrics.get('top3_accuracy', 0)*100:.2f}%",
                    'Top-10': f"{metrics.get('top10_accuracy', 0)*100:.2f}%",
                    'Batches': result.get('num_batches', 0)
                })

    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        return df
    else:
        print("No results found.")
        return None


def summarize_zero_shot_qa(results_dir='zero_shot_qa'):
    """Summarize zero-shot QA results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return None

    print("\n" + "="*80)
    print("ZERO-SHOT QA RESULTS")
    print("="*80)

    data = []
    for ckpt_dir in sorted(results_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        model_name = ckpt_dir.name
        for json_file in sorted(ckpt_dir.glob('*.json')):
            result = load_json_safe(json_file)
            if result:
                label = result.get('label', 'unknown')
                accuracy = result.get('accuracy', 0)
                correct = result.get('correct_answers', 0)
                total = result.get('total_questions', 0)
                data.append({
                    'Model': model_name,
                    'Label': label,
                    'Accuracy': f"{accuracy*100:.2f}%",
                    'Correct': f"{correct}/{total}"
                })

    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        return df
    else:
        print("No results found.")
        return None


def summarize_inverse_design(results_dir='inverse_design'):
    """Summarize inverse design results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return None

    print("\n" + "="*80)
    print("INVERSE DESIGN RESULTS")
    print("="*80)

    data = []
    for ckpt_dir in sorted(results_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        model_name = ckpt_dir.name
        for json_file in sorted(ckpt_dir.glob('*.json')):
            result = load_json_safe(json_file)
            if result:
                property_name = result.get('property', 'unknown')
                unit = result.get('unit', '')
                k_values = result.get('k_values', [])

                for k in k_values:
                    mae = result.get(f'k{k}_overall_mae', 0)
                    random_mae = result.get(f'random_k{k}_overall_mae', 0)
                    corr = result.get(f'k{k}_correlation', 0)
                    data.append({
                        'Model': model_name,
                        'Property': property_name,
                        'K': k,
                        f'MAE ({unit})': f"{mae:.4f}",
                        f'Random MAE': f"{random_mae:.4f}",
                        'Correlation': f"{corr:.4f}"
                    })

    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        return df
    else:
        print("No results found.")
        return None


def summarize_dual_inverse_design(results_dir='inverse_design_dual'):
    """Summarize dual-property inverse design results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return None

    print("\n" + "="*80)
    print("DUAL-PROPERTY INVERSE DESIGN RESULTS")
    print("="*80)

    data = []
    for ckpt_dir in sorted(results_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        model_name = ckpt_dir.name
        for json_file in sorted(ckpt_dir.glob('*.json')):
            result = load_json_safe(json_file)
            if result:
                prop1 = result.get('property1', 'unknown')
                prop2 = result.get('property2', 'unknown')
                unit1 = result.get('unit1', '')
                unit2 = result.get('unit2', '')
                k_values = result.get('k_values', [])

                for k in k_values:
                    mae1 = result.get(f'k{k}_overall_mae1', 0)
                    mae2 = result.get(f'k{k}_overall_mae2', 0)
                    mae_combined = result.get(f'k{k}_overall_mae_combined', 0)
                    data.append({
                        'Model': model_name,
                        'Property 1': prop1,
                        'Property 2': prop2,
                        'K': k,
                        f'MAE1 ({unit1})': f"{mae1:.4f}",
                        f'MAE2 ({unit2})': f"{mae2:.4f}",
                        'Combined MAE': f"{mae_combined:.4f}"
                    })

    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        return df
    else:
        print("No results found.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Summarize evaluation results')
    parser.add_argument('--eval-type', type=str,
                       choices=['retrieval', 'qa', 'inverse', 'dual', 'all'],
                       default='all',
                       help='Type of evaluation to summarize')
    parser.add_argument('--output', type=str, default=None,
                       help='Save summary to CSV file')

    args = parser.parse_args()

    results = {}

    if args.eval_type in ['retrieval', 'all']:
        results['retrieval'] = summarize_zero_shot_retrieval()

    if args.eval_type in ['qa', 'all']:
        results['qa'] = summarize_zero_shot_qa()

    if args.eval_type in ['inverse', 'all']:
        results['inverse'] = summarize_inverse_design()

    if args.eval_type in ['dual', 'all']:
        results['dual'] = summarize_dual_inverse_design()

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for eval_name, df in results.items():
            if df is not None:
                csv_file = output_path.parent / f"{output_path.stem}_{eval_name}.csv"
                df.to_csv(csv_file, index=False)
                print(f"\nSaved {eval_name} results to: {csv_file}")


if __name__ == '__main__':
    main()
