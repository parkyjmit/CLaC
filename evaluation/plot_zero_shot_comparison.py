#!/usr/bin/env python3
"""
Plot and compare zero-shot evaluation results across multiple models.

This script reads JSON results from zero-shot retrieval/QA evaluations
and creates comparison plots with bootstrap confidence intervals.

Usage:
    # Compare specific models
    python evaluation/plot_zero_shot_comparison.py \
        --evaluation-type retrieval \
        --models "model1_last" "model2_last" "untrained_baseline" \
        --output comparison.png

    # Compare all available results
    python evaluation/plot_zero_shot_comparison.py \
        --evaluation-type retrieval \
        --auto-discover \
        --output comparison.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import List, Dict, Tuple

# Set publication-quality defaults
mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Arial', size=5)
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def load_retrieval_results(model_name: str, base_dir: Path = Path('outputs/zero_shot_retrieval')) -> Dict:
    """Load zero-shot retrieval results for a model."""
    result_file = base_dir / model_name / 'retrieval_text.json'

    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, 'r') as f:
        data = json.load(f)

    return data


def load_qa_results(model_name: str, label: str, base_dir: Path = Path('outputs/zero_shot_qa')) -> Dict:
    """Load zero-shot QA results for a model and label."""
    result_file = base_dir / model_name / f'qa_{label}.json'

    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, 'r') as f:
        data = json.load(f)

    return data


def discover_models(evaluation_type: str) -> List[str]:
    """Auto-discover available model results."""
    if evaluation_type == 'retrieval':
        base_dir = Path('outputs/zero_shot_retrieval')
    else:  # qa
        base_dir = Path('outputs/zero_shot_qa')

    if not base_dir.exists():
        return []

    models = [d.name for d in base_dir.iterdir() if d.is_dir()]
    return sorted(models)


def shorten_model_name(name: str, max_length: int = 40) -> str:
    """Shorten long model names for display."""
    if len(name) <= max_length:
        return name

    # Try to keep important parts
    if 'untrained' in name.lower():
        return 'Untrained'
    if 'random' in name.lower():
        return 'Random'

    # Truncate middle
    half = max_length // 2 - 2
    return f"{name[:half]}...{name[-half:]}"


def plot_retrieval_comparison(models: List[str], output_path: str,
                              show_random: bool = True,
                              figsize: Tuple[float, float] = (2.4, 2.4)):
    """
    Create comparison plot for zero-shot retrieval results.

    Args:
        models: List of model names (directory names in zero_shot_retrieval/)
        output_path: Path to save the output figure
        show_random: Whether to include random baseline in comparison
        figsize: Figure size (width, height)
    """
    print(f"Loading results for {len(models)} models...")

    # Load results for all models
    results = {}
    random_baseline_data = None

    for model in models:
        try:
            data = load_retrieval_results(model)
            results[model] = data
            print(f"  ✓ Loaded: {model}")

            # Extract random baseline data (same across all models)
            if random_baseline_data is None and 'random_baseline' in data:
                random_baseline_data = data['random_baseline']
                print(f"  ✓ Found random baseline data")
        except FileNotFoundError as e:
            print(f"  ✗ Skipped: {model} (file not found)")

    if not results:
        print("ERROR: No valid results found!")
        return

    # Reorder results: Random → Untrained → User-specified models
    ordered_results = {}

    # 1. Add random baseline first (if available)
    if show_random and random_baseline_data is not None:
        ordered_results['Random Baseline'] = {
            'overall_metrics': random_baseline_data
        }
        print(f"  ✓ Added random baseline to comparison")

    # 2. Add untrained baseline second (if exists)
    untrained_keys = [k for k in results.keys() if 'untrained' in k.lower()]
    for key in untrained_keys:
        ordered_results[key] = results[key]

    # 3. Add remaining models in user-specified order
    for model in models:
        if model in results and model not in untrained_keys:
            ordered_results[model] = results[model]

    results = ordered_results

    # Prepare data for plotting
    metrics = ['top1_accuracy', 'top3_accuracy', 'top10_accuracy']
    metric_labels = ['Top-1', 'Top-3', 'Top-10']

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.7

    # Color scheme
    colors = {
        'untrained': '#FF6B6B',  # Red
        'random': '#95A5A6',     # Gray
        'trained': '#4ECDC4',    # Teal
    }

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]

        means = []
        errors_lower = []
        errors_upper = []
        bar_colors = []

        for model in model_names:
            data = results[model]

            # Get model metrics
            model_metric = data['overall_metrics'][metric]
            mean = model_metric['mean']
            ci_lower = model_metric['ci_lower']
            ci_upper = model_metric['ci_upper']

            means.append(mean)
            errors_lower.append(mean - ci_lower)
            errors_upper.append(ci_upper - mean)

            # Assign color
            if 'untrained' in model.lower():
                bar_colors.append(colors['untrained'])
            elif 'random' in model.lower():
                bar_colors.append(colors['random'])
            else:
                bar_colors.append(colors['trained'])

        # Create bar plot with error bars (CI)
        bars = ax.bar(x, means, width, color=bar_colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)

        # Add error bars (asymmetric for CI)
        ax.errorbar(x, means, yerr=[errors_lower, errors_upper],
                   fmt='none', ecolor='black', capsize=3,
                   linewidth=1, capthick=1, alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[i] + 0.02,
                   f'{mean_val*100:.1f}%',
                   ha='center', va='bottom', fontsize=5, fontweight='bold')

        # Formatting
        ax.set_ylabel('Accuracy', fontsize=5)
        ax.set_title(f'{label} Accuracy', fontsize=5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([shorten_model_name(m, 25) for m in model_names], #rotation=45, 
                           ha='right', fontsize=5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['trained'], edgecolor='black', label='Trained Model'),
        Patch(facecolor=colors['untrained'], edgecolor='black', label='Untrained Baseline'),
    ]
    if show_random:
        legend_elements.append(
            Patch(facecolor=colors['random'], edgecolor='black', label='Random Baseline')
        )

    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=5,
              frameon=True, edgecolor='black', fancybox=False)

    # plt.suptitle('Zero-Shot Retrieval Performance Comparison',
    #             fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure (create directory if needed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def plot_qa_comparison(models: List[str], label: str, output_path: str,
                      show_random: bool = True,
                      figsize: Tuple[float, float] = (2.4, 2.4),
                      custom_labels: List[str] = None):
    """
    Create comparison plot for zero-shot QA results.

    Args:
        models: List of model names (directory names in zero_shot_qa/)
        label: QA label (e.g., 'structure_question_list')
        output_path: Path to save the output figure
        show_random: Whether to include random baseline
        figsize: Figure size (width, height)
        custom_labels: Optional custom labels for x-axis (must match length of models)
    """
    print(f"Loading QA results for {len(models)} models (label: {label})...")

    # Load results for all models
    results = {}
    random_baseline_data = None

    for model in models:
        try:
            data = load_qa_results(model, label)
            results[model] = data
            print(f"  ✓ Loaded: {model}")

            # Extract random baseline data (same across all models)
            if random_baseline_data is None and 'random_baseline' in data:
                random_baseline_data = data['random_baseline']
                print(f"  ✓ Found random baseline data")
        except FileNotFoundError as e:
            print(f"  ✗ Skipped: {model} (file not found)")

    if not results:
        print("ERROR: No valid results found!")
        return

    # Reorder results: Random → User-specified models
    ordered_results = {}

    # 1. Add random baseline first (if available)
    if show_random and random_baseline_data is not None:
        ordered_results['Random Baseline'] = {
            'accuracy': random_baseline_data
        }
        print(f"  ✓ Added random baseline to comparison")

    # 2. Add models in user-specified order
    for model in models:
        if model in results:
            ordered_results[model] = results[model]

    results = ordered_results

    # Adjust custom_labels to account for random baseline
    if custom_labels:
        if show_random and random_baseline_data is not None:
            # Prepend 'Random' label
            adjusted_labels = ['Random'] + list(custom_labels)
        else:
            adjusted_labels = list(custom_labels)
    else:
        adjusted_labels = None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.7

    # Color scheme: Color-blind friendly palette (Wong 2011)
    palette = [
        '#000000',  # Black
        '#e69f00',  # Orange
        '#56b4e9',  # Sky blue
        '#009e73',  # Bluish green
        '#f0e442',  # Yellow
        '#0072b2',  # Blue
        '#d55e00',  # Vermillion
        '#cc79a7',  # Reddish purple
    ]

    means = []
    stds = []
    errors_lower = []
    errors_upper = []
    bar_colors = []

    color_idx = 0
    for model in model_names:
        data = results[model]

        # Get accuracy metrics
        acc = data['accuracy']
        mean = acc['mean']
        std = acc['std']
        ci_lower = acc['ci_lower']
        ci_upper = acc['ci_upper']

        means.append(mean)
        stds.append(std)
        errors_lower.append(mean - ci_lower)
        errors_upper.append(ci_upper - mean)

        # Assign color from palette
        bar_colors.append(palette[color_idx % len(palette)])
        color_idx += 1

    # Create bar plot with error bars (CI)
    bars = ax.bar(x, means, width, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.3)

    # Add error bars (asymmetric for CI)
    ax.errorbar(x, means, yerr=[errors_lower, errors_upper],
               fmt='none', ecolor='black', capsize=4,
               linewidth=0.3, capthick=0.3, alpha=0.7)

    # Add value labels on bars with error
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[i] + 0.02,
               f'{mean_val*100:.1f}±{std_val*100:.1f}%',
               ha='center', va='bottom', fontsize=5)#, fontweight='bold')

    # Formatting
    ax.set_ylabel('Accuracy', fontsize=5)
    # ax.set_title(f'Zero-Shot QA Performance: {label.replace("_", " ").title()}',
    #             fontsize=12, fontweight='bold')
    ax.set_xticks(x)

    # Use custom labels if provided, otherwise use shortened model names
    if adjusted_labels:
        if len(adjusted_labels) != len(model_names):
            print(f"Warning: custom_labels length ({len(adjusted_labels)}) != models length ({len(model_names)})")
            print("Using default model names instead")
            display_labels = [shorten_model_name(m, 30) for m in model_names]
        else:
            display_labels = adjusted_labels
    else:
        display_labels = [shorten_model_name(m, 30) for m in model_names]

    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Legend removed - each model has distinct color from color-blind friendly palette
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor=palette[i], edgecolor='black', label=model_names[i])
    #     for i in range(len(model_names))
    # ]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
    #          frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()

    # Save figure (create directory if needed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare zero-shot evaluation results across models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific models for retrieval
  python plot_zero_shot_comparison.py \\
      --evaluation-type retrieval \\
      --models "model1_last" "model2_last" "untrained_baseline" \\
      --output retrieval_comparison.png

  # Auto-discover and compare all retrieval results
  python plot_zero_shot_comparison.py \\
      --evaluation-type retrieval \\
      --auto-discover \\
      --output retrieval_all.png

  # Compare QA results
  python plot_zero_shot_comparison.py \\
      --evaluation-type qa \\
      --label structure_question_list \\
      --models "model1_last" "model2_last" \\
      --output qa_comparison.png
        """
    )

    parser.add_argument('--evaluation-type', type=str, required=True,
                       choices=['retrieval', 'qa'],
                       help='Type of evaluation to compare')

    parser.add_argument('--models', type=str, nargs='+',
                       help='List of model names (directory names) to compare')

    parser.add_argument('--auto-discover', action='store_true',
                       help='Automatically discover all available results')

    parser.add_argument('--label', type=str, default='structure_question_list',
                       choices=['structure_question_list', 'composition_question_list', 'oxide_question_list'],
                       help='QA label to compare (only for evaluation-type=qa)')

    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output file path (supports .png, .pdf, .svg)')

    parser.add_argument('--no-random', action='store_true',
                       help='Exclude random baseline from plot')

    parser.add_argument('--figsize', type=float, nargs=2, default=None,
                       help='Figure size (width height) in inches')

    parser.add_argument('--labels', type=str, nargs='+',
                       help='Custom labels for x-axis (must match number of models)')

    args = parser.parse_args()

    # Determine models to plot
    if args.auto_discover:
        models = discover_models(args.evaluation_type)
        if not models:
            print(f"ERROR: No results found in outputs/zero_shot_{args.evaluation_type}/ directory")
            return
        print(f"Auto-discovered {len(models)} models:")
        for m in models:
            print(f"  - {m}")
    elif args.models:
        models = args.models
    else:
        print("ERROR: Must specify either --models or --auto-discover")
        parser.print_help()
        return

    # Create comparison plot
    print(f"\nCreating {args.evaluation_type} comparison plot...")

    if args.evaluation_type == 'retrieval':
        figsize = args.figsize if args.figsize else (10, 4)
        plot_retrieval_comparison(
            models=models,
            output_path=args.output,
            show_random=not args.no_random,
            figsize=figsize
        )
    else:  # qa
        figsize = args.figsize if args.figsize else (2.4, 2.4)
        plot_qa_comparison(
            models=models,
            label=args.label,
            output_path=args.output,
            show_random=not args.no_random,
            figsize=figsize,
            custom_labels=args.labels
        )

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
