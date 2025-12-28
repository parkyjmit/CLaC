#!/usr/bin/env python3
"""
Plot component-wise comparison (e.g., orb vs painn vs cgcnn) from pareto data.

This script aggregates results across all properties and configurations,
then compares average performance by component type (graph encoder, loss type, LLM type).
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import glob

# Configure matplotlib
mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Arial', size=6)
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# Define custom ordering for components
COMPONENT_ORDER = {
    'graph_encoder': ['orb', 'cgcnn', 'painn'],
    'loss_type': ['jsd_dot', 'infonce'],
    'llm_type': ['scibert', 'matscibert', 'opt-125m', 'galactica-125m'],
    'intra_loss': [False, True],
    'vision_intra': [False, True],
    'textual_intra': [False, True],
}


def sort_by_custom_order(data, component_type):
    """
    Sort DataFrame or Series by custom component order.

    Args:
        data: DataFrame or Series with component names as index
        component_type: Type of component (e.g., 'graph_encoder', 'loss_type')

    Returns:
        Sorted data
    """
    if component_type not in COMPONENT_ORDER:
        # If no custom order defined, return as is
        return data

    custom_order = COMPONENT_ORDER[component_type]

    # Get existing components in data
    existing_components = data.index.tolist()

    # Filter custom order to only include existing components
    ordered_components = [c for c in custom_order if c in existing_components]

    # Add any components not in custom order at the end
    remaining_components = [c for c in existing_components if c not in ordered_components]
    final_order = ordered_components + remaining_components

    # Reindex to apply the order
    return data.reindex(final_order)


def load_all_pareto_data(data_dir='pareto_results', pattern='pareto_data_*.csv'):
    """
    Load all pareto data CSV files and combine them.

    Args:
        data_dir: Directory containing CSV files
        pattern: Glob pattern for CSV files

    Returns:
        Combined DataFrame with all data
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {data_dir}/{pattern}")

    print(f"[Load] Found {len(csv_files)} CSV files")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"[Load] Loaded {csv_file.name}: {len(df)} rows")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"[Load] Total combined rows: {len(combined_df)}")

    return combined_df


def compute_component_averages(df, component_column, metrics):
    """
    Compute average metrics for each component type.

    Args:
        df: DataFrame with all data
        component_column: Column to group by (e.g., 'graph_encoder')
        metrics: List of metric columns to average

    Returns:
        DataFrame with average metrics per component
    """
    grouped = df.groupby(component_column)[metrics].mean()
    grouped_std = df.groupby(component_column)[metrics].std()

    return grouped, grouped_std


def plot_component_comparison(df, avg_data, std_data, component_name, metric_groups, output_dir):
    """
    Plot multi-panel comparison of components with bars + scatter overlay.

    Args:
        df: Original DataFrame with all data points
        avg_data: DataFrame with average metrics (index: components, columns: metrics)
        std_data: DataFrame with std metrics
        component_name: Name of component (for title)
        metric_groups: Dict mapping group names to lists of metrics
        output_dir: Output directory for plots
    """
    # Sort data by custom order
    avg_data = sort_by_custom_order(avg_data, component_name)
    std_data = sort_by_custom_order(std_data, component_name)

    n_groups = len(metric_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(2.4 * n_groups, 3.0))

    if n_groups == 1:
        axes = [axes]

    components = avg_data.index.tolist()
    n_components = len(components)

    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for ax_idx, (group_name, metrics) in enumerate(metric_groups.items()):
        ax = axes[ax_idx]

        # Number of metrics in this group
        n_metrics = len(metrics)

        # X positions for bars
        x = np.arange(n_metrics)
        width = 0.8 / n_components
        jitter_strength = 0.08  # Smaller jitter for bar plot

        # Plot for each component
        for i, component in enumerate(components):
            component_color = colors[i % len(colors)]
            offset = (i - n_components/2 + 0.5) * width

            # Get mean values for this component
            values = [avg_data.loc[component, metric] for metric in metrics]
            errors = [std_data.loc[component, metric] for metric in metrics]

            # 1. Plot bars (background, semi-transparent)
            bars = ax.bar(x + offset, values, width,
                         label=component,
                         color=component_color,
                         alpha=0.4,
                         edgecolor=component_color,
                         linewidth=1.5,
                         zorder=1)

            # 2. Plot individual scatter points on top of bars
            for metric_idx, metric in enumerate(metrics):
                # Get all individual data points for this component and metric
                component_data = df[df[component_name] == component][metric].values

                # Add jitter to x positions (centered on bar)
                n_points = len(component_data)
                jitter = np.random.uniform(-jitter_strength, jitter_strength, n_points)
                x_pos = np.full(n_points, metric_idx + offset) + jitter

                # Plot individual points (small, semi-transparent)
                ax.scatter(x_pos, component_data,
                          s=2, alpha=0.5, color=component_color,
                          edgecolors='white', linewidths=0.5, zorder=3)

            # 3. Plot mean markers (large, opaque, on top)
            ax.scatter(x + offset, values,
                      s=15, alpha=1.0, color=component_color,
                      edgecolors='black', linewidths=2.0,
                      marker='D', zorder=5)

            # 4. Add error bars
            ax.errorbar(x + offset, values, yerr=errors,
                       fmt='none', ecolor='black', elinewidth=1.5,
                       capsize=4, capthick=1.5, alpha=0.7, zorder=4)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics],
                          rotation=15, ha='right', fontsize=7)
        ax.set_ylabel('Value', fontsize=8)
        ax.set_title(f'{group_name}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.legend(fontsize=7, loc='best')

        # Set y-axis limits based on metric type
        if 'acc' in group_name.lower() or 'accuracy' in group_name.lower():
            ax.set_ylim([0, 1.05])
        elif 'correlation' in group_name.lower():
            ax.set_ylim([-0.05, 1.05])

        # Spine width
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(width=0.5, labelsize=7)

    # Overall title
    fig.suptitle(f'Component Comparison: {component_name.replace("_", " ").title()}',
                fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = component_name.replace(' ', '_').lower()
    output_file = output_dir / f'component_comparison_{safe_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved to {output_file}")

    output_file_svg = output_dir / f'component_comparison_{safe_name}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved to {output_file_svg}")

    plt.close()


def plot_all_components_single_metric(data, stds, metric, component_types, output_dir):
    """
    Plot a single metric across multiple component types with bars + scatter overlay.

    Args:
        data: DataFrame with all data (not aggregated)
        stds: Not used (kept for compatibility)
        metric: Single metric to plot
        component_types: List of component columns to compare
        output_dir: Output directory
    """
    n_types = len(component_types)
    fig, axes = plt.subplots(1, n_types, figsize=(2.4 * n_types, 3.0))

    if n_types == 1:
        axes = [axes]

    for ax_idx, comp_type in enumerate(component_types):
        ax = axes[ax_idx]

        # Compute averages for this component type
        grouped = data.groupby(comp_type)[metric].mean()
        grouped_std = data.groupby(comp_type)[metric].std()

        # Sort by custom order
        grouped = sort_by_custom_order(grouped, comp_type)
        grouped_std = sort_by_custom_order(grouped_std, comp_type)

        components = grouped.index.tolist()
        mean_values = grouped.values
        std_values = grouped_std.values

        # Color mapping
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # X positions
        x = np.arange(len(components))
        bar_width = 0.7
        jitter_strength = 0.15

        # Plot for each component
        for i, component in enumerate(components):
            component_color = colors[i % len(colors)]

            # Get all individual data points for this component
            component_data = data[data[comp_type] == component][metric].values

            mean_val = mean_values[i]
            std_val = std_values[i]

            # 1. Plot bar (background, semi-transparent)
            ax.bar(i, mean_val, bar_width,
                  color=component_color, alpha=0.4,
                  edgecolor=component_color, linewidth=1.5,
                  zorder=1)

            # 2. Plot individual scatter points on top of bar
            n_points = len(component_data)
            jitter = np.random.uniform(-jitter_strength, jitter_strength, n_points)
            x_pos = np.full(n_points, i) + jitter

            ax.scatter(x_pos, component_data,
                      s=5, alpha=0.5, color=component_color,
                      edgecolors='white', linewidths=0.5, zorder=3)

            # 3. Plot mean marker (large, opaque, on top)
            ax.scatter(i, mean_val,
                      s=18, alpha=1.0, color=component_color,
                      edgecolors='black', linewidths=2.5,
                      marker='D', zorder=5)

            # 4. Add error bar for mean
            ax.errorbar(i, mean_val, yerr=std_val,
                       fmt='none', ecolor='black', elinewidth=2.0,
                       capsize=5, capthick=2.0, alpha=0.7, zorder=4)

            # 5. Add value label above error bar
            ax.text(i, mean_val + std_val + 0.02,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=6, fontweight='bold')

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=8)
        ax.set_title(f'{comp_type.replace("_", " ").title()}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

        # Set y-axis limits based on metric type
        if 'acc' in metric or 'accuracy' in metric:
            ax.set_ylim([0, 1.05])
        elif 'correlation' in metric:
            ax.set_ylim([-0.05, 1.05])
        elif 'mae' in metric:
            # Get max value including error bars
            max_val = max([mean_values[i] + std_values[i] for i in range(len(mean_values))])
            ax.set_ylim([0, max_val * 1.15])

        # Spine width
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(width=0.5, labelsize=7)

    # Overall title
    metric_display = metric.replace('_', ' ').title()
    fig.suptitle(f'{metric_display} Comparison Across Components',
                fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = metric.replace(' ', '_').lower()
    output_file = output_dir / f'metric_comparison_{safe_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved to {output_file}")

    output_file_svg = output_dir / f'metric_comparison_{safe_name}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved to {output_file_svg}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot component-wise comparison from pareto data'
    )
    parser.add_argument('--data-dir', type=str, default='outputs/pareto_results',
                       help='Directory containing pareto CSV files')
    parser.add_argument('--pattern', type=str, default='pareto_data_*.csv',
                       help='Glob pattern for CSV files (use pareto_v2_data_*.csv for v2 data)')
    parser.add_argument('--output-dir', type=str, default='outputs/component_comparison',
                       help='Output directory for plots')
    parser.add_argument('--components', type=str, nargs='+',
                       default=['graph_encoder', 'loss_type', 'llm_type', 'intra_loss'],
                       help='Component types to compare')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['top1_acc', 'top3_acc', 'top10_acc', 'mae', 'correlation'],
                       help='Metrics to plot')

    args = parser.parse_args()

    print("="*80)
    print("COMPONENT COMPARISON ANALYSIS")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Pattern:        {args.pattern}")
    print(f"Components:     {', '.join(args.components)}")
    print(f"Metrics:        {', '.join(args.metrics)}")
    print(f"Output:         {args.output_dir}")
    print("="*80 + "\n")

    # Load data
    df = load_all_pareto_data(args.data_dir, args.pattern)

    # Print summary statistics
    print("\n[Summary] Data statistics:")
    for comp in args.components:
        if comp in df.columns:
            print(f"  {comp}: {df[comp].nunique()} unique values - {df[comp].unique().tolist()}")

    # Plot 1: For each component type, show all metrics
    print("\n[Plotting] Component-wise comparisons...")

    # Define metric groups
    metric_groups = {
        'Retrieval Accuracy': ['top1_acc', 'top3_acc', 'top10_acc'],
        'Property Prediction (MAE)': ['mae'],
        'Property Prediction (Correlation)': ['correlation']
    }

    for component in args.components:
        if component not in df.columns:
            print(f"[Warning] Component '{component}' not found in data, skipping")
            continue

        # Filter available metrics
        available_metrics = [m for m in args.metrics if m in df.columns]

        # Compute averages
        avg_data, std_data = compute_component_averages(df, component, available_metrics)

        print(f"\n[Component: {component}]")
        print(avg_data.to_string())

        # Filter metric groups to only include available metrics
        filtered_groups = {}
        for group_name, group_metrics in metric_groups.items():
            available_in_group = [m for m in group_metrics if m in available_metrics]
            if available_in_group:
                filtered_groups[group_name] = available_in_group

        # Plot
        plot_component_comparison(df, avg_data, std_data, component,
                                 filtered_groups, args.output_dir)

    # Plot 2: For each metric, show all component types
    print("\n[Plotting] Metric-wise comparisons across all components...")

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[Warning] Metric '{metric}' not found in data, skipping")
            continue

        available_components = [c for c in args.components if c in df.columns]

        plot_all_components_single_metric(df, None, metric,
                                         available_components, args.output_dir)

    print("\n" + "="*80)
    print("[DONE] Component comparison analysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
