#!/usr/bin/env python3
"""
Pareto Plot v2 for Zero-shot Retrieval vs Inverse Design Results

New classification scheme:
- Text Encoder (color): llm_type (galactica-125m, matscibert, opt-125m, scibert, etc.)
- Vision Intra (marker): graph_encoder (cgcnn, painn, orb)
- Textual Intra (fill): intra_loss (True/False)

This script:
1. Scans result directories (zero_shot_retrieval/ and inverse_design/)
2. Collects metrics from each model
3. Creates Pareto plots showing trade-offs between retrieval accuracy and inverse design MAE
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple, Optional

# Configure matplotlib for high-quality output
mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Arial', size=5)
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_model_name(model_dir: str) -> Dict[str, str]:
    """
    Parse model directory name to extract components.

    Two formats supported:
    1. Old: 'cgcnn-infonce-galactica-125m-intra_False_epoch=4-step=495'
    2. New: 'painn-infonce-galactica-125m-vTrue-tFalse_epoch=4-step=495'
           'SynData-LMloss-painn-infonce-galactica-125m-vFalse-tTrue_epoch=4-step=495'

    Returns: {
        'graph_encoder': 'cgcnn' or 'painn',
        'loss_type': 'infonce',
        'llm_type': 'galactica-125m',
        'vision_intra': 'True' or 'False',
        'textual_intra': 'True' or 'False',
        'intra_loss': 'True' or 'False' (for backward compatibility),
        'checkpoint': 'epoch=4-step=495',
        'syndata': 'True' or 'False',
        'lm_loss': 'True' or 'False'
    }
    """
    parts = model_dir.split('-')

    # Check for SynData prefix
    syndata = 'False'
    lm_loss = 'False'
    start_idx = 0

    if parts[0] == 'SynData':
        syndata = 'True'
        start_idx = 1
        # Check for LMloss
        if len(parts) > 1 and parts[1] == 'LMloss':
            lm_loss = 'True'
            start_idx = 2

    # Extract components
    graph_encoder = parts[start_idx]  # cgcnn, painn, orb
    loss_type = parts[start_idx + 1]  # infonce, jsd_dot, etc.

    # Find LLM type and intra markers
    llm_parts = []
    vision_intra = None
    textual_intra = None
    intra_loss = None
    checkpoint_idx = None

    for i, part in enumerate(parts[start_idx + 2:], start=start_idx + 2):
        # Check for new format: vTrue/vFalse (vision intra)
        if part.startswith('v') and part[1:] in ['True', 'False']:
            vision_intra = part[1:]  # Extract True/False
            continue

        # Check for new format: tTrue/tFalse (textual intra)
        if part.startswith('t') and part[1:].split('_')[0] in ['True', 'False']:
            textual_intra = part[1:].split('_')[0]  # Extract True/False
            # Check if checkpoint starts here
            if '_' in part:
                checkpoint_idx = i
            continue

        # Check for old format: intra_True/False
        if 'intra_' in part:
            intra_loss = part.split('_')[1]  # 'True' or 'False'
            vision_intra = intra_loss  # Use same value for both
            textual_intra = intra_loss
            checkpoint_idx = i
            break

        # Check for checkpoint pattern (epoch=)
        if 'epoch=' in part or checkpoint_idx is not None:
            checkpoint_idx = i
            break

        # Otherwise, it's part of the LLM name
        llm_parts.append(part)

    llm_type = '-'.join(llm_parts) if llm_parts else 'unknown'

    # Extract checkpoint
    if checkpoint_idx is not None:
        checkpoint = '-'.join(parts[checkpoint_idx:])
    else:
        checkpoint = 'unknown'

    # Set default values if not found
    if vision_intra is None:
        vision_intra = 'unknown'
    if textual_intra is None:
        textual_intra = 'unknown'
    if intra_loss is None:
        intra_loss = vision_intra  # For backward compatibility

    return {
        'model_name': model_dir,
        'graph_encoder': graph_encoder,
        'loss_type': loss_type,
        'llm_type': llm_type,
        'vision_intra': vision_intra,
        'textual_intra': textual_intra,
        'intra_loss': intra_loss,  # Backward compatibility
        'checkpoint': checkpoint,
        'syndata': syndata,
        'lm_loss': lm_loss
    }


def collect_retrieval_results(retrieval_dir: Path) -> pd.DataFrame:
    """
    Collect zero-shot retrieval results from all models.

    Returns:
        DataFrame with columns: model_name, top1_acc, top3_acc, top10_acc
    """
    results = []

    # Scan all model directories
    for model_dir in retrieval_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Look for retrieval_text.json
        result_file = model_dir / 'retrieval_text.json'
        if not result_file.exists():
            print(f"[WARNING] No retrieval_text.json in {model_dir.name}")
            continue

        # Load results
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Extract metrics
        model_info = parse_model_name(model_dir.name)
        model_info.update({
            'top1_acc': data['overall_metrics']['top1_accuracy'],
            'top3_acc': data['overall_metrics']['top3_accuracy'],
            'top10_acc': data['overall_metrics']['top10_accuracy']
        })

        results.append(model_info)

    df = pd.DataFrame(results)
    print(f"[Retrieval] Collected results from {len(df)} models")
    return df


def collect_inverse_design_results(inverse_dir: Path, property_name: str, k_value: int) -> pd.DataFrame:
    """
    Collect inverse design results from all models for a specific property and K value.

    Args:
        inverse_dir: Path to inverse_design directory
        property_name: Property to analyze (e.g., 'band gap', 'density')
        k_value: K value for retrieval (e.g., 3, 10, 30)

    Returns:
        DataFrame with columns: model_name, mae, correlation, random_mae
    """
    results = []

    # Convert property name to filename format
    property_filename = f"inverse_design_{property_name.replace(' ', '_')}.json"

    # Scan all model directories
    for model_dir in inverse_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Look for property result file
        result_file = model_dir / property_filename
        if not result_file.exists():
            print(f"[WARNING] No {property_filename} in {model_dir.name}")
            continue

        # Load results
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Extract metrics for specified K value
        k_key = f'k{k_value}'
        mae_key = f'{k_key}_overall_mae'
        corr_key = f'{k_key}_correlation'
        random_mae_key = f'random_{k_key}_overall_mae'

        if mae_key not in data:
            print(f"[WARNING] No metrics for K={k_value} in {model_dir.name}")
            continue

        # Extract metrics
        model_info = parse_model_name(model_dir.name)
        model_info.update({
            'mae': data[mae_key],
            'correlation': data[corr_key],
            'random_mae': data[random_mae_key],
            'property': property_name,
            'k_value': k_value,
            'unit': data.get('unit', '')
        })

        results.append(model_info)

    df = pd.DataFrame(results)
    print(f"[Inverse Design] Collected results from {len(df)} models for {property_name} (K={k_value})")
    return df


def merge_results(retrieval_df: pd.DataFrame, inverse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge retrieval and inverse design results by model name.
    """
    # Merge on model_name
    merged = pd.merge(
        retrieval_df,
        inverse_df,
        on='model_name',
        how='inner',
        suffixes=('_retrieval', '_inverse')
    )

    # Resolve duplicate columns (keep retrieval version for metadata)
    for col in ['graph_encoder', 'loss_type', 'llm_type', 'intra_loss', 'checkpoint',
                'vision_intra', 'textual_intra', 'syndata', 'lm_loss']:
        if f'{col}_retrieval' in merged.columns:
            merged[col] = merged[f'{col}_retrieval']
            merged.drop(columns=[f'{col}_retrieval', f'{col}_inverse'], inplace=True)

    print(f"[Merge] Successfully merged {len(merged)} models")
    return merged


def apply_filters(df: pd.DataFrame,
                 filter_graph_encoder: Optional[List[str]] = None,
                 filter_llm: Optional[List[str]] = None,
                 filter_loss: Optional[List[str]] = None,
                 filter_intra_loss: Optional[List[str]] = None,
                 filter_models: Optional[List[str]] = None,
                 only_new_naming: bool = True) -> pd.DataFrame:
    """
    Apply filtering criteria to the merged dataframe.

    Args:
        df: Merged dataframe
        filter_graph_encoder: List of graph encoders to keep
        filter_llm: List of LLM types to keep
        filter_loss: List of loss types to keep
        filter_intra_loss: List of intra loss values to keep ('True' and/or 'False')
        filter_models: List of exact model names to keep
        only_new_naming: If True, only keep models with vTrue/False-tTrue/False naming (default: True)

    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()

    # Filter by exact model names (highest priority)
    if filter_models is not None and len(filter_models) > 0:
        filtered_df = filtered_df[filtered_df['model_name'].isin(filter_models)]
        print(f"[Filter] Applied model name filter: kept {len(filtered_df)} models")
        return filtered_df

    # Filter for new naming convention (vTrue/False-tTrue/False)
    if only_new_naming:
        # Keep only models with actual vTrue/False-tTrue/False pattern in name
        # Pattern: contains 'v' followed by True/False and 't' followed by True/False
        import re
        pattern = r'v(True|False)-t(True|False)'
        filtered_df = filtered_df[
            filtered_df['model_name'].str.contains(pattern, regex=True)
        ]
        print(f"[Filter] New naming convention only (vTrue/False-tTrue/False): kept {len(filtered_df)} models")

    # Filter by graph encoder
    if filter_graph_encoder is not None and len(filter_graph_encoder) > 0:
        filtered_df = filtered_df[filtered_df['graph_encoder'].isin(filter_graph_encoder)]
        print(f"[Filter] Graph encoder filter: kept {len(filtered_df)} models")

    # Filter by LLM type
    if filter_llm is not None and len(filter_llm) > 0:
        filtered_df = filtered_df[filtered_df['llm_type'].isin(filter_llm)]
        print(f"[Filter] LLM type filter: kept {len(filtered_df)} models")

    # Filter by loss type
    if filter_loss is not None and len(filter_loss) > 0:
        filtered_df = filtered_df[filtered_df['loss_type'].isin(filter_loss)]
        print(f"[Filter] Loss type filter: kept {len(filtered_df)} models")

    # Filter by intra loss
    if filter_intra_loss is not None and len(filter_intra_loss) > 0:
        # Convert to 'True' or 'False' format
        intra_values = [val.capitalize() for val in filter_intra_loss]
        filtered_df = filtered_df[filtered_df['intra_loss'].isin(intra_values)]
        print(f"[Filter] Intra loss filter: kept {len(filtered_df)} models")

    return filtered_df


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str,
                           x_maximize: bool = True, y_minimize: bool = True) -> np.ndarray:
    """
    Compute Pareto-optimal points (non-dominated solutions).

    Args:
        df: DataFrame with results
        x_col: Column for x-axis metric
        y_col: Column for y-axis metric
        x_maximize: If True, higher x is better
        y_minimize: If True, lower y is better

    Returns:
        Boolean array indicating Pareto-optimal points
    """
    x = df[x_col].values
    y = df[y_col].values

    # Flip signs if we want to maximize/minimize
    x_comp = x if x_maximize else -x
    y_comp = -y if y_minimize else y

    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        # Check if point i is dominated by any other point
        # A point dominates i if it's better in both dimensions
        is_dominated = np.any(
            (x_comp > x_comp[i]) & (y_comp > y_comp[i])
        )
        if is_dominated:
            is_pareto[i] = False

    return is_pareto


def plot_pareto(df: pd.DataFrame,
               x_col: str = 'top1_acc',
               y_col: str = 'mae',
               property_name: str = 'band gap',
               k_value: int = 10,
               output_dir: Path = Path('pareto_results'),
               show_frontier: bool = True,
               show_labels: bool = True):
    """
    Create Pareto plot with new classification scheme:
    - Text Encoder (color): llm_type
    - Vision Intra (marker shape): vision_intra (True/False)
    - Textual Intra (marker fill): textual_intra (True/False)

    Args:
        df: DataFrame with merged results
        x_col: Column for x-axis (retrieval metric)
        y_col: Column for y-axis (inverse design metric)
        property_name: Property name for title
        k_value: K value for title
        output_dir: Directory to save plots
        show_frontier: Whether to highlight Pareto frontier
        show_labels: Whether to show model labels
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define colors for TEXT ENCODERS (LLM types)
    text_encoder_colors = {
        'galactica-125m': '#1f77b4',    # blue
        'matscibert': '#ff7f0e',        # orange
        'opt-125m': '#2ca02c',          # green
        'scibert': '#d62728',           # red
        'galactica-1.3b': '#9467bd',    # purple
        'galactica-6.7b': '#8c564b',    # brown
    }

    # Define markers for VISION INTRA (True/False)
    vision_intra_markers = {
        'True': 's',       # square (filled vision intra)
        'False': 'o',      # circle (no vision intra)
    }

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.7))

    # Compute Pareto frontier
    is_pareto = compute_pareto_frontier(df, x_col, y_col, x_maximize=True, y_minimize=True)

    # Get unit for y-axis label
    unit = df['unit'].iloc[0] if 'unit' in df.columns and len(df) > 0 else ''

    # Reset index to ensure alignment with is_pareto array
    df_reset = df.reset_index(drop=True)

    # Plot all points
    for idx, row in df_reset.iterrows():
        llm_type = row['llm_type']                # Text Encoder (color)
        vision_intra = row['vision_intra']        # Vision Intra (marker shape)
        textual_intra = row['textual_intra']      # Textual Intra (fill)

        # Get color based on text encoder
        color = text_encoder_colors.get(llm_type, 'gray')

        # Get marker based on vision intra
        marker = vision_intra_markers.get(vision_intra, 'o')

        # Determine fill style based on TEXTUAL INTRA
        if textual_intra == 'True':
            # Filled marker for Textual Intra = True
            facecolor = color
            edgecolor = 'black'
            alpha = 0.8
        else:
            # Empty marker for Textual Intra = False
            facecolor = 'none'
            edgecolor = color
            alpha = 1.0

        # Plot point
        ax.scatter(row[x_col], row[y_col],
                  marker=marker, s=50,
                  facecolors=facecolor,
                  edgecolors=edgecolor,
                  linewidths=0.8,
                  alpha=alpha,
                  zorder=10 if is_pareto[idx] else 5)

        # Add label if requested
        if show_labels:
            label = f"{llm_type.split('-')[0]}-v{vision_intra[0]}t{textual_intra[0]}"
            ax.annotate(label,
                       xy=(row[x_col], row[y_col]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=4,
                       alpha=0.6)

    # Highlight Pareto frontier
    if show_frontier and np.any(is_pareto):
        pareto_df = df_reset[is_pareto].sort_values(by=x_col)
        ax.plot(pareto_df[x_col], pareto_df[y_col],
               'r--', linewidth=0.8, alpha=0.5, label='Pareto Frontier', zorder=1)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []

    # Section 1: Text Encoder (colors)
    legend_elements.append(Line2D([0], [0], color='w', label='Text Encoder:',
                                 markerfacecolor='w', markersize=0))
    for llm, color in text_encoder_colors.items():
        if llm in df_reset['llm_type'].values:
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=5,
                                         markeredgecolor='black', markeredgewidth=0.5,
                                         label=f'  {llm}'))

    # Spacer
    legend_elements.append(Line2D([0], [0], color='w', label='',
                                 markerfacecolor='w', markersize=0))

    # Section 2: Vision Intra (markers)
    legend_elements.append(Line2D([0], [0], color='w', label='Vision Intra:',
                                 markerfacecolor='w', markersize=0))
    for v_intra, marker in vision_intra_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                     markerfacecolor='gray', markersize=6,
                                     markeredgecolor='black', markeredgewidth=0.5,
                                     label=f'  {v_intra} ({marker})'))

    # Spacer
    legend_elements.append(Line2D([0], [0], color='w', label='',
                                 markerfacecolor='w', markersize=0))

    # Section 3: Textual Intra (fill)
    legend_elements.append(Line2D([0], [0], color='w', label='Textual Intra:',
                                 markerfacecolor='w', markersize=0))
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='gray', markersize=5,
                                 markeredgecolor='black', markeredgewidth=0.5,
                                 label='  True (filled)'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='none', markersize=5,
                                 markeredgecolor='gray', markeredgewidth=0.5,
                                 label='  False (empty)'))

    ax.legend(handles=legend_elements, loc='best', fontsize=4.5,
             frameon=True, framealpha=0.9, edgecolor='gray', fancybox=False)

    # Labels and title
    ax.set_xlabel(f'Zero-shot Retrieval Top-1 Accuracy', fontsize=6)
    ax.set_ylabel(f'Inverse Design MAE ({unit})', fontsize=6)
    ax.set_title(f'Retrieval vs Inverse Design: {property_name.title()} (K={k_value})',
                fontsize=6, fontweight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.tick_params(width=0.5, length=3, labelsize=5)

    plt.tight_layout()

    # Save plot
    property_safe = property_name.replace(' ', '_')
    output_file = output_dir / f'pareto_v2_retrieval_vs_{property_safe}_k{k_value}.svg'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved to {output_file}")

    # Also save as PNG
    output_file_png = output_dir / f'pareto_v2_retrieval_vs_{property_safe}_k{k_value}.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved to {output_file_png}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Pareto plots v2 for zero-shot retrieval vs inverse design'
    )
    parser.add_argument('--retrieval-dir', type=str,
                       default='zero_shot_retrieval',
                       help='Directory containing retrieval results')
    parser.add_argument('--inverse-dir', type=str,
                       default='inverse_design',
                       help='Directory containing inverse design results')
    parser.add_argument('--property', type=str,
                       default='band gap',
                       choices=['band gap', 'formation energy per atom', 'density',
                               'scintillation attenuation length', 'total magnetization',
                               'energy above hull'],
                       help='Property for inverse design')
    parser.add_argument('--k-value', type=int,
                       default=10,
                       help='K value for inverse design retrieval')
    parser.add_argument('--x-metric', type=str,
                       default='top1_acc',
                       choices=['top1_acc', 'top3_acc', 'top10_acc'],
                       help='Retrieval metric for x-axis')
    parser.add_argument('--y-metric', type=str,
                       default='mae',
                       choices=['mae', 'correlation'],
                       help='Inverse design metric for y-axis')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/pareto_results',
                       help='Output directory for plots and data')
    parser.add_argument('--show-frontier', action='store_true',
                       default=True,
                       help='Show Pareto frontier line')
    parser.add_argument('--show-labels', action='store_true',
                       default=False,
                       help='Show model labels on points')
    parser.add_argument('--save-csv', action='store_true',
                       default=True,
                       help='Save merged data to CSV')

    # Filtering options
    parser.add_argument('--filter-graph-encoder', type=str, nargs='+',
                       choices=['cgcnn', 'painn', 'orb'],
                       help='Filter by graph encoder (e.g., --filter-graph-encoder painn orb)')
    parser.add_argument('--filter-llm', type=str, nargs='+',
                       help='Filter by LLM type (e.g., --filter-llm galactica-125m scibert)')
    parser.add_argument('--filter-loss', type=str, nargs='+',
                       choices=['infonce', 'jsd_dot'],
                       help='Filter by loss type (e.g., --filter-loss infonce)')
    parser.add_argument('--filter-intra-loss', type=str, nargs='+',
                       choices=['True', 'False', 'true', 'false'],
                       help='Filter by intra loss (True or False)')
    parser.add_argument('--filter-models', type=str, nargs='+',
                       help='Filter by exact model names')

    args = parser.parse_args()

    # Convert to Path objects
    retrieval_dir = Path(args.retrieval_dir)
    inverse_dir = Path(args.inverse_dir)
    output_dir = Path(args.output_dir)

    # Check if directories exist
    if not retrieval_dir.exists():
        print(f"[ERROR] Retrieval directory not found: {retrieval_dir}")
        return
    if not inverse_dir.exists():
        print(f"[ERROR] Inverse design directory not found: {inverse_dir}")
        return

    print("="*80)
    print("PARETO PLOT V2 GENERATION")
    print("="*80)
    print(f"Retrieval dir: {retrieval_dir}")
    print(f"Inverse dir:   {inverse_dir}")
    print(f"Property:      {args.property}")
    print(f"K value:       {args.k_value}")
    print(f"X metric:      {args.x_metric}")
    print(f"Y metric:      {args.y_metric}")
    print("="*80 + "\n")
    print("Classification scheme:")
    print("  - Text Encoder (color):   llm_type")
    print("  - Vision Intra (marker):  graph_encoder")
    print("  - Textual Intra (fill):   intra_loss")
    print("="*80 + "\n")

    # Step 1: Collect retrieval results
    print("[Step 1] Collecting zero-shot retrieval results...")
    retrieval_df = collect_retrieval_results(retrieval_dir)

    # Step 2: Collect inverse design results
    print("\n[Step 2] Collecting inverse design results...")
    inverse_df = collect_inverse_design_results(inverse_dir, args.property, args.k_value)

    # Step 3: Merge results
    print("\n[Step 3] Merging results...")
    merged_df = merge_results(retrieval_df, inverse_df)

    if len(merged_df) == 0:
        print("[ERROR] No matching models found between retrieval and inverse design results")
        return

    # Step 3.5: Apply filters
    print("\n[Step 3.5] Applying filters...")
    merged_df = apply_filters(
        merged_df,
        filter_graph_encoder=args.filter_graph_encoder,
        filter_llm=args.filter_llm,
        filter_loss=args.filter_loss,
        filter_intra_loss=args.filter_intra_loss,
        filter_models=args.filter_models,
        only_new_naming=True  # Only keep vTrue/False-tTrue/False models
    )

    if len(merged_df) == 0:
        print("[ERROR] No models remaining after filtering")
        return

    print(f"[Step 3.5] Final model count after filtering: {len(merged_df)}")

    # Step 4: Save merged data
    if args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / f'pareto_v2_data_{args.property.replace(" ", "_")}_k{args.k_value}.csv'
        merged_df.to_csv(csv_file, index=False)
        print(f"\n[Step 4] Saved merged data to {csv_file}")

    # Step 5: Generate Pareto plot
    print("\n[Step 5] Generating Pareto plot v2...")
    plot_pareto(
        merged_df,
        x_col=args.x_metric,
        y_col=args.y_metric,
        property_name=args.property,
        k_value=args.k_value,
        output_dir=output_dir,
        show_frontier=args.show_frontier,
        show_labels=args.show_labels
    )

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total models: {len(merged_df)}")
    print(f"\nRetrieval {args.x_metric}:")
    print(f"  Min:  {merged_df[args.x_metric].min():.4f}")
    print(f"  Max:  {merged_df[args.x_metric].max():.4f}")
    print(f"  Mean: {merged_df[args.x_metric].mean():.4f}")
    print(f"\nInverse Design {args.y_metric}:")
    print(f"  Min:  {merged_df[args.y_metric].min():.4f}")
    print(f"  Max:  {merged_df[args.y_metric].max():.4f}")
    print(f"  Mean: {merged_df[args.y_metric].mean():.4f}")

    # Identify Pareto-optimal models
    is_pareto = compute_pareto_frontier(merged_df, args.x_metric, args.y_metric,
                                       x_maximize=True, y_minimize=True)
    pareto_models = merged_df[is_pareto].sort_values(by=args.x_metric)

    print(f"\nPareto-optimal models ({len(pareto_models)}):")
    for idx, row in pareto_models.iterrows():
        print(f"  {row['graph_encoder']:6s} + {row['llm_type']:20s} + v_intra={row['vision_intra']:5s} + t_intra={row['textual_intra']:5s} | "
              f"{args.x_metric}={row[args.x_metric]:.4f}, {args.y_metric}={row[args.y_metric]:.4f}")

    print("="*80 + "\n")
    print("[DONE] Pareto plot v2 generation completed successfully!")


if __name__ == '__main__':
    main()
