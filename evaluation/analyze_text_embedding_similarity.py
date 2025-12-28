#!/usr/bin/env python3
"""
Analyze text embedding similarity for different property values.

This script:
1. Loads a trained CLaC model
2. Generates text embeddings for different property values (e.g., band gap)
3. Computes cosine similarity between embeddings
4. Plots similarity as a function of property value difference
"""

import sys
from pathlib import Path
# Add parent directory to path to import model module
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from model.clac import CLaCLite

# Configure matplotlib
mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Arial', size=5)
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def get_property_template(property_name, value, threshold_mode=None):
    """
    Get template text for a given property and value.

    Args:
        property_name: Name of property (e.g., 'band gap')
        value: Numeric value of property
        threshold_mode: Optional mode for range-based queries:
                       'greater_than', 'less_than', 'greater_equal', 'less_equal'
                       If None, uses exact value template

    Returns:
        Formatted template text
    """
    templates = {
        'band gap': {
            'unit': 'eV',
            'template': lambda v: f'material with band gap of {v:.2f} eV',
            'greater_than': lambda v: f'material with band gap greater than {v:.2f} eV',
            'less_than': lambda v: f'material with band gap less than {v:.2f} eV',
            'greater_equal': lambda v: f'material with band gap at least {v:.2f} eV',
            'less_equal': lambda v: f'material with band gap at most {v:.2f} eV',
        },
        'formation energy per atom': {
            'unit': 'eV/atom',
            'template': lambda v: f'material with formation energy of {v:.2f} eV/atom',
            'greater_than': lambda v: f'material with formation energy greater than {v:.2f} eV/atom',
            'less_than': lambda v: f'material with formation energy less than {v:.2f} eV/atom',
            'greater_equal': lambda v: f'material with formation energy at least {v:.2f} eV/atom',
            'less_equal': lambda v: f'material with formation energy at most {v:.2f} eV/atom',
        },
        'density': {
            'unit': 'g/cm³',
            'template': lambda v: f'material with density of {v:.2f} g/cm³',
            'greater_than': lambda v: f'material with density greater than {v:.2f} g/cm³',
            'less_than': lambda v: f'material with density less than {v:.2f} g/cm³',
            'greater_equal': lambda v: f'material with density at least {v:.2f} g/cm³',
            'less_equal': lambda v: f'material with density at most {v:.2f} g/cm³',
        },
        'scintillation attenuation length': {
            'unit': 'cm',
            'template': lambda v: f'scintillator with attenuation length of {v:.2f} cm',
            'greater_than': lambda v: f'scintillator with attenuation length greater than {v:.2f} cm',
            'less_than': lambda v: f'scintillator with attenuation length less than {v:.2f} cm',
            'greater_equal': lambda v: f'scintillator with attenuation length at least {v:.2f} cm',
            'less_equal': lambda v: f'scintillator with attenuation length at most {v:.2f} cm',
        },
        'total magnetization': {
            'unit': 'μB',
            'template': lambda v: f'material with magnetization of {v:.2f} μB',
            'greater_than': lambda v: f'material with magnetization greater than {v:.2f} μB',
            'less_than': lambda v: f'material with magnetization less than {v:.2f} μB',
            'greater_equal': lambda v: f'material with magnetization at least {v:.2f} μB',
            'less_equal': lambda v: f'material with magnetization at most {v:.2f} μB',
        },
        'energy above hull': {
            'unit': 'eV/atom',
            'template': lambda v: f'material with energy above hull of {v:.2f} eV/atom',
            'greater_than': lambda v: f'material with energy above hull greater than {v:.2f} eV/atom',
            'less_than': lambda v: f'material with energy above hull less than {v:.2f} eV/atom',
            'greater_equal': lambda v: f'material with energy above hull at least {v:.2f} eV/atom',
            'less_equal': lambda v: f'material with energy above hull at most {v:.2f} eV/atom',
        }
    }

    if property_name not in templates:
        raise ValueError(f"Unknown property: {property_name}")

    # Use threshold mode if specified, otherwise use exact value template
    mode = threshold_mode if threshold_mode else 'template'

    if mode not in templates[property_name]:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    return templates[property_name][mode](value)


def get_property_unit(property_name):
    """
    Get unit string for a given property.

    Args:
        property_name: Name of property

    Returns:
        Unit string (e.g., 'eV', 'g/cm³')
    """
    templates = {
        'band gap': 'eV',
        'formation energy per atom': 'eV/atom',
        'density': 'g/cm³',
        'scintillation attenuation length': 'cm',
        'total magnetization': 'μB',
        'energy above hull': 'eV/atom'
    }

    return templates.get(property_name, '')


def generate_text_embedding(model, text, device='cuda', use_projection=True):
    """
    Generate text embedding from model.

    Args:
        model: Trained CLaC model
        text: Input text string
        device: Device to run on
        use_projection: If True, use projection layer (text_block).
                       If False, use raw LLM features (useful for untrained models)

    Returns:
        Text embedding tensor
    """
    # Tokenize
    inputs = model.tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embedding
    with torch.no_grad():
        _, text_features = model.encode_text(inputs)

        if use_projection:
            # Use projection layer (for trained models)
            text_embedding = model.loss.global_d.text_block(text_features)
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        else:
            # Use raw LLM features (for untrained models or to analyze LLM embeddings)
            text_embedding = F.normalize(text_features, p=2, dim=-1)

    return text_embedding


def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix between embeddings.

    Args:
        embeddings: Tensor of shape (N, D) where N is number of samples

    Returns:
        Similarity matrix of shape (N, N)
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute cosine similarity
    similarity = torch.mm(embeddings, embeddings.t())

    return similarity.cpu().numpy()


def analyze_property_similarity(model, property_name, values, device='cuda', use_projection=True):
    """
    Analyze embedding similarity across different property values.

    Args:
        model: Trained CLaC model
        property_name: Name of property to analyze
        values: List or array of property values to test
        device: Device to run on
        use_projection: If True, use projection layer. If False, use raw LLM features

    Returns:
        Dictionary containing values, embeddings, and similarity matrix
    """
    print(f"\n[Analysis] Property: {property_name}")
    print(f"[Analysis] Testing {len(values)} values: {values[0]:.2f} to {values[-1]:.2f}")
    if use_projection:
        print(f"[Analysis] Using projection layer (text_block)")
    else:
        print(f"[Analysis] Using raw LLM features (no projection)")

    embeddings_list = []

    # Generate embeddings for each value
    for value in tqdm(values, desc="Generating embeddings"):
        text = get_property_template(property_name, value)
        embedding = generate_text_embedding(model, text, device, use_projection=use_projection)
        embeddings_list.append(embedding.squeeze(0))

    # Stack embeddings
    embeddings = torch.stack(embeddings_list)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    return {
        'values': values,
        'embeddings': embeddings.cpu().numpy(),
        'similarity_matrix': similarity_matrix,
        'property_name': property_name
    }


def plot_similarity_heatmap(results, output_dir):
    """
    Plot similarity matrix as heatmap.

    Args:
        results: Dictionary from analyze_property_similarity
        output_dir: Directory to save plot
    """
    values = results['values']
    similarity = results['similarity_matrix']
    property_name = results['property_name']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot heatmap
    im = ax.imshow(similarity, cmap='viridis', aspect='auto',
                   vmin=0, vmax=1, origin='lower')

    # Set ticks
    n_ticks = min(10, len(values))
    tick_indices = np.linspace(0, len(values)-1, n_ticks, dtype=int)
    tick_labels = [f"{values[i]:.2f}" for i in tick_indices]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels)

    # Labels
    ax.set_xlabel(f'{property_name.title()}', fontsize=10)
    ax.set_ylabel(f'{property_name.title()}', fontsize=10)
    ax.set_title(f'Text Embedding Similarity: {property_name.title()}',
                fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=10)

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    property_safe = property_name.replace(' ', '_')
    output_file = output_dir / f'similarity_heatmap_{property_safe}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved heatmap to {output_file}")

    output_file_svg = output_dir / f'similarity_heatmap_{property_safe}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved heatmap to {output_file_svg}")

    plt.close()


def plot_similarity_vs_difference(results, output_dir):
    """
    Plot similarity as a function of value difference.

    For each pair of values, plot their similarity vs their absolute difference.

    Args:
        results: Dictionary from analyze_property_similarity
        output_dir: Directory to save plot
    """
    values = results['values']
    similarity = results['similarity_matrix']
    property_name = results['property_name']

    # Compute all pairwise differences and similarities
    differences = []
    similarities = []

    n = len(values)
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle (no duplicates)
            diff = abs(values[i] - values[j])
            sim = similarity[i, j]
            differences.append(diff)
            similarities.append(sim)

    differences = np.array(differences)
    similarities = np.array(similarities)

    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter plot
    scatter = ax.scatter(differences, similarities, alpha=0.3, s=5, c=similarities,
                        cmap='viridis', vmin=0, vmax=1)

    # Bin and plot mean trend
    n_bins = 20
    bins = np.linspace(0, differences.max(), n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        mask = (differences >= bins[i]) & (differences < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(similarities[mask].mean())
            bin_stds.append(similarities[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    # Plot mean trend
    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], bin_means[valid], 'r-', linewidth=0.5,
           label='Mean similarity', zorder=10)
    ax.fill_between(bin_centers[valid],
                    bin_means[valid] - bin_stds[valid],
                    bin_means[valid] + bin_stds[valid],
                    alpha=0.3, color='red', label='±1 std', zorder=5)

    # Labels
    ax.set_xlabel(f'Absolute Difference in {property_name.title()}', fontsize=10)
    ax.set_ylabel('Text Embedding Similarity', fontsize=10)
    ax.set_title(f'Embedding Similarity vs Value Difference: {property_name.title()}',
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, differences.max() * 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Similarity', rotation=270, labelpad=15, fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    property_safe = property_name.replace(' ', '_')
    output_file = output_dir / f'similarity_vs_difference_{property_safe}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved scatter plot to {output_file}")

    output_file_svg = output_dir / f'similarity_vs_difference_{property_safe}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved scatter plot to {output_file_svg}")

    plt.close()


def plot_similarity_examples(results, output_dir, reference_value=None):
    """
    Plot similarity of all values to a reference value.

    Args:
        results: Dictionary from analyze_property_similarity
        output_dir: Directory to save plot
        reference_value: Value to use as reference (default: middle value)
    """
    values = results['values']
    similarity = results['similarity_matrix']
    property_name = results['property_name']

    # Find reference index
    if reference_value is None:
        ref_idx = len(values) // 2
    else:
        ref_idx = np.argmin(np.abs(values - reference_value))

    ref_value = values[ref_idx]

    # Get similarities to reference
    similarities_to_ref = similarity[ref_idx, :]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot similarity
    ax.plot(values, similarities_to_ref, 'o-', linewidth=0.5, markersize=6,
           label=f'Similarity to {ref_value:.2f}')

    # Highlight reference point
    ax.axvline(ref_value, color='red', linestyle='--', alpha=0.5,
              label=f'Reference: {ref_value:.2f}')
    ax.plot(ref_value, 1.0, 'r*', markersize=15, zorder=10)

    # Labels
    ax.set_xlabel(f'{property_name.title()}', fontsize=10)
    ax.set_ylabel('Text Embedding Similarity', fontsize=10)
    ax.set_title(f'Similarity to Reference Value ({ref_value:.2f}): {property_name.title()}',
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    property_safe = property_name.replace(' ', '_')
    output_file = output_dir / f'similarity_to_reference_{property_safe}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved reference plot to {output_file}")

    output_file_svg = output_dir / f'similarity_to_reference_{property_safe}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved reference plot to {output_file_svg}")

    plt.close()


def analyze_threshold_similarity(model, property_name, values, threshold_value,
                                threshold_mode='greater_than', device='cuda', use_projection=True):
    """
    Analyze embedding similarity for range-based queries (e.g., "greater than X").

    Compares a threshold-based query (e.g., "band gap > 1 eV") with:
    - Values above the threshold
    - Values below the threshold

    Args:
        model: Trained CLaC model
        property_name: Name of property to analyze
        values: List or array of property values to test
        threshold_value: Threshold value for comparison
        threshold_mode: Type of threshold query ('greater_than', 'less_than', etc.)
        device: Device to run on
        use_projection: If True, use projection layer. If False, use raw LLM features

    Returns:
        Dictionary containing threshold query embedding, value embeddings, and similarity results
    """
    print(f"\n[Threshold Analysis] Property: {property_name}")
    print(f"[Threshold Analysis] Threshold: {threshold_value:.2f} ({threshold_mode})")
    print(f"[Threshold Analysis] Testing {len(values)} values: {values[0]:.2f} to {values[-1]:.2f}")

    # Generate embedding for threshold query
    threshold_text = get_property_template(property_name, threshold_value, threshold_mode=threshold_mode)
    print(f"[Threshold Analysis] Query text: '{threshold_text}'")

    threshold_embedding = generate_text_embedding(model, threshold_text, device, use_projection=use_projection)

    # Generate embeddings for each specific value
    embeddings_list = []
    for value in tqdm(values, desc="Generating value embeddings"):
        text = get_property_template(property_name, value)  # Exact value template
        embedding = generate_text_embedding(model, text, device, use_projection=use_projection)
        embeddings_list.append(embedding.squeeze(0))

    value_embeddings = torch.stack(embeddings_list)

    # Compute similarity between threshold query and each value
    threshold_embedding_norm = F.normalize(threshold_embedding, dim=-1)
    value_embeddings_norm = F.normalize(value_embeddings, dim=-1)

    similarities = torch.mm(value_embeddings_norm, threshold_embedding_norm.t()).squeeze(-1)
    similarities = similarities.cpu().numpy()

    # Split values into above/below threshold groups
    values_array = np.array(values)

    if threshold_mode in ['greater_than', 'greater_equal']:
        # For "greater than", we expect higher similarity for values above threshold
        threshold_condition = values_array > threshold_value
        if threshold_mode == 'greater_equal':
            threshold_condition = values_array >= threshold_value
    else:  # less_than, less_equal
        # For "less than", we expect higher similarity for values below threshold
        threshold_condition = values_array < threshold_value
        if threshold_mode == 'less_equal':
            threshold_condition = values_array <= threshold_value

    above_mask = threshold_condition
    below_mask = ~threshold_condition

    # Compute statistics
    above_similarities = similarities[above_mask]
    below_similarities = similarities[below_mask]

    print(f"\n[Results] Values satisfying condition ({np.sum(above_mask)}): "
          f"mean similarity = {above_similarities.mean():.4f} ± {above_similarities.std():.4f}")
    print(f"[Results] Values not satisfying condition ({np.sum(below_mask)}): "
          f"mean similarity = {below_similarities.mean():.4f} ± {below_similarities.std():.4f}")
    print(f"[Results] Difference in mean similarity: {above_similarities.mean() - below_similarities.mean():.4f}")

    return {
        'property_name': property_name,
        'threshold_value': threshold_value,
        'threshold_mode': threshold_mode,
        'threshold_text': threshold_text,
        'values': values,
        'similarities': similarities,
        'above_mask': above_mask,
        'below_mask': below_mask,
        'above_similarities': above_similarities,
        'below_similarities': below_similarities,
    }


def plot_threshold_analysis(threshold_results, output_dir):
    """
    Plot threshold-based similarity analysis.

    Shows:
    1. Similarity vs value with threshold line
    2. Distribution comparison (above vs below threshold)
    3. Box plot comparison

    Args:
        threshold_results: Dictionary from analyze_threshold_similarity
        output_dir: Directory to save plot
    """
    values = threshold_results['values']
    similarities = threshold_results['similarities']
    threshold_value = threshold_results['threshold_value']
    threshold_mode = threshold_results['threshold_mode']
    property_name = threshold_results['property_name']
    above_mask = threshold_results['above_mask']
    below_mask = threshold_results['below_mask']
    above_sims = threshold_results['above_similarities']
    below_sims = threshold_results['below_similarities']

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(7.2, 2.0))

    # Get unit for labels
    unit = get_property_unit(property_name)
    unit_label = f' ({unit})' if unit else ''

    # ===== Subplot 1: Similarity vs Value =====
    ax1 = plt.subplot(1, 3, 1)

    # Plot points with different colors for above/below threshold
    ax1.scatter(values[above_mask], similarities[above_mask],
               c='#ff7f0e', s=10, alpha=0.6, label='Satisfies condition', zorder=3)
    ax1.scatter(values[below_mask], similarities[below_mask],
               c='#1f77b4', s=10, alpha=0.6, label='Does not satisfy', zorder=3)

    # Add threshold line
    ax1.axvline(threshold_value, color='red', linestyle='--', linewidth=1.0,
               alpha=0.7, label=f'Threshold: {threshold_value:.2f}', zorder=2)

    # Connect points with line
    ax1.plot(values, similarities, '-', linewidth=0.5, alpha=0.3, color='gray', zorder=1)

    ax1.set_xlabel(f'{property_name.title()}{unit_label}', fontsize=5)
    ax1.set_ylabel('Similarity to Query', fontsize=5)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(labelsize=5, width=0.5)

    legend = ax1.legend(loc='best', fontsize=4)
    legend.get_frame().set_linewidth(0.5)

    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)

    # ===== Subplot 2: Histogram Comparison =====
    ax2 = plt.subplot(1, 3, 2)

    # Plot histograms
    bins = np.linspace(0, 1, 20)
    ax2.hist(above_sims, bins=bins, alpha=0.5, color='#ff7f0e',
            label=f'Satisfies ({len(above_sims)})', density=True)
    ax2.hist(below_sims, bins=bins, alpha=0.5, color='#1f77b4',
            label=f'Does not ({len(below_sims)})', density=True)

    # Add mean lines
    ax2.axvline(above_sims.mean(), color='#ff7f0e', linestyle='--', linewidth=1.0,
               label=f'Mean (satisfies): {above_sims.mean():.3f}')
    ax2.axvline(below_sims.mean(), color='#1f77b4', linestyle='--', linewidth=1.0,
               label=f'Mean (does not): {below_sims.mean():.3f}')

    ax2.set_xlabel('Similarity to Query', fontsize=5)
    ax2.set_ylabel('Density', fontsize=5)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.tick_params(labelsize=5, width=0.5)

    legend = ax2.legend(loc='best', fontsize=4)
    legend.get_frame().set_linewidth(0.5)

    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)

    # ===== Subplot 3: Box Plot Comparison =====
    ax3 = plt.subplot(1, 3, 3)

    # Create box plot data
    box_data = [below_sims, above_sims]
    box_labels = ['Does not\nsatisfy', 'Satisfies\ncondition']

    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True,
                     widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=4),
                     medianprops=dict(linewidth=1.5, color='black'),
                     boxprops=dict(linewidth=0.5),
                     whiskerprops=dict(linewidth=0.5),
                     capprops=dict(linewidth=0.5))

    # Color boxes
    bp['boxes'][0].set_facecolor('#1f77b4')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('#ff7f0e')
    bp['boxes'][1].set_alpha(0.5)

    # Add mean values as text
    for i, (data, label) in enumerate(zip(box_data, box_labels)):
        mean_val = data.mean()
        std_val = data.std()
        ax3.text(i+1, mean_val, f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=4, fontweight='bold')

    ax3.set_ylabel('Similarity to Query', fontsize=5)
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    ax3.tick_params(labelsize=5, width=0.5)

    for spine in ax3.spines.values():
        spine.set_linewidth(0.5)

    # Overall title with query text
    threshold_text = threshold_results['threshold_text']
    fig.suptitle(f"Query: '{threshold_text}'", fontsize=6, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    property_safe = property_name.replace(' ', '_')
    mode_safe = threshold_mode.replace('_', '-')
    threshold_safe = f"{threshold_value:.2f}".replace('.', 'p')

    output_file = output_dir / f'threshold_analysis_{property_safe}_{mode_safe}_{threshold_safe}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved threshold analysis to {output_file}")

    output_file_svg = output_dir / f'threshold_analysis_{property_safe}_{mode_safe}_{threshold_safe}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved threshold analysis to {output_file_svg}")

    plt.close()


def plot_all_in_one(results, output_dir, reference_value=None):
    """
    Plot all three visualizations in a single figure with subplots.

    Args:
        results: Dictionary from analyze_property_similarity
        output_dir: Directory to save plot
        reference_value: Value to use as reference (default: middle value)
    """
    values = results['values']
    similarity = results['similarity_matrix']
    property_name = results['property_name']

    # Helper function to round to clean numbers
    def round_to_clean(val):
        # For very small ranges (like energy above hull 0-1), use 0.1 increments
        if abs(val) < 2:
            return round(val * 10) / 10  # Round to nearest 0.1
        elif abs(val) < 10:
            return round(val * 2) / 2  # Round to nearest 0.5
        else:
            return round(val)  # Round to nearest integer

    # Find reference indices - use 3 reference values with clean numbers
    if reference_value is None:
        # Use 25%, 50%, 75% of the range with clean numbers
        min_val, max_val = values[0], values[-1]
        value_range = max_val - min_val

        # Generate clean reference values
        ref_candidates = [
            min_val + 0.25 * value_range,
            min_val + 0.50 * value_range,
            min_val + 0.75 * value_range
        ]
    else:
        # Find closest index to specified value, plus nearby values
        center_idx = np.argmin(np.abs(values - reference_value))
        center_val = values[center_idx]

        # Create clean reference values around the center
        min_val, max_val = values[0], values[-1]
        value_range = max_val - min_val

        # Generate references at approximately -25%, 0%, +25% from center
        ref_candidates = [
            max(min_val, center_val - 0.25 * value_range),
            center_val,
            min(max_val, center_val + 0.25 * value_range)
        ]

    # Round to clean numbers
    clean_refs = [round_to_clean(v) for v in ref_candidates]

    # Find indices of these clean values in the actual grid
    # Since we designed the grid to include clean values, they should match exactly (or very close)
    ref_indices = [np.argmin(np.abs(values - ref)) for ref in clean_refs]
    ref_values = [values[idx] for idx in ref_indices]

    # Print diagnostic info
    print(f"[Debug] Selected reference values:")
    for i, (clean, actual, idx) in enumerate(zip(clean_refs, ref_values, ref_indices)):
        print(f"  Ref {i+1}: target={clean:.2f}, actual={actual:.2f}, index={idx}")

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(7.2, 2.0))

    # ===== Subplot 1: Heatmap =====
    ax1 = plt.subplot(1, 3, 1)

    im = ax1.imshow(similarity, cmap='plasma', aspect='auto',
                    vmin=0, vmax=1, origin='lower')

    # Set ticks - use clean tick values
    min_val, max_val = values[0], values[-1]
    value_range = max_val - min_val

    # Determine appropriate tick spacing based on range
    if value_range <= 5:
        tick_spacing = 0.5
    elif value_range <= 20:
        tick_spacing = 2.0
    elif value_range <= 50:
        tick_spacing = 5.0
    else:
        tick_spacing = 10.0

    # Generate clean tick values
    tick_values = np.arange(
        np.ceil(min_val / tick_spacing) * tick_spacing,
        max_val + tick_spacing/2,
        tick_spacing
    )

    # Find closest indices for these tick values
    tick_indices = [np.argmin(np.abs(values - tv)) for tv in tick_values]
    tick_labels = [f"{tv:.1f}" for tv in tick_values]

    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=0, fontsize=5)
    ax1.set_yticks(tick_indices)
    ax1.set_yticklabels(tick_labels, fontsize=5)

    # Get unit for labels
    unit = get_property_unit(property_name)
    unit_label = f' ({unit})' if unit else ''

    ax1.set_xlabel(f'{property_name.title()}{unit_label}', fontsize=5)
    ax1.set_ylabel(f'{property_name.title()}{unit_label}', fontsize=5)
    # ax1.set_title(f'(A) Similarity Matrix', fontsize=10, fontweight='bold')

    # Colorbar
    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Cosine Similarity', rotation=270, labelpad=7, fontsize=5)
    cbar1.ax.tick_params(labelsize=5)
    cbar1.outline.set_linewidth(0.5)

    # Set ax1 spines linewidth
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
    ax1.tick_params(width=0.5)

    # ===== Subplot 2: Similarity to Multiple References (SWAPPED) =====
    ax2 = plt.subplot(1, 3, 2)

    # Define colors for the three reference lines
    ref_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    # Plot similarity for each reference value
    for i, (ref_idx, ref_val, clean_ref, color) in enumerate(zip(ref_indices, ref_values, clean_refs, ref_colors)):
        similarities_to_ref = similarity[ref_idx, :]

        # Plot curve with clean label
        ax2.plot(values, similarities_to_ref, 'o-', linewidth=0.5, markersize=1,
                label=f'Ref: {clean_ref:.1f}', color=color, alpha=0.8)

        # Highlight reference point with vertical line and star at ACTUAL value position
        # The star should be at (ref_val, 1.0) since similarity with itself is always 1.0
        ax2.axvline(ref_val, color=color, linestyle='--', alpha=0.3, linewidth=0.5)
        ax2.plot(ref_val, 1.0, '*', color=color, markersize=5, zorder=10,
                markeredgecolor='black', markeredgewidth=0.5)

    ax2.set_xlabel(f'{property_name.title()}{unit_label}', fontsize=5)
    ax2.set_ylabel('Text Embedding Similarity', fontsize=5)
    # ax2.set_title(f'(B) Similarity to Reference Values', fontsize=10, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.tick_params(labelsize=5, width=0.5)

    # Configure legend with linewidth
    legend = ax2.legend(loc='lower right', fontsize=5)
    legend.get_frame().set_linewidth(0.5)

    # Set all ax spines linewidth
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)
    # ===== Subplot 3: Similarity vs Difference (SWAPPED) =====
    ax3 = plt.subplot(1, 3, 3)

    # Compute all pairwise differences and similarities
    differences = []
    similarities = []

    n = len(values)
    for i in range(n):
        for j in range(i+1, n):
            diff = abs(values[i] - values[j])
            sim = similarity[i, j]
            differences.append(diff)
            similarities.append(sim)

    differences = np.array(differences)
    similarities = np.array(similarities)

    # Scatter plot
    scatter = ax3.scatter(differences, similarities, alpha=0.3, s=2, c=similarities,
                         cmap='viridis', vmin=0, vmax=1)

    # Bin and plot mean trend
    n_bins = 20
    bins = np.linspace(0, differences.max(), n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        mask = (differences >= bins[i]) & (differences < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(similarities[mask].mean())
            bin_stds.append(similarities[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    # Plot mean trend
    valid = ~np.isnan(bin_means)
    ax3.plot(bin_centers[valid], bin_means[valid], 'r-', linewidth=0.5,
            label='Mean similarity', zorder=10)
    ax3.fill_between(bin_centers[valid],
                     bin_means[valid] - bin_stds[valid],
                     bin_means[valid] + bin_stds[valid],
                     alpha=0.3, color='red', label='±1 std', zorder=5)

    # Add colorbar for scatter plot
    cbar3 = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Cosine Similarity', rotation=270, labelpad=7, fontsize=5)
    cbar3.ax.tick_params(labelsize=5)
    cbar3.outline.set_linewidth(0.5)

    ax3.set_xlabel(f'Absolute Difference in {property_name.title()}{unit_label}', fontsize=5)
    ax3.set_ylabel('Text Embedding Similarity', fontsize=5)
    # ax3.set_title(f'(C) Similarity vs Value Difference', fontsize=10, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.set_xlim([0, differences.max() * 1.05])
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.tick_params(labelsize=5, width=0.5)

    # Configure legend with linewidth
    legend = ax3.legend(loc='lower right', fontsize=5)
    legend.get_frame().set_linewidth(0.5)

    # Set all ax spines linewidth
    for spine in ax3.spines.values():
        spine.set_linewidth(0.5)
    # ===== Subplot 3: Similarity vs 
    # Overall title
    # fig.suptitle(f'Text Embedding Similarity Analysis: {property_name.title()}',
    #             fontsize=12, fontweight='bold', y=1.00)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    property_safe = property_name.replace(' ', '_')
    output_file = output_dir / f'similarity_analysis_{property_safe}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved combined plot to {output_file}")

    output_file_svg = output_dir / f'similarity_analysis_{property_safe}.svg'
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"[Plot] Saved combined plot to {output_file_svg}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze text embedding similarity for different property values'
    )
    parser.add_argument('--model-ckpt', type=str, default=None,
                       help='Path to model checkpoint (if not provided, uses untrained model from config)')
    parser.add_argument('--config', type=str, default='config/model/clac-lite.yaml',
                       help='Path to model config file (used only when --model-ckpt is not provided)')
    parser.add_argument('--no-projection', action='store_true',
                       help='Use raw LLM features instead of projection layer (useful for untrained models)')
    parser.add_argument('--property', type=str, default='band gap',
                       choices=['band gap', 'formation energy per atom', 'density',
                               'scintillation attenuation length', 'total magnetization',
                               'energy above hull'],
                       help='Property to analyze')
    parser.add_argument('--min-value', type=float, default=0.0,
                       help='Minimum property value')
    parser.add_argument('--max-value', type=float, default=4.0,
                       help='Maximum property value')
    parser.add_argument('--n-values', type=int, default=50,
                       help='Number of values to test')
    parser.add_argument('--reference-value', type=float, default=None,
                       help='Reference value for similarity comparison')
    parser.add_argument('--threshold-mode', type=str, default=None,
                       choices=['greater_than', 'less_than', 'greater_equal', 'less_equal'],
                       help='Enable threshold analysis mode (e.g., "greater_than" for "> threshold")')
    parser.add_argument('--threshold-value', type=float, default=None,
                       help='Threshold value for range-based analysis (required with --threshold-mode)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='outputs/text_embedding_similarity',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Validate threshold arguments
    if args.threshold_mode and args.threshold_value is None:
        parser.error("--threshold-value is required when --threshold-mode is specified")
    if args.threshold_value is not None and not args.threshold_mode:
        parser.error("--threshold-mode is required when --threshold-value is specified")

    # Determine whether to use projection
    use_projection = not args.no_projection

    print("="*80)
    print("TEXT EMBEDDING SIMILARITY ANALYSIS")
    print("="*80)
    if args.model_ckpt:
        print(f"Model checkpoint: {args.model_ckpt}")
    else:
        print(f"Model: Untrained (random initialization)")
        print(f"Config file:      {args.config}")
    print(f"Property:         {args.property}")
    print(f"Value range:      {args.min_value:.2f} to {args.max_value:.2f}")
    print(f"Number of values: {args.n_values}")
    print(f"Embedding mode:   {'Projection layer' if use_projection else 'Raw LLM features'}")
    print(f"Device:           {args.device}")
    print("="*80 + "\n")

    # Load model
    print("[Step 1] Loading model...")
    if args.model_ckpt:
        # Load from checkpoint
        model = CLaCLite.load_from_checkpoint(args.model_ckpt, map_location={'cuda:0': 'cpu'}, strict=False)
        print(f"[Step 1] Model loaded from checkpoint: {args.model_ckpt}")
    else:
        # Initialize untrained model from config using Hydra
        import hydra
        from hydra import compose, initialize_config_dir

        # Get absolute path to config directory
        config_dir = Path(__file__).parent / "config"
        config_dir = config_dir.resolve()

        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        print(f"[Step 1] Using config directory: {config_dir}")

        # Initialize Hydra with config directory
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Compose config with overrides to use minimal settings
            cfg = compose(
                config_name="config",
                overrides=[
                    "debug=true",  # Use debug mode to avoid heavy setup
                    "resume_from_checkpoint=null",
                ]
            )

            print(f"[Step 1] Config loaded, instantiating model...")

            # Instantiate model using hydra
            model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            print(f"[Step 1] Initialized untrained model from config")

    model = model.to(args.device)
    model.eval()
    print(f"[Step 1] Model ready on device: {args.device}")

    # Generate values with property-specific spacing
    # This ensures clean reference values (integers or 0.5 increments) are exactly on grid points
    print("\n[Step 2] Generating property values...")

    # Define appropriate spacing for each property to hit clean reference values
    property_spacing = {
        'band gap': 0.1,  # 0.0, 0.1, ..., 1.0, ..., 2.0, ..., 3.0, ..., 4.0 (41 values)
        'formation energy per atom': 0.1,  # -3.0, -2.9, ..., -1.0, ..., 0.0, ..., 1.0 (41 values)
        'density': 0.2,  # 2.0, 2.2, ..., 5.0, ..., 10.0 (41 values)
        'scintillation attenuation length': 0.5,  # 0.5, 1.0, ..., 5.0, ..., 20.0 (40 values)
        'total magnetization': 1.0,  # 0.0, 1.0, ..., 10.0, ..., 50.0 (51 values)
        'energy above hull': 0.02,  # 0.0, 0.02, 0.04, ..., 0.1, ..., 0.2, ..., 1.0 (51 values)
    }

    spacing = property_spacing.get(args.property, 0.1)  # Default 0.1 if not specified

    # Generate values using arange with the specified spacing
    values = np.arange(args.min_value, args.max_value + spacing/2, spacing)

    # Ensure we don't exceed max_value due to floating point errors
    values = values[values <= args.max_value + 1e-10]

    print(f"[Step 2] Testing {len(values)} values from {values[0]:.2f} to {values[-1]:.2f}")
    print(f"[Step 2] Spacing: {spacing} (ensures clean reference values on grid)")

    # Decide which analysis to run
    if args.threshold_mode:
        # Run threshold-based analysis
        print("\n[Step 3] Running threshold-based similarity analysis...")
        threshold_results = analyze_threshold_similarity(
            model, args.property, values, args.threshold_value,
            threshold_mode=args.threshold_mode,
            device=args.device,
            use_projection=use_projection
        )

        # Generate plots
        print("\n[Step 4] Generating threshold analysis plots...")
        plot_threshold_analysis(threshold_results, args.output_dir)

    else:
        # Run standard pairwise similarity analysis
        print("\n[Step 3] Computing text embeddings and similarity...")
        results = analyze_property_similarity(model, args.property, values, args.device, use_projection=use_projection)
        print(f"[Step 3] Similarity analysis complete")
        print(f"[Step 3] Similarity matrix shape: {results['similarity_matrix'].shape}")
        print(f"[Step 3] Mean similarity: {results['similarity_matrix'].mean():.4f}")
        print(f"[Step 3] Min similarity:  {results['similarity_matrix'].min():.4f}")
        print(f"[Step 3] Max similarity:  {results['similarity_matrix'].max():.4f}")

        # Generate plots
        print("\n[Step 4] Generating plots...")
        plot_all_in_one(results, args.output_dir, args.reference_value)

    print("\n" + "="*80)
    print("[DONE] Text embedding similarity analysis completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
