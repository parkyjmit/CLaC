import sys
from pathlib import Path
# Add parent directory to path to import data and model modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datamodule import QuestionEvaluationDataModule, CLaCDataModule
from model.clac import CLaCLite
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from scipy import stats
import hydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf


def extract_checkpoint_name(ckpt_path, use_untrained=False):
    """Extract meaningful name from checkpoint path for organizing results."""
    if use_untrained:
        return "untrained_baseline"

    ckpt_path = Path(ckpt_path)
    # Get parent directory name (e.g., 'SynData-LMloss-painn-infonce-galactica-125m-vTrue-tTrue-50epoch')
    parent_dir = ckpt_path.parent.name
    # Get checkpoint file name without extension (e.g., 'last' or 'epoch=99-step=2500')
    ckpt_name = ckpt_path.stem
    # Combine them
    return f"{parent_dir}_{ckpt_name}"


def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence_level=0.95, statistic_fn=np.mean):
    """
    Calculate confidence interval using bootstrapping.

    Args:
        data: Array-like of binary outcomes (0 or 1) or continuous values
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        confidence_level: Confidence level for interval (default: 0.95 for 95% CI)
        statistic_fn: Function to compute statistic (default: np.mean for accuracy)

    Returns:
        dict: {
            'mean': mean of statistic,
            'std': standard deviation of bootstrap distribution,
            'ci_lower': lower bound of confidence interval,
            'ci_upper': upper bound of confidence interval,
            'n_bootstrap': number of bootstrap samples used
        }

    Example:
        >>> accuracies = [1, 0, 1, 1, 0, ...]  # Binary outcomes for each test sample
        >>> result = bootstrap_confidence_interval(accuracies, n_bootstrap=1000)
        >>> print(f"Accuracy: {result['mean']:.3f} ± {result['std']:.3f}")
        >>> print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    """
    data = np.array(data)
    n = len(data)

    if n == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
            'n_bootstrap': 0
        }

    # Original statistic
    original_stat = statistic_fn(data)

    # Bootstrap sampling
    bootstrap_stats = []
    rng = np.random.RandomState(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = rng.choice(data, size=n, replace=True)
        bootstrap_stat = statistic_fn(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'mean': float(original_stat),
        'std': float(np.std(bootstrap_stats)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_bootstrap': n_bootstrap
    }


# Template functions for query text generation (used by both single and dual inverse design)
def get_bandgap_template(value):
    """Return application-specific template based on band gap value."""
    # if value == 0.0:
    #     return 'metallic material with bandgap around {value} eV suitable for electrical conductors and electrodes'
    # elif value <= 1.2:
    #     return 'material with bandgap around {value} eV suitable for infrared photodetectors and thermoelectric devices'
    # elif value <= 1.8:
    #     return 'material with bandgap around {value} eV ideal for solar cell and photovoltaic applications'
    # elif value <= 3.0:
    #     return 'material with bandgap around {value} eV suitable for LED and visible light applications'
    # else:  # > 3.0
    #     return 'wide bandgap material around {value} eV for UV detectors and high-power electronics'
    return 'bandgap around {value} eV'


def get_scintillation_template(value):
    """Return application-specific template based on scintillation attenuation length."""
    # if value <= 1.5:
    #     return 'scintillator with attenuation length around {value} cm for compact detectors and high spatial resolution applications'
    # elif value <= 3.0:
    #     return 'scintillator with attenuation length around {value} cm suitable for medical PET scanners and gamma ray spectroscopy'
    # elif value <= 5.0:
    #     return 'scintillator with attenuation length around {value} cm for radiation monitoring and nuclear security applications'
    # elif value <= 10.0:
    #     return 'scintillator with attenuation length around {value} cm ideal for high-energy physics calorimeters and thick detector systems'
    # else:  # > 10.0
    #     return 'scintillator with long attenuation length around {value} cm for deep penetration cosmic ray detection and large-scale experiments'
    return 'scintillation attenuation length around {value} cm'


def get_magnetization_template(value):
    """Return application-specific template based on total magnetization."""
    # if value == 0.0:
    #     return 'non-magnetic material with magnetization around {value} μB/f.u.'
    # elif value <= 2.0:
    #     return 'weakly magnetic material with magnetization around {value} μB/f.u.'
    # elif value <= 5.0:
    #     return 'ferromagnetic material with magnetization around {value} μB/f.u.'
    # elif value <= 10.0:
    #     return 'ferromagnetic material with magnetization around {value} μB/f.u.'
    # else:  # > 10.0
    #     return 'strongly magnetic material with magnetization around {value} μB/f.u.'
    return 'total magnetization around {value} μB/f.u.'

def get_stability_template(value):
    """Return application-specific template based on energy above hull (stability indicator)."""
    # if value <= 0.05:
    #     return 'highly stable material with energy above hull around {value} eV/atom suitable for commercial production and mass manufacturing'
    # elif value <= 0.15:
    #     return 'metastable material with energy above hull around {value} eV/atom for thin film applications and special synthesis conditions'
    # elif value <= 0.3:
    #     return 'material with energy above hull around {value} eV/atom accessible through high-pressure or high-temperature synthesis'
    # else:  # > 0.3
    #     return 'thermodynamically unstable material with energy above hull around {value} eV/atom for theoretical studies and extreme conditions'
    return 'energy above hull around {value} eV/atom'


def get_density_template(value):
    """Return application-specific template based on density."""
    # if value <= 3.0:
    #     return 'low-density material around {value} g/cm³ suitable for lightweight structures and aerospace applications'
    # elif value <= 5.0:
    #     return 'material with density around {value} g/cm³ ideal for structural ceramics and general engineering applications'
    # elif value <= 8.0:
    #     return 'dense material around {value} g/cm³ for radiation shielding and heavy-duty structural components'
    # else:  # > 8.0
    #     return 'high-density material around {value} g/cm³ for X-ray/gamma-ray shielding and counterweights'
    return 'density around {value} g/cm³'


# ============================================================================
# TEXT-ONLY INVERSE DESIGN: Application-based queries (NO property values)
# ============================================================================

def get_text_only_application_queries():
    """
    Define application-based queries without explicit property values.

    Returns:
        dict: Property -> list of application queries with expected ranges
    """
    return {
        'band gap': [
            {
                'name': 'electrical_conductors',
                'query': 'material suitable for electrical conductors and electrodes',
                'expected_range': (0.0, 0.1),
            },
            {
                'name': 'infrared_detectors',
                'query': 'material suitable for infrared photodetectors and thermoelectric devices',
                'expected_range': (0.1, 1.2),
            },
            {
                'name': 'solar_cells',
                'query': 'material ideal for solar cell and photovoltaic applications',
                'expected_range': (1.1, 1.8),
            },
            {
                'name': 'leds',
                'query': 'material suitable for LED and visible light applications',
                'expected_range': (1.8, 3.0),
            },
            {
                'name': 'uv_detectors',
                'query': 'wide bandgap material for UV detectors and high-power electronics',
                'expected_range': (3.0, 8.7),
            },
        ],
        'density': [
            {
                'name': 'structural_ceramics',
                'query': 'low-density material ideal for structural ceramics material',
                'expected_range': (0.1, 3.0),
            },
            {
                'name': 'lightweight_aerospace',
                'query': 'material suitable for lightweight structures and aerospace applications',
                'expected_range': (2.5, 4.0),
            },
            {
                'name': 'alloys',
                'query': 'material for industrial alloys and structural components',
                'expected_range': (4.0, 8.0),
            },
            {
                'name': 'radiation_shielding',
                'query': 'high-density structural alloy for X-ray shielding and radiation protection',
                'expected_range': (8.0, 12.0),
            },
        ],
        'scintillation attenuation length': [
            {
                'name': 'compact_detectors',
                'query': 'scintillator for compact detectors and high spatial resolution applications',
                'expected_range': (0.0, 1.5),
            },
            {
                'name': 'medical_pet',
                'query': 'scintillator suitable for medical PET scanners and gamma ray spectroscopy',
                'expected_range': (1.5, 3.0),
            },
            {
                'name': 'radiation_monitoring',
                'query': 'scintillator for radiation monitoring and nuclear security applications',
                'expected_range': (3.0, 5.0),
            },
            {
                'name': 'calorimeters',
                'query': 'scintillator ideal for high-energy physics calorimeters and thick detector systems',
                'expected_range': (5.0, 10.0),
            },
            {
                'name': 'cosmic_ray',
                'query': 'scintillator with long attenuation length for deep penetration cosmic ray detection and large-scale experiments',
                'expected_range': (10.0, 24.0),
            },
        ],
        'total magnetization': [
            {
                'name': 'non_magnetic',
                'query': 'non-magnetic material',
                'expected_range': (0.0, 0.5),
            },
            {
                'name': 'weakly_magnetic',
                'query': 'weakly magnetic material',
                'expected_range': (0.5, 5.0),
            },
            {
                'name': 'strongly_magnetic',
                'query': 'strongly magnetic material for permanent magnets',
                'expected_range': (5.0, 20.0),
            },
        ],
        'energy above hull': [
            {
                'name': 'stable',
                'query': 'thermodynamically stable material suitable for commercial production and mass manufacturing',
                'expected_range': (0.0, 0.1),
            },
            {
                'name': 'metastable',
                'query': 'metastable material for thin film applications and special synthesis conditions',
                'expected_range': (0.1, 0.5),
            },
            {
                'name': 'unstable',
                'query': 'hypothetical unstable material for theoretical studies and specialized conditions',
                'expected_range': (0.5, 1.0),
            },
        ],
    }


def evaluate_inverse_design(cfg):
    """
    Inverse materials design evaluation.
    Query: "material with bandgap around X eV"
    Measure: Retrieved materials' actual bandgap values
    """
    print(f"[Inverse Design] Property: {cfg.property}, K values: {cfg.k_values}")

    # Load test dataset
    data_path = {'test': cfg.data_path + '_test.parquet'}
    dataset = load_dataset('parquet', data_files=data_path)['test']
    print(f"[Inverse Design] Loaded {len(dataset)} materials")

    # Load model
    # Use strict=False to allow loading checkpoints with different loss configurations
    model = CLaCLite.load_from_checkpoint(cfg.model_ckpt, map_location={'cuda:0': 'cpu'}, strict=False)
    model = model.to(cfg.device)
    model.eval()
    model.text_encoder.config.output_hidden_states = True

    # Auto-detect graphdatatype from model if not specified
    if cfg.graphdatatype == 'auto':
        # Check model's graph encoder type
        if hasattr(model.graph_encoder, 'orb_model'):
            detected_graphdatatype = 'orb'
            print("[Inverse Design] Auto-detected graph encoder: ORB")
        else:
            detected_graphdatatype = 'torch_geometric'
            print("[Inverse Design] Auto-detected graph encoder: torch_geometric")
    else:
        detected_graphdatatype = cfg.graphdatatype
        print(f"[Inverse Design] Using specified graph encoder: {detected_graphdatatype}")

    # Setup datamodule for graph encoding
    dm = CLaCDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        graphdatatype=detected_graphdatatype,
        tokenizer_model=cfg.llm,
        debug=False,
    )
    dm.setup()
    test_dataloader = dm.test_dataloader()

    # Step 1: Encode all test materials (graphs)
    print("[Inverse Design] Encoding all test materials...")
    all_graph_embeddings = []
    all_property_values = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            graphs, _ = batch
            graphs = graphs.to(cfg.device)

            # Encode graphs
            g_feat = model.graph_encoder(graphs)  # (batch, hidden)
            g_emb = model.loss.global_d.img_block(g_feat)  # Project through img_block
            g_emb = F.normalize(g_emb, p=2, dim=-1)  # Normalize embeddings

            all_graph_embeddings.append(g_emb.cpu())

            # Get property values for this batch
            start_idx = batch_idx * cfg.batch_size
            end_idx = start_idx + len(g_emb)
            batch_properties = [dataset[i][cfg.property] for i in range(start_idx, min(end_idx, len(dataset)))]
            all_property_values.extend(batch_properties)

    all_graph_embeddings = torch.cat(all_graph_embeddings, dim=0)  # (N, hidden)
    all_property_values = np.array(all_property_values)  # (N,)
    print(f"[Inverse Design] Encoded {len(all_graph_embeddings)} materials")

    # Step 2: Define query targets with application-aware templates (use module-level functions)
    property_config = {
        'band gap': {
            'targets': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'template_function': get_bandgap_template,  # Use function for application-aware templates
            'unit': 'eV',
            'precision': 1  # 1 decimal place (e.g., 1.5 eV)
        },
        'formation energy per atom': {
            'targets': [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
            'template': 'material with formation energy around {value} eV per atom',
            'unit': 'eV/atom',
            'precision': 1  # 1 decimal place
        },
        'density': {
            'targets': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'template_function': get_density_template,
            'unit': 'g/cm³',
            'precision': 1  # 1 decimal place
        },
        'scintillation attenuation length': {
            'targets': [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00],
            'template_function': get_scintillation_template,
            'unit': 'cm',
            'precision': 2  # 2 decimal places (e.g., 7.50 cm)
        },
        'total magnetization': {
            'targets': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
            'template_function': get_magnetization_template,
            'unit': 'μB/f.u.',
            'precision': 2  # 2 decimal places for small values (0.01)
        },
        'energy above hull': {
            'targets': [0.0, 0.1, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7, 0.8],
            'template_function': get_stability_template,
            'unit': 'eV/atom',
            'precision': 2  # 2 decimal places (0.01, 0.03, etc.)
        }
    }

    config = property_config[cfg.property]
    query_targets = config['targets']
    precision = config.get('precision', 2)  # Default to 2 decimal places

    # Use template_function if available, otherwise use static template
    if 'template_function' in config:
        # Template function returns format string, we need to fill in the value
        get_template = config['template_function']
    else:
        # Static template
        static_template = config['template']
        get_template = lambda _: static_template

    # Step 3: For each query target, retrieve top-K materials
    results = {
        'query_targets': query_targets,
        'k_values': cfg.k_values,
        'property': cfg.property,
        'unit': config['unit'],  # Save unit for plotting
        'per_query_results': [],
        'all_property_values': all_property_values  # Save for histogram plotting
    }

    print(f"[Inverse Design] Evaluating {len(query_targets)} queries...")
    for target in tqdm(query_targets):
        template = get_template(target)
        # Format value with appropriate precision
        formatted_value = f"{target:.{precision}f}"
        query_text = template.format(value=formatted_value)

        # Encode query text - use model's tokenizer for compatibility
        tokenized = model.tokenizer(query_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        tokenized = {k: v.to(cfg.device) for k, v in tokenized.items()}

        with torch.no_grad():
            _, q_emb = model.encode_text(tokenized)
            q_emb_norm = model.loss.global_d.text_block(q_emb)  # Project through text_block
            q_emb_norm = F.normalize(q_emb_norm, p=2, dim=-1)  # Normalize embeddings

        q_emb_norm = q_emb_norm.cpu()

        # Compute similarity with all materials
        similarities = torch.matmul(q_emb_norm, all_graph_embeddings.T).squeeze()  # (N,)

        # For each K value, retrieve top-K and compute metrics
        query_result = {'target': target, 'query': query_text}

        for k in cfg.k_values:
            # Top-K retrieval
            topk_indices = torch.topk(similarities, k=min(k, len(similarities))).indices.numpy()
            retrieved_values = all_property_values[topk_indices]

            # Filter out NaN/None values
            retrieved_values = retrieved_values[~pd.isna(retrieved_values)]

            if len(retrieved_values) > 0:
                retrieved_mean = float(np.mean(retrieved_values))
                retrieved_std = float(np.std(retrieved_values))
                mae = abs(retrieved_mean - target)
            else:
                retrieved_mean = retrieved_std = mae = float('nan')

            query_result[f'k{k}_mean'] = retrieved_mean
            query_result[f'k{k}_std'] = retrieved_std
            query_result[f'k{k}_mae'] = mae
            query_result[f'k{k}_values'] = retrieved_values.tolist()  # Save individual values for plotting

        results['per_query_results'].append(query_result)

    # Step 4: Compute overall metrics
    for k in cfg.k_values:
        maes = [r[f'k{k}_mae'] for r in results['per_query_results'] if not np.isnan(r[f'k{k}_mae'])]
        results[f'k{k}_overall_mae'] = float(np.mean(maes)) if maes else float('nan')

        # Correlation between query target and retrieved mean
        targets = [r['target'] for r in results['per_query_results']]
        means = [r[f'k{k}_mean'] for r in results['per_query_results']]
        if len(targets) > 1:
            results[f'k{k}_correlation'] = float(np.corrcoef(targets, means)[0, 1])
        else:
            results[f'k{k}_correlation'] = float('nan')

    # Step 5: Random baseline
    print("[Inverse Design] Computing random baseline...")
    random_results = []
    for target in query_targets:
        query_result = {'target': target}
        for k in cfg.k_values:
            random_indices = np.random.choice(len(all_property_values), size=min(k, len(all_property_values)), replace=False)
            random_values = all_property_values[random_indices]
            random_values = random_values[~pd.isna(random_values)]

            if len(random_values) > 0:
                random_mean = float(np.mean(random_values))
                random_mae = abs(random_mean - target)
            else:
                random_mean = random_mae = float('nan')

            query_result[f'k{k}_mean'] = random_mean
            query_result[f'k{k}_mae'] = random_mae

        random_results.append(query_result)

    results['random_baseline'] = random_results

    # Compute random baseline overall MAE
    for k in cfg.k_values:
        random_maes = [r[f'k{k}_mae'] for r in random_results if not np.isnan(r[f'k{k}_mae'])]
        results[f'random_k{k}_overall_mae'] = float(np.mean(random_maes)) if random_maes else float('nan')

    # Step 6: Save results
    # Extract checkpoint name and create checkpoint-specific directory
    use_untrained = getattr(cfg, 'use_untrained', False)
    ckpt_name = extract_checkpoint_name(cfg.model_ckpt, use_untrained=use_untrained)
    output_dir = Path('outputs/inverse_design') / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Inverse Design] Saving results to {output_dir}/")

    # Convert numpy array to list for JSON serialization
    results_to_save = results.copy()
    results_to_save['all_property_values'] = results['all_property_values'].tolist()

    results_file = output_dir / f'inverse_design_{cfg.property.replace(" ", "_")}.json'
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"[Inverse Design] Results saved to {results_file}")

    # Step 7: Print summary
    print("\n" + "="*80)
    print(f"INVERSE DESIGN RESULTS - {cfg.property}")
    print("="*80)
    for k in cfg.k_values:
        print(f"\nK={k}:")
        print(f"  CLaC MAE:    {results[f'k{k}_overall_mae']:.4f} {config['unit']}")
        print(f"  Random MAE:  {results[f'random_k{k}_overall_mae']:.4f} {config['unit']}")
        print(f"  Correlation: {results[f'k{k}_correlation']:.4f}")
    print("="*80 + "\n")

    # Step 8: Visualize
    if cfg.plot:
        plot_inverse_design_results(results, cfg, output_dir)

    return results


def plot_inverse_design_results(results, cfg, output_dir):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.dpi'] = 300
    mpl.rc('font', family='Arial', size=5)
    # For SVG
    mpl.rcParams["svg.fonttype"] = "none"
    # For PDF
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42

    """Create visualization plots for inverse design results."""
    property_name = cfg.property
    k_values = cfg.k_values
    query_targets = results['query_targets']
    unit = results['unit']  # Get unit from results

    # Create figure with subplots
    n_plots = len(k_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(1.8*n_plots, 2.1))
    if n_plots == 1:
        axes = [axes]

    # Create colormap for query targets - property-specific colors
    import matplotlib.cm as cm

    # Define colormap for each property
    property_colormaps = {
        'band gap': 'viridis',
        'density': 'spring',
        'scintillation attenuation length': 'summer',
        'total magnetization': 'autumn',
        'energy above hull': 'winter',
        'formation energy per atom': 'viridis'  # default for any other property
    }

    cmap_name = property_colormaps.get(property_name, 'viridis')
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=min(query_targets), vmax=max(query_targets))

    # Compute axis ranges based on query targets with 15% margin
    # This prevents outlier retrieved values from distorting the plot scale
    query_min = min(query_targets)
    query_max = max(query_targets)
    query_range = query_max - query_min
    margin = query_range * 0.20

    global_min_val = query_min - margin*1.75
    global_max_val = query_max + margin

    # Get candidate pool property values for KDE
    all_property_values = np.array(results['all_property_values'])
    pool_values = all_property_values[~pd.isna(all_property_values)]

    print(f"[Plot] Property: {property_name}")
    print(f"[Plot] Pool values: min={pool_values.min():.4f}, max={pool_values.max():.4f}, count={len(pool_values)}")
    print(f"[Plot] Pool values std: {pool_values.std():.4f}, unique values: {len(np.unique(pool_values))}")
    print(f"[Plot] Query range: [{global_min_val:.4f}, {global_max_val:.4f}]")

    # Check distribution around 0
    near_zero = np.sum(np.abs(pool_values) < 0.01)
    print(f"[Plot] Values near 0.0 (|x|<0.01): {near_zero} ({100*near_zero/len(pool_values):.1f}%)")

    # Compute histogram for candidate pool distribution (number of candidates)
    if len(pool_values) > 1:
        # Use histogram to show actual counts
        hist_counts, hist_bins = np.histogram(pool_values, bins=50, range=(global_min_val, global_max_val))
        y_range_hist = np.array((hist_bins[:-1] + hist_bins[1:]) / 2, dtype=np.float64)
        counts_hist = np.array(hist_counts, dtype=np.float64)
        print(f"[Plot] Histogram computed: {len(hist_counts)} bins")
    else:
        y_range_hist = np.array([global_min_val, global_max_val])
        counts_hist = np.array([0, 0])

    # Max count for scaling
    max_count = counts_hist.max() if len(counts_hist) > 0 and counts_hist.max() > 0 else 1
    print(f"[Plot] Candidate pool counts: min={counts_hist.min():.0f}, max={counts_hist.max():.0f}, total={counts_hist.sum():.0f}")

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        # Extract data
        clac_means = [r[f'k{k}_mean'] for r in results['per_query_results']]
        random_means = [r[f'k{k}_mean'] for r in results['random_baseline']]

        # Create twin axis for histogram (top x-axis)
        ax_hist = ax.twiny()

        # Plot histogram as horizontal filled curve (FIRST - lowest zorder)
        ax_hist.fill_betweenx(y_range_hist, 0, counts_hist, alpha=0.2, color='gray', zorder=-1)
        ax_hist.plot(counts_hist, y_range_hist, color='gray', linewidth=.5, alpha=0.5, zorder=-1)

        # Configure histogram axis (top)
        ax_hist.set_xlim(0, max_count * 1.1)
        ax_hist.set_ylim(global_min_val, global_max_val)
        ax_hist.set_xlabel('Number of Candidates', fontsize=5, color='gray')
        ax_hist.tick_params(axis='x', labelsize=5, colors='gray', width=0.5, length=3)
        ax_hist.spines['top'].set_color('gray')
        ax_hist.spines['top'].set_alpha(0.5)
        ax_hist.spines['top'].set_linewidth(0.5)

        # Set all ax_hist spines linewidth
        for spine in ax_hist.spines.values():
            spine.set_linewidth(0.5)

        # Plot perfect alignment (x=y) using global range
        ax.plot([global_min_val, global_max_val], [global_min_val, global_max_val], 'k--', alpha=0.5, linewidth=.5, zorder=1)

        # Plot individual retrieved values as small transparent points with colors
        for i, target in enumerate(query_targets):
            color = cmap(norm(target))
            if f'k{k}_values' in results['per_query_results'][i]:
                values = results['per_query_results'][i][f'k{k}_values']
                # Add small random jitter to y-axis for better visibility
                y_jitter = np.random.normal(0, 0.02 * (max(query_targets) - min(query_targets)), len(values))
                ax.scatter(values, [target + jitter for jitter in y_jitter],
                          s=1, alpha=0.25, color=color, zorder=2)

        # Add error bars for CLaC (now horizontal)
        clac_stds = [r[f'k{k}_std'] for r in results['per_query_results']]
        ax.errorbar(clac_means, query_targets, xerr=clac_stds, fmt='none',
                   ecolor='black', alpha=0.3, capsize=3, linewidth=.5, capthick=.5, zorder=3)

        # Plot CLaC results (mean) with colors (no label - for actual data)
        scatter = ax.scatter(clac_means, query_targets, s=10, alpha=0.8,
                  c=query_targets, cmap=cmap_name, norm=norm,
                  zorder=4, edgecolors='black', linewidths=.5)

        # Add invisible scatter for legend (gray color)
        ax.scatter([], [], s=10, alpha=0.8, color='gray',
                  edgecolors='black', linewidths=.5,
                  label=f'CLaC (MAE={results[f"k{k}_overall_mae"]:.3f})')#, Corr={results[f"k{k}_correlation"]:.3f})')

        # Plot Random baseline (show MAE in each subplot since it varies with K)
        ax.scatter(random_means, query_targets, s=10, alpha=0.5,
                  color='black', marker='x', zorder=4, linewidths=.5,
                  label=f'Random (MAE={results[f"random_k{k}_overall_mae"]:.3f})')

        ax.set_ylabel(f'Query Target ({unit})')
        ax.set_xlabel(f'Retrieved Value ({unit})')
        ax.set_title(f'K={k}', fontsize=6)

        # Configure legend with linewidth
        legend = ax.legend(loc='lower right', fontsize=5)
        legend.get_frame().set_linewidth(0.5)
        # ax.grid(True, alpha=0.3)

        # Set all ax spines linewidth first
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Remove top and right spines on main axis (after setting linewidth)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set tick parameters
        ax.tick_params(width=0.5, length=3)

        # Set consistent axis limits across all subplots
        ax.set_xlim(global_min_val, global_max_val)
        ax.set_ylim(global_min_val, global_max_val)

        # # Add colorbar for the last subplot
        # if idx == len(k_values) - 1:
        #     cbar = plt.colorbar(scatter, ax=ax)
        #     cbar.set_label(f'Query Target ({property_name})', fontsize=10)

    plt.tight_layout()
    plot_file = output_dir / f'inverse_design_{property_name.replace(" ", "_")}.svg'

    try:
        print(f"[Plot] Attempting to save to: {plot_file}")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Inverse Design] Plot saved to {plot_file}")
    except Exception as e:
        print(f"[Plot] ERROR saving SVG: {e}")
        # Try saving as PNG instead
        plot_file_png = output_dir / f'inverse_design_{property_name.replace(" ", "_")}.png'
        try:
            plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
            print(f"[Plot] Saved as PNG instead: {plot_file_png}")
        except Exception as e2:
            print(f"[Plot] ERROR saving PNG: {e2}")

    plt.close()


def evaluate_text_only_inverse_design(cfg):
    """
    Text-only inverse design evaluation.
    Query by application (NO property values), measure retrieved materials' property distributions.

    This tests whether the model learned application-property relationships from synthetic narratives.
    """
    print(f"[Text-Only Inverse Design] Property: {cfg.property}, K values: {cfg.k_values}")

    # Load test dataset
    data_path = {'test': cfg.data_path + '_test.parquet'}
    dataset = load_dataset('parquet', data_files=data_path)['test']
    print(f"[Text-Only Inverse Design] Loaded {len(dataset)} materials")

    # Load model
    model = CLaCLite.load_from_checkpoint(cfg.model_ckpt, map_location={'cuda:0': 'cpu'}, strict=False)
    model = model.to(cfg.device)
    model.eval()
    model.text_encoder.config.output_hidden_states = True

    # Auto-detect graphdatatype
    if cfg.graphdatatype == 'auto':
        if hasattr(model.graph_encoder, 'orb_model'):
            detected_graphdatatype = 'orb'
            print("[Text-Only Inverse Design] Auto-detected graph encoder: ORB")
        else:
            detected_graphdatatype = 'torch_geometric'
            print("[Text-Only Inverse Design] Auto-detected graph encoder: torch_geometric")
    else:
        detected_graphdatatype = cfg.graphdatatype
        print(f"[Text-Only Inverse Design] Using specified graph encoder: {detected_graphdatatype}")

    # Setup datamodule
    dm = CLaCDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        graphdatatype=detected_graphdatatype,
        tokenizer_model=cfg.llm,
        debug=False,
    )
    dm.setup()
    test_dataloader = dm.test_dataloader()

    # Step 1: Encode all test materials (graphs)
    print("[Text-Only Inverse Design] Encoding all test materials...")
    all_graph_embeddings = []
    all_property_values = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            graphs, _ = batch
            graphs = graphs.to(cfg.device)

            # Encode graphs
            g_feat = model.graph_encoder(graphs)
            g_emb = model.loss.global_d.img_block(g_feat)
            g_emb = F.normalize(g_emb, p=2, dim=-1)

            all_graph_embeddings.append(g_emb.cpu())

            # Get property values for this batch
            start_idx = batch_idx * cfg.batch_size
            end_idx = start_idx + len(g_emb)
            batch_properties = [dataset[i][cfg.property] for i in range(start_idx, min(end_idx, len(dataset)))]
            all_property_values.extend(batch_properties)

    all_graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
    all_property_values = np.array(all_property_values)
    print(f"[Text-Only Inverse Design] Encoded {len(all_graph_embeddings)} materials")

    # Step 2: Get application queries (NO property values)
    application_queries_config = get_text_only_application_queries()

    if cfg.property not in application_queries_config:
        raise ValueError(f"Property '{cfg.property}' not supported for text-only inverse design. "
                        f"Available: {list(application_queries_config.keys())}")

    application_queries = application_queries_config[cfg.property]

    # Get unit for this property
    property_units = {
        'band gap': 'eV',
        'density': 'g/cm³',
        'scintillation attenuation length': 'cm',
        'total magnetization': 'μB/f.u.',
        'energy above hull': 'eV/atom',
        'formation energy per atom': 'eV/atom',
    }
    unit = property_units.get(cfg.property, '')

    # Step 3: For each application query, retrieve top-K materials and analyze distribution
    results = {
        'property': cfg.property,
        'unit': unit,
        'k_values': cfg.k_values,
        'application_results': [],
        'all_property_values': all_property_values.tolist(),
    }

    print(f"[Text-Only Inverse Design] Evaluating {len(application_queries)} application queries...")

    for app_query in tqdm(application_queries):
        app_name = app_query['name']
        query_text = app_query['query']
        expected_min, expected_max = app_query['expected_range']

        print(f"\n[Query] {app_name}: '{query_text}'")
        print(f"[Expected Range] {expected_min} - {expected_max} {unit}")

        # Encode query text
        tokenized = model.tokenizer(query_text, padding='max_length', truncation=True,
                                    max_length=128, return_tensors='pt')
        tokenized = {k: v.to(cfg.device) for k, v in tokenized.items()}

        with torch.no_grad():
            _, q_emb = model.encode_text(tokenized)
            q_emb_norm = model.loss.global_d.text_block(q_emb)
            q_emb_norm = F.normalize(q_emb_norm, p=2, dim=-1)

        q_emb_norm = q_emb_norm.cpu()

        # Compute similarity with all materials
        similarities = torch.matmul(q_emb_norm, all_graph_embeddings.T).squeeze()

        # Compute ground truth: materials in expected range
        valid_mask = ~pd.isna(all_property_values)
        valid_property_values = all_property_values[valid_mask]
        in_range_mask = (valid_property_values >= expected_min) & (valid_property_values <= expected_max)
        num_in_range_total = in_range_mask.sum()

        app_result = {
            'name': app_name,
            'query': query_text,
            'expected_range': (float(expected_min), float(expected_max)),
            'num_materials_in_range': int(num_in_range_total),
        }

        # For each K value
        for k in cfg.k_values:
            # Retrieve top-K
            topk_indices = torch.topk(similarities, k=min(k, len(similarities))).indices.numpy()
            retrieved_values = all_property_values[topk_indices]

            # Filter NaN
            retrieved_values = retrieved_values[~pd.isna(retrieved_values)]

            if len(retrieved_values) == 0:
                app_result[f'k{k}'] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'median': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'precision': 0.0,
                    'recall': 0.0,
                    'values': [],
                }
                continue

            # Compute statistics
            mean_val = float(np.mean(retrieved_values))
            std_val = float(np.std(retrieved_values))
            median_val = float(np.median(retrieved_values))
            min_val = float(np.min(retrieved_values))
            max_val = float(np.max(retrieved_values))

            # Compute precision and recall@k
            in_range_retrieved = ((retrieved_values >= expected_min) & (retrieved_values <= expected_max)).sum()
            precision = float(in_range_retrieved / len(retrieved_values)) if len(retrieved_values) > 0 else 0.0
            # Recall@K: proportion of relevant items found out of min(K, total_relevant)
            recall = float(in_range_retrieved / min(k, num_in_range_total)) if num_in_range_total > 0 else 0.0

            app_result[f'k{k}'] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'min': min_val,
                'max': max_val,
                'precision': precision,
                'recall': recall,
                'num_retrieved_in_range': int(in_range_retrieved),
                'values': retrieved_values.tolist(),
            }

            print(f"  K={k}: Precision={precision:.3f}, Recall@K={recall:.3f}, "
                  f"Mean={mean_val:.3f} {unit}")

        results['application_results'].append(app_result)

    # Step 4: Compute random baseline
    print("\n[Text-Only Inverse Design] Computing random baseline...")
    random_results = []

    for app_query in application_queries:
        app_name = app_query['name']
        expected_min, expected_max = app_query['expected_range']

        random_result = {'name': app_name}

        for k in cfg.k_values:
            random_indices = np.random.choice(len(all_property_values),
                                             size=min(k, len(all_property_values)),
                                             replace=False)
            random_values = all_property_values[random_indices]
            random_values = random_values[~pd.isna(random_values)]

            if len(random_values) > 0:
                in_range = ((random_values >= expected_min) & (random_values <= expected_max)).sum()
                precision = float(in_range / len(random_values))
            else:
                precision = 0.0

            random_result[f'k{k}_precision'] = precision

        random_results.append(random_result)

    results['random_baseline'] = random_results

    # Step 5: Save results
    use_untrained = getattr(cfg, 'use_untrained', False)
    ckpt_name = extract_checkpoint_name(cfg.model_ckpt, use_untrained=use_untrained)
    output_dir = Path('outputs/text_only_inverse_design') / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Text-Only Inverse Design] Saving results to {output_dir}/")

    results_file = output_dir / f'text_only_inverse_design_{cfg.property.replace(" ", "_")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Text-Only Inverse Design] Results saved to {results_file}")

    # Step 6: Print summary
    print("\n" + "="*80)
    print(f"TEXT-ONLY INVERSE DESIGN RESULTS - {cfg.property}")
    print("="*80)

    for app_result in results['application_results']:
        print(f"\nApplication: {app_result['name']}")
        print(f"Query: '{app_result['query']}'")
        print(f"Expected range: {app_result['expected_range'][0]} - {app_result['expected_range'][1]} {unit}")
        print(f"Materials in range: {app_result['num_materials_in_range']}")

        for k in cfg.k_values:
            if f'k{k}' in app_result:
                metrics = app_result[f'k{k}']
                print(f"  K={k:>3}: Precision={metrics['precision']:.3f}, "
                      f"Recall@K={metrics['recall']:.3f}, "
                      f"Mean={metrics['mean']:.2f} {unit}")

    print("="*80 + "\n")

    # Step 7: Visualize
    if cfg.plot:
        plot_text_only_inverse_design_results(results, cfg, output_dir)

    return results


def plot_text_only_inverse_design_results(results, cfg, output_dir):
    """Create visualization plots for text-only inverse design results."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    mpl.rcParams['figure.dpi'] = 300
    mpl.rc('font', family='Arial', size=6)
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    property_name = cfg.property
    unit = results['unit']
    k_values = results['k_values']
    app_results = results['application_results']
    all_property_values = np.array(results['all_property_values'])
    all_property_values = all_property_values[~pd.isna(all_property_values)]

    # Create figure: 2 rows (box plot + precision/recall)
    n_cols = len(k_values)
    fig = plt.figure(figsize=(1.4 * n_cols, 3))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1],
                         hspace=0.3, wspace=0.35,
                         left=0.10, right=0.98, top=0.95, bottom=0.08)

    # Color palette for applications - property-specific colormaps
    property_colormaps = {
        'band gap': 'viridis',
        'density': 'spring',
        'scintillation attenuation length': 'summer',
        'total magnetization': 'autumn',
        'energy above hull': 'winter',
        'formation energy per atom': 'viridis'
    }

    cmap_name = property_colormaps.get(property_name, 'viridis')
    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 0.9, len(app_results)))

    # Row 1: Box plots with expected ranges
    for col_idx, k in enumerate(k_values):
        ax = fig.add_subplot(gs[0, col_idx])

        # Collect data for each application
        app_names = []
        box_data = []
        expected_ranges = []

        for app_idx, app_result in enumerate(app_results):
            app_names.append(app_result['name'].replace('_', ' ').title())
            expected_ranges.append(app_result['expected_range'])

            if f'k{k}' in app_result and len(app_result[f'k{k}']['values']) > 0:
                box_data.append(app_result[f'k{k}']['values'])
            else:
                box_data.append([])

        # Create box plots
        positions = np.arange(len(app_names))
        bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(linewidth=0.8),
                       whiskerprops=dict(linewidth=0.8),
                       capprops=dict(linewidth=0.8),
                       medianprops=dict(linewidth=1.2, color='darkred'))

        # Color boxes with viridis
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Plot individual retrieved values as scatter points with jitter
        for pos, data, color in zip(positions, box_data, colors):
            if len(data) > 0:
                # Add jitter to x-coordinates for better visibility
                jitter = np.random.normal(0, 0.04, size=len(data))
                x_coords = np.ones(len(data)) * pos + jitter
                ax.scatter(x_coords, data, alpha=0.5, s=10, color=color, zorder=1)

        # Overlay expected ranges as horizontal bars
        for pos, (exp_min, exp_max), color in zip(positions, expected_ranges, colors):
            ax.plot([pos - 0.3, pos + 0.3], [exp_min, exp_min], color='black',
                   linewidth=1.5, linestyle='--', alpha=0.7)
            ax.plot([pos - 0.3, pos + 0.3], [exp_max, exp_max], color='black',
                   linewidth=1.5, linestyle='--', alpha=0.7)
            # Shaded expected range
            ax.fill_between([pos - 0.3, pos + 0.3], exp_min, exp_max,
                           color=color, alpha=0.15, zorder=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(app_names, rotation=30, ha='right', fontsize=5)
        ax.set_ylabel(f'{property_name.title()} ({unit})', fontsize=6)
        ax.set_title(f'K={k}', fontsize=7, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

        # Set spine linewidth
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3)

    # Row 2: Precision and Recall bar plots
    for col_idx, k in enumerate(k_values):
        ax = fig.add_subplot(gs[1, col_idx])

        precisions_clac = []
        precisions_random = []
        recalls = []

        for app_idx, app_result in enumerate(app_results):
            if f'k{k}' in app_result:
                precisions_clac.append(app_result[f'k{k}']['precision'])
                recalls.append(app_result[f'k{k}']['recall'])
            else:
                precisions_clac.append(0.0)
                recalls.append(0.0)

            # Random baseline
            random_result = results['random_baseline'][app_idx]
            precisions_random.append(random_result.get(f'k{k}_precision', 0.0))

        x = np.arange(len(app_results))
        width = 0.25

        # Plot bars
        bars1 = ax.bar(x - width, precisions_clac, width, label='Precision (CLaC)',
                      color='steelblue', alpha=0.8, linewidth=0.8, edgecolor='black')
        bars2 = ax.bar(x, precisions_random, width, label='Precision (Random)',
                      color='gray', alpha=0.5, linewidth=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, recalls, width, label='Recall@K (CLaC)',
                      color='coral', alpha=0.8, linewidth=0.8, edgecolor='black')

        ax.set_ylabel('Score', fontsize=6)
        ax.set_title(f'K={k}', fontsize=7, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r['name'].replace('_', ' ').title() for r in app_results],
                          rotation=45, ha='right', fontsize=5)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=5, frameon=True, edgecolor='black', fancybox=False,
                 loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

        # Set spine linewidth
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3)

    # Layout is handled by GridSpec parameters (left, right, top, bottom, wspace, hspace)
    # No need for tight_layout which causes warnings with GridSpec

    # Save plot as both PNG and SVG
    base_name = f'text_only_inverse_design_{property_name.replace(" ", "_")}'

    # Save PNG
    plot_file_png = output_dir / f'{base_name}.png'
    try:
        plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
        print(f"[Text-Only Inverse Design] PNG saved to {plot_file_png}")
    except Exception as e:
        print(f"[Plot] ERROR saving PNG: {e}")

    # Save SVG
    plot_file = output_dir / f'{base_name}.svg'
    try:
        plt.savefig(plot_file, bbox_inches='tight')
        print(f"[Text-Only Inverse Design] SVG saved to {plot_file}")
    except Exception as e:
        print(f"[Plot] ERROR saving SVG: {e}")
        plot_file_png = output_dir / f'{base_name}.png'
        try:
            plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
            print(f"[Plot] Saved as PNG instead: {plot_file_png}")
        except Exception as e2:
            print(f"[Plot] ERROR saving PNG: {e2}")

    plt.close()


def main(cfg):
    print(cfg)

    # Text-only inverse design evaluation (NEW)
    if cfg.evaluation_method == 'text-only-inverse-design':
        evaluate_text_only_inverse_design(cfg)
        return

    # Inverse design evaluation
    if cfg.evaluation_method == 'inverse-design':
        evaluate_inverse_design(cfg)
        return

    # Model loading (load first for auto-detection)
    if cfg.use_untrained:
        # Load untrained (randomly initialized) model from config
        print(f"[{cfg.evaluation_method}] Loading untrained model from config: {cfg.config}")
        config_dir = Path(cfg.config).parent.parent.absolute()
        config_name = Path(cfg.config).stem

        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            hydra_cfg = compose(config_name="config", overrides=[f"model={config_name}", "debug=true"])
            model = hydra.utils.instantiate(hydra_cfg.model, _recursive_=False)

        print(f"[{cfg.evaluation_method}] Initialized untrained model")
    else:
        # Load trained model from checkpoint
        model = CLaCLite.load_from_checkpoint(cfg.model_ckpt, map_location={'cuda:0': 'cpu'}, strict=False)
        print(f"[{cfg.evaluation_method}] Loaded trained model from: {cfg.model_ckpt}")

    model = model.to(cfg.device)
    model.eval()
    model.text_encoder.config.output_hidden_states = True

    # Auto-detect graphdatatype from model if needed
    if cfg.graphdatatype == 'auto':
        if hasattr(model.graph_encoder, 'orb_model'):
            detected_graphdatatype = 'orb'
            print(f"[{cfg.evaluation_method}] Auto-detected graph encoder: ORB")
        else:
            detected_graphdatatype = 'torch_geometric'
            print(f"[{cfg.evaluation_method}] Auto-detected graph encoder: torch_geometric")
    else:
        detected_graphdatatype = cfg.graphdatatype
        print(f"[{cfg.evaluation_method}] Using specified graph encoder: {detected_graphdatatype}")

    # Get tokenizer model name from checkpoint (use model's actual tokenizer)
    # This prevents tokenizer mismatch issues (e.g., MatSciBERT vs Galactica)
    # CLaCTokenizer wraps the actual HuggingFace tokenizer in model.tokenizer.tokenizer
    if hasattr(model, 'tokenizer'):
        # Try to get tokenizer name from wrapped tokenizer (CLaCTokenizer)
        if hasattr(model.tokenizer, 'tokenizer') and hasattr(model.tokenizer.tokenizer, 'name_or_path'):
            tokenizer_model = model.tokenizer.tokenizer.name_or_path
            print(f"[{cfg.evaluation_method}] Using model's tokenizer: {tokenizer_model}")
        # Fallback: try direct name_or_path (for backward compatibility)
        elif hasattr(model.tokenizer, 'name_or_path'):
            tokenizer_model = model.tokenizer.name_or_path
            print(f"[{cfg.evaluation_method}] Using model's tokenizer: {tokenizer_model}")
        else:
            tokenizer_model = cfg.llm
            print(f"[{cfg.evaluation_method}] WARNING: Could not detect tokenizer from model, using config: {tokenizer_model}")
    else:
        tokenizer_model = cfg.llm
        print(f"[{cfg.evaluation_method}] Using specified tokenizer: {tokenizer_model}")

    # DataModule loading (after auto-detection)
    if cfg.evaluation_method == 'zero-shot QA':
        dm = QuestionEvaluationDataModule(
            data_path=cfg.data_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            tokenizer_model=tokenizer_model,  # Use detected tokenizer
            debug=cfg.debug,
            label=cfg.label,
            graphdatatype=detected_graphdatatype  # Use detected graph type
        )
    elif cfg.evaluation_method == 'zero-shot retrieval':
        dm = CLaCDataModule(
            data_path=cfg.data_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            graphdatatype=detected_graphdatatype,  # Use detected type
            tokenizer_model=tokenizer_model,  # Use detected tokenizer
            debug=cfg.debug,
            label=cfg.label
        )
    dm.setup()

    # Get test dataloader
    test_dataloader = dm.test_dataloader()

    # Load raw dataset to get original text (bypass DataModule's tokenization)
    data_path_dict = {'test': cfg.data_path + '_test.parquet'}
    raw_dataset = load_dataset('parquet', data_files=data_path_dict)['test']
    print(f"[{cfg.evaluation_method}] Loaded {len(raw_dataset)} samples from dataset")
    print(f"[{cfg.evaluation_method}] Will re-tokenize with model's tokenizer to avoid mismatch")

    # Evaluation
    g_stack = []
    t_stack = []

    # For zero-shot QA: store per-sample binary outcomes for bootstrapping
    qa_outcomes = []  # Binary outcomes for each sample (1=correct, 0=incorrect)
    qa_num_choices = []  # Number of choices for each QA sample (for random baseline)

    # For zero-shot retrieval: store per-sample binary outcomes for bootstrapping
    top1_outcomes = []  # Binary outcomes for each sample (1=correct, 0=incorrect)
    top3_outcomes = []
    top10_outcomes = []

    for batch_i, batch in tqdm(enumerate(test_dataloader)):
        graphs, _ = batch  # Ignore DataModule's tokenized questions

        # Check if graphs is None (data loading issue)
        if graphs is None:
            print(f"\nWARNING: Batch {batch_i} has None graphs. Skipping...")
            print(f"This usually means graphdatatype mismatch or data loading issue.")
            print(f"Current graphdatatype: {detected_graphdatatype}")
            print(f"Model graph encoder type: {type(model.graph_encoder).__name__}")
            continue

        graphs = graphs.to(model.device)

        # Get raw text from dataset and re-tokenize with model's tokenizer
        start_idx = batch_i * cfg.batch_size
        end_idx = min(start_idx + cfg.batch_size, len(raw_dataset))
        batch_texts = [raw_dataset[i][cfg.label] for i in range(start_idx, end_idx)]

        # Handle different data formats for QA vs retrieval
        if cfg.evaluation_method == 'zero-shot QA':
            # For QA: batch_texts is a list of lists [[q1, q2, q3, ...], [q1, q2, q3, ...], ...]
            # Flatten to a single list and keep track of sizes
            flattened_texts = []
            question_counts = []
            for questions_list in batch_texts:
                flattened_texts.extend(questions_list)
                question_counts.append(len(questions_list))
        elif cfg.evaluation_method == 'zero-shot retrieval':
            # For retrieval: batch_texts is a list of strings ["text1", "text2", ...]
            flattened_texts = batch_texts
            question_counts = [1] * len(batch_texts)  # Each sample has 1 text

        # Re-tokenize with model's own tokenizer (prevents vocab mismatch)
        questions = model.tokenizer(
            flattened_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        questions = {k: v.to(model.device) for k, v in questions.items()}

        # decode batch to text
        with torch.no_grad():
            g_feat = model.graph_encoder(graphs)  # (batch, hidden)
            # Use the unified text encoding method from the model.
            # It returns (loss, features); we only need the features for inference.
            _, t_feat = model.encode_text(questions)  # (total_questions, hidden)

            # Process each sample separately
            if cfg.evaluation_method == 'zero-shot QA':
                t_feat_start = 0
                for sample_idx, num_questions in enumerate(question_counts):
                    # Get text features for this sample's questions
                    t_feat_sample = t_feat[t_feat_start:t_feat_start + num_questions]  # (num_questions, hidden)
                    g_feat_sample = g_feat[sample_idx:sample_idx+1]  # (1, hidden)

                    # Compute similarity between this graph and all its question choices
                    similarity, _, _ = model.loss.global_d(g_feat_sample, t_feat_sample)

                    # measure accuracy. the prediction is the largest similarity and the answer is always the first one
                    prediction = torch.argmax(similarity, dim=-1)
                    is_correct = int(prediction == 0)
                    qa_outcomes.append(is_correct)
                    qa_num_choices.append(num_questions)  # Store number of choices for random baseline

                    t_feat_start += num_questions
            elif cfg.evaluation_method == 'zero-shot retrieval':
                similarity, image_out, text_out = model.loss.global_d(g_feat, t_feat)
                g_stack.append(image_out.cpu())
                t_stack.append(text_out.cpu())
                if batch_i % 8 == 7:
                    g_stack = torch.concat(g_stack, dim=0)
                    t_stack = torch.concat(t_stack, dim=0)
                    gt_logits = torch.matmul(t_stack, g_stack.transpose(0, 1)) * model.loss.global_d.temperature.exp().cpu()

                    self_mask = torch.eye(gt_logits.shape[0], device=gt_logits.device, dtype=torch.bool)
                    comb_sim = torch.cat([gt_logits[self_mask][:,None], gt_logits.masked_fill(self_mask, -torch.inf)], dim=-1)
                    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

                    # Store per-sample binary outcomes for bootstrapping
                    top1_outcomes.extend((sim_argsort == 0).int().tolist())
                    top3_outcomes.extend((sim_argsort < 3).int().tolist())
                    top10_outcomes.extend((sim_argsort < 10).int().tolist())

                    g_stack = []
                    t_stack = []

    if cfg.evaluation_method == 'zero-shot QA':
        # Compute random baseline
        print(f"\n[Random Baseline] Computing random QA performance...")
        # For QA: random choice means 1/N probability where N is number of choices
        # Generate random outcomes based on number of choices
        np.random.seed(42)
        random_qa_outcomes = []
        for num_choices in qa_num_choices:
            # Random guess: 1/num_choices probability of being correct
            random_correct = np.random.binomial(1, 1.0 / num_choices)
            random_qa_outcomes.append(random_correct)

        # Compute bootstrapped confidence intervals for CLaC
        print(f"\n[Bootstrapping] Computing confidence intervals (n_bootstrap=1000)...")
        qa_stats = bootstrap_confidence_interval(qa_outcomes, n_bootstrap=1000)

        # Compute bootstrapped confidence intervals for Random baseline
        random_qa_stats = bootstrap_confidence_interval(random_qa_outcomes, n_bootstrap=1000)

        # Create results dictionary with bootstrapped statistics
        results = {
            'evaluation_method': 'zero-shot QA',
            'label': cfg.label,
            'model_checkpoint': cfg.model_ckpt,
            'num_samples': len(qa_outcomes),
            'batch_size': cfg.batch_size,
            'bootstrap_params': {
                'n_bootstrap': 1000,
                'confidence_level': 0.95,
                'random_seed': 42
            },
            'accuracy': {
                'mean': qa_stats['mean'],
                'std': qa_stats['std'],
                'ci_lower': qa_stats['ci_lower'],
                'ci_upper': qa_stats['ci_upper']
            },
            'random_baseline': {
                'mean': random_qa_stats['mean'],
                'std': random_qa_stats['std'],
                'ci_lower': random_qa_stats['ci_lower'],
                'ci_upper': random_qa_stats['ci_upper']
            }
        }

        # Save results to JSON
        ckpt_name = extract_checkpoint_name(cfg.model_ckpt, use_untrained=cfg.use_untrained)
        output_dir = Path('outputs/zero_shot_qa') / ckpt_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f'qa_{cfg.label}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary with confidence intervals and baseline comparison
        print("\n" + "="*80)
        print(f"ZERO-SHOT QA RESULTS - Label: {cfg.label}")
        print("="*80)
        print(f"Number of test samples: {len(qa_outcomes)}")
        print(f"Bootstrap iterations: {qa_stats['n_bootstrap']}")
        print("-" * 80)
        print("CLaC Model:")
        print(f"  Accuracy: {qa_stats['mean']:.4f} ± {qa_stats['std']:.4f}")
        print(f"            95% CI: [{qa_stats['ci_lower']:.4f}, {qa_stats['ci_upper']:.4f}]")
        print(f"            ({qa_stats['mean']*100:.2f}% ± {qa_stats['std']*100:.2f}%)")
        print(f"  Correct:  {sum(qa_outcomes)} / {len(qa_outcomes)}")
        print("-" * 80)
        print("Random Baseline:")
        print(f"  Accuracy: {random_qa_stats['mean']:.4f} ± {random_qa_stats['std']:.4f}")
        print(f"            95% CI: [{random_qa_stats['ci_lower']:.4f}, {random_qa_stats['ci_upper']:.4f}]")
        print(f"            ({random_qa_stats['mean']*100:.2f}% ± {random_qa_stats['std']*100:.2f}%)")
        print(f"  Correct:  {sum(random_qa_outcomes)} / {len(random_qa_outcomes)}")
        print("-" * 80)
        print(f"Improvement over Random: {(qa_stats['mean'] - random_qa_stats['mean'])*100:.2f}% absolute")
        print("-" * 80)
        print(f"Results saved to: {results_file}")
        print("="*80 + "\n")
    elif cfg.evaluation_method == 'zero-shot retrieval':
        # Compute random baseline
        print(f"\n[Random Baseline] Computing random retrieval performance...")
        n_samples = len(top1_outcomes)

        # For retrieval task: random ranking means uniform probability
        # Top-k accuracy = k / n_samples (for large n_samples)
        # Generate random outcomes for each sample
        np.random.seed(42)
        random_top1_outcomes = np.random.binomial(1, 1.0 / n_samples, n_samples).tolist()
        random_top3_outcomes = np.random.binomial(1, min(3.0 / n_samples, 1.0), n_samples).tolist()
        random_top10_outcomes = np.random.binomial(1, min(10.0 / n_samples, 1.0), n_samples).tolist()

        # Compute bootstrapped confidence intervals for CLaC
        print(f"\n[Bootstrapping] Computing confidence intervals (n_bootstrap=1000)...")
        top1_stats = bootstrap_confidence_interval(top1_outcomes, n_bootstrap=1000)
        top3_stats = bootstrap_confidence_interval(top3_outcomes, n_bootstrap=1000)
        top10_stats = bootstrap_confidence_interval(top10_outcomes, n_bootstrap=1000)

        # Compute bootstrapped confidence intervals for Random baseline
        random_top1_stats = bootstrap_confidence_interval(random_top1_outcomes, n_bootstrap=1000)
        random_top3_stats = bootstrap_confidence_interval(random_top3_outcomes, n_bootstrap=1000)
        random_top10_stats = bootstrap_confidence_interval(random_top10_outcomes, n_bootstrap=1000)

        # Create results dictionary with bootstrapped statistics
        results = {
            'evaluation_method': 'zero-shot retrieval',
            'label': cfg.label,
            'model_checkpoint': cfg.model_ckpt,
            'num_samples': len(top1_outcomes),
            'batch_size': cfg.batch_size,
            'bootstrap_params': {
                'n_bootstrap': 1000,
                'confidence_level': 0.95,
                'random_seed': 42
            },
            'overall_metrics': {
                'top1_accuracy': {
                    'mean': top1_stats['mean'],
                    'std': top1_stats['std'],
                    'ci_lower': top1_stats['ci_lower'],
                    'ci_upper': top1_stats['ci_upper']
                },
                'top3_accuracy': {
                    'mean': top3_stats['mean'],
                    'std': top3_stats['std'],
                    'ci_lower': top3_stats['ci_lower'],
                    'ci_upper': top3_stats['ci_upper']
                },
                'top10_accuracy': {
                    'mean': top10_stats['mean'],
                    'std': top10_stats['std'],
                    'ci_lower': top10_stats['ci_lower'],
                    'ci_upper': top10_stats['ci_upper']
                }
            },
            'random_baseline': {
                'top1_accuracy': {
                    'mean': random_top1_stats['mean'],
                    'std': random_top1_stats['std'],
                    'ci_lower': random_top1_stats['ci_lower'],
                    'ci_upper': random_top1_stats['ci_upper']
                },
                'top3_accuracy': {
                    'mean': random_top3_stats['mean'],
                    'std': random_top3_stats['std'],
                    'ci_lower': random_top3_stats['ci_lower'],
                    'ci_upper': random_top3_stats['ci_upper']
                },
                'top10_accuracy': {
                    'mean': random_top10_stats['mean'],
                    'std': random_top10_stats['std'],
                    'ci_lower': random_top10_stats['ci_lower'],
                    'ci_upper': random_top10_stats['ci_upper']
                }
            }
        }

        # Save results to JSON
        ckpt_name = extract_checkpoint_name(cfg.model_ckpt, use_untrained=cfg.use_untrained)
        output_dir = Path('outputs/zero_shot_retrieval') / ckpt_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f'retrieval_{cfg.label}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary with confidence intervals and baseline comparison
        print("\n" + "="*80)
        print(f"ZERO-SHOT RETRIEVAL RESULTS - Label: {cfg.label}")
        print("="*80)
        print(f"Number of test samples: {len(top1_outcomes)}")
        print(f"Bootstrap iterations: {top1_stats['n_bootstrap']}")
        print("-" * 80)
        print("CLaC Model:")
        print(f"  Top-1 Accuracy:  {top1_stats['mean']:.4f} ± {top1_stats['std']:.4f}")
        print(f"                   95% CI: [{top1_stats['ci_lower']:.4f}, {top1_stats['ci_upper']:.4f}]")
        print(f"                   ({top1_stats['mean']*100:.2f}% ± {top1_stats['std']*100:.2f}%)")
        print()
        print(f"  Top-3 Accuracy:  {top3_stats['mean']:.4f} ± {top3_stats['std']:.4f}")
        print(f"                   95% CI: [{top3_stats['ci_lower']:.4f}, {top3_stats['ci_upper']:.4f}]")
        print(f"                   ({top3_stats['mean']*100:.2f}% ± {top3_stats['std']*100:.2f}%)")
        print()
        print(f"  Top-10 Accuracy: {top10_stats['mean']:.4f} ± {top10_stats['std']:.4f}")
        print(f"                   95% CI: [{top10_stats['ci_lower']:.4f}, {top10_stats['ci_upper']:.4f}]")
        print(f"                   ({top10_stats['mean']*100:.2f}% ± {top10_stats['std']*100:.2f}%)")
        print("-" * 80)
        print("Random Baseline:")
        print(f"  Top-1 Accuracy:  {random_top1_stats['mean']:.4f} ± {random_top1_stats['std']:.4f}")
        print(f"                   95% CI: [{random_top1_stats['ci_lower']:.4f}, {random_top1_stats['ci_upper']:.4f}]")
        print(f"                   ({random_top1_stats['mean']*100:.2f}% ± {random_top1_stats['std']*100:.2f}%)")
        print()
        print(f"  Top-3 Accuracy:  {random_top3_stats['mean']:.4f} ± {random_top3_stats['std']:.4f}")
        print(f"                   95% CI: [{random_top3_stats['ci_lower']:.4f}, {random_top3_stats['ci_upper']:.4f}]")
        print(f"                   ({random_top3_stats['mean']*100:.2f}% ± {random_top3_stats['std']*100:.2f}%)")
        print()
        print(f"  Top-10 Accuracy: {random_top10_stats['mean']:.4f} ± {random_top10_stats['std']:.4f}")
        print(f"                   95% CI: [{random_top10_stats['ci_lower']:.4f}, {random_top10_stats['ci_upper']:.4f}]")
        print(f"                   ({random_top10_stats['mean']*100:.2f}% ± {random_top10_stats['std']*100:.2f}%)")
        print("-" * 80)
        print(f"Improvement over Random:")
        print(f"  Top-1:  {(top1_stats['mean'] - random_top1_stats['mean'])*100:.2f}% absolute")
        print(f"  Top-3:  {(top3_stats['mean'] - random_top3_stats['mean'])*100:.2f}% absolute")
        print(f"  Top-10: {(top10_stats['mean'] - random_top10_stats['mean'])*100:.2f}% absolute")
        print("-" * 80)
        print(f"Results saved to: {results_file}")
        print("="*80 + "\n")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='datafiles/mp_3d_2020_materials_graphs_gpt_questions')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for encoding')
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--llm', type=str, default='facebook/galactica-125m')
    parser.add_argument('--graphdatatype', type=str, default='auto', choices=['orb', 'torch_geometric', 'auto'])
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--label', type=str, default='structure_question_list', choices=['text', 'composition_question_list', 'structure_question_list', 'oxide_question_list'])
    parser.add_argument('--model-ckpt', type=str, default='outputs/your-model.ckpt')
    parser.add_argument('--use-untrained', action='store_true',
                       help='Use untrained (randomly initialized) model as baseline')
    parser.add_argument('--config', type=str, default='config/model/clac-lite.yaml',
                       help='Config file for untrained model (only used with --use-untrained)')

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--evaluation-method', type=str, default='zero-shot QA',
                       choices=['zero-shot QA', 'zero-shot retrieval', 'few-shot QA', 'few-shot retrieval',
                               'inverse-design', 'text-only-inverse-design'])

    # Inverse design specific arguments
    parser.add_argument('--property', type=str, default='band gap',
                       choices=['band gap', 'formation energy per atom', 'density',
                               'scintillation attenuation length', 'total magnetization', 'energy above hull'],
                       help='Property for inverse design evaluation')
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 50, 100],
                       help='K values for top-K retrieval (e.g., --k-values 10 50 100)')
    parser.add_argument('--output-dir', type=str, default='outputs/inverse_design',
                       help='Directory to save inverse design results')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate visualization plots')

    cfg = parser.parse_args()
    main(cfg)