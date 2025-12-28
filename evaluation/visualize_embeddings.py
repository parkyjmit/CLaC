"""
UMAP Visualization of CLaC Embeddings

This script visualizes the learned embedding space using UMAP dimensionality reduction.
It supports coloring by:
- Material properties (band gap, formation energy, etc.)
- Text keywords (presence/absence of specific terms)
- Modality (graph vs text)
"""

import sys
from pathlib import Path
# Add parent directory to path to import data and model modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
import umap
import json

from model.clac import CLaCLite
from data.datamodule import CLaCDataModule

mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Arial', size=5)
# Line widths (reduced to half of defaults)
mpl.rcParams['axes.linewidth'] = 0.4      # axis border (default: 0.8)
mpl.rcParams['lines.linewidth'] = 0.75    # line plots (default: 1.5)
mpl.rcParams['patch.linewidth'] = 0.5     # patch edges (default: 1.0)
mpl.rcParams['xtick.major.width'] = 0.4   # x tick width (default: 0.8)
mpl.rcParams['ytick.major.width'] = 0.4   # y tick width (default: 0.8)
mpl.rcParams['xtick.minor.width'] = 0.3   # x minor tick (default: 0.6)
mpl.rcParams['ytick.minor.width'] = 0.3   # y minor tick (default: 0.6)
# For SVG
mpl.rcParams["svg.fonttype"] = "none"
# For PDF
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42

def save_embeddings_cache(cache_path, embeddings_dict):
    """Save embeddings to cache file."""
    np.savez_compressed(cache_path, **embeddings_dict)
    print(f"✓ Saved embeddings cache to {cache_path}")


def load_embeddings_cache(cache_path):
    """Load embeddings from cache file if it exists."""
    if not cache_path.exists():
        return None

    try:
        print(f"Loading embeddings from cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=False)

        # Convert to dict
        embeddings_dict = {key: cache[key] for key in cache.files}
        print(f"✓ Loaded embeddings from cache (keys: {list(embeddings_dict.keys())})")
        return embeddings_dict

    except Exception as e:
        print(f"⚠ Failed to load cache: {e}")
        return None


def extract_embeddings(model, dataloader, device, embed_type='both', use_projection=True):
    """
    Extract embeddings from the model.

    Args:
        model: Trained CLaC model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        embed_type: 'graph', 'text', or 'both'
        use_projection: If True, use loss.global_d projection blocks to map to common space

    Returns:
        Dictionary with embeddings and metadata
    """
    model.eval()

    graph_embeddings = []
    text_embeddings = []

    # Check if model has global_d for projection
    has_global_d = hasattr(model, 'loss') and hasattr(model.loss, 'global_d')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {embed_type} embeddings"):
            graphs, texts = batch
            graphs = graphs.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}

            if embed_type in ['graph', 'both']:
                # Extract graph embeddings
                g_feat = model.graph_encoder(graphs)

                # Use global_d projection to map to common space if available
                if use_projection and has_global_d:
                    g_feat = model.loss.global_d.img_block(g_feat)
                    g_feat = F.normalize(g_feat, dim=-1)
                else:
                    g_feat = F.normalize(g_feat, dim=-1)

                graph_embeddings.append(g_feat.cpu().numpy())

            if embed_type in ['text', 'both']:
                # Extract text embeddings using the model's encode_text method
                _, t_feat = model.encode_text(texts)

                # Use global_d projection to map to common space if available
                if use_projection and has_global_d:
                    t_feat = model.loss.global_d.text_block(t_feat)
                    t_feat = F.normalize(t_feat, dim=-1)
                else:
                    t_feat = F.normalize(t_feat, dim=-1)

                text_embeddings.append(t_feat.cpu().numpy())

    result = {}
    if embed_type in ['graph', 'both']:
        result['graph'] = np.concatenate(graph_embeddings, axis=0)
    if embed_type in ['text', 'both']:
        result['text'] = np.concatenate(text_embeddings, axis=0)

    return result


def load_metadata(data_path, split='test'):
    """Load dataset metadata (properties, text)."""
    data_path_dict = {split: f"{data_path}_{split}.parquet"}
    dataset = load_dataset('parquet', data_files=data_path_dict)[split]
    return dataset


def get_property_values(dataset, property_name):
    """
    Extract property values from dataset.

    Supported properties:
    - bandgap / band_gap
    - formation_energy_per_atom
    - density
    - scintillation_attenuation_length
    - total_magnetization
    - energy_above_hull
    """
    # Normalize property name
    property_map = {
        'bandgap': 'bandgap',
        'band_gap': 'bandgap',
        'formation_energy': 'formation_energy_per_atom',
        'density': 'density',
        'scintillation': 'scintillation_attenuation_length',
        'magnetization': 'total_magnetization',
        'energy_above_hull': 'energy_above_hull',
        'ehull': 'energy_above_hull',
    }

    normalized_name = property_map.get(property_name.lower(), property_name)

    if normalized_name not in dataset.column_names:
        raise ValueError(f"Property '{property_name}' not found. Available: {dataset.column_names}")

    return np.array(dataset[normalized_name])


def check_keyword_presence(dataset, keywords, text_field='gpt_text'):
    """
    Check if keywords are present in text field.

    Args:
        dataset: HuggingFace dataset
        keywords: List of keywords to search for (case-insensitive)
        text_field: Field name to search in

    Returns:
        Boolean array indicating presence of any keyword
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    keywords_lower = [k.lower() for k in keywords]

    presence = []
    for sample in dataset:
        text = sample.get(text_field, '') or ''
        if not isinstance(text, str):
            text = str(text)
        text_lower = text.lower()
        has_keyword = any(kw in text_lower for kw in keywords_lower)
        presence.append(has_keyword)

    return np.array(presence)


def compute_text_similarity(model, text_query, embeddings, device, datamodule, use_projection=True):
    """
    Compute cosine similarity between text query and embeddings.

    Args:
        model: CLaC model with text encoder
        text_query: Text query string
        embeddings: (N, D) array of embeddings to compare against
        device: Device to run on
        datamodule: CLaCDataModule instance (for tokenizer)
        use_projection: Whether to use projection block

    Returns:
        (N,) array of cosine similarities
    """
    model.eval()

    # Use DataModule's tokenizer directly
    tokenized = datamodule.token_fn(text_query)
    # Convert to tensors and add batch dimension if needed
    inputs = {}
    for k, v in tokenized.items():
        if isinstance(v, list):
            v = torch.tensor(v)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        inputs[k] = v.to(device)

    # Encode query
    with torch.no_grad():
        _, query_feat = model.encode_text(inputs)

        # Apply projection if needed
        has_global_d = hasattr(model, 'loss') and hasattr(model.loss, 'global_d')
        if use_projection and has_global_d:
            query_feat = model.loss.global_d.text_block(query_feat)

        query_feat = F.normalize(query_feat, dim=-1)
        query_feat = query_feat.cpu().numpy()

    # Compute cosine similarity
    # embeddings is (N, D), query_feat is (1, D)
    similarities = np.dot(embeddings, query_feat.T).squeeze()

    return similarities


def apply_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """Apply UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2
    )
    return reducer.fit_transform(embeddings)


def plot_embeddings(coords, colors, labels=None, title='UMAP Embedding Visualization',
                   colorbar_label='Value', save_path=None, cmap='viridis',
                   discrete=False, alpha=0.6, figsize=(1.44, 1.1), palette=None,
                   vmin=None, vmax=None, symmetric_colormap=False):
    """
    Plot UMAP coordinates with colors.

    Args:
        coords: (N, 2) array of UMAP coordinates
        colors: (N,) array of color values
        labels: Optional list of legend labels (for discrete colors)
        title: Plot title (not used)
        colorbar_label: Label for colorbar
        save_path: Path to save figure
        cmap: Colormap name (for continuous values)
        discrete: Whether colors are discrete categories
        alpha: Point transparency
        figsize: Figure size
        palette: Color palette for discrete categories (list of colors)
        vmin: Minimum value for colormap normalization (continuous only)
        vmax: Maximum value for colormap normalization (continuous only)
        symmetric_colormap: If True and vmin/vmax not specified, use symmetric range (text-similarity only)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if discrete:
        # For discrete categories
        unique_values = np.unique(colors)

        # Use palette if provided, otherwise use default matplotlib colors
        if palette is None:
            palette = plt.cm.tab10.colors  # default palette

        for i, val in enumerate(unique_values):
            mask = colors == val
            label = labels[i] if labels else f'Class {val}'
            color = palette[i % len(palette)]  # cycle through palette if needed
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      label=label, alpha=alpha, s=0.3, edgecolors='none', color=color)
        ax.legend(markerscale=2, frameon=True, fancybox=True)
    else:
        # For continuous values
        # If symmetric_colormap is True and vmin/vmax not specified, use symmetric range based on max absolute value
        if symmetric_colormap and (vmin is None or vmax is None):
            max_abs = max(abs(colors.min()), abs(colors.max()))
            vmin = -max_abs if vmin is None else vmin
            vmax = max_abs if vmax is None else vmax

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                           c=colors, cmap=cmap, alpha=alpha,
                           s=0.3, edgecolors='none', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(colorbar_label, rotation=270, labelpad=5, fontsize=5)

    # Remove axis labels, ticks, and grid
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CLaC embeddings with UMAP')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (if not provided, uses randomly initialized model)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (required if --checkpoint is not provided)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset (without _test.parquet)')
    parser.add_argument('--output-dir', type=str, default='outputs/umap_visualizations', help='Output directory')
    parser.add_argument('--embed-type', type=str, choices=['graph', 'text', 'both'], default='both',
                       help='Which embeddings to visualize')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to use')

    # Coloring options
    parser.add_argument('--color-by', type=str, default='modality',
                       choices=['modality', 'property', 'keyword', 'text-similarity'],
                       help='What to use for coloring points')
    parser.add_argument('--property-name', type=str, default='bandgap',
                       help='Property name if --color-by=property (e.g., bandgap, formation_energy, density)')
    parser.add_argument('--keywords', type=str, nargs='+', default=None,
                       help='Keywords to search for if --color-by=keyword (e.g., "magnetic" "semiconductor")')
    parser.add_argument('--text-query', type=str, default=None,
                       help='Text query if --color-by=text-similarity (e.g., "magnetic semiconductor with high conductivity")')

    # UMAP parameters
    parser.add_argument('--n-neighbors', type=int, default=50, help='UMAP n_neighbors')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist')
    parser.add_argument('--metric', type=str, default='cosine', help='UMAP metric')

    # Visualization parameters
    parser.add_argument('--alpha', type=float, default=0.6, help='Point transparency')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for continuous values')
    parser.add_argument('--palette', type=str, nargs='+', default=None,
                       help='Color palette for discrete categories (e.g., "#FF0000" "#0000FF" or "red" "blue")')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for colormap (text-similarity mode)')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for colormap (text-similarity mode)')

    # Embedding extraction options
    parser.add_argument('--use-raw-embeddings', action='store_true',
                       help='Use raw encoder outputs instead of projected embeddings (may have dimension mismatch)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')

    # Caching options
    parser.add_argument('--no-cache', action='store_true',
                       help='Force recompute embeddings even if cache exists')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    print(f"Random seed set to: {args.random_seed}")

    # Validate arguments
    if args.checkpoint is None and args.config is None:
        parser.error("Either --checkpoint or --config must be provided")

    # Load model
    if args.checkpoint is not None:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = CLaCLite.load_from_checkpoint(args.checkpoint, strict=False)
        tokenizer_model = model.hparams.datamodule.tokenizer_model
    else:
        print(f"Initializing model from config: {args.config}")

        # Use Hydra to compose the full config
        from hydra import compose, initialize_config_dir

        # Get absolute path to config directory
        config_file = Path(args.config)
        if not config_file.is_absolute():
            config_file = Path.cwd() / config_file

        config_dir = str(config_file.parent.absolute())
        config_name = config_file.stem

        # Initialize Hydra
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name=config_name)

        # Now cfg.model should exist
        print(f"Loaded config: {config_name}")

        # Create model with parameters from composed config
        model_params = {
            'graph_encoder': cfg.model.graph_encoder,
            'text_augmentation': cfg.model.text_augmentation,
            'loss': cfg.model.loss,
            'optimizer': cfg.model.optimizer,
            'scheduler': cfg.model.scheduler,
            'datamodule': cfg.model.datamodule,  # datamodule is under model
            'augmentation': cfg.model.get('augmentation', True),
        }

        # Add hyperparameters (these are already in the model config via interpolation)
        if 'use_visual_intramodal_loss' in cfg.model:
            model_params['use_visual_intramodal_loss'] = cfg.model.use_visual_intramodal_loss
        if 'use_textual_intramodal_loss' in cfg.model:
            model_params['use_textual_intramodal_loss'] = cfg.model.use_textual_intramodal_loss

        # Add optional model-level parameters
        if 'lm_weight' in cfg.model:
            model_params['lm_weight'] = cfg.model.lm_weight

        # Create model
        model = CLaCLite(**model_params)
        tokenizer_model = cfg.model.datamodule.tokenizer_model  # Also under model

        print("WARNING: Using randomly initialized model (untrained)")

    model = model.to(args.device)
    model.eval()

    # Determine model name from checkpoint path and create output directory
    base_output_dir = Path(args.output_dir)
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        # Extract model name from checkpoint parent directory
        # e.g., outputs/my-model/checkpoint.ckpt -> my-model
        model_name = checkpoint_path.parent.name
    else:
        model_name = "untrained"

    output_dir = base_output_dir / model_name
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # Detect graph encoder type
    encoder_type = type(model.graph_encoder).__name__
    if 'ORB' in encoder_type or 'Orb' in encoder_type:
        graphdatatype = 'orb'
    else:
        graphdatatype = 'torch_geometric'
    print(f"Detected graph encoder: {encoder_type} -> graphdatatype={graphdatatype}")

    # Tokenizer
    print(f"Using tokenizer: {tokenizer_model}")

    # Create datamodule
    dm = CLaCDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        graphdatatype=graphdatatype,
        tokenizer_model=tokenizer_model,
        debug=False,
    )
    dm.setup()

    # Get dataloader
    if args.split == 'train':
        dataloader = dm.train_dataloader()
    elif args.split == 'val':
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.test_dataloader()

    # Determine cache file path
    use_projection = not args.use_raw_embeddings
    projection_str = "projected" if use_projection else "raw"
    cache_filename = f"embeddings_{args.split}_{args.embed_type}_{projection_str}.npz"
    cache_path = output_dir / cache_filename

    # Try to load from cache
    embeddings_dict = None
    if not args.no_cache:
        embeddings_dict = load_embeddings_cache(cache_path)

    # Extract embeddings if not cached
    if embeddings_dict is None:
        print(f"Extracting embeddings from {args.split} split...")
        if use_projection:
            print("Using projected embeddings (global_d projection blocks) for common space")
        else:
            print("Using raw encoder outputs (may have dimension mismatch for 'both' mode)")

        embeddings_dict = extract_embeddings(model, dataloader, args.device, args.embed_type, use_projection=use_projection)

        # Save to cache
        save_embeddings_cache(cache_path, embeddings_dict)

    # Load metadata
    print("Loading metadata...")
    dataset = load_metadata(args.data_path, args.split)
    n_samples = len(dataset)

    # Determine what to visualize
    if args.embed_type == 'both':
        # Combine graph and text embeddings (now in same dimension thanks to projection)
        graph_emb = embeddings_dict['graph']
        text_emb = embeddings_dict['text']
        print(f"Graph embeddings shape: {graph_emb.shape}, Text embeddings shape: {text_emb.shape}")

        embeddings = np.concatenate([graph_emb, text_emb], axis=0)
        modality_labels = np.array(['Graph'] * n_samples + ['Text'] * n_samples)
        dataset_doubled = dataset  # We'll use the same metadata for both
    elif args.embed_type == 'graph':
        embeddings = embeddings_dict['graph']
        modality_labels = np.array(['Graph'] * n_samples)
        dataset_doubled = dataset
    else:  # text
        embeddings = embeddings_dict['text']
        modality_labels = np.array(['Text'] * n_samples)
        dataset_doubled = dataset

    print(f"Total embeddings to visualize: {embeddings.shape[0]} (dimension: {embeddings.shape[1]})")

    # Apply UMAP
    print("Applying UMAP...")
    coords = apply_umap(embeddings, n_neighbors=args.n_neighbors,
                       min_dist=args.min_dist, metric=args.metric,
                       random_state=args.random_seed)

    # Prepare coloring
    if args.color_by == 'modality':
        print("Coloring by modality (graph vs text)...")
        if args.embed_type == 'both':
            colors = (modality_labels == 'Graph').astype(int)
            plot_embeddings(coords, colors, labels=['Text', 'Graph'],
                          title='UMAP: Graph vs Text Embeddings',
                          save_path=output_dir / f'umap_{args.split}_modality.svg',
                          discrete=True, alpha=args.alpha, palette=args.palette)
        else:
            print(f"Only {args.embed_type} embeddings present, skipping modality visualization")

    elif args.color_by == 'property':
        print(f"Coloring by property: {args.property_name}...")
        property_values = get_property_values(dataset, args.property_name)

        if args.embed_type == 'both':
            # Duplicate property values for text embeddings
            property_values = np.concatenate([property_values, property_values])

        plot_embeddings(coords, property_values,
                      title=f'UMAP: Colored by {args.property_name}',
                      colorbar_label=args.property_name,
                      save_path=output_dir / f'umap_{args.split}_property_{args.property_name}.svg',
                      cmap=args.cmap, discrete=False, alpha=args.alpha)

        # Also create separate plots for graph and text if both exist
        if args.embed_type == 'both':
            # Graph only
            graph_mask = modality_labels == 'Graph'
            plot_embeddings(coords[graph_mask], property_values[graph_mask],
                          title=f'UMAP (Graph): Colored by {args.property_name}',
                          colorbar_label=args.property_name,
                          save_path=output_dir / f'umap_{args.split}_graph_property_{args.property_name}.svg',
                          cmap=args.cmap, discrete=False, alpha=args.alpha)

            # Text only
            text_mask = modality_labels == 'Text'
            plot_embeddings(coords[text_mask], property_values[text_mask],
                          title=f'UMAP (Text): Colored by {args.property_name}',
                          colorbar_label=args.property_name,
                          save_path=output_dir / f'umap_{args.split}_text_property_{args.property_name}.svg',
                          cmap=args.cmap, discrete=False, alpha=args.alpha)

    elif args.color_by == 'keyword':
        if not args.keywords:
            raise ValueError("--keywords must be specified when --color-by=keyword")

        print(f"Coloring by keyword presence: {args.keywords}...")
        keyword_presence = check_keyword_presence(dataset, args.keywords)

        if args.embed_type == 'both':
            keyword_presence = np.concatenate([keyword_presence, keyword_presence])

        keywords_str = '_'.join(args.keywords)
        plot_embeddings(coords, keyword_presence.astype(int),
                      labels=['Without keywords', 'With keywords'],
                      title=f'UMAP: Keyword(s) "{", ".join(args.keywords)}" Presence',
                      save_path=output_dir / f'umap_{args.split}_keyword_{keywords_str}.svg',
                      discrete=True, alpha=args.alpha, palette=args.palette)

        # Also create separate plots for graph and text if both exist
        if args.embed_type == 'both':
            graph_mask = modality_labels == 'Graph'
            plot_embeddings(coords[graph_mask], keyword_presence[graph_mask].astype(int),
                          labels=['Without keywords', 'With keywords'],
                          title=f'UMAP (Graph): Keyword(s) "{", ".join(args.keywords)}" Presence',
                          save_path=output_dir / f'umap_{args.split}_graph_keyword_{keywords_str}.svg',
                          discrete=True, alpha=args.alpha, palette=args.palette)

            text_mask = modality_labels == 'Text'
            plot_embeddings(coords[text_mask], keyword_presence[text_mask].astype(int),
                          labels=['Without keywords', 'With keywords'],
                          title=f'UMAP (Text): Keyword(s) "{", ".join(args.keywords)}" Presence',
                          save_path=output_dir / f'umap_{args.split}_text_keyword_{keywords_str}.svg',
                          discrete=True, alpha=args.alpha, palette=args.palette)

    elif args.color_by == 'text-similarity':
        if not args.text_query:
            raise ValueError("--text-query must be specified when --color-by=text-similarity")

        print(f"Coloring by text similarity to query: '{args.text_query}'...")

        # Compute similarity to query for all embeddings
        similarities = compute_text_similarity(model, args.text_query, embeddings, args.device, dm, use_projection=use_projection)

        # Create safe filename from query
        query_safe = args.text_query.replace(' ', '_').replace('/', '_')[:50]

        plot_embeddings(coords, similarities,
                      title=f'UMAP: Similarity to "{args.text_query}"',
                      colorbar_label='Cosine Similarity',
                      save_path=output_dir / f'umap_{args.split}_text_similarity_{query_safe}.svg',
                      cmap='coolwarm', discrete=False, alpha=args.alpha,
                      vmin=args.vmin, vmax=args.vmax, symmetric_colormap=True)

        # Also create separate plots for graph and text if both exist
        if args.embed_type == 'both':
            graph_mask = modality_labels == 'Graph'
            plot_embeddings(coords[graph_mask], similarities[graph_mask],
                          title=f'UMAP (Graph): Similarity to "{args.text_query}"',
                          colorbar_label='Cosine Similarity',
                          save_path=output_dir / f'umap_{args.split}_graph_text_similarity_{query_safe}.svg',
                          cmap='coolwarm', discrete=False, alpha=args.alpha,
                          vmin=args.vmin, vmax=args.vmax, symmetric_colormap=True)

            text_mask = modality_labels == 'Text'
            plot_embeddings(coords[text_mask], similarities[text_mask],
                          title=f'UMAP (Text): Similarity to "{args.text_query}"',
                          colorbar_label='Cosine Similarity',
                          save_path=output_dir / f'umap_{args.split}_text_text_similarity_{query_safe}.svg',
                          cmap='coolwarm', discrete=False, alpha=args.alpha,
                          vmin=args.vmin, vmax=args.vmax, symmetric_colormap=True)

        # Print statistics
        print(f"\nSimilarity statistics:")
        print(f"  Mean: {similarities.mean():.4f}")
        print(f"  Std: {similarities.std():.4f}")
        print(f"  Min: {similarities.min():.4f}")
        print(f"  Max: {similarities.max():.4f}")

        if args.embed_type == 'both':
            graph_sims = similarities[modality_labels == 'Graph']
            text_sims = similarities[modality_labels == 'Text']
            print(f"\nGraph embeddings - Mean: {graph_sims.mean():.4f}, Std: {graph_sims.std():.4f}")
            print(f"Text embeddings  - Mean: {text_sims.mean():.4f}, Std: {text_sims.std():.4f}")

    # Save embeddings and coordinates
    save_data = {
        'umap_coords': coords.tolist(),
        'embed_type': args.embed_type,
        'checkpoint': args.checkpoint,
        'n_samples': n_samples,
        'random_seed': args.random_seed,
        'umap_params': {
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist,
            'metric': args.metric,
            'random_state': args.random_seed,
        }
    }

    if args.embed_type == 'both':
        save_data['modality_labels'] = modality_labels.tolist()

    with open(output_dir / f'umap_{args.split}_data.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
