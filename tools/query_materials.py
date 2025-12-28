"""
Query-based Material Retrieval Tool

Usage:
    python query_materials.py --query "semiconductor with wide band gap for UV detection"
    python query_materials.py --query "lightweight magnetic material" --k 20 --format json
    python query_materials.py --query "stable oxide with band gap around 3 eV" --min-similarity 0.7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset

from data.datamodule import CLaCDataModule
from model.clac import CLaCLite


def load_model_and_data(
    data_path: str,
    model_ckpt: str,
    batch_size: int,
    num_workers: int,
    llm: str,
    graphdatatype: str,
    device: str
) -> tuple:
    """Load CLaC model and dataset."""

    print("Loading model and data...")

    # Load model
    model = CLaCLite.load_from_checkpoint(model_ckpt, map_location={'cuda:0': 'cpu'}, strict=False)
    model = model.to(device)
    model.eval()

    # Auto-detect graphdatatype from model if needed
    if graphdatatype == 'auto':
        if hasattr(model.graph_encoder, 'orb_model'):
            detected_graphdatatype = 'orb'
            print(f"Auto-detected graph encoder: ORB")
        else:
            detected_graphdatatype = 'torch_geometric'
            print(f"Auto-detected graph encoder: torch_geometric")
    else:
        detected_graphdatatype = graphdatatype
        print(f"Using specified graph encoder: {detected_graphdatatype}")

    # Get tokenizer model name from checkpoint
    if hasattr(model, 'tokenizer'):
        if hasattr(model.tokenizer, 'tokenizer') and hasattr(model.tokenizer.tokenizer, 'name_or_path'):
            tokenizer_model = model.tokenizer.tokenizer.name_or_path
            print(f"Using model's tokenizer: {tokenizer_model}")
        elif hasattr(model.tokenizer, 'name_or_path'):
            tokenizer_model = model.tokenizer.name_or_path
            print(f"Using model's tokenizer: {tokenizer_model}")
        else:
            tokenizer_model = llm
            print(f"WARNING: Could not detect tokenizer from model, using config: {tokenizer_model}")
    else:
        tokenizer_model = llm
        print(f"Using specified tokenizer: {tokenizer_model}")

    # Initialize data module
    datamodule = CLaCDataModule(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        graphdatatype=detected_graphdatatype,
        tokenizer_model=tokenizer_model,
        debug=False,
    )
    datamodule.setup()

    return model, datamodule, detected_graphdatatype


def encode_candidate_pool(
    model: CLaCLite,
    datamodule: CLaCDataModule,
    data_path: str,
    device: str,
    batch_size: int,
) -> tuple:
    """Encode all materials in the candidate pool."""

    print("Encoding candidate pool...")

    # Load raw dataset to get formulas and properties
    data_path_dict = {'test': data_path + '_test.parquet'}
    raw_dataset = load_dataset('parquet', data_files=data_path_dict)['test']
    print(f"Loaded {len(raw_dataset)} materials from dataset")

    # Get test dataloader
    test_dataloader = datamodule.test_dataloader()

    # Encode graphs
    graph_embeddings = []
    formulas = []
    properties_list = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Encoding materials"):
            graphs, _ = batch  # Ignore text

            if graphs is None:
                print("WARNING: Batch has None graphs. Skipping...")
                continue

            graphs = graphs.to(device)

            # Encode graphs (following compute_metrics.py pattern)
            g_feat = model.graph_encoder(graphs)
            g_emb = model.loss.global_d.img_block(g_feat)
            embeddings = F.normalize(g_emb, p=2, dim=-1)
            graph_embeddings.append(embeddings.cpu())

            # Get data from raw dataset
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(embeddings)

            for i in range(start_idx, min(end_idx, len(raw_dataset))):
                data = raw_dataset[i]

                # Get formula (key is 'pretty formula' with space, not underscore)
                formula = data.get('pretty formula', data.get('formula', 'Unknown'))
                formulas.append(formula)

                # Extract properties (check for None values)
                props = {}
                if 'band gap' in data and data['band gap'] is not None:
                    props['band gap'] = float(data['band gap'])
                if 'formation energy per atom' in data and data['formation energy per atom'] is not None:
                    props['formation_energy'] = float(data['formation energy per atom'])
                if 'density' in data and data['density'] is not None:
                    props['density'] = float(data['density'])
                if 'scintillation attenuation length' in data and data['scintillation attenuation length'] is not None:
                    props['scintillation_length'] = float(data['scintillation attenuation length'])
                if 'total magnetization' in data and data['total magnetization'] is not None:
                    props['magnetization'] = float(data['total magnetization'])
                if 'energy above hull' in data and data['energy above hull'] is not None:
                    props['e_above_hull'] = float(data['energy above hull'])

                properties_list.append(props)

    graph_embeddings = torch.cat(graph_embeddings, dim=0)

    print(f"Encoded {len(formulas)} materials")

    return graph_embeddings, formulas, properties_list


def query_materials(
    query_text: str,
    model: CLaCLite,
    graph_embeddings: torch.Tensor,
    formulas: List[str],
    properties_list: List[Dict],
    device: str,
    k: int = 10,
    min_similarity: float = 0.0
) -> List[Dict]:
    """Query materials using text description."""

    # Encode query text using model's tokenizer (following compute_metrics.py pattern)
    with torch.no_grad():
        inputs = model.tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _, t_feat = model.encode_text(inputs)
        text_embedding = model.loss.global_d.text_block(t_feat)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)

    # Compute similarities
    text_embedding = text_embedding.cpu()
    similarities = torch.cosine_similarity(
        text_embedding.unsqueeze(0),
        graph_embeddings.unsqueeze(1),
        dim=2
    ).squeeze()

    # Filter by minimum similarity
    if min_similarity > 0:
        mask = similarities >= min_similarity
        if not mask.any():
            print(f"Warning: No materials found with similarity >= {min_similarity}")
            print(f"Max similarity: {similarities.max():.4f}")
            return []

        filtered_indices = torch.where(mask)[0]
        filtered_similarities = similarities[mask]

        # Sort by similarity
        sorted_indices = torch.argsort(filtered_similarities, descending=True)[:k]
        top_k_indices = filtered_indices[sorted_indices]
        top_k_similarities = filtered_similarities[sorted_indices]
    else:
        # Get top-k
        top_k_similarities, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))

    # Prepare results
    results = []
    for idx, sim in zip(top_k_indices, top_k_similarities):
        result = {
            'rank': len(results) + 1,
            'formula': formulas[idx],
            'similarity': float(sim),
            'properties': properties_list[idx]
        }
        results.append(result)

    return results


def format_table(results: List[Dict], query_text: str) -> str:
    """Format results as a table."""

    if not results:
        return "No materials found matching the query."

    # Build table header
    lines = []
    lines.append("=" * 120)
    lines.append(f"Query: {query_text}")
    lines.append(f"Found {len(results)} materials")
    lines.append("=" * 120)
    lines.append("")

    # Property columns to show
    prop_cols = ['band gap', 'formation_energy', 'density', 'e_above_hull', 'magnetization']
    prop_names = ['Band Gap', 'Form. Energy', 'Density', 'E_hull', 'Magnet.']
    prop_units = ['(eV)', '(eV/atom)', '(g/cm³)', '(eV/atom)', '(μB)']

    # Header
    header = f"{'Rank':<6} {'Formula':<20} {'Similarity':<12}"
    for name, unit in zip(prop_names, prop_units):
        header += f" {name:<12}"
    lines.append(header)

    # Units row
    units_row = f"{'':6} {'':20} {'':12}"
    for unit in prop_units:
        units_row += f" {unit:<12}"
    lines.append(units_row)
    lines.append("-" * 120)

    # Data rows
    for result in results:
        row = f"{result['rank']:<6} {result['formula']:<20} {result['similarity']:<12.4f}"

        for prop_key in prop_cols:
            if prop_key in result['properties']:
                value = result['properties'][prop_key]
                row += f" {value:<12.3f}"
            else:
                row += f" {'N/A':<12}"

        lines.append(row)

    lines.append("=" * 120)

    return "\n".join(lines)


def save_json(results: List[Dict], query_text: str, output_path: str):
    """Save results to JSON file."""

    output = {
        'query': query_text,
        'num_results': len(results),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")


def save_csv(results: List[Dict], output_path: str):
    """Save results to CSV file."""

    if not results:
        print("No results to save")
        return

    # Flatten properties into columns
    rows = []
    for result in results:
        row = {
            'rank': result['rank'],
            'formula': result['formula'],
            'similarity': result['similarity']
        }
        row.update(result['properties'])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Query materials using text descriptions")

    # Required arguments
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query describing desired material properties"
    )

    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='/home/lucky/Projects/CLaC-revision/datafiles/mp_3d_2020_gpt_narratives',
        help='Path to dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=12,
        help='Number of workers for data loading'
    )

    # Model arguments
    parser.add_argument(
        '--model-ckpt',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default='facebook/galactica-125m',
        help='LLM tokenizer model name'
    )
    parser.add_argument(
        '--graphdatatype',
        type=str,
        default='auto',
        choices=['orb', 'torch_geometric', 'auto'],
        help='Graph encoder type'
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu)"
    )

    # Retrieval arguments
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of materials to retrieve"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum similarity threshold (0.0-1.0)"
    )

    # Output arguments
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json", "csv"],
        default="table",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (for json/csv formats)"
    )

    # Performance arguments
    parser.add_argument(
        "--cache-embeddings",
        type=str,
        default=None,
        help="Path to save/load cached embeddings"
    )

    args = parser.parse_args()

    # Load model and data
    model, datamodule, graphdatatype = load_model_and_data(
        data_path=args.data_path,
        model_ckpt=args.model_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        llm=args.llm,
        graphdatatype=args.graphdatatype,
        device=args.device
    )

    # Load or compute embeddings
    if args.cache_embeddings and Path(args.cache_embeddings).exists():
        print(f"Loading cached embeddings from: {args.cache_embeddings}")
        cache = torch.load(args.cache_embeddings)
        graph_embeddings = cache['embeddings']
        formulas = cache['formulas']
        properties_list = cache['properties']
    else:
        graph_embeddings, formulas, properties_list = encode_candidate_pool(
            model=model,
            datamodule=datamodule,
            data_path=args.data_path,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Save cache if requested
        if args.cache_embeddings:
            print(f"Saving embeddings to: {args.cache_embeddings}")
            torch.save({
                'embeddings': graph_embeddings,
                'formulas': formulas,
                'properties': properties_list
            }, args.cache_embeddings)

    # Query materials
    print(f"\nQuerying: '{args.query}'")
    results = query_materials(
        query_text=args.query,
        model=model,
        graph_embeddings=graph_embeddings,
        formulas=formulas,
        properties_list=properties_list,
        device=args.device,
        k=args.k,
        min_similarity=args.min_similarity
    )

    # Output results
    if args.format == "table":
        print("\n" + format_table(results, args.query))

    elif args.format == "json":
        output_path = args.output or f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(results, args.query, output_path)
        # Also print table
        print("\n" + format_table(results, args.query))

    elif args.format == "csv":
        output_path = args.output or f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_csv(results, output_path)
        # Also print table
        print("\n" + format_table(results, args.query))


if __name__ == "__main__":
    main()
