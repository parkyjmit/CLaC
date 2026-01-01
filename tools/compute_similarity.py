#!/usr/bin/env python3
"""
Compute Structure-Text Similarity for Specific Materials

This tool computes the similarity between a specific material (by mp-id or material-id)
and a text prompt using a trained CLaC model.

Features:
- Search materials in local parquet dataset
- Fallback to Materials Project API if not found locally
- Support for multiple mp-ids in one query
- Save results to JSON

Usage:
    # Single mp-id with text prompt (search local dataset first)
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-id mp-19990 \
        --text "Materials for solid-state batteries applications"

    # Multiple mp-ids
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-ids mp-19990 mp-1234 mp-5678 \
        --text "High energy density cathode materials"

    # Use Materials Project API (when material not in local dataset)
    # Method 1: Set in .env file (recommended)
    # Add to .env file: MP_API_KEY=your_api_key_here
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-id mp-99999 \
        --text "Solid-state electrolytes"

    # Method 2: Set environment variable
    # export MP_API_KEY="your_api_key_here"
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-id mp-99999 \
        --text "Solid-state electrolytes"

    # Method 3: Pass as command-line argument
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-id mp-99999 \
        --text "Battery materials" \
        --mp-api-key "your_api_key"

    # Always use Materials Project API (skip local dataset)
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-id mp-19990 \
        --text "Battery materials" \
        --use-mp-api

    # Output to JSON file
    python tools/compute_similarity.py \
        --model outputs/my-model/last.ckpt \
        --mp-id mp-19990 \
        --text "Solid-state electrolytes" \
        --output results.json

Requirements for Materials Project API:
    pip install mp-api

Get API key from: https://next-gen.materialsproject.org/api
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Load environment variables from .env file
_env_loaded = False
_env_path = Path(__file__).parent.parent / '../.env'

try:
    from dotenv import load_dotenv
    if _env_path.exists():
        load_dotenv(_env_path)
        _env_loaded = True
        # Check if MP_API_KEY was actually loaded
        mp_key = os.environ.get('MP_API_KEY')
        print(f"✓ Loaded .env from {_env_path}")
        if mp_key:
            print(f"✓ MP_API_KEY found: {mp_key[:10]}...{mp_key[-4:]}")
        else:
            print(f"⚠ WARNING: MP_API_KEY not found in .env file")
    else:
        print(f"⚠ WARNING: .env file not found at {_env_path}")
except ImportError:
    print(f"⚠ WARNING: python-dotenv not installed")
    print(f"  Install with: pip install python-dotenv")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.clac import CLaCLite
from data.datamodule import (
    get_atoms_from_data,
    atom_to_torch_graph_data,
    atom_to_orb_graph_data
)


def load_model(model_ckpt: str, device: str, graphdatatype: str = 'auto', llm: str = 'facebook/galactica-125m'):
    """Load CLaC model from checkpoint."""

    print(f"Loading model from: {model_ckpt}")

    # Load model
    model = CLaCLite.load_from_checkpoint(model_ckpt, map_location={'cuda:0': 'cpu'}, strict=False)
    model = model.to(device)
    model.eval()

    # Auto-detect graphdatatype from model if needed
    if graphdatatype == 'auto':
        if hasattr(model.graph_encoder, 'orb_model'):
            detected_graphdatatype = 'orb'
            print(f"✓ Auto-detected graph encoder: ORB")
        else:
            detected_graphdatatype = 'torch_geometric'
            print(f"✓ Auto-detected graph encoder: torch_geometric")
    else:
        detected_graphdatatype = graphdatatype
        print(f"✓ Using specified graph encoder: {detected_graphdatatype}")

    # Get tokenizer model name from checkpoint
    if hasattr(model, 'tokenizer'):
        if hasattr(model.tokenizer, 'tokenizer') and hasattr(model.tokenizer.tokenizer, 'name_or_path'):
            tokenizer_model = model.tokenizer.tokenizer.name_or_path
            print(f"✓ Using model's tokenizer: {tokenizer_model}")
        elif hasattr(model.tokenizer, 'name_or_path'):
            tokenizer_model = model.tokenizer.name_or_path
            print(f"✓ Using model's tokenizer: {tokenizer_model}")
        else:
            tokenizer_model = llm
            print(f"⚠ WARNING: Could not detect tokenizer from model, using config: {tokenizer_model}")
    else:
        tokenizer_model = llm
        print(f"✓ Using specified tokenizer: {tokenizer_model}")

    print(f"✓ Model loaded successfully\n")

    return model, detected_graphdatatype, tokenizer_model


def find_material_by_id(data_path: str, material_id: str, id_column: str = 'material-id') -> Optional[Dict]:
    """
    Find a material by its ID in the parquet dataset.

    Args:
        data_path: Base path to dataset (without _train/_val/_test suffix)
        material_id: Material ID to search for (e.g., 'mp-19990')
        id_column: Column name for material ID (default: 'material-id')

    Returns:
        Dictionary containing material data, or None if not found
    """

    # Try all splits
    for split in ['test', 'val', 'train']:
        parquet_path = f"{data_path}_{split}.parquet"

        if not Path(parquet_path).exists():
            continue

        df = pd.read_parquet(parquet_path)

        # Check if id_column exists
        if id_column not in df.columns:
            continue

        # Find material
        matches = df[df[id_column] == material_id]

        if len(matches) > 0:
            print(f"✓ Found {material_id} in {split} split")
            row_dict = matches.iloc[0].to_dict()
            row_dict['_split'] = split
            row_dict['_index'] = matches.index[0]
            return row_dict

    return None


def fetch_from_materials_project(mp_id: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch material data from Materials Project API.

    Args:
        mp_id: Materials Project ID (e.g., 'mp-19990')
        api_key: Materials Project API key (if None, reads from MP_API_KEY env variable)

    Returns:
        Dictionary with structure info, or None if fetch failed
    """
    # Check imports
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        error_msg = str(e)
        print(f"⚠ ERROR: Cannot import mp_api: {error_msg}")
        if "NotRequired" in error_msg or "typing" in error_msg:
            print("  Fix: pip install --upgrade typing_extensions")
            print("  (mp-api requires Python 3.11+ or typing_extensions for Python 3.10)")
        else:
            print("  Install with: pip install mp-api")
        return None

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as e:
        print(f"⚠ ERROR: Cannot import pymatgen/ase: {e}")
        print("  Install with: pip install pymatgen ase")
        return None

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get('MP_API_KEY')
        if api_key is None:
            print("⚠ ERROR: Materials Project API key not found")
            print("  Set MP_API_KEY environment variable or use --mp-api-key argument")
            print("  Get API key from: https://next-gen.materialsproject.org/api")
            return None

    print(f"  Fetching {mp_id} from Materials Project API...")

    try:
        with MPRester(api_key) as mpr:
            # Get material summary (includes structure and properties)
            try:
                summary = mpr.materials.summary.search(material_ids=[mp_id])
            except Exception as e:
                print(f"  ✗ Error searching for {mp_id}: {type(e).__name__}: {e}")
                return None

            if not summary or len(summary) == 0:
                print(f"  ✗ Material {mp_id} not found in Materials Project")
                return None

            mat_data = summary[0]

            # Get structure
            try:
                structure = mat_data.structure
            except AttributeError:
                print(f"  ✗ No structure available for {mp_id}")
                return None

            # Convert pymatgen Structure to ASE Atoms
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)

            # Extract properties and build material dictionary
            material_dict = {
                'atoms': {
                    'numbers': atoms.get_atomic_numbers().tolist(),
                    'positions': atoms.get_positions().tolist(),
                    'cell': atoms.get_cell().tolist(),
                    'pbc': atoms.get_pbc().tolist()
                },
                'pretty formula': mat_data.formula_pretty if hasattr(mat_data, 'formula_pretty') else structure.composition.reduced_formula,
                'reduced_formula': structure.composition.reduced_formula,
                'space group symbol': str(mat_data.symmetry.symbol) if hasattr(mat_data, 'symmetry') and hasattr(mat_data.symmetry, 'symbol') else 'Unknown',
                'crystal system': str(mat_data.symmetry.crystal_system) if hasattr(mat_data, 'symmetry') and hasattr(mat_data.symmetry, 'crystal_system') else 'Unknown',
                '_source': 'materials_project_api',
                '_mp_id': mp_id
            }

            # Add available properties
            if hasattr(mat_data, 'band_gap'):
                material_dict['band gap (eV)'] = mat_data.band_gap
            if hasattr(mat_data, 'formation_energy_per_atom'):
                material_dict['formation energy per atom (eV/atom)'] = mat_data.formation_energy_per_atom
            if hasattr(mat_data, 'density'):
                material_dict['density (g/cm³)'] = mat_data.density
            if hasattr(mat_data, 'energy_above_hull'):
                material_dict['energy above hull (eV/atom)'] = mat_data.energy_above_hull
            if hasattr(mat_data, 'total_magnetization'):
                material_dict['total magnetization (μB/f.u.)'] = mat_data.total_magnetization

            print(f"  ✓ Fetched {mp_id} from Materials Project API")
            return material_dict

    except Exception as e:
        print(f"  ✗ Error fetching {mp_id} from Materials Project: {e}")
        return None


def encode_structure(model, material_data: Dict, graphdatatype: str, device: str) -> torch.Tensor:
    """Encode a single material structure to embedding."""

    from ase import Atoms as ASEAtoms

    # Get atoms object from data
    # Check if data is from MP API (has 'numbers', 'positions', 'cell', 'pbc')
    if '_source' in material_data and material_data['_source'] == 'materials_project_api':
        # Data from MP API: reconstruct ASE Atoms directly
        atoms_dict = material_data['atoms']
        atoms = ASEAtoms(
            numbers=atoms_dict['numbers'],
            positions=atoms_dict['positions'],
            cell=atoms_dict['cell'],
            pbc=atoms_dict['pbc']
        )
    else:
        # Data from local dataset: use standard conversion
        atoms = get_atoms_from_data(material_data)

    # Convert to graph
    if graphdatatype == 'torch_geometric':
        from torch_geometric.data import Batch
        graph_data = atom_to_torch_graph_data(atoms)
        batch = Batch.from_data_list([graph_data]).to(device)
    elif graphdatatype == 'orb':
        from orb_models.dataset.ase_dataset import batch_graphs
        from orb_models.forcefield import pretrained
        orbff = pretrained.orb_v2(device='cpu', precision='float32-high')
        graph_data = atom_to_orb_graph_data(atoms, orbff.system_config)
        batch = batch_graphs([graph_data]).to(device)
    else:
        raise ValueError(f"Unsupported graphdatatype: {graphdatatype}")

    # Encode structure
    with torch.no_grad():
        g_feat = model.graph_encoder(batch)
        g_emb = model.loss.global_d.img_block(g_feat)
        g_emb = F.normalize(g_emb, p=2, dim=-1)

    return g_emb


def encode_text(model, text: str, device: str) -> torch.Tensor:
    """Encode text to embedding."""

    with torch.no_grad():
        inputs = model.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        _, t_feat = model.encode_text(inputs)
        t_emb = model.loss.global_d.text_block(t_feat)
        t_emb = F.normalize(t_emb, p=2, dim=-1)

    return t_emb


def compute_similarity(structure_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
    """Compute cosine similarity between structure and text embeddings."""

    # Cosine similarity (already normalized, so just dot product)
    similarity = (structure_emb * text_emb).sum().item()

    return similarity


def format_properties(material_data: Dict) -> Dict:
    """Extract and format relevant properties from material data."""

    props = {}

    # Common property keys (with space, as used in parquet)
    property_mapping = {
        'pretty formula': 'formula',
        'reduced_formula': 'reduced_formula',
        'space group symbol': 'space_group',
        'crystal system': 'crystal_system',
        'band gap (eV)': 'band_gap_eV',
        'formation energy per atom (eV/atom)': 'formation_energy_eV_atom',
        'density (g/cm³)': 'density_g_cm3',
        'energy above hull (eV/atom)': 'e_above_hull_eV_atom',
        'total magnetization (μB/f.u.)': 'magnetization_uB',
        'scintillation attenuation length (cm)': 'scintillation_length_cm',
        'oxide type': 'oxide_type',
        'volume (Å³)': 'volume_A3',
    }

    for key, name in property_mapping.items():
        if key in material_data and material_data[key] is not None:
            props[name] = material_data[key]

    return props


def main():
    parser = argparse.ArgumentParser(
        description='Compute structure-text similarity for specific materials',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model', '--model-ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--graphdatatype', type=str, default='auto',
                       choices=['auto', 'torch_geometric', 'orb'],
                       help='Graph data type (default: auto-detect)')
    parser.add_argument('--llm', type=str, default='facebook/galactica-125m',
                       help='LLM tokenizer (default: facebook/galactica-125m)')

    # Data arguments
    parser.add_argument('--data-path', type=str, default='datafiles/mp_3d_2020_gpt_narratives',
                       help='Path to dataset (without _train/_val/_test suffix)')
    parser.add_argument('--id-column', type=str, default='material-id',
                       help='Column name for material ID (default: material-id)')

    # Query arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mp-id', type=str,
                      help='Single material ID to query (e.g., mp-19990)')
    group.add_argument('--mp-ids', type=str, nargs='+',
                      help='Multiple material IDs to query')

    parser.add_argument('--text', type=str, required=True,
                       help='Text prompt to compute similarity with')

    # Materials Project API arguments
    parser.add_argument('--mp-api-key', type=str, default=None,
                       help='Materials Project API key (default: read from MP_API_KEY env variable)')
    parser.add_argument('--use-mp-api', action='store_true',
                       help='Always use Materials Project API (skip local dataset search)')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')

    args = parser.parse_args()

    # Get list of material IDs
    if args.mp_id:
        material_ids = [args.mp_id]
    else:
        material_ids = args.mp_ids

    print("="*80)
    print("Structure-Text Similarity Computation")
    print("="*80)
    print(f"Material IDs: {', '.join(material_ids)}")
    print(f"Text prompt: \"{args.text}\"")
    print(f"Model: {args.model}")
    print("="*80)
    print()

    # Load model
    model, graphdatatype, _ = load_model(
        args.model,
        args.device,
        args.graphdatatype,
        args.llm
    )

    # Encode text prompt once
    print(f"Encoding text prompt...")
    text_emb = encode_text(model, args.text, args.device)
    print(f"✓ Text embedding shape: {text_emb.shape}\n")

    # Process each material
    results = []

    for material_id in tqdm(material_ids, desc="Processing materials"):
        # Find material in dataset (skip if --use-mp-api is set)
        if args.use_mp_api:
            material_data = None
        else:
            material_data = find_material_by_id(args.data_path, material_id, args.id_column)

        # If not found in dataset, try Materials Project API
        if material_data is None:
            if not args.use_mp_api:
                print(f"  Material {material_id} not found in local dataset")

            # Try fetching from Materials Project API
            material_data = fetch_from_materials_project(material_id, args.mp_api_key)

            if material_data is None:
                print(f"✗ Material {material_id} not found in dataset or Materials Project API")
                results.append({
                    'material_id': material_id,
                    'found': False,
                    'error': 'Material not found in dataset or Materials Project API'
                })
                continue

        # Encode structure
        if args.verbose:
            print(f"  Encoding structure for {material_id}...")

        try:
            structure_emb = encode_structure(model, material_data, graphdatatype, args.device)

            if args.verbose:
                print(f"  ✓ Structure embedding shape: {structure_emb.shape}")

            # Compute similarity
            similarity = compute_similarity(structure_emb, text_emb)

            # Extract properties
            properties = format_properties(material_data)

            # Store result
            result = {
                'material_id': material_id,
                'found': True,
                'similarity': float(similarity),
                'properties': properties,
                'text_prompt': args.text,
                'source': material_data.get('_source', 'local_dataset'),
                'split': material_data.get('_split', 'unknown')
            }

            results.append(result)

            if not args.verbose:
                print(f"  ✓ {material_id}: similarity = {similarity:.4f}")

        except Exception as e:
            print(f"  ✗ Error processing {material_id}: {e}")
            results.append({
                'material_id': material_id,
                'found': True,
                'error': str(e)
            })

    print()
    print("="*80)
    print("Results Summary")
    print("="*80)

    # Print summary
    for result in results:
        if not result['found']:
            print(f"{result['material_id']:15s} NOT FOUND")
        elif 'error' in result:
            print(f"{result['material_id']:15s} ERROR: {result['error']}")
        else:
            sim = result['similarity']
            formula = result['properties'].get('formula', 'N/A')
            print(f"{result['material_id']:15s} {formula:20s} similarity = {sim:.4f}")

    print("="*80)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'text_prompt': args.text,
            'model': args.model,
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
