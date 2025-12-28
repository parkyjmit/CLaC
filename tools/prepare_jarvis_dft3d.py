#!/usr/bin/env python
"""
Prepare JARVIS dft_3d dataset for supervised learning.
Downloads data from JARVIS, splits into train/val/test (80/10/10), and saves as parquet.
"""

import pandas as pd
import numpy as np
from jarvis.db.figshare import data
from pathlib import Path
import argparse
import ast


def main(output_dir: str = "datafiles", dataset_name: str = "dft_3d_2021"):
    """
    Download JARVIS dft_3d dataset and prepare train/val/test splits.

    Args:
        output_dir: Directory to save parquet files
        dataset_name: Name of JARVIS dataset (default: dft_3d)
    """
    print(f"Loading JARVIS {dataset_name} dataset...")
    dft_3d_data = data(dataset_name)

    # Convert to DataFrame
    df = pd.DataFrame(dft_3d_data)
    print(f"Loaded {len(df)} materials")
    print(f"Columns: {df.columns.tolist()}")

    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"  Total materials: {len(df)}")

    # Check for key properties
    key_properties = [
        'formation_energy_peratom',
        'optb88vdw_bandgap',
        'mbj_bandgap',
        'total_magnetization',
        'ehull',
        'bulk_modulus_kv',
        'shear_modulus_gv',
    ]

    print("\nProperty availability:")
    for prop in key_properties:
        if prop in df.columns:
            non_null = df[prop].notna().sum()
            print(f"  {prop}: {non_null} / {len(df)} ({non_null/len(df)*100:.1f}%)")

    # Clean data types for parquet compatibility
    print("\nCleaning data types for parquet compatibility...")
    for col in df.columns:
        # Special handling for 'atoms' column - keep as dict/object
        if col == 'atoms':
            print(f"  Kept {col} as dict/object type (structural data)")
            continue

        # Convert object columns with mixed types to strings first
        if df[col].dtype == 'object':
            # Check if the column should be numeric
            # Try to convert to numeric, coercing errors to NaN
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # If most values converted successfully, use numeric type
                if numeric_col.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_col
                    print(f"  Converted {col} to numeric (float)")
                else:
                    # Keep as string, but clean up
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace(['nan', 'None', 'NA', 'N/A', 'na'], np.nan)
                    print(f"  Kept {col} as string type")
            except Exception:
                # If conversion fails, keep as string
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'None', 'NA', 'N/A', 'na'], np.nan)
                print(f"  Kept {col} as string type (conversion failed)")

    # Shuffle dataset
    print("\nShuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train/val/test (80/10/10)
    n = len(df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to parquet files
    base_name = f"jarvis_{dataset_name}"
    train_file = output_path / f"{base_name}_train.parquet"
    val_file = output_path / f"{base_name}_val.parquet"
    test_file = output_path / f"{base_name}_test.parquet"

    print(f"\nSaving parquet files to {output_path}...")
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)

    print(f"  ✓ {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  ✓ {val_file} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  ✓ {test_file} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Verify saved files
    print("\nVerifying saved files...")
    train_check = pd.read_parquet(train_file)
    val_check = pd.read_parquet(val_file)
    test_check = pd.read_parquet(test_file)

    assert len(train_check) == len(train_df), "Train size mismatch!"
    assert len(val_check) == len(val_df), "Val size mismatch!"
    assert len(test_check) == len(test_df), "Test size mismatch!"

    print("✓ All files saved and verified successfully!")

    # Print example properties for supervised learning
    print("\n" + "="*60)
    print("Available properties for supervised learning:")
    print("="*60)

    # Properties with good coverage
    good_properties = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            coverage = df[col].notna().sum() / len(df)
            if coverage > 0.5:  # At least 50% coverage
                good_properties.append((col, coverage))

    good_properties.sort(key=lambda x: x[1], reverse=True)

    for prop, coverage in good_properties[:20]:  # Top 20
        print(f"  {prop:<40} {coverage*100:>5.1f}% coverage")

    print("\n" + "="*60)
    print(f"Dataset path: {output_path}/{base_name}")
    print("="*60)
    print("\nUsage in config:")
    print(f"  data_path: {output_path}/{base_name}")
    print(f"  label: formation_energy_peratom  # or any property above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare JARVIS dft_3d dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datafiles",
        help="Output directory for parquet files (default: datafiles)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dft_3d_2021",
        help="JARVIS dataset name (default: dft_3d)"
    )

    args = parser.parse_args()
    main(output_dir=args.output_dir, dataset_name=args.dataset)
