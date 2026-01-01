"""
Split dataset into train, val, test sets.

This script splits a parquet file into train (90%), val (5%), and test (5%) sets.

Usage:
    python data/get_splits.py --input datafiles/mp_3d_2020_gpt_narratives.parquet
    python data/get_splits.py --input datafiles/mp_3d_2020_gpt_narratives.parquet --train-ratio 0.8 --val-ratio 0.1
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def split_dataset(input_path: str,
                  train_ratio: float = 0.9,
                  val_ratio: float = 0.05,
                  seed: int = 42):
    """
    Split dataset into train, val, test sets.

    Args:
        input_path: Path to input parquet file
        train_ratio: Ratio for training set (default: 0.9)
        val_ratio: Ratio for validation set (default: 0.05)
        seed: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    total_samples = len(df)
    print(f"Total samples: {total_samples}")

    # Shuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate split indices
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    # Split data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Add dummy 'y' field
    if 'y' not in train_df.columns:
        print(f"\nAdding 'y' field (answer index = 0) for zero-shot QA evaluation...")
        train_df['y'] = 0.0
        val_df['y'] = 0.0
        test_df['y'] = 0.0

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")

    # Generate output paths
    output_dir = input_path.parent
    base_name = input_path.stem  # Remove .parquet extension

    train_path = output_dir / f"{base_name}_train.parquet"
    val_path = output_dir / f"{base_name}_val.parquet"
    test_path = output_dir / f"{base_name}_test.parquet"

    # Save splits
    print(f"\nSaving splits...")
    train_df.to_parquet(train_path)
    print(f"  Train: {train_path}")

    val_df.to_parquet(val_path)
    print(f"  Val:   {val_path}")

    test_df.to_parquet(test_path)
    print(f"  Test:  {test_path}")

    print(f"\n✓ Done! Created {len(train_df) + len(val_df) + len(test_df)} total samples.")


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default split (90/5/5)
  python data/get_splits.py --input datafiles/mp_3d_2020_gpt_narratives.parquet

  # Custom split ratios (80/10/10)
  python data/get_splits.py \
      --input datafiles/mp_3d_2020_gpt_narratives.parquet \
      --train-ratio 0.8 \
      --val-ratio 0.1

  # With custom random seed
  python data/get_splits.py \
      --input datafiles/mp_3d_2020_gpt_narratives.parquet \
      --seed 123
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input parquet file')

    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio for training set (default: 0.9)')

    parser.add_argument('--val-ratio', type=float, default=0.05,
                       help='Ratio for validation set (default: 0.05). '
                            'Test ratio will be 1 - train_ratio - val_ratio')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError(f"Invalid ratios: train_ratio + val_ratio must be <= 1.0")

    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(f"All ratios must be positive")

    print(f"Split ratios: train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={test_ratio:.2f}")

    # Split dataset
    split_dataset(
        input_path=args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
