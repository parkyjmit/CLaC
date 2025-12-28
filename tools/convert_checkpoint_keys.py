"""
Convert old checkpoint keys to new format for backward compatibility.
Usage: python tools/convert_checkpoint_keys.py --input old.ckpt --output new.ckpt
"""
import torch
import argparse
from pathlib import Path


def convert_checkpoint_keys(checkpoint_path, output_path=None):
    """
    Convert checkpoint keys from old format to new format.

    Old format: use_intramodal_loss
    New format: use_visual_intramodal_loss, use_textual_intramodal_loss
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    modified = False

    # Check if old key exists
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']

        if 'use_intramodal_loss' in hparams and 'use_visual_intramodal_loss' not in hparams:
            print(f"Converting 'use_intramodal_loss' to new format")
            use_intramodal = hparams['use_intramodal_loss']
            hparams['use_visual_intramodal_loss'] = use_intramodal
            hparams['use_textual_intramodal_loss'] = use_intramodal
            modified = True
            print(f"  use_visual_intramodal_loss = {use_intramodal}")
            print(f"  use_textual_intramodal_loss = {use_intramodal}")

    # Also handle state_dict keys if needed
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        keys_to_rename = {}

        # Add any state_dict key renaming logic here if needed
        for old_key, new_key in keys_to_rename.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
                modified = True
                print(f"  Renamed: {old_key} -> {new_key}")

    if modified:
        if output_path is None:
            output_path = checkpoint_path.replace('.ckpt', '_converted.ckpt')

        print(f"Saving converted checkpoint to {output_path}")
        torch.save(ckpt, output_path)
        print("Conversion complete!")
    else:
        print("No conversion needed - checkpoint already in new format")

    return output_path if modified else checkpoint_path


def batch_convert(input_dir, output_dir=None):
    """Convert all checkpoints in a directory"""
    input_path = Path(input_dir)
    checkpoints = list(input_path.rglob('*.ckpt'))

    print(f"Found {len(checkpoints)} checkpoints to convert")

    for ckpt_path in checkpoints:
        if output_dir:
            output_path = Path(output_dir) / ckpt_path.relative_to(input_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path)
        else:
            output_path = None

        try:
            convert_checkpoint_keys(str(ckpt_path), output_path)
        except Exception as e:
            print(f"Error converting {ckpt_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert checkpoint keys for backward compatibility')
    parser.add_argument('--input', type=str, required=True, help='Input checkpoint file or directory')
    parser.add_argument('--output', type=str, help='Output checkpoint file or directory')
    parser.add_argument('--batch', action='store_true', help='Batch convert all checkpoints in input directory')

    args = parser.parse_args()

    if args.batch:
        batch_convert(args.input, args.output)
    else:
        convert_checkpoint_keys(args.input, args.output)
