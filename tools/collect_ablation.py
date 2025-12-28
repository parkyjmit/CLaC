import csv
import glob
import os
import yaml
import pandas as pd


def find_metrics_csv(run_dir: str) -> str | None:
    # Try common locations
    direct = os.path.join(run_dir, 'metrics.csv')
    if os.path.exists(direct):
        return direct
    # Lightning CSVLogger typically nests into version directories
    matches = glob.glob(os.path.join(run_dir, '**', 'metrics.csv'), recursive=True)
    return matches[0] if matches else None


def pick_best_metric(df: pd.DataFrame, metric: str, mode: str):
    if metric not in df.columns:
        # Some metrics may be logged under epoch-level; try last occurrence
        cols = [c for c in df.columns if c.endswith(metric)]
        if cols:
            metric = cols[0]
        else:
            return None
    sdf = df.dropna(subset=[metric])
    if sdf.empty:
        return None
    return sdf[metric].min() if mode == 'min' else sdf[metric].max()


def main():
    rows = []
    for run_dir in glob.glob('outputs/*/*/'):
        cfg_path = os.path.join(run_dir, '.hydra', 'config.yaml')
        metrics_path = find_metrics_csv(run_dir)
        if not (os.path.exists(cfg_path) and metrics_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            df = pd.read_csv(metrics_path)
        except Exception:
            continue

        monitor = cfg.get('trainer', {}).get('monitor_metric', 'val_loss')
        mode = cfg.get('trainer', {}).get('monitor_metric_mode', 'min')
        best = pick_best_metric(df, monitor, mode)
        if best is None:
            continue

        model_cfg = cfg.get('model', {})
        graph_encoder = model_cfg.get('graph_encoder', {}).get('_target_')
        text_model = model_cfg.get('datamodule', {}).get('tokenizer_model')
        augmentation = model_cfg.get('augmentation')
        lm_weight = model_cfg.get('lm_weight')
        loss_cfg = model_cfg.get('loss', {})

        row = {
            'run_dir': run_dir,
            'monitor': monitor,
            'mode': mode,
            'best_metric': best,
            'graph_encoder': graph_encoder,
            'text_model': text_model,
            'augmentation': augmentation,
            'lm_weight': lm_weight,
            'loss_type': loss_cfg.get('type'),
            'image_prior': loss_cfg.get('image_prior'),
            'text_prior': loss_cfg.get('text_prior'),
            'visual_ssl': loss_cfg.get('visual_self_supervised'),
            'textual_ssl': loss_cfg.get('textual_self_supervised'),
            'seed': cfg.get('trainer', {}).get('random_seed'),
        }
        rows.append(row)

    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/ablation_summary.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'saved: {out_path}, rows={len(rows)}')


if __name__ == '__main__':
    main()

