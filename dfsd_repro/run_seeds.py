"""Run a configuration for independent seeds, preserving configs and summaries."""
import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from dfsd_repro.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--modes', nargs='+', default=['dfsd_main'])
    p.add_argument('--metrics-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Write configurations without starting experiments')
    args = p.parse_args()
    cfg = load_config(args.config)
    dest = cfg.output_root / 'seed_runs'
    dest.mkdir(parents=True, exist_ok=True)
    groups = {}
    for seed in args.seeds:
        raw = asdict(cfg)
        raw.update(seed=seed, experiment_name=f'{cfg.experiment_name}_seed{seed}', modes=args.modes)
        raw['subspace_cache_dir'] = str(Path(cfg.subspace_cache_dir) / f'seed{seed}')
        path = dest / f'seed{seed}.json'
        path.write_text(json.dumps(raw, indent=2) + '\n')
        if args.prepare_only:
            continue
        command = [sys.executable, '-m', 'dfsd_repro.run', '--config', str(path)]
        if args.metrics_only:
            command.append('--metrics-only')
        subprocess.run(command, check=True)
        seed_cfg = load_config(path)
        for mode in args.modes:
            with (seed_cfg.score_dir(mode) / 'metrics.csv').open() as f:
                for row in csv.DictReader(f):
                    for metric in ['FPR', 'AUROC']:
                        groups.setdefault((mode, row['dataset'], metric), []).append(float(row[metric]))
    if groups:
        with (dest / 'summary.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mode', 'dataset', 'metric', 'n_seeds', 'mean_percent', 'std_percent'])
            for key, values in groups.items():
                writer.writerow([*key, len(values), mean(values), stdev(values) if len(values)>1 else 0.0])
    print(dest)


if __name__ == '__main__':
    main()
