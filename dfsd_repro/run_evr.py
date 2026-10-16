"""Evaluate retained KPCA eigenvalue-mass thresholds on one fixed configuration."""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from dfsd_repro.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--evr', type=float, nargs='+', default=[0.1, 0.5, 0.9, 0.99, 1.0])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--metrics-only', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    dest = cfg.output_root / 'evr_configs'
    dest.mkdir(parents=True, exist_ok=True)
    for value in args.evr:
        if not 0 < value <= 1:
            p.error('EVR must lie in (0,1]; run cpp_dice separately as the component-off reference.')
        raw = asdict(cfg)
        raw.update(evr=value, modes=['dfsd_main'])
        path = dest / f'evr{value:g}.json'
        path.write_text(json.dumps(raw, indent=2) + '\n')
        if not args.prepare_only:
            command = [sys.executable, '-m', 'dfsd_repro.run', '--config', str(path)]
            if args.metrics_only:command.append('--metrics-only')
            subprocess.run(command, check=True)
    print(dest)


if __name__ == '__main__':
    main()
