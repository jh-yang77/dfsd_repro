from __future__ import annotations

import argparse

from dfsd_repro.config import load_config
from dfsd_repro.dfsd import run_mode, set_reproducible_state
from dfsd_repro.metrics import summarize_score_dir, write_metrics_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run clean DFSD reproduction experiments.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--modes", nargs="*", default=None, help="Override modes from config.")
    parser.add_argument("--c-pp", choices=["dice", "ash", "scale"], default=None, help="Override C-PP component.")
    parser.add_argument("--pca-components", type=int, default=None, help="Override retained PCA components for PCA ablations.")
    parser.add_argument("--metrics-only", action="store_true", help="Only summarize existing scores.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.modes:
        cfg.modes = args.modes
    if args.c_pp:
        cfg.c_pp = args.c_pp
    if args.pca_components is not None:
        cfg.pca_components = args.pca_components

    set_reproducible_state(cfg)
    for mode in cfg.modes:
        if not args.metrics_only:
            run_mode(cfg, mode)

        score_dir = cfg.score_dir(mode)
        rows = summarize_score_dir(score_dir, cfg.out_datasets)
        write_metrics_csv(rows, score_dir / "metrics.csv")
        avg = rows[-1]
        print(
            f"{mode}: AVG FPR95={100 * avg['FPR']:.2f}, "
            f"AUROC={100 * avg['AUROC']:.2f}, AUIN={100 * avg['AUIN']:.2f}"
        )


if __name__ == "__main__":
    main()
