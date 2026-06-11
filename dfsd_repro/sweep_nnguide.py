from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from dfsd_repro.config import config_as_legacy_args, load_config
from dfsd_repro.dfsd import (
    classifier_params,
    extract_features_and_logits,
    load_or_extract_train_features,
    set_reproducible_state,
)
from dfsd_repro.metrics import summarize_score_dir, write_metrics_csv
from dfsd_repro.utils.ood_utils import get_dataloader, get_model, get_outdataset


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep NNGuide K values using the official score formula.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--ks", nargs="+", type=int, required=True, help="K values to evaluate.")
    return parser.parse_args()


def _write_scores_for_loader(loader, model, cfg, scaled_train_features, ks, output_paths: dict[int, Path]):
    max_k = max(ks)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    files = {k: open(path, "w", encoding="utf-8") for k, path in output_paths.items()}
    try:
        for batch_idx, data in enumerate(loader):
            images = data[0].cuda()
            features, outputs = extract_features_and_logits(model, images, cfg)
            query = torch.from_numpy(features.astype(np.float32)).cuda()
            guided = torch.mm(query, scaled_train_features.T)
            values = torch.topk(guided, min(max_k, guided.shape[1]), dim=1).values
            energy = logsumexp(outputs, axis=-1)
            for k in ks:
                k_eff = min(k, values.shape[1])
                guidance = values[:, :k_eff].mean(dim=1).cpu().numpy()
                scores = energy * guidance
                for score in scores:
                    files[k].write(f"{score}\n")
            if batch_idx % 10 == 0:
                print(f"nnguide_sweep: {batch_idx + 1}/{len(loader)}")
    finally:
        for f in files.values():
            f.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ks = sorted(set(args.ks))
    set_reproducible_state(cfg)

    backbone_args = config_as_legacy_args(cfg)
    backbone_args.p = None
    loader_in, num_classes = get_dataloader(cfg.in_dataset, cfg.batch_size, backbone_args)
    model = get_model(backbone_args, num_classes).eval().cuda()

    feature_id_train, _, _ = load_or_extract_train_features(cfg, model, num_classes)
    weight_matrix, bias_vector = classifier_params(model, cfg)
    train_logits = feature_id_train @ weight_matrix.T + bias_vector
    train_energy = logsumexp(train_logits, axis=-1)
    scaled_train_features = torch.from_numpy((feature_id_train * train_energy[:, None]).astype(np.float32)).cuda()

    in_paths = {}
    for k in ks:
        cfg.nnguide_k = k
        in_paths[k] = cfg.score_dir("nnguide") / "in_scores.txt"
    _write_scores_for_loader(loader_in, model, cfg, scaled_train_features, ks, in_paths)

    for out_dataset in cfg.out_datasets:
        loader_out = get_outdataset(out_dataset, cfg.in_dataset, cfg.batch_size)
        out_paths = {}
        for k in ks:
            cfg.nnguide_k = k
            out_paths[k] = cfg.score_dir("nnguide") / out_dataset / "out_scores.txt"
        _write_scores_for_loader(loader_out, model, cfg, scaled_train_features, ks, out_paths)

    for k in ks:
        cfg.nnguide_k = k
        score_dir = cfg.score_dir("nnguide")
        rows = summarize_score_dir(score_dir, cfg.out_datasets)
        write_metrics_csv(rows, score_dir / "metrics.csv")
        avg = rows[-1]
        print(f"nnguide k={k}: AVG FPR95={100 * avg['FPR']:.2f}, AUROC={100 * avg['AUROC']:.2f}")


if __name__ == "__main__":
    main()
