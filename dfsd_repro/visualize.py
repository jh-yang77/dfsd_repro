from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.linalg import norm
from sklearn.manifold import TSNE

from dfsd_repro.config import ExperimentConfig, config_as_legacy_args, load_config, package_path
from dfsd_repro.dfsd import (
    extract_features_and_logits,
    fit_or_load_subspaces,
    score_numpy,
    set_reproducible_state,
)
from dfsd_repro.utils.ood_utils import get_dataloader, get_model, get_outdataset


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "figure.dpi": 150,
        }
    )


def load_scores(path: str | Path):
    arr = np.loadtxt(path)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr


def subsample_mask(x, y, mask, n, seed=0):
    x_masked = x[mask]
    y_masked = y[mask]
    if n is None or n <= 0 or n >= len(x_masked):
        return x_masked, y_masked
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_masked), size=n, replace=False)
    return x_masked[idx], y_masked[idx]


def score_dir(cfg: ExperimentConfig, mode: str) -> Path:
    return cfg.score_dir(mode)


def plot_complementarity(cfg: ExperimentConfig, out_dataset: str, output_dir: str | Path):
    """Plot C-PP/C-SP complementarity using saved score files."""
    set_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c_pp_dir = score_dir(cfg, "cpp_dice")
    c_sp_dir = score_dir(cfg, "csp_kpca")
    dfsd_dir = score_dir(cfg, "dfsd_main")

    xin = load_scores(c_pp_dir / "in_scores.txt")
    yin = load_scores(c_sp_dir / "in_scores.txt")
    sin = load_scores(dfsd_dir / "in_scores.txt")
    xout = load_scores(c_pp_dir / out_dataset / "out_scores.txt")
    yout = load_scores(c_sp_dir / out_dataset / "out_scores.txt")
    sout = load_scores(dfsd_dir / out_dataset / "out_scores.txt")

    lam_cpp = np.quantile(xin, 0.05)
    lam_csp = np.quantile(yin, 0.05)
    lam_cm = np.quantile(sin, 0.05)

    ood_by_cpp = xout < lam_cpp
    ood_by_csp = yout < lam_csp
    ood_by_cm = sout < lam_cm

    caught_by_both = ood_by_cpp & ood_by_csp
    miss_cpp_catch_cm = (~ood_by_cpp) & ood_by_cm
    miss_csp_catch_cm = (~ood_by_csp) & ood_by_cm

    xin_p, yin_p = subsample_mask(xin, yin, np.ones(len(xin), dtype=bool), 2000, seed=1)
    x_both_p, y_both_p = subsample_mask(xout, yout, caught_by_both, 2000, seed=2)
    x_c1_p, y_c1_p = subsample_mask(xout, yout, miss_cpp_catch_cm, 2000, seed=3)
    x_c2_p, y_c2_p = subsample_mask(xout, yout, miss_csp_catch_cm, 2000, seed=4)

    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.scatter(
        x_both_p,
        y_both_p,
        c="gray",
        s=12,
        alpha=0.15,
        marker=".",
        label="OOD (Caught by both parts)",
        zorder=1,
    )
    ax.scatter(xin_p, yin_p, c="#3498db", s=12, alpha=0.25, marker="o", label="ID", zorder=2)
    ax.scatter(
        x_c1_p,
        y_c1_p,
        c="#e74c3c",
        s=12,
        alpha=0.45,
        marker="x",
        linewidth=1.8,
        label="OOD: Missed by C-PP, caught by DFSD",
        zorder=3,
    )
    ax.scatter(
        x_c2_p,
        y_c2_p,
        c="#e67e22",
        s=12,
        alpha=0.45,
        marker="+",
        linewidth=1.8,
        label="OOD: Missed by C-SP, caught by DFSD",
        zorder=3,
    )

    ax.autoscale(enable=True, axis="both", tight=True)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0] - 0.5, x_lim[1] + 0.5)
    ax.set_ylim(y_lim[0] - 0.5, y_lim[1] + 0.5)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    line_opts = {"color": "k", "linestyle": "--", "linewidth": 2, "alpha": 0.7, "zorder": 4}
    ax.axvline(lam_cpp, **line_opts)
    ax.axhline(lam_csp, **line_opts)
    xx = np.linspace(x_lim[0], x_lim[1], 200)
    yy = lam_cm - xx
    ax.plot(
        xx,
        yy,
        color="#8e44ad",
        linestyle="--",
        linewidth=2.5,
        alpha=0.7,
        zorder=4,
        label=r"Fusion Boundary ($x+y=\lambda_{DFSD}$)",
    )

    ax.text(
        lam_cpp,
        y_lim[1] - (y_lim[1] - y_lim[0]) * 0.02,
        r" $\lambda_{C-PP}$",
        verticalalignment="top",
        horizontalalignment="left",
        fontweight="bold",
    )
    ax.text(
        x_lim[1] - (x_lim[1] - x_lim[0]) * 0.02,
        lam_csp,
        r"$\lambda_{C-SP}$ ",
        verticalalignment="bottom",
        horizontalalignment="right",
        fontweight="bold",
    )

    ax.set_xlabel("OOD Scores: C-PP", fontweight="bold")
    ax.set_ylabel("OOD Scores: C-SP", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="gray", fancybox=False)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = output_dir / f"complementarity_{out_dataset}.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out_path}")


def _threshold_success(id_scores, ood_scores):
    return ood_scores < np.quantile(id_scores, 0.05)


def _plot_tsne_space(x_id, x_ood, labels_ood, success_type, out_dataset, output_dir: Path):
    color_id = "#4169E1"
    color_other = "#F08080"
    color_cpp = "#8B0000"
    color_csp = "#800080"
    color_dfsd = "#FF8C00"

    rng = np.random.default_rng(42)
    ood_idx = rng.choice(len(x_ood), min(len(x_ood), len(x_id) // 2), replace=False)
    x_id_sub = x_id
    x_ood_sub = x_ood[ood_idx]
    labels_sub = labels_ood[ood_idx]

    x_all = np.concatenate([x_id_sub, x_ood_sub], axis=0)
    is_id = np.zeros(len(x_all), dtype=bool)
    is_id[: len(x_id_sub)] = True
    x_2d = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto").fit_transform(x_all)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_2d[is_id, 0], x_2d[is_id, 1], s=8, c=color_id, alpha=0.35, label="ID")
    ood_points = x_2d[len(x_id_sub) :]
    ax.scatter(
        ood_points[labels_sub == 0, 0],
        ood_points[labels_sub == 0, 1],
        s=10,
        c=color_other,
        alpha=0.4,
        label="OOD (Other)",
    )

    if success_type == "C-PP":
        ax.scatter(
            ood_points[labels_sub == 1, 0],
            ood_points[labels_sub == 1, 1],
            s=35,
            c=color_cpp,
            alpha=0.85,
            marker="*",
            label="OOD (C-PP-only)",
        )
        title = "C-PP Space"
    elif success_type == "C-SP":
        ax.scatter(
            ood_points[labels_sub == 2, 0],
            ood_points[labels_sub == 2, 1],
            s=35,
            c=color_csp,
            alpha=0.85,
            marker="*",
            label="OOD (C-SP-only)",
        )
        title = "C-SP Space"
    else:
        ax.scatter(
            ood_points[labels_sub == 3, 0],
            ood_points[labels_sub == 3, 1],
            s=28,
            c=color_dfsd,
            alpha=0.85,
            marker="D",
            label="OOD (DFSD-only)",
        )
        title = "Raw Feature Space"

    ax.set_title(f"{title} - {out_dataset}", fontsize=13)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(fontsize=10, loc="best", markerscale=1.3, framealpha=0.9)
    fig.tight_layout()

    out_path = output_dir / f"{success_type.replace('-', '').lower()}_space_{out_dataset}.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_tsne_spaces(cfg: ExperimentConfig, out_dataset: str, output_dir: str | Path, max_id_batches: int | None = None):
    """Plot C-PP and C-SP t-SNE spaces with DFSD success labels."""
    set_paper_style()
    set_reproducible_state(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = config_as_legacy_args(cfg)
    loader_in, num_classes = get_dataloader(cfg.in_dataset, cfg.batch_size, args)
    model = get_model(args, num_classes).eval().cuda()
    alpha, estimators, u = fit_or_load_subspaces(cfg, model, "kpca", num_classes)

    info = np.load(package_path("features", f"{cfg.in_dataset}_{cfg.model_arch}_feat_stat.npy"))
    contrib = info[None, :] * model.fc.weight.data.cpu().numpy()
    thresh = np.percentile(contrib, cfg.p, axis=1, keepdims=True)
    mask = torch.tensor(contrib > thresh, dtype=torch.float32).cuda()

    def collect(loader, limit_batches=None):
        raw_all, cpp_all, csp_all, c_pp_scores, c_sp_scores, dfsd_scores = [], [], [], [], [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                images = data[0].cuda()
                features, outputs = extract_features_and_logits(model, images, cfg)
                feat_tensor = torch.tensor(features, dtype=torch.float32).cuda()
                raw_all.append(features)
                cpp_all.append((feat_tensor @ mask.mT).cpu().numpy())
                csp = np.zeros((features.shape[0], len(estimators)))
                for i, estimator in enumerate(estimators):
                    reconstructed = estimator.inverse_transform(estimator.transform(features - u))
                    csp[:, i] = norm((features - u) - reconstructed, axis=-1) * alpha[i]
                csp_all.append(csp)
                c_pp_scores.append(score_numpy(features, outputs, alpha, estimators, u, "cpp_dice"))
                c_sp_scores.append(score_numpy(features, outputs, alpha, estimators, u, "csp_kpca"))
                dfsd_scores.append(score_numpy(features, outputs, alpha, estimators, u, "dfsd_main"))
        return (
            np.concatenate(raw_all),
            np.concatenate(cpp_all),
            np.concatenate(csp_all),
            np.concatenate(c_pp_scores),
            np.concatenate(c_sp_scores),
            np.concatenate(dfsd_scores),
        )

    id_raw, id_cpp_feat, id_csp_feat, id_cpp_score, id_csp_score, id_dfsd_score = collect(loader_in, max_id_batches)
    loader_out = get_outdataset(out_dataset, cfg.in_dataset, cfg.batch_size)
    ood_raw, ood_cpp_feat, ood_csp_feat, ood_cpp_score, ood_csp_score, ood_dfsd_score = collect(loader_out)

    cpp_success = _threshold_success(id_cpp_score, ood_cpp_score)
    csp_success = _threshold_success(id_csp_score, ood_csp_score)
    dfsd_success = _threshold_success(id_dfsd_score, ood_dfsd_score)
    labels = np.zeros_like(ood_cpp_score, dtype=int)
    labels[cpp_success & ~csp_success & ~dfsd_success] = 1
    labels[csp_success & ~cpp_success & ~dfsd_success] = 2
    labels[dfsd_success & ~cpp_success & ~csp_success] = 3

    _plot_tsne_space(id_cpp_feat, ood_cpp_feat, labels, "C-PP", out_dataset, output_dir)
    _plot_tsne_space(id_csp_feat, ood_csp_feat, labels, "C-SP", out_dataset, output_dir)
    _plot_tsne_space(id_raw, ood_raw, labels, "Raw", out_dataset, output_dir)


def plot_evr_curves(dataset: str, seed: int, output_dir: str | Path):
    set_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evr_info_path = package_path("ood_checkpoints", dataset, f"kpca_FULL_seed{seed}", "kpca_evr_info.npy")
    if not evr_info_path.exists():
        raise FileNotFoundError(evr_info_path)
    evr_info = np.load(evr_info_path, allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(9, 6))
    first_ratios = []
    for info in evr_info.values():
        lambdas = np.real(info["lambdas"])
        lambdas = lambdas[lambdas > 1e-10]
        evr = lambdas / np.sum(lambdas)
        cum_evr = np.cumsum(evr)
        first_ratios.append(evr[0])
        ax.plot(range(1, len(cum_evr) + 1), cum_evr, lw=0.8, alpha=0.35, color="tab:blue")

    mean_ratio = float(np.mean(first_ratios))
    ax.axhline(y=mean_ratio, color="red", lw=1.2, ls="--", alpha=0.8, label=f"Mean first EVR = {mean_ratio:.2f}")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio (EVR)")
    ax.set_title(f"KPCA cumulative EVR curves ({dataset}, seed={seed})", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    out_path = output_dir / f"kpca_evr_{dataset}_seed{seed}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_bar_line_ablation(spec_path: str | Path, output_dir: str | Path):
    set_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    labels = spec["labels"]
    x = np.arange(len(labels))
    for metric in spec["metrics"]:
        fig, ax = plt.subplots(figsize=tuple(metric.get("figsize", [3.0, 2.4])))
        values = metric["values"]
        ax.bar(x, values, color=metric.get("bar_color", "#A8C8E6"), edgecolor="none", linewidth=0.6)
        ax.plot(
            x,
            values,
            color=metric.get("line_color", "steelblue"),
            marker="o",
            linestyle="--",
            linewidth=1.2,
        )
        if "ylim" in metric:
            ax.set_ylim(*metric["ylim"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric["ylabel"])
        for i, val in enumerate(values):
            ax.text(i, val + metric.get("label_offset", 0.2), f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out_path = output_dir / metric["filename"]
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DFSD paper visualizations.")
    parser.add_argument("--config", default=None, help="DFSD config JSON.")
    parser.add_argument(
        "--figure",
        required=True,
        choices=["complementarity", "tsne_spaces", "evr", "bar_line_ablation"],
    )
    parser.add_argument("--out-dataset", default="SVHN")
    parser.add_argument("--output-dir", default="dfsd_repro/figures")
    parser.add_argument("--dataset", default="CIFAR-10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--spec", default=None, help="Ablation figure spec JSON.")
    parser.add_argument("--max-id-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.figure in {"complementarity", "tsne_spaces"}:
        if args.config is None:
            raise ValueError("--config is required for this figure.")
        cfg = load_config(args.config)
        if args.figure == "complementarity":
            plot_complementarity(cfg, args.out_dataset, args.output_dir)
        else:
            plot_tsne_spaces(cfg, args.out_dataset, args.output_dir, max_id_batches=args.max_id_batches)
    elif args.figure == "evr":
        plot_evr_curves(args.dataset, args.seed, args.output_dir)
    elif args.figure == "bar_line_ablation":
        if args.spec is None:
            raise ValueError("--spec is required for bar_line_ablation.")
        plot_bar_line_ablation(args.spec, args.output_dir)


if __name__ == "__main__":
    main()
