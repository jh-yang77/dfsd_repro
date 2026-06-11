from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_OOD = {
    "CIFAR-10": ["SVHN", "LSUN", "LSUN_resize", "iSUN", "dtd", "places365"],
    "CIFAR-100": ["SVHN", "LSUN", "LSUN_resize", "iSUN", "dtd", "places365"],
    "imagenet": ["dtd", "sun", "inat", "places"],
}

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_path(*parts: str) -> Path:
    return PACKAGE_ROOT.joinpath(*parts)


@dataclass
class ExperimentConfig:
    experiment_name: str
    in_dataset: str
    model_arch: str = "densenet"
    model_name: str = "densenet"
    p: int | None = 90
    dim: int = 250
    seed: int = 1
    gpu: str = "0"
    batch_size: int = 224
    epochs: int = 100
    layers: int = 100
    depth: int = 40
    width: int = 4
    clip_threshold: float = 1.0
    base_output_dir: str = "results"
    subspace_cache_dir: str = "cache/subspaces"
    out_datasets: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=lambda: ["dfsd_main"])
    fit_class_fraction: float = 1.0
    score_class_fraction: float = 1.0
    force_refit: bool = False
    num_workers: int = 2
    c_pp: str = "dice"
    ash_percentile: int = 90
    scale_percentile: int = 90
    imagenet_train_samples_per_class: int = 100
    pca_components: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls(**raw)
        if not cfg.out_datasets:
            cfg.out_datasets = list(DEFAULT_OOD[cfg.in_dataset])
        if cfg.c_pp not in {"dice", "ash", "scale"}:
            raise ValueError(f"Unsupported c_pp={cfg.c_pp!r}; use dice, ash, or scale.")
        return cfg

    @property
    def output_root(self) -> Path:
        return Path(self.base_output_dir) / self.experiment_name

    def score_dir(self, mode: str) -> Path:
        mode = {
            "dfsd": "dfsd_main",
            "dfsd_kpca": "dfsd_main",
            "dfsd_pca": "dfsd_main_pca",
            "c_pp": "cpp_dice",
            "cpp": "cpp_dice",
            "c_sp_kpca": "csp_kpca",
            "c_sp_pca": "csp_pca",
        }.get(mode, mode)
        if mode == "csp_pca" and self.pca_components is not None:
            return self.output_root / f"{mode}_components{self.pca_components}"
        if mode.startswith("csp"):
            return self.output_root / mode
        if mode == "cpp_dice":
            return self.output_root / (mode if self.c_pp == "dice" else f"cpp_{self.c_pp}")
        if mode == "dfsd_main":
            return self.output_root / (mode if self.c_pp == "dice" else f"{mode}_{self.c_pp}")
        if mode == "dfsd_main_pca" and self.pca_components is not None:
            return self.output_root / f"{mode}_components{self.pca_components}"
        return self.output_root / mode


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        return ExperimentConfig.from_dict(json.load(f))


def config_as_legacy_args(cfg: ExperimentConfig):
    class Args:
        pass

    args = Args()
    args.in_dataset = cfg.in_dataset
    args.name = cfg.model_name
    args.model_arch = cfg.model_arch
    args.p = cfg.p
    args.dim = cfg.dim
    args.gpu = cfg.gpu
    args.use_nng = False
    args.epochs = cfg.epochs
    args.layers = cfg.layers
    args.depth = cfg.depth
    args.width = cfg.width
    args.clip_threshold = cfg.clip_threshold
    args.batch_size = cfg.batch_size
    args.base_dir = cfg.base_output_dir
    args.method = "kpca"
    args.method_args = {}
    args.c_pp = cfg.c_pp
    args.ash_percentile = cfg.ash_percentile
    args.scale_percentile = cfg.scale_percentile
    args.imagenet_train_samples_per_class = cfg.imagenet_train_samples_per_class
    args.pca_components = cfg.pca_components
    return args
