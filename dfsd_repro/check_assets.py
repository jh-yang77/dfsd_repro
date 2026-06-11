from __future__ import annotations

import argparse
from pathlib import Path

from dfsd_repro.config import ExperimentConfig, load_config, package_path


DATASET_URLS = {
    "CIFAR-10": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "CIFAR-100": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "SVHN": "http://ufldl.stanford.edu/housenumbers/",
    "DTD": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
    "Places365": "http://places2.csail.mit.edu/download.html",
    "LSUN": "https://www.yf.io/p/lsun",
    "iSUN": "https://github.com/facebookresearch/odin",
    "iNaturalist/SUN/Places/OpenImage-O": "https://github.com/deeplearning-wisc/large_scale_ood",
    "ImageNet-1K": "https://image-net.org/download.php",
}


def _paths_for(cfg: ExperimentConfig) -> list[Path]:
    paths: list[Path] = []
    if cfg.in_dataset in {"CIFAR-10", "CIFAR-100"} and cfg.model_arch == "densenet":
        paths.extend(
            [
                package_path("checkpoints", cfg.in_dataset, "densenet", f"checkpoint_{cfg.epochs}.pth.tar"),
                package_path("id_datasets", cfg.in_dataset),
            ]
        )
    elif cfg.in_dataset == "imagenet" and cfg.model_arch == "resnet50":
        paths.extend(
            [
                package_path("id_datasets", "imagenet", "val"),
                package_path("id_datasets", "imagenet", "train"),
            ]
        )
    else:
        raise ValueError(f"No asset checklist for {cfg.in_dataset}/{cfg.model_arch}")

    for out_dataset in cfg.out_datasets:
        if out_dataset == "SVHN":
            paths.append(package_path("ood_datasets", "svhn"))
        elif out_dataset == "dtd":
            paths.append(package_path("ood_datasets", "dtd", "images"))
        elif out_dataset == "places365":
            paths.append(package_path("ood_datasets", "places365"))
        elif out_dataset == "LSUN":
            paths.append(package_path("ood_datasets", "LSUN"))
        elif out_dataset == "LSUN_resize":
            paths.append(package_path("ood_datasets", "LSUN_resize"))
        elif out_dataset == "iSUN":
            paths.append(package_path("ood_datasets", "iSUN"))
        elif out_dataset == "inat":
            paths.append(package_path("ood_datasets", "iNaturalist"))
        elif out_dataset == "places":
            paths.append(package_path("ood_datasets", "Places"))
        elif out_dataset == "sun":
            paths.append(package_path("ood_datasets", "SUN"))
    return paths


def check_config(cfg: ExperimentConfig) -> bool:
    missing = [path for path in _paths_for(cfg) if not path.exists()]
    if not missing:
        print(f"[ok] {cfg.experiment_name}: all required assets are present")
        return True

    print(f"[missing] {cfg.experiment_name}")
    for path in missing:
        print(f"  - {path}")
    print("\nDownload/source references:")
    for name, url in DATASET_URLS.items():
        print(f"  - {name}: {url}")
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Check DFSD data/model assets before running experiments.")
    parser.add_argument("--config", action="append", required=True, help="Config JSON. Can be passed multiple times.")
    return parser.parse_args()


def main():
    args = parse_args()
    ok = True
    for config_path in args.config:
        ok = check_config(load_config(config_path)) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
