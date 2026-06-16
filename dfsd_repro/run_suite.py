from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_CONFIGS = [
    "dfsd_repro/configs/cifar10_dfsd.json",
    "dfsd_repro/configs/cifar100_dfsd.json",
    "dfsd_repro/configs/imagenet_dfsd.json",
]


def run_command(command: list[str]):
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DFSD reproduction configs.")
    parser.add_argument("--config", action="append", default=None, help="Config JSON. Defaults to CIFAR-10/100/ImageNet.")
    parser.add_argument("--modes", nargs="*", default=None, help="Override modes for all configs.")
    parser.add_argument("--c-pp", choices=["dice", "ash", "scale"], default=None, help="Override C-PP component.")
    parser.add_argument("--metrics-only", action="store_true", help="Summarize existing score files only.")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = args.config or DEFAULT_CONFIGS

    for config_path in configs:
        command = [sys.executable, "-m", "dfsd_repro.run", "--config", config_path]
        if args.modes:
            command.extend(["--modes", *args.modes])
        if args.c_pp:
            command.extend(["--c-pp", args.c_pp])
        if args.metrics_only:
            command.append("--metrics-only")
        run_command(command)


if __name__ == "__main__":
    main()
