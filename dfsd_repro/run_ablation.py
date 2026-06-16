from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]):
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed DFSD component ablations.")
    parser.add_argument("--config", required=True, help="Dataset config JSON.")
    parser.add_argument("--metrics-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    for c_pp in ["dice", "ash", "scale"]:
        command = [
            sys.executable,
            "-m",
            "dfsd_repro.run",
            "--config",
            args.config,
            "--c-pp",
            c_pp,
            "--modes",
            "dfsd_main",
            "cpp_dice",
        ]
        if args.metrics_only:
            command.append("--metrics-only")
        run(command)

    command = [
        sys.executable,
        "-m",
        "dfsd_repro.run",
        "--config",
        args.config,
        "--modes",
        "dfsd_main_pca",
        "csp_kpca",
        "csp_pca",
    ]
    if args.metrics_only:
        command.append("--metrics-only")
    run(command)


if __name__ == "__main__":
    main()
