from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def get_curve(known, novel, method=None):
    known.sort()
    novel.sort()

    all_scores = np.concatenate((known, novel))
    all_scores.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == "row":
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for idx in range(num_k + num_n):
        if k == num_k:
            tp[idx + 1 :] = tp[idx]
            fp[idx + 1 :] = np.arange(fp[idx] - 1, -1, -1)
            break
        if n == num_n:
            tp[idx + 1 :] = np.arange(tp[idx] - 1, -1, -1)
            fp[idx + 1 :] = fp[idx]
            break
        if novel[n] < known[k]:
            n += 1
            tp[idx + 1] = tp[idx]
            fp[idx + 1] = fp[idx] - 1
        else:
            k += 1
            tp[idx + 1] = tp[idx] - 1
            fp[idx + 1] = fp[idx]

    j = num_k + num_n - 1
    for _ in range(num_k + num_n - 1):
        if all_scores[j] == all_scores[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)
    return tp, fp, fpr_at_tpr95


def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = {}

    results["FPR"] = fpr_at_tpr95

    tpr = np.concatenate([[1.0], tp / tp[0], [0.0]])
    fpr = np.concatenate([[1.0], fp / fp[0], [0.0]])
    results["AUROC"] = -np.trapz(1.0 - fpr, tpr)

    results["DTERR"] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    denom = tp + fp
    denom[denom == 0.0] = -1.0
    pin_ind = np.concatenate([[True], denom > 0.0, [True]])
    pin = np.concatenate([[0.5], tp / denom, [0.0]])
    results["AUIN"] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.0] = -1.0
    pout_ind = np.concatenate([[True], denom > 0.0, [True]])
    pout = np.concatenate([[0.0], (fp[0] - fp) / denom, [0.5]])
    results["AUOUT"] = np.trapz(pout[pout_ind], 1.0 - fpr[pout_ind])

    return results


def summarize_score_dir(score_dir: str | Path, out_datasets: list[str], method="kpca"):
    score_dir = Path(score_dir)
    known = np.loadtxt(score_dir / "in_scores.txt")
    rows = []
    for out_dataset in out_datasets:
        novel = np.loadtxt(score_dir / out_dataset / "out_scores.txt")
        metrics = cal_metric(known.copy(), novel.copy(), method)
        rows.append({"dataset": out_dataset, **metrics})
    avg = {k: float(np.mean([row[k] for row in rows])) for k in ["FPR", "DTERR", "AUROC", "AUIN", "AUOUT"]}
    rows.append({"dataset": "AVG", **avg})
    return rows


def write_metrics_csv(rows, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "FPR", "AUROC", "AUIN", "DTERR", "AUOUT"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "FPR": f"{100 * row['FPR']:.4f}",
                    "AUROC": f"{100 * row['AUROC']:.4f}",
                    "AUIN": f"{100 * row['AUIN']:.4f}",
                    "DTERR": f"{100 * row['DTERR']:.4f}",
                    "AUOUT": f"{100 * row['AUOUT']:.4f}",
                }
            )

