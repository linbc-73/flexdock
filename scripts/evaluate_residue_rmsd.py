#!/usr/bin/env python3
"""
Evaluate residue RMSD predictions exported by scripts/predict_residue_rmsd.py.

Expected CSV columns (regression):
  - complex_name
  - chain_id
  - residue_id
  - predicted_rmsd
  - target_rmsd

Expected CSV columns (classification):
  - complex_name
  - chain_id
  - residue_id
  - predicted_class
  - target_class
  - target_rmsd
  - prob_class_0 ... prob_class_4

Example:
  python scripts/evaluate_residue_rmsd.py \
      --input residue_rmsd_predictions.csv \
      --output_json residue_rmsd_eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residue RMSD prediction CSV")
    parser.add_argument("--input", required=True, help="Path to prediction CSV")
    parser.add_argument(
        "--group_by_complex",
        action="store_true",
        help="Also report macro metrics (average over complexes)",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save metrics as JSON",
    )
    return parser.parse_args()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)
    if y_true_std == 0 or y_pred_std == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa for ordinal classification."""
    if y_true.size < 2:
        return float("nan")
    classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(classes)
    if n_classes < 2:
        return float("nan")

    # Build observed confusion matrix (normalized).
    observed = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        observed[t, p] += 1
    observed /= observed.sum()

    # Expected matrix from marginal distributions.
    row_marginals = observed.sum(axis=1, keepdims=True)
    col_marginals = observed.sum(axis=0, keepdims=True)
    expected = row_marginals @ col_marginals

    # Quadratic weights.
    weights = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = np.sum(weights * observed)
    den = np.sum(weights * expected)
    if den == 0:
        return float("nan")
    return float(1.0 - num / den)


def topk_hit(y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: float = 2.0) -> float:
    """
    Fraction of complexes where among top-k lowest predicted residues,
    at least one residue has target RMSD <= threshold.
    """
    if y_true.size == 0:
        return float("nan")
    k_eff = min(k, y_true.size)
    idx = np.argsort(y_pred)[:k_eff]
    return float(np.any(y_true[idx] <= threshold))


def main() -> None:
    args = parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    classification_mode = {"predicted_class", "target_class"}.issubset(df.columns)

    if classification_mode:
        required = {
            "complex_name",
            "chain_id",
            "residue_id",
            "predicted_class",
            "target_class",
            "target_rmsd",
        }
    else:
        required = {
            "complex_name",
            "chain_id",
            "residue_id",
            "predicted_rmsd",
            "target_rmsd",
        }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()

    if classification_mode:
        df["predicted_class"] = pd.to_numeric(df["predicted_class"], errors="coerce").astype(int)
        df["target_class"] = pd.to_numeric(df["target_class"], errors="coerce").astype(int)
        df["target_rmsd"] = pd.to_numeric(df["target_rmsd"], errors="coerce")
        df = df.dropna(subset=["predicted_class", "target_class", "target_rmsd"]).copy()

        y_pred = df["predicted_class"].to_numpy(dtype=np.int64)
        y_true = df["target_class"].to_numpy(dtype=np.int64)

        accuracy = float(np.mean(y_pred == y_true))
        classes = sorted(int(c) for c in set(y_true) | set(y_pred))
        per_class = {}
        for c in classes:
            tp = int(np.sum((y_pred == c) & (y_true == c)))
            fp = int(np.sum((y_pred == c) & (y_true != c)))
            fn = int(np.sum((y_pred != c) & (y_true == c)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[str(c)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(np.sum(y_true == c)),
            }

        macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
        macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))
        macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

        confusion = pd.crosstab(
            pd.Series(y_true, name="target"),
            pd.Series(y_pred, name="predicted"),
        ).to_dict()

        # Ordinal-aware metrics.
        class_mae = float(np.mean(np.abs(y_pred - y_true)))
        adjacent_accuracy = float(np.mean(np.abs(y_pred - y_true) <= 1))
        qwk = quadratic_weighted_kappa(y_true, y_pred)

        metrics = {
            "mode": "classification",
            "n_rows": int(len(df)),
            "n_complexes": int(df["complex_name"].nunique()),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "class_mae": class_mae,
            "adjacent_accuracy": adjacent_accuracy,
            "quadratic_weighted_kappa": qwk,
            "per_class": per_class,
            "confusion_matrix": confusion,
        }

        per_complex = []
        for complex_name, g in df.groupby("complex_name", sort=False):
            gy_true = g["target_class"].to_numpy(dtype=np.int64)
            gy_pred = g["predicted_class"].to_numpy(dtype=np.int64)
            per_complex.append(
                {
                    "complex_name": complex_name,
                    "n_residues": int(len(g)),
                    "accuracy": float(np.mean(gy_pred == gy_true)),
                }
            )

        if args.group_by_complex and per_complex:
            pc = pd.DataFrame(per_complex)
            metrics["macro_accuracy"] = float(pc["accuracy"].mean())

        print_keys = [
            "n_rows",
            "n_complexes",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "class_mae",
            "adjacent_accuracy",
            "quadratic_weighted_kappa",
            "macro_accuracy",
        ]
    else:
        df["predicted_rmsd"] = pd.to_numeric(df["predicted_rmsd"], errors="coerce")
        df["target_rmsd"] = pd.to_numeric(df["target_rmsd"], errors="coerce")
        df = df.dropna(subset=["predicted_rmsd", "target_rmsd"]).copy()

        y_pred = df["predicted_rmsd"].to_numpy(dtype=np.float64)
        y_true = df["target_rmsd"].to_numpy(dtype=np.float64)

        metrics = {
            "mode": "regression",
            "n_rows": int(len(df)),
            "n_complexes": int(df["complex_name"].nunique()),
            "pred_mean": float(np.mean(y_pred)),
            "target_mean": float(np.mean(y_true)),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "pearson": pearsonr(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "frac_abs_err_le_0.5": float(np.mean(np.abs(y_pred - y_true) <= 0.5)),
            "frac_abs_err_le_1.0": float(np.mean(np.abs(y_pred - y_true) <= 1.0)),
            "frac_abs_err_le_2.0": float(np.mean(np.abs(y_pred - y_true) <= 2.0)),
        }

        per_complex = []
        grouped = df.groupby("complex_name", sort=False)
        for complex_name, g in grouped:
            gy_true = g["target_rmsd"].to_numpy(dtype=np.float64)
            gy_pred = g["predicted_rmsd"].to_numpy(dtype=np.float64)
            per_complex.append(
                {
                    "complex_name": complex_name,
                    "n_residues": int(len(g)),
                    "mae": mae(gy_true, gy_pred),
                    "rmse": rmse(gy_true, gy_pred),
                    "pearson": pearsonr(gy_true, gy_pred),
                    "top1_hit_le2": topk_hit(gy_true, gy_pred, k=1, threshold=2.0),
                    "top3_hit_le2": topk_hit(gy_true, gy_pred, k=3, threshold=2.0),
                    "top5_hit_le2": topk_hit(gy_true, gy_pred, k=5, threshold=2.0),
                }
            )

        if args.group_by_complex and per_complex:
            pc = pd.DataFrame(per_complex)
            metrics["macro_mae"] = float(pc["mae"].mean())
            metrics["macro_rmse"] = float(pc["rmse"].mean())
            metrics["macro_pearson"] = float(pc["pearson"].replace([np.inf, -np.inf], np.nan).dropna().mean())
            metrics["macro_top1_hit_le2"] = float(pc["top1_hit_le2"].mean())
            metrics["macro_top3_hit_le2"] = float(pc["top3_hit_le2"].mean())
            metrics["macro_top5_hit_le2"] = float(pc["top5_hit_le2"].mean())

        print_keys = [
            "n_rows",
            "n_complexes",
            "pred_mean",
            "target_mean",
            "mae",
            "rmse",
            "pearson",
            "r2",
            "frac_abs_err_le_0.5",
            "frac_abs_err_le_1.0",
            "frac_abs_err_le_2.0",
            "macro_mae",
            "macro_rmse",
            "macro_pearson",
            "macro_top1_hit_le2",
            "macro_top3_hit_le2",
            "macro_top5_hit_le2",
        ]

    print("=== Residue RMSD Evaluation ===")
    for k in print_keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                if math.isnan(v):
                    print(f"{k}: nan")
                else:
                    print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics JSON to: {out_path}")


if __name__ == "__main__":
    main()
