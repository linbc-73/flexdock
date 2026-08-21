"""Attach external residue-RMSD predictions to cached heterographs.

This script reads a CSV produced by ``scripts/predict_residue_rmsd.py`` and
writes a new cache directory where each ``heterograph-*.pt`` file has an
additional ``receptor.residue_rmsd_class`` attribute. The docking inference
code can then use these predicted classes for ``bb_sigma_mode=class`` and
``sc_sigma_mode=class`` even when the docking model itself was not trained
with a residue-RMSD head.

Example:
    python scripts/preprocess/attach_residue_rmsd_predictions.py \
        --input residue_rmsd_predictions.csv \
        --source_cache /path/to/original/cache \
        --target_cache /path/to/cache_with_pred_classes \
        --default_class 1
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file from scripts/predict_residue_rmsd.py (classification mode).",
    )
    parser.add_argument(
        "--source_cache",
        required=True,
        help="Original cache directory containing heterograph-*.pt files.",
    )
    parser.add_argument(
        "--target_cache",
        required=True,
        help="Output cache directory to write modified graphs.",
    )
    parser.add_argument(
        "--default_class",
        type=int,
        default=1,
        help="Class index to use for residues not present in the CSV (default: 1).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of flexibility classes (default: 3).",
    )
    return parser.parse_args()


def read_csv_predictions(csv_path):
    """Read CSV and return dict: {(complex_name, chain_id, residue_id): class}."""
    predictions = defaultdict(dict)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            complex_name = row["complex_name"]
            chain_id = row["chain_id"]
            residue_id = int(row["residue_id"])
            predicted_class = int(row["predicted_class"])
            predictions[complex_name][(chain_id, residue_id)] = predicted_class
    return predictions


def chain_id_to_str(chain_id):
    """CSV chain_id may be a letter or a numeric string."""
    return str(chain_id).strip()


def attach_predictions_to_graph(graph, pred_dict, default_class, num_classes):
    """Attach ``receptor.residue_rmsd_class`` to a single graph."""
    num_residues = graph["receptor"].x.shape[0]
    classes = torch.full(
        (num_residues,),
        fill_value=default_class,
        dtype=torch.long,
    )

    if not hasattr(graph["receptor"], "chain_idx") or not hasattr(
        graph["receptor"], "residue_number"
    ):
        raise ValueError(
            "Graph receptor nodes must have 'chain_idx' and 'residue_number' "
            "attributes to map CSV predictions."
        )

    chain_idx = graph["receptor"].chain_idx
    residue_number = graph["receptor"].residue_number
    chain_letters = getattr(graph, "chain_letters", None)

    if chain_idx.numel() != num_residues or residue_number.numel() != num_residues:
        raise ValueError(
            f"chain_idx/residue_number length mismatch: "
            f"{chain_idx.numel()}/{residue_number.numel()} vs {num_residues}"
        )

    for i in range(num_residues):
        cidx = int(chain_idx[i].item())
        resnum = int(residue_number[i].item())

        # Try letter-based chain id first, then numeric index.
        key_letter = None
        if chain_letters is not None and cidx < len(chain_letters):
            key_letter = (chain_letters[cidx].strip(), resnum)
        key_numeric = (str(cidx), resnum)

        pred_class = None
        if key_letter is not None and key_letter in pred_dict:
            pred_class = pred_dict[key_letter]
        elif key_numeric in pred_dict:
            pred_class = pred_dict[key_numeric]

        if pred_class is not None:
            classes[i] = max(0, min(num_classes - 1, pred_class))

    graph["receptor"].residue_rmsd_class = classes
    return graph


def main():
    args = parse_args()

    os.makedirs(args.target_cache, exist_ok=True)
    predictions = read_csv_predictions(args.input)

    source_files = [
        f
        for f in os.listdir(args.source_cache)
        if f.startswith("heterograph-") and f.endswith(".pt")
    ]

    processed = 0
    skipped = 0
    for fname in sorted(source_files):
        complex_name = fname[len("heterograph-") : -len(".pt")]
        source_path = os.path.join(args.source_cache, fname)
        target_path = os.path.join(args.target_cache, fname)

        if complex_name not in predictions:
            # Copy unchanged if no predictions available.
            graph = torch.load(source_path, weights_only=False)
            torch.save(graph, target_path)
            skipped += 1
            continue

        graph = torch.load(source_path, weights_only=False)
        try:
            graph = attach_predictions_to_graph(
                graph,
                predictions[complex_name],
                args.default_class,
                args.num_classes,
            )
            torch.save(graph, target_path)
            processed += 1
        except ValueError as e:
            print(f"Skipping {complex_name}: {e}", flush=True)
            skipped += 1

    print(
        f"Done: {processed} graphs updated, {skipped} copied/skipped. "
        f"Output: {args.target_cache}",
        flush=True,
    )


if __name__ == "__main__":
    main()
