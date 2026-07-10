"""Inference script for residue-level apo-holo RMSD regression.

Loads a trained ResidueRMSDModule checkpoint and evaluates it on a cached
heterograph split. Writes a CSV with per-residue predictions and ground-truth
RMSD values (only for residues flagged as ``nearby_residues``).

Example:
    python scripts/predict_residue_rmsd.py \
        --config configs/residue_rmsd/residue_rmsd.yaml \
        --checkpoint train_log/residue_rmsd/residue-rmsd-baseline/best_model.pt \
        --split data/pdbbind_test_split.txt \
        --output residue_rmsd_predictions.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

from flexdock.data.transforms.docking import construct_transform
from flexdock.data.parse.base import read_strings_from_txt
from flexdock.models.pl_modules.residue_rmsd import ResidueRMSDModule


class CachedResidueRMSDDataset(Dataset):
    def __init__(self, cache_path, complex_names, transform=None):
        super().__init__(root=None, transform=transform)
        self.cache_path = cache_path
        self.complex_names = complex_names
        self.transform = transform

    def len(self):
        return len(self.complex_names)

    def get(self, idx):
        name = self.complex_names[idx]
        graph = torch.load(f"{self.cache_path}/heterograph-{name}.pt")
        if hasattr(graph, "mol"):
            delattr(graph, "mol")
        if "mol" in graph:
            del graph["mol"]
        if hasattr(graph, "rmsd_matching"):
            delattr(graph, "rmsd_matching")
        if "rmsd_matching" in graph:
            del graph["rmsd_matching"]
        if self.transform is not None:
            graph = self.transform(graph)
        # Store name as a list-like attribute so PyG collate keeps it.
        graph.name = [name]
        return graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to residue_rmsd config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--split",
        default=None,
        help="Test split file (defaults to cfg.data.split_val)",
    )
    parser.add_argument(
        "--output",
        default="residue_rmsd_predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (defaults to cfg.data.batch_size)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N complexes (for testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    split_path = args.split if args.split is not None else cfg.data.split_val
    cache_path = cfg.data.cache_path
    batch_size = args.batch_size if args.batch_size is not None else cfg.data.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank_zero_info = lambda msg: print(msg, flush=True)  # noqa: E731
    rank_zero_info(f"Loading checkpoint from {args.checkpoint}")
    model = ResidueRMSDModule.load_from_checkpoint(
        args.checkpoint,
        model_cfg=cfg.model,
        sigma_cfg=cfg.sigma,
        training_cfg=cfg.training,
    )
    model.to(device)
    model.eval()

    transform = construct_transform(cfg=cfg.transforms, mode="val", task="residue_rmsd")

    complex_names = read_strings_from_txt(split_path)
    available_names = [
        name
        for name in complex_names
        if os.path.exists(f"{cache_path}/heterograph-{name}.pt")
    ]
    if args.limit is not None:
        available_names = available_names[: args.limit]
    rank_zero_info(
        f"Found {len(available_names)}/{len(complex_names)} cached complexes"
    )

    dataset = CachedResidueRMSDDataset(
        cache_path=cache_path,
        complex_names=available_names,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    from tqdm import tqdm

    rows = []
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Residue RMSD inference")
        for batch_idx_global, batch in pbar:
            batch = batch.to(device)
            outputs = model(batch)
            pred = outputs["residue_rmsd_pred"].cpu()
            target = batch["receptor"].residue_rmsd.cpu()
            mask = batch["receptor"].nearby_residues.cpu()
            batch_idx = batch["receptor"].batch.cpu()
            names = batch.name  # list of complex names

            # If a complex has no nearby residues in this batch, skip cheaply.
            batch_rows = []
            for i in range(len(mask)):
                if not mask[i]:
                    continue
                complex_name = names[batch_idx[i].item()]
                if isinstance(complex_name, list):
                    complex_name = complex_name[0]
                # Per-complex residue index (after pocket reduction).
                batch_rows.append(
                    {
                        "complex_name": complex_name,
                        "batch_residue_idx": i,
                        "predicted_rmsd": f"{pred[i].item():.4f}",
                        "target_rmsd": f"{target[i].item():.4f}",
                    }
                )
            rows.extend(batch_rows)
            pbar.set_postfix(
                {
                    "rows": len(rows),
                    "per_batch": len(batch_rows),
                }
            )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "complex_name",
                "batch_residue_idx",
                "predicted_rmsd",
                "target_rmsd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rank_zero_info(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
