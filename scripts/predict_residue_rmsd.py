"""Inference script for residue-level apo-holo RMSD prediction.

Loads a trained ResidueRMSDModule checkpoint and evaluates it on a cached
heterograph split. In regression mode the CSV contains predicted and target
RMSD values; in classification mode it contains predicted/target class labels,
the continuous target RMSD, and per-class probabilities. Only residues flagged
as ``nearby_residues`` are written.

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
        "--cache_path",
        default=None,
        help="Cache directory (defaults to cfg.data.cache_path)",
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
    parser.add_argument(
        "--soft",
        action="store_true",
        help="For classification, use the expected class (rounded probability "
             "distribution) instead of argmax. Often improves ordinal accuracy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    split_path = args.split if args.split is not None else cfg.data.split_val
    cache_path = args.cache_path if args.cache_path is not None else cfg.data.cache_path
    batch_size = args.batch_size if args.batch_size is not None else cfg.data.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank_zero_info = lambda msg: print(msg, flush=True)  # noqa: E731
    rank_zero_info(f"Loading checkpoint from {args.checkpoint}")

    # Use the checkpoint's own model architecture hyperparameters so that the
    # loaded weights match the constructed network. The config file is still
    # used for data paths and transforms.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_hparams = checkpoint.get("hyper_parameters", {})
    ckpt_model_cfg = ckpt_hparams.get("model_cfg", cfg.model)

    model = ResidueRMSDModule.load_from_checkpoint(
        args.checkpoint,
        model_cfg=ckpt_model_cfg,
        sigma_cfg=cfg.sigma,
        training_cfg=cfg.training,
    )
    model.to(device)
    model.eval()

    classification = getattr(model, "classification", False)
    num_classes = getattr(model, "num_classes", 1)

    # Keep the data transform consistent with the model's task mode.
    ckpt_classification = getattr(ckpt_model_cfg, "residue_rmsd_classification", False)
    ckpt_bins = list(getattr(ckpt_model_cfg, "residue_rmsd_bins", [])) if ckpt_classification else []
    cfg.transforms.residue_rmsd_classification = ckpt_classification
    cfg.transforms.residue_rmsd_bins = ckpt_bins

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

            mask = batch["receptor"].nearby_residues.cpu()
            batch_idx = batch["receptor"].batch.cpu()
            names = batch.name  # list of complex names
            chain_idx = batch["receptor"].chain_idx.cpu()
            residue_number = batch["receptor"].residue_number.cpu()
            chain_letters = getattr(batch, "chain_letters", None)

            if classification:
                logits = outputs["residue_rmsd_class_logits"].cpu()
                probs = torch.softmax(logits, dim=-1)
                target_class = batch["receptor"].residue_rmsd_class.cpu()
                target_rmsd = batch["receptor"].residue_rmsd.cpu()
                if args.soft:
                    # Expected (mean) class from the predicted distribution.
                    classes = torch.arange(num_classes, dtype=torch.float32)
                    pred_class = torch.round(
                        (probs * classes).sum(dim=-1)
                    ).clamp(min=0, max=num_classes - 1).long()
                else:
                    pred_class = logits.argmax(dim=-1)
            else:
                pred = outputs["residue_rmsd_pred"].cpu()
                target = batch["receptor"].residue_rmsd.cpu()

            # If a complex has no nearby residues in this batch, skip cheaply.
            batch_rows = []
            for i in range(len(mask)):
                if not mask[i]:
                    continue
                graph_idx = batch_idx[i].item()
                complex_name = names[graph_idx]
                if isinstance(complex_name, list):
                    complex_name = complex_name[0]

                cidx = chain_idx[i].item()
                if (
                    chain_letters is not None
                    and graph_idx < len(chain_letters)
                    and cidx < len(chain_letters[graph_idx])
                ):
                    chain_id = chain_letters[graph_idx][cidx]
                else:
                    chain_id = str(cidx)
                residue_id = int(residue_number[i].item())

                if classification:
                    row = {
                        "complex_name": complex_name,
                        "chain_id": chain_id,
                        "residue_id": residue_id,
                        "predicted_class": int(pred_class[i].item()),
                        "target_class": int(target_class[i].item()),
                        "target_rmsd": f"{target_rmsd[i].item():.4f}",
                    }
                    for c in range(num_classes):
                        row[f"prob_class_{c}"] = f"{probs[i, c].item():.4f}"
                else:
                    row = {
                        "complex_name": complex_name,
                        "chain_id": chain_id,
                        "residue_id": residue_id,
                        "predicted_rmsd": f"{pred[i].item():.4f}",
                        "target_rmsd": f"{target[i].item():.4f}",
                    }
                batch_rows.append(row)
            rows.extend(batch_rows)
            pbar.set_postfix(
                {
                    "rows": len(rows),
                    "per_batch": len(batch_rows),
                }
            )

    if classification:
        fieldnames = [
            "complex_name",
            "chain_id",
            "residue_id",
            "predicted_class",
            "target_class",
            "target_rmsd",
        ] + [f"prob_class_{c}" for c in range(num_classes)]
    else:
        fieldnames = [
            "complex_name",
            "chain_id",
            "residue_id",
            "predicted_rmsd",
            "target_rmsd",
        ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rank_zero_info(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
