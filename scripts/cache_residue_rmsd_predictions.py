"""Cache residue-level RMSD predictions back into heterograph cache files.

Loads a trained ResidueRMSDModule checkpoint, runs inference on each cached
heterograph, and writes the predicted per-residue RMSD into the
``receptor.residue_rmsd_pred`` field of the original (full) graph.

Already-processed graphs (those that already contain ``residue_rmsd_pred``)
are skipped, so the script can be restarted safely.

Example:
    python scripts/cache_residue_rmsd_predictions.py \
        --config configs/residue_rmsd/residue_rmsd.yaml \
        --checkpoint train_log/residue_rmsd/residue-rmsd-baseline/best_model.pt \
        --source_cache data/pdbbind_train/cache \
        --output_cache data/pdbbind_train/cache_with_pred \
        --split data/pdbbind_train_split.txt
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from flexdock.data.feature.protein import get_binding_pocket_masks
from flexdock.data.parse.base import read_strings_from_txt
from flexdock.data.transforms.docking import construct_transform
from flexdock.models.pl_modules.residue_rmsd import ResidueRMSDModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the residue_rmsd config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the ResidueRMSDModule checkpoint (.pt / .ckpt).",
    )
    parser.add_argument(
        "--source_cache",
        required=True,
        help="Directory containing heterograph-{name}.pt files.",
    )
    parser.add_argument(
        "--output_cache",
        required=True,
        help="Directory where updated heterograph files will be written. "
             "Can be the same as --source_cache for in-place update.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Text file with one complex name per line. "
             "If None, all heterograph-*.pt files in source_cache are processed.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N complexes (useful for testing).",
    )
    return parser.parse_args()


def maybe_remove_collate_unfriendly_attrs(graph):
    """Remove attributes that can break PyG collate / Batch construction."""
    for key in ("mol", "rmsd_matching"):
        if hasattr(graph, key):
            delattr(graph, key)
        if key in graph:
            del graph[key]
    return graph


def compute_pocket_res_mask(full_graph):
    """Compute the full-length boolean pocket mask used by PocketTransform."""
    # The same defaults as PocketTransform.compute_pocket.
    _, pocket_res_mask, _, _ = get_binding_pocket_masks(
        atom_pos=full_graph["atom"].orig_apo_pos,
        ref_atom_pos=full_graph["atom"].orig_holo_pos,
        lig_pos=full_graph["ligand"].pos,
        ca_mask=full_graph["atom"].ca_mask,
        atom_rec_index=full_graph["atom", "atom_rec_contact", "receptor"].edge_index[1],
        pocket_cutoff=5.0,
        pocket_buffer=20.0,
        pocket_min_size=1,
    )
    if pocket_res_mask.dtype != torch.bool:
        pocket_res_mask = pocket_res_mask > 0
    return pocket_res_mask


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    os.makedirs(args.output_cache, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print(f"Loading checkpoint from {args.checkpoint}", flush=True)
    model = ResidueRMSDModule.load_from_checkpoint(
        args.checkpoint,
        model_cfg=cfg.model,
        sigma_cfg=cfg.sigma,
        training_cfg=cfg.training,
    )
    model.to(device)
    model.eval()

    transform = construct_transform(
        cfg=cfg.transforms, mode="val", task="residue_rmsd"
    )

    if args.split is not None:
        names = read_strings_from_txt(args.split)
    else:
        names = [
            fname.replace("heterograph-", "").replace(".pt", "")
            for fname in os.listdir(args.source_cache)
            if fname.startswith("heterograph-") and fname.endswith(".pt")
        ]

    names = [
        name
        for name in names
        if os.path.exists(os.path.join(args.source_cache, f"heterograph-{name}.pt"))
    ]

    if args.limit is not None:
        names = names[: args.limit]

    print(
        f"Found {len(names)} complexes to process in {args.source_cache}",
        flush=True,
    )

    processed = skipped = failed = 0

    with torch.no_grad():
        for name in tqdm(names, desc="Predicting residue RMSD"):
            src_path = os.path.join(args.source_cache, f"heterograph-{name}.pt")
            dst_path = os.path.join(args.output_cache, f"heterograph-{name}.pt")

            # Skip if the output already has the predicted field.
            if os.path.exists(dst_path):
                existing = torch.load(dst_path, map_location="cpu")
                if hasattr(existing["receptor"], "residue_rmsd_pred"):
                    skipped += 1
                    continue
                full_graph = existing
            else:
                full_graph = torch.load(src_path, map_location="cpu")

            full_graph = maybe_remove_collate_unfriendly_attrs(full_graph)

            try:
                # Run the residue_rmsd transform on a clone so the full graph
                # stays intact for mapping predictions back to full residue length.
                model_input = transform(full_graph.clone())
                if model_input is None:
                    raise ValueError("Transform returned None")

                model_input = model_input.to(device)
                batch = Batch.from_data_list([model_input])
                outputs = model(batch)
                pred = outputs["residue_rmsd_pred"].detach().cpu()

                # Map pocket-level predictions back to the full receptor length.
                n_res_full = full_graph["receptor"].x.shape[0]
                full_pred = torch.zeros(n_res_full, dtype=pred.dtype)

                pocket_res_mask = compute_pocket_res_mask(full_graph)
                n_pocket = int(pocket_res_mask.sum())

                if pred.numel() != n_pocket:
                    raise ValueError(
                        f"Prediction size mismatch for {name}: "
                        f"pred={pred.numel()}, pocket_residues={n_pocket}"
                    )

                full_pred[pocket_res_mask] = pred
                full_graph["receptor"].residue_rmsd_pred = full_pred

                torch.save(full_graph, dst_path)
                processed += 1

            except Exception as e:
                failed += 1
                print(f"Failed on {name}: {e}", flush=True)

    print(
        f"Done: processed={processed}, skipped={skipped}, failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
