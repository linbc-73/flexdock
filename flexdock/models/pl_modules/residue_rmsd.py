from typing import Any
from functools import partial

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info

from flexdock.models.networks import get_model
from flexdock.sampling.docking.diffusion import t_to_sigma as t_to_sigma_compl


def compute_residue_rmsd_class_weights(
    dataset,
    weight_type: str = "sqrt_inverse",
    max_complexes: int | None = None,
    num_workers: int = 0,
):
    """Estimate per-class weights for residue RMSD classification.

    Samples ``max_complexes`` graphs from ``dataset`` and counts the class
    distribution among ``receptor.nearby_residues``. Returns a list of floats
    that can be passed as ``loss.class_weight``.

    Args:
        dataset: A PyG-style dataset yielding transformed heterographs.
        weight_type: ``"inverse"`` for 1/freq, ``"sqrt_inverse"`` for
            1/sqrt(freq). Any other value raises.
        max_complexes: If given, only sample this many complexes for speed.
        num_workers: Number of workers for temporary DataLoader.

    Returns:
        List of float weights, one per class.
    """
    from torch.utils.data import DataLoader as TorchDataLoader

    rank_zero_info(
        f"Computing residue RMSD class weights (type={weight_type}, "
        f"max_complexes={max_complexes})..."
    )

    indices = list(range(len(dataset)))
    if max_complexes is not None and max_complexes < len(indices):
        import random

        random.seed(42)
        indices = random.sample(indices, max_complexes)

    temp_loader = TorchDataLoader(
        dataset,
        batch_size=1,
        sampler=indices,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )

    class_counts = None
    for graph in temp_loader:
        mask = graph["receptor"].nearby_residues
        if mask.sum() == 0:
            continue
        classes = graph["receptor"].residue_rmsd_class[mask].long()
        max_class = int(classes.max().item())
        if class_counts is None:
            class_counts = torch.zeros(max_class + 1, dtype=torch.float32)
        elif max_class >= len(class_counts):
            old = class_counts
            class_counts = torch.zeros(max_class + 1, dtype=torch.float32)
            class_counts[: len(old)] = old
        class_counts += torch.bincount(classes, minlength=len(class_counts)).float()

    if class_counts is None:
        raise RuntimeError("No residue RMSD class labels found in sampled training data.")

    # Avoid division by zero for missing classes.
    class_counts = torch.clamp(class_counts, min=1.0)
    freq = class_counts / class_counts.sum()

    if weight_type == "inverse":
        weights = 1.0 / freq
    elif weight_type == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(freq)
    else:
        raise ValueError(f"Unknown weight_type={weight_type}")

    # Normalize so the smallest weight is 1.0.
    weights = weights / weights.min()
    weight_list = [float(w) for w in weights.tolist()]
    rank_zero_info(f"Computed class weights: {weight_list}")
    return weight_list


class ResidueRMSDModule(pl.LightningModule):
    """Lightning module for residue-level apo-holo RMSD prediction.

    Reuses the FlexDock docking score network as a feature extractor and adds
    a per-residue head. The head can be either a regression head (single
    positive RMSD value) or a classification head (logits over ordered RMSD
    bins). Only residues flagged as ``receptor.nearby_residues`` are supervised.
    """

    def __init__(
        self,
        model_cfg,
        sigma_cfg,
        training_cfg,
        sampler_cfg=None,
        loss_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.sigma_cfg = sigma_cfg
        self.training_cfg = training_cfg
        self.loss_cfg = loss_cfg or {}

        self.classification = getattr(model_cfg, "residue_rmsd_classification", False)
        self.bins = list(getattr(model_cfg, "residue_rmsd_bins", [])) if self.classification else None
        self.num_classes = len(self.bins) + 1 if self.classification else 1

        # Classification-specific loss options.
        self.label_smoothing = float(getattr(self.loss_cfg, "label_smoothing", 0.0))
        class_weight = getattr(self.loss_cfg, "class_weight", None)
        self.register_buffer("class_weight", self._parse_class_weight(class_weight))

        # Provide a dummy t_to_sigma; time is fixed to 0 for this task.
        t_to_sigma = partial(t_to_sigma_compl, args=sigma_cfg)
        self.model = get_model(
            args=model_cfg,
            t_to_sigma=t_to_sigma,
            confidence_mode=False,
            device=self.device,
        )

    def _parse_class_weight(self, class_weight):
        if class_weight is None or class_weight == "none":
            return None
        # OmegaConf lists are ListConfig; convert them to plain Python lists.
        if hasattr(class_weight, "_content"):
            class_weight = list(class_weight)
        if isinstance(class_weight, (list, tuple)):
            return torch.tensor(class_weight, dtype=torch.float32)
        if isinstance(class_weight, torch.Tensor):
            return class_weight.float()
        if class_weight in ("inverse", "sqrt_inverse"):
            raise ValueError(
                f"class_weight='{class_weight}' must be resolved to a list of floats "
                "before constructing ResidueRMSDModule (done automatically by "
                "scripts/train/train_residue_rmsd.py)."
            )
        raise ValueError(
            f"class_weight must be 'none', 'inverse', 'sqrt_inverse', a list of floats, "
            f"or a tensor, got {class_weight}"
        )

    def forward(self, batch):
        return self.model(batch, fast_updates=True)

    def _compute_loss(self, outputs, batch):
        mask = batch["receptor"].nearby_residues

        if mask.sum() == 0:
            return (
                batch["receptor"].pos.new_tensor(0.0),
                batch["receptor"].pos.new_tensor(0.0),
            )

        if self.classification:
            pred = outputs["residue_rmsd_class_logits"]
            target = batch["receptor"].residue_rmsd_class.long()
            if pred.numel() == 0:
                return pred.new_tensor(0.0), pred.new_tensor(0.0)
            loss = F.cross_entropy(
                pred[mask],
                target[mask],
                weight=self.class_weight,
                label_smoothing=self.label_smoothing,
            )
            accuracy = (pred[mask].argmax(dim=-1) == target[mask]).float().mean()
            return loss, accuracy
        else:
            pred = outputs["residue_rmsd_pred"]
            target = batch["receptor"].residue_rmsd
            if pred.numel() == 0:
                return pred.new_tensor(0.0), pred.new_tensor(0.0)
            loss = F.mse_loss(pred[mask], target[mask])
            mae = F.l1_loss(pred[mask], target[mask])
            return loss, mae

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, metric = self._compute_loss(outputs, batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        metric_name = "train_accuracy" if self.classification else "train_mae"
        self.log(
            metric_name,
            metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, metric = self._compute_loss(outputs, batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        metric_name = "val_accuracy" if self.classification else "val_mae"
        self.log(
            metric_name,
            metric,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )

        mask = batch["receptor"].nearby_residues
        if mask.sum() > 0 and not self.classification:
            pred = outputs["residue_rmsd_pred"]
            target = batch["receptor"].residue_rmsd
            pred_masked = pred[mask]
            target_masked = target[mask]
            correlation = self._pearson_correlation(pred_masked, target_masked)
            self.log(
                "val_pearson",
                correlation,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch.num_graphs,
            )
        return loss

    @staticmethod
    def _pearson_correlation(x, y):
        mean_x = x.mean()
        mean_y = y.mean()
        xm = x - mean_x
        ym = y - mean_y
        r_num = (xm * ym).sum()
        r_den = torch.sqrt((xm**2).sum() * (ym**2).sum())
        return r_num / (r_den + 1e-8)

    def configure_optimizers(self):
        optimizer_cls = (
            torch.optim.AdamW
            if getattr(self.training_cfg, "adamw", False) == "adamw"
            else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.training_cfg.lr),
            weight_decay=getattr(self.training_cfg, "w_decay", 0.0),
        )

        scheduler = None
        scheduler_name = getattr(self.training_cfg, "scheduler", None)
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.7,
                patience=getattr(self.training_cfg, "scheduler_patience", 10),
                min_lr=float(self.training_cfg.lr) / 100,
            )
        elif scheduler_name == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=getattr(self.training_cfg, "scheduler_t_max", 250),
                eta_min=1e-7,
            )

        optim_dict = {"optimizer": optimizer}
        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
                "name": scheduler_name,
            }
        return optim_dict

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Light-weight checkpoint without sampler-specific state.
        pass

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        pass
