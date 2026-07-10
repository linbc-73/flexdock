from typing import Any
from functools import partial

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info

from flexdock.models.networks import get_model
from flexdock.sampling.docking.diffusion import t_to_sigma as t_to_sigma_compl


class ResidueRMSDModule(pl.LightningModule):
    """Lightning module for residue-level apo-holo RMSD regression.

    Reuses the FlexDock docking score network as a feature extractor and adds
    a per-residue regression head. Only residues flagged as
    ``receptor.nearby_residues`` are supervised.
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

        # Provide a dummy t_to_sigma; time is fixed to 0 for this task.
        t_to_sigma = partial(t_to_sigma_compl, args=sigma_cfg)
        self.model = get_model(
            args=model_cfg,
            t_to_sigma=t_to_sigma,
            confidence_mode=False,
            device=self.device,
        )

    def forward(self, batch):
        return self.model(batch, fast_updates=True)

    def _compute_loss(self, outputs, batch):
        pred = outputs["residue_rmsd_pred"]
        target = batch["receptor"].residue_rmsd
        mask = batch["receptor"].nearby_residues

        if pred.numel() == 0:
            return pred.new_tensor(0.0), pred.new_tensor(0.0)

        if mask.sum() == 0:
            return pred.new_tensor(0.0), pred.new_tensor(0.0)

        loss = F.mse_loss(pred[mask], target[mask])
        mae = F.l1_loss(pred[mask], target[mask])
        return loss, mae

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, mae = self._compute_loss(outputs, batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "train_mae",
            mae,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, mae = self._compute_loss(outputs, batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "val_mae",
            mae,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )

        pred = outputs["residue_rmsd_pred"]
        target = batch["receptor"].residue_rmsd
        mask = batch["receptor"].nearby_residues
        if mask.sum() > 0:
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
