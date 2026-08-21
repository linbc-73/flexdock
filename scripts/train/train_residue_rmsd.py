"""Training entry point for residue-level apo-holo RMSD regression.

This script mirrors ``scripts/train/train_config.py`` but defaults to the
residue_rmsd config. You can also invoke the generic config trainer directly::

    python scripts/train/train_config.py configs/residue_rmsd/residue_rmsd.yaml

Usage:
    python scripts/train/train_residue_rmsd.py [CONFIG_FILE] [OVERRIDES...]
"""

import sys
import os
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from omegaconf import OmegaConf
import torch
import logging

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger, Logger
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.plugins.precision import FSDPPrecision

from flexdock.data.parse.base import save_config
from flexdock.data.modules.training import setup_training_datamodule
from flexdock.models.pl_modules import setup_model
from flexdock.models.pl_modules.residue_rmsd import compute_residue_rmsd_class_weights
from flexdock.models.layers.tensor_product import TensorProductConvLayer
from flexdock.utils.callbacks import setup_callbacks


def setup_strategy(cfg):
    strategy_str = cfg.type.lower()

    if strategy_str == "auto":
        rank_zero_info("INFO: Strategy automatically selected by lightning: pl_strategy='auto'")
        return "auto"

    if not torch.cuda.is_available():
        return DDPStrategy(find_unused_parameters=True)

    SHARDING_STRATEGY = {
        "full": "FULL_SHARD",
        "hybrid": "HYBRID_SHARD",
        "none": "NO_SHARD",
        "grad": "SHARD_GRAD_OP",
    }

    strategy_kwargs = {}
    if strategy_str == "ddp":
        rank_zero_info("DDP: pl_strategy=DDPStrategy(find_unused_parameters=True)")
        return DDPStrategy(find_unused_parameters=True)
    else:
        rank_zero_info("INFO: Option 0: pl_strategy = FSDPStrategy(sharding_strategy=...)")
        strategy_kwargs["sharding_strategy"] = SHARDING_STRATEGY.get(cfg.sharding_strategy)

    if "awp" in strategy_str:
        rank_zero_info("INFO: FSDP - Auto-Wrap-Policy")
        strategy_kwargs["auto_wrap_policy"] = {TensorProductConvLayer}

    if "ac" in strategy_str:
        rank_zero_info("INFO: FSDP - Activation Checkpointing")
        strategy_kwargs["activation_checkpointing_policy"] = {TensorProductConvLayer}

    precision = cfg.get("precision", None)
    if precision is not None:
        rank_zero_info(f"Precision={precision}")
        strategy_kwargs["precision_plugin"] = FSDPPrecision(precision=precision)

    return FSDPStrategy(**strategy_kwargs)


def setup_logger(cfg) -> Optional[Logger]:
    logger_cfg = cfg.logger
    if logger_cfg.wandb:
        return WandbLogger(
            entity=logger_cfg.entity,
            project=logger_cfg.project,
            name=logger_cfg.name,
            tags=logger_cfg.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    return None


def main(config_file, args):
    assert os.path.exists(config_file), f"Config file not found: {config_file}"
    raw_config = OmegaConf.load(config_file)

    args = OmegaConf.from_dotlist(args)
    cfg = OmegaConf.merge(raw_config, args)
    OmegaConf.resolve(cfg)

    logging.getLogger().setLevel("INFO")

    rank_zero_info(f"Running with seed {cfg.seed}")
    seed_everything(cfg.seed)

    data_module = setup_training_datamodule(data_cfg=cfg.data, transform_cfg=cfg.transforms)

    # For residue RMSD classification, auto-compute class weights if requested.
    if cfg.data.task == "residue_rmsd" and getattr(cfg.model, "residue_rmsd_classification", False):
        loss_cfg = cfg.get("loss", OmegaConf.create({}))
        class_weight = getattr(loss_cfg, "class_weight", None)
        if class_weight in ("inverse", "sqrt_inverse"):
            weight_list = compute_residue_rmsd_class_weights(
                dataset=data_module._train_dataset,
                weight_type=class_weight,
                max_complexes=getattr(loss_cfg, "class_weight_max_complexes", 1000),
                num_workers=getattr(cfg.data, "num_workers", 0),
            )
            cfg.loss.class_weight = weight_list

    model = setup_model(cfg, task=cfg.data.task)

    run_dir = os.path.join(cfg.log_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    strategy = setup_strategy(cfg.strategy)
    callbacks = setup_callbacks(args=cfg.callbacks, run_dir=run_dir, task=cfg.data.task)
    logger = setup_logger(cfg=cfg)

    trainer_cfg = cfg.trainer
    trainer = pl.Trainer(**trainer_cfg, strategy=strategy, callbacks=callbacks, logger=logger)

    config_out = os.path.join(run_dir, "model_parameters.yml")
    save_config(cfg, config_out)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.get("restart_ckpt", None))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        config_file = "configs/residue_rmsd/residue_rmsd.yaml"
        args = []
    else:
        config_file = sys.argv[1]
        args = sys.argv[2:]

    main(config_file=config_file, args=args)
