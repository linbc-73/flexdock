from flexdock.models.pl_modules.docking import FlexDockModule
from flexdock.models.pl_modules.filtering import FilteringModule
from flexdock.models.pl_modules.relaxation import RelaxFlowModule
from flexdock.models.pl_modules.residue_rmsd import ResidueRMSDModule


TASK_TO_MODULES = {
    "docking": FlexDockModule,
    "filtering": FilteringModule,
    "relaxation": RelaxFlowModule,
    "residue_rmsd": ResidueRMSDModule,
}


def setup_model(cfg, task: str = "docking"):
    model_cls = TASK_TO_MODULES.get(task, None)
    if model_cls is None:
        raise ValueError(
            f"Task of type={task} not supported. Supported tasks are {list(TASK_TO_MODULES.keys())}"
        )

    if task == "docking":
        return FlexDockModule(
            model_cfg=cfg.model,
            sigma_cfg=cfg.sigma,
            training_cfg=cfg.training,
            sampler_cfg=cfg.sampler,
            loss_cfg=cfg.loss,
        )
    elif task == "residue_rmsd":
        return ResidueRMSDModule(
            model_cfg=cfg.model,
            sigma_cfg=cfg.sigma,
            training_cfg=cfg.training,
            sampler_cfg=cfg.get("sampler", None),
            loss_cfg=cfg.get("loss", None),
        )
    else:
        raise NotImplementedError(
            "Config based pipeline is WIP for confidence and relaxation"
        )
