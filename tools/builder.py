import os

import torch
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import print_log
from utils.misc import (
    GradualWarmupScheduler,
    build_lambda_bnsche,
    build_lambda_sche,
    worker_init_fn,
)


def _to_plain_kwargs(cfg):
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def dataset_builder(cfg, split_cfg):
    dataset = build_dataset_from_cfg(split_cfg)
    shuffle = split_cfg.subset == "train"
    dataloader_kwargs = dict(
        batch_size=split_cfg.bs,
        num_workers=int(cfg.runtime.num_workers),
        drop_last=split_cfg.subset == "train",
        worker_init_fn=worker_init_fn,
        persistent_workers=int(cfg.runtime.num_workers) > 0,
    )

    if cfg.runtime.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, **dataloader_kwargs)
    return sampler, dataloader


def model_builder(config):
    return build_model_from_cfg(config)


def build_optimizer(base_model, cfg, stage=None):
    opti_config = cfg.optimizer

    if opti_config.type == "AdamW":
        def add_weight_decay(model, weight_decay=1e-5, skip_list=(), stage=None):
            decay = []
            no_decay = []
            if stage is not None:
                if stage == 1:
                    parameters = [
                        (name, param)
                        for name, param in model.named_parameters()
                        if ("primitive_segmentation" not in name and "plane_segmentation" not in name)
                    ]
                elif stage == 2:
                    parameters = [
                        (name, param)
                        for name, param in model.named_parameters()
                        if ("primitive_segmentation" in name or "plane_segmentation" in name)
                    ]
                else:
                    parameters = list(model.named_parameters())
            else:
                parameters = list(model.named_parameters())

            for name, param in parameters:
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay, "weight_decay": weight_decay},
            ]

        param_groups = add_weight_decay(
            base_model,
            weight_decay=opti_config.kwargs.weight_decay,
            stage=stage,
        )
        optimizer = optim.AdamW(param_groups, **_to_plain_kwargs(opti_config.kwargs))
    else:
        if stage is not None:
            if stage == 1:
                parameters = [
                    param
                    for name, param in base_model.named_parameters()
                    if ("primitive_segmentation" not in name and "plane_segmentation" not in name)
                ]
            elif stage == 2:
                parameters = [
                    param
                    for name, param in base_model.named_parameters()
                    if ("primitive_segmentation" in name or "plane_segmentation" in name)
                ]
            else:
                parameters = base_model.parameters()
        else:
            parameters = base_model.parameters()

        if opti_config.type == "Adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, parameters),
                **_to_plain_kwargs(opti_config.kwargs),
            )
        elif opti_config.type == "SGD":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, parameters),
                **_to_plain_kwargs(opti_config.kwargs),
            )
        else:
            raise NotImplementedError()

    return optimizer


def build_scheduler(base_model, optimizer1, optimizer2, optimizer3, cfg, last_epoch=-1):
    sche_config = cfg.scheduler
    if sche_config.type == "LambdaLR":
        if last_epoch == -1:
            scheduler1 = build_lambda_sche(optimizer1, sche_config.kwargs, last_epoch=last_epoch)
            scheduler2 = build_lambda_sche(optimizer2, sche_config.kwargs, last_epoch=last_epoch)
            scheduler3 = build_lambda_sche(optimizer3, sche_config.kwargs, last_epoch=last_epoch)
        else:
            scheduler1 = build_lambda_sche(optimizer1, sche_config.kwargs, last_epoch=last_epoch)
            scheduler2 = build_lambda_sche(
                optimizer2,
                sche_config.kwargs,
                last_epoch=max(-1, last_epoch - cfg.loss.first_stage),
            )
            scheduler3 = build_lambda_sche(
                optimizer3,
                sche_config.kwargs,
                last_epoch=max(-1, last_epoch - cfg.loss.second_stage),
            )
    elif sche_config.type == "StepLR":
        scheduler1 = torch.optim.lr_scheduler.StepLR(
            optimizer1, last_epoch=last_epoch, **_to_plain_kwargs(sche_config.kwargs)
        )
        scheduler2 = torch.optim.lr_scheduler.StepLR(
            optimizer2,
            last_epoch=max(-1, last_epoch - cfg.loss.first_stage),
            **_to_plain_kwargs(sche_config.kwargs),
        )
        scheduler3 = torch.optim.lr_scheduler.StepLR(
            optimizer3,
            last_epoch=max(-1, last_epoch - cfg.loss.second_stage),
            **_to_plain_kwargs(sche_config.kwargs),
        )
    elif sche_config.type == "GradualWarmup":
        scheduler_steplr1 = torch.optim.lr_scheduler.StepLR(
            optimizer1, last_epoch=last_epoch, **_to_plain_kwargs(sche_config.kwargs_1)
        )
        scheduler_steplr2 = torch.optim.lr_scheduler.StepLR(
            optimizer2,
            last_epoch=max(-1, last_epoch - cfg.loss.first_stage),
            **_to_plain_kwargs(sche_config.kwargs_1),
        )
        scheduler_steplr3 = torch.optim.lr_scheduler.StepLR(
            optimizer3,
            last_epoch=max(-1, last_epoch - cfg.loss.second_stage),
            **_to_plain_kwargs(sche_config.kwargs_1),
        )
        scheduler1 = GradualWarmupScheduler(
            optimizer1, after_scheduler=scheduler_steplr1, **_to_plain_kwargs(sche_config.kwargs_2)
        )
        scheduler2 = GradualWarmupScheduler(
            optimizer2, after_scheduler=scheduler_steplr2, **_to_plain_kwargs(sche_config.kwargs_2)
        )
        scheduler3 = GradualWarmupScheduler(
            optimizer3, after_scheduler=scheduler_steplr3, **_to_plain_kwargs(sche_config.kwargs_2)
        )
    else:
        raise NotImplementedError()

    if cfg.get("bnmscheduler") is not None:
        bnsche_config = cfg.bnmscheduler
        if bnsche_config.type == "Lambda":
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)
        scheduler1 = [scheduler1, bnscheduler]
        scheduler2 = [scheduler2, bnscheduler]
        scheduler3 = [scheduler3, bnscheduler]

    return scheduler1, scheduler2, scheduler3


def resume_model(base_model, cfg, logger=None):
    ckpt_path = os.path.join(cfg.paths.run_dir, "ckpt-last.pth")
    if not os.path.exists(ckpt_path):
        print_log(f"[RESUME INFO] no checkpoint file from path {ckpt_path}...", logger=logger)
        return 0, 0
    print_log(f"[RESUME INFO] Loading model weights from {ckpt_path}...", logger=logger)

    local_rank = int(cfg.runtime.local_rank)
    if cfg.runtime.use_gpu:
        map_location = {f"cuda:{0}": f"cuda:{local_rank}"}
    else:
        map_location = "cpu"
    state_dict = torch.load(ckpt_path, map_location=map_location)

    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict["base_model"].items()}
    base_model.load_state_dict(base_ckpt)

    start_epoch = state_dict["epoch"] + 1
    best_metrics = state_dict["best_metrics"]
    print_log(
        f"[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})",
        logger=logger,
    )
    return start_epoch, best_metrics


def resume_optimizer(optimizer1, optimizer2, optimizer3, cfg, logger=None):
    ckpt_path = os.path.join(cfg.paths.run_dir, "ckpt-last.pth")
    if not os.path.exists(ckpt_path):
        print_log(f"[RESUME INFO] no checkpoint file from path {ckpt_path}...", logger=logger)
        return False, False, False
    print_log(f"[RESUME INFO] Loading optimizer from {ckpt_path}...", logger=logger)
    state_dict = torch.load(ckpt_path, map_location="cpu")

    loaded_flags = [False, False, False]
    for idx, opt in enumerate([optimizer1, optimizer2, optimizer3], start=1):
        key = f"optimizer{idx}"
        if key not in state_dict:
            print_log(f"[RESUME WARN] '{key}' not found in checkpoint. Using fresh state.", logger=logger)
            continue
        try:
            opt.load_state_dict(state_dict[key])
            loaded_flags[idx - 1] = True
        except (ValueError, RuntimeError) as e:
            print_log(f"[RESUME WARN] Skipping load for {key} due to incompatibility: {e}", logger=logger)
    return tuple(loaded_flags)


def save_checkpoint(base_model, optimizer1, optimizer2, optimizer3, epoch, metrics, best_metrics, prefix, cfg, logger=None):
    if int(cfg.runtime.local_rank) != 0:
        return

    state_dict = base_model.module.state_dict() if hasattr(base_model, "module") else base_model.state_dict()
    save_path = os.path.join(cfg.paths.run_dir, f"{prefix}.pth")
    torch.save(
        {
            "base_model": state_dict,
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "optimizer3": optimizer3.state_dict(),
            "epoch": epoch,
            "metrics": metrics if metrics is not None else dict(),
            "best_metrics": best_metrics if best_metrics is not None else dict(),
        },
        save_path,
    )
    print_log(f"Save checkpoint at {save_path}", logger=logger)


def load_model(base_model, ckpt_path, logger=None):
    ckpt_path = to_absolute_path(ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(f"no checkpoint file from path {ckpt_path}...")
    print_log(f"Loading weights from {ckpt_path}...", logger=logger)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if state_dict.get("model") is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict["model"].items()}
    elif state_dict.get("base_model") is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict["base_model"].items()}
    else:
        raise RuntimeError("mismatch of ckpt weight")
    base_model.load_state_dict(base_ckpt)

    epoch = state_dict.get("epoch", -1)
    metrics = state_dict.get("metrics", "No Metrics")
    print_log(f"ckpts @ {epoch} epoch( performance = {str(metrics):s})", logger=logger)
    return epoch
