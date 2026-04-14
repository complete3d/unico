import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tensorboardX import SummaryWriter

from tools import run_net
from utils import dist_utils, misc
from utils.config import (
    apply_resume_config,
    log_config_to_file,
    save_resolved_config,
)
from utils.logger import get_root_logger


def _require_helper_script() -> None:
    if os.environ.get("UNICO_USE_HELPER_SCRIPT") != "1":
        raise RuntimeError(
            "Training must be launched via the helper scripts. "
            "Use `bash scripts/train_dp.sh ...` or `bash scripts/train_ddp.sh ...`."
        )


def _resolve_launcher(cfg: DictConfig) -> str:
    launcher = str(cfg.runtime.launcher)
    if launcher != "auto":
        return launcher
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = os.environ.get("RANK")
    if world_size > 1 or rank is not None:
        return "pytorch"
    return "none"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _require_helper_script()
    cfg = apply_resume_config(cfg, HydraConfig.get().overrides.task)

    cfg.runtime.use_gpu = bool(cfg.runtime.use_gpu and torch.cuda.is_available())
    if cfg.runtime.use_gpu:
        torch.backends.cudnn.benchmark = True

    cfg.runtime.launcher = _resolve_launcher(cfg)
    cfg.runtime.distributed = cfg.runtime.launcher != "none"
    if cfg.runtime.distributed:
        dist_utils.init_dist(cfg.runtime.launcher)
        _, world_size = dist_utils.get_dist_info()
        cfg.runtime.world_size = world_size
    else:
        cfg.runtime.world_size = 1

    if cfg.runtime.distributed:
        assert cfg.training.total_bs % cfg.runtime.world_size == 0
        cfg.dataset.train.bs = cfg.training.total_bs // cfg.runtime.world_size
        cfg.dataset.val.bs = cfg.training.total_bs // cfg.runtime.world_size
    else:
        cfg.dataset.train.bs = cfg.training.total_bs
        cfg.dataset.val.bs = cfg.training.total_bs

    local_rank = int(cfg.runtime.local_rank)
    log_name = "unico"
    log_file = os.path.join(
        cfg.paths.run_dir,
        "train.log" if local_rank == 0 else f"train.rank{local_rank}.log",
    )
    logger = get_root_logger(log_file=log_file, name=log_name)

    train_writer = None
    val_writer = None
    if local_rank == 0:
        train_writer = SummaryWriter(os.path.join(cfg.paths.tensorboard_dir, "train"))
        val_writer = SummaryWriter(os.path.join(cfg.paths.tensorboard_dir, "test"))
        save_resolved_config(cfg, os.path.join(cfg.paths.run_dir, "config.resolved.yaml"))

    logger.info(f"Distributed training: {cfg.runtime.distributed}")
    logger.info(f"Launcher: {cfg.runtime.launcher}")
    logger.info(f"SyncBatchNorm enabled: {bool(cfg.runtime.distributed and cfg.runtime.sync_bn)}")
    logger.info(f"Hydra output dir: {HydraConfig.get().runtime.output_dir}")
    log_config_to_file(cfg, logger=logger)

    if cfg.runtime.seed is not None:
        logger.info(
            "Set random seed to %s, deterministic: %s",
            cfg.runtime.seed,
            cfg.runtime.deterministic,
        )
        misc.set_random_seed(
            int(cfg.runtime.seed) + local_rank,
            deterministic=bool(cfg.runtime.deterministic),
        )

    if cfg.runtime.distributed:
        assert local_rank == torch.distributed.get_rank()

    run_net(cfg, train_writer, val_writer, logger_name=log_name)


if __name__ == "__main__":
    main()
