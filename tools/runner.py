import time

import torch
import torch.nn as nn

from tools import builder
from utils import dist_utils, misc
from utils.AverageMeter import AverageMeter
from utils.logger import get_logger, print_log


def _unwrap_model(base_model):
    return base_model.module if hasattr(base_model, "module") else base_model


def _move_train_batch(data, dataset_name, cfg, device):
    if dataset_name == "ABCMulti":
        gt = data[0].to(device)
        gt_index = data[1].to(device)
        gt_coeff = data[2].to(device)
        gt_type = data[3].to(device)
        pc = gt
        npoints = cfg.dataset.train.N_POINTS
        pc, _ = misc.seprate_point_cloud(
            gt,
            npoints,
            [int(npoints * 1 / 4), int(npoints * 3 / 4)],
            fixed_points=None,
        )
        pc = pc.to(device)
    elif dataset_name in {"ABCPlane", "BuildingNL"}:
        gt = data[0].to(device)
        gt_index = data[1].to(device)
        gt_coeff = data[2].to(device)
        gt_type = data[3].to(device)
        pc = data[4].to(device)
    else:
        raise NotImplementedError(f"Train phase do not support {dataset_name}")

    return gt, gt_index, gt_coeff, gt_type, pc


def run_net(cfg, train_writer=None, val_writer=None, logger_name="unico"):
    logger = get_logger(logger_name)
    local_rank = int(cfg.runtime.local_rank)
    device = torch.device(f"cuda:{local_rank}" if cfg.runtime.use_gpu else "cpu")

    (train_sampler, train_dataloader), (_, test_dataloader) = (
        builder.dataset_builder(cfg, cfg.dataset.train),
        builder.dataset_builder(cfg, cfg.dataset.val),
    )

    base_model = builder.model_builder(cfg.model)
    base_model.to(device)

    start_epoch = 0
    best_metrics = None
    metrics = None

    if cfg.runtime.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, cfg, logger=logger)
    elif cfg.runtime.start_ckpt is not None:
        start_epoch = builder.load_model(base_model, cfg.runtime.start_ckpt, logger=logger)

    print_log("Trainable_parameters:", logger=logger)
    print_log("=" * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log("=" * 25, logger=logger)

    print_log("Untrainable_parameters:", logger=logger)
    print_log("=" * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log("=" * 25, logger=logger)

    if cfg.runtime.distributed:
        if bool(cfg.runtime.sync_bn):
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    elif cfg.runtime.use_gpu:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).to(device)
    else:
        print_log("Using CPU model ...", logger=logger)

    optimizer1 = builder.build_optimizer(base_model, cfg, stage=1)
    optimizer2 = builder.build_optimizer(base_model, cfg, stage=2)
    optimizer3 = builder.build_optimizer(base_model, cfg, stage=3)

    def _count_optim_params(optim):
        try:
            return sum(p.numel() for g in optim.param_groups for p in g.get("params", []))
        except Exception:
            return -1

    print_log(
        f"[OPTIM INFO] stage1 params: {_count_optim_params(optimizer1)}; "
        f"stage2 params: {_count_optim_params(optimizer2)}; "
        f"stage3 params: {_count_optim_params(optimizer3)}",
        logger=logger,
    )

    if cfg.runtime.resume:
        builder.resume_optimizer(optimizer1, optimizer2, optimizer3, cfg, logger=logger)
        sched_last_epoch = start_epoch - 1
    else:
        sched_last_epoch = -1

    scheduler1, scheduler2, scheduler3 = builder.build_scheduler(
        base_model, optimizer1, optimizer2, optimizer3, cfg, last_epoch=sched_last_epoch
    )

    base_model.zero_grad()
    last_test_losses = None
    for epoch in range(start_epoch, int(cfg.training.max_epoch) + 1):
        if epoch < cfg.loss.first_stage:
            consider_metric = cfg.loss.consider_metric[0]
        elif epoch < cfg.loss.second_stage:
            consider_metric = cfg.loss.consider_metric[1]
        else:
            consider_metric = cfg.loss.consider_metric[2]

        if epoch == cfg.loss.first_stage:
            best_metrics = None
        if epoch == cfg.loss.second_stage:
            best_metrics = None

        if cfg.runtime.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_losses = AverageMeter(
            [
                "loss_denoised",
                "loss_coarse",
                "chamfer_norm1_loss",
                "classification_loss",
                "mask_loss",
                "dice_loss",
                "primitive_chamfer_loss",
                "primitive_normal_loss",
                "total_loss_stage3",
            ]
        )

        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            gt, gt_index, gt_coeff, gt_type, pc = _move_train_batch(
                data, cfg.dataset.train.NAME, cfg, device
            )

            num_iter += 1
            ret = base_model(pc, epoch=epoch)
            model_ref = _unwrap_model(base_model)
            losses = model_ref.get_loss(cfg.loss, ret, gt, gt_index, gt_coeff, gt_type, epoch)

            if epoch < cfg.loss.first_stage:
                loss = losses["total_loss_stage1"]
                loss.backward()
                if num_iter == cfg.training.step_per_update:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(),
                        cfg.get("grad_norm_clip", 10),
                        norm_type=2,
                    )
                    num_iter = 0
                    optimizer1.step()
                    base_model.zero_grad()
            elif epoch < cfg.loss.second_stage:
                loss = losses["total_loss_stage2"]
                loss.backward()
                if num_iter == cfg.training.step_per_update:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(),
                        cfg.get("grad_norm_clip", 10),
                        norm_type=2,
                    )
                    num_iter = 0
                    optimizer2.step()
                    base_model.zero_grad()
            else:
                loss = losses["total_loss_stage3"]
                loss.backward()
                if num_iter == cfg.training.step_per_update:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(),
                        cfg.get("grad_norm_clip", 10),
                        norm_type=2,
                    )
                    num_iter = 0
                    optimizer3.step()
                    base_model.zero_grad()

            losses_dict = {}
            if cfg.runtime.distributed:
                for key, value in losses.items():
                    losses_dict[key] = dist_utils.reduce_tensor(value, cfg) * 1000
            else:
                for key, value in losses.items():
                    losses_dict[key] = value * 1000
            train_losses.update(losses_dict)

            if cfg.runtime.distributed and cfg.runtime.use_gpu:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                for key, value in losses.items():
                    train_writer.add_scalar(f"Loss/Batch/{key}", value.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 10 == 0:
                if epoch < cfg.loss.first_stage:
                    lr = optimizer1.param_groups[0]["lr"]
                elif epoch < cfg.loss.second_stage:
                    lr = optimizer2.param_groups[0]["lr"]
                else:
                    lr = optimizer3.param_groups[0]["lr"]
                print_log(
                    f"[Epoch {epoch}/{cfg.training.max_epoch}][Batch {idx + 1}/{n_batches}] | "
                    f"BatchTime = {batch_time.val():.3f}s | "
                    f'Losses = [{", ".join(f"{l:.3f}" for l in train_losses.val())}] | '
                    f"lr = {lr:.6f}",
                    logger=logger,
                )

            if cfg.scheduler.type == "GradualWarmup" and n_itr < cfg.scheduler.kwargs_2.total_epoch:
                if epoch < cfg.loss.first_stage:
                    scheduler1.step()
                elif epoch < cfg.loss.second_stage:
                    scheduler2.step()
                else:
                    scheduler3.step()

        if isinstance(scheduler1, list):
            if epoch < cfg.loss.first_stage:
                for item in scheduler1:
                    item.step()
            elif epoch < cfg.loss.second_stage:
                for item in scheduler2:
                    item.step()
            else:
                for item in scheduler3:
                    item.step()
        else:
            if epoch < cfg.loss.first_stage:
                scheduler1.step()
            elif epoch < cfg.loss.second_stage:
                scheduler2.step()
            else:
                scheduler3.step()

        epoch_end_time = time.time()

        if train_writer is not None:
            for key in losses.keys():
                train_writer.add_scalar(f"Loss/Epoch/{key}", train_losses.avg(key=key), epoch)
            print_log(
                f"[Training] Epoch: {epoch} | EpochTime = {epoch_end_time - epoch_start_time:.3f}s | "
                f'Losses = [{", ".join(f"{l:.4f}" for l in train_losses.avg())}]',
                logger=logger,
            )

        if epoch % int(cfg.training.val_freq) == 0:
            test_losses = validate(base_model, test_dataloader, epoch, val_writer, cfg, device, logger=logger)
            last_test_losses = test_losses
            if best_metrics is None:
                best_metrics = test_losses[consider_metric]
                metrics = test_losses

            if test_losses[consider_metric] < best_metrics:
                best_metrics = test_losses[consider_metric]
                metrics = test_losses
                builder.save_checkpoint(
                    base_model,
                    optimizer1,
                    optimizer2,
                    optimizer3,
                    epoch,
                    metrics,
                    best_metrics,
                    "ckpt-best",
                    cfg,
                    logger=logger,
                )

        builder.save_checkpoint(
            base_model,
            optimizer1,
            optimizer2,
            optimizer3,
            epoch,
            metrics,
            best_metrics,
            "ckpt-last",
            cfg,
            logger=logger,
        )
        if (
            ((cfg.loss.first_stage - epoch) < 2 and (cfg.loss.first_stage - epoch) >= 0)
            or ((cfg.loss.second_stage - epoch) < 2 and (cfg.loss.second_stage - epoch) >= 0)
            or (cfg.training.max_epoch - epoch) < 2
        ) and last_test_losses is not None:
            metrics = last_test_losses
            builder.save_checkpoint(
                base_model,
                optimizer1,
                optimizer2,
                optimizer3,
                epoch,
                metrics,
                best_metrics,
                f"ckpt-epoch-{epoch:03d}",
                cfg,
                logger=logger,
            )

    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, cfg, device, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()

    test_losses = AverageMeter(
        [
            "loss_coarse",
            "chamfer_norm1_loss",
            "classification_loss",
            "mask_loss",
            "dice_loss",
            "primitive_chamfer_loss",
            "primitive_normal_loss",
            "total_loss_stage3",
        ]
    )

    n_samples = len(test_dataloader)
    with torch.no_grad():
        for idx, (model_ids, data) in enumerate(test_dataloader):
            gt, gt_index, gt_coeff, gt_type, pc = _move_train_batch(
                data, cfg.dataset.val.NAME, cfg, device
            )

            ret = base_model(pc, epoch=epoch)
            model_ref = _unwrap_model(base_model)
            losses = model_ref.get_loss(cfg.loss, ret, gt, gt_index, gt_coeff, gt_type, epoch)

            losses_dict = {}
            for key, value in losses.items():
                if cfg.runtime.distributed:
                    losses_dict[key] = dist_utils.reduce_tensor(value, cfg) * 1000
                else:
                    losses_dict[key] = value * 1000

            test_losses.update(losses_dict)
            if idx % 10 == 0:
                print_log(
                    f'[Epoch {epoch}/{cfg.training.max_epoch}][Batch {idx + 1}/{n_samples}] '
                    f'Losses = [{", ".join(f"{l:.3f}" for l in test_losses.val())}]',
                    logger=logger,
                )

        if cfg.runtime.distributed and cfg.runtime.use_gpu:
            torch.cuda.synchronize()

    if val_writer is not None:
        for key in losses.keys():
            val_writer.add_scalar(f"Loss/Epoch/{key}", test_losses.avg(key=key), epoch)

    print_log(
        f'[Validation] Epoch: {epoch} | Losses = [{", ".join(f"{l:.4f}" for l in test_losses.avg())}]',
        logger=logger,
    )

    metrics = {}
    for key in losses.keys():
        metrics[key] = test_losses.avg(key=key)
    return metrics
