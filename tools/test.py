import os

import numpy as np
import torch
from hydra.utils import to_absolute_path
from tqdm import tqdm

from datasets.data_transforms import Compose
from datasets.io import IO
from tools import builder


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


def _grad_F(x, th):
    A, B, C, D, E, F_, G, H, I, J = th
    gx = 2 * (A * x[..., 0] + D * x[..., 1] + E * x[..., 2] + G)
    gy = 2 * (D * x[..., 0] + B * x[..., 1] + F_ * x[..., 2] + H)
    gz = 2 * (E * x[..., 0] + F_ * x[..., 1] + C * x[..., 2] + I)
    return np.stack([gx, gy, gz], axis=-1)


def _F_val(x, th):
    A, B, C, D, E, F_, G, H, I, J = th
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    return (
        A * X * X
        + B * Y * Y
        + C * Z * Z
        + 2 * D * X * Y
        + 2 * E * X * Z
        + 2 * F_ * Y * Z
        + 2 * G * X
        + 2 * H * Y
        + 2 * I * Z
        + J
    )


def quadric_normals_at_points(points_xyz, ids, coeffs_10):
    normals = np.zeros((points_xyz.shape[0], 3), dtype=np.float64)
    if coeffs_10 is None or len(coeffs_10) == 0:
        return normals.astype(np.float32)

    ids = ids.astype(int)
    for gid in np.unique(ids):
        if gid < 0 or gid >= coeffs_10.shape[0]:
            continue
        idx = np.where(ids == gid)[0]
        if idx.size == 0:
            continue
        x = points_xyz[idx]
        th = coeffs_10[gid]
        g0 = _grad_F(x, th)
        f0 = _F_val(x, th)
        denom = (np.linalg.norm(g0, axis=-1) ** 2 + 1e-15)[..., None]
        x1 = x - (f0[..., None] / denom) * g0
        g1 = _grad_F(x1, th)
        normals[idx] = _normalize(g1)
    return normals.astype(np.float32)


def build_model_from_config(cfg):
    device_name = cfg.evaluate.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    base_model = builder.model_builder(cfg.model)
    builder.load_model(base_model, cfg.evaluate.ckpt_path)
    base_model.to(device)
    base_model.eval()
    return base_model, device


def load_test_ids(test_list_file):
    with open(test_list_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return [ln.split(".")[0] for ln in lines]


def add_noise_to_points(pc, std=0.0):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(mean=0.0, std=std)
        pc[i, :, 0:3] += jittered_data
    return pc


def _predict_instances(ret, output_format, threshold, param_mode, out_dir, model_id):
    raw_cls = ret["class_prob"].softmax(dim=-1)
    class_prob = (1.0 - raw_cls[..., -1]).unsqueeze(-1)
    pred_masks = ret["pred_masks"].sigmoid()
    rebuild_points = ret["rebuild_points"]

    m = pred_masks > 0.5
    c_m = (pred_masks * m).sum(-1) / torch.clamp(m.sum(-1), min=1)
    keep = torch.bitwise_and(m.sum(-1) > 1, (c_m * class_prob.squeeze(-1)) > threshold)

    heatmap = class_prob * keep.float().unsqueeze(-1) * pred_masks
    assigned_query = heatmap.argmax(dim=1)
    assigned_score = heatmap.max(dim=1).values
    assigned_query[assigned_score < 0.5] = -1

    B, _ = assigned_query.shape
    for b in range(B):
        query_ids = assigned_query[b].unique()
        query_ids = query_ids[query_ids >= 0]
        probs = ret["class_prob"][b].softmax(dim=-1)
        type_ids = probs.argmax(dim=-1)

        batch_instances = []
        accepted_qids = []
        for qid in query_ids:
            qid_int = int(qid.item())
            tid = int(type_ids[qid_int].item())
            if probs.shape[-1] == 5 and tid == 4:
                continue
            if probs.shape[-1] == 2 and tid == 1:
                continue

            point_mask = assigned_query[b] == qid
            if point_mask.sum() >= 3:
                current_points = (
                    rebuild_points[b]
                    .reshape(512, -1, 3)[point_mask, :, :]
                    .reshape(-1, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_instances.append(
                    np.concatenate(
                        (
                            current_points,
                            np.ones((current_points.shape[0], 1)) * len(accepted_qids),
                        ),
                        axis=1,
                    )
                )
                accepted_qids.append(qid_int)

        batch_instances = (
            np.concatenate(batch_instances, axis=0)
            if batch_instances
            else np.zeros((0, 4), dtype=np.float32)
        )

        if output_format in ("vg", "seg"):
            if batch_instances.shape[0] == 0:
                print(f"[WARN] {model_id}: no valid primitives after filtering; skipping export.")
                continue

            xyz_lab = batch_instances
            pts_xyz = xyz_lab[:, :3].astype(np.float32)
            local_ids = xyz_lab[:, 3].astype(int)

            quadric_params = ret.get("quadrics", None)
            group_info = []
            coeffs_list = []
            if quadric_params is not None and accepted_qids:
                new_gid = 0
                for orig_qid in accepted_qids:
                    if orig_qid >= type_ids.shape[0]:
                        continue
                    tid = int(type_ids[orig_qid].item())
                    if quadric_params.shape[-1] == 10:
                        params10 = quadric_params[b, orig_qid].detach().cpu().numpy()[:10]
                    elif quadric_params.shape[-1] == 4:
                        params10 = quadric_params[b, orig_qid].detach().cpu().numpy()[:4]
                        params10[:3] = params10[:3] / 2
                        params10 = np.concatenate([np.zeros(6, dtype=np.float32), params10], axis=0)
                        params10 = params10 / np.linalg.norm(params10)
                    else:
                        raise ValueError(
                            f"Unsupported number of quadric parameters: {quadric_params.shape[-1]}"
                        )
                    group_info.append({"id": new_gid, "type": tid, "parameters": params10})
                    coeffs_list.append(params10)
                    new_gid += 1

            coeffs_10 = np.asarray(coeffs_list, dtype=np.float32) if coeffs_list else None
            normals = quadric_normals_at_points(pts_xyz, local_ids, coeffs_10)

            if output_format == "vg":
                from utils.save_vg import save_vg

                out_path = os.path.join(out_dir, f"{model_id}.vg")
                save_vg(
                    xyz_lab,
                    out_path,
                    group_info=group_info if group_info else None,
                    normals=normals,
                )
            else:
                from utils.save_seg import save_seg

                out_path = os.path.join(out_dir, f"{model_id}.seg")
                seg_group_info = [
                    {"id": entry["id"], "type": entry["type"], "parameters": entry["parameters"]}
                    for entry in (group_info or [])
                ]
                save_seg(
                    xyz_lab,
                    out_path,
                    group_info=seg_group_info,
                    normals=normals,
                    param_mode=param_mode,
                )
        else:
            out_path = os.path.join(out_dir, f"{model_id}.xyz")
            np.savetxt(out_path, batch_instances)


def run_inference_over_list(
    model,
    test_ids,
    data_root,
    mode,
    out_dir,
    device,
    data_mode,
    output_format,
    noise_std,
    threshold,
    param_mode,
):
    os.makedirs(out_dir, exist_ok=True)
    if data_mode not in ("npy", "ply"):
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    transform = Compose(
        [
            {"callback": "UpSamplePoints", "parameters": {"n_points": 2048}, "objects": ["input"]},
            {"callback": "ToTensor", "objects": ["input"]},
        ]
    )

    for model_id in tqdm(test_ids):
        pc_base = os.path.join(data_root, "eval", mode, f"{model_id}_00")
        pc_file = f"{pc_base}.{data_mode}"
        if not os.path.exists(pc_file):
            raise FileNotFoundError(
                f"Partial PC file not found for ID '{model_id}' under base '{pc_base}'."
            )
        pc_ndarray = IO.get(pc_file).astype(np.float32)
        pc_ndarray_normalized = transform({"input": pc_ndarray[:, :3]})
        if noise_std > 0.0:
            noisy = add_noise_to_points(
                pc_ndarray_normalized["input"].unsqueeze(0),
                std=noise_std,
            )
            pc_ndarray_normalized["input"] = noisy.squeeze(0)

        with torch.no_grad():
            ret = model(pc_ndarray_normalized["input"].unsqueeze(0).to(device), epoch=600)

        _predict_instances(ret, output_format, threshold, param_mode, out_dir, model_id)


def run_inference_over_dataset(model, dataloader, device, output_format, noise_std, threshold, param_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for model_ids, data in tqdm(dataloader):
        model_id = model_ids[0]
        pc = data[4]
        if noise_std > 0.0:
            pc = add_noise_to_points(pc.clone(), std=noise_std)

        with torch.no_grad():
            ret = model(pc.to(device), epoch=600)

        _predict_instances(ret, output_format, threshold, param_mode, out_dir, model_id)


def run_inference(cfg):
    out_dir = to_absolute_path(cfg.evaluate.output_dir)
    model, device = build_model_from_config(cfg)
    output_format = str(cfg.evaluate.output_format).lower()
    noise_std = float(cfg.evaluate.noise_std)
    threshold = float(cfg.evaluate.threshold)
    param_mode = cfg.evaluate.param_mode

    if cfg.evaluate.input_source == "abcmulti_eval":
        data_root = to_absolute_path(cfg.evaluate.data_root)
        test_list_file = to_absolute_path(cfg.evaluate.test_list_file)
        test_ids = load_test_ids(test_list_file)
        run_inference_over_list(
            model,
            test_ids,
            data_root=data_root,
            mode=cfg.evaluate.mode,
            out_dir=out_dir,
            device=device,
            data_mode=cfg.evaluate.data_mode,
            output_format=output_format,
            noise_std=noise_std,
            threshold=threshold,
            param_mode=param_mode,
        )
    elif cfg.evaluate.input_source == "dataset":
        _, dataloader = builder.dataset_builder(cfg, cfg.dataset.test)
        run_inference_over_dataset(
            model,
            dataloader,
            device=device,
            output_format=output_format,
            noise_std=noise_std,
            threshold=threshold,
            param_mode=param_mode,
            out_dir=out_dir,
        )
    else:
        raise ValueError(f"Unsupported evaluate.input_source: {cfg.evaluate.input_source}")
