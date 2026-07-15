"""Run TRPM model inference and save predicted t2 point clouds as geometry assets.

This bridges the TRPM models with src/evaluate.py by:
1. Loading a trained TRPM checkpoint
2. Running inference per variant to produce predicted t2 point clouds
3. Saving them as point_cloud_clean.npz in the geometry_assets layout

After running this, src/evaluate.py can be used directly on the output.

Output layout:
    {output_root}/{triplet_id}/{variant}/t2/point_cloud_clean.npz

Usage (single GPU):
    python src/trpm/build_predicted_geometry.py \
        --runs-root runs/trpm_small_cam_depth \
        --output-root geometry_assets_predicted/trpm_small_cam_depth \
        --protocol strict --crop corn

Usage (multi-GPU, 8 GPUs):
    torchrun --nproc_per_node=8 src/trpm/build_predicted_geometry.py \
        --runs-root runs/trpm_small_cam_depth \
        --output-root geometry_assets_predicted/trpm_small_cam_depth \
        --protocol strict --crop corn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loto import build_all_folds
from trpm.dataset import _load_pointmaps, _gps_alignment, _to_gps, _load_camera_data


def _load_model_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module_path), class_name)


DEFAULT_CONFIG: dict[str, Any] = {
    "vggt_output_root": "vggt_outputs/t1t2_paired_v16_o8",
    "geometry_assets_root": "geometry_assets",
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "runs_root": "runs/trpm_small",
    "output_root": "geometry_assets_predicted/trpm_small",
    "protocols": ["strict"],
    "crops": ["corn"],
    "conf_threshold": 0.02,
    "pred_conf_threshold": 0.02,
    "n_points": 50_000,
    "seed": 42,
    "device": "auto",
    "model_kwargs": {
        "num_t3_points": 1024,
        "cond_dim": 192,
    },
}


# ── DDP helpers ───────────────────────────────────────────────────────────────

def _is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ


def _setup_distributed() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size()


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_main() -> bool:
    return _get_rank() == 0


# ── utilities ─────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    if not _is_main():
        return
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def log_all(msg: str) -> None:
    """Log from any rank (for debugging)."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][rank={_get_rank()}] {msg}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build predicted geometry assets from TRPM model.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--protocol", choices=["target_date", "strict"], action="append", default=None)
    p.add_argument("--crop", action="append", default=None)
    p.add_argument("--test-date", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--model-class", type=str, default=None,
                   help="Dotted model class path, e.g. trpm.trpm_small_cam_depth.TRPMSmallCamDepth")
    p.add_argument("--geometry-assets-root", type=Path, default=None,
                   help="Path to existing build_geometry_assets.py output (for t1/t3 ground truth).")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config and args.config.exists():
        cfg.update(read_yaml(args.config))
    if args.runs_root:
        cfg["runs_root"] = args.runs_root
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.device:
        cfg["device"] = args.device
    if args.protocol:
        cfg["protocols"] = args.protocol
    if args.crop:
        cfg["crops"] = args.crop
    if args.test_date:
        cfg["test_date"] = args.test_date
    cfg["skip_existing"] = args.skip_existing
    if args.model_class:
        cfg["model_class"] = args.model_class
    if args.geometry_assets_root:
        cfg["geometry_assets_root"] = args.geometry_assets_root
    pred_thr = cfg.get("pred_conf_threshold", cfg.get("conf_threshold", 0.02))
    cfg["pred_conf_threshold"] = pred_thr
    cfg.setdefault("model_kwargs", {})["conf_threshold"] = pred_thr
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["geometry_assets_root"] = Path(cfg["geometry_assets_root"])
    cfg["runs_root"] = Path(cfg["runs_root"])
    cfg["output_root"] = Path(cfg["output_root"])
    cfg["triplets_path"] = Path(cfg["triplets_path"])
    return cfg


# ── helpers ───────────────────────────────────────────────────────────────────


def _compute_pad_mode_intrinsics(date_dir: Path) -> np.ndarray:
    """Compute dataset_cameras intrinsics scaled to VGGT resolution with pad mode."""
    import json as _json
    cam = _json.loads((date_dir / "dataset_cameras.json").read_text())
    intr = cam["intrinsics"]
    pm_shape = np.load(date_dir / "predictions" / "point_map.npy", mmap_mode="r").shape
    H, W = pm_shape[1], pm_shape[2]
    w_orig, h_orig = float(intr["w"]), float(intr["h"])
    scale = min(W / w_orig, H / h_orig)
    pad_left = (W - w_orig * scale) / 2
    pad_top = (H - h_orig * scale) / 2
    return np.array([
        [intr["fl_x"] * scale, 0.0, intr["cx"] * scale + pad_left],
        [0.0, intr["fl_y"] * scale, intr["cy"] * scale + pad_top],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _vggt_ext_to_c2w(ext_3x4: np.ndarray) -> np.ndarray:
    """Invert VGGT world-to-camera extrinsic [3, 4] → camera-to-world [4, 4]."""
    R = ext_3x4[:, :3].astype(np.float64)
    t = ext_3x4[:, 3].astype(np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w

def _list_variants(vggt_root: Path, triplet_id: str) -> list[str]:
    d = vggt_root / triplet_id
    if not d.exists():
        return []
    return sorted(v.name for v in d.iterdir() if v.is_dir())


def _pointmap_to_cloud(pm: np.ndarray, pc: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Flatten [H, W, 3] point map → (N, 3) points + (N,) confidence."""
    conf = pc.reshape(-1)
    mask = conf >= conf_threshold
    xyz = pm.reshape(-1, 3)[mask].astype(np.float32)
    c = conf[mask].astype(np.float32)
    return xyz, c


def _voxel_downsample(pts: np.ndarray, conf: np.ndarray, voxel_size: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0 or voxel_size <= 0:
        return pts, conf
    keys = (pts / voxel_size).astype(np.int64)
    _, uidx = np.unique(keys, axis=0, return_index=True)
    return pts[uidx], conf[uidx]


def _postprocess(pts: np.ndarray, conf: np.ndarray, n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0:
        return pts, conf
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    keep = dists <= np.quantile(dists, 0.995)
    pts, conf = pts[keep], conf[keep]
    pts, conf = _voxel_downsample(pts, conf)
    if n_points > 0 and len(pts) > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), n_points, replace=False)
        pts, conf = pts[idx], conf[idx]
    return pts, conf


def _load_date_cloud(date_dir: Path, conf_threshold: float, n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load VGGT point maps for a date, align to GPS, return (points, confidence)."""
    pred_dir = date_dir / "predictions"
    pm_all = np.load(pred_dir / "point_map.npy", mmap_mode="r")
    pc_all = np.load(pred_dir / "point_confidence.npy", mmap_mode="r")
    alignment = _gps_alignment(date_dir)
    all_pts, all_conf = [], []
    for vi in range(pm_all.shape[0]):
        pm = pm_all[vi].astype(np.float32)
        pc = pc_all[vi].astype(np.float32)
        if alignment is not None:
            pm = _to_gps(pm, *alignment)
        xyz, c = _pointmap_to_cloud(pm, pc, conf_threshold)
        if len(xyz):
            all_pts.append(xyz)
            all_conf.append(c)
    if not all_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    pts = np.concatenate(all_pts)
    conf = np.concatenate(all_conf)
    return _postprocess(pts, conf, n_points, seed)


# ── TRPM inference ────────────────────────────────────────────────────────────

def _unproject_depth_to_world(
    depth: np.ndarray,
    K: np.ndarray,
    T_c2w: np.ndarray,
) -> np.ndarray:
    """Unproject depth [H, W] using intrinsics K [3,3] and c2w pose [4,4] → world points [H*W, 3]."""
    H, W = depth.shape
    fu, fv = K[0, 0], K[1, 1]
    cu, cv = K[0, 2], K[1, 2]
    us = np.arange(W, dtype=np.float64)
    vs = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    d = depth.astype(np.float64)
    x_cam = (uu - cu) / fu * d
    y_cam = (vv - cv) / fv * d
    z_cam = d
    pts_cam = np.stack([x_cam.ravel(), y_cam.ravel(), z_cam.ravel(), np.ones(H * W)], axis=1)  # [N, 4]
    pts_world = (T_c2w.astype(np.float64) @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)


def _is_depth_model(model: torch.nn.Module) -> bool:
    """Check if model is TRPMSmallCamDepth (has depth_head)."""
    from trpm.trpm_small_cam_depth import TRPMSmallCamDepth
    return isinstance(model, TRPMSmallCamDepth)


@torch.no_grad()
def predict_t2_cloud(
    model: torch.nn.Module,
    vggt_root: Path,
    triplet_id: str,
    variant: str,
    tau: float,
    device: str,
    conf_threshold: float,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run TRPM model, return predicted t2 cloud as (points [N,3], confidence [N,]).

    For depth models (TRPMSmallCamDepth): unprojects D2_hat using K2 + T2_c2w,
    matching how build_geometry_assets.py builds ground truth from depth maps.

    For other cam models: uses P2_cam_hat transformed to world space.
    For base models: uses P2_hat directly.
    """
    from trpm.trpm_small_cam import TRPMSmallCam
    is_cam = isinstance(model, TRPMSmallCam)
    is_depth = _is_depth_model(model)

    t1_dir = vggt_root / triplet_id / variant / "t1"
    t2_dir = vggt_root / triplet_id / variant / "t2"
    t3_dir = vggt_root / triplet_id / variant / "t3"
    if not all((d / "predictions" / "point_map.npy").exists() for d in (t1_dir, t2_dir, t3_dir)):
        return None

    nv1 = np.load(t1_dir / "predictions" / "point_map.npy", mmap_mode="r").shape[0]
    nv3 = np.load(t3_dir / "predictions" / "point_map.npy", mmap_mode="r").shape[0]
    align1 = _gps_alignment(t1_dir)
    align3 = _gps_alignment(t3_dir)

    # For depth unprojection: use t1 dataset_cameras intrinsics (pad-mode scaled)
    # + t1 VGGT extrinsics (per view c2w) + t1 GPS alignment
    ext1_all = np.load(t1_dir / "predictions" / "extrinsic.npy").astype(np.float64)  # [V, 3, 4]
    K1_pad = _compute_pad_mode_intrinsics(t1_dir)  # [3, 3]

    P1, C1, P3, C3 = [], [], [], []
    for vi in range(nv1):
        pm1, pc1 = _load_pointmaps(t1_dir / "predictions", vi)
        if align1 is not None:
            pm1 = _to_gps(pm1, *align1)
        P1.append(torch.from_numpy(pm1).permute(2, 0, 1))
        C1.append(torch.from_numpy(pc1).unsqueeze(0))
    for vi in range(nv3):
        pm3, pc3 = _load_pointmaps(t3_dir / "predictions", vi)
        if align3 is not None:
            pm3 = _to_gps(pm3, *align3)
        P3.append(torch.from_numpy(pm3).permute(2, 0, 1))
        C3.append(torch.from_numpy(pc3).unsqueeze(0))

    P3_t = torch.stack(P3).unsqueeze(0).to(device)
    C3_t = torch.stack(C3).unsqueeze(0).to(device)

    cam1_data = _load_camera_data(t1_dir) if is_cam else None
    cam2_data = _load_camera_data(t2_dir) if is_cam else None
    cam3_data = _load_camera_data(t3_dir) if is_cam else None
    if is_cam and (cam1_data is None or cam2_data is None or cam3_data is None):
        return None

    if is_cam:
        T1_all = torch.from_numpy(cam1_data[0]).to(device)
        T2_all = torch.from_numpy(cam2_data[0]).to(device)
        T3_t = torch.from_numpy(cam3_data[0]).unsqueeze(0).to(device)
        K2_all = torch.from_numpy(cam2_data[1]).to(device)
        K3_t = torch.from_numpy(cam3_data[1]).unsqueeze(0).to(device)
        T2_np = cam2_data[0]  # [V, 4, 4]
        K2_np = cam2_data[1]  # [V, 3, 3]

    tau_t = torch.tensor([[tau]], dtype=torch.float32, device=device)
    all_pts, all_conf = [], []

    if is_cam:
        # Batch all views in one forward pass for speed.
        V = nv1
        P1_batch = torch.stack(P1).to(device)
        C1_batch = torch.stack(C1).to(device)
        V3 = P3_t.shape[1]
        P3_batch = P3_t.expand(V, -1, -1, -1, -1)
        C3_batch = C3_t.expand(V, -1, -1, -1, -1)
        T2_batch = T2_all[:V]
        T1_batch = T1_all[:V]
        T3_batch = T3_t.expand(V, -1, -1, -1)
        K2_batch = K2_all[:V]
        K3_batch = K3_t.expand(V, -1, -1, -1)
        tau_batch = tau_t.expand(V, -1)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out = model(
                P1_batch, C1_batch, P3_batch, C3_batch,
                T2_batch, T1_batch, T3_batch,
                K2_batch, K3_batch, tau_batch,
            )

        if is_depth:
            d2_all = out["D2_hat"].float().cpu().numpy()
            gate_all = out["G"].float().cpu().numpy()
            for vi in range(V):
                d2_hat = d2_all[vi, 0]
                gate = gate_all[vi, 0]
                # Unproject with t1 pad-mode intrinsics → camera frame
                # Then t1 VGGT extrinsic c2w → VGGT world
                c2w_vi = _vggt_ext_to_c2w(ext1_all[vi])
                pts_vggt = _unproject_depth_to_world(d2_hat, K1_pad, c2w_vi)
                # Apply t1 GPS alignment → GPS/dataset world
                if align1 is not None:
                    s, R, t_vec = align1
                    pts_world = (s * R @ pts_vggt.astype(np.float64).T + t_vec[:, None]).T.astype(np.float32)
                else:
                    pts_world = pts_vggt
                conf_flat = gate.ravel()
                depth_flat = d2_hat.ravel()
                valid = (
                    np.isfinite(pts_world).all(axis=1)
                    & np.isfinite(conf_flat)
                    & (conf_flat >= conf_threshold)
                    & (depth_flat > 0)
                )
                xyz = pts_world[valid].astype(np.float32)
                c = conf_flat[valid].astype(np.float32)
                if len(xyz):
                    all_pts.append(xyz)
                    all_conf.append(c)
        else:
            p2_cam_all = out["P2_cam_hat"].float()
            gate_all = out["G"].float().cpu().numpy()
            R2 = T2_batch[:, :3, :3]
            c2 = T2_batch[:, :3, 3]
            pts_world = (torch.bmm(R2, p2_cam_all.view(V, 3, -1)) + c2.unsqueeze(-1))
            pts_world = pts_world.view(V, 3, -1).permute(0, 2, 1).cpu().numpy()
            H, W = p2_cam_all.shape[2], p2_cam_all.shape[3]
            for vi in range(V):
                p2_hat = pts_world[vi].reshape(H, W, 3)
                p2_conf = gate_all[vi, 0]
                xyz, c = _pointmap_to_cloud(p2_hat, p2_conf, conf_threshold)
                if len(xyz):
                    all_pts.append(xyz)
                    all_conf.append(c)
    else:
        # Base model: per-view loop (base model takes single [B,3,H,W] for P3)
        P3_flat = P3_t.squeeze(0)
        C3_flat = C3_t.squeeze(0)
        for vi in range(nv1):
            P1_b = P1[vi].unsqueeze(0).to(device)
            C1_b = C1[vi].unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = model(P1_b, C1_b, P3_flat[:1], C3_flat[:1], tau_t)
            p2_hat = out["P2_hat"].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            p2_conf = out["G"].squeeze().float().cpu().numpy()
            xyz, c = _pointmap_to_cloud(p2_hat, p2_conf, conf_threshold)
            if len(xyz):
                all_pts.append(xyz)
                all_conf.append(c)

    if not all_pts:
        return None
    pts = np.concatenate(all_pts)
    conf = np.concatenate(all_conf)
    return _postprocess(pts, conf, n_points, seed)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Setup distributed if launched via torchrun
    distributed = _is_distributed()
    if distributed:
        local_rank, world_size = _setup_distributed()
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        world_size = 1
        device = None  # will be set below

    args = parse_args()
    cfg = build_config(args)

    if not distributed:
        device = f"cuda" if torch.cuda.is_available() else "cpu"
        if cfg.get("device") and cfg["device"] != "auto":
            device = cfg["device"]

    rank = _get_rank()
    log(f"build_predicted_geometry: world_size={world_size} device={device}")

    all_folds = build_all_folds(cfg["triplets_path"])
    output_root = cfg["output_root"]
    if _is_main():
        output_root.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    vggt_root = cfg["vggt_output_root"]
    geometry_assets_root = cfg["geometry_assets_root"]
    conf_threshold = cfg["conf_threshold"]
    pred_conf_threshold = cfg["pred_conf_threshold"]
    n_points = cfg["n_points"]
    seed = cfg["seed"]
    skip_existing = cfg.get("skip_existing", False)

    # Collect all work items: (fold_id, protocol, triplet, variant, tau, checkpoint)
    work_items: list[dict[str, Any]] = []
    for protocol in cfg["protocols"]:
        for fold in all_folds.get(protocol, []):
            if fold["crop"] not in cfg["crops"]:
                continue
            if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                continue
            if not fold["test_triplets"]:
                continue

            fold_id = fold["fold_id"]
            checkpoint = cfg["runs_root"] / protocol / fold_id / "best_model.pt"
            if not checkpoint.exists():
                log(f"skip fold={fold_id}: no checkpoint at {checkpoint}")
                continue

            for triplet in fold["test_triplets"]:
                left_date = triplet["left_date"]
                middle_date = triplet["middle_date"]
                right_date = triplet["right_date"]
                crop = triplet["crop"]
                triplet_id = f"{left_date}_{middle_date}_{right_date}_{crop}"
                tau = float(triplet["tau"])

                variants = _list_variants(vggt_root, triplet_id)
                for variant in variants:
                    t2_out = output_root / triplet_id / variant / "t2" / "point_cloud_clean.npz"
                    if skip_existing and t2_out.exists():
                        continue
                    work_items.append({
                        "fold_id": fold_id,
                        "protocol": protocol,
                        "triplet_id": triplet_id,
                        "variant": variant,
                        "tau": tau,
                        "checkpoint": checkpoint,
                    })

    log(f"total work items: {len(work_items)}")

    # Distribute work items round-robin across ranks
    my_items = work_items[rank::world_size]
    log_all(f"rank={rank} processing {len(my_items)} items")

    # Group by checkpoint to avoid reloading model
    items_by_checkpoint: dict[Path, list[dict]] = {}
    for item in my_items:
        items_by_checkpoint.setdefault(item["checkpoint"], []).append(item)

    summary: list[dict[str, Any]] = []

    for checkpoint, items in items_by_checkpoint.items():
        model_cls = _load_model_class(cfg.get("model_class", "trpm.trpm_small_cam.TRPMSmallCam"))
        model = model_cls(**cfg.get("model_kwargs", {})).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        log_all(f"loaded model from {checkpoint} ({len(items)} items)")

        for item in tqdm(items, desc=f"rank={rank}", disable=(rank != 0)):
            triplet_id = item["triplet_id"]
            variant = item["variant"]
            tau = item["tau"]

            # Ensure t1/t3 ground truth exists in geometry_assets_root
            for date_label in ("t1", "t3"):
                gt_out = geometry_assets_root / triplet_id / variant / date_label / "point_cloud_clean.npz"
                if gt_out.exists():
                    continue
                date_dir = vggt_root / triplet_id / variant / date_label
                if not (date_dir / "predictions" / "point_map.npy").exists():
                    continue
                pts_gt, conf_gt = _load_date_cloud(date_dir, conf_threshold, n_points, seed)
                if len(pts_gt) == 0:
                    continue
                gt_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    gt_out,
                    points_raw=pts_gt,
                    points_aligned=pts_gt,
                    points_normalized=pts_gt,
                    confidence=conf_gt,
                )

            # Run model for t2 prediction
            result = predict_t2_cloud(
                model, vggt_root, triplet_id, variant, tau, device,
                pred_conf_threshold, n_points, seed,
            )
            if result is None:
                continue
            pts, conf = result
            t2_out = output_root / triplet_id / variant / "t2" / "point_cloud_clean.npz"
            t2_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                t2_out,
                points_raw=pts,
                points_aligned=pts,
                points_normalized=pts,
                confidence=conf,
            )

            summary.append({
                "triplet_id": triplet_id,
                "variant": variant,
                "n_points_t2": int(len(pts)),
                "status": "completed",
            })

    # Each rank writes its own partial summary
    if distributed:
        dist.barrier()
        write_json(output_root / f"build_summary_rank{rank}.json", summary)
        dist.barrier()
        # Rank 0 merges all partial summaries
        if _is_main():
            all_summary = []
            for r in range(world_size):
                part_path = output_root / f"build_summary_rank{r}.json"
                if part_path.exists():
                    all_summary.extend(json.loads(part_path.read_text()))
                    part_path.unlink()
            write_json(output_root / "build_summary.json", all_summary)
            log(f"done. {len(all_summary)} variants processed. summary → {output_root / 'build_summary.json'}")
    else:
        write_json(output_root / "build_summary.json", summary)
        log(f"done. {len(summary)} variants processed. summary → {output_root / 'build_summary.json'}")

    if distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()

