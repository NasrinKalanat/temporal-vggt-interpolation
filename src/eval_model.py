"""Evaluate TemporalVGGTv1 checkpoints on LOTO test folds.

Evaluates the trained model (best_model.pt per fold from runs_root) and five
point-map baselines (B0-B4) on each LOTO test split.

Baselines use VGGT t1/t3 point maps from vggt_output_root as inputs.
All metrics are computed against VGGT t2 teacher predictions as reference.

PointMap-L1/L2 are only computed for the trained model (pixel-aligned maps).
Baselines get NaN for those two metrics.

Usage:
    # Evaluate trained run + baselines:
    python src/eval_model.py --config configs/train_model_v1.yaml \\
        --runs-root runs/model_v1 --output-root eval/model_v1

    # Baselines only:
    python src/eval_model.py --config configs/train_model_v1.yaml \\
        --output-root eval/baselines

    # Single fold:
    python src/eval_model.py --config configs/train_model_v1.yaml \\
        --runs-root runs/model_v1 --crop corn --protocol strict --test-date 20230831
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.triplet_dataset import TemporalTripletDataset
from dataset.vggt_feature_cache import VGGTFeatureCache
from losses.geometry import compute_metrics
from loto import build_loto_folds, compute_tau, load_triplets


# ─── config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "vggt_output_root": "vggt_outputs/camera_consistent_triplets_v16_o4",
    "runs_root": None,
    "output_root": "eval/model_v1",
    "protocols": ["target_date", "strict"],
    "crops": ["corn", "soybean"],
    "test_date": None,
    "seed": 42,
    "device": "cuda:0",
    "conf_threshold": 0.02,
    "image_preprocess_mode": "pad",
    "n_points": 50_000,
    "distance_threshold": 0.05,
    "voxel_size": 0.05,
    "eval_alpha": 0.5,
    "eval_beta": 0.5,
    "model_module": None,
    "model_class": None,
    "model_kwargs": {},
    # Used when t1/t3 predictions are absent (t2_only inference runs).
    "vggt_model_id": "facebook/VGGT-1B",
    "vggt_device": "auto",
    "feature_cache_root": "/vast/xjia/nak168/vggt_cache",  # null to disable
}

BASELINES = [
    "B0_t1_date_copy",
    "B1_t3_date_copy",
    "B2_nearest_date_copy",
    "B3_linear_interpolation",
    "B4_temporal_weighted_union",
]


# ─── utilities ────────────────────────────────────────────────────────────────

def _no_collate(x: list) -> Any:
    """Return the single sample unchanged (bypass default tensor stacking)."""
    return x[0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required: pip install pyyaml") from exc
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/train_model_v1.yaml"))
    p.add_argument("--runs-root", type=Path, default=None,
                   help="runs/ directory from train.py (contains fold subdirs with best_model.pt)")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--crop", choices=["corn", "soybean"], action="append", default=None)
    p.add_argument("--protocol", choices=["target_date", "strict"], action="append", default=None)
    p.add_argument("--test-date", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--save-clouds", action="store_true",
                   help="Save predicted and reference point clouds (.npy) for visualization")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of test samples evaluated per fold (useful for quick checks)")
    p.add_argument("--vggt-output-root", "--vggt_output_root", type=Path, default=None)
    p.add_argument("--conf-threshold", "--conf_threshold", type=float, default=None)
    p.add_argument("--distance-threshold", "--distance_threshold", type=float, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config.exists():
        cfg.update(read_yaml(args.config))
    if args.runs_root is not None:
        cfg["runs_root"] = args.runs_root
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.crop is not None:
        cfg["crops"] = args.crop
    if args.protocol is not None:
        cfg["protocols"] = args.protocol
    if args.test_date is not None:
        cfg["test_date"] = args.test_date
    if args.device is not None:
        cfg["device"] = args.device
    cfg["save_clouds"] = args.save_clouds
    cfg["max_samples"] = args.max_samples
    if args.vggt_output_root is not None:
        cfg["vggt_output_root"] = args.vggt_output_root
    if args.conf_threshold is not None:
        cfg["conf_threshold"] = args.conf_threshold
    if args.distance_threshold is not None:
        cfg["distance_threshold"] = args.distance_threshold
    cfg["triplets_path"] = Path(cfg["triplets_path"])
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["output_root"] = Path(cfg["output_root"])
    if cfg.get("runs_root"):
        cfg["runs_root"] = Path(cfg["runs_root"])
    return cfg


# ─── triplet helpers ──────────────────────────────────────────────────────────

def _triplet_id(t: dict[str, Any]) -> str:
    return f"{t['left_date']}_{t['middle_date']}_{t['right_date']}_{t['crop']}"


def fold_test_indices(fold: dict[str, Any], dataset: TemporalTripletDataset) -> list[int]:
    test_ids = {_triplet_id(t) for t in fold["test_triplets"]}
    return [i for i, e in enumerate(dataset.index) if e["triplet_id"] in test_ids]


def triplet_is_adjacent(triplet_entry: dict[str, Any], all_dates: list[str]) -> bool:
    """True if t1 and t3 are the immediate neighbours of t2 in the sorted date list."""
    sorted_dates = sorted(all_dates)
    t1, t2, t3 = triplet_entry["t1_date"], triplet_entry["t2_date"], triplet_entry["t3_date"]
    try:
        idx = sorted_dates.index(t2)
    except ValueError:
        return False
    prev = sorted_dates[idx - 1] if idx > 0 else None
    nxt  = sorted_dates[idx + 1] if idx < len(sorted_dates) - 1 else None
    return t1 == prev and t3 == nxt


# ─── Umeyama alignment helpers ───────────────────────────────────────────────

def _umeyama_similarity(
    src: np.ndarray, dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, R, t) such that dst ≈ scale * R @ src + t."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
    cov = (dst_c.T @ src_c) / n
    U, S_vals, Vt = np.linalg.svd(cov)
    d = float(np.sign(np.linalg.det(U @ Vt)))
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).astype(np.float64)
    scale = float(np.sum(S_vals * np.diag(D)) / var_src) if var_src > 0 else 1.0
    t = (mu_d - scale * R @ mu_s).astype(np.float64)
    return scale, R, t


def _vggt_to_dataset_alignment(
    date_dir: Path,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama (scale, R, t) mapping VGGT world space → NeRFStudio GPS world space.

    Aligns VGGT camera centers (from extrinsic.npy) to NeRFStudio camera centers
    (from dataset_cameras.json transform_matrices), same approach as
    build_geometry_assets.py.
    """
    extrinsics = np.load(date_dir / "predictions" / "extrinsic.npy")  # [S, 3, 4]
    cams = json.loads((date_dir / "dataset_cameras.json").read_text())
    frames = cams["frames"]
    n = min(len(extrinsics), len(frames))
    vggt_centers = np.array(
        [-extrinsics[i, :, :3].T @ extrinsics[i, :, 3] for i in range(n)],
        dtype=np.float64,
    )
    dataset_centers = np.array(
        [np.array(frames[i]["transform_matrix"], dtype=np.float64)[:3, 3] for i in range(n)],
        dtype=np.float64,
    )
    return _umeyama_similarity(vggt_centers, dataset_centers)


# ─── point map / cloud helpers ────────────────────────────────────────────────

def load_pointmap_cloud(
    date_dir: Path,
    conf_threshold: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    """Load point_map.npy + confidence, filter, subsample → [N, 3] float32."""
    pm  = np.load(date_dir / "predictions" / "point_map.npy").astype(np.float32)  # [S, H, W, 3]
    pc  = np.load(date_dir / "predictions" / "point_confidence.npy").astype(np.float32)
    if pc.ndim == 4:
        pc = pc[..., 0]
    pts = pm.reshape(-1, 3)
    conf = pc.reshape(-1)
    pts = pts[conf >= conf_threshold]
    if n_points > 0 and len(pts) > n_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return pts


_lazy_vggt_runner: dict[tuple[str, str], Any] = {}


def _get_lazy_vggt_runner(model_id: str, device: str) -> Any:
    key = (model_id, device)
    if key not in _lazy_vggt_runner:
        from vggt_pipeline.execute_vggt import get_vggt_runner
        log(f"loading VGGT runner for live t1/t3 inference model_id={model_id} device={device}")
        _lazy_vggt_runner[key] = get_vggt_runner(model_id=model_id, device=device, use_cache=True)
    return _lazy_vggt_runner[key]


def load_or_infer_pointcloud(
    date_dir: Path,
    conf_threshold: float,
    n_points: int,
    seed: int,
    vggt_model_id: str = "facebook/VGGT-1B",
    vggt_device: str = "auto",
    image_preprocess_mode: str = "pad",
) -> np.ndarray:
    """Load point cloud from pre-computed predictions, or run VGGT live if missing.

    Falls back to live inference when predictions/point_map.npy is absent, which
    happens for t1/t3 dirs produced by a t2_only inference run.  Image paths are
    read from the selected_images.json that is always written by run_vggt_inference.
    """
    pred_path = date_dir / "predictions" / "point_map.npy"
    if pred_path.exists():
        return load_pointmap_cloud(date_dir, conf_threshold, n_points, seed)

    from vggt_pipeline.execute_vggt import run_vggt_inference_in_memory
    selected = json.loads((date_dir / "selected_images.json").read_text())
    image_paths = [entry["image_path"] for entry in selected]
    runner = _get_lazy_vggt_runner(vggt_model_id, vggt_device)
    preds = run_vggt_inference_in_memory(image_paths, runner, image_preprocess_mode)

    pm = preds["point_map"].numpy()
    pc = preds["point_confidence"].numpy()
    if pc.ndim == 4:
        pc = pc[..., 0]
    pts = pm.reshape(-1, 3)
    conf = pc.reshape(-1)
    pts = pts[conf >= conf_threshold]
    if n_points > 0 and len(pts) > n_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return pts.astype(np.float32)


def apply_baseline(
    name: str,
    pts_t1: np.ndarray,
    pts_t3: np.ndarray,
    tau: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    if name == "B0_t1_date_copy":
        return pts_t1
    if name == "B1_t3_date_copy":
        return pts_t3
    if name == "B2_nearest_date_copy":
        return pts_t1 if tau <= 0.5 else pts_t3
    if name == "B3_linear_interpolation":
        n = min(len(pts_t1), len(pts_t3), n_points)
        rng = np.random.default_rng(seed)
        i1 = rng.choice(len(pts_t1), n, replace=False)
        i3 = rng.choice(len(pts_t3), n, replace=False)
        return (1.0 - tau) * pts_t1[i1] + tau * pts_t3[i3]
    if name == "B4_temporal_weighted_union":
        n1 = max(1, round(n_points * (1.0 - tau)))
        n3 = max(1, round(n_points * tau))
        rng = np.random.default_rng(seed)
        idx1 = rng.choice(len(pts_t1), min(n1, len(pts_t1)), replace=False)
        idx3 = rng.choice(len(pts_t3), min(n3, len(pts_t3)), replace=False)
        return np.concatenate([pts_t1[idx1], pts_t3[idx3]], axis=0)
    raise ValueError(f"Unknown baseline: {name}")


def load_raw_confidence(date_dir: Path) -> np.ndarray | None:
    """Load raw flattened confidence array from predictions/; None if file missing."""
    path = date_dir / "predictions" / "point_confidence.npy"
    if not path.exists():
        return None
    pc = np.load(path).astype(np.float32)
    if pc.ndim == 4:
        pc = pc[..., 0]
    return pc.reshape(-1)


def _conf_stats(conf: np.ndarray, threshold: float) -> dict:
    """Confidence distribution stats + filtering summary."""
    total  = int(len(conf))
    passed = int((conf >= threshold).sum())
    return {
        "total": total,
        "passed": passed,
        "fraction_passed": float(passed / total) if total > 0 else 0.0,
        "mean":   float(conf.mean()),
        "median": float(np.median(conf)),
        "min":    float(conf.min()),
        "max":    float(conf.max()),
        "p25":    float(np.percentile(conf, 25)),
        "p75":    float(np.percentile(conf, 75)),
        "p95":    float(np.percentile(conf, 95)),
    }


def pointmap_to_cloud(
    pm: torch.Tensor,
    conf: torch.Tensor,
    conf_threshold: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    """Flatten [Q, H, W, 3] point map + [Q, H, W] conf → filtered [N, 3] numpy."""
    pts_np = pm.reshape(-1, 3).cpu().numpy().astype(np.float32)
    conf_np = conf.reshape(-1).cpu().numpy()
    mask = conf_np >= conf_threshold
    pts_np = pts_np[mask]
    if n_points > 0 and len(pts_np) > n_points:
        rng = np.random.default_rng(seed)
        pts_np = pts_np[rng.choice(len(pts_np), n_points, replace=False)]
    return pts_np


def pointmap_l1_l2(
    pred_pm: torch.Tensor,
    target_pm: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """PointMap-L1 and L2 averaged over valid (masked) pixels across all query views."""
    diff = pred_pm - target_pm          # [Q, H, W, 3]
    valid = mask.bool()                 # [Q, H, W]
    if valid.sum() == 0:
        return float("nan"), float("nan")
    d = diff[valid]                     # [K, 3]
    l1 = float(d.abs().mean())
    l2 = float(d.norm(dim=-1).mean())
    return l1, l2


# ─── model ────────────────────────────────────────────────────────────────────

def load_model(cfg: dict[str, Any], checkpoint: Path, device: str) -> torch.nn.Module:
    module = importlib.import_module(cfg["model_module"])
    cls = getattr(module, cfg["model_class"])

    # __init__ loads HuggingFace base weights and freshly initialises trainable
    # components (LoRA, time_encoder, query_grid, point_head, camera_head).
    # load_state_dict below unconditionally overwrites EVERY parameter with the
    # trained checkpoint, so HuggingFace init values do not survive into eval.
    model = cls(**cfg.get("model_kwargs", {})).to(device)

    # Snapshot a trainable parameter before loading so we can verify the
    # checkpoint actually changed it (guards against silent load failures).
    probe_name, probe_before = next(
        ((n, p.detach().clone()) for n, p in model.named_parameters() if p.requires_grad),
        (None, None),
    )

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    # strict=True (the default) raises if any key is missing or unexpected,
    # catching architecture mismatches between the saved run and this config.
    model.load_state_dict(state, strict=True)
    del state

    # Explicitly move to device after load_state_dict in case any sub-module
    # or buffer ended up on CPU during checkpoint loading.
    model = model.to(device)

    actual_device = next(model.parameters()).device
    log(f"  model device after load: {actual_device}")
    expected = torch.device(device)
    device_ok = (
        actual_device.type == expected.type
        and (expected.index is None or actual_device.index == expected.index)
    )
    if not device_ok:
        raise RuntimeError(
            f"Model is on {actual_device} but expected {device}. "
            "Check that CUDA is available and device string is correct."
        )

    if probe_name is not None and probe_before is not None:
        probe_after = dict(model.named_parameters())[probe_name].detach()
        if torch.equal(probe_before, probe_after):
            raise RuntimeError(
                f"Checkpoint load verification failed: trainable parameter "
                f"'{probe_name}' is identical before and after load_state_dict. "
                "The checkpoint may not contain trained weights."
            )

    model.eval()
    if actual_device.type == "cuda":
        mem_gb = torch.cuda.memory_allocated(actual_device) / 1024 ** 3
        log(f"  GPU memory after model load: {mem_gb:.2f} GB")
    return model


# ─── per-sample evaluation ────────────────────────────────────────────────────

def evaluate_sample(
    sample: dict[str, Any],
    model: torch.nn.Module | None,
    device: str,
    cfg: dict[str, Any],
    cloud_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one sample: run baselines and model, compute all metrics."""
    conf_thr  = cfg.get("conf_threshold", 0.02)
    n_points  = cfg.get("n_points", 50_000)
    seed      = cfg.get("seed", 42)
    threshold = cfg.get("distance_threshold", 0.05)
    voxel_sz  = cfg.get("voxel_size", 0.05)
    alpha     = cfg.get("eval_alpha", 0.5)
    beta      = cfg.get("eval_beta", 0.5)

    variant_dir: Path = sample["variant_dir"]
    tau = compute_tau(sample["t1_date"], sample["t2_date"], sample["t3_date"])

    # Load t1 / t3 point clouds for baselines (falls back to live VGGT if t2_only run).
    vggt_model_id = cfg.get("vggt_model_id", "facebook/VGGT-1B")
    vggt_device   = cfg.get("vggt_device", "auto")
    img_mode      = cfg.get("image_preprocess_mode", "pad")
    pts_t1 = load_or_infer_pointcloud(
        variant_dir / "t1", conf_thr, n_points, seed, vggt_model_id, vggt_device, img_mode,
    )
    pts_t3 = load_or_infer_pointcloud(
        variant_dir / "t3", conf_thr, n_points, seed, vggt_model_id, vggt_device, img_mode,
    )

    # Reference: teacher t2 point cloud.
    pts_t2_ref = load_pointmap_cloud(variant_dir / "t2", conf_thr, n_points, seed)

    # Confidence stats (before and after threshold) for precomputed sources.
    conf_stats: dict = {"conf_threshold": conf_thr}
    for src_key, src_dir in [
        ("t1",     variant_dir / "t1"),
        ("t3",     variant_dir / "t3"),
        ("t2_ref", variant_dir / "t2"),
    ]:
        raw = load_raw_confidence(src_dir)
        if raw is not None:
            conf_stats[src_key] = _conf_stats(raw, conf_thr)
            s = conf_stats[src_key]
            log(f"  conf [{src_key:6s}]  mean={s['mean']:.3f}  median={s['median']:.3f}"
                f"  min={s['min']:.3f}  max={s['max']:.3f}"
                f"  passed={s['passed']}/{s['total']} ({s['fraction_passed']:.1%})")

    row: dict[str, Any] = {
        "triplet_id": sample["triplet_id"],
        "crop": sample["crop"],
        "t1_date": sample["t1_date"],
        "t2_date": sample["t2_date"],
        "t3_date": sample["t3_date"],
        "variant": sample["variant"],
        "tau": tau,
        "is_adjacent": sample.get("is_adjacent", None),
    }

    for baseline in BASELINES:
        pred = apply_baseline(baseline, pts_t1, pts_t3, tau, n_points, seed)
        row[baseline] = compute_metrics(pred, pts_t2_ref, threshold=threshold,
                                        voxel_size=voxel_sz, alpha=alpha, beta=beta)

    if model is not None:
        # Build a single-sample batch (same structure as DataLoader output).
        batch = {
            "images_t1": sample["images_t1"].unsqueeze(0).to(device),  # [1, V1, 3, H, W]
            "images_t3": sample["images_t3"].unsqueeze(0).to(device),
            "t1_cache_key": sample.get("t1_cache_key"),
            "t3_cache_key": sample.get("t3_cache_key"),
            "camera_t1": {k: v.unsqueeze(0) for k, v in sample["camera_t1"].items()},
            "camera_t3": {k: v.unsqueeze(0) for k, v in sample["camera_t3"].items()},
            "camera_t2_query": {k: v.unsqueeze(0) for k, v in sample["camera_t2_query"].items()},
            "date_t1": sample["date_t1"].unsqueeze(0),
            "date_t2": sample["date_t2"].unsqueeze(0),
            "date_t3": sample["date_t3"].unsqueeze(0),
            "t1_day": sample["t1_day"].unsqueeze(0),
            "t2_day": sample["t2_day"].unsqueeze(0),
            "t3_day": sample["t3_day"].unsqueeze(0),
        }

        with torch.no_grad():
            if str(device).startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch)
            else:
                outputs = model(batch)

        pred_points = outputs["pred_points"][0]  # [Q, H, W, 3]
        pred_conf   = outputs["pred_conf"][0]    # [Q, H, W]
        conf_np_pred = pred_conf.float().reshape(-1).cpu().numpy()
        conf_stats["model_pred"] = _conf_stats(conf_np_pred, conf_thr)
        s = conf_stats["model_pred"]
        log(f"  conf [model ]  mean={s['mean']:.3f}  median={s['median']:.3f}"
            f"  min={s['min']:.3f}  max={s['max']:.3f}"
            f"  passed={s['passed']}/{s['total']} ({s['fraction_passed']:.1%})")

        # Teacher maps for PointMap-L1/L2 — align resolution if needed.
        target_pm   = sample["target_point_maps_t2"].to(device)    # [Q, H, W, 3]
        target_mask = sample["target_masks_t2"].to(device)         # [Q, H, W]

        if target_pm.shape[-3:-1] != pred_points.shape[-3:-1]:
            Q = pred_points.shape[0]
            Hp, Wp = pred_points.shape[-3], pred_points.shape[-2]
            target_pm = F.interpolate(
                target_pm.permute(0, 3, 1, 2),
                size=(Hp, Wp), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()
            target_mask = F.interpolate(
                target_mask.unsqueeze(1).float(),
                size=(Hp, Wp), mode="nearest",
            ).squeeze(1)

        l1, l2 = pointmap_l1_l2(pred_points.float(), target_pm.float(), target_mask)

        pred_cloud = pointmap_to_cloud(pred_points.float(), pred_conf.float(), conf_thr, n_points, seed)
        metrics = compute_metrics(pred_cloud, pts_t2_ref, threshold=threshold,
                                  voxel_size=voxel_sz, alpha=alpha, beta=beta)
        metrics["pointmap_l1"] = l1
        metrics["pointmap_l2"] = l2
        metrics["n_pred_points"] = int(len(pred_cloud))
        metrics["n_ref_points"] = int(len(pts_t2_ref))
        metrics["pred_conf_mean"] = float(pred_conf.mean())
        metrics["pred_conf_median"] = float(pred_conf.median())
        metrics["pred_conf_min"] = float(pred_conf.min())
        metrics["pred_conf_max"] = float(pred_conf.max())
        row["model"] = metrics

        if cloud_output_dir is not None:
            cloud_output_dir.mkdir(parents=True, exist_ok=True)
            key = f"{row['t1_date']}_{row['t2_date']}_{row['t3_date']}_{row['crop']}_{sample['variant']}"

            # Ref cloud: GT Umeyama — t2 VGGT space → GPS.
            scale_r, R_r, t_r = _vggt_to_dataset_alignment(variant_dir / "t2")

            def _to_gps_ref(pts: np.ndarray) -> np.ndarray:
                return (scale_r * R_r @ pts.astype(np.float64).T + t_r[:, None]).T.astype(np.float32)

            # Pred cloud: Umeyama from predicted camera centers → GPS when available,
            # otherwise fall back to the same GT alignment as ref.
            if "pred_extrinsic" in outputs:
                pred_cam = outputs["pred_extrinsic"][0, :, :3].cpu().numpy().astype(np.float64)  # [Q, 3]
                gps_cam  = sample["camera_t2_query"]["transform_matrix"][:, :3, 3].numpy().astype(np.float64)  # [Q, 3]
                scale_p, R_p, t_p = _umeyama_similarity(pred_cam, gps_cam)

                def _to_gps_pred(pts: np.ndarray) -> np.ndarray:
                    return (scale_p * R_p @ pts.astype(np.float64).T + t_p[:, None]).T.astype(np.float32)
            else:
                _to_gps_pred = _to_gps_ref

            np.save(cloud_output_dir / f"{key}_pred.npy", _to_gps_pred(pred_cloud))
            np.save(cloud_output_dir / f"{key}_ref.npy", _to_gps_ref(pts_t2_ref))

    row["conf_stats"] = conf_stats
    return row


# ─── aggregation ──────────────────────────────────────────────────────────────

def _mean_metrics(rows: list[dict], method: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if method not in row:
            continue
        for k, v in row[method].items():
            if isinstance(v, float) and not np.isnan(v):
                buckets.setdefault(k, []).append(v)
    return {k: float(np.mean(vs)) for k, vs in buckets.items()}


def aggregate_results(
    rows: list[dict],
    method_keys: list[str],
) -> dict[str, Any]:
    """Return aggregated metrics split by adjacency and overall."""
    adjacent    = [r for r in rows if r.get("is_adjacent") is True]
    multi_gap   = [r for r in rows if r.get("is_adjacent") is False]

    out: dict[str, Any] = {"overall": {}, "adjacent": {}, "multi_gap": {}}
    for method in method_keys:
        out["overall"][method]   = _mean_metrics(rows, method)
        out["adjacent"][method]  = _mean_metrics(adjacent, method)
        out["multi_gap"][method] = _mean_metrics(multi_gap, method)
    return out


# ─── fold evaluation ──────────────────────────────────────────────────────────

def evaluate_fold(
    fold: dict[str, Any],
    dataset: TemporalTripletDataset,
    model: torch.nn.Module | None,
    cfg: dict[str, Any],
    device: str,
    all_dates_by_crop: dict[str, list[str]],
    output_dir: Path,
) -> dict[str, Any]:
    test_idx = fold_test_indices(fold, dataset)
    if not test_idx:
        log(f"  fold={fold['fold_id']}: no test samples found, skipping")
        return {"fold_id": fold["fold_id"], "status": "skipped_no_data"}

    crop = fold["crop"]
    all_dates = all_dates_by_crop.get(crop, [])

    method_keys = list(BASELINES) + (["model"] if model is not None else [])

    cloud_dir = (output_dir / "clouds") if cfg.get("save_clouds") and model is not None else None

    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        test_idx = test_idx[:max_samples]

    nw = cfg.get("num_workers", 2)
    loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=1, shuffle=False,
        num_workers=nw, pin_memory=False,
        persistent_workers=nw > 0,
        collate_fn=_no_collate,
    )

    if model is not None:
        log(f"  model loaded — will run inference on {len(test_idx)} samples")
    else:
        log(f"  no model — evaluating baselines only on {len(test_idx)} samples")

    first_inference_done = False
    rows: list[dict] = []
    pbar = tqdm(
        zip(test_idx, loader), total=len(test_idx),
        desc=fold["fold_id"], leave=False, dynamic_ncols=True,
    )
    for i, sample in pbar:
        sample["variant_dir"] = dataset.index[i]["variant_dir"]
        sample["is_adjacent"] = triplet_is_adjacent(dataset.index[i], all_dates)
        try:
            row = evaluate_sample(sample, model, device, cfg, cloud_output_dir=cloud_dir)
            if model is not None and not first_inference_done:
                first_inference_done = True
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / 1024 ** 3
                    log(f"  first inference done — GPU memory in use: {mem_gb:.2f} GB")
            rows.append(row)
        except Exception as e:
            log(
                f"  skip {dataset.index[i]['triplet_id']}"
                f" variant={dataset.index[i]['variant']}: {e}"
            )

    aggregated = aggregate_results(rows, method_keys)

    # Aggregate per-source conf stats across all samples and save to conf_stats.json.
    conf_stats_buckets: dict[str, dict[str, list]] = {}
    conf_thr_val = None
    for row in rows:
        cs = row.get("conf_stats", {})
        conf_thr_val = conf_thr_val or cs.get("conf_threshold")
        for src, stats in cs.items():
            if src == "conf_threshold" or not isinstance(stats, dict):
                continue
            for k, v in stats.items():
                conf_stats_buckets.setdefault(src, {}).setdefault(k, []).append(v)
    conf_stats_summary = {"conf_threshold": conf_thr_val}
    for src, metrics in conf_stats_buckets.items():
        conf_stats_summary[src] = {k: float(np.mean(vs)) for k, vs in metrics.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "conf_stats.json", conf_stats_summary)

    result = {
        "fold_id": fold["fold_id"],
        "crop": fold["crop"],
        "protocol": fold["protocol"],
        "test_date": fold["test_date"],
        "n_test_samples": len(rows),
        "aggregated": aggregated,
        "sample_rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "eval_result.json", result)

    # Print summary for each method.
    for method in method_keys:
        m = aggregated["overall"].get(method, {})
        log(f"  {fold['fold_id']} [{method}]  "
            f"chamfer={m.get('asymmetric_chamfer', float('nan')):.4f}  "
            f"f1={m.get('f1', float('nan')):.4f}  "
            f"pm_l1={m.get('pointmap_l1', float('nan')):.4f}")

    return result


# ─── device info ──────────────────────────────────────────────────────────────

def _log_device_info(device: str) -> None:
    import platform, socket
    log(f"  host={socket.gethostname()}  pid={__import__('os').getpid()}"
        f"  python={platform.python_version()}  torch={torch.__version__}")
    if not torch.cuda.is_available():
        log("  CUDA not available — running on CPU")
        return
    n = torch.cuda.device_count()
    log(f"  CUDA available  visible_devices={n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        log(f"    cuda:{i}  {props.name}"
            f"  total={total/1024**3:.1f}GB"
            f"  free={free/1024**3:.1f}GB"
            f"  SM={props.major}.{props.minor}"
            f"  multiprocessors={props.multi_processor_count}")
    requested = torch.device(device)
    idx = requested.index if requested.index is not None else 0
    log(f"  using cuda:{idx} ({torch.cuda.get_device_name(idx)})")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)
    device = cfg["device"]

    log(f"device={device}  output={cfg['output_root']}")
    _log_device_info(device)

    cache_root = cfg.get("feature_cache_root")
    feature_cache = VGGTFeatureCache(cache_root) if cache_root else None
    log(f"VGGT feature cache: {cache_root or 'disabled'}")

    log("Scanning vggt_output_root...")
    dataset = TemporalTripletDataset(
        vggt_output_root=cfg["vggt_output_root"],
        image_preprocess_mode=cfg.get("image_preprocess_mode", "pad"),
        conf_threshold=cfg.get("conf_threshold", 0.02),
        num_query_views=cfg.get("model_kwargs", {}).get("num_query_views", 1),
        seed=cfg.get("seed", 42),
        feature_cache=feature_cache,
    )
    log(f"Dataset: {len(dataset)} completed variants")

    triplets = load_triplets(cfg["triplets_path"])

    # Build date lookup per crop for adjacency classification.
    all_dates_by_crop: dict[str, list[str]] = {}
    for t in triplets:
        c = t["crop"]
        all_dates_by_crop.setdefault(c, set()).update(
            [t["left_date"], t["middle_date"], t["right_date"]]
        )
    all_dates_by_crop = {c: sorted(ds) for c, ds in all_dates_by_crop.items()}

    has_model_cfg = cfg.get("model_module") and cfg.get("model_class")

    cfg["output_root"].mkdir(parents=True, exist_ok=True)
    write_json(cfg["output_root"] / "eval_config.json", {
        k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()
    })

    all_results: list[dict] = []
    for protocol in cfg["protocols"]:
        for crop in cfg["crops"]:
            folds = build_loto_folds(triplets, crop, protocol)
            for fold in folds:
                if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                    continue
                if not fold["test_triplets"]:
                    continue

                log(f"--- fold={fold['fold_id']} test_date={fold['test_date']} "
                    f"n_test_triplets={len(fold['test_triplets'])} ---")

                # Load model checkpoint for this fold (if runs_root provided).
                model: torch.nn.Module | None = None
                if has_model_cfg and cfg.get("runs_root"):
                    ckpt = cfg["runs_root"] / protocol / fold["fold_id"] / "best_model.pt"
                    if ckpt.exists():
                        log(f"  loading checkpoint: {ckpt}")
                        model = load_model(cfg, ckpt, device)
                        model.feature_cache = feature_cache
                    else:
                        log(f"  no checkpoint at {ckpt} — evaluating baselines only")

                fold_dir = cfg["output_root"] / protocol / fold["fold_id"]
                result = evaluate_fold(
                    fold, dataset, model, cfg, device,
                    all_dates_by_crop, fold_dir,
                )
                all_results.append({k: v for k, v in result.items() if k != "sample_rows"})

                # Free model VRAM between folds.
                del model
                torch.cuda.empty_cache()

    write_json(cfg["output_root"] / "eval_summary.json", all_results)
    log(f"Done. Summary: {cfg['output_root'] / 'eval_summary.json'}")

    # Print cross-fold aggregate per method.
    _print_summary(all_results)


def _print_summary(all_results: list[dict]) -> None:
    """Print overall and per-crop method averages across all folds."""
    from collections import defaultdict

    # Collect per-method per-split: list of metric dicts.
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for fold_res in all_results:
        for split in ("overall", "adjacent", "multi_gap"):
            for method, metrics in fold_res.get("aggregated", {}).get(split, {}).items():
                buckets[split][method].append(metrics)

    def _avg(mlist: list[dict]) -> dict[str, float]:
        keys = mlist[0].keys() if mlist else []
        return {k: float(np.nanmean([m[k] for m in mlist if k in m])) for k in keys}

    log("=" * 72)
    for split in ("overall", "adjacent", "multi_gap"):
        log(f"  [{split}]")
        for method, mlist in sorted(buckets[split].items()):
            m = _avg(mlist)
            log(f"    {method:35s}  "
                f"chamfer={m.get('asymmetric_chamfer', float('nan')):.4f}  "
                f"f1={m.get('f1', float('nan')):.4f}  "
                f"pm_l1={m.get('pointmap_l1', float('nan')):.4f}")
    log("=" * 72)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

