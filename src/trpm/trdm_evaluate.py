"""Evaluate TRDM by unprojecting predicted depth maps and comparing baselines.

This is intentionally separate from the existing TRPM evaluators so the point-map
TRPM path remains unchanged. TRDM predicts target-view depth; this evaluator
converts each predicted ``D2_hat`` into a t2 world-space point cloud using K2 and
T2_c2w, aligns it to dataset/GPS coordinates, and evaluates against t2 depth
geometry.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.geometry import compute_metrics
from loto import build_all_folds
from trpm.trdm_dataset import TRDMDepthDataset, _load_depth_and_conf, trdm_collate
from trpm.trdm_geometry import (
    apply_similarity,
    gps_alignment,
    load_intrinsics,
    unproject_depth_numpy,
    vggt_extrinsic_to_c2w_np,
)


BASELINES = [
    "B0_t1_date_copy",
    "B1_t3_date_copy",
    "B2_nearest_date_copy",
    "B3_linear_depth_cloud_interpolation",
    "B4_temporal_weighted_depth_cloud_union",
]


DEFAULT_CONFIG: dict[str, Any] = {
    "model_class": "trpm.trdm_model.TRDM",
    "vggt_output_root": "vggt_outputs/t1t2_paired_v16_o8",
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "runs_root": "runs/trdm",
    "output_root": "evaluation/trdm",
    "protocols": ["strict"],
    "crops": ["corn"],
    "test_date": None,
    "seed": 42,
    "device": "auto",
    "image_preprocess_mode": "pad",
    "conf_threshold": 0.02,
    "pred_conf_threshold": 0.0,
    "n_points": 50_000,
    "distance_threshold": 0.05,
    "voxel_size": 0.05,
    "eval_alpha": 0.5,
    "eval_beta": 0.5,
    "save_clouds": False,
    "baselines_only": False,
    "eval_batch_size": 4,
    "compute_normals": False,
    "shard_wait_poll_s": 10,
    "shard_wait_timeout_s": 3600,
    "model_kwargs": {},
}


def distributed_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def log(message: str) -> None:
    rank, _, world_size = distributed_info()
    prefix = f"[rank {rank}/{world_size}] " if world_size > 1 else ""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix}{message}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TRDM depth predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_trdm.yaml"))
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", choices=["target_date", "strict"], action="append", default=None)
    parser.add_argument("--crop", action="append", default=None)
    parser.add_argument("--test-date", default=None)
    parser.add_argument("--save-clouds", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config and args.config.exists():
        cfg.update(read_yaml(args.config))
        if "eval_output_root" in cfg:
            cfg["output_root"] = cfg["eval_output_root"]
    if args.runs_root is not None:
        cfg["runs_root"] = args.runs_root
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.device is not None:
        cfg["device"] = args.device
    if args.protocol is not None:
        cfg["protocols"] = args.protocol
    if args.crop is not None:
        cfg["crops"] = args.crop
    if args.test_date is not None:
        cfg["test_date"] = args.test_date
    cfg["save_clouds"] = bool(args.save_clouds or cfg.get("save_clouds", False))
    cfg["baselines_only"] = bool(args.baselines_only or cfg.get("baselines_only", False))
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["triplets_path"] = Path(cfg["triplets_path"])
    cfg["runs_root"] = Path(cfg["runs_root"])
    cfg["output_root"] = Path(cfg["output_root"])
    return cfg


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def triplet_id_from_fold_entry(entry: dict[str, Any]) -> str:
    return f"{entry['left_date']}_{entry['middle_date']}_{entry['right_date']}_{entry['crop']}"


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _list_variants(vggt_root: Path, triplet_id: str) -> list[str]:
    triplet_dir = vggt_root / triplet_id
    if not triplet_dir.exists():
        return []
    return sorted(path.name for path in triplet_dir.iterdir() if path.is_dir())


def _voxel_downsample(points: np.ndarray, confidence: np.ndarray, voxel_size: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0 or voxel_size <= 0:
        return points, confidence
    keys = (points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[unique_idx], confidence[unique_idx]


def _postprocess_cloud(
    points: np.ndarray,
    confidence: np.ndarray,
    n_points: int,
    seed: int,
    voxel_size: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points.astype(np.float32), confidence.astype(np.float32)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points, confidence = points[finite], confidence[finite]
    if len(points) == 0:
        return points.astype(np.float32), confidence.astype(np.float32)
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    keep = distances <= np.quantile(distances, 0.995)
    points, confidence = points[keep], confidence[keep]
    points, confidence = _voxel_downsample(points, confidence, voxel_size=voxel_size)
    if n_points > 0 and len(points) > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), n_points, replace=False)
        points, confidence = points[idx], confidence[idx]
    return points.astype(np.float32), confidence.astype(np.float32)


def _load_c2w(date_dir: Path) -> np.ndarray:
    extrinsics = np.load(date_dir / "predictions" / "extrinsic.npy").astype(np.float32)
    return np.stack([vggt_extrinsic_to_c2w_np(ext) for ext in extrinsics]).astype(np.float32)


@lru_cache(maxsize=96)
def _load_date_bundle_cached(
    date_dir_str: str,
    preprocess_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    date_dir = Path(date_dir_str)
    depth_all, conf_all = _load_depth_and_conf(date_dir)
    height, width = depth_all.shape[1], depth_all.shape[2]
    K_all = load_intrinsics(date_dir, height, width, preprocess_mode)
    T_all = _load_c2w(date_dir)
    alignment = gps_alignment(date_dir)
    return depth_all, conf_all, K_all, T_all, alignment


def _depth_view_cloud_from_bundle(
    depth_all: np.ndarray,
    conf_all: np.ndarray,
    K_all: np.ndarray,
    T_all: np.ndarray,
    alignment: Any,
    view_idx: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = depth_all[view_idx]
    confidence = conf_all[view_idx]
    points = unproject_depth_numpy(depth, K_all[view_idx], T_all[view_idx])
    points = apply_similarity(points, alignment)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(depth.reshape(-1))
        & (depth.reshape(-1) > 0)
        & np.isfinite(confidence.reshape(-1))
        & (confidence.reshape(-1) >= conf_threshold)
    )
    return points[valid].astype(np.float32), confidence.reshape(-1)[valid].astype(np.float32)


def _depth_view_cloud(
    date_dir: Path,
    view_idx: int,
    preprocess_mode: str,
    conf_threshold: float,
    align_to_gps: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    depth_all, conf_all, K_all, T_all, alignment = _load_date_bundle_cached(
        str(date_dir), preprocess_mode
    )
    if not align_to_gps:
        alignment = None
    return _depth_view_cloud_from_bundle(
        depth_all, conf_all, K_all, T_all, alignment, view_idx, conf_threshold
    )


def _load_date_cloud(
    date_dir: Path,
    preprocess_mode: str,
    conf_threshold: float,
    n_points: int,
    seed: int,
    voxel_size: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    depth_all, conf_all, K_all, T_all, alignment = _load_date_bundle_cached(
        str(date_dir), preprocess_mode
    )
    views: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    all_confidence: list[np.ndarray] = []
    for view_idx in range(depth_all.shape[0]):
        points, confidence = _depth_view_cloud_from_bundle(
            depth_all, conf_all, K_all, T_all, alignment, view_idx, conf_threshold
        )
        if len(points):
            views.append(np.concatenate([points, confidence[:, None]], axis=1))
            all_points.append(points)
            all_confidence.append(confidence)
    if not all_points:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), views
    points = np.concatenate(all_points, axis=0)
    confidence = np.concatenate(all_confidence, axis=0)
    points, confidence = _postprocess_cloud(points, confidence, n_points, seed, voxel_size=voxel_size)
    return points, confidence, views


def apply_baseline(
    baseline: str,
    points_t1: np.ndarray,
    points_t2: np.ndarray,
    points_t3: np.ndarray,
    tau: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    if baseline == "B0_t1_date_copy":
        return points_t1
    if baseline == "B1_t3_date_copy":
        return points_t3
    if baseline == "B2_nearest_date_copy":
        return points_t1 if tau <= 0.5 else points_t3
    if baseline == "B3_linear_depth_cloud_interpolation":
        count = min(len(points_t1), len(points_t3), n_points)
        if count == 0:
            return np.zeros((0, 3), dtype=np.float32)
        rng = np.random.default_rng(seed)
        idx1 = rng.choice(len(points_t1), count, replace=False)
        idx3 = rng.choice(len(points_t3), count, replace=False)
        return ((1.0 - tau) * points_t1[idx1] + tau * points_t3[idx3]).astype(np.float32)
    if baseline == "B4_temporal_weighted_depth_cloud_union":
        if len(points_t1) == 0 and len(points_t3) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        rng = np.random.default_rng(seed)
        count_t1 = min(max(1, int(round(n_points * (1.0 - tau)))), len(points_t1))
        count_t3 = min(max(1, int(round(n_points * tau))), len(points_t3))
        clouds = []
        if count_t1:
            clouds.append(points_t1[rng.choice(len(points_t1), count_t1, replace=False)])
        if count_t3:
            clouds.append(points_t3[rng.choice(len(points_t3), count_t3, replace=False)])
        return np.concatenate(clouds, axis=0).astype(np.float32)
    raise ValueError(f"Unknown baseline: {baseline}")


def _metrics(
    pred_points: np.ndarray,
    ref_points: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    return compute_metrics(
        pred_points[:, :3],
        ref_points[:, :3],
        threshold=cfg["distance_threshold"],
        voxel_size=cfg["voxel_size"],
        alpha=cfg["eval_alpha"],
        beta=cfg["eval_beta"],
        compute_normals=bool(cfg.get("compute_normals", False)),
    )


def _avg_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [
            metrics[key]
            for metrics in metric_list
            if isinstance(metrics.get(key), float) and not np.isnan(metrics[key])
        ]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def aggregate_rows(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    per_view: dict[str, Any] = {}
    for method in methods:
        merged_values = [row["merged"][method] for row in rows if method in row.get("merged", {})]
        per_view_values = [row["per_view_avg"][method] for row in rows if method in row.get("per_view_avg", {})]
        if merged_values:
            merged[method] = _avg_metrics(merged_values)
        if per_view_values:
            per_view[method] = _avg_metrics(per_view_values)
    return {"merged": merged, "per_view_avg": per_view}


def _dataset_lookup(dataset: TRDMDepthDataset) -> dict[tuple[str, str], list[int]]:
    lookup: dict[tuple[str, str], list[int]] = {}
    for idx, entry in enumerate(dataset.index):
        lookup.setdefault((entry["triplet_id"], entry["variant"]), []).append(idx)
    return lookup


@torch.no_grad()
def predict_variant_depth_clouds(
    model: torch.nn.Module,
    dataset: TRDMDepthDataset,
    sample_indices: list[int],
    device: str,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    model.eval()
    all_points: list[np.ndarray] = []
    all_confidence: list[np.ndarray] = []
    per_view_clouds: list[np.ndarray] = []
    preprocess_mode = cfg.get("image_preprocess_mode", "pad")
    n_points = cfg["n_points"]
    seed = cfg.get("seed", 42)
    voxel_size = cfg.get("voxel_size", 0.05)
    pred_conf_threshold = cfg.get("pred_conf_threshold", 0.0)
    batch_size = max(1, int(cfg.get("eval_batch_size", 1)))

    for start in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[start : start + batch_size]
        samples = [dataset[sample_idx] for sample_idx in batch_indices]
        batch = move_batch(trdm_collate(samples), device)
        outputs = model(batch)
        depths = outputs["D2_hat"][:, 0].float().cpu().numpy()
        gates = outputs.get("G", outputs["gate"])[:, 0].float().cpu().numpy()

        for batch_offset, sample_idx in enumerate(batch_indices):
            depth = depths[batch_offset]
            gate = gates[batch_offset]
            entry = dataset.index[sample_idx]
            t2_dir = entry["t2_dir"]
            view_idx = entry["view_idx"]
            _, _, K_all, T_all, alignment = _load_date_bundle_cached(str(t2_dir), preprocess_mode)
            points = unproject_depth_numpy(depth, K_all[view_idx], T_all[view_idx])
            points = apply_similarity(points, alignment)
            confidence = np.ones((points.shape[0],), dtype=np.float32)
            valid = np.isfinite(points).all(axis=1) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
            if pred_conf_threshold > 0:
                valid &= np.isfinite(gate.reshape(-1)) & (gate.reshape(-1) >= pred_conf_threshold)
                confidence = gate.reshape(-1).astype(np.float32)
            points = points[valid].astype(np.float32)
            confidence = confidence[valid].astype(np.float32)
            if len(points):
                per_view_clouds.append(np.concatenate([points, confidence[:, None]], axis=1))
                all_points.append(points)
                all_confidence.append(confidence)

    if not all_points:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), per_view_clouds
    points = np.concatenate(all_points, axis=0)
    confidence = np.concatenate(all_confidence, axis=0)
    points, confidence = _postprocess_cloud(points, confidence, n_points, seed, voxel_size=voxel_size)
    return points, confidence, per_view_clouds


def evaluate_variant(
    model: torch.nn.Module | None,
    dataset: TRDMDepthDataset | None,
    sample_indices: list[int],
    triplet_id: str,
    variant: str,
    tau: float,
    cfg: dict[str, Any],
    device: str,
    clouds_dir: Path | None,
) -> dict[str, Any] | None:
    vggt_root = cfg["vggt_output_root"]
    preprocess_mode = cfg.get("image_preprocess_mode", "pad")
    conf_threshold = cfg["conf_threshold"]
    n_points = cfg["n_points"]
    seed = cfg.get("seed", 42)
    voxel_size = cfg.get("voxel_size", 0.05)

    variant_dir = vggt_root / triplet_id / variant
    t1_dir = variant_dir / "t1"
    t2_dir = variant_dir / "t2"
    t3_dir = variant_dir / "t3"
    if not all(date_dir.exists() for date_dir in (t1_dir, t2_dir, t3_dir)):
        return None

    points_t1, _, views_t1 = _load_date_cloud(t1_dir, preprocess_mode, conf_threshold, n_points, seed, voxel_size)
    points_t2, _, views_t2 = _load_date_cloud(t2_dir, preprocess_mode, conf_threshold, n_points, seed, voxel_size)
    points_t3, _, views_t3 = _load_date_cloud(t3_dir, preprocess_mode, conf_threshold, n_points, seed, voxel_size)
    if len(points_t2) == 0:
        return None

    merged: dict[str, dict[str, float]] = {}
    per_view: dict[str, list[dict[str, float]]] = {method: [] for method in BASELINES}

    baseline_clouds: dict[str, np.ndarray] = {}
    for baseline in BASELINES:
        pred = apply_baseline(baseline, points_t1, points_t2, points_t3, tau, n_points, seed)
        baseline_clouds[baseline] = pred
        merged[baseline] = _metrics(pred, points_t2, cfg)

    view_count = min(len(views_t1), len(views_t2))
    for view_idx in range(view_count):
        ref_view = views_t2[view_idx][:, :3]
        t1_view = views_t1[view_idx][:, :3]
        t3_view = views_t3[min(view_idx, len(views_t3) - 1)][:, :3] if views_t3 else points_t3
        if len(ref_view) == 0:
            continue
        for baseline in BASELINES:
            pred_view = apply_baseline(baseline, t1_view, ref_view, t3_view, tau, n_points, seed + view_idx)
            per_view[baseline].append(_metrics(pred_view, ref_view, cfg))

    if model is not None and dataset is not None and sample_indices:
        pred_points, pred_conf, pred_views = predict_variant_depth_clouds(
            model, dataset, sample_indices, device, cfg
        )
        if len(pred_points):
            merged["trdm"] = _metrics(pred_points, points_t2, cfg)
            per_view["trdm"] = []
            for view_idx in range(min(len(pred_views), len(views_t2))):
                ref_view = views_t2[view_idx][:, :3]
                pred_view = pred_views[view_idx][:, :3]
                if len(ref_view) and len(pred_view):
                    per_view["trdm"].append(_metrics(pred_view, ref_view, cfg))
            if clouds_dir is not None:
                save_path = clouds_dir / triplet_id / variant
                save_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path / "trdm_pred_merged.npz", points=pred_points, confidence=pred_conf)
                np.savez_compressed(save_path / "t2_ref_merged.npz", points=points_t2)
                for view_idx, view_cloud in enumerate(pred_views):
                    np.savez_compressed(save_path / f"trdm_pred_view_{view_idx:02d}.npz", points=view_cloud[:, :3], confidence=view_cloud[:, 3])
                for view_idx, view_cloud in enumerate(views_t2):
                    np.savez_compressed(save_path / f"t2_ref_view_{view_idx:02d}.npz", points=view_cloud[:, :3], confidence=view_cloud[:, 3])
                for baseline, cloud in baseline_clouds.items():
                    np.savez_compressed(save_path / f"{baseline}.npz", points=cloud)

    per_view_avg = {
        method: _avg_metrics(metrics)
        for method, metrics in per_view.items()
        if metrics
    }
    return {
        "triplet_id": triplet_id,
        "variant": variant,
        "tau": tau,
        "n_t1_points": int(len(points_t1)),
        "n_t2_points": int(len(points_t2)),
        "n_t3_points": int(len(points_t3)),
        "merged": merged,
        "per_view_avg": per_view_avg,
    }


def evaluate_fold(
    fold: dict[str, Any],
    model: torch.nn.Module | None,
    dataset: TRDMDepthDataset | None,
    cfg: dict[str, Any],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clouds_dir = output_dir / "clouds" if cfg.get("save_clouds") else None
    if clouds_dir is not None:
        clouds_dir.mkdir(parents=True, exist_ok=True)

    lookup = _dataset_lookup(dataset) if dataset is not None else {}
    variant_rows: list[dict[str, Any]] = []
    rank, local_rank, world_size = distributed_info()
    test_triplets = [
        triplet for idx, triplet in enumerate(fold["test_triplets"])
        if idx % world_size == rank
    ]
    log(
        f"fold={fold['fold_id']} assigned {len(test_triplets)}/"
        f"{len(fold['test_triplets'])} triplets on local_rank={local_rank}"
    )

    for triplet in tqdm(test_triplets, desc=f"fold={fold['fold_id']} rank={rank}", position=local_rank):
        triplet_id = triplet_id_from_fold_entry(triplet)
        tau = float(triplet["tau"])
        for variant in _list_variants(cfg["vggt_output_root"], triplet_id):
            row = evaluate_variant(
                model,
                dataset,
                lookup.get((triplet_id, variant), []),
                triplet_id,
                variant,
                tau,
                cfg,
                device,
                clouds_dir,
            )
            if row is not None:
                variant_rows.append(row)

    methods = list(BASELINES) + (["trdm"] if model is not None else [])
    aggregated = aggregate_rows(variant_rows, methods)
    result = {
        "fold_id": fold["fold_id"],
        "crop": fold["crop"],
        "protocol": fold["protocol"],
        "test_date": fold["test_date"],
        "rank": rank,
        "world_size": world_size,
        "n_variants": len(variant_rows),
        "aggregated": aggregated,
        "variant_rows": variant_rows,
    }
    write_json(output_dir / f"eval_result_rank{rank:02d}.json", result)
    if world_size == 1:
        write_json(output_dir / "eval_result.json", result)
    return result


def merge_fold_shards(fold: dict[str, Any], output_dir: Path, world_size: int, methods: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = output_dir / f"eval_result_rank{rank:02d}.json"
        if path.exists():
            rows.extend(json.loads(path.read_text()).get("variant_rows", []))
    result = {
        "fold_id": fold["fold_id"],
        "crop": fold["crop"],
        "protocol": fold["protocol"],
        "test_date": fold["test_date"],
        "n_variants": len(rows),
        "aggregated": aggregate_rows(rows, methods),
        "variant_rows": rows,
    }
    write_json(output_dir / "eval_result.json", result)
    return result


def wait_for_fold_shards(output_dir: Path, world_size: int, cfg: dict[str, Any]) -> None:
    expected = [output_dir / f"eval_result_rank{rank_idx:02d}.json" for rank_idx in range(world_size)]
    poll_s = max(1.0, float(cfg.get("shard_wait_poll_s", 10)))
    timeout_s = float(cfg.get("shard_wait_timeout_s", 3600))
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    last_log = 0.0
    while True:
        missing = [path.name for path in expected if not path.exists()]
        if not missing:
            return
        now = time.monotonic()
        if now - last_log >= poll_s:
            log(f"waiting for fold shards in {output_dir}; missing={missing}")
            last_log = now
        if deadline is not None and now >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {len(missing)} fold shard(s) in {output_dir}: {missing}"
            )
        time.sleep(poll_s)


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    metrics = ["asymmetric_chamfer", "f1", "precision", "recall", "voxel_iou", "height_median_error"]
    lines = ["# TRDM Evaluation Report", ""]
    for result in results:
        lines.append(f"## {result['protocol']} / {result['fold_id']} / test={result['test_date']}")
        for section in ("merged", "per_view_avg"):
            lines.append("")
            lines.append(f"### {section}")
            lines.append("| method | " + " | ".join(metrics) + " |")
            lines.append("|---|" + "|".join(["---"] * len(metrics)) + "|")
            for method, values in result.get("aggregated", {}).get(section, {}).items():
                row = " | ".join(f"{values.get(metric, float('nan')):.4f}" for metric in metrics)
                lines.append(f"| {method} | {row} |")
        lines.append("")
    path.write_text("\n".join(lines))


def load_checkpoint_model(checkpoint: Path, cfg: dict[str, Any], device: str) -> torch.nn.Module | None:
    if cfg.get("baselines_only") or not checkpoint.exists():
        return None
    model_cls = load_model_class(cfg["model_class"])
    model = model_cls(**cfg.get("model_kwargs", {})).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    rank, local_rank, world_size = distributed_info()
    if cfg["device"] == "auto" and torch.cuda.is_available() and world_size > 1:
        device = f"cuda:{local_rank}"
    else:
        device = choose_device(cfg["device"])
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)

    cfg["output_root"].mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(cfg["output_root"] / "eval_config.json", cfg)

    all_folds = build_all_folds(cfg["triplets_path"])
    fold_jobs: list[dict[str, Any]] = []
    for protocol in cfg["protocols"]:
        for fold in all_folds.get(protocol, []):
            if fold["crop"] not in cfg["crops"]:
                continue
            if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                continue
            if fold["test_triplets"]:
                fold_jobs.append(fold)

    all_needed_triplet_ids = {
        triplet_id_from_fold_entry(triplet)
        for fold in fold_jobs
        for triplet in fold["test_triplets"]
    }
    rank_needed_triplet_ids = {
        triplet_id_from_fold_entry(triplet)
        for fold in fold_jobs
        for idx, triplet in enumerate(fold["test_triplets"])
        if idx % world_size == rank
    }
    log(
        f"startup: folds={len(fold_jobs)} test_triplets={len(all_needed_triplet_ids)} "
        f"rank_triplets={len(rank_needed_triplet_ids)} device={device} root={cfg['vggt_output_root']}"
    )

    dataset: TRDMDepthDataset | None = None
    if not cfg.get("baselines_only"):
        log("indexing TRDM samples for selected folds")
        dataset = TRDMDepthDataset(
            cfg["vggt_output_root"],
            preprocess_mode=cfg.get("image_preprocess_mode", "pad"),
            triplet_ids=rank_needed_triplet_ids,
        )
        log(f"dataset samples={len(dataset)}")

    local_results = []
    methods = list(BASELINES) + ([] if cfg.get("baselines_only") else ["trdm"])
    for fold in fold_jobs:
        checkpoint = cfg["runs_root"] / fold["protocol"] / fold["fold_id"] / "best_model.pt"
        model = load_checkpoint_model(checkpoint, cfg, device)
        if model is None:
            log(f"fold={fold['fold_id']} evaluating baselines only")
        else:
            log(f"fold={fold['fold_id']} loaded {checkpoint}")
        output_dir = cfg["output_root"] / fold["protocol"] / fold["fold_id"]
        local_results.append(evaluate_fold(fold, model, dataset, cfg, device, output_dir))

    if world_size == 1:
        summary = [{key: value for key, value in result.items() if key != "variant_rows"} for result in local_results]
        write_json(cfg["output_root"] / "eval_summary.json", summary)
        write_report(cfg["output_root"] / "eval_report.md", summary)
        log(f"done → {cfg['output_root'] / 'eval_report.md'}")
        return

    if rank == 0:
        for fold in fold_jobs:
            output_dir = cfg["output_root"] / fold["protocol"] / fold["fold_id"]
            wait_for_fold_shards(output_dir, world_size, cfg)
            merge_fold_shards(fold, output_dir, world_size, methods)
        summary = []
        for fold in fold_jobs:
            result = json.loads((cfg["output_root"] / fold["protocol"] / fold["fold_id"] / "eval_result.json").read_text())
            summary.append({key: value for key, value in result.items() if key != "variant_rows"})
        write_json(cfg["output_root"] / "eval_summary.json", summary)
        write_report(cfg["output_root"] / "eval_report.md", summary)
        log(f"done → {cfg['output_root'] / 'eval_report.md'}")


if __name__ == "__main__":
    main()
