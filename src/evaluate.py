"""Evaluate baselines and trained models on all LOTO test folds.

Baselines (no training required):
    B0: B0_t1_date_copy — use t1 point cloud as prediction
    B1: B1_t3_date_copy — use t3 point cloud as prediction
    B2: B2_nearest_date_copy — use whichever endpoint is temporally closer to t2
    B3: B3_linear_point_map_interpolation — element-wise (1-tau)*t1 + tau*t3
    B4: B4_temporal_weighted_point_map_union — sample from t1 and t3 weighted by tau

Optionally evaluates a trained model (from train.py checkpoints) on each fold.

Usage:
    # Baselines only:
    python src/evaluate.py --config configs/train.yaml

    # Include trained model:
    python src/evaluate.py --config configs/train.yaml --runs-root runs/

    # Distributed (multi-GPU):
    torchrun --nproc_per_node=4 src/evaluate.py --config configs/train.yaml --protocol strict --crop corn
"""
from __future__ import annotations

import argparse
import importlib
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

from losses.geometry import compute_metrics
from loto import build_all_folds, compute_tau
from models.base import TemporalGeometryPredictor as TemporalCloudInterpolator

DEFAULT_CONFIG: dict[str, Any] = {
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "geometry_root": "geometry_assets",
    "output_root": "evaluation",
    "protocols": ["target_date", "strict"],
    "crops": ["corn", "soybean"],
    "n_points": 50_000,
    "seed": 42,
    "conf_threshold": 0.02,
    "distance_threshold": 0.05,
    "voxel_size": 0.05,
    "eval_alpha": 0.5,
    "eval_beta": 0.5,
    "device": "auto",
    "runs_root": None,
    "model_module": None,
    "model_class": None,
    "model_kwargs": {},
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

def log(message: str) -> None:
    if not _is_main():
        return
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required.") from exc
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal interpolation baselines and models.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--geometry-root", type=Path, default=None)
    parser.add_argument("--predicted-root", type=Path, default=None,
                        help="Root for predicted t2 clouds. If set, t2 is loaded from here, t1/t3 from geometry-root.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="Directory from train.py; if set, evaluates trained model checkpoints.")
    parser.add_argument("--protocol", choices=["target_date", "strict"], action="append", default=None)
    parser.add_argument("--crop", choices=["corn", "soybean"], action="append", default=None)
    parser.add_argument("--test-date", default=None, help="Only evaluate this test date.")
    parser.add_argument("--max-variants", type=int, default=None,
                        help="Max variants per triplet (for quick testing).")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if args.config.exists():
        yaml_cfg = read_yaml(args.config)
        config.update(yaml_cfg)
        # eval_output_root (if present) overrides output_root so evaluate.py
        # doesn't write into the training runs/ directory.
        if "eval_output_root" in yaml_cfg:
            config["output_root"] = yaml_cfg["eval_output_root"]
    if args.geometry_root is not None:
        config["geometry_root"] = args.geometry_root
    if args.predicted_root is not None:
        config["predicted_root"] = args.predicted_root
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.runs_root is not None:
        config["runs_root"] = args.runs_root
    if args.protocol is not None:
        config["protocols"] = args.protocol
    if args.crop is not None:
        config["crops"] = args.crop
    if args.device is not None:
        config["device"] = args.device
    if args.test_date is not None:
        config["test_date"] = args.test_date
    config["max_variants"] = args.max_variants
    config["triplets_path"] = Path(config["triplets_path"])
    config["geometry_root"] = Path(config["geometry_root"])
    if config.get("predicted_root"):
        config["predicted_root"] = Path(config["predicted_root"])
    config["output_root"] = Path(config["output_root"])
    if config.get("runs_root"):
        config["runs_root"] = Path(config["runs_root"])
    return config


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_cloud(
    geometry_root: Path,
    triplet_id: str,
    variant: str,
    date_label: str,
    n_points: int,
    seed: int,
    conf_threshold: float = 0.02,
) -> np.ndarray | None:
    """Load from geometry_assets/{triplet_id}/{variant}/{date_label}/point_cloud_clean.npz."""
    path = geometry_root / triplet_id / variant / date_label / "point_cloud_clean.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "points_normalized" in data:
        points = data["points_normalized"].astype(np.float32)
    elif "points_aligned" in data:
        points = data["points_aligned"].astype(np.float32)
    elif "points" in data:
        points = data["points"].astype(np.float32)
    else:
        return None
    if "confidence" in data:
        conf = data["confidence"].astype(np.float32)
        points = points[conf >= conf_threshold]
    if len(points) == 0:
        return None
    if n_points > 0 and len(points) > n_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), n_points, replace=False)]
    return points


def list_variants(geometry_root: Path, triplet_id: str) -> list[str]:
    """List available variants for a triplet in geometry_assets."""
    d = geometry_root / triplet_id
    if not d.exists():
        return []
    return sorted(v.name for v in d.iterdir() if v.is_dir())


def apply_baseline(
    baseline: str,
    pts_t1: np.ndarray,
    pts_t3: np.ndarray,
    tau: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    if baseline == "B0_t1_date_copy":
        return pts_t1
    if baseline == "B1_t3_date_copy":
        return pts_t3
    if baseline == "B2_nearest_date_copy":
        return pts_t1 if tau <= 0.5 else pts_t3
    if baseline == "B3_linear_point_map_interpolation":
        # Element-wise linear blend on equal-size random subsamples.
        # Does NOT assume pointwise correspondence — treats this as a geometric blend.
        n = min(len(pts_t1), len(pts_t3), n_points)
        rng = np.random.default_rng(seed)
        i1 = rng.choice(len(pts_t1), n, replace=False)
        i3 = rng.choice(len(pts_t3), n, replace=False)
        return (1.0 - tau) * pts_t1[i1] + tau * pts_t3[i3]
    if baseline == "B4_temporal_weighted_point_map_union":
        # Sample (1-tau) fraction from t1, tau fraction from t3.
        n1 = max(1, int(round(n_points * (1 - tau))))
        n3 = max(1, int(round(n_points * tau)))
        rng = np.random.default_rng(seed)
        idx1 = rng.choice(len(pts_t1), min(n1, len(pts_t1)), replace=False)
        idx3 = rng.choice(len(pts_t3), min(n3, len(pts_t3)), replace=False)
        return np.concatenate([pts_t1[idx1], pts_t3[idx3]], axis=0)
    raise ValueError(f"Unknown baseline: {baseline}")


def _avg_metrics(metric_list: list[dict]) -> dict:
    """Average a list of metric dicts, skipping NaN values."""
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if isinstance(m.get(k), float) and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_triplet(
    triplet: dict[str, Any],
    geometry_root: Path,
    baselines: list[str],
    n_points: int,
    seed: int,
    threshold: float,
    voxel_size: float,
    model: TemporalCloudInterpolator | None,
    device: str,
    conf_threshold: float = 0.02,
    alpha: float = 0.5,
    beta: float = 0.5,
    max_variants: int | None = None,
    predicted_root: Path | None = None,
) -> dict[str, Any] | None:
    crop = triplet["crop"]
    left_date = triplet["left_date"]
    middle_date = triplet["middle_date"]
    right_date = triplet["right_date"]
    triplet_id = f"{left_date}_{middle_date}_{right_date}_{crop}"
    tau = compute_tau(left_date, middle_date, right_date)

    variants = list_variants(geometry_root, triplet_id)
    if not variants:
        return None
    if max_variants is not None:
        variants = variants[:max_variants]

    method_keys = list(baselines)
    if predicted_root is not None:
        method_keys.append("predicted")
    if model is not None:
        method_keys.append("model")
    variant_metrics: dict[str, list[dict]] = {k: [] for k in method_keys}

    for variant in variants:
        pts_t1 = load_cloud(geometry_root, triplet_id, variant, "t1", n_points, seed, conf_threshold)
        pts_t2 = load_cloud(geometry_root, triplet_id, variant, "t2", n_points, seed, conf_threshold)
        pts_t3 = load_cloud(geometry_root, triplet_id, variant, "t3", n_points, seed + 1, conf_threshold)

        if pts_t1 is None or pts_t2 is None or pts_t3 is None:
            continue

        for baseline in baselines:
            pred = apply_baseline(baseline, pts_t1, pts_t3, tau, n_points, seed)
            m = compute_metrics(pred, pts_t2, threshold=threshold, voxel_size=voxel_size,
                                alpha=alpha, beta=beta)
            variant_metrics[baseline].append(m)

        if predicted_root is not None:
            pts_pred = load_cloud(predicted_root, triplet_id, variant, "t2", n_points, seed, conf_threshold)
            if pts_pred is not None:
                m = compute_metrics(pts_pred, pts_t2, threshold=threshold, voxel_size=voxel_size,
                                    alpha=alpha, beta=beta)
                variant_metrics["predicted"].append(m)

        if model is not None:
            model.eval()
            with torch.no_grad():
                t1_tensor = torch.from_numpy(pts_t1).to(device)
                t3_tensor = torch.from_numpy(pts_t3).to(device)
                tau_tensor = torch.tensor(tau, dtype=torch.float32).to(device)
                pred_tensor = model(t1_tensor, t3_tensor, tau_tensor)
                pred_np = pred_tensor.cpu().numpy().astype(np.float32)
            m = compute_metrics(pred_np, pts_t2, threshold=threshold, voxel_size=voxel_size,
                                alpha=alpha, beta=beta)
            variant_metrics["model"].append(m)

    # Average metrics across variants for this triplet
    row: dict[str, Any] = {
        "triplet_id": triplet_id,
        "crop": crop,
        "left_date": left_date,
        "middle_date": middle_date,
        "right_date": right_date,
        "tau": tau,
        "sensor_consistent": triplet.get("sensor_consistent", True),
        "n_variants": len(variants),
    }
    for method in method_keys:
        if variant_metrics[method]:
            row[method] = _avg_metrics(variant_metrics[method])

    return row


def aggregate_metrics(rows: list[dict[str, Any]], method_keys: list[str]) -> dict[str, dict[str, float]]:
    """Average metrics across all rows for each method."""
    agg: dict[str, dict[str, list[float]]] = {k: {} for k in method_keys}
    for row in rows:
        for method in method_keys:
            if method not in row:
                continue
            for metric, val in row[method].items():
                if isinstance(val, float) and not np.isnan(val):
                    agg[method].setdefault(metric, []).append(val)
    return {
        method: {metric: float(np.mean(vals)) for metric, vals in metrics.items()}
        for method, metrics in agg.items()
    }


def load_model_for_fold(
    fold_id: str,
    runs_root: Path,
    protocol: str,
    model_cls: type,
    model_kwargs: dict,
    device: str,
) -> TemporalCloudInterpolator | None:
    checkpoint = runs_root / protocol / fold_id / "best_model.pt"
    if not checkpoint.exists():
        log(f"no checkpoint found for fold={fold_id} at {checkpoint}")
        return None
    model = model_cls(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def evaluate_fold(
    fold: dict[str, Any],
    config: dict[str, Any],
    device: str,
    model_cls: type | None,
    fold_output_dir: Path,
) -> dict[str, Any]:
    fold_id = fold["fold_id"]
    protocol = fold["protocol"]
    baselines = [
        "B0_t1_date_copy",
        "B1_t3_date_copy",
        "B2_nearest_date_copy",
        "B3_linear_point_map_interpolation",
        "B4_temporal_weighted_point_map_union",
    ]
    method_keys = list(baselines)

    if config.get("predicted_root"):
        method_keys.append("predicted")

    model: TemporalCloudInterpolator | None = None
    if model_cls is not None and config.get("runs_root"):
        model = load_model_for_fold(
            fold_id, config["runs_root"], protocol,
            model_cls, config.get("model_kwargs", {}), device
        )
        if model is not None:
            method_keys.append("model")

    # Distribute triplets across ranks
    all_triplets = fold["test_triplets"]
    rank = _get_rank()
    world_size = _get_world_size()
    my_triplets = all_triplets[rank::world_size]

    rows: list[dict[str, Any]] = []
    for triplet in tqdm(my_triplets, desc=f"fold={fold_id} rank={rank}", disable=(rank != 0)):
        row = evaluate_triplet(
            triplet=triplet,
            geometry_root=config["geometry_root"],
            baselines=baselines,
            n_points=config["n_points"],
            seed=config["seed"],
            threshold=config["distance_threshold"],
            voxel_size=config["voxel_size"],
            model=model,
            device=device,
            conf_threshold=config.get("conf_threshold", 0.02),
            alpha=config.get("eval_alpha", 0.5),
            beta=config.get("eval_beta", 0.5),
            max_variants=config.get("max_variants"),
            predicted_root=config.get("predicted_root"),
        )
        if row is not None:
            rows.append(row)

    # Gather rows from all ranks
    if _is_distributed():
        dist.barrier()
        all_rows_gathered = [None] * world_size
        dist.all_gather_object(all_rows_gathered, rows)
        rows = [r for rank_rows in all_rows_gathered for r in rank_rows]

    aggregated = aggregate_metrics(rows, method_keys)
    result = {
        "fold_id": fold_id,
        "crop": fold["crop"],
        "protocol": protocol,
        "test_date": fold["test_date"],
        "n_test": len(rows),
        "aggregated": aggregated,
        "triplet_rows": rows,
    }

    if _is_main():
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_output_dir / "eval_result.json", result)

        for method, metrics in aggregated.items():
            log(f"fold={fold_id} {method}: asymmetric_chamfer={metrics.get('asymmetric_chamfer', float('nan')):.4f} f1={metrics.get('f1', 0.0):.4f}")

    return result


def main() -> None:
    # Setup distributed if launched via torchrun
    distributed = _is_distributed()
    if distributed:
        local_rank, world_size = _setup_distributed()
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        world_size = 1
        device = None

    args = parse_args()
    config = build_config(args)

    if not distributed:
        device = choose_device(config["device"])

    log(f"evaluate: world_size={world_size} device={device}")

    model_cls: type | None = None
    if config.get("model_module") and config.get("model_class"):
        module = importlib.import_module(config["model_module"])
        model_cls = getattr(module, config["model_class"])

    all_folds = build_all_folds(config["triplets_path"])

    output_root = config["output_root"]
    if _is_main():
        output_root.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    all_results: list[dict[str, Any]] = []
    for protocol in config["protocols"]:
        folds = all_folds.get(protocol, [])
        for fold in folds:
            if fold["crop"] not in config["crops"]:
                continue
            if not fold["test_triplets"]:
                continue
            if config.get("test_date") and fold["test_date"] != config["test_date"]:
                continue
            log(f"--- evaluating fold={fold['fold_id']} n_test={fold['n_test']} ---")
            fold_output_dir = output_root / protocol / fold["fold_id"]
            result = evaluate_fold(fold, config, device, model_cls, fold_output_dir)
            all_results.append({k: v for k, v in result.items() if k != "triplet_rows"})

    if _is_main():
        write_json(output_root / "eval_summary.json", all_results)
        log(f"evaluation complete. summary: {output_root / 'eval_summary.json'}")

    if distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

