"""Evaluate TRPM-Small and baselines on LOTO test folds."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.geometry import compute_metrics
from loto import build_all_folds
from trpm.dataset import _load_pointmaps, _gps_alignment, _to_gps


def _load_model_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module_path), class_name)


# ── config ────────────────────────────────────────────────────────────────────

BASELINES = [
    "B0_t1_date_copy",
    "B1_t3_date_copy",
    "B2_nearest_date_copy",
    "B3_linear_point_map_interpolation",
    "B4_temporal_weighted_point_map_union",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "vggt_output_root": "vggt_outputs/t1t2_paired_v16_o8",
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "runs_root": "runs/trpm_small",
    "output_root": "evaluation/trpm_small",
    "protocols": ["target_date", "strict"],
    "crops": ["corn"],
    "conf_threshold": 0.02,
    "pred_conf_threshold": 0.02,
    "n_points": 50_000,
    "seed": 42,
    "distance_threshold": 0.05,
    "voxel_size": 0.05,
    "eval_alpha": 0.5,
    "eval_beta": 0.5,
    "device": "auto",
    "model_kwargs": {
        "num_t3_points": 1024,
        "cond_dim": 192,
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--protocol", choices=["target_date", "strict"], action="append", default=None)
    p.add_argument("--crop", action="append", default=None)
    p.add_argument("--test-date", default=None)
    p.add_argument("--save-clouds", action="store_true",
                   help="Save pred/ref clouds in GPS space for visualize.py")
    p.add_argument("--baselines-only", action="store_true",
                   help="Skip model inference, evaluate baselines only")
    p.add_argument("--baseline-cache", type=Path, default=None,
                   help="Directory to cache/reuse per-triplet baseline metrics")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config and args.config.exists():
        cfg.update(read_yaml(args.config))
        if "eval_output_root" in cfg:
            cfg["output_root"] = cfg["eval_output_root"]
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
    cfg["save_clouds"] = args.save_clouds
    cfg["baselines_only"] = args.baselines_only
    cfg["baseline_cache"] = args.baseline_cache
    pred_thr = cfg.get("pred_conf_threshold", cfg.get("conf_threshold", 0.02))
    cfg["pred_conf_threshold"] = pred_thr
    cfg.setdefault("model_kwargs", {})["conf_threshold"] = pred_thr
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["runs_root"]        = Path(cfg["runs_root"])
    cfg["output_root"]      = Path(cfg["output_root"])
    cfg["triplets_path"]    = Path(cfg.get("triplets_path", "prepared_data/subsets/benchmark_triplets.json"))
    return cfg


# ── point map / cloud helpers ─────────────────────────────────────────────────

def _pred_dir(vggt_root: Path, triplet_id: str, variant: str, t: str) -> Path:
    return vggt_root / triplet_id / variant / t / "predictions"


def _list_variants(vggt_root: Path, triplet_id: str) -> list[str]:
    d = vggt_root / triplet_id
    if not d.exists():
        return []
    return sorted(v.name for v in d.iterdir() if v.is_dir())


def _pointmap_to_cloud(pm: np.ndarray, pc: np.ndarray, conf_threshold: float, n_points: int, seed: int) -> np.ndarray:
    """Flatten [H, W, 3] point map to (N, 4) cloud (xyz + conf) filtered by confidence."""
    conf = pc.reshape(-1)
    mask = conf >= conf_threshold
    xyz  = pm.reshape(-1, 3)[mask]
    c    = conf[mask, None]
    pts  = np.concatenate([xyz, c], axis=1).astype(np.float32)  # [N, 4]
    if n_points > 0 and len(pts) > n_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return pts


def _voxel_downsample(pts: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """Fast voxel downsampling using dict hashing. Uses only xyz (first 3 cols)."""
    if len(pts) == 0:
        return pts
    keys = (pts[:, :3] / voxel_size).astype(np.int32)
    seen: dict[tuple, int] = {}
    for i, k in enumerate(map(tuple, keys)):
        if k not in seen:
            seen[k] = i
    return pts[np.array(list(seen.values()), dtype=np.int64)]


def _postprocess_cloud(combined: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    """Outlier removal + voxel downsampling + subsampling. Works on [N,3] or [N,4]."""
    if len(combined) == 0:
        return combined
    xyz = combined[:, :3]
    centroid = xyz.mean(axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    combined = combined[dists <= np.quantile(dists, 0.995)]
    combined = _voxel_downsample(combined)
    if n_points > 0 and len(combined) > n_points:
        rng = np.random.default_rng(seed)
        combined = combined[rng.choice(len(combined), n_points, replace=False)]
    return combined


def _load_views(
    date_dir: Path,
    conf_threshold: float,
    seed: int,
) -> list[np.ndarray]:
    """Load each view of a date as a separate [N,4] array (xyz+conf)."""
    pred_dir  = date_dir / "predictions"
    pm_all    = np.load(pred_dir / "point_map.npy",        mmap_mode="r")
    pc_all    = np.load(pred_dir / "point_confidence.npy", mmap_mode="r")
    alignment = _gps_alignment(date_dir)
    views = []
    for vi in range(pm_all.shape[0]):
        pm = pm_all[vi].astype(np.float32)
        pc = pc_all[vi].astype(np.float32)
        if alignment is not None:
            pm = _to_gps(pm, *alignment)
        views.append(_pointmap_to_cloud(pm, pc, conf_threshold, 0, seed))
    return views


def _load_date_cloud(
    date_dir: Path,
    conf_threshold: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    """Load all views of a date, convert to GPS space, concatenate into one cloud."""
    pred_dir  = date_dir / "predictions"
    pm_all    = np.load(pred_dir / "point_map.npy",        mmap_mode="r")
    pc_all    = np.load(pred_dir / "point_confidence.npy", mmap_mode="r")
    alignment = _gps_alignment(date_dir)
    parts = []
    for vi in range(pm_all.shape[0]):
        pm = pm_all[vi].astype(np.float32)
        pc = pc_all[vi].astype(np.float32)
        if alignment is not None:
            pm = _to_gps(pm, *alignment)
        pts = _pointmap_to_cloud(pm, pc, conf_threshold, 0, seed)
        if len(pts):
            parts.append(pts)
    if not parts:
        return np.zeros((0, 3), np.float32)
    return _postprocess_cloud(np.concatenate(parts, axis=0), n_points, seed)


def _load_variant_clouds(
    vggt_root: Path,
    triplet_id: str,
    variant: str,
    conf_threshold: float,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load t1, t2, t3 clouds for a single variant. Returns None if any date is missing."""
    result = []
    for t in ("t1", "t2", "t3"):
        date_dir = vggt_root / triplet_id / variant / t
        if not (date_dir / "predictions" / "point_map.npy").exists():
            return None
        pts = _load_date_cloud(date_dir, conf_threshold, n_points, seed)
        if len(pts) == 0:
            return None
        result.append(pts)
    return tuple(result)  # type: ignore[return-value]


# ── baselines ─────────────────────────────────────────────────────────────────

def _z_align(pts: np.ndarray, z_ref: float, percentile: float = 1.0) -> np.ndarray:
    """Shift pts[:, 2] so its ground level (low percentile) matches z_ref."""
    pts = pts.copy()
    pts[:, 2] += z_ref - np.percentile(pts[:, 2], percentile)
    return pts


def apply_baseline(
    baseline: str,
    pts_t1: np.ndarray,
    pts_t2: np.ndarray,
    pts_t3: np.ndarray,
    tau: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    z_ref = np.percentile(pts_t2[:, 2], 1.0)
    pts_t1 = _z_align(pts_t1, z_ref)
    pts_t3 = _z_align(pts_t3, z_ref)
    if baseline == "B0_t1_date_copy":
        return pts_t1
    if baseline == "B1_t3_date_copy":
        return pts_t3
    if baseline == "B2_nearest_date_copy":
        return pts_t1 if tau <= 0.5 else pts_t3
    if baseline == "B3_linear_point_map_interpolation":
        n = min(len(pts_t1), len(pts_t3), n_points)
        rng = np.random.default_rng(seed)
        i1 = rng.choice(len(pts_t1), n, replace=False)
        i3 = rng.choice(len(pts_t3), n, replace=False)
        return (1.0 - tau) * pts_t1[i1] + tau * pts_t3[i3]
    if baseline == "B4_temporal_weighted_point_map_union":
        n1 = max(1, int(round(n_points * (1 - tau))))
        n3 = max(1, int(round(n_points * tau)))
        rng = np.random.default_rng(seed)
        idx1 = rng.choice(len(pts_t1), min(n1, len(pts_t1)), replace=False)
        idx3 = rng.choice(len(pts_t3), min(n3, len(pts_t3)), replace=False)
        return np.concatenate([pts_t1[idx1], pts_t3[idx3]], axis=0)
    raise ValueError(f"Unknown baseline: {baseline}")


# ── TRPM inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_trpm_variant(
    model: torch.nn.Module,
    vggt_root: Path,
    triplet_id: str,
    variant: str,
    tau: float,
    device: str,
    conf_threshold: float,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, list] | None:
    """Run model over all views of a single variant, return (merged_cloud, per_view_list)."""
    from trpm.trpm_small_cam import TRPMSmallCam
    from trpm.dataset import _load_camera_data
    is_cam = isinstance(model, TRPMSmallCam)

    t1_dir = vggt_root / triplet_id / variant / "t1"
    t2_dir = vggt_root / triplet_id / variant / "t2"
    t3_dir = vggt_root / triplet_id / variant / "t3"
    if not all((d / "predictions" / "point_map.npy").exists() for d in (t1_dir, t2_dir, t3_dir)):
        return None

    nv1 = np.load(t1_dir / "predictions" / "point_map.npy", mmap_mode="r").shape[0]
    nv3 = np.load(t3_dir / "predictions" / "point_map.npy", mmap_mode="r").shape[0]
    align1 = _gps_alignment(t1_dir)
    align3 = _gps_alignment(t3_dir)

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

    P3_t = torch.stack(P3).unsqueeze(0).to(device)   # [1, V3, 3, H, W]
    C3_t = torch.stack(C3).unsqueeze(0).to(device)   # [1, V3, 1, H, W]

    cam1_data = _load_camera_data(t1_dir) if is_cam else None
    cam2_data = _load_camera_data(t2_dir) if is_cam else None
    cam3_data = _load_camera_data(t3_dir) if is_cam else None
    if is_cam and (cam1_data is None or cam2_data is None or cam3_data is None):
        return None

    if is_cam:
        T1_all = torch.from_numpy(cam1_data[0]).to(device)              # [V1, 4, 4]
        T2_all = torch.from_numpy(cam2_data[0]).to(device)              # [V1, 4, 4]
        T3_t   = torch.from_numpy(cam3_data[0]).unsqueeze(0).to(device) # [1, V3, 4, 4]
        K2_all = torch.from_numpy(cam2_data[1]).to(device)              # [V1, 3, 3]
        K3_t   = torch.from_numpy(cam3_data[1]).unsqueeze(0).to(device) # [1, V3, 3, 3]

    tau_t = torch.tensor([[tau]], dtype=torch.float32, device=device)
    all_pts: list[np.ndarray] = []

    for vi in range(nv1):
        P1_b = P1[vi].unsqueeze(0).to(device)              # [1, 3, H, W]
        C1_b = C1[vi].unsqueeze(0).to(device)

        if is_cam:
            T2_b = T2_all[vi].unsqueeze(0)                 # [1, 4, 4]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = model(
                    P1_b, C1_b,
                    P3_t, C3_t,
                    T2_b,
                    T1_all[vi].unsqueeze(0),
                    T3_t,
                    K2_all[vi].unsqueeze(0),
                    K3_t,
                    tau_t,
                )
            p2_cam = out["P2_cam_hat"].float()             # [1, 3, H, W]
            R2 = T2_b[:, :3, :3]
            c2 = T2_b[:, :3,  3]
            pts_world = (torch.bmm(R2, p2_cam.view(1, 3, -1)) + c2.unsqueeze(-1)).view_as(p2_cam)
            p2_hat  = pts_world.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            p2_conf = out["G"].squeeze().float().cpu().numpy()              # [H, W]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = model(P1_b, C1_b, P3_t, C3_t, tau_t)
            p2_hat  = out["P2_hat"].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            p2_conf = out["G"].squeeze().float().cpu().numpy()

        pts = _pointmap_to_cloud(p2_hat, p2_conf, conf_threshold, 0, seed)
        if len(pts):
            all_pts.append(pts)

    if not all_pts:
        return None
    merged = _postprocess_cloud(np.concatenate(all_pts, axis=0), n_points, seed)
    return merged, all_pts

# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_metrics(rows: list[dict[str, Any]], method_keys: list[str]) -> dict[str, dict[str, float]]:
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


# ── markdown report ───────────────────────────────────────────────────────────

def write_markdown_report(path: Path, all_results: list[dict[str, Any]]) -> None:
    metrics = [
        "asymmetric_chamfer", "f1", "precision", "recall",
        "voxel_iou", "normal_consistency", "height_median_error",
    ]
    lines = ["# TRPM Evaluation Report\n"]
    for result in all_results:
        lines.append(
            f"## fold={result['fold_id']}  "
            f"protocol={result['protocol']}  "
            f"n={result['n_test']}\n"
        )
        header = "| method | " + " | ".join(metrics) + " |"
        sep    = "|--------|" + "|".join(["-------"] * len(metrics)) + "|"
        lines += [header, sep]
        for method, m in result["aggregated"].items():
            vals = " | ".join(f"{m.get(k, float('nan')):.4f}" for k in metrics)
            lines.append(f"| {method} | {vals} |")
        lines.append("")
    path.write_text("\n".join(lines))


# ── fold evaluation ───────────────────────────────────────────────────────────

def _avg_metrics(metric_list: list[dict]) -> dict:
    """Average a list of metric dicts, skipping NaN values."""
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if isinstance(m.get(k), float) and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_fold(
    fold: dict[str, Any],
    model: torch.nn.Module | None,
    cfg: dict[str, Any],
    device: str,
    fold_output_dir: Path,
) -> dict[str, Any]:
    vggt_root           = cfg["vggt_output_root"]
    conf_threshold      = cfg["conf_threshold"]
    pred_conf_threshold = cfg.get("pred_conf_threshold", conf_threshold)
    n_points            = cfg["n_points"]
    seed                = cfg.get("seed", 42)
    threshold           = cfg["distance_threshold"]
    voxel_size          = cfg["voxel_size"]
    alpha               = cfg["eval_alpha"]
    beta                = cfg["eval_beta"]
    save_clouds         = cfg.get("save_clouds", False) and model is not None
    baseline_cache      = cfg.get("baseline_cache")  # Path or None

    method_keys = list(BASELINES) + (["trpm"] if model is not None else [])
    rows: list[dict[str, Any]] = []

    clouds_dir = fold_output_dir / "clouds" if save_clouds else None
    if clouds_dir is not None:
        clouds_dir.mkdir(parents=True, exist_ok=True)

    for triplet in tqdm(fold["test_triplets"], desc=f"fold={fold['fold_id']}"):
        crop        = triplet["crop"]
        left_date   = triplet["left_date"]
        middle_date = triplet["middle_date"]
        right_date  = triplet["right_date"]
        triplet_id  = f"{left_date}_{middle_date}_{right_date}_{crop}"
        tau         = float(triplet["tau"])

        variants = _list_variants(vggt_root, triplet_id)
        if not variants:
            log(f"  skip {triplet_id}: no variants found")
            continue

        # Per-method list of per-variant metric dicts
        variant_metrics: dict[str, list[dict]] = {k: [] for k in method_keys}
        pred_views_all: list[np.ndarray] = []
        ref_views_all:  list[np.ndarray] = []

        # Load cached baseline metrics if available
        cache_file = Path(baseline_cache) / f"{triplet_id}.json" if baseline_cache else None
        cached_baselines: dict | None = None
        if cache_file and cache_file.exists():
            cached_baselines = json.loads(cache_file.read_text())

        for variant in tqdm(variants, desc=f"  {triplet_id}", leave=False):
            if cached_baselines is not None:
                # Reuse cached per-variant baseline metrics
                for baseline in BASELINES:
                    if baseline in cached_baselines.get(variant, {}):
                        variant_metrics[baseline].append(cached_baselines[variant][baseline])
                need_data = model is not None
            else:
                need_data = True

            if not need_data:
                continue

            v_clouds = _load_variant_clouds(vggt_root, triplet_id, variant,
                                            conf_threshold, n_points, seed)
            if v_clouds is None:
                continue
            pts_t1, pts_t2, pts_t3 = v_clouds

            if cached_baselines is None:
                for baseline in BASELINES:
                    pred = apply_baseline(baseline, pts_t1, pts_t2, pts_t3, tau, n_points, seed)
                    m = compute_metrics(pred[:, :3], pts_t2[:, :3], threshold=threshold,
                                        voxel_size=voxel_size, alpha=alpha, beta=beta)
                    variant_metrics[baseline].append(m)

            if model is not None:
                result = predict_trpm_variant(
                    model, vggt_root, triplet_id, variant, tau, device,
                    pred_conf_threshold, n_points, seed,
                )
                if result is not None:
                    pred_cloud, pred_views = result
                    m = compute_metrics(pred_cloud[:, :3], pts_t2[:, :3], threshold=threshold,
                                        voxel_size=voxel_size, alpha=alpha, beta=beta)
                    variant_metrics["trpm"].append(m)
                    if save_clouds:
                        pred_views_all.extend(pred_views)
                        ref_views_all.extend(
                            _load_views(vggt_root / triplet_id / variant / "t2",
                                        conf_threshold, seed)
                        )

        # Save baseline metrics to cache if we just computed them
        if cache_file and cached_baselines is None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                v: {b: variant_metrics[b][i] for b in BASELINES if i < len(variant_metrics[b])}
                for i, v in enumerate(variants)
            }
            cache_file.write_text(json.dumps(cache_data, indent=2))

        # Average metrics across variants
        row: dict[str, Any] = {
            "triplet_id": triplet_id, "crop": crop,
            "left_date": left_date, "middle_date": middle_date, "right_date": right_date,
            "tau": tau, "sensor_consistent": triplet.get("sensor_consistent", True),
            "n_variants": len(variants),
        }
        for method in method_keys:
            if variant_metrics[method]:
                row[method] = _avg_metrics(variant_metrics[method])

        if save_clouds and pred_views_all:
            np.savez(clouds_dir / f"{triplet_id}_pred.npz",
                     **{f"v{i:02d}": v for i, v in enumerate(pred_views_all)})
            np.savez(clouds_dir / f"{triplet_id}_ref.npz",
                     **{f"v{i:02d}": v for i, v in enumerate(ref_views_all)})

        rows.append(row)

    aggregated = aggregate_metrics(rows, method_keys)

    result = {
        "fold_id": fold["fold_id"],
        "crop": fold["crop"],
        "protocol": fold["protocol"],
        "test_date": fold["test_date"],
        "n_test": len(rows),
        "aggregated": aggregated,
        "triplet_rows": rows,
    }
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(fold_output_dir / "eval_result.json", result)

    for method, metrics in aggregated.items():
        log(f"  fold={fold['fold_id']} {method}: "
            f"chamfer={metrics.get('asymmetric_chamfer', float('nan')):.4f}  "
            f"f1={metrics.get('f1', 0.0):.4f}  "
            f"precision={metrics.get('precision', 0.0):.4f}  "
            f"recall={metrics.get('recall', 0.0):.4f}  "
            f"voxel_iou={metrics.get('voxel_iou', 0.0):.4f}")

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    cfg    = build_config(args)
    device = choose_device(cfg["device"])

    all_folds   = build_all_folds(cfg["triplets_path"])
    output_root = cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for protocol in cfg["protocols"]:
        for fold in all_folds.get(protocol, []):
            if fold["crop"] not in cfg["crops"]:
                continue
            if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                continue
            if not fold["test_triplets"]:                continue

            fold_id    = fold["fold_id"]
            checkpoint = cfg["runs_root"] / protocol / fold_id / "best_model.pt"

            model: torch.nn.Module | None = None
            if not cfg.get("baselines_only") and checkpoint.exists():
                model_cls = _load_model_class(cfg.get("model_class", "trpm.model.TRPMSmall"))
                model = model_cls(**cfg.get("model_kwargs", {})).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                model.eval()
                log(f"--- fold={fold_id} protocol={protocol} n_test={fold['n_test']} (model + baselines) ---")
            else:
                log(f"--- fold={fold_id} protocol={protocol} n_test={fold['n_test']} (baselines only, no checkpoint at {checkpoint}) ---")

            fold_output_dir = output_root / protocol / fold_id
            result = evaluate_fold(fold, model, cfg, device, fold_output_dir)
            all_results.append({k: v for k, v in result.items() if k != "triplet_rows"})

    write_json(output_root / "eval_summary.json", all_results)
    log(f"done. summary → {output_root / 'eval_summary.json'}")

    write_markdown_report(output_root / "eval_report.md", all_results)
    log(f"report → {output_root / 'eval_report.md'}")


if __name__ == "__main__":
    main()

