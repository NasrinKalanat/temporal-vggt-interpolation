"""Build fused point cloud geometry assets from VGGT triplet inference outputs.

For each (triplet_id, variant, date_label) triple in vggt_output, loads depth_map.npy
and depth_confidence.npy, unprojects into dataset (NeRFStudio) world space via
per-variant Umeyama alignment using dataset_cameras.json saved alongside predictions,
then optionally normalizes with crop-level params derived from the reference date.

All crop dates share the same NeRFStudio GPS coordinate frame, so per-variant
Umeyama alignment gives automatic cross-date alignment within a triplet.

Reference dates:
    corn:     20230812
    soybean:  20230822

Output layout mirrors the inference output:
    geometry_assets/
    └── {triplet_id}/
        └── {variant}/
            ├── t1/
            │   ├── point_cloud_clean.npz
            │   └── geometry_metadata.json
            ├── t2/  (same)
            └── t3/  (same)

Usage:
    python src/vggt_pipeline/build_geometry_assets.py \
        --vggt-root vggt_output_triplets --output-root geometry_assets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_DATES: dict[str, str] = {
    "corn": "20230812",
    "soybean": "20230822",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build geometry assets from VGGT triplet outputs.")
    parser.add_argument("--vggt-root", type=Path, default=Path("vggt_outputs/t1t2_paired_v16_o8"))
    parser.add_argument("--output-root", type=Path, default=Path("geometry_assets"))
    parser.add_argument("--triplet-id", action="append", default=None)
    parser.add_argument("--middle-date", type=str, default=None,
                        help="Only process triplets with this middle date (e.g. 20230831).")
    parser.add_argument("--conf-threshold", type=float, default=0.02)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--outlier-quantile", type=float, default=0.005)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--normalize", action="store_true", default=False,
        help="Apply per-axis affine normalization (subtract crop center/ground_z, divide by scale). "
             "When omitted, points_normalized == points_aligned (NeRFStudio world space).",
    )
    return parser.parse_args()


# ── normalization helpers ──────────────────────────────────────────────────────

def normalize_confidence(conf: np.ndarray) -> np.ndarray:
    mn, mx = conf.min(), conf.max()
    if mx > mn:
        return (conf - mn) / (mx - mn)
    return np.ones_like(conf)


def compute_normalization_params(points: np.ndarray) -> dict[str, float]:
    center_x = float(np.median(points[:, 0]))
    center_y = float(np.median(points[:, 1]))
    ground_z = float(np.percentile(points[:, 2], 2.0))
    xy_centered = points[:, :2] - np.array([center_x, center_y])
    scale = float(np.percentile(np.linalg.norm(xy_centered, axis=1), 95))
    if scale <= 0:
        scale = 1.0
    return {"center_x": center_x, "center_y": center_y, "ground_z": ground_z, "scale": scale}


def normalize_points(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    ground_z: float,
    scale: float,
) -> np.ndarray:
    pts = points.astype(np.float64)
    pts[:, 0] = (pts[:, 0] - center_x) / scale
    pts[:, 1] = (pts[:, 1] - center_y) / scale
    pts[:, 2] = (pts[:, 2] - ground_z) / scale
    return pts.astype(np.float32)


def compute_roi_bounds(points_normalized: np.ndarray) -> dict[str, float]:
    lo = np.percentile(points_normalized, 1, axis=0)
    hi = np.percentile(points_normalized, 99, axis=0)
    margin = np.maximum(0.1 * (hi - lo), 0.05)
    lo_bound = lo - margin
    hi_bound = hi + margin
    return {
        "x_min": float(lo_bound[0]), "x_max": float(hi_bound[0]),
        "y_min": float(lo_bound[1]), "y_max": float(hi_bound[1]),
        "z_min": float(max(lo_bound[2], -0.05)),
        "z_max": float(hi_bound[2]),
    }


def apply_roi_mask(points: np.ndarray, roi: dict[str, float]) -> np.ndarray:
    return (
        (points[:, 0] >= roi["x_min"]) & (points[:, 0] <= roi["x_max"]) &
        (points[:, 1] >= roi["y_min"]) & (points[:, 1] <= roi["y_max"]) &
        (points[:, 2] >= roi["z_min"]) & (points[:, 2] <= roi["z_max"])
    )


def remove_outliers(points: np.ndarray, conf: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, conf
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    threshold = np.quantile(dists, 1.0 - quantile)
    keep = dists <= threshold
    return points[keep], conf[keep]


def voxel_downsample(points: np.ndarray, conf: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0 or voxel_size <= 0:
        return points, conf
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_idx, axis=0, return_index=True)
    return points[unique_indices], conf[unique_indices]


def subsample(points: np.ndarray, conf: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, conf
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], conf[idx]


def parse_triplet_crop_date(triplet_id: str, date_label: str) -> tuple[str | None, str | None]:
    """Parse crop and actual date from triplet_id + date_label.

    triplet_id format: '{t1_date}_{t2_date}_{t3_date}_{crop}'
    date_label: 't1', 't2', or 't3'
    """
    parts = triplet_id.split("_")
    if len(parts) < 4:
        return None, None
    crop = parts[-1]
    dates = parts[:-1]
    label_map = {"t1": 0, "t2": 1, "t3": 2}
    idx = label_map.get(date_label)
    if idx is None or idx >= len(dates):
        return None, None
    return crop, dates[idx]


# ── dataset camera helpers ─────────────────────────────────────────────────────

def scale_intrinsics_for_pad_mode(
    w_orig: float, h_orig: float,
    fl_x: float, fl_y: float, cx: float, cy: float,
    vggt_w: int, vggt_h: int,
) -> tuple[float, float, float, float]:
    """Scale full-resolution intrinsics to VGGT depth map padded to (vggt_w, vggt_h)."""
    scale = min(vggt_w / w_orig, vggt_h / h_orig)
    pad_left = (vggt_w - w_orig * scale) / 2
    pad_top = (vggt_h - h_orig * scale) / 2
    return fl_x * scale, fl_y * scale, cx * scale + pad_left, cy * scale + pad_top


def vggt_extrinsic_to_c2w(ext_3x4: np.ndarray) -> np.ndarray:
    """Invert VGGT world-to-camera extrinsic (3, 4) → camera-to-world (4, 4)."""
    R = ext_3x4[:, :3].astype(np.float64)
    t = ext_3x4[:, 3].astype(np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama similarity: (scale, R, t) such that dst ≈ scale * R @ src + t."""
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


def load_dataset_cameras(date_dir: Path) -> dict[str, Any] | None:
    """Load cameras from dataset_cameras.json saved by run_vggt_inference."""
    cam_path = date_dir / "dataset_cameras.json"
    if not cam_path.exists():
        return None
    cams = read_json(cam_path)
    if len(cams.get("frames", [])) < 4:
        return None
    return cams


def extract_depth_cloud(
    date_dir: Path,
    dataset_cameras: dict[str, Any],
    conf_threshold: float,
    stride: int,
    outlier_quantile: float,
    voxel_size: float,
    max_points: int,
    seed: int,
    preprocess_mode: str = "pad",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unproject VGGT depth maps into dataset (NeRFStudio) world space via Umeyama.

    Returns (points_dataset, points_raw_vggt, conf, view_index) where:
      - points_dataset: (N, 3) in NeRFStudio GPS world space
      - points_raw_vggt: (N, 3) in VGGT world space
      - conf: (N,) normalized confidence
      - view_index: (N,) source view index for each retained point
    """
    pred_dir = date_dir / "predictions"
    depth_map = np.load(pred_dir / "depth_map.npy")
    depth_conf = np.load(pred_dir / "depth_confidence.npy")
    extrinsics = np.load(pred_dir / "extrinsic.npy")   # (S, 3, 4)

    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]

    S, H, W = depth_map.shape
    intrinsics = dataset_cameras["intrinsics"]
    frames = dataset_cameras["frames"]

    fl_x = float(intrinsics["fl_x"])
    fl_y = float(intrinsics["fl_y"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    w_orig = float(intrinsics["w"])
    h_orig = float(intrinsics["h"])

    if preprocess_mode == "pad":
        fl_x_s, fl_y_s, cx_s, cy_s = scale_intrinsics_for_pad_mode(w_orig, h_orig, fl_x, fl_y, cx, cy, W, H)
    else:
        fl_x_s = fl_x * (W / w_orig)
        fl_y_s = fl_y * (H / h_orig)
        cx_s = cx * (W / w_orig)
        cy_s = cy * (H / h_orig)

    n_frames = min(S, len(frames))
    vggt_centers = np.array(
        [-extrinsics[i, :, :3].T @ extrinsics[i, :, 3] for i in range(n_frames)],
        dtype=np.float64,
    )
    dataset_centers = np.array(
        [np.array(frames[i]["transform_matrix"], dtype=np.float64)[:3, 3] for i in range(n_frames)],
        dtype=np.float64,
    )
    scale_a, R_align, t_align = umeyama_similarity(vggt_centers, dataset_centers)

    depth_conf_norm = normalize_confidence(depth_conf)

    stride = max(1, stride)
    us = np.arange(0, W, stride, dtype=np.float64)
    vs = np.arange(0, H, stride, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)

    all_pts_ds: list[np.ndarray] = []
    all_pts_vggt: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_view_idx: list[np.ndarray] = []

    for i in range(n_frames):
        d = depth_map[i, ::stride, ::stride].astype(np.float64)
        c = depth_conf_norm[i, ::stride, ::stride].astype(np.float32)

        x_c = (uu - cx_s) / fl_x_s * d
        y_c = (vv - cy_s) / fl_y_s * d
        z_c = d

        pts_cam = np.stack([x_c.ravel(), y_c.ravel(), z_c.ravel()], axis=1)
        ones = np.ones((len(pts_cam), 1), dtype=np.float64)
        pts_cam_h = np.concatenate([pts_cam, ones], axis=1)

        c2w = vggt_extrinsic_to_c2w(extrinsics[i])
        pts_vggt = (c2w @ pts_cam_h.T).T[:, :3].astype(np.float32)
        pts_ds = (scale_a * R_align @ pts_vggt.T + t_align[:, None]).T.astype(np.float32)

        conf_flat = c.ravel()
        valid = (
            np.isfinite(pts_ds).all(axis=1)
            & np.isfinite(conf_flat)
            & (conf_flat >= conf_threshold)
        )

        all_pts_ds.append(pts_ds[valid])
        all_pts_vggt.append(pts_vggt[valid])
        all_conf.append(conf_flat[valid])
        all_view_idx.append(np.full(int(valid.sum()), i, dtype=np.int16))

    if not all_pts_ds:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int16)

    return (
        np.concatenate(all_pts_ds, axis=0),
        np.concatenate(all_pts_vggt, axis=0),
        np.concatenate(all_conf, axis=0),
        np.concatenate(all_view_idx, axis=0),
    )


def _filter_consistent(
    pts_ds: np.ndarray, pts_vggt: np.ndarray, conf: np.ndarray, view_idx: np.ndarray,
    outlier_quantile: float, voxel_size: float, max_points: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter outliers, voxel-downsample, and subsample; keep pts_vggt consistent."""
    if len(pts_ds) == 0:
        return pts_ds, pts_vggt, conf, view_idx

    centroid = pts_ds.mean(axis=0)
    dists = np.linalg.norm(pts_ds - centroid, axis=1)
    threshold = np.quantile(dists, 1.0 - outlier_quantile)
    keep = dists <= threshold
    pts_ds, pts_vggt, conf, view_idx = pts_ds[keep], pts_vggt[keep], conf[keep], view_idx[keep]

    if voxel_size > 0 and len(pts_ds) > 0:
        voxel_idx = np.floor(pts_ds / voxel_size).astype(np.int64)
        _, uidx = np.unique(voxel_idx, axis=0, return_index=True)
        pts_ds, pts_vggt, conf, view_idx = pts_ds[uidx], pts_vggt[uidx], conf[uidx], view_idx[uidx]

    if max_points > 0 and len(pts_ds) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts_ds), size=max_points, replace=False)
        pts_ds, pts_vggt, conf, view_idx = pts_ds[idx], pts_vggt[idx], conf[idx], view_idx[idx]

    return pts_ds, pts_vggt, conf, view_idx


def discover_triplet_date_variants(vggt_root: Path) -> list[tuple[str, str, str]]:
    """Return (triplet_id, variant, date_label) triples with completed predictions."""
    triples = []
    for triplet_dir in sorted(vggt_root.iterdir()):
        if not triplet_dir.is_dir():
            continue
        for variant_dir in sorted(triplet_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            for date_dir in sorted(variant_dir.iterdir()):
                if date_dir.is_dir() and (date_dir / "predictions" / "depth_map.npy").exists():
                    triples.append((triplet_dir.name, variant_dir.name, date_dir.name))
    return triples


def build_geometry_for_date(
    triplet_id: str,
    variant: str,
    date_label: str,
    vggt_root: Path,
    output_root: Path,
    conf_threshold: float,
    stride: int,
    outlier_quantile: float,
    voxel_size: float,
    max_points: int,
    seed: int,
    skip_existing: bool,
    normalize: bool = False,
    crop_norm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = output_root / triplet_id / variant / date_label
    out_path = out_dir / "point_cloud_clean.npz"

    if skip_existing and out_path.exists():
        log(f"skip existing {triplet_id}/{variant}/{date_label}")
        return {"triplet_id": triplet_id, "variant": variant, "date_label": date_label, "status": "skipped"}

    date_dir = vggt_root / triplet_id / variant / date_label
    preprocess_mode = "pad"
    req_path = date_dir / "run_request.json"
    if req_path.exists():
        preprocess_mode = read_json(req_path).get("image_preprocess_mode", "pad")

    dataset_cams = load_dataset_cameras(date_dir)

    if dataset_cams is not None:
        pts_ds, pts_vggt, conf, view_idx = extract_depth_cloud(
            date_dir, dataset_cams, conf_threshold, stride,
            0.0, 0.0, 0, seed, preprocess_mode,
        )
        pts_ds, pts_vggt, conf, view_idx = _filter_consistent(
            pts_ds, pts_vggt, conf, view_idx, outlier_quantile, voxel_size, max_points, seed,
        )
        geometry_source = "depth_dataset_cameras"
    else:
        log(f"WARNING: dataset_cameras.json missing for {triplet_id}/{variant}/{date_label}; using point_map (VGGT space)")
        pred_dir = date_dir / "predictions"
        pm = np.load(pred_dir / "point_map.npy")
        pc = np.load(pred_dir / "point_confidence.npy")
        s = max(1, stride)
        pm_strided = pm[:, ::s, ::s, :]
        pts_ds = pm_strided.reshape(-1, 3).astype(np.float32)
        conf = pc[:, ::s, ::s].reshape(-1).astype(np.float32)
        view_idx = np.repeat(
            np.arange(pm_strided.shape[0], dtype=np.int16),
            pm_strided.shape[1] * pm_strided.shape[2],
        )
        conf = normalize_confidence(conf)
        valid = np.isfinite(pts_ds).all(1) & np.isfinite(conf) & (conf >= conf_threshold)
        pts_ds, conf, view_idx = pts_ds[valid], conf[valid], view_idx[valid]
        pts_vggt = pts_ds.copy()
        pts_ds, pts_vggt, conf, view_idx = _filter_consistent(
            pts_ds, pts_vggt, conf, view_idx, outlier_quantile, voxel_size, max_points, seed,
        )
        geometry_source = "point_map_vggt"

    log(f"{triplet_id}/{variant}/{date_label}: filtered n={len(pts_ds)} source={geometry_source}")

    if normalize:
        if crop_norm is not None:
            center_x = crop_norm["center_x"]
            center_y = crop_norm["center_y"]
            ground_z = crop_norm["ground_z"]
            scale = crop_norm["scale"]
            roi_bounds = crop_norm["roi_bounds"]
            norm_type = "crop_level"
        else:
            params = compute_normalization_params(pts_ds)
            center_x, center_y = params["center_x"], params["center_y"]
            ground_z, scale = params["ground_z"], params["scale"]
            roi_bounds = compute_roi_bounds(normalize_points(pts_ds, center_x, center_y, ground_z, scale))
            norm_type = "per_scene"

        pts_normalized = normalize_points(pts_ds, center_x, center_y, ground_z, scale)
        roi_mask = apply_roi_mask(pts_normalized, roi_bounds)
        pts_normalized = pts_normalized[roi_mask]
        pts_aligned = pts_ds[roi_mask]
        pts_raw = pts_vggt[roi_mask]
        conf_out = conf[roi_mask]
        view_idx_out = view_idx[roi_mask]
    else:
        norm_type = "none"
        pts_normalized = pts_ds
        pts_aligned = pts_ds
        pts_raw = pts_vggt
        conf_out = conf
        view_idx_out = view_idx

    log(f"{triplet_id}/{variant}/{date_label}: final n={len(pts_normalized)} norm={norm_type}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points_raw=pts_raw,
        points_aligned=pts_aligned,
        points_normalized=pts_normalized,
        confidence=conf_out,
        view_index=view_idx_out.astype(np.int16),
    )

    norm_meta: dict[str, Any] = {"type": norm_type}
    if normalize:
        norm_meta.update({
            "center_x": center_x, "center_y": center_y,
            "ground_z": ground_z, "scale": scale, "roi_bounds": roi_bounds,
        })

    meta: dict[str, Any] = {
        "triplet_id": triplet_id,
        "variant": variant,
        "date_label": date_label,
        "status": "completed",
        "geometry_source": geometry_source,
        "n_points_pre_roi": int(len(pts_ds)),
        "n_points": int(len(pts_normalized)),
        "stride": stride,
        "conf_threshold": conf_threshold,
        "normalization": norm_meta,
    }
    write_json(out_dir / "geometry_metadata.json", meta)
    return meta


def main() -> None:
    args = parse_args()

    all_triples = discover_triplet_date_variants(args.vggt_root)
    if not all_triples:
        raise RuntimeError(f"No completed predictions found under {args.vggt_root}")

    if args.triplet_id:
        filter_set = set(args.triplet_id)
        all_triples = [(tid, var, dl) for tid, var, dl in all_triples if tid in filter_set]

    if args.middle_date:
        all_triples = [
            (tid, var, dl) for tid, var, dl in all_triples
            if tid.split("_")[1] == args.middle_date
        ]

    log(f"building geometry for {len(all_triples)} (triplet, variant, date) triples output_root={args.output_root}")

    # Compute crop-level normalization from reference date clouds.
    crop_norms: dict[str, dict[str, Any]] = {}
    if not args.normalize:
        log("normalization disabled — points_normalized will equal points_aligned")
    else:
        crop_groups: dict[str, list[tuple[str, str, str]]] = {}
        for triplet_id, variant, date_label in all_triples:
            crop, _ = parse_triplet_crop_date(triplet_id, date_label)
            if crop:
                crop_groups.setdefault(crop, []).append((triplet_id, variant, date_label))

        for crop, triples in crop_groups.items():
            ref_date = REFERENCE_DATES.get(crop)
            ref_triple = next(
                ((tid, var, dl) for tid, var, dl in triples
                 if parse_triplet_crop_date(tid, dl)[1] == ref_date),
                None,
            )
            if ref_triple is None:
                log(f"WARNING: reference date {ref_date} not found for crop={crop}; will use per-scene normalization")
                continue

            ref_tid, ref_var, ref_dl = ref_triple
            ref_date_dir = args.vggt_root / ref_tid / ref_var / ref_dl
            preprocess_mode = "pad"
            req_path = ref_date_dir / "run_request.json"
            if req_path.exists():
                preprocess_mode = read_json(req_path).get("image_preprocess_mode", "pad")

            dataset_cams = load_dataset_cameras(ref_date_dir)
            if dataset_cams is not None:
                ref_pts, _, _, _ = extract_depth_cloud(
                    ref_date_dir, dataset_cams,
                    args.conf_threshold, args.stride,
                    args.outlier_quantile, args.voxel_size, args.max_points, args.seed, preprocess_mode,
                )
            else:
                log(f"WARNING: dataset_cameras.json missing for ref {ref_tid}/{ref_var}/{ref_dl}; using point_map")
                pred_dir = ref_date_dir / "predictions"
                pm = np.load(pred_dir / "point_map.npy")
                pc = np.load(pred_dir / "point_confidence.npy")
                s = max(1, args.stride)
                ref_pts = pm[:, ::s, ::s, :].reshape(-1, 3).astype(np.float32)
                conf = pc[:, ::s, ::s].reshape(-1).astype(np.float32)
                conf = normalize_confidence(conf)
                valid = np.isfinite(ref_pts).all(1) & np.isfinite(conf) & (conf >= args.conf_threshold)
                ref_pts = ref_pts[valid]

            if len(ref_pts) == 0:
                log(f"WARNING: empty reference cloud for crop={crop}; will use per-scene normalization")
                continue

            params = compute_normalization_params(ref_pts)
            center_x, center_y = params["center_x"], params["center_y"]
            ground_z, scale = params["ground_z"], params["scale"]
            ref_norm = normalize_points(ref_pts, center_x, center_y, ground_z, scale)
            roi_bounds = compute_roi_bounds(ref_norm)
            crop_norms[crop] = {
                "reference_date": ref_date,
                "reference_triplet_id": ref_tid,
                "reference_variant": ref_var,
                "reference_date_label": ref_dl,
                "center_x": center_x,
                "center_y": center_y,
                "ground_z": ground_z,
                "scale": scale,
                "roi_bounds": roi_bounds,
            }
            log(
                f"crop={crop} normalization from {ref_tid}/{ref_var}/{ref_dl}: "
                f"center=({center_x:.4f},{center_y:.4f}) ground_z={ground_z:.4f} scale={scale:.4f}"
            )

    if crop_norms:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "crop_normalization.json", crop_norms)

    summary: list[dict[str, Any]] = []
    for triplet_id, variant, date_label in all_triples:
        crop, _ = parse_triplet_crop_date(triplet_id, date_label)
        crop_norm = crop_norms.get(crop) if crop else None
        meta = build_geometry_for_date(
            triplet_id=triplet_id,
            variant=variant,
            date_label=date_label,
            vggt_root=args.vggt_root,
            output_root=args.output_root,
            conf_threshold=args.conf_threshold,
            stride=args.stride,
            outlier_quantile=args.outlier_quantile,
            voxel_size=args.voxel_size,
            max_points=args.max_points,
            seed=args.seed,
            skip_existing=args.skip_existing,
            normalize=args.normalize,
            crop_norm=crop_norm,
        )
        summary.append(meta)

    write_json(args.output_root / "geometry_summary.json", summary)
    log(f"done. summary: {args.output_root / 'geometry_summary.json'}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
