"""Dataset for temporal triplet training with images and VGGT-derived supervision.

Each sample is a (t1, t2, t3) temporal triplet loaded from a completed VGGT
inference variant under vggt_output_root/{triplet_id}/variant_XX/{t1,t2,t3}/.

The dataset indexes all completed variants and returns, per sample:
  - t1 and t3 multi-view images and camera parameters (context)
  - t2 camera parameters for num_query_views randomly selected views (query)
  - VGGT-predicted t2 point maps / depths for those query views (supervision)

Returned camera dicts are compatible with build_camera_features() in
models/camera_encoding.py, which expects:
    transform_matrix [V, 4, 4], fl_x/fl_y/cx/cy/img_w/img_h [V],
    avg_pos [3], scale [] (scalar)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> tuple[float, int]:
    """Return (day_of_year, ordinal) for a date string like '20230812'."""
    d = datetime.strptime(date_str, "%Y%m%d").date()
    return float(d.timetuple().tm_yday), d.toordinal()


def _load_images(date_dir: Path, mode: str) -> torch.Tensor:
    """Load all selected images for one date dir. Returns [V, 3, H, W] float32."""
    from vggt.utils.load_fn import load_and_preprocess_images

    selected = json.loads((date_dir / "selected_images.json").read_text())
    paths = [entry["image_path"] for entry in selected]
    return load_and_preprocess_images(paths, mode=mode)


def _load_cameras(date_dir: Path) -> dict[str, Any]:
    """Read dataset_cameras.json saved by run_vggt_inference."""
    return json.loads((date_dir / "dataset_cameras.json").read_text())


def _load_geometry(
    date_dir: Path,
    query_indices: list[int],
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Load available VGGT predictions for selected query view indices.

    Returns:
        Optional tensors for point/depth/camera predictions. Missing files are
        returned as None so cached-feature-only training can skip them.
    """
    pred_dir = date_dir / "predictions"
    query = np.array(query_indices)

    def load_optional(name: str, squeeze_channel: bool = False) -> torch.Tensor | None:
        path = pred_dir / name
        if not path.exists():
            return None
        arr = np.load(path).astype(np.float32)
        if squeeze_channel and arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return torch.from_numpy(arr[query])

    return (
        load_optional("point_map.npy"),
        load_optional("point_confidence.npy", squeeze_channel=True),
        load_optional("depth_map.npy", squeeze_channel=True),
        load_optional("depth_confidence.npy", squeeze_channel=True),
        load_optional("extrinsic.npy"),
        load_optional("intrinsic.npy"),
    )


def _compute_scene_normalization(
    cams_t1: dict[str, Any], cams_t3: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute avg_pos [3] and scale [] from the union of t1 and t3 camera centers."""
    positions = []
    for cams in (cams_t1, cams_t3):
        for frame in cams["frames"]:
            m = np.array(frame["transform_matrix"], dtype=np.float64)
            positions.append(m[:3, 3])
    positions = np.array(positions, dtype=np.float32)
    avg_pos = positions.mean(axis=0)
    dists = np.linalg.norm(positions - avg_pos, axis=1)
    scale = 1.0 / (float(np.percentile(dists, 90)) + 1e-6)
    return torch.from_numpy(avg_pos), torch.tensor(scale, dtype=torch.float32)


def _build_camera_dict(
    cams: dict[str, Any],
    frame_indices: list[int],
    avg_pos: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build a camera dict for build_camera_features() from dataset_cameras.json."""
    intrinsics = cams["intrinsics"]
    frames = cams["frames"]
    V = len(frame_indices)

    transform_matrices = torch.stack([
        torch.tensor(frames[i]["transform_matrix"], dtype=torch.float32)
        for i in frame_indices
    ])  # [V, 4, 4]

    fl_x = torch.full((V,), float(intrinsics["fl_x"]), dtype=torch.float32)
    fl_y = torch.full((V,), float(intrinsics["fl_y"]), dtype=torch.float32)
    cx   = torch.full((V,), float(intrinsics["cx"]),   dtype=torch.float32)
    cy   = torch.full((V,), float(intrinsics["cy"]),   dtype=torch.float32)
    img_w = torch.full((V,), float(intrinsics["w"]),   dtype=torch.float32)
    img_h = torch.full((V,), float(intrinsics["h"]),   dtype=torch.float32)

    return {
        "transform_matrix": transform_matrices,
        "fl_x": fl_x, "fl_y": fl_y,
        "cx": cx, "cy": cy,
        "img_w": img_w, "img_h": img_h,
        "avg_pos": avg_pos,
        "scale": scale,
    }


def _parse_triplet_dir(triplet_dir_name: str) -> tuple[str, str, str, str] | None:
    """Parse '{t1_date}_{t2_date}_{t3_date}_{crop}' → (t1, t2, t3, crop) or None."""
    parts = triplet_dir_name.split("_")
    if len(parts) < 4:
        return None
    crop = parts[-1]
    dates = parts[:-1]
    if len(dates) != 3 or not all(d.isdigit() and len(d) == 8 for d in dates):
        return None
    return dates[0], dates[1], dates[2], crop


# ── live-inference helpers ────────────────────────────────────────────────────

class _LiveContextImages(Dataset):
    """Images-only view of LiveTripletDataset — loads t1/t3 images without running VGGT."""

    def __init__(self, entries: list[dict[str, Any]], image_preprocess_mode: str) -> None:
        self.entries = entries
        self.image_preprocess_mode = image_preprocess_mode

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from vggt.utils.load_fn import load_and_preprocess_images
        entry = self.entries[idx]
        batch = entry["batch_triplets"]
        paths_t1, _ = _resolve_date_views_live([t["v1"] for t in batch])
        paths_t3, _ = _resolve_date_views_live([t["v3"] for t in batch])
        return {
            "images_t1": load_and_preprocess_images(paths_t1, mode=self.image_preprocess_mode),
            "images_t3": load_and_preprocess_images(paths_t3, mode=self.image_preprocess_mode),
        }



def _resolve_date_views_live(
    date_views: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Resolve absolute image paths and build dataset_cameras from triplet view dicts.

    Mirrors resolve_date_views() in run_vggt_inference.py but returns only
    (image_paths, dataset_cameras) — no disk writes.
    """
    transforms_path = Path(date_views[0]["source_transforms_path"])
    scene_root = transforms_path.parent
    transforms = json.loads(transforms_path.read_text())

    image_paths: list[str] = []
    camera_frames: list[dict[str, Any]] = []
    for i, view in enumerate(date_views):
        abs_path = str((scene_root / view["image_path"]).resolve())
        image_paths.append(abs_path)
        camera_frames.append({
            "frame_index": i,
            "file_path": view["image_path"],
            "image_path": abs_path,
            "transform_matrix": view.get("transform_matrix", []),
        })

    dataset_cameras: dict[str, Any] = {
        "scene_root": str(scene_root),
        "source_transforms_path": str(transforms_path),
        "intrinsics": {k: transforms.get(k) for k in (
            "w", "h", "fl_x", "fl_y", "cx", "cy",
            "k1", "k2", "k3", "k4", "p1", "p2", "camera_model",
        )},
        "frames": camera_frames,
    }
    return image_paths, dataset_cameras


def _load_geometry_live(
    image_paths: list[str],
    runner: Any,
    image_preprocess_mode: str,
    query_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run live VGGT inference and return geometry for selected query views."""
    from vggt_pipeline.execute_vggt import run_vggt_inference_in_memory

    preds = run_vggt_inference_in_memory(image_paths, runner, image_preprocess_mode)

    pm  = preds["point_map"]         # [S, H, W, 3]
    pc  = preds["point_confidence"]  # [S, H, W]
    dm  = preds["depth_map"]         # [S, H, W]
    dc  = preds["depth_confidence"]  # [S, H, W]
    ext = preds["extrinsic"]         # [S, 3, 4]
    intr = preds["intrinsic"]        # [S, 3, 3]

    if pc.ndim == 4 and pc.shape[-1] == 1:
        pc = pc[..., 0]
    if dm.ndim == 4 and dm.shape[-1] == 1:
        dm = dm[..., 0]
    if dc.ndim == 4 and dc.shape[-1] == 1:
        dc = dc[..., 0]

    q = torch.tensor(query_indices)
    return pm[q], pc[q], dm[q], dc[q], ext[q], intr[q]


# ── dataset ───────────────────────────────────────────────────────────────────

class TemporalTripletDataset(Dataset):
    """Loads (t1-images+cameras, t3-images+cameras, t2-query-cameras, t2-geometry).

    Scans vggt_output_root for all completed (triplet_id, variant) pairs where
    t1/, t2/, t3/ subdirs each have predictions/point_map.npy.

    Each __getitem__ randomly selects num_query_views from t2's cameras
    (deterministic per index for reproducibility).
    """

    def __init__(
        self,
        vggt_output_root: Path,
        image_preprocess_mode: str = "pad",
        conf_threshold: float = 0.02,
        num_query_views: int = 1,
        seed: int = 42,
        feature_cache=None,  # VGGTFeatureCache | None
    ) -> None:
        self.vggt_output_root = Path(vggt_output_root)
        self.image_preprocess_mode = image_preprocess_mode
        self.conf_threshold = conf_threshold
        self.num_query_views = num_query_views
        self.seed = seed
        self.feature_cache = feature_cache
        self.index = self._build_index()

    def _build_index(self) -> list[dict[str, Any]]:
        index = []
        for triplet_dir in sorted(self.vggt_output_root.iterdir()):
            if not triplet_dir.is_dir():
                continue
            parsed = _parse_triplet_dir(triplet_dir.name)
            if parsed is None:
                continue
            t1_date, t2_date, t3_date, crop = parsed

            for variant_dir in sorted(triplet_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                # t1/t2/t3 need camera metadata. Prediction .npy files are
                # optional for cached-feature-only training.
                if not all(
                    (variant_dir / dl / "dataset_cameras.json").exists()
                    for dl in ("t1", "t2", "t3")
                ):
                    continue
                index.append({
                    "variant_dir": variant_dir,
                    "triplet_id": triplet_dir.name,
                    "crop": crop,
                    "t1_date": t1_date,
                    "t2_date": t2_date,
                    "t3_date": t3_date,
                    "variant": variant_dir.name,
                })
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.index[idx]
        variant_dir: Path = entry["variant_dir"]
        t1_date, t2_date, t3_date = entry["t1_date"], entry["t2_date"], entry["t3_date"]

        t1_dir = variant_dir / "t1"
        t2_dir = variant_dir / "t2"
        t3_dir = variant_dir / "t3"

        # Compute cache keys (no I/O).
        cache = self.feature_cache
        t1_key = cache.key(t1_dir) if cache else None
        t3_key = cache.key(t3_dir) if cache else None
        t2_key = cache.key(t2_dir) if cache else None

        # Skip image loading entirely when features are cached — images are only
        # needed to run the frozen VGGT aggregator, which is skipped on cache hit.
        images_t1 = None if (t1_key and cache.exists(t1_key)) else _load_images(t1_dir, self.image_preprocess_mode)
        images_t3 = None if (t3_key and cache.exists(t3_key)) else _load_images(t3_dir, self.image_preprocess_mode)
        images_t2 = None if (t2_key and cache.exists(t2_key)) else _load_images(t2_dir, self.image_preprocess_mode)

        # Load cameras for all three dates
        cams_t1 = _load_cameras(t1_dir)
        cams_t2 = _load_cameras(t2_dir)
        cams_t3 = _load_cameras(t3_dir)

        # Scene normalization from context cameras (t1 + t3)
        avg_pos, scale = _compute_scene_normalization(cams_t1, cams_t3)

        # Select query views from t2 — deterministic per (seed, idx)
        n_t2 = len(cams_t2["frames"])
        q = min(self.num_query_views, n_t2)
        rng = np.random.default_rng(self.seed + idx)
        query_indices = rng.choice(n_t2, size=q, replace=False).tolist()

        # Build camera dicts for model
        all_t1 = list(range(len(cams_t1["frames"])))
        all_t3 = list(range(len(cams_t3["frames"])))
        camera_t1 = _build_camera_dict(cams_t1, all_t1, avg_pos, scale)
        camera_t3 = _build_camera_dict(cams_t3, all_t3, avg_pos, scale)
        camera_t2_query = _build_camera_dict(cams_t2, query_indices, avg_pos, scale)

        # Load t2 predictions for selected query views
        pt_maps, pt_conf, depths, d_conf, vggt_ext, vggt_intr = _load_geometry(t2_dir, query_indices)
        # pt_maps: [Q, H, W, 3], pt_conf/depths/d_conf: [Q, H, W]
        # vggt_ext: [Q, 3, 4], vggt_intr: [Q, 3, 3]

        # Parse dates
        doy_t1, ord_t1 = _parse_date(t1_date)
        doy_t2, ord_t2 = _parse_date(t2_date)
        doy_t3, ord_t3 = _parse_date(t3_date)

        return {
            # Context inputs
            "images_t1": images_t1,                                     # [V1, 3, H, W]
            "images_t3": images_t3,                                     # [V3, 3, H, W]
            "images_t2": images_t2,                                     # [V2, 3, H, W] or None
            # Cache keys (strings → collated as list by DataLoader; None when cache disabled)
            "t1_cache_key": t1_key,
            "t3_cache_key": t3_key,
            "t2_cache_key": t2_key,
            "camera_t1": camera_t1,
            "camera_t3": camera_t3,
            "camera_t2_query": camera_t2_query,
            # Temporal features
            "date_t1": torch.tensor(doy_t1, dtype=torch.float32),
            "date_t2": torch.tensor(doy_t2, dtype=torch.float32),
            "date_t3": torch.tensor(doy_t3, dtype=torch.float32),
            "t1_day": torch.tensor(ord_t1, dtype=torch.long),
            "t2_day": torch.tensor(ord_t2, dtype=torch.long),
            "t3_day": torch.tensor(ord_t3, dtype=torch.long),
            # Supervision
            "target_point_maps_t2": pt_maps,                            # [Q, H, W, 3]
            "target_point_confidence_t2": pt_conf,                      # [Q, H, W]
            "target_masks_t2": (pt_conf > self.conf_threshold).float() if pt_conf is not None else None,
            "target_depths_t2": depths,                                  # [Q, H, W]
            "target_depth_masks_t2": (d_conf > self.conf_threshold).float() if d_conf is not None else None,
            "target_vggt_extrinsic_t2": vggt_ext,                      # [Q, 3, 4]
            "target_vggt_intrinsic_t2": vggt_intr,                     # [Q, 3, 3]
            # Metadata
            "triplet_id": entry["triplet_id"],
            "crop": entry["crop"],
            "t1_date": t1_date,
            "t2_date": t2_date,
            "t3_date": t3_date,
            "variant": entry["variant"],
            "query_view_indices": query_indices,
        }


class LiveTripletDataset(Dataset):
    """Identical sample format to TemporalTripletDataset but runs VGGT on-the-fly.

    Indexes from all_triplets entries (not from pre-computed output dirs).
    Each entry's matched triplets are windowed into variants the same way as
    run_vggt_inference.py.

    NOTE: Use num_workers=0 in DataLoader — the VGGT runner holds GPU state
    that cannot be shared across fork-based workers.
    """

    def __init__(
        self,
        all_triplets: list[dict[str, Any]],
        vggt_runner: Any,
        n_views: int = 8,
        max_overlap_views: int = 2,
        max_variants: int | None = None,
        image_preprocess_mode: str = "pad",
        conf_threshold: float = 0.02,
        num_query_views: int = 1,
        seed: int = 42,
    ) -> None:
        self.vggt_runner = vggt_runner
        self.n_views = n_views
        self.max_overlap_views = max_overlap_views
        self.max_variants = max_variants
        self.image_preprocess_mode = image_preprocess_mode
        self.conf_threshold = conf_threshold
        self.num_query_views = num_query_views
        self.seed = seed
        self.index = self._build_index(all_triplets)

    def _build_index(self, all_triplets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stride = max(1, self.n_views - self.max_overlap_views)
        index = []
        for entry in all_triplets:
            matched = entry["triplets"]
            if len(matched) < self.n_views:
                continue
            tid = f"{entry['t1']}_{entry['t2']}_{entry['t3']}_{entry['crop']}"
            batch_idx = 0
            start = 0
            while start + self.n_views <= len(matched):
                index.append({
                    "triplet_id": tid,
                    "crop": entry["crop"],
                    "t1_date": entry["t1"],
                    "t2_date": entry["t2"],
                    "t3_date": entry["t3"],
                    "variant": f"variant_{batch_idx:02d}",
                    "batch_triplets": matched[start: start + self.n_views],
                })
                start += stride
                batch_idx += 1
                if self.max_variants is not None and batch_idx >= self.max_variants:
                    break
        return index

    def __len__(self) -> int:
        return len(self.index)

    def context_images_subset(self, indices: list[int]) -> "_LiveContextImages":
        """Return a dataset that loads only images_t1/images_t3 for the given indices."""
        return _LiveContextImages(
            [self.index[i] for i in indices],
            self.image_preprocess_mode,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from vggt.utils.load_fn import load_and_preprocess_images

        entry = self.index[idx]
        batch = entry["batch_triplets"]

        views_t1 = [t["v1"] for t in batch]
        views_t2 = [t["v2"] for t in batch]
        views_t3 = [t["v3"] for t in batch]

        image_paths_t1, cams_t1 = _resolve_date_views_live(views_t1)
        image_paths_t2, cams_t2 = _resolve_date_views_live(views_t2)
        image_paths_t3, cams_t3 = _resolve_date_views_live(views_t3)

        images_t1 = load_and_preprocess_images(image_paths_t1, mode=self.image_preprocess_mode)
        images_t3 = load_and_preprocess_images(image_paths_t3, mode=self.image_preprocess_mode)

        avg_pos, scale = _compute_scene_normalization(cams_t1, cams_t3)

        n_t2 = len(cams_t2["frames"])
        q = min(self.num_query_views, n_t2)
        rng = np.random.default_rng(self.seed + idx)
        query_indices = rng.choice(n_t2, size=q, replace=False).tolist()

        all_t1 = list(range(len(cams_t1["frames"])))
        all_t3 = list(range(len(cams_t3["frames"])))
        camera_t1 = _build_camera_dict(cams_t1, all_t1, avg_pos, scale)
        camera_t3 = _build_camera_dict(cams_t3, all_t3, avg_pos, scale)
        camera_t2_query = _build_camera_dict(cams_t2, query_indices, avg_pos, scale)

        pt_maps, pt_conf, depths, d_conf, vggt_ext, vggt_intr = _load_geometry_live(
            image_paths_t2, self.vggt_runner, self.image_preprocess_mode, query_indices,
        )

        doy_t1, ord_t1 = _parse_date(entry["t1_date"])
        doy_t2, ord_t2 = _parse_date(entry["t2_date"])
        doy_t3, ord_t3 = _parse_date(entry["t3_date"])

        return {
            "images_t1": images_t1,
            "images_t3": images_t3,
            "camera_t1": camera_t1,
            "camera_t3": camera_t3,
            "camera_t2_query": camera_t2_query,
            "date_t1": torch.tensor(doy_t1, dtype=torch.float32),
            "date_t2": torch.tensor(doy_t2, dtype=torch.float32),
            "date_t3": torch.tensor(doy_t3, dtype=torch.float32),
            "t1_day": torch.tensor(ord_t1, dtype=torch.long),
            "t2_day": torch.tensor(ord_t2, dtype=torch.long),
            "t3_day": torch.tensor(ord_t3, dtype=torch.long),
            "target_point_maps_t2": pt_maps,
            "target_point_confidence_t2": pt_conf,
            "target_masks_t2": (pt_conf > self.conf_threshold).float(),
            "target_depths_t2": depths,
            "target_depth_masks_t2": (d_conf > self.conf_threshold).float(),
            "target_vggt_extrinsic_t2": vggt_ext,
            "target_vggt_intrinsic_t2": vggt_intr,
            "triplet_id": entry["triplet_id"],
            "crop": entry["crop"],
            "t1_date": entry["t1_date"],
            "t2_date": entry["t2_date"],
            "t3_date": entry["t3_date"],
            "variant": entry["variant"],
            "query_view_indices": query_indices,
        }
