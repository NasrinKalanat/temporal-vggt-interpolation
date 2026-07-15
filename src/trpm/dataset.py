"""Dataset for TRPM-Small: loads all views per variant in GPS world space.

GPS conversion uses Umeyama alignment: VGGT camera centers (from extrinsic.npy,
-R.T @ t) matched against GPS camera centers (dataset_cameras.json transform_matrix[:3,3]).

One sample = one variant. Returns all V views per date as [V, 3, H, W].
Train loop reshapes [B, V, 3, H, W] → [B*V, 3, H, W] to treat views as batch.

Camera-aware extension (TRPMSmallCam):
    Also returns per-view c2w poses and scaled intrinsics from dataset_cameras.json.
    Poses:      T1_c2w [V, 4, 4], T2_c2w [V, 4, 4], T3_c2w [V3, 4, 4]
    Intrinsics: K2 [V, 3, 3]  (t2 intrinsics scaled to point-map resolution)
    TRPMSmall ignores these keys; TRPMSmallCam uses them.

Directory layout expected:
    vggt_output_root/{triplet_id}/variant_XX/{t1,t2,t3}/predictions/
        point_map.npy        [S, H, W, 3]
        point_confidence.npy [S, H, W]
        extrinsic.npy        [S, 3, 4]   world-to-cam
    vggt_output_root/{triplet_id}/variant_XX/{t1,t2,t3}/
        dataset_cameras.json  frames[i].transform_matrix  4x4 c2w GPS
                              intrinsics  COLMAP intrinsics at original resolution
"""
from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _parse_triplet_dir(name: str) -> tuple[str, str, str, str] | None:
    parts = name.split("_")
    if len(parts) < 4:
        return None
    crop = parts[-1]
    dates = parts[:-1]
    if len(dates) != 3 or not all(d.isdigit() and len(d) == 8 for d in dates):
        return None
    return dates[0], dates[1], dates[2], crop


def _parse_date(date_str: str) -> int:
    return datetime.strptime(date_str, "%Y%m%d").date().toordinal()


def _load_pointmaps(pred_dir: Path, view_idx: int) -> tuple[np.ndarray, np.ndarray]:
    pm = np.load(pred_dir / "point_map.npy", mmap_mode="r")[view_idx].astype(np.float32)
    pc = np.load(pred_dir / "point_confidence.npy", mmap_mode="r")[view_idx].astype(np.float32)
    if pc.ndim == 3:
        pc = pc[..., 0]
    return pm, pc


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, R, t) such that dst ≈ scale * R @ src + t."""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = float(np.mean(np.sum(sc ** 2, axis=1)))
    cov = (dc.T @ sc) / n
    U, S, Vt = np.linalg.svd(cov)
    d = float(np.sign(np.linalg.det(U @ Vt)))
    D = np.diag([1., 1., d])
    R = U @ D @ Vt
    scale = float(np.sum(S * np.diag(D)) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t.astype(np.float64)


def _gps_alignment(date_dir: Path) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Compute Umeyama (scale, R, t): VGGT world → GPS world.

    VGGT camera centers: -R.T @ t from extrinsic.npy [S, 3, 4] (world-to-cam).
    GPS camera centers: transform_matrix[:3, 3] from dataset_cameras.json (c2w GPS).
    """
    ext_path = date_dir / "predictions" / "extrinsic.npy"
    cam_path = date_dir / "dataset_cameras.json"
    if not ext_path.exists() or not cam_path.exists():
        return None
    ext    = np.load(ext_path, mmap_mode="r").astype(np.float64)   # [S, 3, 4]
    frames = json.loads(cam_path.read_text())["frames"]
    n = min(len(frames), ext.shape[0])
    if n < 4:
        return None
    vggt_centers = np.array(
        [-ext[i, :, :3].T @ ext[i, :, 3] for i in range(n)], dtype=np.float64
    )
    gps_centers = np.array(
        [np.array(frames[i]["transform_matrix"], dtype=np.float64)[:3, 3] for i in range(n)],
        dtype=np.float64,
    )
    return _umeyama(vggt_centers, gps_centers)


def _to_gps(pm: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform [H, W, 3] point map from VGGT world to GPS world space."""
    H, W = pm.shape[:2]
    pts = pm.reshape(-1, 3).astype(np.float64)
    pts_gps = (scale * R @ pts.T + t[:, None]).T
    return pts_gps.reshape(H, W, 3).astype(np.float32)


def _load_camera_data(date_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load c2w poses and COLMAP intrinsics scaled to point-map resolution.

    Returns:
        poses:      [S, 4, 4]  c2w GPS transform matrices
        intrinsics: [S, 3, 3]  K matrices scaled to point-map resolution
    Returns None if data is missing or incomplete.
    """
    cam_path = date_dir / "dataset_cameras.json"
    pm_path  = date_dir / "predictions" / "point_map.npy"
    if not cam_path.exists() or not pm_path.exists():
        return None
    cam    = json.loads(cam_path.read_text())
    frames = cam["frames"]
    intr   = cam.get("intrinsics") or {}

    pm_shape = np.load(pm_path, mmap_mode="r").shape  # [S, H, W, 3]
    H, W = pm_shape[1], pm_shape[2]
    S    = min(len(frames), pm_shape[0])

    orig_w = intr.get("w") or 1
    orig_h = intr.get("h") or 1
    sx = W / orig_w
    sy = H / orig_h

    fl_x = (intr.get("fl_x") or 1.0) * sx
    fl_y = (intr.get("fl_y") or 1.0) * sy
    cx   = (intr.get("cx")   or W / 2) * sx
    cy   = (intr.get("cy")   or H / 2) * sy

    K = np.array([[fl_x, 0.,   cx],
                  [0.,   fl_y, cy],
                  [0.,   0.,   1.]], dtype=np.float32)

    poses = np.stack(
        [np.array(frames[i]["transform_matrix"], dtype=np.float32) for i in range(S)]
    )  # [S, 4, 4]
    intrinsics = np.stack([K] * S)  # [S, 3, 3]  same K for all views of this date

    return poses, intrinsics


def _preprocess_pointmap(
    pm: np.ndarray, pc: np.ndarray, conf_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out low-conf and outlier pixels in-place. Keeps spatial layout intact.

    Args:
        pm: [H, W, 3]  point map
        pc: [H, W]     confidence
    Returns:
        pm, pc with invalid pixels zeroed
    """
    mask = pc >= conf_threshold                          # [H, W] valid conf
    valid_pts = pm[mask]                                 # [N, 3]
    if len(valid_pts) > 0:
        centroid = valid_pts.mean(axis=0)
        dists = np.linalg.norm(valid_pts - centroid, axis=1)
        dist_threshold = np.quantile(dists, 0.995)
        outlier_mask = np.zeros_like(mask)
        outlier_mask[mask] = dists > dist_threshold      # outliers among valid pts
        mask = mask & ~outlier_mask
    pm = pm.copy()
    pc = pc.copy()
    pm[~mask] = 0.0
    pc[~mask] = 0.0
    return pm, pc


def _load_preprocessed(
    pred_dir: Path, view_idx: int, conf_threshold: float,
    alignment: tuple | None,
    cache_dir: Path | None,
    vggt_root: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load GPS-aligned, preprocessed (pm, pc) for one view, using disk cache.

    Cache mirrors the source structure: {cache_dir}/{pred_dir relative to vggt_root}/v{i}_{thr}.npz
    """
    if cache_dir is not None and vggt_root is not None:
        thr_hex = struct.pack('>d', conf_threshold).hex()
        rel = pred_dir.relative_to(vggt_root)
        cache_path = cache_dir / rel / f"v{view_idx}_{thr_hex}.npz"
        if cache_path.exists():
            d = np.load(cache_path)
            return d["pm"], d["pc"]
    else:
        cache_path = None
    pm, pc = _load_pointmaps(pred_dir, view_idx)
    if alignment is not None:
        pm = _to_gps(pm, *alignment)
    pm, pc = _preprocess_pointmap(pm, pc, conf_threshold)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, pm=pm, pc=pc)
    return pm, pc


def _load_rgb_image(date_dir: Path, view_idx: int, H: int, W: int) -> np.ndarray:
    """Load RGB image for one view and resize to point-map resolution.

    Returns:
        rgb: [3, H, W], float32, range [0, 1]
    """
    cam_path = date_dir / "dataset_cameras.json"
    cam = json.loads(cam_path.read_text())
    frames = cam["frames"]
    frame = frames[view_idx]

    # Try selected_images.json first (has absolute image_path)
    sel_path = date_dir / "selected_images.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        if view_idx < len(sel) and "image_path" in sel[view_idx]:
            img_path = Path(sel[view_idx]["image_path"])
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = img.resize((W, H), Image.BILINEAR)
                rgb = np.asarray(img).astype(np.float32) / 255.0
                return np.transpose(rgb, (2, 0, 1))

    # Fallback: resolve from frame file_path
    rel_path = frame.get("file_path") or frame.get("image_path") or frame.get("name")
    if rel_path is None:
        raise RuntimeError(f"No image path found for view {view_idx} in {cam_path}")

    img_path = Path(rel_path)
    if not img_path.is_absolute():
        candidates = [
            date_dir / img_path,
            date_dir / "images" / img_path.name,
            date_dir.parent / img_path,
            date_dir.parent / "images" / img_path.name,
        ]
        img_path = next((p for p in candidates if p.exists()), None)

    if img_path is None or not img_path.exists():
        raise FileNotFoundError(f"Could not find RGB image for view {view_idx}: {rel_path}")

    img = Image.open(img_path).convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


class PointMapTripletDataset(Dataset):
    """One sample per variant. Returns all V views per date in GPS world space.

    Each item always contains:
        P1, C1: [V,  3, H, W] / [V,  1, H, W]  — t1 in GPS space
        P2, C2: [V,  3, H, W] / [V,  1, H, W]  — t2 in GPS space (target)
        P3, C3: [V3, 3, H, W] / [V3, 1, H, W]  — t3 in GPS space
        tau:    [1]

    Camera-aware extras (always present when dataset_cameras.json has intrinsics):
        T1_c2w: [V,  4, 4]   t1 camera-to-world poses (GPS)
        T2_c2w: [V,  4, 4]   t2 camera-to-world poses (GPS)
        T3_c2w: [V3, 4, 4]   t3 camera-to-world poses (GPS)
        K1:     [V,  3, 3]   t1 intrinsics scaled to point-map resolution
        K2:     [V,  3, 3]   t2 intrinsics scaled to point-map resolution
        K3:     [V3, 3, 3]   t3 intrinsics scaled to point-map resolution

    Color extras (when load_rgb=True):
        I1: [V,  3, H, W]   t1 RGB images
        I2: [V,  3, H, W]   t2 RGB images (supervision only)
        I3: [V3, 3, H, W]   t3 RGB images

    TRPMSmall ignores the camera-aware keys. TRPMSmallCam uses them.
    TRPMSmallCamColor uses both camera-aware and color keys.
    """

    def __init__(self, vggt_output_root: Path, conf_threshold: float = 1.0, cache_dir: Path | None = None, load_rgb: bool = False, load_depth: bool = False) -> None:
        self.conf_threshold = conf_threshold
        self.vggt_root = Path(vggt_output_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.index = self._build_index(self.vggt_root)

    def _build_index(self, root: Path) -> list[dict[str, Any]]:
        index = []
        for triplet_dir in sorted(root.iterdir()):
            if not triplet_dir.is_dir():
                continue
            parsed = _parse_triplet_dir(triplet_dir.name)
            if parsed is None:
                continue
            t1_date, t2_date, t3_date, _ = parsed
            tau = ((_parse_date(t2_date) - _parse_date(t1_date)) /
                   max(_parse_date(t3_date) - _parse_date(t1_date), 1))

            for variant_dir in sorted(triplet_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                t1_dir = variant_dir / "t1"
                t2_dir = variant_dir / "t2"
                t3_dir = variant_dir / "t3"
                if not all(
                    (d / "predictions" / "point_map.npy").exists() and
                    (d / "predictions" / "extrinsic.npy").exists() and
                    (d / "dataset_cameras.json").exists()
                    for d in (t1_dir, t2_dir, t3_dir)
                ):
                    continue
                num_views = np.load(
                    t1_dir / "predictions" / "point_map.npy", mmap_mode="r"
                ).shape[0]
                num_views_t3 = np.load(
                    t3_dir / "predictions" / "point_map.npy", mmap_mode="r"
                ).shape[0]
                index.append({
                    "t1_dir": t1_dir,
                    "t2_dir": t2_dir,
                    "t3_dir": t3_dir,
                    "tau": float(tau),
                    "num_views": num_views,
                    "num_views_t3": num_views_t3,
                    "triplet_id": triplet_dir.name,
                    "variant": variant_dir.name,
                })
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.index[idx]
        V  = entry["num_views"]
        V3 = entry["num_views_t3"]

        P1_views, C1_views = [], []
        P2_views, C2_views = [], []
        P3_views, C3_views = [], []
        I1_views, I2_views, I3_views = [], [], []

        align1 = _gps_alignment(entry["t1_dir"])
        align2 = _gps_alignment(entry["t2_dir"])
        align3 = _gps_alignment(entry["t3_dir"])

        for vi in range(V):
            pm1, pc1 = _load_preprocessed(entry["t1_dir"] / "predictions", vi, self.conf_threshold, align1, self.cache_dir, self.vggt_root)
            pm2, pc2 = _load_preprocessed(entry["t2_dir"] / "predictions", vi, self.conf_threshold, align2, self.cache_dir, self.vggt_root)
            P1_views.append(torch.from_numpy(pm1).permute(2, 0, 1))
            C1_views.append(torch.from_numpy(pc1).unsqueeze(0))
            P2_views.append(torch.from_numpy(pm2).permute(2, 0, 1))
            C2_views.append(torch.from_numpy(pc2).unsqueeze(0))
            if self.load_rgb:
                H, W = pm1.shape[:2]
                I1_views.append(torch.from_numpy(_load_rgb_image(entry["t1_dir"], vi, H, W)))
                I2_views.append(torch.from_numpy(_load_rgb_image(entry["t2_dir"], vi, H, W)))

        for vi in range(V3):
            pm3, pc3 = _load_preprocessed(entry["t3_dir"] / "predictions", vi, self.conf_threshold, align3, self.cache_dir, self.vggt_root)
            P3_views.append(torch.from_numpy(pm3).permute(2, 0, 1))
            C3_views.append(torch.from_numpy(pc3).unsqueeze(0))
            if self.load_rgb:
                H, W = pm3.shape[:2]
                I3_views.append(torch.from_numpy(_load_rgb_image(entry["t3_dir"], vi, H, W)))

        sample: dict[str, Any] = {
            "P1":  torch.stack(P1_views),
            "C1":  torch.stack(C1_views),
            "P2":  torch.stack(P2_views),
            "C2":  torch.stack(C2_views),
            "P3":  torch.stack(P3_views),
            "C3":  torch.stack(C3_views),
            "tau": torch.tensor([entry["tau"]], dtype=torch.float32),
            "triplet_id": entry["triplet_id"],
            "variant":    entry["variant"],
        }

        if self.load_rgb:
            sample["I1"] = torch.stack(I1_views)  # [V,  3, H, W]
            sample["I2"] = torch.stack(I2_views)  # [V,  3, H, W]
            sample["I3"] = torch.stack(I3_views)  # [V3, 3, H, W]

        # Depth map for t2 (from depth_map.npy)
        if self.load_depth:
            D2_views = []
            pred_dir = entry["t2_dir"] / "predictions"
            depth_path = pred_dir / "depth_map.npy"
            if depth_path.exists():
                dm = np.load(depth_path, mmap_mode="r")  # [S, H, W] or [S, H, W, 1]
                for vi in range(V):
                    d = dm[vi].astype(np.float32)
                    if d.ndim == 3 and d.shape[-1] == 1:
                        d = d[..., 0]
                    D2_views.append(torch.from_numpy(d).unsqueeze(0))  # [1, H, W]
                sample["D2"] = torch.stack(D2_views)  # [V, 1, H, W]

        # Camera-aware extras — load for all three dates
        cam1 = _load_camera_data(entry["t1_dir"])
        cam2 = _load_camera_data(entry["t2_dir"])
        cam3 = _load_camera_data(entry["t3_dir"])
        if cam1 is not None and cam2 is not None and cam3 is not None:
            poses1, K1 = cam1
            poses2, K2 = cam2
            poses3, K3 = cam3
            sample["T1_c2w"] = torch.from_numpy(poses1)   # [V,  4, 4]
            sample["T2_c2w"] = torch.from_numpy(poses2)   # [V,  4, 4]
            sample["T3_c2w"] = torch.from_numpy(poses3)   # [V3, 4, 4]
            sample["K1"]     = torch.from_numpy(K1)       # [V,  3, 3]
            sample["K2"]     = torch.from_numpy(K2)       # [V,  3, 3]
            sample["K3"]     = torch.from_numpy(K3)       # [V3, 3, 3]

        return sample

