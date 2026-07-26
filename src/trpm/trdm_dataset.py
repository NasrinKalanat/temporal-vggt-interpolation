"""Dataset for TRDM depth-map training/evaluation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from trpm.trdm_geometry import load_intrinsics, vggt_extrinsic_to_c2w_np


def parse_triplet_dir(name: str) -> tuple[str, str, str, str] | None:
    parts = name.split("_")
    if len(parts) < 4:
        return None
    crop = parts[-1]
    dates = parts[:-1]
    if len(dates) != 3 or not all(date.isdigit() and len(date) == 8 for date in dates):
        return None
    return dates[0], dates[1], dates[2], crop


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y%m%d")


def date_to_ordinal(date_str: str) -> int:
    return parse_date(date_str).date().toordinal()


def compute_time_feat(t1_date: str, t2_date: str, t3_date: str) -> np.ndarray:
    t1 = parse_date(t1_date)
    t2 = parse_date(t2_date)
    t3 = parse_date(t3_date)
    left_gap = max((t2 - t1).days, 0)
    right_gap = max((t3 - t2).days, 0)
    total_gap = max((t3 - t1).days, 1)
    tau = left_gap / total_gap
    return np.array(
        [
            tau,
            1.0 - tau,
            left_gap / 365.0,
            right_gap / 365.0,
            total_gap / 365.0,
        ],
        dtype=np.float32,
    )


def _load_depth_and_conf(date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred_dir = date_dir / "predictions"
    depth = np.load(pred_dir / "depth_map.npy", mmap_mode="r").astype(np.float32)
    conf = np.load(pred_dir / "depth_confidence.npy", mmap_mode="r").astype(np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    return depth, conf


def _load_c2w_stack(date_dir: Path) -> np.ndarray:
    extrinsics = np.load(date_dir / "predictions" / "extrinsic.npy").astype(np.float32)
    return np.stack([vggt_extrinsic_to_c2w_np(ext) for ext in extrinsics]).astype(np.float32)


class TRDMDepthDataset(Dataset):
    """One item is one target t1/t2 view plus all t3 context depth maps."""

    def __init__(
        self,
        vggt_output_root: Path | str,
        preprocess_mode: str = "pad",
        require_dataset_cameras: bool = True,
    ):
        self.vggt_root = Path(vggt_output_root)
        self.preprocess_mode = preprocess_mode
        self.require_dataset_cameras = require_dataset_cameras
        self.index = self._build_index()

    def _date_ready(self, date_dir: Path) -> bool:
        pred_dir = date_dir / "predictions"
        required = [
            pred_dir / "depth_map.npy",
            pred_dir / "depth_confidence.npy",
            pred_dir / "extrinsic.npy",
        ]
        if self.require_dataset_cameras:
            required.append(date_dir / "dataset_cameras.json")
        return all(path.exists() for path in required)

    def _build_index(self) -> list[dict[str, Any]]:
        index: list[dict[str, Any]] = []
        if not self.vggt_root.exists():
            return index
        for triplet_dir in sorted(self.vggt_root.iterdir()):
            if not triplet_dir.is_dir():
                continue
            parsed = parse_triplet_dir(triplet_dir.name)
            if parsed is None:
                continue
            t1_date, t2_date, t3_date, crop = parsed
            time_feat = compute_time_feat(t1_date, t2_date, t3_date)
            for variant_dir in sorted(triplet_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                t1_dir = variant_dir / "t1"
                t2_dir = variant_dir / "t2"
                t3_dir = variant_dir / "t3"
                if not all(self._date_ready(date_dir) for date_dir in (t1_dir, t2_dir, t3_dir)):
                    continue
                depth_t1 = np.load(t1_dir / "predictions" / "depth_map.npy", mmap_mode="r")
                depth_t2 = np.load(t2_dir / "predictions" / "depth_map.npy", mmap_mode="r")
                num_views = min(depth_t1.shape[0], depth_t2.shape[0])
                for view_idx in range(num_views):
                    index.append(
                        {
                            "triplet_id": triplet_dir.name,
                            "variant": variant_dir.name,
                            "view_idx": view_idx,
                            "crop": crop,
                            "t1_date": t1_date,
                            "t2_date": t2_date,
                            "t3_date": t3_date,
                            "t1_dir": t1_dir,
                            "t2_dir": t2_dir,
                            "t3_dir": t3_dir,
                            "tau": float(time_feat[0]),
                            "time_feat": time_feat,
                        }
                    )
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.index[idx]
        view_idx = entry["view_idx"]
        D1_all, C1_all = _load_depth_and_conf(entry["t1_dir"])
        D2_all, C2_all = _load_depth_and_conf(entry["t2_dir"])
        D3_all, C3_all = _load_depth_and_conf(entry["t3_dir"])
        height, width = D1_all.shape[1], D1_all.shape[2]

        K1_all = load_intrinsics(entry["t1_dir"], height, width, self.preprocess_mode)
        K2_all = load_intrinsics(entry["t2_dir"], height, width, self.preprocess_mode)
        K3_all = load_intrinsics(entry["t3_dir"], height, width, self.preprocess_mode)
        T1_all = _load_c2w_stack(entry["t1_dir"])
        T2_all = _load_c2w_stack(entry["t2_dir"])
        T3_all = _load_c2w_stack(entry["t3_dir"])

        sample: dict[str, Any] = {
            "D1": torch.from_numpy(D1_all[view_idx]).unsqueeze(0),
            "C1": torch.from_numpy(C1_all[view_idx]).unsqueeze(0),
            "D2": torch.from_numpy(D2_all[view_idx]).unsqueeze(0),
            "C2": torch.from_numpy(C2_all[view_idx]).unsqueeze(0),
            "D3": torch.from_numpy(D3_all).unsqueeze(1),
            "C3": torch.from_numpy(C3_all).unsqueeze(1),
            "K1": torch.from_numpy(K1_all[view_idx]),
            "K2": torch.from_numpy(K2_all[view_idx]),
            "K3": torch.from_numpy(K3_all),
            "T1_c2w": torch.from_numpy(T1_all[view_idx]),
            "T2_c2w": torch.from_numpy(T2_all[view_idx]),
            "T3_c2w": torch.from_numpy(T3_all),
            "tau": torch.tensor([entry["tau"]], dtype=torch.float32),
            "time_feat": torch.from_numpy(entry["time_feat"]),
            "metadata": {
                "triplet_id": entry["triplet_id"],
                "variant": entry["variant"],
                "view_idx": view_idx,
                "crop": entry["crop"],
                "t1_date": entry["t1_date"],
                "t2_date": entry["t2_date"],
                "t3_date": entry["t3_date"],
            },
        }
        return sample


def trdm_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out
