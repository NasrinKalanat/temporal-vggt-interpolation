"""Persistent disk cache for frozen VGGT intermediate features.

Cache key derivation (no file I/O):
  date_dir layout: {root}/{subset}/{t1}_{t2}_{t3}_{crop}/variant_{t2v}_{t3v}/{t1|t3}/

  For t1: key = "{t1_date}_{crop}_pair{t2v}"
    — t1 images are determined by the pair index (t2v), not the t3 view index.
      variant_03_02/t1 and variant_03_01/t1 hold IDENTICAL t1 images.

  For t3: key = "{t3_date}_{crop}_t3v{t3v}"
    — t3 images are determined solely by the t3 view index.

Cache file = {cache_root}/{key}.pt
Features   = list of 4 CPU bfloat16 tensors, one per cache layer.

Writes are atomic (temp file + os.replace) so multiple training processes
can generate and read the cache concurrently without corruption.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


class VGGTFeatureCache:

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ── key ───────────────────────────────────────────────────────────────────

    def key(self, date_dir: Path) -> Optional[str]:
        """Derive a deduplicated cache key from the directory path hierarchy.

        Handles two variant naming conventions:

        t1t2_paired:  variant_{t2v}_{t3v}/{t1|t3}
          t1 key = "{t1_date}_{crop}_pair{t2v:02d}"   (same t2v → same t1 images)
          t3 key = "{t3_date}_{crop}_t3v{t3v:02d}"

        camera_consistent:  variant_{n}/{t1|t3}
          t1 key = "{t1_date}_{crop}_v{n:03d}_t1"     (per-variant, no dedup)
          t3 key = "{t3_date}_{crop}_v{n:03d}_t3"

        Returns None if the path does not match the expected layout.
        """
        endpoint = date_dir.name
        if endpoint not in ("t1", "t2", "t3"):
            return None

        variant = date_dir.parent.name
        triplet = date_dir.parent.parent.name

        vparts = variant.split("_")
        if not vparts or vparts[0] != "variant":
            return None
        try:
            indices = [int(p) for p in vparts[1:]]
        except ValueError:
            return None
        if not indices:
            return None

        tparts = triplet.split("_")
        if len(tparts) < 4:
            return None
        crop  = tparts[-1]
        dates = tparts[:-1]
        if len(dates) != 3 or not all(d.isdigit() and len(d) == 8 for d in dates):
            return None
        t1_date, _, t3_date = dates

        t2_date = dates[1]

        if len(indices) == 2:
            # t1t2_paired: variant_{t2v}_{t3v}
            t2v, t3v = indices
            if endpoint == "t1":
                return f"{t1_date}_{crop}_pair{t2v:02d}"
            elif endpoint == "t2":
                return f"{t2_date}_{crop}_t2_pair{t2v:02d}"
            return f"{t3_date}_{crop}_t3v{t3v:02d}"
        else:
            # camera_consistent: variant_{n}
            v = indices[0]
            if endpoint == "t2":
                return f"{t2_date}_{crop}_v{v:03d}_t2"
            date = t1_date if endpoint == "t1" else t3_date
            return f"{date}_{crop}_v{v:03d}_{endpoint}"

    # ── path ──────────────────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        return self.cache_root / f"{key}.pt"

    def exists(self, key: Optional[str]) -> bool:
        return key is not None and self._path(key).exists()

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, key: Optional[str]) -> Optional[list[torch.Tensor]]:
        """Return list of 4 CPU bf16 tensors, or None on miss / corrupt file."""
        if key is None:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            return None

    # ── write ─────────────────────────────────────────────────────────────────

    def put(self, key: Optional[str], features: list[torch.Tensor]) -> None:
        """Atomically write features. No-op if key is None or file already exists."""
        if key is None:
            return
        p = self._path(key)
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        cpu_feats = [f.detach().cpu().to(torch.bfloat16) for f in features]
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        torch.save(cpu_feats, tmp)
        try:
            os.replace(tmp, p)   # atomic on POSIX; overwrites safely if two processes race
        except Exception:
            try:
                tmp.unlink()
            except Exception:
                pass

