"""Count view triplets and sliding-window variants per t1_t2_t3_crop entry.

Usage:
    python src/count_triplet_variants.py \
        [--triplets prepared_data/camera_consistent_triplets.json] \
        [--n-views 16] [--max-overlap-views 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def n_variants(n_triplets: int, n_views: int, max_overlap_views: int) -> int:
    if n_triplets < n_views:
        return 0
    stride = max(1, n_views - max_overlap_views)
    return (n_triplets - n_views) // stride + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=Path,
                        default=Path("prepared_data/camera_consistent_triplets.json"))
    parser.add_argument("--n-views", type=int, default=16)
    parser.add_argument("--max-overlap-views", type=int, default=4)
    parser.add_argument("--max-variants", type=int, default=None)
    args = parser.parse_args()

    data = read_json(args.triplets)
    n_views = args.n_views
    max_overlap = args.max_overlap_views
    stride = max(1, n_views - max_overlap)

    print(f"File:            {args.triplets}")
    print(f"n_views:         {n_views}")
    print(f"max_overlap:     {max_overlap}  (stride={stride})")
    print(f"max_variants:    {args.max_variants if args.max_variants is not None else 'unlimited'}")
    print(f"Entries:         {len(data)}")
    print()

    # Group by crop
    by_crop: dict[str, list[dict]] = {}
    for entry in data:
        by_crop.setdefault(entry["crop"], []).append(entry)

    total_triplet_entries = 0
    total_view_triplets = 0
    total_variants = 0
    total_skipped = 0

    for crop, entries in sorted(by_crop.items()):
        print(f"{'─'*70}")
        print(f"Crop: {crop}  ({len(entries)} t1_t2_t3 combinations)")
        print(f"{'─'*70}")
        print(f"  {'t1':>8}  {'t2':>8}  {'t3':>8}  {'views':>6}  {'variants':>8}  {'note'}")

        crop_views = 0
        crop_variants = 0
        crop_skipped = 0

        for e in sorted(entries, key=lambda x: (x["t1"], x["t2"], x["t3"])):
            nv = len(e["triplets"])
            nvar = n_variants(nv, n_views, max_overlap)
            if args.max_variants is not None:
                nvar = min(nvar, args.max_variants)
            note = f"(< {n_views} views, skipped)" if nv < n_views else ""
            print(f"  {e['t1']:>8}  {e['t2']:>8}  {e['t3']:>8}  {nv:>6}  {nvar:>8}  {note}")
            crop_views += nv
            crop_variants += nvar
            if nv < n_views:
                crop_skipped += 1

        print(f"  {'SUBTOTAL':>27}  {crop_views:>6}  {crop_variants:>8}")
        if crop_skipped:
            print(f"  ({crop_skipped} entries skipped — fewer than {n_views} views)")
        print()

        total_triplet_entries += len(entries)
        total_view_triplets += crop_views
        total_variants += crop_variants
        total_skipped += crop_skipped

    print(f"{'═'*70}")
    print(f"TOTAL  entries={total_triplet_entries}  view_triplets={total_view_triplets}  "
          f"variants={total_variants}  skipped_entries={total_skipped}")
    print(f"       Total VGGT forward passes = {total_variants * 3}  "
          f"(variants × 3 dates)")


if __name__ == "__main__":
    main()

