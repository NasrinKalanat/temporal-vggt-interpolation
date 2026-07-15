"""Build a triplets JSON from completed VGGT inference variants in vggt_output/.

Scans vggt_output/ for variants with predictions/point_map.npy, determines which
(crop, date, batch_idx) combinations are available, then generates one triplet entry
per (left_date, middle_date, right_date, view_batch) where all three dates have that
batch index completed.

Unlike build_expanded_triplets.py (which computes expected counts from frame counts),
this script reads what actually exists on disk — run it after vggt inference.

Usage:
    python src/data_prep/build_available_triplets.py \
        --vggt-root vggt_output \
        --triplets prepared_data/subsets/benchmark_triplets.json \
        --output prepared_data/subsets/available_triplets.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import itertools
import random


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_scene_crop_date(scene_id: str) -> tuple[str | None, str | None]:
    parts = scene_id.split("_")
    if len(parts) < 3:
        return None, None
    crop = parts[-1]
    date = "".join(filter(str.isdigit, parts[-2]))
    if crop not in ("corn", "soybean") or len(date) != 8:
        return None, None
    return crop, date


def scan_available_variants(vggt_root: Path) -> dict[tuple[str, str], list[int]]:
    """Return {(crop, date): sorted list of completed batch indices}."""
    available: dict[tuple[str, str], list[int]] = defaultdict(list)

    if not vggt_root.exists():
        return available

    for scene_dir in sorted(vggt_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        crop, date = parse_scene_crop_date(scene_dir.name)
        if crop is None:
            continue
        for variant_dir in sorted(scene_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            # Only handle views_NN variants
            m = re.fullmatch(r"views_(\d+)", variant_dir.name)
            if m is None:
                continue
            if (variant_dir / "predictions" / "point_map.npy").exists():
                available[(crop, date)].append(int(m.group(1)))

    for key in available:
        random.shuffle(available[key])
    return available


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build triplets JSON from completed vggt_output variants.")
    p.add_argument("--vggt-root", type=Path, default=Path("vggt_output"))
    p.add_argument("--triplets", type=Path, default=Path("prepared_data/subsets/benchmark_triplets.json"))
    p.add_argument("--output", type=Path, default=Path("prepared_data/subsets/available_triplets.json"))
    p.add_argument("--limit", type=int, default=0, help="Limit number of variation for each triplet, 0 for no limit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    available = scan_available_variants(args.vggt_root)

    if not available:
        print(f"No completed variants found under {args.vggt_root}")
        print("Run vggt inference first: python src/vggt_pipeline/run_vggt_inference.py")
        return

    print(f"Found completed variants for {len(available)} (crop, date) pairs:")
    for (crop, date), batches in sorted(available.items()):
        print(f"  {crop} {date}: {len(batches)} batches")

    date_triplets: list[dict[str, Any]] = read_json(args.triplets).get("triplets", [])

    expanded: list[dict[str, Any]] = []
    skipped = 0
    for t in date_triplets:
        crop = t["crop"]
        b1 = available.get((crop, t["left_date"]), [])
        b2 = available.get((crop, t["middle_date"]), [])
        b3 = available.get((crop, t["right_date"]), [])
        if not b1 or not b2 or not b3:
            skipped += 1
            continue
        triplet_combination = list(itertools.islice(itertools.product(b1, b2, b3), 0, args.limit if args.limit > 0 else None))
        print(f"total possible combinations: {len(b1) * len(b2) * len(b3)}")
        limit_statement = f"Limit: {args.limit}" if args.limit > 0 else ""
        print(f"selected combinations: {len(triplet_combination)} {limit_statement}")
        for batch_idx1, batch_idx2, batch_idx3 in triplet_combination:
            entry = {}
            entry["view_batch_t1"] = batch_idx1
            entry["view_batch_t2"] = batch_idx2
            entry["view_batch_t3"] = batch_idx3
            expanded.append(entry)
        t['variations'] = expanded

    n_adjacent = sum(1 for t in expanded if t.get("is_adjacent"))
    n_multigap = len(expanded) - n_adjacent
    by_crop: dict[str, int] = {}
    for t in date_triplets:
        by_crop[t["crop"]] = by_crop.get(t["crop"], 0) + len(t['variations'])
    for crop, count in sorted(by_crop.items()):
        print(f"  {crop}: {count}")

    num_triplets = sum(by_crop.values())
    output = {
        "n_triplets": num_triplets,
        "n_adjacent": n_adjacent,
        "n_multigap": n_multigap,
        "by_crop": by_crop,
        "vggt_root": str(args.vggt_root),
        "triplets": date_triplets,
    }
    print("Writing ")
    write_json(args.output, output)
    print(f"\nwrote {num_triplets} triplets to {args.output}  (skipped {skipped} date-triplets with no shared batches)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()

