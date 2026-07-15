"""Build expanded triplets JSON with multiple view batches per scene.

For each (t1, t2, t3) date-triplet, generates one entry per valid view_batch index
using the same batch index across all three dates. The number of batches per scene
is derived from the cleaned_frames valid_frame_count and the sliding-window formula:
    n_batches = 1 + (n_frames - n_views) // stride
    stride = n_views - max_overlap_views

Output triplets carry a view_batch field that tells the dataset which views_NN/ variant
to load for all three dates.

Usage:
    python src/data_prep/build_expanded_triplets.py \
        --triplets prepared_data/subsets/benchmark_triplets.json \
        --cleaned-frames prepared_data/cleaned_frames \
        --inference-config configs/vggt_inference.yaml \
        --output prepared_data/subsets/expanded_triplets.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required.") from exc
    return yaml.safe_load(path.read_text()) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build expanded triplets with multiple view batches.")
    p.add_argument("--triplets", type=Path, default=Path("prepared_data/subsets/benchmark_triplets.json"))
    p.add_argument("--cleaned-frames", type=Path, default=Path("prepared_data/cleaned_frames"))
    p.add_argument("--inference-config", type=Path, default=Path("configs/vggt_inference.yaml"))
    p.add_argument("--output", type=Path, default=Path("prepared_data/subsets/expanded_triplets.json"))
    p.add_argument("--n-views", type=int, default=None, help="Override n_views from inference config")
    p.add_argument("--max-overlap-views", type=int, default=None, help="Override max_overlap_views from inference config")
    return p.parse_args()


def n_view_batches(n_frames: int, n_views: int, max_overlap_views: int) -> int:
    if n_frames < n_views:
        return 0
    stride = max(1, n_views - max_overlap_views)
    return 1 + (n_frames - n_views) // stride


def main() -> None:
    args = parse_args()

    # Load n_views and max_overlap_views from inference config, allow CLI override.
    inf_cfg = load_yaml(args.inference_config) if args.inference_config.exists() else {}
    n_views = args.n_views if args.n_views is not None else int(inf_cfg.get("n_views", 32))
    max_overlap_views = (
        args.max_overlap_views if args.max_overlap_views is not None
        else int(inf_cfg.get("max_overlap_views", 0))
    )
    stride = max(1, n_views - max_overlap_views)
    print(f"n_views={n_views}  max_overlap_views={max_overlap_views}  stride={stride}")

    # Load frame counts per (crop, date) from cleaned_frames JSONs.
    frames_per_scene: dict[tuple[str, str], int] = {}
    for p in sorted(args.cleaned_frames.glob("nerfstudio_matic*.json")):
        data = read_json(p)
        n = data["valid_frame_count"]
        parts = p.stem.split("_")
        crop = parts[-1]
        date = parts[-2][-8:]
        frames_per_scene[(crop, date)] = n

    print(f"loaded frame counts for {len(frames_per_scene)} scenes")

    # Load original date-triplets.
    raw = read_json(args.triplets)
    date_triplets: list[dict[str, Any]] = raw.get("triplets", [])

    # Expand: generate max(k1, k2, k3) triplets per date-triplet; shorter lists cycle with modulo.
    expanded: list[dict[str, Any]] = []
    skipped = 0
    for t in date_triplets:
        crop = t["crop"]
        k1 = n_view_batches(frames_per_scene.get((crop, t["left_date"]), 0), n_views, max_overlap_views)
        k2 = n_view_batches(frames_per_scene.get((crop, t["middle_date"]), 0), n_views, max_overlap_views)
        k3 = n_view_batches(frames_per_scene.get((crop, t["right_date"]), 0), n_views, max_overlap_views)
        if k1 == 0 or k2 == 0 or k3 == 0:
            skipped += 1
            continue
        for i in range(max(k1, k2, k3)):
            entry = dict(t)
            entry["view_batch_t1"] = i % k1
            entry["view_batch_t2"] = i % k2
            entry["view_batch_t3"] = i % k3
            expanded.append(entry)

    n_adjacent = sum(1 for t in expanded if t.get("is_adjacent"))
    n_multigap = sum(1 for t in expanded if not t.get("is_adjacent"))
    by_crop = {}
    for t in expanded:
        by_crop.setdefault(t["crop"], 0)
        by_crop[t["crop"]] += 1

    output = {
        "n_triplets": len(expanded),
        "n_adjacent": n_adjacent,
        "n_multigap": n_multigap,
        "by_crop": by_crop,
        "n_views": n_views,
        "max_overlap_views": max_overlap_views,
        "stride": stride,
        "triplets": expanded,
    }
    write_json(args.output, output)
    print(f"wrote {len(expanded)} triplets to {args.output}  (skipped {skipped} date-triplets with insufficient frames)")
    for crop, count in sorted(by_crop.items()):
        print(f"  {crop}: {count}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()

