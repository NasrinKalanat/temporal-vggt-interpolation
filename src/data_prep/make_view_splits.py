from __future__ import annotations

import json
from pathlib import Path

from .common import (
    load_config,
    merge_cli_paths,
    parse_args,
    stable_shuffle,
    write_json,
)


def split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n > 0 and n_train == 0:
        n_train = 1
    if n - n_train > 1 and n_val == 0:
        n_val = 1
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return n_train, n_val


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    output_root = Path(config["output_root"])
    subset_path = output_root / "subsets/benchmark_subset.json"
    if not subset_path.exists():
        raise FileNotFoundError("Missing benchmark subset. Run scripts/data_prep/build_benchmark_subset.py first.")

    subset = json.loads(subset_path.read_text())
    scenes = subset.get("scenes", [])

    ratios = config["split_ratios"]
    train_ratio = float(ratios.get("train", 0.7))
    val_ratio = float(ratios.get("val", 0.15))

    seed = int(config["random_seed"])
    summary: list[dict[str, str | int]] = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        cleaned_frames_path = output_root / f"cleaned_frames/{scene_id}.json"
        if not cleaned_frames_path.exists():
            continue

        payload = json.loads(cleaned_frames_path.read_text())
        frames = payload.get("frames", [])

        indexed = [{"frame_index": i, "file_path": f.get("file_path", "")} for i, f in enumerate(frames)]
        shuffled = stable_shuffle(indexed, seed + sum(ord(c) for c in scene_id))

        n_train, n_val = split_indices(len(shuffled), train_ratio, val_ratio)
        train_frames = shuffled[:n_train]
        val_frames = shuffled[n_train : n_train + n_val]
        test_frames = shuffled[n_train + n_val :]

        split_payload = {
            "scene_id": scene_id,
            "seed": seed,
            "ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": max(0.0, 1.0 - train_ratio - val_ratio),
            },
            "n_total": len(shuffled),
            "train_frames": train_frames,
            "val_frames": val_frames,
            "test_frames": test_frames,
        }
        write_json(output_root / f"splits/{scene_id}.json", split_payload)

        summary.append(
            {
                "scene_id": scene_id,
                "n_total": len(shuffled),
                "n_train": len(train_frames),
                "n_val": len(val_frames),
                "n_test": len(test_frames),
            }
        )

    write_json(output_root / "manifests/view_split_summary.json", summary)


if __name__ == "__main__":
    main()

