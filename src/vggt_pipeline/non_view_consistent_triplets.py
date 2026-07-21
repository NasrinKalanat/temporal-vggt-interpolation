"""Create temporal triplets with independent, non-view-consistent view pools.

For each date triplet t1 < t2 < t3 and crop, this script stores all available
views for t1, t2, and t3. The inference script later builds variants by sliding
independent windows over each date's view list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_pipeline.camera_consistent_triplets import (
    _compact_view,
    _load_views_using_t2_world,
    _parse_offset,
    load_yaml_config,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_manifest": "prepared_data/manifests/dataset_manifest.json",
    "subset_manifest": "prepared_data/subsets/benchmark_subset.json",
    "output_path": "prepared_data/non_view_consistent_triplets.json",
    "selected_dates": [],
    "selected_crops": [],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create triplets with independent t1/t2/t3 view pools."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/non_view_consistent_triplets.yaml"),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(load_yaml_config(args.config))
    return cfg


def load_views_for_date(
    transforms_path: str | Path,
    date_label: str,
    t2_scale: float,
    t2_offset: Any,
) -> list[dict[str, Any]]:
    views = _load_views_using_t2_world(
        transforms_path,
        date_label,
        t2_scale,
        t2_offset,
    )
    return [_compact_view(view) for view in views]


def main(config: dict[str, Any]) -> None:
    selected_dates = sorted(config["selected_dates"])
    dataset = read_json(Path(config["subset_manifest"]))
    scenes = dataset["scenes"]

    by_crop: dict[str, dict[str, Any]] = {crop: {} for crop in config["selected_crops"]}
    for scene in scenes:
        if scene["crop"] in by_crop:
            by_crop[scene["crop"]][scene["date"]] = scene

    all_entries: list[dict[str, Any]] = []

    for crop in config["selected_crops"]:
        by_date = by_crop[crop]
        for i, t1 in enumerate(selected_dates):
            if t1 not in by_date:
                continue
            for j, t2 in enumerate(selected_dates):
                if j <= i or t2 not in by_date:
                    continue

                t2_data = json.loads(Path(by_date[t2]["transforms_path"]).read_text())
                t2_scale = float(t2_data.get("scale", 1.0))
                t2_offset = _parse_offset(t2_data.get("offset", [0.0, 0.0, 0.0]))

                views_t1 = load_views_for_date(
                    by_date[t1]["transforms_path"], t1, t2_scale, t2_offset
                )
                views_t2 = load_views_for_date(
                    by_date[t2]["transforms_path"], t2, t2_scale, t2_offset
                )
                if not views_t1 or not views_t2:
                    continue

                for k, t3 in enumerate(selected_dates):
                    if k <= j or t3 not in by_date:
                        continue

                    views_t3 = load_views_for_date(
                        by_date[t3]["transforms_path"], t3, t2_scale, t2_offset
                    )
                    if not views_t3:
                        continue

                    print(
                        f"{t1}_{t2}_{t3}_{crop}: "
                        f"views t1={len(views_t1)} t2={len(views_t2)} t3={len(views_t3)}"
                    )
                    all_entries.append({
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "crop": crop,
                        "coordinate_reference": "t2_world",
                        "views_t1": views_t1,
                        "views_t2": views_t2,
                        "views_t3": views_t3,
                    })

    for crop in config["selected_crops"]:
        n_entries = sum(1 for entry in all_entries if entry["crop"] == crop)
        print(f"{crop}: {n_entries} (t1,t2,t3) entries")

    write_json(Path(config["output_path"]), all_entries)


if __name__ == "__main__":
    args = parse_args()
    main(build_config(args))
