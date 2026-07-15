"""Find camera-consistent t1-t2 view pairs; t3 views are unconstrained.

For each (t1, t2, t3) date combination:
  - Match t1 and t2 views by camera position + viewing direction (same as
    camera_consistent_triplets.py, but only for the t1-t2 pair).
  - Load all t3 views without any camera-consistency constraint.

Output JSON structure:
  [
    {
      "t1": "20230812", "t2": "20230817", "t3": "20230822", "crop": "corn",
      "pairs": [
        {
          "v1": { ...view fields... },
          "v2": { ...view fields... },
          "metrics": { "position_distance_m": ..., ... },
          "score": 0.12
        },
        ...
      ],
      "views_t3": [ { ...view fields... }, ... ]
    },
    ...
  ]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_consistent_triplets import (
    _build_pair_candidates,
    _compact_view,
    _load_views_using_t2_world,
    _parse_offset,
    load_yaml_config,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_manifest": "prepared_data/manifests/dataset_manifest.json",
    "subset_manifest": "prepared_data/subsets/benchmark_subset.json",
    "output_path": "prepared_data/t1t2_paired_triplets.json",
    "selected_dates": [],
    "selected_crops": [],
    "max_position_distance_m": 0.1,
    "max_view_angle_deg": 3.0,
    "max_tilt_difference_deg": 3.0,
    "max_oblique_yaw_difference_deg": 5.0,
    "use_xy_position_only": False,
    "one_to_one": True,
    "max_pairs": None,
}


def find_t1t2_pairs(
    t1_transforms_path,
    t2_transforms_path,
    date_labels=("t1", "t2"),
    max_position_distance_m=0.1,
    max_view_angle_deg=3.0,
    max_tilt_difference_deg=3.0,
    max_oblique_yaw_difference_deg=5.0,
    use_xy_position_only=False,
    one_to_one=True,
    max_pairs=None,
):
    """Find camera-consistent view pairs between t1 and t2.

    Uses t2 as the reference coordinate system (same convention as
    camera_consistent_triplets.py).

    Returns list of dicts, each with v1, v2, metrics, score.
    """
    t2_data = json.loads(Path(t2_transforms_path).read_text())
    t2_scale = float(t2_data.get("scale", 1.0))
    t2_offset = _parse_offset(t2_data.get("offset", [0.0, 0.0, 0.0]))

    views1 = _load_views_using_t2_world(t1_transforms_path, date_labels[0], t2_scale, t2_offset)
    views2 = _load_views_using_t2_world(t2_transforms_path, date_labels[1], t2_scale, t2_offset)

    raw_pairs = _build_pair_candidates(
        views1, views2,
        max_position_distance_m=max_position_distance_m,
        max_view_angle_deg=max_view_angle_deg,
        max_tilt_difference_deg=max_tilt_difference_deg,
        max_oblique_yaw_difference_deg=max_oblique_yaw_difference_deg,
        use_xy_position_only=use_xy_position_only,
    )

    # Score = position + angle cost (same weights as triplet scorer, but for one pair)
    candidates = []
    for p in raw_pairs:
        v1 = views1[p["index_a"]]
        v2 = views2[p["index_b"]]
        m = p["metrics"]
        score = float(
            1.00 * m["position_distance_m"]
            + 0.35 * m["view_angle_difference_deg"]
            + 0.15 * m["tilt_difference_deg"]
            + 0.10 * m["yaw_difference_deg"]
        )
        candidates.append({"v1": v1, "v2": v2, "metrics": m, "score": score})

    candidates.sort(key=lambda x: x["score"])

    if not one_to_one:
        result = candidates[:max_pairs] if max_pairs else candidates
    else:
        selected, used1, used2 = [], set(), set()
        for c in candidates:
            k1 = c["v1"]["frame_index"]
            k2 = c["v2"]["frame_index"]
            if k1 in used1 or k2 in used2:
                continue
            selected.append(c)
            used1.add(k1)
            used2.add(k2)
            if max_pairs and len(selected) >= max_pairs:
                break
        result = selected

    # Compact views and drop frame_index
    out = []
    for c in result:
        v1 = dict(_compact_view(c["v1"]))
        v2 = dict(_compact_view(c["v2"]))
        out.append({"v1": v1, "v2": v2, "metrics": c["metrics"], "score": c["score"]})
    return out


def load_t3_views(t3_transforms_path, date_label, t2_scale, t2_offset):
    """Load all t3 views in t2 world coordinates, compacted."""
    views = _load_views_using_t2_world(t3_transforms_path, date_label, t2_scale, t2_offset)
    return [_compact_view(v) for v in views]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find camera-consistent t1-t2 pairs; t3 views are free."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/t1t2_paired_triplets.yaml"))
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(DEFAULT_CONFIG)
    cfg.update(load_yaml_config(args.config))
    return cfg


def main(config: dict[str, Any]):
    selected_dates = sorted(config["selected_dates"])
    dataset = read_json(Path(config["subset_manifest"]))
    scenes = dataset["scenes"]

    by_crop: dict[str, dict] = {crop: {} for crop in config["selected_crops"]}
    for scene in scenes:
        if scene["crop"] in by_crop:
            by_crop[scene["crop"]][scene["date"]] = scene

    all_entries = []

    for crop in config["selected_crops"]:
        by_date = by_crop[crop]
        for i, t1 in enumerate(selected_dates):
            if t1 not in by_date:
                continue
            for j, t2 in enumerate(selected_dates):
                if j <= i or t2 not in by_date:
                    continue
                # Resolve t2 coordinate reference once per (t1, t2) pair
                t2_data = json.loads(Path(by_date[t2]["transforms_path"]).read_text())
                t2_scale = float(t2_data.get("scale", 1.0))
                t2_offset = _parse_offset(t2_data.get("offset", [0.0, 0.0, 0.0]))

                pairs = find_t1t2_pairs(
                    t1_transforms_path=by_date[t1]["transforms_path"],
                    t2_transforms_path=by_date[t2]["transforms_path"],
                    date_labels=(t1, t2),
                    max_position_distance_m=config["max_position_distance_m"],
                    max_view_angle_deg=config["max_view_angle_deg"],
                    max_tilt_difference_deg=config["max_tilt_difference_deg"],
                    max_oblique_yaw_difference_deg=config["max_oblique_yaw_difference_deg"],
                    use_xy_position_only=config["use_xy_position_only"],
                    one_to_one=config["one_to_one"],
                    max_pairs=config["max_pairs"],
                )

                if not pairs:
                    continue

                for k, t3 in enumerate(selected_dates):
                    if k <= j or t3 not in by_date:
                        continue

                    views_t3 = load_t3_views(
                        by_date[t3]["transforms_path"],
                        date_label=t3,
                        t2_scale=t2_scale,
                        t2_offset=t2_offset,
                    )

                    print(f"{t1}_{t2}_{t3}_{crop}: {len(pairs)} pairs, {len(views_t3)} free t3 views")
                    all_entries.append({
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "crop": crop,
                        "pairs": pairs,
                        "views_t3": views_t3,
                    })

    for crop in config["selected_crops"]:
        n = sum(1 for e in all_entries if e["crop"] == crop)
        print(f"{crop}: {n} (t1,t2,t3) entries")

    write_json(Path(config["output_path"]), all_entries)


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    main(config)

