from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path

from .common import (
    load_config,
    merge_cli_paths,
    parse_args,
    read_csv,
    write_json,
)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _days_between(a: str, b: str) -> int:
    return (_parse_date(b) - _parse_date(a)).days


def _compute_tau(left: str, middle: str, right: str) -> float:
    span = _days_between(left, right)
    return _days_between(left, middle) / span if span > 0 else 0.5


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    output_root = Path(config["output_root"])
    scene_manifest_path = output_root / "manifests/scene_manifest.csv"
    if not scene_manifest_path.exists():
        raise FileNotFoundError("Missing scene manifest. Run src/data_prep/extract_scene_metadata.py first.")

    scene_rows = read_csv(scene_manifest_path)

    selected_dates = set(config["selected_dates"])
    selected_crops = set(config["selected_crops"])
    selected_platforms = set(config.get("selected_platforms", ["matic"]))

    selected_scenes = [
        row
        for row in scene_rows
        if row["date"] in selected_dates
        and row["crop"] in selected_crops
        and row["platform"] in selected_platforms
    ]

    selected_scenes = sorted(selected_scenes, key=lambda r: (r["crop"], r["date"], r["platform"], r["scene_id"]))

    by_crop: dict[str, list[dict[str, str]]] = {}
    for row in selected_scenes:
        by_crop.setdefault(row["crop"], []).append(
            {
                "scene_id": row["scene_id"],
                "date": row["date"],
                "platform": row["platform"],
                "scene_path": row["scene_path"],
            }
        )

    triplets: list[dict] = []
    for crop in sorted(by_crop):
        date_to_sensor: dict[str, str] = {r["date"]: r["platform"] for r in by_crop[crop]}
        dates = sorted(date_to_sensor)

        # Generate all valid triplets: all (i, j, k) with i < j < k and same platform
        for left, middle, right in combinations(dates, 3):
            ls = date_to_sensor[left]
            ms = date_to_sensor[middle]
            rs = date_to_sensor[right]
            if ls != ms or ms != rs:
                continue

            sorted_dates = dates
            li, mi, ri = sorted_dates.index(left), sorted_dates.index(middle), sorted_dates.index(right)
            is_adjacent = (mi == li + 1) and (ri == mi + 1)

            left_gap = _days_between(left, middle)
            right_gap = _days_between(middle, right)
            total_gap = _days_between(left, right)
            tau = _compute_tau(left, middle, right)

            triplets.append(
                {
                    "crop": crop,
                    "left_date": left,
                    "middle_date": middle,
                    "right_date": right,
                    "left_sensor": ls,
                    "middle_sensor": ms,
                    "right_sensor": rs,
                    "sensor_consistent": True,
                    "is_adjacent": is_adjacent,
                    "tau": round(tau, 4),
                    "left_gap_days": left_gap,
                    "right_gap_days": right_gap,
                    "total_gap_days": total_gap,
                    "pattern": f"{left} -> {middle} <- {right}",
                }
            )

    write_json(
        output_root / "subsets/benchmark_subset.json",
        {
            "selected_dates": sorted(selected_dates),
            "selected_crops": sorted(selected_crops),
            "selected_platforms": sorted(selected_platforms),
            "n_scenes": len(selected_scenes),
            "scenes": selected_scenes,
            "scenes_by_crop": by_crop,
        },
    )

    write_json(
        output_root / "subsets/benchmark_triplets.json",
        {
            "n_triplets": len(triplets),
            "n_adjacent": sum(1 for t in triplets if t["is_adjacent"]),
            "n_multigap": sum(1 for t in triplets if not t["is_adjacent"]),
            "triplets": triplets,
        },
    )


if __name__ == "__main__":
    main()

