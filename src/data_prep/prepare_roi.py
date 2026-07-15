from __future__ import annotations

import json
from pathlib import Path

from .common import load_config, merge_cli_paths, parse_args, read_csv, write_json


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    output_root = Path(config["output_root"])
    if not bool(config.get("build_roi", True)):
        write_json(output_root / "roi/roi_manifest.json", {"build_roi": False, "entries": []})
        return

    subset_path = output_root / "subsets/benchmark_subset.json"
    nadir_manifest_path = output_root / "manifests/nadir_manifest.csv"
    if not subset_path.exists() or not nadir_manifest_path.exists():
        raise FileNotFoundError("Missing subset or nadir manifest. Run subset and nadir scripts first.")

    subset = json.loads(subset_path.read_text())
    subset_scene_ids = {x["scene_id"] for x in subset.get("scenes", [])}
    nadir_rows = {row["scene_id"]: row for row in read_csv(nadir_manifest_path)}

    roi_entries: list[dict[str, object]] = []

    for scene_id in sorted(subset_scene_ids):
        nadir = nadir_rows.get(scene_id)
        has_nadir = nadir is not None and str(nadir.get("nadir_found", "")).lower() in {"true", "1"}

        width = None
        height = None
        if has_nadir:
            try:
                width = int(nadir.get("nadir_width", "") or 0)
                height = int(nadir.get("nadir_height", "") or 0)
            except ValueError:
                width = None
                height = None

        roi = None
        if width and height:
            roi = {
                "type": "pixel_bbox",
                "x0": 0,
                "y0": 0,
                "x1": width - 1,
                "y1": height - 1,
            }

        roi_entries.append(
            {
                "scene_id": scene_id,
                "has_nadir": has_nadir,
                "nadir_path": nadir.get("nadir_path", "") if nadir else "",
                "roi": roi,
                "mask_available": False,
                "notes": "default_full_image_roi" if roi else "no_nadir_roi",
            }
        )

    write_json(
        output_root / "roi/roi_manifest.json",
        {
            "build_roi": True,
            "n_scenes": len(roi_entries),
            "entries": roi_entries,
        },
    )


if __name__ == "__main__":
    main()

