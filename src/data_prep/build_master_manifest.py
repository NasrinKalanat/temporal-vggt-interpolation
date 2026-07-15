from __future__ import annotations

import json
from pathlib import Path

from .common import read_csv, write_csv, write_json, parse_args, load_config, merge_cli_paths


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def as_bool(v: str | bool | None) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)
    output_root = Path(config["output_root"])

    scene_manifest = read_csv(output_root / "manifests/scene_manifest.csv")
    validation_rows = {r["scene_id"]: r for r in read_csv(output_root / "manifests/frame_validation_manifest.csv")}
    pix4d_rows = {r["scene_id"]: r for r in read_csv(output_root / "manifests/pix4d_manifest.csv")}
    nadir_rows = {r["scene_id"]: r for r in read_csv(output_root / "manifests/nadir_manifest.csv")}

    subset = read_json(output_root / "subsets/benchmark_subset.json")
    subset_ids = {s["scene_id"] for s in subset.get("scenes", [])}

    split_summary = {
        r["scene_id"]: r for r in read_json(output_root / "manifests/view_split_summary.json")
    } if (output_root / "manifests/view_split_summary.json").exists() else {}

    roi_manifest = read_json(output_root / "roi/roi_manifest.json")
    roi_entries = {e["scene_id"]: e for e in roi_manifest.get("entries", [])} if isinstance(roi_manifest, dict) else {}

    rows: list[dict[str, object]] = []

    for scene in scene_manifest:
        scene_id = scene["scene_id"]
        val = validation_rows.get(scene_id, {})
        pix = pix4d_rows.get(scene_id, {})
        nadir = nadir_rows.get(scene_id, {})
        split = split_summary.get(scene_id, {})
        roi = roi_entries.get(scene_id, {})

        rows.append(
            {
                "scene_id": scene_id,
                "scene_path": scene["scene_path"],
                "date": scene["date"],
                "crop": scene["crop"],
                "platform": scene["platform"],
                "transforms_path": scene["transforms_path"],
                "clean_transforms_path": str((output_root / f"cleaned_transforms/{scene_id}/transforms_clean.json").resolve()),
                "frame_count": scene["frame_count"],
                "valid_frames": val.get("valid_frames", ""),
                "invalid_frames": val.get("invalid_frames", ""),
                "frame_validation_status": val.get("status", ""),
                "pix4d_found": as_bool(pix.get("pix4d_found")),
                "pix4d_project_path": pix.get("pix4d_project_path", ""),
                "nadir_found": as_bool(nadir.get("nadir_found")),
                "nadir_path": nadir.get("nadir_path", ""),
                "in_benchmark_subset": scene_id in subset_ids,
                "split_path": str((output_root / f"splits/{scene_id}.json").resolve()) if scene_id in split_summary else "",
                "split_n_train": split.get("n_train", ""),
                "split_n_val": split.get("n_val", ""),
                "split_n_test": split.get("n_test", ""),
                "roi_available": bool(roi),
                "roi_notes": roi.get("notes", "") if roi else "",
                "notes": scene.get("notes", ""),
            }
        )

    fieldnames = [
        "scene_id",
        "scene_path",
        "date",
        "crop",
        "platform",
        "transforms_path",
        "clean_transforms_path",
        "frame_count",
        "valid_frames",
        "invalid_frames",
        "frame_validation_status",
        "pix4d_found",
        "pix4d_project_path",
        "nadir_found",
        "nadir_path",
        "in_benchmark_subset",
        "split_path",
        "split_n_train",
        "split_n_val",
        "split_n_test",
        "roi_available",
        "roi_notes",
        "notes",
    ]

    write_csv(output_root / "manifests/dataset_manifest.csv", rows, fieldnames)
    write_json(output_root / "manifests/dataset_manifest.json", rows)


if __name__ == "__main__":
    main()

