from __future__ import annotations

from pathlib import Path

from .common import (
    load_config,
    merge_cli_paths,
    parse_args,
    read_csv,
    resolve_frame_path,
    safe_json_load,
    write_csv,
    write_json,
)


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    output_root = Path(config["output_root"])
    manifest_path = output_root / "manifests/scene_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing scene manifest. Run scripts/data_prep/extract_scene_metadata.py first.")

    scenes = read_csv(manifest_path)
    validation_rows: list[dict[str, str | int]] = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_root = Path(scene["scene_path"])
        transforms_path = Path(scene["transforms_path"])

        if not transforms_path.exists():
            validation_rows.append(
                {
                    "scene_id": scene_id,
                    "total_frames": 0,
                    "valid_frames": 0,
                    "invalid_frames": 0,
                    "status": "missing_transforms_json",
                }
            )
            continue

        data = safe_json_load(transforms_path)
        frames = data.get("frames", [])

        frame_audit: list[dict[str, str | int | bool]] = []
        valid_frames: list[dict] = []

        for idx, frame in enumerate(frames):
            file_path = str(frame.get("file_path", ""))
            resolved = resolve_frame_path(scene_root, file_path)
            exists = resolved.exists()
            status = "ok" if exists else "missing_file"

            frame_audit.append(
                {
                    "scene_id": scene_id,
                    "frame_index": idx,
                    "file_path": file_path,
                    "resolved_path": str(resolved.resolve()),
                    "exists": exists,
                    "status": status,
                }
            )

            if exists:
                valid_frames.append(frame)

        write_csv(
            output_root / f"frame_audit/{scene_id}.csv",
            frame_audit,
            ["scene_id", "frame_index", "file_path", "resolved_path", "exists", "status"],
        )

        write_json(
            output_root / f"cleaned_frames/{scene_id}.json",
            {
                "scene_id": scene_id,
                "source_transforms": str(transforms_path.resolve()),
                "valid_frame_count": len(valid_frames),
                "dropped_frame_count": len(frames) - len(valid_frames),
                "frames": valid_frames,
            },
        )

        cleaned_transforms = dict(data)
        cleaned_transforms["frames"] = valid_frames
        write_json(
            output_root / f"cleaned_transforms/{scene_id}/transforms_clean.json",
            cleaned_transforms,
        )

        validation_rows.append(
            {
                "scene_id": scene_id,
                "total_frames": len(frames),
                "valid_frames": len(valid_frames),
                "invalid_frames": len(frames) - len(valid_frames),
                "status": "ok" if len(valid_frames) == len(frames) else "has_missing_frames",
            }
        )

    write_csv(
        output_root / "manifests/frame_validation_manifest.csv",
        validation_rows,
        ["scene_id", "total_frames", "valid_frames", "invalid_frames", "status"],
    )
    write_json(output_root / "manifests/frame_validation_manifest.json", validation_rows)


if __name__ == "__main__":
    main()

