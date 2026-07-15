from __future__ import annotations

from pathlib import Path

from .common import (
    load_config,
    merge_cli_paths,
    parse_args,
    parse_nadir_filename,
    read_csv,
    tiff_size,
    write_csv,
    write_json,
)


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_root"])

    scene_manifest_path = output_root / "manifests/scene_manifest.csv"
    if not scene_manifest_path.exists():
        raise FileNotFoundError("Missing scene manifest. Run scripts/data_prep/extract_scene_metadata.py first.")

    scenes = read_csv(scene_manifest_path)

    nadir_root = dataset_root / "Nadir_view_images"
    nadir_index: dict[tuple[str, str], Path] = {}
    if nadir_root.exists():
        for p in nadir_root.iterdir():
            if not p.is_file():
                continue
            parsed = parse_nadir_filename(p.name)
            if parsed is None:
                continue
            nadir_index[parsed] = p

    rows: list[dict[str, str | int | bool]] = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        crop = scene["crop"]
        date = scene["date"]

        matched = nadir_index.get((crop, date))
        width = None
        height = None
        if matched is not None:
            width, height = tiff_size(matched)

        rows.append(
            {
                "scene_id": scene_id,
                "date": date,
                "crop": crop,
                "nadir_found": matched is not None,
                "nadir_path": str(matched.resolve()) if matched else "",
                "nadir_width": width if width is not None else "",
                "nadir_height": height if height is not None else "",
            }
        )

    fieldnames = [
        "scene_id",
        "date",
        "crop",
        "nadir_found",
        "nadir_path",
        "nadir_width",
        "nadir_height",
    ]

    write_csv(output_root / "manifests/nadir_manifest.csv", rows, fieldnames)
    write_json(output_root / "manifests/nadir_manifest.json", rows)


if __name__ == "__main__":
    main()

