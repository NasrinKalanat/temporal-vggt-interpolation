from __future__ import annotations

from pathlib import Path

from .common import (
    ensure_output_tree,
    load_config,
    merge_cli_paths,
    parse_args,
    parse_nadir_filename,
    scene_parts_from_name,
    write_csv,
    write_json,
)


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_root"])
    ensure_output_tree(output_root)

    pix4d_root = dataset_root / "PIX4D_calibrated parameters"
    nadir_root = dataset_root / "Nadir_view_images"

    pix4d_folders = {
        p.name: str(p.resolve())
        for p in sorted(pix4d_root.iterdir())
        if p.is_dir()
    } if pix4d_root.exists() else {}

    nadir_index: dict[tuple[str, str], str] = {}
    if nadir_root.exists():
        for file in sorted(nadir_root.iterdir()):
            if not file.is_file():
                continue
            parsed = parse_nadir_filename(file.name)
            if parsed is None:
                continue
            nadir_index[parsed] = str(file.resolve())

    rows: list[dict[str, str]] = []

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        parts = scene_parts_from_name(child.name)
        if parts is None:
            continue

        pix4d_candidates = [f"{parts.date}_{parts.crop}", parts.date]
        matched_pix4d = ""
        for cand in pix4d_candidates:
            if cand in pix4d_folders:
                matched_pix4d = pix4d_folders[cand]
                break

        row = {
            "scene_id": parts.scene_id,
            "scene_path": str(child.resolve()),
            "date": parts.date,
            "crop": parts.crop,
            "platform": parts.platform,
            "transforms_path": str((child / "transforms.json").resolve()),
            "transforms_test_bak_path": str((child / "transforms_test.bak").resolve()),
            "images_dir": str((child / "images").resolve()),
            "images_2_dir": str((child / "images_2").resolve()),
            "images_4_dir": str((child / "images_4").resolve()),
            "images_8_dir": str((child / "images_8").resolve()),
            "pix4d_path": matched_pix4d,
            "nadir_path": nadir_index.get((parts.crop, parts.date), ""),
        }
        rows.append(row)

    fieldnames = [
        "scene_id",
        "scene_path",
        "date",
        "crop",
        "platform",
        "transforms_path",
        "transforms_test_bak_path",
        "images_dir",
        "images_2_dir",
        "images_4_dir",
        "images_8_dir",
        "pix4d_path",
        "nadir_path",
    ]

    write_csv(output_root / "inventory/scene_inventory.csv", rows, fieldnames)
    write_json(output_root / "inventory/scene_inventory.json", rows)

    write_json(
        output_root / "logs/scan_dataset_summary.json",
        {
            "dataset_root": str(dataset_root.resolve()),
            "n_scenes": len(rows),
            "n_pix4d_projects": len(pix4d_folders),
            "n_nadir_pairs": len(nadir_index),
        },
    )


if __name__ == "__main__":
    main()

