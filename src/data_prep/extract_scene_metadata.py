from __future__ import annotations

from pathlib import Path

from .common import (
    count_images,
    load_config,
    merge_cli_paths,
    parse_args,
    read_csv,
    safe_json_load,
    write_csv,
    write_json,
)


def main() -> None:
    args = parse_args()
    config = merge_cli_paths(load_config(args.config), args.dataset_root, args.output_root)

    output_root = Path(config["output_root"])
    inventory_path = output_root / "inventory/scene_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("Missing inventory. Run scripts/data_prep/scan_dataset.py first.")

    inventory = read_csv(inventory_path)
    rows: list[dict[str, str | int | float]] = []

    for item in inventory:
        scene_id = item["scene_id"]
        scene_path = Path(item["scene_path"])
        transforms_path = Path(item["transforms_path"])

        notes: list[str] = []
        frame_count = 0
        width = ""
        height = ""
        camera_model = ""
        fl_x = ""
        fl_y = ""
        k1 = ""
        k2 = ""
        k3 = ""
        k4 = ""
        p1 = ""
        p2 = ""

        if transforms_path.exists():
            data = safe_json_load(transforms_path)
            frames = data.get("frames", [])
            frame_count = len(frames)
            width = data.get("w", "")
            height = data.get("h", "")
            camera_model = data.get("camera_model", "")
            fl_x = data.get("fl_x", "")
            fl_y = data.get("fl_y", "")
            k1 = data.get("k1", "")
            k2 = data.get("k2", "")
            k3 = data.get("k3", "")
            k4 = data.get("k4", "")
            p1 = data.get("p1", "")
            p2 = data.get("p2", "")
        else:
            notes.append("missing_transforms_json")

        images_count = count_images(scene_path / "images")
        images_2_count = count_images(scene_path / "images_2")
        images_4_count = count_images(scene_path / "images_4")
        images_8_count = count_images(scene_path / "images_8")

        has_images_2 = (scene_path / "images_2").exists()
        has_images_4 = (scene_path / "images_4").exists()
        has_images_8 = (scene_path / "images_8").exists()

        if not has_images_2:
            notes.append("missing_images_2")
        if not has_images_4:
            notes.append("missing_images_4")
        if not has_images_8:
            notes.append("missing_images_8")
        if frame_count != images_count:
            notes.append(f"frame_count_mismatch:{frame_count}!={images_count}")

        rows.append(
            {
                "scene_id": scene_id,
                "scene_path": item["scene_path"],
                "date": item["date"],
                "crop": item["crop"],
                "platform": item["platform"],
                "transforms_path": item["transforms_path"],
                "frame_count": frame_count,
                "images_count": images_count,
                "images_2_count": images_2_count,
                "images_4_count": images_4_count,
                "images_8_count": images_8_count,
                "images_2_exists": has_images_2,
                "images_4_exists": has_images_4,
                "images_8_exists": has_images_8,
                "width": width,
                "height": height,
                "camera_model": camera_model,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "k4": k4,
                "p1": p1,
                "p2": p2,
                "pix4d_path": item.get("pix4d_path", ""),
                "nadir_path": item.get("nadir_path", ""),
                "notes": "|".join(notes),
            }
        )

    fieldnames = [
        "scene_id",
        "scene_path",
        "date",
        "crop",
        "platform",
        "transforms_path",
        "frame_count",
        "images_count",
        "images_2_count",
        "images_4_count",
        "images_8_count",
        "images_2_exists",
        "images_4_exists",
        "images_8_exists",
        "width",
        "height",
        "camera_model",
        "fl_x",
        "fl_y",
        "k1",
        "k2",
        "k3",
        "k4",
        "p1",
        "p2",
        "pix4d_path",
        "nadir_path",
        "notes",
    ]

    write_csv(output_root / "manifests/scene_manifest.csv", rows, fieldnames)
    write_json(output_root / "manifests/scene_manifest.json", rows)


if __name__ == "__main__":
    main()

