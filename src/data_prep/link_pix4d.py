from __future__ import annotations

from pathlib import Path

from .common import (
    load_config,
    merge_cli_paths,
    parse_args,
    read_csv,
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

    pix4d_root = dataset_root / "PIX4D_calibrated parameters"
    project_paths = {p.name: p for p in pix4d_root.iterdir() if p.is_dir()} if pix4d_root.exists() else {}

    rows: list[dict[str, str | int | bool]] = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        date = scene["date"]
        crop = scene["crop"]

        candidates = [f"{date}_{crop}", date]
        match_name = ""
        match_path = None

        for cand in candidates:
            if cand in project_paths:
                match_name = cand
                match_path = project_paths[cand]
                break

        has_match = match_path is not None

        input_cameras = ""
        calibrated_cameras = ""
        external_txt = ""
        n_json_files = 0
        n_txt_files = 0

        if match_path is not None:
            input_candidate = match_path / "input_cameras.json"
            calibrated_candidate = match_path / "calibrated_cameras.json"
            if input_candidate.exists():
                input_cameras = str(input_candidate.resolve())
            if calibrated_candidate.exists():
                calibrated_cameras = str(calibrated_candidate.resolve())

            txt_candidates = sorted(match_path.glob("*calibrated_external_camera_parameters.txt"))
            if txt_candidates:
                external_txt = str(txt_candidates[0].resolve())

            n_json_files = len(list(match_path.glob("*.json")))
            n_txt_files = len(list(match_path.glob("*.txt")))

        rows.append(
            {
                "scene_id": scene_id,
                "date": date,
                "crop": crop,
                "pix4d_project_name": match_name,
                "pix4d_project_path": str(match_path.resolve()) if match_path else "",
                "pix4d_found": has_match,
                "input_cameras_json": input_cameras,
                "calibrated_cameras_json": calibrated_cameras,
                "external_camera_parameters_txt": external_txt,
                "n_json_files": n_json_files,
                "n_txt_files": n_txt_files,
            }
        )

    fieldnames = [
        "scene_id",
        "date",
        "crop",
        "pix4d_project_name",
        "pix4d_project_path",
        "pix4d_found",
        "input_cameras_json",
        "calibrated_cameras_json",
        "external_camera_parameters_txt",
        "n_json_files",
        "n_txt_files",
    ]

    write_csv(output_root / "manifests/pix4d_manifest.csv", rows, fieldnames)
    write_json(output_root / "manifests/pix4d_manifest.json", rows)


if __name__ == "__main__":
    main()

