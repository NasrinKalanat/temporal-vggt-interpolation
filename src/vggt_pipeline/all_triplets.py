import argparse
import json
from pathlib import Path
from typing import Any
import random

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_manifest": "prepared_data/manifests/dataset_manifest.json",
    "subset_manifest": "prepared_data/subsets/benchmark_subset.json",
    "output_path": "prepared_data/all_triplet.json",
    "selected_dates": [],
    "selected_crops": [],
    "seed": 42
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for config files.") from exc
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must be a key-value mapping.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT inference for benchmark scenes."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/all_triplet.yaml")
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = dict(DEFAULT_CONFIG)
    config.update(load_yaml_config(args.config))
    return config


def main(config: dict[str, Any]):
    random.seed(config["seed"])

    selected_dates = sorted(config["selected_dates"])

    dataset = read_json(Path(config["subset_manifest"]))
    scenes = dataset["scenes"]
    by_crop = {crop: {} for crop in config["selected_crops"]}

    for scene in scenes:
        by_crop[scene["crop"]][scene["date"]] = scene
    all_triplets = []
    for crop in config["selected_crops"]:
        by_date = by_crop[crop]
        for i in range(len(selected_dates)):
            t1 = selected_dates[i]
            if t1 not in by_date:
                continue
            for j in range(i + 1, len(selected_dates)):
                t2 = selected_dates[j]
                if t2 not in by_date:
                    continue
                for k in range(j + 1, len(selected_dates)):
                    t3 = selected_dates[k]
                    if t3 not in by_date:
                        continue
                    t1_transforms = read_json(Path(by_date[t1]["transforms_path"]))
                    t2_transforms = read_json(Path(by_date[t2]["transforms_path"]))
                    t3_transforms = read_json(Path(by_date[t3]["transforms_path"]))
                    random.shuffle(t1_transforms["frames"])
                    random.shuffle(t2_transforms["frames"])
                    random.shuffle(t3_transforms["frames"])
                    triplets = []
                    def create_view(date_label, transform_path, frame: dict[str, Any]):
                        return {
                            "date": date_label,
                            "colmap_im_id": frame.get("colmap_im_id", None),
                            "image_path": frame.get("file_path", None),
                            "source_transforms_path": transform_path,
                            "transform_matrix": frame.get("transform_matrix", None)
                        }
                    for frame1, frame2, frame3 in zip(t1_transforms["frames"], t2_transforms["frames"], t3_transforms["frames"]):
                        triplets.append({
                            "v1": create_view("t1", by_date[t1]["transforms_path"], frame1),
                            "v2": create_view("t2", by_date[t2]["transforms_path"], frame2),
                            "v3": create_view("t3", by_date[t3]["transforms_path"], frame3),
                        })
                    print(f"Triplet {t1}_{t2}_{t3}_{crop}: {len(triplets)} views")
                    all_triplets.append(
                        {
                            "t1": t1,
                            "t2": t2,
                            "t3": t3,
                            "crop": crop,
                            "triplets": triplets,
                        }
                    )
    for crop in config["selected_crops"]:
        print(
            f"{crop}: {len(list(filter(lambda x: x['crop'] == crop, all_triplets)))} triplets"
        )
    write_json(Path(config["output_path"]), all_triplets)


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    main(config)

