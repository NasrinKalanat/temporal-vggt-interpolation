from __future__ import annotations

import argparse
import csv
import json
import random
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_root": "data",
    "output_root": "prepared_data",
    "selected_dates": [
        "20230817", "20230822", "20230827", "20230831",
        "20230906", "20230911", "20230917", "20230922",
    ],
    "selected_crops": ["corn", "soybean"],
    "selected_platforms": ["matic"],
    "random_seed": 42,
    "build_roi": True,
}

SCENE_RE = re.compile(r"^nerfstudio_(matic|mapper)(\d{8})(?:_(corn|soybean))?$")
NADIR_RE = re.compile(r"^(corn|soy)_(\d{4})\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class SceneParts:
    scene_id: str
    platform: str
    date: str
    crop: str


def parse_args(default_config: str = "configs/prepare_data.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def load_config(config_path: Path | None) -> dict[str, Any]:
    config: dict[str, Any] = dict(DEFAULT_CONFIG)

    if config_path and config_path.exists():
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required to read YAML config. Install pyyaml or use JSON config."
                ) from exc
            file_cfg = yaml.safe_load(config_path.read_text()) or {}
        elif config_path.suffix.lower() == ".json":
            file_cfg = json.loads(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

        if not isinstance(file_cfg, dict):
            raise ValueError("Config file must contain a dictionary/object at top level")
        config.update(file_cfg)

    return config


def merge_cli_paths(config: dict[str, Any], dataset_root: Path | None, output_root: Path | None) -> dict[str, Any]:
    merged = dict(config)
    if dataset_root is not None:
        merged["dataset_root"] = str(dataset_root)
    if output_root is not None:
        merged["output_root"] = str(output_root)
    return merged


def scene_parts_from_name(scene_name: str) -> SceneParts | None:
    match = SCENE_RE.match(scene_name)
    if not match:
        return None
    platform, date, crop = match.groups()
    return SceneParts(scene_id=scene_name, platform=platform, date=date, crop=crop or "unknown")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def count_images(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and is_image_file(p))


def safe_json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def resolve_frame_path(scene_root: Path, file_path: str) -> Path:
    # Nerfstudio commonly stores frame paths relative to scene root.
    clean = file_path.strip().replace("\\", "/")
    if clean.startswith("./"):
        clean = clean[2:]
    return (scene_root / clean)


def stable_shuffle(items: list[Any], seed: int) -> list[Any]:
    out = list(items)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def parse_nadir_filename(name: str) -> tuple[str, str] | None:
    match = NADIR_RE.match(name)
    if not match:
        return None
    crop_token, mmdd = match.groups()
    crop = "corn" if crop_token.lower() == "corn" else "soybean"
    return crop, f"2023{mmdd}"


def tiff_size(path: Path) -> tuple[int | None, int | None]:
    """Read TIFF width/height from tags 256/257 (baseline TIFF)."""
    data = path.read_bytes()
    if len(data) < 8:
        return None, None

    byte_order = data[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None, None

    version = struct.unpack(endian + "H", data[2:4])[0]
    if version != 42:
        return None, None

    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    if ifd_offset + 2 > len(data):
        return None, None

    n_entries = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    cursor = ifd_offset + 2

    width = None
    height = None

    for _ in range(n_entries):
        if cursor + 12 > len(data):
            break
        tag, field_type, count, value = struct.unpack(endian + "HHII", data[cursor : cursor + 12])
        cursor += 12

        if count < 1:
            continue

        if field_type == 3:  # SHORT
            parsed_value = value & 0xFFFF
        elif field_type == 4:  # LONG
            parsed_value = value
        else:
            continue

        if tag == 256:
            width = int(parsed_value)
        elif tag == 257:
            height = int(parsed_value)

    return width, height


def ensure_output_tree(output_root: Path) -> None:
    paths = [
        "inventory",
        "manifests",
        "frame_audit",
        "cleaned_frames",
        "cleaned_transforms",
        "subsets",
        "splits",
        "roi/masks_coarse",
        "logs",
    ]
    for rel in paths:
        (output_root / rel).mkdir(parents=True, exist_ok=True)

