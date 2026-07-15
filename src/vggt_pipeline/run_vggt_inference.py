"""Run VGGT inference over all triplet combinations from all_triplets.json.

For each t1_t2_t3_crop entry, applies a sliding window (n_views, max_overlap_views)
over the pre-ordered matched triplets (no shuffling — order is already set).
Each variant runs three separate VGGT forward passes, one per date (t1, t2, t3).

Output layout:
    output_root/
    └── {t1}_{t2}_{t3}_{crop}/
        └── variant_{i:02d}/
            ├── t1/
            │   ├── input_images.txt
            │   ├── selected_images.json
            │   ├── dataset_cameras.json
            │   ├── run_request.json
            │   ├── run_status.json
            │   └── predictions/
            ├── t2/  (same)
            └── t3/  (same)

Usage:
    python src/vggt_pipeline/run_vggt_inference.py [--config configs/triplet_inference.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "all_triplets_path": "prepared_data/all_triplets.json",
    "output_root": "vggt_output_triplets",
    "triplet_ids": None,
    "crops": None,
    "n_views": 8,
    "max_overlap_views": 2,
    "max_variants": None,
    "model_id": "facebook/VGGT-1B",
    "device": "auto",
    "image_preprocess_mode": "pad",
    "skip_existing": True,
    "t2_only": False,
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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
    parser = argparse.ArgumentParser(description="Run VGGT inference over triplet combinations.")
    parser.add_argument("--config", type=Path, default=Path("configs/triplet_inference.yaml"))
    parser.add_argument("--all-triplets-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--triplet-id", action="append", default=None)
    parser.add_argument("--crop", action="append", default=None)
    parser.add_argument("--n-views", type=int, default=None)
    parser.add_argument("--max-overlap-views", type=int, default=None)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--image-preprocess-mode", choices=["pad", "crop"], default=None)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=None)
    parser.add_argument("--t2-only", dest="t2_only", action="store_true", default=None,
                        help="Only run VGGT inference for t2; write metadata for t1/t3 without predictions.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = dict(DEFAULT_CONFIG)
    config.update(load_yaml_config(args.config))

    if args.all_triplets_path is not None:
        config["all_triplets_path"] = args.all_triplets_path
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.triplet_id is not None:
        config["triplet_ids"] = args.triplet_id
    if args.crop is not None:
        config["crops"] = args.crop
    if args.n_views is not None:
        config["n_views"] = args.n_views
    if args.max_overlap_views is not None:
        config["max_overlap_views"] = args.max_overlap_views
    if args.max_variants is not None:
        config["max_variants"] = args.max_variants
    if args.model_id is not None:
        config["model_id"] = args.model_id
    if args.device is not None:
        config["device"] = args.device
    if args.image_preprocess_mode is not None:
        config["image_preprocess_mode"] = args.image_preprocess_mode
    if args.skip_existing is not None:
        config["skip_existing"] = args.skip_existing
    if args.t2_only is not None:
        config["t2_only"] = args.t2_only

    config["all_triplets_path"] = Path(config["all_triplets_path"])
    config["output_root"] = Path(config["output_root"])
    config["n_views"] = int(config["n_views"])
    config["max_overlap_views"] = int(config["max_overlap_views"])
    if config.get("max_variants") is not None:
        config["max_variants"] = int(config["max_variants"])
    if isinstance(config.get("triplet_ids"), str):
        config["triplet_ids"] = [config["triplet_ids"]]
    if isinstance(config.get("crops"), str):
        config["crops"] = [config["crops"]]
    return config


def triplet_id(entry: dict[str, Any]) -> str:
    return f"{entry['t1']}_{entry['t2']}_{entry['t3']}_{entry['crop']}"


def compute_triplet_batches(
    triplets: list[dict[str, Any]],
    n_views: int,
    max_overlap_views: int,
    max_variants: int | None,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Sliding window over pre-ordered matched triplets — no shuffling."""
    if len(triplets) < n_views:
        return []
    stride = max(1, n_views - max_overlap_views)
    batches = []
    start = 0
    batch_idx = 0
    while start + n_views <= len(triplets):
        batches.append((batch_idx, triplets[start : start + n_views]))
        start += stride
        batch_idx += 1
        if max_variants is not None and len(batches) >= max_variants:
            break
    return batches


def resolve_date_views(
    date_views: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Resolve absolute image paths and build dataset_cameras for one date's views.

    Each view dict comes from all_triplets.json and has:
        image_path, source_transforms_path, transform_matrix, colmap_im_id
    """
    transforms_path = Path(date_views[0]["source_transforms_path"])
    scene_root = transforms_path.parent
    transforms = read_json(transforms_path)

    image_paths: list[str] = []
    selected_images: list[dict[str, Any]] = []
    camera_frames: list[dict[str, Any]] = []

    for i, view in enumerate(date_views):
        rel_path = view["image_path"]
        abs_path = str((scene_root / rel_path).resolve())
        image_paths.append(abs_path)
        selected_images.append({"frame_index": i, "file_path": rel_path, "image_path": abs_path})
        camera_frames.append({
            "frame_index": i,
            "file_path": rel_path,
            "image_path": abs_path,
            "transform_matrix": view.get("transform_matrix", []),
        })

    dataset_cameras = {
        "scene_root": str(scene_root),
        "source_transforms_path": str(transforms_path),
        "intrinsics": {
            "w": transforms.get("w"),
            "h": transforms.get("h"),
            "fl_x": transforms.get("fl_x"),
            "fl_y": transforms.get("fl_y"),
            "cx": transforms.get("cx"),
            "cy": transforms.get("cy"),
            "k1": transforms.get("k1"),
            "k2": transforms.get("k2"),
            "k3": transforms.get("k3"),
            "k4": transforms.get("k4"),
            "p1": transforms.get("p1"),
            "p2": transforms.get("p2"),
            "camera_model": transforms.get("camera_model"),
        },
        "frames": camera_frames,
    }
    return image_paths, selected_images, dataset_cameras


def run_date_inference(
    date_label: str,
    date_str: str,
    date_views: list[dict[str, Any]],
    variant_dir: Path,
    config: dict[str, Any],
    runner: Any,
    label: str,
    skip_vggt: bool = False,
) -> dict[str, Any]:
    from vggt_pipeline.execute_vggt import run_vggt_inference_from_image_paths

    date_dir = variant_dir / date_label
    pred_dir = date_dir / "predictions"

    if skip_vggt:
        # t2_only mode: write camera/image metadata but skip the forward pass.
        if config.get("skip_existing") and (date_dir / "dataset_cameras.json").exists():
            log(f"{label} skip existing metadata date={date_str}")
            return {"date": date_str, "date_label": date_label, "status": "skipped_t2_only"}
        date_dir.mkdir(parents=True, exist_ok=True)
        image_paths, selected_images, dataset_cameras = resolve_date_views(date_views)
        Path(date_dir / "input_images.txt").write_text("\n".join(image_paths) + "\n")
        write_json(date_dir / "selected_images.json", selected_images)
        write_json(date_dir / "dataset_cameras.json", dataset_cameras)
        write_json(date_dir / "run_status.json", {
            "date": date_str, "date_label": date_label, "status": "skipped_t2_only",
        })
        log(f"{label} metadata only (t2_only) date={date_str}")
        return {"date": date_str, "date_label": date_label, "status": "skipped_t2_only"}

    all_pred_files = [
        "point_map.npy", "point_confidence.npy",
        "extrinsic.npy", "intrinsic.npy",
        "depth_map.npy", "depth_confidence.npy",
    ]
    if config.get("skip_existing") and all((pred_dir / f).exists() for f in all_pred_files):
        log(f"{label} skip existing date={date_str}")
        return {"date": date_str, "status": "skipped"}

    date_dir.mkdir(parents=True, exist_ok=True)
    image_paths, selected_images, dataset_cameras = resolve_date_views(date_views)

    list_path = date_dir / "input_images.txt"
    list_path.write_text("\n".join(image_paths) + "\n")
    write_json(date_dir / "selected_images.json", selected_images)
    write_json(date_dir / "dataset_cameras.json", dataset_cameras)
    write_json(date_dir / "run_request.json", {
        "date": date_str,
        "date_label": date_label,
        "n_input_views": len(image_paths),
        "model_id": config["model_id"],
        "image_preprocess_mode": config["image_preprocess_mode"],
    })

    metadata = run_vggt_inference_from_image_paths(
        image_paths=image_paths,
        output_dir=date_dir,
        runner=runner,
        image_preprocess_mode=config["image_preprocess_mode"],
        input_image_list_path=list_path,
    )

    status = {
        "date": date_str,
        "date_label": date_label,
        "status": "completed",
        "duration_sec": metadata.get("duration_sec"),
    }
    write_json(date_dir / "run_status.json", status)
    log(f"{label} done date={date_str} duration_sec={metadata.get('duration_sec'):.1f}")
    return status


def run_variant(
    entry: dict[str, Any],
    batch_idx: int,
    batch_triplets: list[dict[str, Any]],
    config: dict[str, Any],
    output_root: Path,
    runner: Any,
    label: str,
) -> dict[str, Any]:
    tid = triplet_id(entry)
    variant_name = f"variant_{batch_idx:02d}"
    variant_dir = output_root / tid / variant_name

    date_labels = ["t1", "t2", "t3"]
    view_keys = ["v1", "v2", "v3"]
    date_strs = [entry["t1"], entry["t2"], entry["t3"]]
    t2_only = config.get("t2_only", False)

    if config.get("skip_existing"):
        all_pred_files_t2 = [
            "point_map.npy", "point_confidence.npy",
            "extrinsic.npy", "intrinsic.npy",
            "depth_map.npy", "depth_confidence.npy",
        ]
        t2_done = all(
            (variant_dir / "t2" / "predictions" / f).exists() for f in all_pred_files_t2
        )
        ctx_done = all(
            (variant_dir / dl / "dataset_cameras.json").exists() for dl in ("t1", "t3")
        )
        if t2_only and t2_done and ctx_done:
            log(f"{label} skip existing triplet={tid} variant={variant_name}")
            return {"triplet_id": tid, "variant": variant_name, "status": "skipped"}
        all_pred_files = [
            "point_map.npy", "point_confidence.npy",
            "extrinsic.npy", "intrinsic.npy",
            "depth_map.npy", "depth_confidence.npy",
        ]
        if not t2_only and all(
            all((variant_dir / dl / "predictions" / f).exists() for f in all_pred_files)
            for dl in date_labels
        ):
            log(f"{label} skip existing triplet={tid} variant={variant_name}")
            return {"triplet_id": tid, "variant": variant_name, "status": "skipped"}

    log(f"{label} start triplet={tid} variant={variant_name} n_views={len(batch_triplets)}")
    variant_results = []
    for date_label, view_key, date_str in zip(date_labels, view_keys, date_strs):
        date_views = [t[view_key] for t in batch_triplets]
        date_label_str = f"{label}[{date_label}]"
        skip_vggt = t2_only and date_label != "t2"
        status = run_date_inference(
            date_label, date_str, date_views, variant_dir, config, runner, date_label_str,
            skip_vggt=skip_vggt,
        )
        variant_results.append(status)

    all_done = all(
        r["status"] in ("completed", "skipped", "skipped_t2_only") for r in variant_results
    )
    return {
        "triplet_id": tid,
        "variant": variant_name,
        "status": "completed" if all_done else "partial",
        "dates": variant_results,
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vggt_pipeline.execute_vggt import get_vggt_runner

    args = parse_args()
    config = build_config(args)

    all_triplets: list[dict[str, Any]] = read_json(config["all_triplets_path"])

    filter_ids = config.get("triplet_ids")
    if filter_ids:
        filter_set = set(filter_ids)
        all_triplets = [e for e in all_triplets if triplet_id(e) in filter_set]
        if not all_triplets:
            raise ValueError(f"No triplet entries matched triplet_ids={filter_ids}")

    filter_crops = config.get("crops")
    if filter_crops:
        crop_set = set(filter_crops)
        all_triplets = [e for e in all_triplets if e["crop"] in crop_set]
        if not all_triplets:
            raise ValueError(f"No triplet entries matched crops={filter_crops}")

    output_root = config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", {k: str(v) for k, v in config.items()})

    log(
        f"triplets={len(all_triplets)} n_views={config['n_views']} "
        f"max_overlap_views={config['max_overlap_views']} "
        f"max_variants={config['max_variants']} output_root={output_root}"
    )

    log(f"loading VGGT model_id={config['model_id']}")
    runner = get_vggt_runner(model_id=config["model_id"], device=config["device"], use_cache=True)
    log("model loaded")

    summary: list[dict[str, Any]] = []
    for i, entry in enumerate(all_triplets, start=1):
        tid = triplet_id(entry)
        batches = compute_triplet_batches(
            entry["triplets"], config["n_views"], config["max_overlap_views"], config["max_variants"]
        )
        if not batches:
            log(f"[{i}/{len(all_triplets)}] triplet={tid} fewer than {config['n_views']} triplets, skipping")
            summary.append({"triplet_id": tid, "status": "skipped_insufficient_views"})
            continue

        log(f"[{i}/{len(all_triplets)}] triplet={tid} n_triplets={len(entry['triplets'])} n_variants={len(batches)}")
        for batch_idx, batch_triplets in batches:
            label = f"[{i}/{len(all_triplets)}][{batch_idx+1}/{len(batches)}]"
            result = run_variant(entry, batch_idx, batch_triplets, config, output_root, runner, label)
            summary.append(result)

    write_json(output_root / "run_summary.json", summary)
    log(f"all done. summary: {output_root / 'run_summary.json'}")


if __name__ == "__main__":
    main()

