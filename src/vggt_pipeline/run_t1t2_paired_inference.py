"""Run VGGT inference for t1t2-paired triplets (free t3).

Input: t1t2_paired_triplets.json (from t1t2_paired_triplets.py)
  Each entry has:
    - pairs: camera-consistent (v1, v2) view pairs for t1 and t2
    - views_t3: all t3 views, unconstrained

Variant generation:
  1. Slide a window of n_views over the sorted t1-t2 pairs → t1t2 windows.
  2. For each t1t2 window, sample t3_variants_per_window random windows of
     n_views from views_t3.
  3. Each (t1t2_window, t3_window) combination becomes one variant.
  Total variants per entry ≤ n_t1t2_windows × t3_variants_per_window.

Variant naming: variant_{t1t2_idx:02d}_{t3_idx:02d}

Output layout (same as run_vggt_inference.py):
    output_root/
    └── {t1}_{t2}_{t3}_{crop}/
        └── variant_{t1t2_idx:02d}_{t3_idx:02d}/
            ├── t1/  (input_images.txt, selected_images.json, dataset_cameras.json, predictions/)
            ├── t2/  (same)
            └── t3/  (same)

Usage:
    python src/vggt_pipeline/run_t1t2_paired_inference.py \\
        --config configs/t1t2_paired_inference.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_pipeline.execute_vggt import (
    get_vggt_runner,
    run_vggt_inference_from_image_paths,
    save_cached_layers,
    cached_layers_exist,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "paired_triplets_path": "prepared_data/t1t2_paired_triplets.json",
    "output_root": "vggt_outputs/t1t2_paired",
    "triplet_ids": None,
    "crops": None,
    "n_views": 16,
    "max_overlap_views": 4,
    "max_t1t2_windows": None,          # cap on sliding windows over t1-t2 pairs per entry
    "t3_variants_per_window": 3,        # number of random t3 windows per t1t2 window
    "seed": 42,
    "model_id": "facebook/VGGT-1B",
    "device": "auto",
    "image_preprocess_mode": "pad",
    "skip_existing": True,
    "t2_only": False,
    "point_map_only": False,
    "dry_run": False,
    "t2_cache_layers": [4, 11, 17, 23],  # aggregator layer indices to cache for t2
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
    parser = argparse.ArgumentParser(
        description="Run VGGT inference for t1t2-paired triplets with free t3."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/t1t2_paired_inference.yaml"))
    parser.add_argument("--paired-triplets-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--triplet-id", action="append", default=None)
    parser.add_argument("--crop", action="append", default=None)
    parser.add_argument("--n-views", type=int, default=None)
    parser.add_argument("--max-overlap-views", type=int, default=None)
    parser.add_argument("--max-t1t2-windows", type=int, default=None)
    parser.add_argument("--t3-variants-per-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-preprocess-mode", choices=["pad", "crop"], default=None)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=None)
    parser.add_argument("--t2-only", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Total number of GPU workers for data-parallel sharding.")
    parser.add_argument("--gpu-rank", type=int, default=0,
                        help="This worker's rank (0-indexed). Each rank processes its shard of variants.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = dict(DEFAULT_CONFIG)
    config.update(load_yaml_config(args.config))

    if args.paired_triplets_path is not None:
        config["paired_triplets_path"] = args.paired_triplets_path
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
    if args.max_t1t2_windows is not None:
        config["max_t1t2_windows"] = args.max_t1t2_windows
    if args.t3_variants_per_window is not None:
        config["t3_variants_per_window"] = args.t3_variants_per_window
    if args.seed is not None:
        config["seed"] = args.seed
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
    if args.dry_run is not None:
        config["dry_run"] = args.dry_run

    config["num_gpus"] = args.num_gpus
    config["gpu_rank"] = args.gpu_rank
    # Override device to use the GPU corresponding to this worker's rank
    if args.num_gpus > 1 and args.device is None:
        config["device"] = f"cuda:{args.gpu_rank}"

    config["paired_triplets_path"] = Path(config["paired_triplets_path"])
    config["output_root"] = Path(config["output_root"])
    if isinstance(config.get("triplet_ids"), str):
        config["triplet_ids"] = [config["triplet_ids"]]
    if isinstance(config.get("crops"), str):
        config["crops"] = [config["crops"]]
    return config


def entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['t1']}_{entry['t2']}_{entry['t3']}_{entry['crop']}"


def sliding_windows(items: list, n: int, max_overlap: int, max_windows: int | None) -> list[tuple[int, list]]:
    """Sliding window over items. Returns list of (window_idx, window_items)."""
    if len(items) < n:
        return []
    stride = max(1, n - max_overlap)
    windows = []
    start = 0
    while start + n <= len(items):
        windows.append((len(windows), items[start: start + n]))
        start += stride
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def sample_t3_windows(
    views_t3: list[dict],
    n: int,
    max_overlap: int,
    k: int,
    rng: random.Random,
) -> list[tuple[int, list]]:
    """Sample k random windows of size n from views_t3.

    Generates all valid windows, then picks k without replacement (or all if fewer).
    Returns list of (t3_idx, window_views).
    """
    if len(views_t3) < n:
        return []
    stride = max(1, n - max_overlap)
    all_starts = list(range(0, len(views_t3) - n + 1, stride))
    chosen = rng.sample(all_starts, min(k, len(all_starts)))
    chosen.sort()  # deterministic ordering within a variant set
    return [(idx, views_t3[start: start + n]) for idx, start in enumerate(chosen)]


def resolve_date_views(
    date_views: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Resolve absolute image paths and build dataset_cameras for one date's views."""
    transforms_path = Path(date_views[0]["source_transforms_path"])
    scene_root = transforms_path.parent
    transforms = read_json(transforms_path)

    image_paths: list[str] = []
    selected_images: list[dict[str, Any]] = []
    camera_frames: list[dict[str, Any]] = []

    for i, view in enumerate(date_views):
        abs_path = str((scene_root / view["image_path"]).resolve())
        image_paths.append(abs_path)
        selected_images.append({"frame_index": i, "file_path": view["image_path"], "image_path": abs_path})
        camera_frames.append({
            "frame_index": i,
            "file_path": view["image_path"],
            "image_path": abs_path,
            "transform_matrix": view.get("transform_matrix", []),
        })

    dataset_cameras = {
        "scene_root": str(scene_root),
        "source_transforms_path": str(transforms_path),
        "intrinsics": {k: transforms.get(k) for k in (
            "w", "h", "fl_x", "fl_y", "cx", "cy",
            "k1", "k2", "k3", "k4", "p1", "p2", "camera_model",
        )},
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
    date_dir = variant_dir / date_label
    pred_dir = date_dir / "predictions"

    if skip_vggt:
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

    if config.get("skip_existing") and (pred_dir / "point_map.npy").exists() and (pred_dir / "extrinsic.npy").exists():
        # Even if full inference is skipped, ensure cached layers exist for t2
        cache_layers = config.get("t2_cache_layers", [4, 11, 17, 23])
        if date_label == "t2" and not cached_layers_exist(date_dir, cache_layers):
            image_paths, _, _ = resolve_date_views(date_views)
            save_cached_layers(
                image_paths=image_paths,
                output_dir=date_dir,
                runner=runner,
                image_preprocess_mode=config["image_preprocess_mode"],
                cache_layers=cache_layers,
            )
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

    # Save cached layers for t2 (reuses preprocessed images via save_cached_layers)
    if date_label == "t2":
        save_cached_layers(
            image_paths=image_paths,
            output_dir=date_dir,
            runner=runner,
            image_preprocess_mode=config["image_preprocess_mode"],
            cache_layers=config.get("t2_cache_layers", [4, 11, 17, 23]),
        )

    status = {
        "date": date_str,
        "date_label": date_label,
        "status": "completed",
        "duration_sec": metadata.get("duration_sec"),
    }
    write_json(date_dir / "run_status.json", status)
    log(f"{label} done date={date_str} duration={metadata.get('duration_sec'):.1f}s")
    return status


def run_variant(
    entry: dict[str, Any],
    t1t2_idx: int,
    t3_idx: int,
    t1_views: list[dict],
    t2_views: list[dict],
    t3_views: list[dict],
    config: dict[str, Any],
    output_root: Path,
    runner: Any,
    label: str,
) -> dict[str, Any]:
    eid = entry_id(entry)
    variant_name = f"variant_{t1t2_idx:02d}_{t3_idx:02d}"
    variant_dir = output_root / eid / variant_name
    t2_only = config.get("t2_only", False)

    if config.get("skip_existing"):
        cache_layers = config.get("t2_cache_layers", [4, 11, 17, 23])
        t2_cached = cached_layers_exist(variant_dir / "t2", cache_layers)
        t2_done = (variant_dir / "t2" / "predictions" / "point_map.npy").exists()
        ctx_done = all((variant_dir / dl / "dataset_cameras.json").exists() for dl in ("t1", "t3"))
        if t2_only and t2_done and t2_cached and ctx_done:
            log(f"{label} skip existing entry={eid} variant={variant_name}")
            return {"entry_id": eid, "variant": variant_name, "status": "skipped"}
        if not t2_only and t2_cached and all(
            (variant_dir / dl / "predictions" / "point_map.npy").exists()
            and (variant_dir / dl / "predictions" / "extrinsic.npy").exists()
            for dl in ("t1", "t2", "t3")
        ):
            log(f"{label} skip existing entry={eid} variant={variant_name}")
            return {"entry_id": eid, "variant": variant_name, "status": "skipped"}

    log(f"{label} start entry={eid} variant={variant_name}")
    variant_results = []
    for date_label, date_str, date_views in [
        ("t1", entry["t1"], t1_views),
        ("t2", entry["t2"], t2_views),
        ("t3", entry["t3"], t3_views),
    ]:
        skip_vggt = t2_only and date_label != "t2"
        status = run_date_inference(
            date_label, date_str, date_views, variant_dir, config, runner,
            f"{label}[{date_label}]", skip_vggt=skip_vggt,
        )
        variant_results.append(status)

    all_done = all(r["status"] in ("completed", "skipped", "skipped_t2_only") for r in variant_results)
    return {
        "entry_id": eid,
        "variant": variant_name,
        "status": "completed" if all_done else "partial",
        "dates": variant_results,
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)

    rng = random.Random(config.get("seed", 42))

    all_entries: list[dict[str, Any]] = read_json(config["paired_triplets_path"])

    if config.get("triplet_ids"):
        id_set = set(config["triplet_ids"])
        all_entries = [e for e in all_entries if entry_id(e) in id_set]
        if not all_entries:
            raise ValueError(f"No entries matched triplet_ids={config['triplet_ids']}")

    if config.get("crops"):
        crop_set = set(config["crops"])
        all_entries = [e for e in all_entries if e["crop"] in crop_set]
        if not all_entries:
            raise ValueError(f"No entries matched crops={config['crops']}")

    output_root = config["output_root"]

    n_views          = int(config["n_views"])
    max_overlap      = int(config["max_overlap_views"])
    max_t1t2_windows = config.get("max_t1t2_windows")
    t3_variants      = int(config["t3_variants_per_window"])

    dry_run = config.get("dry_run", False)

    log(
        f"{'[DRY RUN] ' if dry_run else ''}"
        f"entries={len(all_entries)} n_views={n_views} max_overlap={max_overlap} "
        f"max_t1t2_windows={max_t1t2_windows} t3_variants_per_window={t3_variants} "
        f"output_root={output_root}"
    )

    if dry_run:
        total_variants = 0
        for entry_idx, entry in enumerate(all_entries, start=1):
            eid = entry_id(entry)
            pairs = entry["pairs"]
            views_t3 = entry["views_t3"]

            t1t2_windows = sliding_windows(pairs, n_views, max_overlap, max_t1t2_windows)
            if not t1t2_windows:
                print(f"  {eid}: 0 variants (fewer than {n_views} pairs)")
                continue

            t3_pool = sample_t3_windows(views_t3, n_views, max_overlap, t3_variants * len(t1t2_windows), rng)
            if not t3_pool:
                print(f"  {eid}: 0 variants (fewer than {n_views} t3 views)")
                continue

            n_variants = min(len(t3_pool), len(t1t2_windows) * t3_variants)
            total_variants += n_variants
            print(f"  {eid}: {n_variants} variants (t1t2_windows={len(t1t2_windows)}, t3_pool={len(t3_pool)})")

        print(f"\nNum triplets: {len(all_entries)}")
        print(f"Total variants: {total_variants}")
        return

    log(f"loading VGGT model_id={config['model_id']}")
    runner = get_vggt_runner(model_id=config["model_id"], device=config["device"], use_cache=True)
    log("model loaded")

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", {k: str(v) for k, v in config.items()})

    # Build full list of work items (entry, t1t2_idx, t3_idx, views)
    work_items: list[tuple] = []

    for entry_idx, entry in enumerate(all_entries, start=1):
        eid = entry_id(entry)
        pairs    = entry["pairs"]
        views_t3 = entry["views_t3"]

        t1t2_windows = sliding_windows(pairs, n_views, max_overlap, max_t1t2_windows)
        if not t1t2_windows:
            continue

        t3_pool = sample_t3_windows(views_t3, n_views, max_overlap, t3_variants * len(t1t2_windows), rng)
        if not t3_pool:
            continue

        t3_cursor = 0
        for t1t2_idx, pair_window in t1t2_windows:
            t1_views = [p["v1"] for p in pair_window]
            t2_views = [p["v2"] for p in pair_window]

            for t3_idx in range(t3_variants):
                if t3_cursor >= len(t3_pool):
                    break
                _, t3_window = t3_pool[t3_cursor]
                t3_cursor += 1
                work_items.append((entry_idx, entry, t1t2_idx, t3_idx, t1_views, t2_views, t3_window))

    # Shard work items across GPUs
    num_gpus = config["num_gpus"]
    gpu_rank = config["gpu_rank"]
    my_items = work_items[gpu_rank::num_gpus]

    log(
        f"rank={gpu_rank}/{num_gpus} total_variants={len(work_items)} "
        f"my_variants={len(my_items)} device={config['device']}"
    )

    summary: list[dict[str, Any]] = []

    for item_idx, (entry_idx, entry, t1t2_idx, t3_idx, t1_views, t2_views, t3_window) in enumerate(my_items, start=1):
        eid = entry_id(entry)
        n_t1t2 = len(sliding_windows(entry["pairs"], n_views, max_overlap, max_t1t2_windows))
        label = (
            f"[rank{gpu_rank} {item_idx}/{len(my_items)}]"
            f"[{entry_idx}/{len(all_entries)}]"
            f"[t1t2={t1t2_idx+1}/{n_t1t2},t3={t3_idx+1}/{t3_variants}]"
        )
        result = run_variant(
            entry, t1t2_idx, t3_idx,
            t1_views, t2_views, t3_window,
            config, output_root, runner, label,
        )
        summary.append(result)

    summary_path = output_root / f"run_summary_rank{gpu_rank}.json"
    write_json(summary_path, summary)
    log(f"rank {gpu_rank} done. summary: {summary_path}")

if __name__ == "__main__":
    main()

