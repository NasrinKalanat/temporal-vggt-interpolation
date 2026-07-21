"""Run VGGT inference for non-view-consistent temporal triplets.

Input: non_view_consistent_triplets.json from non_view_consistent_triplets.py.

Variant generation:
  1. Slide independent windows of n_views over t1, t2, and t3 views.
  2. Combine t1/t2/t3 windows into variants.
  3. If max_variants_per_triplet is null, generate every valid combination.
     Otherwise, sample up to that many combinations deterministically.

Variant naming: variant_{t1_idx:02d}_{t2_idx:02d}_{t3_idx:02d}
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_pipeline.execute_vggt import (
    get_vggt_runner,
    normalize_prediction_outputs,
)
from vggt_pipeline.run_t1t2_paired_inference import (
    outputs_for_date,
    predictions_exist,
    read_json,
    resolve_date_views,
    run_date_inference,
    sliding_windows,
    write_json,
)
from vggt_pipeline.camera_consistent_triplets import load_yaml_config

DEFAULT_CONFIG: dict[str, Any] = {
    "triplets_path": "prepared_data/non_view_consistent_triplets.json",
    "output_root": "vggt_outputs/non_view_consistent",
    "triplet_ids": None,
    "crops": None,
    "n_views": 16,
    "max_overlap_views": 4,
    "max_windows_per_date": None,
    "max_variants_per_triplet": None,
    "require_distinct_view_windows": True,
    "seed": 42,
    "model_id": "facebook/VGGT-1B",
    "device": "auto",
    "image_preprocess_mode": "pad",
    "skip_existing": True,
    "t2_only": False,
    "prediction_outputs": "all",
    "point_map_only": False,
    "dry_run": False,
    "t2_cache_layers": [4, 11, 17, 23],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT inference for non-view-consistent temporal triplets."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/non_view_consistent_inference.yaml"))
    parser.add_argument("--triplets-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--triplet-id", action="append", default=None)
    parser.add_argument("--crop", action="append", default=None)
    parser.add_argument("--n-views", type=int, default=None)
    parser.add_argument("--max-overlap-views", type=int, default=None)
    parser.add_argument("--max-windows-per-date", type=int, default=None)
    parser.add_argument("--max-variants-per-triplet", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-preprocess-mode", choices=["pad", "crop"], default=None)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=None)
    parser.add_argument("--allow-same-view-windows", dest="require_distinct_view_windows", action="store_false", default=None)
    parser.add_argument("--t2-only", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-rank", type=int, default=0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    file_config = load_yaml_config(args.config)
    config.update(file_config)

    if args.triplets_path is not None:
        config["triplets_path"] = args.triplets_path
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
    if args.max_windows_per_date is not None:
        config["max_windows_per_date"] = args.max_windows_per_date
    if args.max_variants_per_triplet is not None:
        config["max_variants_per_triplet"] = args.max_variants_per_triplet
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
    if args.require_distinct_view_windows is not None:
        config["require_distinct_view_windows"] = args.require_distinct_view_windows
    if args.t2_only is not None:
        config["t2_only"] = args.t2_only
    if args.dry_run is not None:
        config["dry_run"] = args.dry_run

    raw_prediction_outputs = config.get("prediction_outputs")
    if config.get("point_map_only") and "prediction_outputs" not in file_config:
        config["prediction_outputs"] = ["point_map", "point_confidence"]
    elif raw_prediction_outputs is None or raw_prediction_outputs == "all":
        config["prediction_outputs"] = list(normalize_prediction_outputs("all")) + ["cached_layers"]
    else:
        config["prediction_outputs"] = list(normalize_prediction_outputs(raw_prediction_outputs))

    config["num_gpus"] = args.num_gpus
    config["gpu_rank"] = args.gpu_rank
    if args.num_gpus > 1 and args.device is None:
        config["device"] = f"cuda:{args.gpu_rank}"

    config["triplets_path"] = Path(config["triplets_path"])
    config["output_root"] = Path(config["output_root"])
    if isinstance(config.get("triplet_ids"), str):
        config["triplet_ids"] = [config["triplet_ids"]]
    if isinstance(config.get("crops"), str):
        config["crops"] = [config["crops"]]
    return config


def entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['t1']}_{entry['t2']}_{entry['t3']}_{entry['crop']}"


def view_signature(window: list[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(int(view["frame_index"]) for view in window)


def windows_are_distinct(*windows: list[dict[str, Any]]) -> bool:
    signatures = [view_signature(window) for window in windows]
    return len(set(signatures)) == len(signatures)


def build_entry_variants(
    entry: dict[str, Any],
    n_views: int,
    max_overlap: int,
    max_windows_per_date: int | None,
    max_variants_per_triplet: int | None,
    require_distinct_view_windows: bool,
    rng: random.Random,
) -> list[tuple[int, int, int, list[dict], list[dict], list[dict]]]:
    t1_windows = sliding_windows(entry["views_t1"], n_views, max_overlap, max_windows_per_date)
    t2_windows = sliding_windows(entry["views_t2"], n_views, max_overlap, max_windows_per_date)
    t3_windows = sliding_windows(entry["views_t3"], n_views, max_overlap, max_windows_per_date)
    if not t1_windows or not t2_windows or not t3_windows:
        return []

    variants = []
    for (t1_idx, t1_views), (t2_idx, t2_views), (t3_idx, t3_views) in itertools.product(
        t1_windows, t2_windows, t3_windows
    ):
        if require_distinct_view_windows and not windows_are_distinct(t1_views, t2_views, t3_views):
            continue
        variants.append((t1_idx, t2_idx, t3_idx, t1_views, t2_views, t3_views))

    if max_variants_per_triplet is not None and len(variants) > max_variants_per_triplet:
        variants = rng.sample(variants, max_variants_per_triplet)
        variants.sort(key=lambda item: item[:3])
    return variants


def variant_name(t1_idx: int, t2_idx: int, t3_idx: int) -> str:
    return f"variant_{t1_idx:02d}_{t2_idx:02d}_{t3_idx:02d}"


def run_variant(
    entry: dict[str, Any],
    t1_idx: int,
    t2_idx: int,
    t3_idx: int,
    t1_views: list[dict[str, Any]],
    t2_views: list[dict[str, Any]],
    t3_views: list[dict[str, Any]],
    config: dict[str, Any],
    output_root: Path,
    runner: Any,
    label: str,
) -> dict[str, Any]:
    eid = entry_id(entry)
    name = variant_name(t1_idx, t2_idx, t3_idx)
    variant_dir = output_root / eid / name
    t2_only = config.get("t2_only", False)

    if config.get("skip_existing"):
        cache_layers = config.get("t2_cache_layers", [4, 11, 17, 23])
        t2_done = predictions_exist(
            variant_dir / "t2",
            outputs_for_date("t2", config.get("prediction_outputs")),
            cache_layers=cache_layers,
        )
        ctx_done = all((variant_dir / date_label / "dataset_cameras.json").exists() for date_label in ("t1", "t3"))
        if t2_only and t2_done and ctx_done:
            log(f"{label} skip existing entry={eid} variant={name}")
            return {"entry_id": eid, "variant": name, "status": "skipped"}
        if not t2_only and all(
            predictions_exist(
                variant_dir / date_label,
                outputs_for_date(date_label, config.get("prediction_outputs")),
                cache_layers=cache_layers,
            )
            for date_label in ("t1", "t2", "t3")
        ):
            log(f"{label} skip existing entry={eid} variant={name}")
            return {"entry_id": eid, "variant": name, "status": "skipped"}

    log(f"{label} start entry={eid} variant={name}")
    variant_results = []
    for date_label, date_str, date_views in [
        ("t1", entry["t1"], t1_views),
        ("t2", entry["t2"], t2_views),
        ("t3", entry["t3"], t3_views),
    ]:
        skip_vggt = t2_only and date_label != "t2"
        status = run_date_inference(
            date_label,
            date_str,
            date_views,
            variant_dir,
            config,
            runner,
            f"{label}[{date_label}]",
            skip_vggt=skip_vggt,
        )
        variant_results.append(status)

    complete_statuses = {"completed", "skipped", "skipped_t2_only", "metadata_only"}
    all_done = all(result["status"] in complete_statuses for result in variant_results)
    return {
        "entry_id": eid,
        "variant": name,
        "window_indices": {"t1": t1_idx, "t2": t2_idx, "t3": t3_idx},
        "status": "completed" if all_done else "partial",
        "dates": variant_results,
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)
    rng = random.Random(config.get("seed", 42))

    all_entries: list[dict[str, Any]] = read_json(config["triplets_path"])
    if config.get("triplet_ids"):
        id_set = set(config["triplet_ids"])
        all_entries = [entry for entry in all_entries if entry_id(entry) in id_set]
        if not all_entries:
            raise ValueError(f"No entries matched triplet_ids={config['triplet_ids']}")
    if config.get("crops"):
        crop_set = set(config["crops"])
        all_entries = [entry for entry in all_entries if entry["crop"] in crop_set]
        if not all_entries:
            raise ValueError(f"No entries matched crops={config['crops']}")

    output_root = config["output_root"]
    n_views = int(config["n_views"])
    max_overlap = int(config["max_overlap_views"])
    max_windows_per_date = config.get("max_windows_per_date")
    max_variants_per_triplet = config.get("max_variants_per_triplet")
    require_distinct = bool(config.get("require_distinct_view_windows", True))
    dry_run = config.get("dry_run", False)

    log(
        f"{'[DRY RUN] ' if dry_run else ''}"
        f"entries={len(all_entries)} n_views={n_views} max_overlap={max_overlap} "
        f"max_windows_per_date={max_windows_per_date} "
        f"max_variants_per_triplet={max_variants_per_triplet} "
        f"require_distinct_view_windows={require_distinct} output_root={output_root}"
    )

    work_items: list[tuple] = []
    for entry_idx, entry in enumerate(all_entries, start=1):
        variants = build_entry_variants(
            entry,
            n_views,
            max_overlap,
            max_windows_per_date,
            max_variants_per_triplet,
            require_distinct,
            rng,
        )
        if dry_run:
            print(f"  {entry_id(entry)}: {len(variants)} variants")
        for variant in variants:
            work_items.append((entry_idx, entry, *variant))

    if dry_run:
        print(f"\nNum triplets: {len(all_entries)}")
        print(f"Total variants: {len(work_items)}")
        return

    needs_vggt = bool(config.get("prediction_outputs"))
    if needs_vggt:
        log(f"loading VGGT model_id={config['model_id']}")
        runner = get_vggt_runner(model_id=config["model_id"], device=config["device"], use_cache=True)
        log("model loaded")
    else:
        runner = None
        log("prediction_outputs is empty; VGGT model will not be loaded")

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", {key: str(value) for key, value in config.items()})

    num_gpus = int(config["num_gpus"])
    gpu_rank = int(config["gpu_rank"])
    my_items = work_items[gpu_rank::num_gpus]

    log(
        f"rank={gpu_rank}/{num_gpus} total_variants={len(work_items)} "
        f"my_variants={len(my_items)} device={config['device']}"
    )

    summary: list[dict[str, Any]] = []
    for item_idx, item in enumerate(my_items, start=1):
        entry_idx, entry, t1_idx, t2_idx, t3_idx, t1_views, t2_views, t3_views = item
        label = (
            f"[rank{gpu_rank} {item_idx}/{len(my_items)}]"
            f"[{entry_idx}/{len(all_entries)}]"
            f"[t1={t1_idx},t2={t2_idx},t3={t3_idx}]"
        )
        summary.append(
            run_variant(
                entry,
                t1_idx,
                t2_idx,
                t3_idx,
                t1_views,
                t2_views,
                t3_views,
                config,
                output_root,
                runner,
                label,
            )
        )

    summary_path = output_root / f"run_summary_rank{gpu_rank}.json"
    write_json(summary_path, summary)
    log(f"rank {gpu_rank} done. summary: {summary_path}")


if __name__ == "__main__":
    main()
