"""Evaluate TRPM-Small and baselines on LOTO test folds — multi-process version.

Distributes triplets within each fold across processes using LOCAL_RANK / WORLD_SIZE.
No NCCL / process group needed — each process works fully independently.
Rank 0 waits for all shard files then assembles the final summary.

Launch with torchrun (handles process spawning):

    torchrun --nproc_per_node=N src/trpm/evaluate_ddp.py \\
        --config configs/train_trpm_small.yaml \\
        --runs-root runs/trpm_small \\
        --output-root evaluation/trpm_small

    # with cloud saving:
    torchrun --nproc_per_node=N src/trpm/evaluate_ddp.py \\
        --config configs/train_trpm_small.yaml \\
        --runs-root runs/trpm_small \\
        --output-root evaluation/trpm_small \\
        --save-clouds
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trpm.evaluate import (
    build_config,
    evaluate_fold,
    parse_args,
    write_json,
    write_markdown_report,
    aggregate_metrics,
    BASELINES,
    _load_model_class,
)
from loto import build_all_folds


def log(msg: str, rank: int) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][rank{rank}] {msg}", flush=True)


def main() -> None:
    rank       = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device     = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    args = parse_args()
    cfg  = build_config(args)
    cfg["device"] = device

    all_folds   = build_all_folds(cfg["triplets_path"])
    output_root = cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect fold jobs (same filtering as before).
    fold_jobs: list[tuple[dict[str, Any], Path | None]] = []
    for protocol in cfg["protocols"]:
        for fold in all_folds.get(protocol, []):
            if fold["crop"] not in cfg["crops"]:
                continue
            if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                continue
            if not fold["test_triplets"]:
                continue
            checkpoint = cfg["runs_root"] / protocol / fold["fold_id"] / "best_model.pt"
            fold_jobs.append((fold, checkpoint if checkpoint.exists() else None))

    for fold, checkpoint in fold_jobs:
        fold_id  = fold["fold_id"]
        protocol = fold["protocol"]

        # Slice this rank's triplets round-robin.
        all_triplets  = fold["test_triplets"]
        my_triplets   = [t for i, t in enumerate(all_triplets) if i % world_size == rank]
        log(f"fold={fold_id} assigned {len(my_triplets)}/{len(all_triplets)} triplets", rank)

        model: torch.nn.Module | None = None
        if not cfg.get("baselines_only") and checkpoint is not None:
            model_cls = _load_model_class(cfg.get("model_class", "trpm.model.TRPMSmall"))
            model = model_cls(**cfg.get("model_kwargs", {})).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()

        # Run evaluate_fold on the sliced fold.
        sliced_fold = {**fold, "test_triplets": my_triplets, "n_test": len(my_triplets)}
        fold_output_dir = output_root / protocol / fold_id
        result = evaluate_fold(sliced_fold, model, cfg, device, fold_output_dir)

        # Write per-rank shard with triplet_rows included for merging.
        write_json(fold_output_dir / f"shard_{rank:02d}.json", result)
        log(f"fold={fold_id} shard written", rank)

    # Rank 0 waits for all shards, merges rows, aggregates, writes summary.
    if rank == 0:
        for fold, _ in fold_jobs:
            fold_id         = fold["fold_id"]
            protocol        = fold["protocol"]
            fold_output_dir = output_root / protocol / fold_id

            shard_paths = [fold_output_dir / f"shard_{r:02d}.json" for r in range(world_size)]
            log(f"fold={fold_id} waiting for {world_size} shards...", rank)
            while True:
                if all(p.exists() for p in shard_paths):
                    break
                time.sleep(5)

            # Merge triplet_rows from all shards, re-aggregate.
            all_rows: list[dict[str, Any]] = []
            for p in shard_paths:
                all_rows.extend(json.loads(p.read_text()).get("triplet_rows", []))

            method_keys = list(BASELINES) + (["trpm"] if any("trpm" in r for r in all_rows) else [])
            aggregated  = aggregate_metrics(all_rows, method_keys)

            merged = {
                "fold_id":   fold["fold_id"],
                "crop":      fold["crop"],
                "protocol":  fold["protocol"],
                "test_date": fold["test_date"],
                "n_test":    len(all_rows),
                "aggregated": aggregated,
                "triplet_rows": all_rows,
            }
            write_json(fold_output_dir / "eval_result.json", merged)

        all_results = []
        for fold, _ in fold_jobs:
            p = output_root / fold["protocol"] / fold["fold_id"] / "eval_result.json"
            r = json.loads(p.read_text())
            all_results.append({k: v for k, v in r.items() if k != "triplet_rows"})

        write_json(output_root / "eval_summary.json", all_results)
        log(f"done. summary → {output_root / 'eval_summary.json'}", rank)

        write_markdown_report(output_root / "eval_report.md", all_results)
        log(f"report → {output_root / 'eval_report.md'}", rank)


if __name__ == "__main__":
    main()

