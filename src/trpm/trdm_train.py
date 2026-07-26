"""Train TRDM with leave-one-date-out folds."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loto import build_loto_folds, load_triplets
from trpm.trdm_dataset import TRDMDepthDataset, trdm_collate
from trpm.trdm_loss import trdm_loss


DEFAULT_CONFIG: dict[str, Any] = {
    "model_class": "trpm.trdm_model.TRDM",
    "vggt_output_root": "vggt_outputs/t1t2_paired_v16_o8",
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "output_root": "runs/trdm",
    "protocols": ["strict"],
    "crops": ["corn"],
    "test_date": None,
    "val_date": None,
    "seed": 42,
    "epochs": 100,
    "batch_size": 8,
    "num_workers": 4,
    "val_every": 1,
    "device": "auto",
    "grad_clip": 1.0,
    "image_preprocess_mode": "pad",
    "model_kwargs": {},
    "optimizer": {"lr": 2e-4, "weight_decay": 1e-4, "betas": [0.9, 0.999]},
    "scheduler": {"warmup_epochs": 5, "min_lr": 1e-6},
    "early_stopping": {"patience": 15, "min_delta": 1e-4},
    "loss": {
        "log_depth_weight": 1.0,
        "depth_gradient_weight": 0.2,
        "depth_chamfer_weight": 0.01,
        "residual_weight": 0.01,
        "gate_weight": 0.001,
        "conf_threshold": 0.02,
        "depth_smooth_l1_beta": 0.05,
        "chamfer_num_points": 4096,
        "eps": 1e-4,
    },
}


LOSS_METRIC_KEYS = [
    "loss",
    "loss_log_depth",
    "loss_depth_gradient",
    "loss_depth_chamfer",
    "loss_residual",
    "loss_gate",
    "valid_ratio",
    "warp_valid_ratio",
    "mean_gate",
    "mean_abs_delta_logD",
    "mean_D2_hat",
    "mean_D2",
]


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, int]:
    if not is_distributed():
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def log(message: str) -> None:
    if not is_main_process():
        return
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRDM.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_trdm.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--test-date", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config.exists():
        loaded = read_yaml(args.config)
        cfg.update(loaded)
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.device is not None:
        cfg["device"] = args.device
    if args.test_date is not None:
        cfg["test_date"] = args.test_date
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["triplets_path"] = Path(cfg["triplets_path"])
    cfg["output_root"] = Path(cfg["output_root"])
    return cfg


def choose_device(device: str, local_rank: int = 0) -> str:
    if is_distributed():
        return f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def cosine_lr(epoch_zero: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if warmup > 0 and epoch_zero < warmup:
        return base_lr * float(epoch_zero + 1) / float(warmup)
    if total <= warmup:
        return base_lr
    progress = (epoch_zero - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def triplet_id_from_fold_entry(entry: dict[str, Any]) -> str:
    return f"{entry['left_date']}_{entry['middle_date']}_{entry['right_date']}_{entry['crop']}"


def fold_indices(fold: dict[str, Any], dataset: TRDMDepthDataset) -> tuple[list[int], list[int]]:
    train_ids = {triplet_id_from_fold_entry(entry) for entry in fold["train_triplets"]}
    val_ids = {triplet_id_from_fold_entry(entry) for entry in fold["val_triplets"]}
    train_idx = [i for i, item in enumerate(dataset.index) if item["triplet_id"] in train_ids]
    val_idx = [i for i, item in enumerate(dataset.index) if item["triplet_id"] in val_ids]
    return train_idx, val_idx


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def aggregate_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def reduce_metrics(total: dict[str, float], count: int, device: str) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return aggregate_metrics(total, count)
    keys = LOSS_METRIC_KEYS
    values = [total.get(key, 0.0) for key in keys] + [float(count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced_count = max(float(tensor[-1].item()), 1.0)
    return {key: float(tensor[idx].item() / reduced_count) for idx, key in enumerate(keys)}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def should_show_progress() -> bool:
    return is_main_process()


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_cfg: dict[str, Any],
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False, disable=not should_show_progress()):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = trdm_loss(outputs, batch, loss_cfg)
        losses["loss"].backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        count += 1
    return reduce_metrics(totals, count, device)


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="val", leave=False, disable=not should_show_progress()):
        batch = move_batch(batch, device)
        outputs = model(batch)
        losses = trdm_loss(outputs, batch, loss_cfg)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        count += 1
    return reduce_metrics(totals, count, device)


def train_fold(
    fold: dict[str, Any],
    dataset: TRDMDepthDataset,
    cfg: dict[str, Any],
    device: str,
    output_dir: Path,
    distributed: bool = False,
) -> None:
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    train_idx, val_idx = fold_indices(fold, dataset)
    if not train_idx:
        log(f"skip {fold['fold_id']}: no train samples")
        return
    log(f"fold={fold['fold_id']} train={len(train_idx)} val={len(val_idx)}")

    model_cls = load_model_class(cfg["model_class"])
    model = model_cls(**cfg.get("model_kwargs", {})).to(device)
    if distributed:
        ddp_kwargs = {}
        if device.startswith("cuda"):
            ddp_kwargs = {"device_ids": [torch.device(device).index], "output_device": torch.device(device).index}
        model = DDP(model, **ddp_kwargs)
    log(f"params={sum(param.numel() for param in unwrap_model(model).parameters()):,}")

    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    train_subset = Subset(dataset, train_idx)
    rank = get_rank()
    world_size = dist.get_world_size() if distributed else 1
    val_rank_idx = val_idx[rank::world_size] if distributed else val_idx
    val_subset = Subset(dataset, val_rank_idx) if val_idx else None
    train_sampler = DistributedSampler(train_subset, shuffle=True) if distributed else None

    loader_kwargs = {
        "batch_size": cfg.get("batch_size", 8),
        "num_workers": cfg.get("num_workers", 4),
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "collate_fn": trdm_collate,
        "pin_memory": device.startswith("cuda"),
    }
    train_loader = DataLoader(train_subset, **loader_kwargs)
    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg.get("batch_size", 8),
            num_workers=cfg.get("num_workers", 4),
            shuffle=False,
            collate_fn=trdm_collate,
            pin_memory=device.startswith("cuda"),
        )

    total_epochs = cfg.get("epochs", 100)
    sched_cfg = cfg.get("scheduler", {})
    warmup = sched_cfg.get("warmup_epochs", 5)
    min_lr = sched_cfg.get("min_lr", 1e-6)
    base_lr = opt_cfg.get("lr", 2e-4)
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg.setdefault("conf_threshold", cfg.get("conf_threshold", 0.02))
    best_val = float("inf")
    best_epoch = 0
    history = []
    early_cfg = cfg.get("early_stopping", {})
    patience = early_cfg.get("patience", 15)
    min_delta = early_cfg.get("min_delta", 1e-4)
    stale_epochs = 0

    for epoch in range(1, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = cosine_lr(epoch - 1, warmup, total_epochs, base_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_cfg,
            cfg.get("grad_clip", 1.0),
        )
        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_metrics.items()}}
        if val_loader is not None and epoch % cfg.get("val_every", 1) == 0:
            val_metrics = eval_epoch(model, val_loader, device, loss_cfg)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            val_loss = val_metrics.get("loss", float("inf"))
            improved = val_loss < best_val - min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                stale_epochs = 0
                if is_main_process():
                    torch.save(unwrap_model(model).state_dict(), output_dir / "best_model.pt")
            else:
                stale_epochs += 1
            log(
                f"epoch={epoch}/{total_epochs} lr={lr:.2e} "
                f"train={train_metrics.get('loss', 0):.4f} "
                f"val={val_loss:.4f} "
                f"valid={val_metrics.get('valid_ratio', 0):.3f} "
                f"warp={val_metrics.get('warp_valid_ratio', 0):.3f} "
                f"gate={val_metrics.get('mean_gate', 0):.3f} "
                f"best={best_val:.4f}@{best_epoch}"
            )
            if patience and stale_epochs >= patience:
                log(f"early stopping after {stale_epochs} stale validation checks")
                if distributed:
                    dist.barrier()
                break
        else:
            log(
                f"epoch={epoch}/{total_epochs} lr={lr:.2e} "
                f"train={train_metrics.get('loss', 0):.4f} "
                f"valid={train_metrics.get('valid_ratio', 0):.3f} "
                f"warp={train_metrics.get('warp_valid_ratio', 0):.3f}"
            )
        history.append(row)
        if is_main_process():
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                output_dir / "last_checkpoint.pt",
            )
    if is_main_process():
        if best_epoch == 0:
            torch.save(unwrap_model(model).state_dict(), output_dir / "best_model.pt")
        write_json(output_dir / "training_history.json", history)
    if distributed:
        dist.barrier()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    local_rank, rank, world_size = setup_distributed()
    try:
        torch.manual_seed(cfg.get("seed", 42) + rank)
        device = choose_device(cfg.get("device", "auto"), local_rank)
        if is_main_process():
            cfg["output_root"].mkdir(parents=True, exist_ok=True)
            write_json(cfg["output_root"] / "train_config.json", cfg)
        if is_distributed():
            dist.barrier()
        dataset = TRDMDepthDataset(
            cfg["vggt_output_root"],
            preprocess_mode=cfg.get("image_preprocess_mode", "pad"),
        )
        log(f"dataset samples={len(dataset)} root={cfg['vggt_output_root']} world_size={world_size}")
        triplets = load_triplets(cfg["triplets_path"])
        for protocol in cfg["protocols"]:
            for crop in cfg["crops"]:
                folds = build_loto_folds(triplets, crop, protocol, cfg.get("val_date"))
                for fold in folds:
                    if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                        continue
                    fold_dir = cfg["output_root"] / protocol / fold["fold_id"]
                    train_fold(fold, dataset, cfg, device, fold_dir, distributed=is_distributed())
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
