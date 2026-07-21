"""Train TemporalVGGTv1 with LOTO cross-validation (DDP version).

Usage:
    torchrun --nproc_per_node=N src/train_ddp.py --config configs/train_model_v1.yaml

Distributes data across N GPUs. Each GPU processes batch_size=1.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, IO

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset.triplet_dataset import LiveTripletDataset, TemporalTripletDataset, _load_images
from dataset.vggt_feature_cache import VGGTFeatureCache
from losses.pointmap_loss import temporal_vggt_pointmap_loss
from losses.cache_loss import cached_feature_loss
from losses.camera_loss import camera_loss_t2q
from loto import build_loto_folds, load_triplets
from models.feature_cache_utils import run_cached_endpoint


# ─── DDP setup / teardown ─────────────────────────────────────────────────────

def setup_ddp() -> tuple[int, int]:
    """Initialize DDP process group. Returns (local_rank, world_size)."""
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    """Destroy the DDP process group."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    return dist.get_rank() == 0


# ─── utilities ────────────────────────────────────────────────────────────────

_log_files: list[IO[str]] = []


def log(msg: str) -> None:
    if not is_main_process():
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in _log_files:
        print(line, file=f, flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required") from exc
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, obj: Any) -> None:
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ─── config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "vggt_output_root": "vggt_output",
    "output_root": "runs/model_v1",
    "vggt_inference_mode": "precomputed",
    "all_triplets_path": "prepared_data/all_triplets.json",
    "vggt_device": "auto",
    "vggt_model_id": "facebook/VGGT-1B",
    "n_views": 8,
    "max_overlap_views": 2,
    "max_variants": None,
    "protocols": ["target_date", "strict"],
    "crops": ["corn", "soybean"],
    "test_date": None,
    "seed": 42,
    "image_preprocess_mode": "pad",
    "conf_threshold": 0.02,
    "feature_cache_root": "/home/ec2-user/workspace/canopy-org/vggt_cache",
    "freeze_point_head": False,
    "model_module": "models.temporal_vggt_v1",
    "model_class": "TemporalVGGTv1",
    "model_kwargs": {},
    "resume": False,
    "epochs": 50,
    "batch_size": 1,
    "num_workers": 2,
    "val_every": 5,
    "grad_clip": 1.0,
    "init_checkpoint": None,
    "optimizer": {
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
    },
    "scheduler": {
        "warmup_epochs": 2,
        "min_lr": 1e-6,
    },
    "loss": {
        "alpha": 0.2,
        "lambda_grad": 1.0,
        "use_gradient_loss": True,
        "pred_conf_clamp_min": 1e-6,
        "pred_conf_clamp_max": 100.0,
        "cache_weight": 0.0,
        "cache_smooth_l1_beta": 0.1,
        "cache_cos_weight": 0.1,
        "cache_normalize": True,
        "cache_patch_only": True,
        "pointmap_weight": 1.0,
        "delta_reg_weight": 0.0,
        "gate_reg_weight": 0.0,
    },
    "teacher": {
        "conf_mask_threshold": 0.2,
        "conf_threshold_type": "quantile",
        "use_weighted_reg_loss": False,
        "weight_clip_min": 0.25,
        "weight_clip_max": 4.0,
    },
    "camera_loss": {
        "lambda_camera": 0.1,
        "gamma": 0.6,
        "weight_trans": 1.0,
        "weight_rot": 1.0,
        "weight_focal": 0.5,
    },
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config.exists():
        cfg.update(read_yaml(args.config))
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.crop is not None:
        cfg["crops"] = args.crop
    if args.test_date is not None:
        cfg["test_date"] = args.test_date
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = args.init_checkpoint
    cfg["triplets_path"] = Path(cfg["triplets_path"])
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["output_root"] = Path(cfg["output_root"])
    cfg["all_triplets_path"] = Path(cfg["all_triplets_path"])
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TemporalVGGTv1 with LOTO (DDP).")
    p.add_argument("--config", type=Path, default=Path("configs/train_model_v1.yaml"))
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--crop", action="append", default=None)
    p.add_argument("--test-date", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    return p.parse_args()


# ─── fold ↔ dataset bridging ──────────────────────────────────────────────────

def _triplet_id(t: dict[str, Any]) -> str:
    return f"{t['left_date']}_{t['middle_date']}_{t['right_date']}_{t['crop']}"


def fold_dataset_indices(
    fold: dict[str, Any],
    dataset: TemporalTripletDataset,
) -> tuple[list[int], list[int]]:
    train_ids = {_triplet_id(t) for t in fold["train_triplets"]}
    val_ids   = {_triplet_id(t) for t in fold["val_triplets"]}
    train_idx = [i for i, e in enumerate(dataset.index) if e["triplet_id"] in train_ids]
    val_idx   = [i for i, e in enumerate(dataset.index) if e["triplet_id"] in val_ids]
    return train_idx, val_idx


# ─── collate ──────────────────────────────────────────────────────────────────

def _collate(samples: list) -> dict:
    out: dict = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if any(v is None for v in vals):
            out[k] = None if all(v is None for v in vals) else vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], dict):
            out[k] = {dk: torch.stack([v[dk] for v in vals]) for dk in vals[0]}
        elif isinstance(vals[0], list) and len(vals[0]) > 0 and isinstance(vals[0][0], torch.Tensor):
            out[k] = [torch.stack([v[i] for v in vals]) for i in range(len(vals[0]))]
        elif isinstance(vals[0], str):
            out[k] = vals
        else:
            out[k] = vals
    return out


# ─── model ──────────────────────────────────────────────────────────────────────

def build_model(cfg: dict[str, Any], device: str, feature_cache=None) -> torch.nn.Module:
    module = importlib.import_module(cfg["model_module"])
    cls = getattr(module, cfg["model_class"])
    model = cls(**cfg.get("model_kwargs", {}))
    model = model.to(device)
    if cfg.get("gradient_checkpointing", False):
        log("  gradient checkpointing enabled")
    model.feature_cache = feature_cache
    if cfg.get("freeze_point_head", False):
        model.freeze_point_head = True
        if hasattr(model, "point_head"):
            for p in model.point_head.parameters():
                p.requires_grad_(False)
            model.point_head = None
        log("  point_head: FROZEN and removed from memory")
    else:
        model.freeze_point_head = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"  model={cfg['model_class']}  trainable={n_train:,}  total={n_total:,}  device={device}")
    return model


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> DDP:
    """Wrap model in DistributedDataParallel."""
    return DDP(model, device_ids=[local_rank], find_unused_parameters=True)


# ─── DataLoader with DistributedSampler ───────────────────────────────────────

def build_loaders(
    train_subset: Subset,
    val_subset: Subset | None,
    cfg: dict[str, Any],
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DistributedSampler, DataLoader | None, DistributedSampler | None]:
    """Build train/val DataLoaders with DistributedSampler."""
    live_mode = cfg.get("vggt_inference_mode", "precomputed") == "live"
    nw = 0 if live_mode else cfg.get("num_workers", 2)

    train_sampler = DistributedSampler(
        train_subset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.get("batch_size", 1),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        collate_fn=_collate,
    )

    val_loader = None
    val_sampler = None
    if val_subset is not None:
        val_sampler = DistributedSampler(
            val_subset, num_replicas=world_size, rank=rank, shuffle=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg.get("batch_size", 1),
            sampler=val_sampler,
            num_workers=nw,
            pin_memory=nw > 0,
            persistent_workers=nw > 0,
            collate_fn=_collate,
        )

    return train_loader, train_sampler, val_loader, val_sampler


# ─── optimizer / LR schedule ──────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-2),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )


def cosine_lr(epoch: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ─── t2 teacher cache ─────────────────────────────────────────────────────────

def select_query_cached_views(
    cached_layers: list[torch.Tensor],
    query_view_indices: Any,
    device: str,
) -> list[torch.Tensor]:
    """Select/reorder cached t2 features to match camera_t2_query view order."""
    if query_view_indices is None:
        return cached_layers
    if torch.is_tensor(query_view_indices):
        indices = query_view_indices.to(device=device, dtype=torch.long)
    else:
        indices = torch.tensor(query_view_indices, device=device, dtype=torch.long)
    if indices.dim() == 1:
        indices = indices.unsqueeze(0)

    selected: list[torch.Tensor] = []
    for layer in cached_layers:
        if indices.max().item() >= layer.shape[1]:
            raise RuntimeError(
                f"query_view_indices max={indices.max().item()} exceeds cached t2 view count={layer.shape[1]}"
            )
        gather_idx = indices[:, :, None, None].expand(-1, -1, layer.shape[2], layer.shape[3])
        selected.append(layer.gather(dim=1, index=gather_idx))
    return selected


def resolve_t2_teacher_cached(
    batch: dict,
    model: torch.nn.Module,
    device: str,
) -> list[torch.Tensor] | None:
    """Load or compute t2 teacher cached layers, caching to disk on first use."""
    if not hasattr(model, "_run_vggt_endpoint") or model.aggregator is None:
        return None
    cached_layers = run_cached_endpoint(
        model,
        batch.get("images_t2"),
        batch["date_t1"].shape[0],
        batch.get("t2_cache_key"),
        "t2",
    )
    if getattr(model, "target_view_cache_requires_query_order", False):
        cached_layers = select_query_cached_views(
            cached_layers, batch.get("query_view_indices"), device
        )
    return cached_layers


# ─── loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    outputs: dict,
    batch: dict,
    device: str,
    cfg: dict[str, Any],
    model: torch.nn.Module = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg    = cfg.get("loss", {})
    loss_dict: dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=device)

    def _missing_batch_value(key: str) -> bool:
        value = batch.get(key)
        return value is None or (
            isinstance(value, (list, tuple)) and any(item is None for item in value)
        )

    # --- Pointmap loss (skipped when point head is frozen / not in outputs) ---
    pointmap_weight = loss_cfg.get("pointmap_weight", 1.0)
    if pointmap_weight > 0 and "pred_points" in outputs:
        if any(_missing_batch_value(k) for k in (
            "target_point_maps_t2",
            "target_point_confidence_t2",
            "target_masks_t2",
        )):
            raise RuntimeError(
                "Pointmap loss is enabled, but t2 point_map/point_confidence "
                "predictions are missing. Generate them or set pointmap_weight: 0."
            )
        pred_points = outputs["pred_points"]
        pred_conf   = outputs["pred_conf"]

        teacher_points = batch["target_point_maps_t2"].to(device)
        teacher_conf   = batch["target_point_confidence_t2"].to(device)
        base_mask      = batch["target_masks_t2"].to(device).bool()

        if teacher_points.shape[-3:-1] != pred_points.shape[-3:-1]:
            B, Q = teacher_points.shape[:2]
            Hp, Wp = pred_points.shape[-3], pred_points.shape[-2]
            teacher_points = F.interpolate(
                teacher_points.view(B * Q, *teacher_points.shape[2:]).permute(0, 3, 1, 2),
                size=(Hp, Wp), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1).contiguous().view(B, Q, Hp, Wp, 3)
            teacher_conf = F.interpolate(
                teacher_conf.view(B * Q, 1, *teacher_conf.shape[2:]),
                size=(Hp, Wp), mode="bilinear", align_corners=False,
            ).view(B, Q, Hp, Wp)
            base_mask = F.interpolate(
                base_mask.view(B * Q, 1, *base_mask.shape[2:]).float(),
                size=(Hp, Wp), mode="nearest",
            ).view(B, Q, Hp, Wp).bool()

        teacher_cfg = cfg.get("teacher", {})
        pm_dict = temporal_vggt_pointmap_loss(
            pred_points=pred_points,
            pred_conf=pred_conf,
            teacher_points=teacher_points,
            teacher_conf=teacher_conf,
            base_mask=base_mask,
            teacher_conf_mask_threshold=teacher_cfg.get("conf_mask_threshold", 0.2),
            teacher_conf_threshold_type=teacher_cfg.get("conf_threshold_type", "quantile"),
            use_teacher_conf_weighted_reg=teacher_cfg.get("use_weighted_reg_loss", False),
            teacher_conf_weight_clip_min=teacher_cfg.get("weight_clip_min", 0.25),
            teacher_conf_weight_clip_max=teacher_cfg.get("weight_clip_max", 4.0),
            alpha=loss_cfg.get("alpha", 0.2),
            lambda_grad=loss_cfg.get("lambda_grad", 1.0),
            use_gradient_loss=loss_cfg.get("use_gradient_loss", True),
            pred_conf_clamp_min=loss_cfg.get("pred_conf_clamp_min", 1e-6),
            pred_conf_clamp_max=loss_cfg.get("pred_conf_clamp_max", 100.0),
        )
        total_loss = total_loss + pointmap_weight * pm_dict["loss_pointmap"]
        loss_dict.update(pm_dict)

    # --- Cached feature loss ---
    cache_weight = loss_cfg.get("cache_weight", 0.0)
    if cache_weight > 0 and "pred_cached_layers" in outputs:
        teacher_cached = resolve_t2_teacher_cached(batch, model, device)
        if teacher_cached is not None:
            cache_dict = cached_feature_loss(
                pred_layers=outputs["pred_cached_layers"],
                teacher_layers=teacher_cached,
                patch_start_idx=5,
                smooth_l1_beta=loss_cfg.get("cache_smooth_l1_beta", 0.1),
                cos_weight=loss_cfg.get("cache_cos_weight", 0.1),
                patch_only=loss_cfg.get("cache_patch_only", True),
                normalize=loss_cfg.get("cache_normalize", True),
            )
            total_loss = total_loss + cache_weight * cache_dict["loss_cache"]
            loss_dict.update(cache_dict)

    # --- Gate regularization ---
    gate_reg_weight = loss_cfg.get("gate_reg_weight", 0.0)
    if gate_reg_weight > 0 and model is not None and hasattr(model, "gates"):
        gate_reg = sum(g.abs().sum() for g in model.gates)
        total_loss = total_loss + gate_reg_weight * gate_reg
        loss_dict["loss_gate_reg"] = gate_reg

    # --- Delta regularization ---
    delta_reg_weight = loss_cfg.get("delta_reg_weight", 0.0)
    if delta_reg_weight > 0 and "pred_cached_layers" in outputs:
        delta_reg = sum(layer.pow(2).mean() for layer in outputs["pred_cached_layers"])
        total_loss = total_loss + delta_reg_weight * delta_reg
        loss_dict["loss_delta_reg"] = delta_reg

    if "pred_pose_enc_list_t2q" in outputs and "pred_points" in outputs:
        if any(_missing_batch_value(k) for k in (
            "target_vggt_extrinsic_t2",
            "target_vggt_intrinsic_t2",
        )):
            raise RuntimeError(
                "Camera loss is enabled by model outputs, but t2 extrinsic/intrinsic "
                "predictions are missing. Generate them or disable camera loss outputs."
            )
        pred_points = outputs["pred_points"]
        cam_cfg = cfg.get("camera_loss", {})
        lambda_camera = cam_cfg.get("lambda_camera", 0.1)
        image_size_hw = (pred_points.shape[-3], pred_points.shape[-2])
        cam_dict = camera_loss_t2q(
            predictions=outputs,
            batch=batch,
            image_size_hw=image_size_hw,
            gamma=cam_cfg.get("gamma", 0.6),
            weight_trans=cam_cfg.get("weight_trans", 1.0),
            weight_rot=cam_cfg.get("weight_rot", 1.0),
            weight_focal=cam_cfg.get("weight_focal", 0.5),
        )
        total_loss = total_loss + lambda_camera * cam_dict["loss_camera"]
        loss_dict.update(cam_dict)

    return total_loss, loss_dict


# ─── metric all-reduce ────────────────────────────────────────────────────────

def reduce_metrics(totals: dict[str, float], n: int, world_size: int) -> dict[str, float]:
    """All-reduce summed metrics and count across ranks, return averaged metrics."""
    keys = sorted(totals.keys())
    vals = [totals[k] for k in keys] + [float(n)]
    t = torch.tensor(vals, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_n = t[-1].item()
    return {k: t[i].item() / max(total_n, 1) for i, k in enumerate(keys)}


# ─── train / eval loops ───────────────────────────────────────────────────────

def _autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _accumulate(totals: dict, metrics: dict) -> None:
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.detach()
        totals[k] = totals.get(k, 0.0) + float(v)


def _total_loss(metrics: dict, lambda_camera: float, cfg: dict = None) -> float:
    loss_cfg = (cfg or {}).get("loss", {})
    total = 0.0
    pm_w = loss_cfg.get("pointmap_weight", 1.0)
    total += pm_w * metrics.get("loss_pointmap", 0.0)
    total += lambda_camera * metrics.get("loss_camera", 0.0)
    cache_w = loss_cfg.get("cache_weight", 0.0)
    total += cache_w * metrics.get("loss_cache", 0.0)
    total += loss_cfg.get("gate_reg_weight", 0.0) * metrics.get("loss_gate_reg", 0.0)
    total += loss_cfg.get("delta_reg_weight", 0.0) * metrics.get("loss_delta_reg", 0.0)
    return total


def train_epoch(
    model: DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cfg: dict[str, Any],
    grad_clip: float,
    world_size: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    show_pbar = is_main_process()
    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True, disable=not show_pbar)
    for batch in pbar:
        optimizer.zero_grad()
        with _autocast(device):
            outputs = model(batch)
        loss, loss_dict = compute_loss(outputs, batch, device, cfg, model=model.module)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip
            )
        optimizer.step()
        _accumulate(totals, loss_dict)
        n += 1
        if show_pbar:
            pbar.set_postfix(loss=f"{loss_dict.get('loss_pointmap', 0):.4f}")
    return reduce_metrics(totals, n, world_size)


def eval_epoch(
    model: DDP,
    loader: DataLoader,
    device: str,
    cfg: dict[str, Any],
    world_size: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    show_pbar = is_main_process()
    with torch.no_grad():
        pbar = tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True, disable=not show_pbar)
        for batch in pbar:
            with _autocast(device):
                outputs = model(batch)
            _, loss_dict = compute_loss(outputs, batch, device, cfg, model=model.module)
            _accumulate(totals, loss_dict)
            n += 1
            if show_pbar:
                pbar.set_postfix(loss=f"{loss_dict.get('loss_pointmap', 0):.4f}")
    return reduce_metrics(totals, n, world_size)


# ─── checkpoint ───────────────────────────────────────────────────────────────

def _save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    best_epoch: int,
    patience_count: int,
    history: list,
) -> None:
    """Only rank 0 saves; all ranks barrier after."""
    if is_main_process():
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "history": history,
        }, output_dir / "last_checkpoint.pt")
        torch.save(model.module.state_dict(), output_dir / "last_model.pt")
    dist.barrier()


def _save_best(output_dir: Path, model: DDP) -> None:
    if is_main_process():
        torch.save(model.module.state_dict(), output_dir / "best_model.pt")
    dist.barrier()


# ─── fold training ────────────────────────────────────────────────────────────

def train_fold(
    fold: dict[str, Any],
    dataset: Dataset,
    cfg: dict[str, Any],
    device: str,
    local_rank: int,
    world_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    fold_id = fold["fold_id"]
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    fold_log = None
    if is_main_process():
        fold_log = (output_dir / "train.log").open("w", buffering=1)
        _log_files.append(fold_log)

    try:
        return _train_fold_inner(fold, fold_id, dataset, cfg, device, local_rank, world_size, output_dir)
    finally:
        if fold_log:
            _log_files.remove(fold_log)
            fold_log.close()


def _train_fold_inner(
    fold: dict[str, Any],
    fold_id: str,
    dataset: Dataset,
    cfg: dict[str, Any],
    device: str,
    local_rank: int,
    world_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    train_idx, val_idx = fold_dataset_indices(fold, dataset)

    if not train_idx:
        log(f"  fold={fold_id}: no training samples, skipping")
        return {"fold_id": fold_id, "status": "skipped_no_data"}

    log(f"  fold={fold_id}: train_variants={len(train_idx)}  val_variants={len(val_idx)}")

    feature_cache = dataset.feature_cache if hasattr(dataset, "feature_cache") else None
    model = build_model(cfg, device, feature_cache=feature_cache)
    optimizer = build_optimizer(model, cfg)

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx) if val_idx else None

    train_loader, train_sampler, val_loader, val_sampler = build_loaders(
        train_subset, val_subset, cfg, world_size, dist.get_rank(),
    )

    sched_cfg    = cfg.get("scheduler", {})
    base_lr      = cfg.get("optimizer", {}).get("lr", 1e-4)
    min_lr       = sched_cfg.get("min_lr", 1e-6)
    warmup       = sched_cfg.get("warmup_epochs", 2)
    total_epochs = cfg.get("epochs", 50)
    grad_clip    = cfg.get("grad_clip", 1.0)
    val_every    = cfg.get("val_every", 5)

    es_cfg        = cfg.get("early_stopping", {})
    es_patience   = es_cfg.get("patience", None)
    es_min_delta  = es_cfg.get("min_delta", 1e-4)

    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    history: list[dict] = []
    start_epoch = 1

    # Resume from checkpoint only when explicitly enabled.
    resume_path = output_dir / "last_checkpoint.pt"
    resume_enabled = cfg.get("resume", False)
    if resume_enabled and resume_path.exists():
        log(f"  resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        best_epoch     = ckpt.get("best_epoch", 0)
        patience_count = ckpt.get("patience_count", 0)
        history        = ckpt.get("history", [])
        log(f"  resumed at epoch {start_epoch}/{total_epochs}  best_val={best_val_loss:.6f} (ep {best_epoch})")
    else:
        if resume_path.exists() and not resume_enabled:
            log(f"  resume disabled; ignoring existing checkpoint {resume_path}")
        ckpt_path = cfg.get("init_checkpoint")
        if ckpt_path is not None:
            state = torch.load(Path(ckpt_path), map_location=device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            log(f"  init_checkpoint: loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    # Wrap in DDP after loading weights
    model = wrap_ddp(model, local_rank)

    for epoch in range(start_epoch, total_epochs + 1):
        lr = cosine_lr(epoch - 1, warmup, total_epochs, base_lr, min_lr)
        set_lr(optimizer, lr)
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(model, train_loader, optimizer, device, cfg, grad_clip, world_size)
        row = {"epoch": epoch, "lr": lr,
               **{f"train_{k}": v for k, v in train_metrics.items()}}

        if val_loader and epoch % val_every == 0:
            if val_sampler:
                val_sampler.set_epoch(epoch)
            val_metrics = eval_epoch(model, val_loader, device, cfg, world_size)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

            lambda_camera = cfg.get("camera_loss", {}).get("lambda_camera", 0.1)
            cache_weight = cfg.get("loss", {}).get("cache_weight", 0.0)
            if cache_weight > 0 and "loss_cache" in val_metrics:
                val_loss = val_metrics["loss_cache"]
            else:
                val_loss = (
                    val_metrics.get("loss_point_reg", float("inf"))
                    + lambda_camera * val_metrics.get("loss_camera", 0.0)
                )

            if val_loss < best_val_loss - es_min_delta:
                best_val_loss  = val_loss
                best_epoch     = epoch
                patience_count = 0
                _save_best(output_dir, model)
            else:
                patience_count += 1

            patience_str = f"  patience={patience_count}/{es_patience}" if es_patience else ""
            log(f"  epoch={epoch}/{total_epochs} lr={lr:.2e}"
                f"  total={_total_loss(train_metrics, lambda_camera, cfg):.6f}"
                + (f"  pm={train_metrics.get('loss_pointmap', 0):.6f}" if "loss_pointmap" in train_metrics else "")
                + (f"  cache={train_metrics.get('loss_cache', 0):.6f}" if "loss_cache" in train_metrics else "")
                + (f"  l1={train_metrics.get('loss_cache_l1', 0):.6f}" if "loss_cache_l1" in train_metrics else "")
                + (f"  cos={train_metrics.get('loss_cache_cos', 0):.6f}" if "loss_cache_cos" in train_metrics else "")
                + (f"  delta={train_metrics.get('loss_delta_reg', 0):.6f}" if "loss_delta_reg" in train_metrics else "")
                + (f"  cam={train_metrics.get('loss_camera', 0):.6f}" if "loss_camera" in train_metrics else "")
                + f"  | val_crit={val_loss:.6f}"
                + (f"  val_pm={val_metrics.get('loss_pointmap', 0):.6f}" if "loss_pointmap" in val_metrics else "")
                + (f"  val_cache={val_metrics.get('loss_cache', 0):.6f}" if "loss_cache" in val_metrics else "")
                + (f"  val_cam={val_metrics.get('loss_camera', 0):.6f}" if "loss_camera" in val_metrics else "")
                + f"  best={best_val_loss:.6f} (ep {best_epoch})"
                + patience_str)

            if es_patience and patience_count >= es_patience:
                log(f"  early stopping: no improvement for {es_patience} val checks")
                history.append(row)
                _save_checkpoint(output_dir, epoch, model, optimizer,
                                 best_val_loss, best_epoch, patience_count, history)
                break
        else:
            lambda_camera = cfg.get("camera_loss", {}).get("lambda_camera", 0.1)
            log(f"  epoch={epoch}/{total_epochs} lr={lr:.2e}"
                f"  total={_total_loss(train_metrics, lambda_camera, cfg):.6f}"
                + (f"  pm={train_metrics.get('loss_pointmap', 0):.6f}" if "loss_pointmap" in train_metrics else "")
                + (f"  cache={train_metrics.get('loss_cache', 0):.6f}" if "loss_cache" in train_metrics else "")
                + (f"  l1={train_metrics.get('loss_cache_l1', 0):.6f}" if "loss_cache_l1" in train_metrics else "")
                + (f"  cos={train_metrics.get('loss_cache_cos', 0):.6f}" if "loss_cache_cos" in train_metrics else "")
                + (f"  delta={train_metrics.get('loss_delta_reg', 0):.6f}" if "loss_delta_reg" in train_metrics else "")
                + (f"  cam={train_metrics.get('loss_camera', 0):.6f}" if "loss_camera" in train_metrics else ""))

        history.append(row)
        _save_checkpoint(output_dir, epoch, model, optimizer,
                         best_val_loss, best_epoch, patience_count, history)

    if not val_loader:
        _save_best(output_dir, model)
        best_epoch = total_epochs

    write_json(output_dir / "training_history.json", history)

    result = {
        "fold_id":        fold_id,
        "status":         "completed",
        "best_epoch":     best_epoch,
        "best_val_loss":  best_val_loss if val_loader else None,
        "stopped_early":  es_patience is not None and patience_count >= es_patience,
        "train_variants": len(train_idx),
        "val_variants":   len(val_idx),
        "checkpoint":     str((output_dir / "best_model.pt").resolve()),
    }
    write_json(output_dir / "fold_result.json", result)
    return result


# ─── main ─────────────────────────────────────────────────────────────────────

def main(local_rank: int, world_size: int) -> None:
    device = f"cuda:{local_rank}"
    args = parse_args()
    cfg  = build_config(args)

    log(f"DDP: world_size={world_size}  local_rank={local_rank}  device={device}")
    log(f"output={cfg['output_root']}")

    num_query_views = cfg.get("model_kwargs", {}).get("num_query_views", 1)
    live_mode = cfg.get("vggt_inference_mode", "precomputed") == "live"

    cache_root = cfg.get("feature_cache_root")
    feature_cache = VGGTFeatureCache.from_config(cache_root, cfg) if cache_root else None
    if feature_cache:
        log(f"VGGT feature cache: {cache_root} namespace={feature_cache.namespace}")

    if live_mode:
        from vggt_pipeline.execute_vggt import get_vggt_runner
        vggt_device = cfg.get("vggt_device", "auto")
        vggt_model_id = cfg.get("vggt_model_id", "facebook/VGGT-1B")
        log(f"live VGGT mode: model_id={vggt_model_id} vggt_device={vggt_device}")
        vggt_runner = get_vggt_runner(model_id=vggt_model_id, device=vggt_device, use_cache=True)
        all_triplets = json.loads(cfg["all_triplets_path"].read_text())
        dataset: Dataset = LiveTripletDataset(
            all_triplets=all_triplets,
            vggt_runner=vggt_runner,
            n_views=cfg.get("n_views", 8),
            max_overlap_views=cfg.get("max_overlap_views", 2),
            max_variants=cfg.get("max_variants"),
            image_preprocess_mode=cfg.get("image_preprocess_mode", "pad"),
            conf_threshold=cfg.get("conf_threshold", 0.02),
            num_query_views=num_query_views,
            seed=cfg.get("seed", 42),
        )
        log(f"Dataset: {len(dataset)} pairs (live mode)")
    else:
        log("Scanning vggt_output_root for completed variants...")
        dataset = TemporalTripletDataset(
            vggt_output_root=cfg["vggt_output_root"],
            image_preprocess_mode=cfg.get("image_preprocess_mode", "pad"),
            conf_threshold=cfg.get("conf_threshold", 0.02),
            num_query_views=num_query_views,
            seed=cfg.get("seed", 42),
            feature_cache=feature_cache,
        )
        log(f"Dataset: {len(dataset)} completed (triplet, variant) pairs")

    # Feature cache warmup — all ranks participate (sharded by rank)
    if feature_cache is not None and not live_mode:
        from train import warmup_feature_cache_sharded
        warmup_model = build_model(cfg, device, feature_cache=feature_cache)
        warmup_feature_cache_sharded(
            warmup_model, dataset, device,
            cfg.get("image_preprocess_mode", "pad"),
            rank=dist.get_rank(), world_size=world_size,
        )
        del warmup_model
        torch.cuda.empty_cache()
        dist.barrier()

    triplets = load_triplets(cfg["triplets_path"])

    if is_main_process():
        cfg["output_root"].mkdir(parents=True, exist_ok=True)
    write_json(cfg["output_root"] / "train_config.json", cfg)
    dist.barrier()

    main_log = None
    if is_main_process():
        main_log = (cfg["output_root"] / "train.log").open("w", buffering=1)
        _log_files.append(main_log)

    summary: list[dict] = []
    for protocol in cfg["protocols"]:
        for crop in cfg["crops"]:
            folds = build_loto_folds(triplets, crop, protocol)
            for fold in folds:
                if cfg["test_date"] and fold["test_date"] != cfg["test_date"]:
                    continue
                fold_dir = cfg["output_root"] / protocol / fold["fold_id"]
                log(f"--- fold={fold['fold_id']} protocol={protocol} "
                    f"crop={crop} test_date={fold['test_date']} ---")
                result = train_fold(fold, dataset, cfg, device, local_rank, world_size, fold_dir)
                summary.append(result)

    write_json(cfg["output_root"] / "train_summary.json", summary)
    log(f"Done. Summary: {cfg['output_root'] / 'train_summary.json'}")

    if main_log:
        _log_files.remove(main_log)
        main_log.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    local_rank, world_size = setup_ddp()
    try:
        main(local_rank, world_size)
    finally:
        cleanup_ddp()
