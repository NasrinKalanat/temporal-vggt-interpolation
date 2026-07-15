"""Train TemporalVGGTv1 with LOTO cross-validation (multi-GPU via DeepSpeed).

Usage:
    deepspeed src/train_dist.py \
        --deepspeed_config configs/deepspeed_config.json \
        --config configs/train_residual_endpoint_adaln.yaml

Builds LOTO folds from triplets_path using loto.py, then for each
(protocol, crop, [test_date]) fold:
  - Filters TemporalTripletDataset index by the fold's train/val triplet IDs
  - Trains with DeepSpeed ZeRO-2 + AdamW + cosine-warmup LR
  - Validates every val_every epochs; saves best checkpoint by val loss

Use --test-date to train a single fold instead of all folds.
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
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import deepspeed

from dataset.triplet_dataset import LiveTripletDataset, TemporalTripletDataset, _load_images
from dataset.vggt_feature_cache import VGGTFeatureCache
from losses.pointmap_loss import temporal_vggt_pointmap_loss
from losses.camera_loss import camera_loss_t2q
from loto import build_loto_folds, load_triplets


# ─── utilities ────────────────────────────────────────────────────────────────

_log_files: list[IO[str]] = []
_is_main: bool = True  # set to False on non-rank-0 processes


def log(msg: str) -> None:
    if not _is_main:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ─── config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "vggt_output_root": "vggt_output",
    "output_root": "runs/model_v1",
    # live VGGT inference
    "vggt_inference_mode": "precomputed",   # "precomputed" | "live"
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
    "feature_cache_root": "/home/ec2-user/workspace/canopy-org/vggt_cache",  # null to disable
    "model_module": "models.temporal_vggt_v1",
    "model_class": "TemporalVGGTv1",
    "model_kwargs": {},
    "epochs": 50,
    "batch_size": 1,
    "num_workers": 2,
    "val_every": 5,
    "device": "cuda:0",
    "grad_clip": 1.0,
    "init_checkpoint": None,        # path to a .pt state_dict to initialize from
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
    if args.device is not None:
        cfg["device"] = args.device
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = args.init_checkpoint
    cfg["triplets_path"] = Path(cfg["triplets_path"])
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["output_root"] = Path(cfg["output_root"])
    cfg["all_triplets_path"] = Path(cfg["all_triplets_path"])
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TemporalVGGTv1 with LOTO cross-validation.")
    p.add_argument("--config", type=Path, default=Path("configs/train_model_v1.yaml"))
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--crop", action="append", default=None,
                   help="Crop to train (repeatable). Overrides config crops.")
    p.add_argument("--test-date", default=None,
                   help="Train only the fold for this test date (e.g. 20230822).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Path to a saved state_dict (.pt) to initialize model weights before training.")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank passed by DeepSpeed launcher.")
    p.add_argument("--deepspeed_config", type=Path,
                   default=Path("configs/deepspeed_config.json"),
                   help="Path to DeepSpeed JSON config.")
    return p.parse_args()


# ─── fold ↔ dataset bridging ──────────────────────────────────────────────────

def _triplet_id(t: dict[str, Any]) -> str:
    """Convert a loto triplet dict to the TemporalTripletDataset triplet_id format."""
    return f"{t['left_date']}_{t['middle_date']}_{t['right_date']}_{t['crop']}"


def fold_dataset_indices(
    fold: dict[str, Any],
    dataset: TemporalTripletDataset,
) -> tuple[list[int], list[int]]:
    """Return train and val index lists into dataset for this LOTO fold."""
    train_ids = {_triplet_id(t) for t in fold["train_triplets"]}
    val_ids   = {_triplet_id(t) for t in fold["val_triplets"]}

    train_idx = [i for i, e in enumerate(dataset.index) if e["triplet_id"] in train_ids]
    val_idx   = [i for i, e in enumerate(dataset.index) if e["triplet_id"] in val_ids]

    return train_idx, val_idx


# ─── model ────────────────────────────────────────────────────────────────────

def build_model(cfg: dict[str, Any], device: str, feature_cache=None) -> torch.nn.Module:
    module = importlib.import_module(cfg["model_module"])
    cls = getattr(module, cfg["model_class"])
    model = cls(**cfg.get("model_kwargs", {}))
    # Second .to(device) after __init__ in case VGGT sub-modules registered
    # buffers or re-instantiated layers on CPU during from_pretrained.
    model = model.to(device)
    actual_device = next(model.parameters()).device
    expected = torch.device(device)
    device_ok = (
        actual_device.type == expected.type
        and (expected.index is None or actual_device.index == expected.index)
    )
    if not device_ok:
        raise RuntimeError(
            f"Model is on {actual_device} after build but expected {device}. "
            "Check that CUDA is available and device string is correct."
        )
    if cfg.get("gradient_checkpointing", False):
        log("  gradient checkpointing enabled")
    # init_checkpoint is NOT loaded here — callers load it explicitly,
    # only when there is no last_checkpoint.pt to resume from.
    model.feature_cache = feature_cache
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"  model={cfg['model_class']}  trainable={n_train:,}  total={n_total:,}  device={actual_device}")
    return model


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


# ─── loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    outputs: dict,
    batch: dict,
    device: str,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_points = outputs["pred_points"]   # [B, Q, H, W, 3]
    pred_conf   = outputs["pred_conf"]     # [B, Q, H, W]

    teacher_points = batch["target_point_maps_t2"].to(device)         # [B, Q, H, W, 3]
    teacher_conf   = batch["target_point_confidence_t2"].to(device)   # [B, Q, H, W]
    base_mask      = batch["target_masks_t2"].to(device).bool()       # [B, Q, H, W]

    # Align spatial resolution if DPTHead output differs from teacher size.
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

    loss_cfg    = cfg.get("loss", {})
    teacher_cfg = cfg.get("teacher", {})

    loss_dict = temporal_vggt_pointmap_loss(
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

    total_loss = loss_dict["loss_pointmap"]

    if "pred_pose_enc_list_t2q" in outputs:
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


# ─── train / eval loops ───────────────────────────────────────────────────────

def _collate(samples: list) -> dict:
    """Collate sample dicts while preserving mixed cached/uncached images."""
    out: dict = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if any(v is None for v in vals):
            out[k] = None if all(v is None for v in vals) else vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], dict):
            out[k] = {dk: torch.stack([v[dk] for v in vals]) for dk in vals[0]}
        elif isinstance(vals[0], str):
            out[k] = vals
        else:
            out[k] = vals
    return out


def _autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _accumulate(totals: dict, metrics: dict) -> None:
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.detach()
        totals[k] = totals.get(k, 0.0) + float(v)


def _total_loss(metrics: dict, lambda_camera: float) -> float:
    return metrics.get("loss_pointmap", 0.0) + lambda_camera * metrics.get("loss_camera", 0.0)


def _save_checkpoint(
    output_dir: Path,
    epoch: int,
    model_engine,
    optimizer,
    best_val_loss: float,
    best_epoch: int,
    patience_count: int,
    history: list,
) -> None:
    if not _is_main:
        return
    model_state = model_engine.module.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "history": history,
    }, output_dir / "last_checkpoint.pt")
    torch.save(model_state, output_dir / "last_model.pt")


def _check_weights_match_checkpoint(
    model: torch.nn.Module,
    ckpt_state: dict,
    max_params: int = 5,
) -> None:
    """Verify that trainable model parameters match the checkpoint state dict.

    Samples up to max_params trainable parameters, computes max absolute
    difference between model and checkpoint values, and logs the result.
    Logs an explicit WARNING if any mismatch exceeds floating-point rounding.
    """
    mismatches: list[str] = []
    matches:    list[str] = []
    checked = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in ckpt_state:
            continue
        ckpt_tensor = ckpt_state[name].to(param.device, param.dtype)
        max_diff = (param.data - ckpt_tensor).abs().max().item()
        entry = f"{name}  max_diff={max_diff:.2e}  norm={param.data.norm():.4f}"
        if max_diff > 1e-5:
            mismatches.append(entry)
        else:
            matches.append(entry)
        checked += 1
        if checked >= max_params:
            break
    if not checked:
        log("  weight check: no trainable keys shared with checkpoint (all missing?)")
        return
    for m in matches:
        log(f"  weight check OK : {m}")
    for m in mismatches:
        log(f"  weight check MISMATCH: {m}")
    if mismatches:
        log(f"  WARNING: {len(mismatches)} parameter(s) do not match checkpoint — weights may have been overwritten")
    else:
        log(f"  weight check: all {checked} sampled trainable params match checkpoint")


# ─── feature cache warmup ─────────────────────────────────────────────────────

def warmup_feature_cache(
    model: torch.nn.Module,
    dataset: TemporalTripletDataset,
    device: str,
    image_preprocess_mode: str,
) -> None:
    """Pre-populate the frozen VGGT feature cache for all unique endpoints in dataset.

    Collects unique (cache_key, date_dir) pairs across the full dataset index,
    filters to those not yet on disk, and runs the frozen aggregator on each
    with a tqdm progress bar. Safe to call from multiple processes simultaneously.
    """
    cache = getattr(model, "feature_cache", None)
    if cache is None or not hasattr(dataset, "index"):
        return

    # Collect unique missing entries: key → date_dir
    missing: dict[str, Path] = {}
    for entry in dataset.index:
        vdir: Path = entry["variant_dir"]
        for endpoint in ("t1", "t3"):
            date_dir = vdir / endpoint
            key = cache.key(date_dir)
            if key and key not in missing and not cache.exists(key):
                missing[key] = date_dir

    total = len(missing)
    if total == 0:
        log("Feature cache: all entries present, skipping warmup")
        return

    log(f"Feature cache: warming up {total} missing entries...")
    model.eval()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    )

    with torch.no_grad(), autocast_ctx:
        for key, date_dir in tqdm(missing.items(), desc="cache warmup", unit="view"):
            if cache.exists(key):   # another process may have written it between scan and now
                continue
            images = _load_images(date_dir, image_preprocess_mode).unsqueeze(0).to(device)
            B, S = 1, images.shape[1]
            feats = model._run_vggt_endpoint(images, B, S)
            cache.put(key, feats)

    log(f"Feature cache: warmup done ({total} entries)")
    model.train()


def train_epoch(
    model_engine,
    loader: DataLoader,
    optimizer,
    device: str,
    cfg: dict[str, Any],
    grad_clip: float,
) -> dict[str, float]:
    model_engine.train()
    totals: dict[str, float] = {}
    n = 0
    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True, disable=not _is_main)
    for batch in pbar:
        with _autocast(device):
            if cfg.get("gradient_checkpointing"):
                outputs = torch.utils.checkpoint.checkpoint(
                    lambda: model_engine(batch), use_reentrant=False
                )
            else:
                outputs = model_engine(batch)
        loss, loss_dict = compute_loss(outputs, batch, device, cfg)
        model_engine.backward(loss)
        model_engine.step()
        _accumulate(totals, loss_dict)
        n += 1
        pbar.set_postfix(loss=f"{loss_dict.get('loss_pointmap', 0):.4f}")
    return {k: v / max(n, 1) for k, v in totals.items()}


def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True, disable=not _is_main)
        for batch in pbar:
            with _autocast(device):
                outputs = model(batch)
            _, loss_dict = compute_loss(outputs, batch, device, cfg)
            _accumulate(totals, loss_dict)
            n += 1
            pbar.set_postfix(loss=f"{loss_dict.get('loss_pointmap', 0):.4f}")
    # All-reduce metrics across ranks
    for k in list(totals.keys()):
        t = torch.tensor(totals[k], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        totals[k] = t.item()
    n_tensor = torch.tensor(n, device=device)
    dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
    n_total = int(n_tensor.item())
    return {k: v / max(n_total, 1) for k, v in totals.items()}


# ─── fold training ────────────────────────────────────────────────────────────

def train_fold(
    fold: dict[str, Any],
    dataset: Dataset,
    cfg: dict[str, Any],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    fold_id = fold["fold_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_log = (output_dir / "train.log").open("w", buffering=1)
    _log_files.append(fold_log)
    try:
        return _train_fold_inner(fold, fold_id, dataset, cfg, device, output_dir)
    finally:
        _log_files.remove(fold_log)
        fold_log.close()


def _train_fold_inner(
    fold: dict[str, Any],
    fold_id: str,
    dataset: Dataset,
    cfg: dict[str, Any],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    train_idx, val_idx = fold_dataset_indices(fold, dataset)

    if not train_idx:
        log(f"  fold={fold_id}: no training samples in vggt_output_root, skipping")
        return {"fold_id": fold_id, "status": "skipped_no_data"}

    log(f"  fold={fold_id}: train_variants={len(train_idx)}  val_variants={len(val_idx)}")

    model     = build_model(cfg, device, feature_cache=dataset.feature_cache if hasattr(dataset, "feature_cache") else None)

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx) if val_idx else None

    live_mode = cfg.get("vggt_inference_mode", "precomputed") == "live"
    if live_mode:
        from dataset.triplet_dataset import LiveTripletDataset as _LTD
        assert isinstance(dataset, _LTD), "live mode requires LiveTripletDataset"

    # num_workers=0 in live mode — VGGT runner can't be shared across forked workers.
    nw = 0 if live_mode else cfg.get("num_workers", 2)
    loader_kwargs = dict(
        batch_size=cfg.get("batch_size", 1),
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        collate_fn=_collate,
    )

    train_sampler = DistributedSampler(train_subset, shuffle=True)
    train_loader = DataLoader(train_subset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs) if val_idx else None

    sched_cfg    = cfg.get("scheduler", {})
    base_lr      = cfg.get("optimizer", {}).get("lr", 1e-4)
    min_lr       = sched_cfg.get("min_lr", 1e-6)
    warmup       = sched_cfg.get("warmup_epochs", 2)
    total_epochs = cfg.get("epochs", 50)
    grad_clip    = cfg.get("grad_clip", 1.0)
    val_every    = cfg.get("val_every", 5)

    es_cfg        = cfg.get("early_stopping", {})
    es_patience   = es_cfg.get("patience", None)   # None = disabled
    es_min_delta  = es_cfg.get("min_delta", 1e-4)
    patience_count = 0

    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    history: list[dict] = []

    resume_path = output_dir / "last_checkpoint.pt"
    start_epoch = 1
    resume_state: dict | None = None   # kept to re-verify weights at first epoch start
    resume_optim_state: dict | None = None
    if resume_path.exists():
        log(f"  resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        resume_state = ckpt["model_state_dict"]
        resume_optim_state = ckpt.get("optimizer_state_dict")
        missing, unexpected = model.load_state_dict(resume_state, strict=False)
        if missing:
            log(f"  resume: {len(missing)} missing keys (new layers use random init): {missing[:3]}")
        if unexpected:
            log(f"  resume: {len(unexpected)} unexpected keys (ignored): {unexpected[:3]}")
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        best_epoch     = ckpt.get("best_epoch", 0)
        patience_count = ckpt.get("patience_count", 0)
        history        = ckpt.get("history", [])
        log(f"  resumed at epoch {start_epoch}/{total_epochs}  best_val={best_val_loss:.6f} (ep {best_epoch})")
        log("  verifying loaded weights match checkpoint...")
        _check_weights_match_checkpoint(model, resume_state)
    else:
        # Only apply init_checkpoint when there is no resume checkpoint.
        ckpt_path = cfg.get("init_checkpoint")
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log(f"  init_checkpoint: {len(missing)} missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if unexpected:
                log(f"  init_checkpoint: {len(unexpected)} unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
            if not missing and not unexpected:
                log(f"  init_checkpoint: loaded {ckpt_path} (all keys matched)")
            else:
                log(f"  init_checkpoint: loaded {ckpt_path} ({len(missing)} missing, {len(unexpected)} unexpected)")

    # ── DeepSpeed initialize ──
    ds_config = json.loads(Path(cfg["deepspeed_config"]).read_text())
    opt_cfg = cfg.get("optimizer", {})
    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": opt_cfg.get("lr", 1e-4),
            "weight_decay": opt_cfg.get("weight_decay", 1e-2),
            "betas": opt_cfg.get("betas", [0.9, 0.999]),
        },
    }
    ds_config["gradient_clipping"] = grad_clip
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
        config=ds_config,
    )
    if resume_optim_state is not None:
        try:
            optimizer.load_state_dict(resume_optim_state)
            log("  optimizer state restored from checkpoint")
        except Exception as e:
            log(f"  WARNING: could not restore optimizer state: {e}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        lr = cosine_lr(epoch - 1, warmup, total_epochs, base_lr, min_lr)
        set_lr(optimizer, lr)

        if epoch == start_epoch and resume_state is not None:
            log(f"  epoch {epoch}: re-verifying weights before first training step...")
            _check_weights_match_checkpoint(model, resume_state)

        train_metrics = train_epoch(model_engine, train_loader, optimizer, device, cfg, grad_clip)
        row = {"epoch": epoch, "lr": lr,
               **{f"train_{k}": v for k, v in train_metrics.items()}}

        if val_loader and epoch % val_every == 0:
            val_metrics = eval_epoch(model_engine, val_loader, device, cfg)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

            # Checkpoint criterion: geometry + camera accuracy only.
            # Excludes confidence regularizer and gradient term (training stabilizers).
            lambda_camera = cfg.get("camera_loss", {}).get("lambda_camera", 0.1)
            val_loss = (
                val_metrics.get("loss_point_reg", float("inf"))
                + lambda_camera * val_metrics.get("loss_camera", 0.0)
            )

            if val_loss < best_val_loss - es_min_delta:
                best_val_loss  = val_loss
                best_epoch     = epoch
                patience_count = 0
                if _is_main:
                    torch.save(model_engine.module.state_dict(), output_dir / "best_model.pt")
            else:
                patience_count += 1

            patience_str = (
                f"  patience={patience_count}/{es_patience}" if es_patience else ""
            )
            log(f"  epoch={epoch}/{total_epochs} lr={lr:.2e}"
                f"  total={_total_loss(train_metrics, lambda_camera):.6f}"
                f"  pm={train_metrics.get('loss_pointmap', 0):.6f}"
                f"  reg={train_metrics.get('loss_point_reg', 0):.6f}"
                f"  grad={train_metrics.get('loss_point_grad', 0):.6f}"
                f"  conf={train_metrics.get('loss_point_conf_reg', 0):.6f}"
                + (f"  cam={train_metrics.get('loss_camera', 0):.6f}" if "loss_camera" in train_metrics else "")
                + f"  | val_crit={val_loss:.6f}"
                f"  val_pm={val_metrics.get('loss_pointmap', 0):.6f}"
                f"  val_reg={val_metrics.get('loss_point_reg', 0):.6f}"
                + (f"  val_cam={val_metrics.get('loss_camera', 0):.6f}" if "loss_camera" in val_metrics else "")
                + f"  best={best_val_loss:.6f} (ep {best_epoch})"
                + patience_str)

            if es_patience and patience_count >= es_patience:
                log(f"  early stopping: no improvement for {es_patience} val checks")
                history.append(row)
                _save_checkpoint(output_dir, epoch, model_engine, optimizer,
                                 best_val_loss, best_epoch, patience_count, history)
                dist.barrier()
                break
        else:
            lambda_camera = cfg.get("camera_loss", {}).get("lambda_camera", 0.1)
            log(f"  epoch={epoch}/{total_epochs} lr={lr:.2e}"
                f"  total={_total_loss(train_metrics, lambda_camera):.6f}"
                f"  pm={train_metrics.get('loss_pointmap', 0):.6f}"
                f"  reg={train_metrics.get('loss_point_reg', 0):.6f}"
                f"  grad={train_metrics.get('loss_point_grad', 0):.6f}"
                f"  conf={train_metrics.get('loss_point_conf_reg', 0):.6f}"
                + (f"  cam={train_metrics.get('loss_camera', 0):.6f}" if "loss_camera" in train_metrics else ""))

        history.append(row)
        _save_checkpoint(output_dir, epoch, model_engine, optimizer,
                         best_val_loss, best_epoch, patience_count, history)
        dist.barrier()

    if not val_loader:
        # No validation — treat the final checkpoint as the best.
        if _is_main:
            torch.save(model_engine.module.state_dict(), output_dir / "best_model.pt")
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

def main() -> None:
    args   = parse_args()
    cfg    = build_config(args)

    # ── Distributed init ──
    deepspeed.init_distributed()
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    is_main = (global_rank == 0)

    global _is_main
    _is_main = is_main

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    cfg["device"] = device
    cfg["deepspeed_config"] = str(args.deepspeed_config)

    if is_main:
        log(f"device={device}  world_size={world_size}  output={cfg['output_root']}")

    num_query_views = cfg.get("model_kwargs", {}).get("num_query_views", 1)
    live_mode = cfg.get("vggt_inference_mode", "precomputed") == "live"

    cache_root = cfg.get("feature_cache_root")
    feature_cache = VGGTFeatureCache.from_config(cache_root, cfg) if cache_root else None
    if feature_cache:
        log(f"VGGT feature cache: {cache_root} namespace={feature_cache.namespace}")
    else:
        log("VGGT feature cache: disabled")

    if live_mode:
        from vggt_pipeline.execute_vggt import get_vggt_runner
        vggt_device = cfg.get("vggt_device", "auto")
        vggt_model_id = cfg.get("vggt_model_id", "facebook/VGGT-1B")
        log(f"live VGGT mode: loading model_id={vggt_model_id} on vggt_device={vggt_device}")
        vggt_runner = get_vggt_runner(model_id=vggt_model_id, device=vggt_device, use_cache=True)
        log("VGGT model loaded")
        log(f"Building LiveTripletDataset from {cfg['all_triplets_path']}")
        all_triplets: list[dict[str, Any]] = json.loads(cfg["all_triplets_path"].read_text())
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
        log(f"Dataset: {len(dataset)} (triplet, variant) pairs (live mode)")
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

    # Pre-compute missing cache entries upfront before any fold training.
    if feature_cache is not None and not live_mode:
        warmup_model = build_model(cfg, device, feature_cache=feature_cache)
        warmup_feature_cache(warmup_model, dataset, device, cfg.get("image_preprocess_mode", "pad"))
        del warmup_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    triplets = load_triplets(cfg["triplets_path"])

    cfg["output_root"].mkdir(parents=True, exist_ok=True)
    write_json(cfg["output_root"] / "train_config.json", cfg)

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
                    f"crop={crop} test_date={fold['test_date']} "
                    f"val_date={fold.get('val_date')} ---")
                result = train_fold(fold, dataset, cfg, device, fold_dir)
                summary.append(result)

    write_json(cfg["output_root"] / "train_summary.json", summary)
    log(f"Done. Summary: {cfg['output_root'] / 'train_summary.json'}")

    _log_files.remove(main_log)
    main_log.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
