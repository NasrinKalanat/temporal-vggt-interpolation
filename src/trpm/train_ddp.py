"""Train TRPM-Small with LOTO cross-validation (DDP version).

Usage:
    torchrun --nproc_per_node=N src/trpm/train_ddp.py --config configs/train_trpm_small.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, IO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from trpm.dataset import PointMapTripletDataset
from trpm.loss import trpm_loss
from loto import build_loto_folds, load_triplets


def _load_model_class(class_path: str):
    """Load a model class from a dotted path, e.g. 'trpm.model.TRPMSmall'."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module_path), class_name)


# ── DDP setup / teardown ──────────────────────────────────────────────────────

def setup_ddp() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main() -> bool:
    return dist.get_rank() == 0


# ── utilities ─────────────────────────────────────────────────────────────────

_log_files: list[IO[str]] = []


def log(msg: str) -> None:
    if not is_main():
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in _log_files:
        print(line, file=f, flush=True)


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, obj: Any) -> None:
    if not is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "vggt_output_root": "vggt_outputs/t1t2_paired_v16_o8",
    "triplets_path": "prepared_data/subsets/benchmark_triplets.json",
    "output_root": "runs/trpm_small",
    "protocols": ["target_date", "strict"],
    "crops": ["corn"],
    "test_date": None,
    "seed": 42,
    "epochs": 100,
    "batch_size": 8,
    "num_workers": 4,
    "val_every": 5,
    "grad_clip": 1.0,
    "model_kwargs": {
        "num_t3_points": 1024,
        "cond_dim": 192,
    },
    "optimizer": {"lr": 1e-4, "weight_decay": 1e-2},
    "scheduler": {"warmup_epochs": 5, "min_lr": 1e-6},
    "loss": {
        "lambda_chamfer": 0.05,
        "lambda_res": 0.01,
        "lambda_gate": 0.001,
        "chamfer_warmup_epoch": 11,
    },
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config and Path(args.config).exists():
        cfg.update(read_yaml(Path(args.config)))
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.test_date:
        cfg["test_date"] = args.test_date
    if args.epochs:
        cfg["epochs"] = args.epochs
    pred_thr = cfg.get("pred_conf_threshold", 0.02)
    cfg.setdefault("model_kwargs", {})["conf_threshold"] = pred_thr
    cfg.setdefault("loss", {})["conf_threshold"] = pred_thr
    cfg["vggt_output_root"] = Path(cfg["vggt_output_root"])
    cfg["triplets_path"]    = Path(cfg["triplets_path"])
    cfg["output_root"]      = Path(cfg["output_root"])
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--test-date", default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


# ── LOTO fold helpers ─────────────────────────────────────────────────────────

def _triplet_id(t: dict) -> str:
    return f"{t['left_date']}_{t['middle_date']}_{t['right_date']}_{t['crop']}"


def fold_indices(fold: dict, dataset: PointMapTripletDataset) -> tuple[list[int], list[int]]:
    train_ids = {_triplet_id(t) for t in fold["train_triplets"]}
    val_ids   = {_triplet_id(t) for t in fold["val_triplets"]}
    train_idx = [i for i, e in enumerate(dataset.index) if e["triplet_id"] in train_ids]
    val_idx   = [i for i, e in enumerate(dataset.index) if e["triplet_id"] in val_ids]
    return train_idx, val_idx


# ── optimizer / scheduler ─────────────────────────────────────────────────────

def cosine_lr(epoch: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    p = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


# ── collate ───────────────────────────────────────────────────────────────────

def _collate(samples: list) -> dict:
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


# ── metric all-reduce ─────────────────────────────────────────────────────────

def reduce_metrics(totals: dict[str, float], n: int, device: str) -> dict[str, float]:
    keys = sorted(totals.keys())
    vals = torch.tensor([totals[k] for k in keys] + [float(n)], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    total_n = vals[-1].item()
    return {k: vals[i].item() / max(total_n, 1) for i, k in enumerate(keys)}


# ── train / eval loops ────────────────────────────────────────────────────────

def _is_cam_depth_model(model) -> bool:
    from trpm.trpm_small_cam_depth import TRPMSmallCamDepth
    m = model.module if hasattr(model, "module") else model
    return isinstance(m, TRPMSmallCamDepth)


def _is_cam_color_model(model) -> bool:
    from trpm.trpm_small_cam_color import TRPMSmallCamColor
    m = model.module if hasattr(model, "module") else model
    return isinstance(m, TRPMSmallCamColor)


def _is_cam_model(model) -> bool:
    from trpm.trpm_small_cam import TRPMSmallCam
    m = model.module if hasattr(model, "module") else model
    return isinstance(m, TRPMSmallCam)


def _step_base(model, batch, device, loss_cfg, use_chamfer):
    """Step for TRPMSmall: flattens V into batch dimension."""
    P1  = batch["P1"].to(device)
    C1  = batch["C1"].to(device)
    P3  = batch["P3"].to(device)
    C3  = batch["C3"].to(device)
    P2  = batch["P2"].to(device)
    C2  = batch["C2"].to(device)
    tau = batch["tau"].to(device)   # [B, 1]

    B, V = P1.shape[:2]
    def _flat(x): return x.flatten(0, 1)          # [B*V, C, H, W]
    tau_bv = tau.unsqueeze(1).expand(B, V, 1).flatten(0, 1)  # [B*V, 1]

    out = model(_flat(P1), _flat(C1), _flat(P3), _flat(C3), tau_bv)
    return trpm_loss(
        P2_hat=out["P2_hat"], delta_P=out["delta_P"], G=out["G"],
        P2=_flat(P2), C2=_flat(C2),
        conf_threshold=loss_cfg.get("conf_threshold", 0.02),
        lambda_chamfer=loss_cfg.get("lambda_chamfer", 0.05),
        lambda_res=loss_cfg.get("lambda_res", 0.01),
        lambda_gate=loss_cfg.get("lambda_gate", 0.001),
        use_chamfer=use_chamfer,
    )


def _step_cam_depth(model, batch, device, loss_cfg, use_chamfer):
    """Step for TRPMSmallCamDepth: includes depth target from depth_map.npy."""
    from trpm.trpm_small_cam import world_to_cam

    P1     = batch["P1"].to(device)
    C1     = batch["C1"].to(device)
    P2     = batch["P2"].to(device)
    C2     = batch["C2"].to(device)
    P3     = batch["P3"].to(device)
    C3     = batch["C3"].to(device)
    D2     = batch["D2"].to(device)
    tau    = batch["tau"].to(device)
    T1_c2w = batch["T1_c2w"].to(device)
    T2_c2w = batch["T2_c2w"].to(device)
    T3_c2w = batch["T3_c2w"].to(device)
    K2     = batch["K2"].to(device)
    K3     = batch["K3"].to(device)

    B, V  = P1.shape[:2]
    V3    = P3.shape[1]

    def _flat(x): return x.flatten(0, 1)

    P3_exp  = P3.unsqueeze(1).expand(B, V, V3, *P3.shape[2:]).flatten(0, 1)
    C3_exp  = C3.unsqueeze(1).expand(B, V, V3, *C3.shape[2:]).flatten(0, 1)
    T3_exp  = T3_c2w.unsqueeze(1).expand(B, V, V3, 4, 4).flatten(0, 1)
    K3_exp  = K3.unsqueeze(1).expand(B, V, V3, 3, 3).flatten(0, 1)
    tau_exp = tau.unsqueeze(1).expand(B, V, 1).flatten(0, 1)

    out = model(
        _flat(P1), _flat(C1),
        P3_exp, C3_exp,
        _flat(T2_c2w), _flat(T1_c2w), T3_exp,
        _flat(K2), K3_exp, tau_exp,
    )

    P2_cam = world_to_cam(_flat(P2), _flat(T2_c2w))

    # depth_mode: "depth_only" | "point_only" | "point_depth"
    depth_mode = loss_cfg.get("depth_mode", "depth_only")
    if depth_mode == "depth_only":
        P2_target = P2_cam.detach()
        lambda_depth = loss_cfg.get("lambda_depth", 0.1)
    elif depth_mode == "point_only":
        P2_target = P2_cam
        lambda_depth = 0.0
    else:  # "point_depth"
        P2_target = P2_cam
        lambda_depth = loss_cfg.get("lambda_depth", 0.1)

    return trpm_loss(
        P2_hat=out["P2_cam_hat"], delta_P=out["delta_P"], G=out["G"],
        P2=P2_target, C2=_flat(C2),
        D2_hat=out["D2_hat"], D2=_flat(D2), K2=_flat(K2),
        conf_threshold=loss_cfg.get("conf_threshold", 0.02),
        lambda_chamfer=loss_cfg.get("lambda_chamfer", 0.05),
        lambda_res=loss_cfg.get("lambda_res", 0.01),
        lambda_gate=loss_cfg.get("lambda_gate", 0.001),
        lambda_depth=lambda_depth,
        use_chamfer=use_chamfer,
    )


def _step_cam_point(model, batch, device, loss_cfg, use_chamfer):
    """Step for TRPMSmallCam: flattens V into batch, broadcasts t3 to match."""
    from trpm.trpm_small_cam import world_to_cam

    P1     = batch["P1"].to(device)      # [B, V,  3, H, W]
    C1     = batch["C1"].to(device)
    P2     = batch["P2"].to(device)
    C2     = batch["C2"].to(device)
    P3     = batch["P3"].to(device)      # [B, V3, 3, H, W]
    C3     = batch["C3"].to(device)
    tau    = batch["tau"].to(device)     # [B, 1]
    T1_c2w = batch["T1_c2w"].to(device) # [B, V,  4, 4]
    T2_c2w = batch["T2_c2w"].to(device)
    T3_c2w = batch["T3_c2w"].to(device) # [B, V3, 4, 4]
    K2     = batch["K2"].to(device)     # [B, V,  3, 3]
    K3     = batch["K3"].to(device)     # [B, V,  3, 3]

    B, V  = P1.shape[:2]
    V3    = P3.shape[1]

    def _flat(x): return x.flatten(0, 1)  # [B*V, ...]

    # Broadcast t3 and tau to match B*V
    P3_exp  = P3.unsqueeze(1).expand(B, V, V3, *P3.shape[2:]).flatten(0, 1)   # [B*V, V3, 3, H, W]
    C3_exp  = C3.unsqueeze(1).expand(B, V, V3, *C3.shape[2:]).flatten(0, 1)
    T3_exp  = T3_c2w.unsqueeze(1).expand(B, V, V3, 4, 4).flatten(0, 1)       # [B*V, V3, 4, 4]
    K3_exp  = K3.unsqueeze(1).expand(B, V, V3, 3, 3).flatten(0, 1)
    tau_exp = tau.unsqueeze(1).expand(B, V, 1).flatten(0, 1)                  # [B*V, 1]

    out = model(
        _flat(P1), _flat(C1),
        P3_exp, C3_exp,
        _flat(T2_c2w), _flat(T1_c2w), T3_exp,
        _flat(K2), K3_exp, tau_exp,
    )

    P2_cam = world_to_cam(_flat(P2), _flat(T2_c2w))  # [B*V, 3, H, W]
    return trpm_loss(
        P2_hat=out["P2_cam_hat"], delta_P=out["delta_P"], G=out["G"],
        P2=P2_cam, C2=_flat(C2),
        conf_threshold=loss_cfg.get("conf_threshold", 0.02),
        lambda_chamfer=loss_cfg.get("lambda_chamfer", 0.05),
        lambda_res=loss_cfg.get("lambda_res", 0.01),
        lambda_gate=loss_cfg.get("lambda_gate", 0.001),
        use_chamfer=use_chamfer,
    )


def _step_cam_color(model, batch, device, loss_cfg, use_chamfer):
    """Step for TRPMSmallCamColor: includes RGB inputs and color loss."""
    from trpm.trpm_small_cam import world_to_cam

    P1     = batch["P1"].to(device)
    C1     = batch["C1"].to(device)
    P2     = batch["P2"].to(device)
    C2     = batch["C2"].to(device)
    P3     = batch["P3"].to(device)
    C3     = batch["C3"].to(device)
    I1     = batch["I1"].to(device)
    I2     = batch["I2"].to(device)
    I3     = batch["I3"].to(device)
    tau    = batch["tau"].to(device)
    T1_c2w = batch["T1_c2w"].to(device)
    T2_c2w = batch["T2_c2w"].to(device)
    T3_c2w = batch["T3_c2w"].to(device)
    K2     = batch["K2"].to(device)
    K3     = batch["K3"].to(device)

    B, V  = P1.shape[:2]
    V3    = P3.shape[1]

    def _flat(x): return x.flatten(0, 1)

    P3_exp  = P3.unsqueeze(1).expand(B, V, V3, *P3.shape[2:]).flatten(0, 1)
    C3_exp  = C3.unsqueeze(1).expand(B, V, V3, *C3.shape[2:]).flatten(0, 1)
    I3_exp  = I3.unsqueeze(1).expand(B, V, V3, *I3.shape[2:]).flatten(0, 1)
    T3_exp  = T3_c2w.unsqueeze(1).expand(B, V, V3, 4, 4).flatten(0, 1)
    K3_exp  = K3.unsqueeze(1).expand(B, V, V3, 3, 3).flatten(0, 1)
    tau_exp = tau.unsqueeze(1).expand(B, V, 1).flatten(0, 1)

    out = model(
        _flat(P1), _flat(C1), _flat(I1),
        P3_exp, C3_exp, I3_exp,
        _flat(T2_c2w), _flat(T1_c2w), T3_exp,
        _flat(K2), K3_exp, tau_exp,
    )

    P2_cam = world_to_cam(_flat(P2), _flat(T2_c2w))
    return trpm_loss(
        P2_hat=out["P2_cam_hat"], delta_P=out["delta_P"], G=out["G"],
        P2=P2_cam, C2=_flat(C2),
        RGB2_hat=out["RGB2_hat"], RGB2=_flat(I2),
        conf_threshold=loss_cfg.get("conf_threshold", 0.02),
        lambda_chamfer=loss_cfg.get("lambda_chamfer", 0.05),
        lambda_res=loss_cfg.get("lambda_res", 0.01),
        lambda_gate=loss_cfg.get("lambda_gate", 0.001),
        lambda_rgb=loss_cfg.get("lambda_rgb", 0.05),
        use_chamfer=use_chamfer,
    )


def _step(model, batch, device, loss_cfg, use_chamfer):
    if _is_cam_color_model(model):
        return _step_cam_color(model, batch, device, loss_cfg, use_chamfer)
    if _is_cam_depth_model(model):
        return _step_cam_depth(model, batch, device, loss_cfg, use_chamfer)
    if _is_cam_model(model):
        return _step_cam_point(model, batch, device, loss_cfg, use_chamfer)
    return _step_base(model, batch, device, loss_cfg, use_chamfer)


def train_epoch(model, loader, optimizer, device, loss_cfg, grad_clip, use_chamfer):
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in tqdm(loader, desc="train", leave=False, disable=not is_main()):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ld = _step(model, batch, device, loss_cfg, use_chamfer)
        ld["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for k, v in ld.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())
        n += 1
    return reduce_metrics(totals, n, device)


def eval_epoch(model, loader, device, loss_cfg):
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val  ", leave=False, disable=not is_main()):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld = _step(model, batch, device, loss_cfg, use_chamfer=False)
            for k, v in ld.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            n += 1
    return reduce_metrics(totals, n, device)


# ── checkpoint ────────────────────────────────────────────────────────────────

def _save_checkpoint(output_dir, epoch, model, optimizer, best_val_loss, best_epoch, history):
    if is_main():
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
        }, output_dir / "last_checkpoint.pt")
        torch.save(model.module.state_dict(), output_dir / "last_model.pt")
    dist.barrier()


def _save_best(output_dir, model):
    if is_main():
        torch.save(model.module.state_dict(), output_dir / "best_model.pt")
    dist.barrier()


# ── fold training ─────────────────────────────────────────────────────────────

def train_fold(fold, dataset, cfg, device, local_rank, world_size, output_dir):
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    fold_log = None
    if is_main():
        fold_log = (output_dir / "train.log").open("w", buffering=1)
        _log_files.append(fold_log)

    try:
        train_idx, val_idx = fold_indices(fold, dataset)
        if not train_idx:
            log(f"  fold={fold['fold_id']}: no training samples, skipping")
            return

        log(f"  fold={fold['fold_id']}: train={len(train_idx)}  val={len(val_idx)}")

        model = _load_model_class(cfg.get("model_class", "trpm.model.TRPMSmall"))(**cfg.get("model_kwargs", {})).to(device)
        log(f"  params={sum(p.numel() for p in model.parameters()):,}")

        opt_cfg   = cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
        )

        # Resume before wrapping in DDP
        resume_path = output_dir / "last_checkpoint.pt"
        start_epoch   = 1
        best_val_loss = float("inf")
        best_epoch    = 0
        history: list = []
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_epoch    = ckpt.get("best_epoch", 0)
            history       = ckpt.get("history", [])
            log(f"  resumed at epoch {start_epoch}")

        model = DDP(model, device_ids=[local_rank])

        loader_kw = dict(
            batch_size=cfg.get("batch_size", 8),
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=_collate,
        )
        rank = dist.get_rank()
        train_subset  = Subset(dataset, train_idx)
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader  = DataLoader(train_subset, sampler=train_sampler, **loader_kw)

        val_loader = val_sampler = None
        if val_idx:
            val_subset  = Subset(dataset, val_idx)
            val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)
            val_loader  = DataLoader(val_subset, sampler=val_sampler, **loader_kw)

        sched_cfg    = cfg.get("scheduler", {})
        base_lr      = opt_cfg.get("lr", 1e-4)
        min_lr       = sched_cfg.get("min_lr", 1e-6)
        warmup       = sched_cfg.get("warmup_epochs", 5)
        total_epochs = cfg.get("epochs", 100)
        grad_clip    = cfg.get("grad_clip", 1.0)
        val_every    = cfg.get("val_every", 5)
        loss_cfg     = cfg.get("loss", {})
        chamfer_start = loss_cfg.get("chamfer_warmup_epoch", 11)

        for epoch in range(start_epoch, total_epochs + 1):
            lr = cosine_lr(epoch - 1, warmup, total_epochs, base_lr, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            train_sampler.set_epoch(epoch)

            use_chamfer = epoch >= chamfer_start
            train_m = train_epoch(model, train_loader, optimizer, device, loss_cfg, grad_clip, use_chamfer)
            row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_m.items()}}

            if val_loader and epoch % val_every == 0:
                val_sampler.set_epoch(epoch)
                val_m = eval_epoch(model, val_loader, device, loss_cfg)
                row.update({f"val_{k}": v for k, v in val_m.items()})
                val_loss = val_m.get("loss_point", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch    = epoch
                    _save_best(output_dir, model)
                log(f"  epoch={epoch}/{total_epochs} lr={lr:.2e}"
                    f"  train_total={train_m.get('loss', 0):.4f}"
                    f"  train_point={train_m.get('loss_point', 0):.4f}"
                    f"  val_total={val_m.get('loss', 0):.4f}"
                    f"  val_point={val_loss:.4f}"
                    f"  chamfer={val_m.get('loss_chamfer', 0):.4f}"
                    f"  res={val_m.get('loss_res', 0):.4f}"
                    f"  gate={val_m.get('loss_gate', 0):.4f}"
                    f"  rgb={val_m.get('loss_rgb', 0):.4f}"
                    f"  best={best_val_loss:.4f} (ep {best_epoch})")
            else:
                log(f"  epoch={epoch}/{total_epochs} lr={lr:.2e}"
                    f"  train_total={train_m.get('loss', 0):.4f}"
                    f"  train_point={train_m.get('loss_point', 0):.4f}"
                    f"  chamfer={train_m.get('loss_chamfer', 0):.4f}"
                    f"  res={train_m.get('loss_res', 0):.4f}"
                    f"  gate={train_m.get('loss_gate', 0):.4f}"
                    f"  rgb={train_m.get('loss_rgb', 0):.4f}")

            history.append(row)
            _save_checkpoint(output_dir, epoch, model, optimizer, best_val_loss, best_epoch, history)

        if not val_loader:
            _save_best(output_dir, model)

        write_json(output_dir / "training_history.json", history)
        log(f"  fold done. best_epoch={best_epoch}  best_val_loss={best_val_loss:.4f}")

    finally:
        if fold_log:
            _log_files.remove(fold_log)
            fold_log.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main(local_rank: int, world_size: int) -> None:
    device = f"cuda:{local_rank}"
    args = parse_args()
    cfg  = build_config(args)

    log(f"DDP: world_size={world_size}  local_rank={local_rank}")
    log(f"output={cfg['output_root']}")

    dataset = PointMapTripletDataset(cfg["vggt_output_root"], conf_threshold=cfg.get("conf_threshold", 1.0), cache_dir=cfg.get("preprocess_cache_dir"), load_rgb="color" in cfg.get("model_class", "").lower(), load_depth="depth" in cfg.get("model_class", "").lower())
    log(f"Dataset: {len(dataset)} samples")

    triplets = load_triplets(cfg["triplets_path"])

    if is_main():
        cfg["output_root"].mkdir(parents=True, exist_ok=True)
    write_json(cfg["output_root"] / "train_config.json", cfg)
    dist.barrier()

    main_log = None
    if is_main():
        main_log = (cfg["output_root"] / "train.log").open("w", buffering=1)
        _log_files.append(main_log)

    for protocol in cfg["protocols"]:
        for crop in cfg["crops"]:
            for fold in build_loto_folds(triplets, crop, protocol):
                if cfg.get("test_date") and fold["test_date"] != cfg["test_date"]:
                    continue
                fold_dir = cfg["output_root"] / protocol / fold["fold_id"]
                log(f"--- fold={fold['fold_id']} protocol={protocol} crop={crop} ---")
                train_fold(fold, dataset, cfg, device, local_rank, world_size, fold_dir)

    log("Done.")
    if main_log:
        _log_files.remove(main_log)
        main_log.close()


if __name__ == "__main__":
    local_rank, world_size = setup_ddp()
    try:
        main(local_rank, world_size)
    finally:
        cleanup_ddp()

