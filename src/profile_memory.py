"""Profile GPU memory and timing for TemporalVGGT with/without gradient checkpointing.

Usage:
    conda run -n 4d python src/profile_memory.py --device cuda:1
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mb(bytes_: int) -> str:
    return f"{bytes_ / 1024**2:.0f} MB"


def param_mb(module: torch.nn.Module, trainable_only: bool = False) -> float:
    return sum(
        p.numel() * p.element_size()
        for p in module.parameters()
        if (not trainable_only or p.requires_grad)
    ) / 1024**2


def make_batch(S: int, H: int, W: int, device: str) -> dict:
    return {
        "images_t1":     torch.rand(1, S, 3, H, W).to(device),
        "images_t3":     torch.rand(1, S, 3, H, W).to(device),
        "doy_t1":        torch.tensor([224.0]).to(device),
        "doy_t2":        torch.tensor([234.0]).to(device),
        "doy_t3":        torch.tensor([260.0]).to(device),
        "day_index_t1":  torch.tensor([738377], dtype=torch.long).to(device),
        "day_index_t2":  torch.tensor([738387], dtype=torch.long).to(device),
        "day_index_t3":  torch.tensor([738413], dtype=torch.long).to(device),
    }


def build_model(gradient_checkpointing: bool, device: str) -> torch.nn.Module:
    from src.models.temporal_vggt_old import TemporalVGGT
    model = TemporalVGGT(
        vggt_model_id="facebook/VGGT-1B",
        num_target_views=32,
        target_query_grid=(16, 16),
        lora_layers=[18, 19, 20, 21, 22, 23],
        film_layers=[20, 21, 22, 23],
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        film_hidden_dim=256,
        max_encoder_views=8,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    model.train()
    return model


def run_one_step(model, batch, device: str) -> tuple[float, float, int]:
    """One forward+backward+step. Returns (fwd_s, bwd_s, peak_MB)."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(batch)
    loss = outputs["point_maps"].mean()
    del outputs
    torch.cuda.synchronize(device)
    t_fwd = time.perf_counter() - t0

    t1 = time.perf_counter()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    t_bwd = time.perf_counter() - t1

    peak = torch.cuda.max_memory_allocated(device)
    return t_fwd, t_bwd, peak // (1024**2)


def profile(gradient_checkpointing: bool, batch, device: str, warmup: int = 1, reps: int = 3):
    label = "grad_ckpt=ON " if gradient_checkpointing else "grad_ckpt=OFF"
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    model = build_model(gradient_checkpointing, device)
    print(f"  model params: total={param_mb(model):.0f} MB  "
          f"trainable={param_mb(model, True):.0f} MB")

    # Warm up
    for _ in range(warmup):
        run_one_step(model, batch, device)

    fwds, bwds, peaks = [], [], []
    for i in range(reps):
        f, b, p = run_one_step(model, batch, device)
        fwds.append(f); bwds.append(b); peaks.append(p)
        print(f"  rep {i+1}: fwd={f:.2f}s  bwd={b:.2f}s  total={f+b:.2f}s  peak={p} MB")

    avg_f = sum(fwds)/reps
    avg_b = sum(bwds)/reps
    avg_p = sum(peaks)//reps
    print(f"  avg:   fwd={avg_f:.2f}s  bwd={avg_b:.2f}s  total={avg_f+avg_b:.2f}s  peak={avg_p} MB")

    del model
    torch.cuda.empty_cache()
    return avg_f, avg_b, avg_p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    torch.cuda.set_device(device)

    batch = make_batch(S=32, H=518, W=518, device=device)

    f_off, b_off, p_off = profile(False, batch, device, args.warmup, args.reps)
    f_on,  b_on,  p_on  = profile(True,  batch, device, args.warmup, args.reps)

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  {'':25s} {'no ckpt':>10s}  {'ckpt':>10s}  {'overhead':>10s}")
    print(f"  {'forward':25s} {f_off:>9.2f}s  {f_on:>9.2f}s  {f_on/f_off - 1:>+9.0%}")
    print(f"  {'backward':25s} {b_off:>9.2f}s  {b_on:>9.2f}s  {b_on/b_off - 1:>+9.0%}")
    print(f"  {'total':25s} {f_off+b_off:>9.2f}s  {f_on+b_on:>9.2f}s  {(f_on+b_on)/(f_off+b_off) - 1:>+9.0%}")
    print(f"  {'peak memory':25s} {p_off:>8d}MB  {p_on:>8d}MB  {p_on/p_off - 1:>+9.0%}")


if __name__ == "__main__":
    main()

