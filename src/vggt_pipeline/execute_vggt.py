from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

_RUNNER_CACHE: dict[tuple[str, str], "VggtRunner"] = {}


@dataclass
class VggtRunner:
    model: VGGT
    model_source: str
    device: str
    dtype: torch.dtype


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def get_vggt_runner(model_id: str, device: str = "auto", use_cache: bool = True) -> VggtRunner:
    selected_device = choose_device(device)
    cache_key = (model_id, selected_device)
    if use_cache and cache_key in _RUNNER_CACHE:
        log(f"reusing loaded model model_id={model_id} device={selected_device}")
        return _RUNNER_CACHE[cache_key]

    dtype = choose_dtype(selected_device)
    log(f"loading model from Hugging Face model_id={model_id} device={selected_device}")
    model = VGGT.from_pretrained(model_id)
    model.eval()
    model = model.to(selected_device)
    runner = VggtRunner(model=model, model_source=f"from_pretrained:{model_id}", device=selected_device, dtype=dtype)

    if use_cache:
        _RUNNER_CACHE[cache_key] = runner
    return runner


_DEFAULT_CACHE_LAYERS = [4, 11, 17, 23]


def cached_layers_exist(output_dir: Path, cache_layers: list[int] | None = None) -> bool:
    """Check if all cached layer .npy files already exist under output_dir/predictions/."""
    layers = cache_layers or _DEFAULT_CACHE_LAYERS
    pred_dir = output_dir / "predictions"
    return all((pred_dir / f"cached_layer_{l:02d}.npy").exists() for l in layers)


def save_cached_layers(
    image_paths: list[str],
    output_dir: Path,
    runner: VggtRunner,
    image_preprocess_mode: str = "pad",
    cache_layers: list[int] | None = None,
) -> dict[str, str]:
    """Run VGGT aggregator and save intermediate cached layers as .npy files.

    Saves layers at indices `cache_layers` (default [4,11,17,23]) from the
    aggregator's output_list. Each saved array has shape [S, T, D] where
    S=num_views, T=num_tokens_per_view, D=2*C (2048 for VGGT-1B).

    Skips entirely if all files already exist.

    Returns dict mapping layer index to saved file path.
    """
    layers = cache_layers or _DEFAULT_CACHE_LAYERS
    pred_dir = output_dir / "predictions"

    if all((pred_dir / f"cached_layer_{l:02d}.npy").exists() for l in layers):
        log(f"cached layers already exist, skipping: {output_dir}")
        return {l: str(pred_dir / f"cached_layer_{l:02d}.npy") for l in layers}

    images = load_and_preprocess_images(image_paths, mode=image_preprocess_mode)
    images = images.to(runner.device)

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=runner.dtype)
        if runner.device.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            images_batch = images[None]
            aggregated_tokens_list, _ = runner.model.aggregator(images_batch)

    pred_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for l in layers:
        arr = aggregated_tokens_list[l].squeeze(0).float().cpu().numpy()  # [S, T, D]
        path = pred_dir / f"cached_layer_{l:02d}.npy"
        np.save(path, arr)
        saved[l] = str(path.resolve())

    log(f"saved cached layers {layers} to {pred_dir}")
    return saved


def run_vggt_inference_in_memory(
    image_paths: list[str],
    runner: VggtRunner,
    image_preprocess_mode: str = "pad",
) -> dict[str, torch.Tensor]:
    """Run VGGT forward pass and return CPU float32 tensors (no disk I/O).

    Returns dict with keys: extrinsic [S,3,4], intrinsic [S,3,3],
    depth_map [S,H,W], depth_confidence [S,H,W], point_map [S,H,W,3],
    point_confidence [S,H,W].  Shapes mirror the saved .npy files from
    run_vggt_inference_from_image_paths.
    """
    images = load_and_preprocess_images(image_paths, mode=image_preprocess_mode)
    images = images.to(runner.device)

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=runner.dtype)
        if runner.device.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = runner.model.aggregator(images_batch)
            pose_enc = runner.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = runner.model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            point_map, point_conf = runner.model.point_head(aggregated_tokens_list, images_batch, ps_idx)

    return {
        "extrinsic":        extrinsic.squeeze(0).float().cpu(),
        "intrinsic":        intrinsic.squeeze(0).float().cpu(),
        "depth_map":        depth_map.squeeze(0).float().cpu(),
        "depth_confidence": depth_conf.squeeze(0).float().cpu(),
        "point_map":        point_map.squeeze(0).float().cpu(),
        "point_confidence": point_conf.squeeze(0).float().cpu(),
    }


def run_vggt_inference_from_image_paths(
    image_paths: list[str],
    output_dir: Path,
    runner: VggtRunner,
    image_preprocess_mode: str = "pad",
    input_image_list_path: Path | None = None,
) -> dict[str, Any]:
    log(f"start inference output_dir={output_dir} num_images={len(image_paths)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_and_preprocess_images(image_paths, mode=image_preprocess_mode)
    images = images.to(runner.device)
    log(f"preprocessed image tensor shape={tuple(images.shape)}")

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=runner.dtype)
        if runner.device == "cuda"
        else nullcontext()
    )

    start = time.time()
    with torch.no_grad():
        with autocast_ctx:
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = runner.model.aggregator(images_batch)
            pose_enc = runner.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = runner.model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            point_map, point_conf = runner.model.point_head(aggregated_tokens_list, images_batch, ps_idx)

    duration_sec = time.time() - start
    log(f"forward pass complete duration_sec={duration_sec:.3f}")

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str, arr: np.ndarray) -> str:
        path = pred_dir / name
        np.save(path, arr)
        return str(path.resolve())

    point_np = point_map.squeeze(0).detach().cpu().numpy()
    point_conf_np = point_conf.squeeze(0).detach().cpu().numpy()

    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_np = depth_map.squeeze(0).detach().cpu().numpy()
    depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy()

    outputs = {
        "point_map": save("point_map.npy", point_np),
        "point_confidence": save("point_confidence.npy", point_conf_np),
        "extrinsic": save("extrinsic.npy", extrinsic_np),
        "intrinsic": save("intrinsic.npy", intrinsic_np),
        "depth_map": save("depth_map.npy", depth_np),
        "depth_confidence": save("depth_confidence.npy", depth_conf_np),
    }
    metadata: dict[str, Any] = {
        "model_source": runner.model_source,
        "device": runner.device,
        "dtype": str(runner.dtype).replace("torch.", ""),
        "num_images": len(image_paths),
        "image_preprocess_mode": image_preprocess_mode,
        "input_image_list": str(input_image_list_path.resolve()) if input_image_list_path else None,
        "output_dir": str(output_dir.resolve()),
        "duration_sec": duration_sec,
        "outputs": outputs,
    }
    meta_path = output_dir / "prediction_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log(f"saved predictions to {pred_dir}, metadata to {meta_path}")
    return metadata

