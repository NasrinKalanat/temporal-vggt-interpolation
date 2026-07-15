"""Shared helpers for per-sample VGGT feature-cache batching."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def _cache_keys(keys: Any, batch_size: int) -> list[str | None]:
    if keys is None:
        return [None] * batch_size
    if isinstance(keys, str):
        return [keys] * batch_size
    if len(keys) != batch_size:
        raise ValueError(f"Expected {batch_size} cache keys, got {len(keys)}.")
    return list(keys)


def _image_entries(images: Any, batch_size: int) -> list[torch.Tensor | None]:
    if images is None:
        return [None] * batch_size
    if isinstance(images, torch.Tensor):
        if images.shape[0] != batch_size:
            raise ValueError(f"Expected image batch {batch_size}, got {images.shape[0]}.")
        return [images[i : i + 1] for i in range(batch_size)]
    if len(images) != batch_size:
        raise ValueError(f"Expected {batch_size} image entries, got {len(images)}.")
    return [
        image.unsqueeze(0) if isinstance(image, torch.Tensor) and image.dim() == 4 else image
        for image in images
    ]


def _single_sample_features(features: list[torch.Tensor], key: str | None) -> list[torch.Tensor]:
    if not features:
        raise ValueError(f"Cache entry {key!r} is empty.")
    if features[0].shape[0] != 1:
        raise ValueError(f"Cache entry {key!r} is not single-sample.")
    return features


def run_cached_endpoint(
    model: torch.nn.Module,
    images: Any,
    batch_size: int,
    cache_keys: Any,
    endpoint_name: str,
) -> list[torch.Tensor]:
    """Resolve endpoint features using one cache key per sample."""
    if not hasattr(model, "_run_vggt_endpoint") or model.aggregator is None:
        raise RuntimeError("Model cannot compute VGGT endpoint features.")

    cache = getattr(model, "feature_cache", None)
    device = next(model.parameters()).device
    keys = _cache_keys(cache_keys, batch_size)
    image_list = _image_entries(images, batch_size)
    resolved: list[list[torch.Tensor] | None] = [None] * batch_size
    missing: list[int] = []

    for i, key in enumerate(keys):
        cached = cache.get(key) if cache is not None and key is not None else None
        if cached is None:
            missing.append(i)
        else:
            resolved[i] = [f.to(device) for f in _single_sample_features(cached, key)]

    if missing:
        by_shape: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i in missing:
            image = image_list[i]
            if image is None:
                raise RuntimeError(
                    f"Missing {endpoint_name} cache for sample {i}, but images were not loaded."
                )
            by_shape[tuple(image.shape[1:])].append(i)

        for indices in by_shape.values():
            batch_images = torch.cat([image_list[i] for i in indices], dim=0).to(device)
            with torch.no_grad():
                features = model._run_vggt_endpoint(
                    batch_images, batch_images.shape[0], batch_images.shape[1]
                )
            for offset, i in enumerate(indices):
                sample_features = [feature[offset : offset + 1] for feature in features]
                resolved[i] = sample_features
                key = keys[i]
                if cache is not None and key is not None:
                    cache.put(key, sample_features)

    return [
        torch.cat([features[layer_idx] for features in resolved], dim=0)
        for layer_idx in range(len(resolved[0]))
    ]
