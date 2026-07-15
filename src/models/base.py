"""Abstract base class for temporal dense geometry prediction.

Subclass this and implement forward() to plug into the training loop.

Example:

    class MyModel(TemporalGeometryPredictor):
        def forward(self, batch):
            # batch["images_t1"]: [B, M1, 3, H, W]
            # batch["images_t3"]: [B, M3, 3, H, W]
            # batch["doy_t1/t2/t3"]: [B]
            # batch["day_index_t1/t2/t3"]: [B]
            # return dict with "point_maps": [B, K, 3, H_out, W_out]
            ...

Set model_module and model_class in configs/train.yaml to use a model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class TemporalGeometryPredictor(nn.Module, ABC):
    """Predict dense geometry at t2 from multi-view images at t1 and t3."""

    @abstractmethod
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: {
                "images_t1":      [B, M1, 3, H, W] in [0, 1]
                "images_t3":      [B, M3, 3, H, W] in [0, 1]
                "doy_t1":         [B] float, day-of-year for t1
                "doy_t2":         [B] float, day-of-year for t2
                "doy_t3":         [B] float, day-of-year for t3
                "day_index_t1":   [B] int, date ordinal for t1
                "day_index_t2":   [B] int, date ordinal for t2
                "day_index_t3":   [B] int, date ordinal for t3
            }
        Returns:
            {
                "point_maps":        [B, K, 3, H_out, W_out]  required
                "depth_maps":        [B, K, 1, H_out, W_out]  optional
                "point_confidence":  [B, K, 1, H_out, W_out]  optional
                "depth_confidence":  [B, K, 1, H_out, W_out]  optional
            }
        """
        ...

