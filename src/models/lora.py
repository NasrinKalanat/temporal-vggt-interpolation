"""LoRA (Low-Rank Adaptation) wrappers for Linear and Conv2d layers."""
import logging
import math
from typing import List, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank delta.

    forward(x) = frozen_linear(x) + (lora_B(lora_A(x))) * (alpha / rank)

    lora_B is zero-initialized so the delta is zero at the start, preserving
    the pretrained VGGT behavior.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class LoRAConv2d(nn.Module):
    """Wraps a frozen nn.Conv2d with a trainable low-rank delta via 1×1 convolutions.

    forward(x) = frozen_conv(x) + (lora_B(lora_A(x))) * (alpha / rank)

    The low-rank path uses 1×1 convolutions regardless of the original kernel size,
    effectively learning a channel-mixing residual. lora_B is zero-initialized so
    the output equals the frozen conv at initialization.
    """

    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.scaling = alpha / rank

        in_channels = conv.in_channels
        out_channels = conv.out_channels

        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def apply_lora_to_dpt_head(
    head: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    target_kernel_sizes: Optional[List[int]] = None,
    target_layer_types: Optional[List[str]] = None,
) -> int:
    """Freeze the DPTHead and inject LoRA adapters into Conv2d/Linear layers.

    Args:
        head:                The DPTHead module (or any nn.Module).
        rank:                LoRA rank.
        alpha:               LoRA alpha (scaling = alpha / rank).
        dropout:             Dropout applied in the LoRA path.
        target_modules:      List of sub-module name prefixes to target.
                             If None, no name-based filtering (all names match).
                             Examples: ["projects", "scratch"] or ["projects", "resize_layers"].
        target_kernel_sizes: List of kernel sizes to target for Conv2d layers.
                             If None, all kernel sizes match.
                             Examples: [1] for only 1×1 convs, [1, 3] for 1×1 and 3×3.
        target_layer_types:  List of layer type strings to target: "conv2d", "linear".
                             If None, both Conv2d and Linear are targeted.
                             Examples: ["conv2d"] to skip Linear layers.

    Returns:
        Number of LoRA-wrapped layers.
    """
    # Step 1: Freeze all existing parameters
    for p in head.parameters():
        p.requires_grad_(False)

    # Step 2: Determine filters
    if target_modules is not None:
        target_prefixes: Optional[Set[str]] = set(target_modules)
    else:
        target_prefixes = None

    allowed_kernel_sizes: Optional[Set[int]] = (
        set(target_kernel_sizes) if target_kernel_sizes is not None else None
    )

    allowed_types: Optional[Set[str]] = (
        {t.lower() for t in target_layer_types} if target_layer_types is not None else None
    )

    def _matches_prefix(name: str) -> bool:
        if target_prefixes is None:
            return True
        for prefix in target_prefixes:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False

    def _matches_conv(child: nn.Conv2d) -> bool:
        if allowed_types is not None and "conv2d" not in allowed_types:
            return False
        if allowed_kernel_sizes is not None:
            k = child.kernel_size
            # kernel_size is (h, w); we check if both dims match any allowed size
            if not (k[0] in allowed_kernel_sizes and k[1] in allowed_kernel_sizes):
                return False
        return True

    def _matches_linear(child: nn.Linear) -> bool:
        if allowed_types is not None and "linear" not in allowed_types:
            return False
        return True

    # Step 3: Walk module tree and replace Conv2d/Linear with LoRA wrappers
    count = 0
    replacements = []  # (parent, attr_name, wrapped_module)

    for name, module in head.named_modules():
        if not _matches_prefix(name):
            continue
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if not _matches_prefix(full_name):
                continue
            if isinstance(child, nn.Conv2d) and _matches_conv(child):
                wrapped = LoRAConv2d(child, rank, alpha, dropout)
                replacements.append((module, child_name, wrapped))
                count += 1
            elif isinstance(child, nn.Linear) and _matches_linear(child):
                wrapped = LoRALinear(child, rank, alpha, dropout)
                replacements.append((module, child_name, wrapped))
                count += 1

    # Apply replacements
    for parent, attr_name, wrapped in replacements:
        setattr(parent, attr_name, wrapped)

    logger.info(
        f"LoRA applied to DPTHead: {count} layers wrapped "
        f"(rank={rank}, alpha={alpha}, dropout={dropout}, "
        f"targets={target_modules or 'all'}, "
        f"kernel_sizes={target_kernel_sizes or 'all'}, "
        f"layer_types={target_layer_types or 'all'})"
    )
    return count

