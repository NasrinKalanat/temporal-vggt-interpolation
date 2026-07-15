"""Point-map prediction heads for Temporal-VGGT v4.

All heads share the same interface:
    __init__(dim_in, patch_size, **kwargs)
    forward(x: [B, C, Hp, Wp]) -> [B, 4, H, W]   raw output (activations applied by caller)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GELU(),
    )


class PointHeadSmall(nn.Module):
    """Lightweight point head with two upsampling stages.

    LN → 1×1 Conv(C,128) → ConvBlock(128,128) → ×2 upsample
       → ConvBlock(128,64) → upsample to Hout,Wout
       → ConvBlock(64,32) → 1×1 Conv(32,4)
    """

    def __init__(self, dim_in: int = 1024, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.norm   = nn.LayerNorm(dim_in)
        self.proj   = nn.Conv2d(dim_in, 128, kernel_size=1)
        self.block1 = _conv_block(128, 128)
        self.block2 = _conv_block(128, 64)
        self.block3 = _conv_block(64, 32)
        self.out    = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, Hp, Wp] token grid
        Returns:
            [B, 4, H, W] raw point-map output
        """
        _, _, Hp, Wp = x.shape
        Hout, Wout = Hp * self.patch_size, Wp * self.patch_size

        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.proj(x)                                                             # [B, 128, Hp,   Wp  ]
        x = self.block1(x)                                                           # [B, 128, Hp,   Wp  ]
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # [B, 128, Hp*2, Wp*2]
        x = self.block2(x)                                                           # [B,  64, Hp*2, Wp*2]
        if x.shape[-2:] != (Hout, Wout):
            x = F.interpolate(x, size=(Hout, Wout), mode="bilinear", align_corners=False)
        x = self.block3(x)                                                           # [B,  32, Hout, Wout]
        return self.out(x)                                                           # [B,   4, Hout, Wout]


class PointHeadLarge(nn.Module):
    """Multi-scale point head with progressive upsampling.

    LN → 1×1 Conv(C,256) → ConvBlock(256,256) → ×2 upsample
       → ConvBlock(256,128) → ×2 upsample
       → ConvBlock(128,64) → upsample to Hout,Wout
       → ConvBlock(64,32) → 1×1 Conv(32,4)
    """

    def __init__(self, dim_in: int = 1024, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.norm   = nn.LayerNorm(dim_in)
        self.proj   = nn.Conv2d(dim_in, 256, kernel_size=1)
        self.block1 = _conv_block(256, 256)
        self.block2 = _conv_block(256, 128)
        self.block3 = _conv_block(128, 64)
        self.block4 = _conv_block(64, 32)
        self.out    = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, Hp, Wp] token grid
        Returns:
            [B, 4, H, W] raw point-map output
        """
        _, _, Hp, Wp = x.shape
        Hout, Wout = Hp * self.patch_size, Wp * self.patch_size

        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.proj(x)                                                              # [B, 256, Hp,   Wp  ]
        x = self.block1(x)                                                            # [B, 256, Hp,   Wp  ]
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)   # [B, 256, Hp*2, Wp*2]
        x = self.block2(x)                                                            # [B, 128, Hp*2, Wp*2]
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)   # [B, 128, Hp*4, Wp*4]
        x = self.block3(x)                                                            # [B,  64, Hp*4, Wp*4]
        if x.shape[-2:] != (Hout, Wout):
            x = F.interpolate(x, size=(Hout, Wout), mode="bilinear", align_corners=False)
        x = self.block4(x)                                                            # [B,  32, Hout, Wout]
        return self.out(x)                                                            # [B,   4, Hout, Wout]


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "PointHeadSmall": PointHeadSmall,
    "PointHeadLarge": PointHeadLarge,
}


def get_point_head_class(name: str) -> type:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown point head class {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]

