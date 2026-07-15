"""Camera feature encoding for Temporal-VGGT v1.

Builds a 13-dim camera feature from intrinsics and extrinsics, then projects
it to the token dimension via a small MLP.

Camera feature layout (13 dims total):
    [rot6d (6), t_norm (3), fx_norm (1), fy_norm (1), cx_norm (1), cy_norm (1)]
"""
import torch
import torch.nn as nn


class CameraEmbedding(nn.Module):
    """Projects a 13-dim camera feature vector to token dimension via MLP."""

    def __init__(self, cam_dim: int = 13, token_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cam_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, camera_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_feat: [..., 13]
        Returns:
            embedding: [..., token_dim]
        """
        return self.net(camera_feat)


def build_camera_features(
    transform_matrix: torch.Tensor,  # [..., 4, 4] cam-to-world
    fl_x: torch.Tensor,              # [...] focal length x (pixels)
    fl_y: torch.Tensor,              # [...] focal length y (pixels)
    cx: torch.Tensor,                # [...] principal point x (pixels)
    cy: torch.Tensor,                # [...] principal point y (pixels)
    img_w: torch.Tensor,             # [...] or scalar image width
    img_h: torch.Tensor,             # [...] or scalar image height
    avg_pos: torch.Tensor,           # [..., 3] or [3] scene-center for translation norm
    scale: torch.Tensor,             # [...] or scalar scene scale for translation norm
    **_ignored,                      # absorbs extra keys (distortion etc.)
) -> torch.Tensor:
    """Build the 13-dim camera feature vector.

    Extrinsics: rotation 6D encoding (first two columns of R) + normalized translation.
    Intrinsics: normalized focal lengths and principal point.

    Returns:
        camera_feat: [..., 13]
    """
    R = transform_matrix[..., :3, :3]  # [..., 3, 3]
    t = transform_matrix[..., :3, 3]   # [..., 3]

    # Rotation 6D: first two columns of R, flattened
    rot6d = R[..., :2].reshape(*R.shape[:-2], 6)  # [..., 6]

    # Normalized translation: (t - avg_pos) * scale
    # avg_pos may be [3] or [..., 3]; scale may be scalar or [...]
    avg_pos = avg_pos.to(t)
    scale = scale.to(t)

    # Broadcast avg_pos and scale to match t's shape if needed
    while avg_pos.dim() < t.dim():
        avg_pos = avg_pos.unsqueeze(-2)
    while scale.dim() < t.dim() - 1:
        scale = scale.unsqueeze(-1)

    t_norm = (t - avg_pos) * scale.unsqueeze(-1)  # [..., 3]

    # Normalized intrinsics
    fl_x = fl_x.to(t)
    fl_y = fl_y.to(t)
    cx = cx.to(t)
    cy = cy.to(t)
    img_w = img_w.to(t) if isinstance(img_w, torch.Tensor) else torch.tensor(img_w, dtype=t.dtype, device=t.device)
    img_h = img_h.to(t) if isinstance(img_h, torch.Tensor) else torch.tensor(img_h, dtype=t.dtype, device=t.device)

    fx_norm = (fl_x / img_w).unsqueeze(-1)  # [..., 1]
    fy_norm = (fl_y / img_h).unsqueeze(-1)
    cx_norm = (cx / img_w).unsqueeze(-1)
    cy_norm = (cy / img_h).unsqueeze(-1)

    return torch.cat([rot6d, t_norm, fx_norm, fy_norm, cx_norm, cy_norm], dim=-1)  # [..., 13]

