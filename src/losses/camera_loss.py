"""Camera pose loss for TemporalVGGTv1.

Supervises the camera head against teacher (VGGT) camera predictions for the
t2 query views.  Uses the same absT_quaR_FoV encoding and stage-weighted L1
loss as the original VGGT camera loss.

Expected batch keys (set by TemporalTripletDataset):
    target_vggt_extrinsic_t2:  [B, Q, 3, 4]  world-to-camera (VGGT space)
    target_vggt_intrinsic_t2:  [B, Q, 3, 3]  camera intrinsics (pixels)

Expected prediction keys (set by TemporalVGGTv1.forward when use_camera_head=True):
    pred_pose_enc_list_t2q:    list of [B, Q, 9]  one tensor per refinement iter
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def camera_loss_t2q(
    predictions: dict,
    batch: dict,
    image_size_hw: tuple[int, int],
    gamma: float = 0.6,
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    weight_focal: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Stage-weighted L1 camera loss against teacher predictions.

    Args:
        predictions:  model output dict (must contain 'pred_pose_enc_list_t2q')
        batch:        data batch (must contain teacher extrinsics / intrinsics)
        image_size_hw: (H, W) of the preprocessed images, needed to compute FoV
        gamma:        decay weight for earlier refinement stages (final stage = 1.0)
        weight_trans: loss weight for translation component
        weight_rot:   loss weight for rotation (quaternion) component
        weight_focal: loss weight for FoV component

    Returns:
        dict with keys: loss_camera, loss_T, loss_R, loss_FL
    """
    pred_list = predictions["pred_pose_enc_list_t2q"]   # list of [B, Q, 9]
    n_stages = len(pred_list)

    gt_ext  = batch["target_vggt_extrinsic_t2"].to(pred_list[0].device)   # [B, Q, 3, 4]
    gt_intr = batch["target_vggt_intrinsic_t2"].to(pred_list[0].device)   # [B, Q, 3, 3]

    # Convert teacher cameras to the same absT_quaR_FoV encoding
    gt_pose = extri_intri_to_pose_encoding(gt_ext, gt_intr, image_size_hw)  # [B, Q, 9]

    total_T = total_R = total_FL = gt_pose.new_zeros(1).squeeze()

    for stage_idx, pred_pose in enumerate(pred_list):
        stage_w = gamma ** (n_stages - stage_idx - 1)

        loss_T  = (pred_pose[..., :3] - gt_pose[..., :3]).abs().mean()
        loss_R  = (pred_pose[..., 3:7] - gt_pose[..., 3:7]).abs().mean()
        loss_FL = (pred_pose[..., 7:]  - gt_pose[..., 7:]).abs().mean()

        total_T  = total_T  + stage_w * loss_T
        total_R  = total_R  + stage_w * loss_R
        total_FL = total_FL + stage_w * loss_FL

    avg_T  = total_T  / n_stages
    avg_R  = total_R  / n_stages
    avg_FL = total_FL / n_stages

    loss = weight_trans * avg_T + weight_rot * avg_R + weight_focal * avg_FL

    return {
        "loss_camera": loss,
        "loss_T":  avg_T,
        "loss_R":  avg_R,
        "loss_FL": avg_FL,
    }

