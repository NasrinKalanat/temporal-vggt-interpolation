"""Target-view temporal VGGT cached-feature decoder.

Predicts VGGT cached layers for target t2 camera views from t1/t3 cached
features. Unlike the residual endpoint model, this model does not add an
update to t1 features; it decodes a fresh t2 feature grid from learned target
queries conditioned on t2 camera rays.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from vggt.heads.dpt_head import DPTHead
from vggt.layers.mlp import Mlp
from vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.models.vggt import VGGT

from models.feature_cache_utils import run_cached_endpoint
from models.time_encoding import EndpointTimeEncoder, build_relative_gap_features

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_LAYERS = [4, 11, 17, 23]


class RayMLP(nn.Module):
    """Map per-patch camera-ray features to token offsets."""

    def __init__(self, ray_dim: int = 8, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ray_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, ray_feat: torch.Tensor) -> torch.Tensor:
        return self.net(ray_feat)


class CrossAttention(nn.Module):
    """Pre-norm cross-attention from target query tokens to source memory."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        batch, num_q, channels = q.shape
        num_kv = kv.shape[1]
        heads, head_dim = self.num_heads, self.head_dim

        q_heads = self.q_proj(self.norm_q(q)).view(batch, num_q, heads, head_dim).transpose(1, 2)
        kv_norm = self.norm_kv(kv)
        k_heads = self.k_proj(kv_norm).view(batch, num_kv, heads, head_dim).transpose(1, 2)
        v_heads = self.v_proj(kv_norm).view(batch, num_kv, heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, num_q, channels)
        return q + self.out_proj(out)


class TargetViewTemporalDecoderBlock(nn.Module):
    """Self-attention on t2 query + cross-attention to t1/t3 memory + MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        rope: Optional[RotaryPositionEmbedding2D] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout
        self.rope = rope

        self.norm_sa = nn.LayerNorm(dim)
        self.qkv_sa = nn.Linear(dim, 3 * dim)
        self.out_sa = nn.Linear(dim, dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, num_tokens, channels = query.shape
        heads, head_dim = self.num_heads, self.head_dim

        qkv = self.qkv_sa(self.norm_sa(query)).reshape(
            batch, num_tokens, 3, heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        q_heads, k_heads, v_heads = qkv.unbind(0)
        if self.rope is not None and pos is not None:
            q_heads = self.rope(q_heads, pos)
            k_heads = self.rope(k_heads, pos)

        attn_out = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        query = query + self.out_sa(attn_out.transpose(1, 2).reshape(batch, num_tokens, channels))
        query = self.cross_attn(query, memory)
        query = query + self.ffn(self.norm_ffn(query))
        return query


class TargetViewTemporalVGGT(nn.Module):
    """Predict t2 cached VGGT features on explicit t2 target camera views."""

    target_view_cache_requires_query_order = True

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        embed_dim: int = 1024,
        vggt_dim: Optional[int] = None,
        d_model: int = 512,
        num_query_views: int = 16,
        num_decoder_blocks: int = 1,
        num_transformer_layers: Optional[int] = None,
        num_heads: int = 8,
        num_transformer_heads: Optional[int] = None,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        ray_dim: int = 8,
        time_hidden_dim: int = 256,
        cache_layers: Optional[List[int]] = None,
        cached_layers: Optional[List[int]] = None,
        patch_start_idx: int = 5,
        image_size: int = 518,
        patch_size: int = 14,
        image_preprocess_mode: str = "pad",
        use_rope_in_self_attention: bool = True,
        init_point_head_from_vggt: bool = True,
        use_gradient_checkpoint: bool = False,
        skip_aggregator: bool = False,
        freeze_vggt_backbone: bool = True,
        freeze_point_head: bool = True,
        train_point_head: bool = False,
        point_head_mode: str = "full",
        **_unused,
    ):
        super().__init__()
        del num_query_views, freeze_vggt_backbone, train_point_head, point_head_mode

        if cache_layers is None:
            cache_layers = cached_layers or _DEFAULT_CACHE_LAYERS
        assert len(cache_layers) == 4, "VGGT DPTHead expects exactly 4 cache layers"

        self.embed_dim = embed_dim
        self.D = int(vggt_dim or (2 * embed_dim))
        self.d_model = d_model
        self.cache_layers = sorted(cache_layers)
        self.cache_set = set(self.cache_layers)
        self.patch_start_idx = patch_start_idx
        self.image_size = image_size
        self.image_preprocess_mode = image_preprocess_mode
        self.patch_size = patch_size
        self.patch_h = image_size // patch_size
        self.patch_w = image_size // patch_size
        self.num_patch_tokens = self.patch_h * self.patch_w
        self.num_tokens = patch_start_idx + self.num_patch_tokens
        self.use_gradient_checkpoint = use_gradient_checkpoint

        num_layers = num_transformer_layers if num_transformer_layers is not None else num_decoder_blocks
        heads = num_transformer_heads if num_transformer_heads is not None else num_heads

        if skip_aggregator:
            logger.info("skip_aggregator=True: aggregator not loaded (feature cache assumed full)")
            self.aggregator = None
            vggt = VGGT.from_pretrained(vggt_model_id) if init_point_head_from_vggt else None
        else:
            logger.info(f"Loading VGGT from {vggt_model_id!r}")
            vggt = VGGT.from_pretrained(vggt_model_id)
            self.aggregator = vggt.aggregator
            self.patch_start_idx = self.aggregator.patch_start_idx
            for param in self.aggregator.parameters():
                param.requires_grad_(False)

        self.down_norms = nn.ModuleList([nn.LayerNorm(self.D) for _ in self.cache_layers])
        self.down_projs = nn.ModuleList([nn.Linear(self.D, d_model) for _ in self.cache_layers])
        self.up_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.cache_layers])
        self.up_projs = nn.ModuleList([nn.Linear(d_model, self.D) for _ in self.cache_layers])
        for proj in list(self.down_projs) + list(self.up_projs):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        self.learned_t2_queries = nn.ParameterList([
            nn.Parameter(torch.empty(1, 1, self.num_tokens, d_model))
            for _ in self.cache_layers
        ])
        for query in self.learned_t2_queries:
            nn.init.trunc_normal_(query, std=0.02)

        self.ray_mlp = RayMLP(ray_dim=ray_dim, d_model=d_model)
        self.time_encoder = EndpointTimeEncoder(out_dim=d_model, hidden_dim=time_hidden_dim)

        rope = RotaryPositionEmbedding2D(frequency=100) if use_rope_in_self_attention else None
        self.position_getter = PositionGetter()
        self.decoder_blocks = nn.ModuleList([
            TargetViewTemporalDecoderBlock(
                dim=d_model,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope=rope,
            )
            for _ in range(num_layers)
        ])

        if init_point_head_from_vggt and vggt is not None and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
            self.point_head.intermediate_layer_idx = [0, 1, 2, 3]
        else:
            self.point_head = DPTHead(
                dim_in=self.D,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
        if freeze_point_head:
            for param in self.point_head.parameters():
                param.requires_grad_(False)
        if vggt is not None:
            del vggt

    def train(self, mode: bool = True):
        super().train(mode)
        if self.aggregator is not None:
            self.aggregator.eval()
        return self

    def _run_vggt_endpoint(self, images: torch.Tensor, B: int, S: int) -> List[torch.Tensor]:
        """Run frozen VGGT on one endpoint and collect configured cache layers."""
        agg = self.aggregator
        H_img, W_img = images.shape[-2:]
        patch_h = H_img // agg.patch_size
        patch_w = W_img // agg.patch_size

        with torch.no_grad():
            imgs_norm = (images - agg._resnet_mean) / agg._resnet_std
            patch_tokens = agg.patch_embed(imgs_norm.view(B * S, 3, H_img, W_img))
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            _, _, channels = patch_tokens.shape
            del imgs_norm

            camera_tok = slice_expand_and_flatten(agg.camera_token, B, S)
            register_tok = slice_expand_and_flatten(agg.register_token, B, S)
            tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)
            del camera_tok, register_tok, patch_tokens
            _, num_tokens, _ = tokens.shape

            pos = None
            if agg.rope is not None:
                pos = agg.position_getter(B * S, patch_h, patch_w, device=tokens.device)
                pos = pos + 1
                pos_special = torch.zeros(
                    B * S, agg.patch_start_idx, 2,
                    device=tokens.device, dtype=pos.dtype,
                )
                pos = torch.cat([pos_special, pos], dim=1)

            frame_idx = 0
            global_idx = 0
            cached: List[torch.Tensor] = []
            for layer_count in range(agg.aa_block_num):
                tokens, frame_idx, frame_ints = agg._process_frame_attention(
                    tokens, B, S, num_tokens, channels, frame_idx, pos=pos
                )
                tokens, global_idx, global_ints = agg._process_global_attention(
                    tokens, B, S, num_tokens, channels, global_idx, pos=pos
                )
                frame_feat = frame_ints[0]
                global_feat = global_ints[0]
                del frame_ints, global_ints
                if layer_count in self.cache_set:
                    cached.append(torch.cat([frame_feat, global_feat], dim=-1))
                del frame_feat, global_feat
        return cached

    def _run_vggt_endpoints(
        self,
        images_t1,
        images_t3,
        B: int,
        t1_key=None,
        t3_key=None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            run_cached_endpoint(self, images_t1, B, t1_key, "t1"),
            run_cached_endpoint(self, images_t3, B, t3_key, "t3"),
        )

    def _time_offset(
        self,
        endpoint_date: torch.Tensor,
        target_date: torch.Tensor,
        rel_gap: torch.Tensor,
    ) -> torch.Tensor:
        return self.time_encoder.additive(endpoint_date, target_date, rel_gap)[:, None, None, :]

    def _scaled_intrinsics(
        self,
        camera: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl_x = camera["fl_x"]
        fl_y = camera["fl_y"]
        cx = camera["cx"]
        cy = camera["cy"]
        img_w = camera["img_w"]
        img_h = camera["img_h"]

        if self.image_preprocess_mode == "pad":
            image_size = torch.as_tensor(self.image_size, dtype=fl_x.dtype, device=fl_x.device)
            scale = torch.minimum(image_size / img_w, image_size / img_h)
            pad_left = (image_size - img_w * scale) * 0.5
            pad_top = (image_size - img_h * scale) * 0.5
            return fl_x * scale, fl_y * scale, cx * scale + pad_left, cy * scale + pad_top

        width = torch.as_tensor(self.image_size, dtype=fl_x.dtype, device=fl_x.device)
        height = torch.as_tensor(self.image_size, dtype=fl_x.dtype, device=fl_x.device)
        return fl_x * (width / img_w), fl_y * (height / img_h), cx * (width / img_w), cy * (height / img_h)

    def _build_ray_features(self, camera: Dict[str, torch.Tensor], num_patch_tokens: int) -> torch.Tensor:
        if num_patch_tokens != self.num_patch_tokens:
            side = int(math.sqrt(num_patch_tokens))
            if side * side != num_patch_tokens:
                raise ValueError(
                    f"Cannot build square patch ray grid for {num_patch_tokens} patch tokens."
                )
            patch_h = patch_w = side
            patch_size_y = self.image_size / patch_h
            patch_size_x = self.image_size / patch_w
        else:
            patch_h, patch_w = self.patch_h, self.patch_w
            patch_size_y = patch_size_x = self.patch_size

        device = camera["transform_matrix"].device
        dtype = camera["transform_matrix"].dtype
        ys = (torch.arange(patch_h, device=device, dtype=dtype) + 0.5) * patch_size_y
        xs = (torch.arange(patch_w, device=device, dtype=dtype) + 0.5) * patch_size_x
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        u = uu.reshape(1, 1, -1)
        v = vv.reshape(1, 1, -1)

        fl_x, fl_y, cx, cy = self._scaled_intrinsics(camera)
        x_cam = (u - cx.unsqueeze(-1)) / fl_x.unsqueeze(-1)
        y_cam = (v - cy.unsqueeze(-1)) / fl_y.unsqueeze(-1)
        z_cam = torch.ones_like(x_cam)
        ray_cam = F.normalize(torch.stack([x_cam, y_cam, z_cam], dim=-1), dim=-1)

        transform = camera["transform_matrix"]
        rotation = transform[..., :3, :3]
        ray_world = torch.einsum("bsij,bspj->bspi", rotation, ray_cam)
        ray_world = F.normalize(ray_world, dim=-1)

        center = transform[..., :3, 3]
        avg_pos = camera["avg_pos"].to(center)
        scale = camera["scale"].to(center)
        center_norm = (center - avg_pos.unsqueeze(1)) * scale.view(-1, 1, 1)
        center_norm = center_norm.unsqueeze(2).expand(-1, -1, num_patch_tokens, -1)

        u_norm = (2.0 * (u / float(self.image_size)) - 1.0).expand_as(x_cam).unsqueeze(-1)
        v_norm = (2.0 * (v / float(self.image_size)) - 1.0).expand_as(y_cam).unsqueeze(-1)

        return torch.cat([ray_world, center_norm, u_norm, v_norm], dim=-1)

    def _add_ray_embedding(self, tokens: torch.Tensor, ray_feat: torch.Tensor) -> torch.Tensor:
        special = tokens[:, :, : self.patch_start_idx]
        patches = tokens[:, :, self.patch_start_idx :]
        if patches.shape[2] != ray_feat.shape[2]:
            raise ValueError(
                f"Ray/token patch mismatch: tokens have {patches.shape[2]} patches, "
                f"rays have {ray_feat.shape[2]}."
            )
        patches = patches + self.ray_mlp(ray_feat)
        return torch.cat([special, patches], dim=2)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]
        batch_size = batch["date_t1"].shape[0]

        if "rel_gap_feat" in batch:
            rel_gap = batch["rel_gap_feat"].to(device)
        else:
            rel_gap = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)
        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)

        cached_t1, cached_t3 = self._run_vggt_endpoints(
            images_t1, images_t3, batch_size,
            t1_key=batch.get("t1_cache_key"),
            t3_key=batch.get("t3_cache_key"),
        )

        camera_t1 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["camera_t1"].items()}
        camera_t3 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["camera_t3"].items()}
        camera_t2 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["camera_t2_query"].items()}

        num_t1 = camera_t1["transform_matrix"].shape[-3]
        num_t3 = camera_t3["transform_matrix"].shape[-3]
        num_t2 = camera_t2["transform_matrix"].shape[-3]
        if cached_t1[0].shape[1] > num_t1:
            cached_t1 = [features[:, :num_t1] for features in cached_t1]
        if cached_t3[0].shape[1] > num_t3:
            cached_t3 = [features[:, :num_t3] for features in cached_t3]

        num_tokens = cached_t1[0].shape[2]
        num_patch_tokens = num_tokens - self.patch_start_idx
        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} VGGT cache tokens, got {num_tokens}. "
                "Set image_size/patch_size/patch_start_idx to match the cached features."
            )

        ray_t1 = self._build_ray_features(camera_t1, num_patch_tokens)
        ray_t3 = self._build_ray_features(camera_t3, num_patch_tokens)
        ray_t2 = self._build_ray_features(camera_t2, num_patch_tokens)

        time_t1 = self._time_offset(date_t1, date_t2, rel_gap)
        time_t3 = self._time_offset(date_t3, date_t2, rel_gap)
        time_t2 = self._time_offset(date_t2, date_t2, rel_gap)
        pos_t2 = self.position_getter(batch_size, num_t2, num_tokens, device)

        pred_cached_layers: List[torch.Tensor] = []
        for layer_idx in range(len(self.cache_layers)):
            f1 = cached_t1.pop(0)
            f3 = cached_t3.pop(0)

            f1_low = self.down_projs[layer_idx](self.down_norms[layer_idx](f1))
            f3_low = self.down_projs[layer_idx](self.down_norms[layer_idx](f3))
            f1_low = self._add_ray_embedding(f1_low + time_t1, ray_t1)
            f3_low = self._add_ray_embedding(f3_low + time_t3, ray_t3)

            memory = torch.cat(
                [
                    f1_low.reshape(batch_size, num_t1 * num_tokens, self.d_model),
                    f3_low.reshape(batch_size, num_t3 * num_tokens, self.d_model),
                ],
                dim=1,
            )
            del f1, f3, f1_low, f3_low

            target_query = self.learned_t2_queries[layer_idx].expand(
                batch_size, num_t2, num_tokens, self.d_model
            )
            target_query = self._add_ray_embedding(target_query + time_t2, ray_t2)
            query = target_query.reshape(batch_size, num_t2 * num_tokens, self.d_model)
            del target_query

            hidden = query
            for block in self.decoder_blocks:
                if self.use_gradient_checkpoint and self.training:
                    hidden = grad_checkpoint(block, hidden, memory, pos_t2, use_reentrant=False)
                else:
                    hidden = block(hidden, memory, pos_t2)
            del memory

            hidden = hidden.reshape(batch_size, num_t2, num_tokens, self.d_model)
            pred = self.up_projs[layer_idx](self.up_norms[layer_idx](hidden))
            pred_cached_layers.append(pred)

        if getattr(self, "freeze_point_head", False):
            return {"pred_cached_layers": pred_cached_layers}

        mock_images = torch.zeros(
            batch_size, num_t2, 3, self.image_size, self.image_size,
            device=device,
            dtype=pred_cached_layers[0].dtype,
        )
        pred_points, pred_conf = self.point_head(
            pred_cached_layers,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )
        return {
            "pred_points": pred_points,
            "pred_conf": pred_conf,
            "pred_cached_layers": pred_cached_layers,
        }

    def trainable_parameter_groups(self) -> List[Dict]:
        return [{"params": [param for param in self.parameters() if param.requires_grad]}]
