Below is a detailed implementation document for **TRDM: Temporal Residual Depth-Map Network**. I made it a single concrete plan, with depth as the primary representation. It uses `depth_map.npy` / `depth_confidence.npy` as the main training signal, which matches your existing depth-loading and depth-loss direction. 

# TRDM: Temporal Residual Depth-Map Network

## 1. Goal

Implement **TRDM**, a camera-aware temporal model that predicts the intermediate-date target depth map `D2_hat` from endpoint dates `t1` and `t3`.

The model predicts geometry through depth:

```text
D2_hat: [B, 1, H, W]
```

The predicted depth map is later unprojected with the target camera intrinsics and pose to produce a 3D point cloud for evaluation.

The model does not predict 3D coordinates directly. It predicts a target-view depth image, and the camera model converts that depth image into geometry.

---

## 2. Core Formulation

TRDM uses residual temporal interpolation in **log-depth space**.

The prediction equation is:

```text
D2_hat = exp(log(D1_to_t2 + eps) + tau * G * ΔlogD)
```

where:

```text
D1_to_t2   = t1 depth warped into the target t2 camera view
D2_hat     = predicted t2 depth map
tau        = temporal position of t2 between t1 and t3
G          = learned per-pixel gate in [0, 1]
ΔlogD      = learned per-pixel log-depth residual
eps        = small constant for numerical stability
```

The gate controls where temporal geometry changes should be applied.

The residual is predicted in log-depth space because depth must remain positive and multiplicative depth changes are easier to model than raw additive changes.

---

## 3. Required Files

For each date, the dataset should load:

```text
depth_map.npy
depth_confidence.npy
extrinsic.npy
intrinsic.npy
dataset_cameras.json
```

The model should use:

```text
t1 depth and confidence
t2 depth and confidence
t3 depth and confidence
t1 camera intrinsics and pose
t2 camera intrinsics and pose
t3 camera intrinsics and pose
date information for t1, t2, t3
```

The target `t2` depth map is used only for training supervision.

At inference, the model uses:

```text
D1, C1, K1, T1_c2w
D3, C3, K3, T3_c2w
K2, T2_c2w
date_t1, date_t2, date_t3
```

and predicts:

```text
D2_hat
```

---

## 4. Tensor Shapes

Use the following shape convention:

```text
B   = batch size
V3  = number of t3 context views
H   = depth map height
W   = depth map width
K   = number of sampled t3 context points
```

Training tensors:

```text
D1:       [B, 1, H, W]
C1:       [B, 1, H, W]

D2:       [B, 1, H, W]
C2:       [B, 1, H, W]

D3:       [B, V3, 1, H, W]
C3:       [B, V3, 1, H, W]

K1:       [B, 3, 3]
K2:       [B, 3, 3]
K3:       [B, V3, 3, 3]

T1_c2w:   [B, 4, 4]
T2_c2w:   [B, 4, 4]
T3_c2w:   [B, V3, 4, 4]

tau:      [B, 1]
```

Model outputs:

```text
D2_hat:     [B, 1, H, W]
G:          [B, 1, H, W]
delta_logD: [B, 1, H, W]
```

---

## 5. Camera Convention

Use the same camera convention consistently across:

```text
depth unprojection
depth warping
ray construction
evaluation export
```

The implementation should use camera-to-world matrices:

```text
T_c2w: [B, 4, 4]
```

To transform world points into a target camera frame, compute:

```text
T_w2c = inverse(T_c2w)
```

Then:

```text
P_cam = T_w2c @ P_world
```

Depth is defined as the camera-frame forward-axis coordinate. For this implementation, use the standard pinhole camera convention:

```text
x = (u - cx) / fx * depth
y = (v - cy) / fy * depth
z = depth
```

So camera-frame depth is `z`.

---

## 6. Module Overview

TRDM contains these components:

```text
1. depth warping module
2. target ray-map builder
3. t3 depth-context sampler
4. t3 PointNet encoder
5. relative pose encoder
6. time encoder
7. fusion MLP
8. U-Net encoder-decoder
9. depth residual head
10. gate head
11. depth losses
12. depth unprojection/export function
```

The high-level flow is:

```text
D1 + K1 + T1_c2w + K2 + T2_c2w
        ↓
warp t1 depth into target t2 view
        ↓
D1_to_t2, C1_to_t2, M1_to_t2

D3 + C3 + K3 + T3_c2w + T2_c2w
        ↓
sample and transform t3 depth context into t2 camera frame
        ↓
z3

relative poses + time features
        ↓
z_pose, z_time

D1_to_t2 + C1_to_t2 + M1_to_t2 + xy + target rays + tau
        ↓
U-Net with FiLM conditioning
        ↓
delta_logD, G
        ↓
D2_hat
```

---

## 7. Depth Warping: `D1` to Target `t2` View

Before passing `D1` into the network, warp it into the target `t2` camera view.

### 7.1 Unproject `D1` to 3D

For every valid pixel `(u, v)` in `D1`:

```python
x = (u - cx1) / fx1 * D1
y = (v - cy1) / fy1 * D1
z = D1
```

This creates:

```text
P1_cam1: [B, H, W, 3]
```

Convert to homogeneous coordinates:

```text
P1_cam1_h: [B, H, W, 4]
```

Then transform to world:

```text
P1_world = T1_c2w @ P1_cam1_h
```

### 7.2 Transform `P1_world` into the `t2` camera frame

```text
T2_w2c = inverse(T2_c2w)
P1_cam2 = T2_w2c @ P1_world
```

### 7.3 Project into the `t2` image plane

For each transformed point:

```python
u2 = fx2 * X2 / Z2 + cx2
v2 = fy2 * Y2 / Z2 + cy2
```

where:

```text
P1_cam2 = [X2, Y2, Z2]
```

Only keep points where:

```text
Z2 > 0
0 <= u2 < W
0 <= v2 < H
C1 > confidence threshold
```

### 7.4 Rasterize with z-buffer

Multiple source pixels may project to the same target pixel. Keep the closest valid point by depth:

```text
D1_to_t2[v2, u2] = min Z2
```

Also produce:

```text
C1_to_t2: confidence of selected projected point
M1_to_t2: binary valid mask
```

Final warped tensors:

```text
D1_to_t2: [B, 1, H, W]
C1_to_t2: [B, 1, H, W]
M1_to_t2: [B, 1, H, W]
```

Use nearest-neighbor splatting first. Do not use bilinear splatting in the first implementation.

---

## 8. Target Ray Map

Build a target ray map from `K2`.

For every target pixel `(u, v)`:

```python
x = (u - cx2) / fx2
y = (v - cy2) / fy2
ray = normalize([x, y, 1])
```

Output:

```text
ray2: [B, 3, H, W]
```

Also build normalized pixel coordinates:

```text
xy: [B, 2, H, W]
```

with:

```python
x_norm = 2 * u / (W - 1) - 1
y_norm = 2 * v / (H - 1) - 1
```

---

## 9. Encoder Input

The U-Net input has exactly 9 channels:

```text
D1_to_t2     1
C1_to_t2     1
M1_to_t2     1
xy grid      2
target ray2  3
tau_map      1
----------------
total        9
```

Shape:

```text
x_in: [B, 9, H, W]
```

Before concatenation:

```python
D1_to_t2_safe = torch.where(
    M1_to_t2 > 0,
    torch.log(D1_to_t2.clamp_min(eps)),
    torch.zeros_like(D1_to_t2),
)
```

Use `log(D1_to_t2)` as the depth input, not raw depth.

So the actual first channel is:

```text
log_D1_to_t2
```

The final encoder input is:

```python
x_in = torch.cat(
    [
        log_D1_to_t2,
        C1_to_t2,
        M1_to_t2,
        xy,
        ray2,
        tau_map,
    ],
    dim=1,
)
```

---

## 10. t3 Depth Context

The model should use all available `t3` views as temporal context.

For each `t3` view:

1. Unproject `D3` using `K3`.
2. Transform the resulting 3D points into world coordinates using `T3_c2w`.
3. Transform world points into the target `t2` camera frame using `T2_w2c`.
4. Sample valid confident points.
5. Encode sampled points with PointNet.

### 10.1 Sample Feature Vector

Each sampled `t3` depth point should have 13 features:

```text
xyz in t2 camera frame       3
source depth                 1
confidence                   1
source uv                    2
source ray in t2 frame       3
source camera center in t2   3
-------------------------------
total                       13
```

So:

```text
Q3: [B, K, 13]
```

where `K` is the total number of sampled points across all `t3` views.

Use:

```text
K = 4096
```

per training sample.

If fewer than `4096` valid points exist, sample with replacement.

If more than `4096` valid points exist, random sample without replacement.

### 10.2 PointNet Encoder

Use a small PointNet:

```python
Linear(13, 64)
SiLU
Linear(64, 128)
SiLU
Linear(128, 128)
SiLU
max_pool over K
```

Output:

```text
z3: [B, 128]
```

---

## 11. Relative Pose Encoder

Encode the relative geometry between endpoint cameras and the target camera.

For each sample, compute:

```text
relative t1 camera center in t2 frame:      3
relative t1 forward direction in t2 frame:  3
mean t3 camera center in t2 frame:          3
std t3 camera center in t2 frame:           3
mean t3 forward direction in t2 frame:      3
```

Total pose feature dimension:

```text
15
```

Pass through:

```python
Linear(15, 128)
SiLU
Linear(128, 128)
SiLU
```

Output:

```text
z_pose: [B, 128]
```

---

## 12. Time Encoder

Use temporal features:

```text
tau
1 - tau
left_gap_days
right_gap_days
total_gap_days
```

Normalize day gaps by:

```text
365.0
```

Input:

```text
time_feat: [B, 5]
```

Encoder:

```python
Linear(5, 128)
SiLU
Linear(128, 128)
SiLU
```

Output:

```text
z_time: [B, 128]
```

---

## 13. Conditioning Fusion

Concatenate:

```text
z3:     [B, 128]
z_pose: [B, 128]
z_time: [B, 128]
```

Then:

```text
z_all: [B, 384]
```

Fusion MLP:

```python
Linear(384, 256)
SiLU
Linear(256, 256)
SiLU
```

Output:

```text
z_cond: [B, 256]
```

This conditioning vector is injected into the U-Net decoder using FiLM.

---

## 14. U-Net Backbone

Use a compact U-Net.

### 14.1 Encoder

```text
e0: ConvBlock(9, 32)       → [B, 32, H, W]
e1: DownBlock(32, 64)      → [B, 64, H/2, W/2]
e2: DownBlock(64, 128)     → [B, 128, H/4, W/4]
e3: DownBlock(128, 192)    → [B, 192, H/8, W/8]
```

### 14.2 Decoder

```text
d2: UpBlock(192 + 128, 128) with FiLM(z_cond)
d1: UpBlock(128 + 64, 64)   with FiLM(z_cond)
d0: UpBlock(64 + 32, 32)    with FiLM(z_cond)
```

Final decoder feature:

```text
f: [B, 32, H, W]
```

### 14.3 ConvBlock

Each ConvBlock:

```text
Conv2d
GroupNorm
SiLU
Conv2d
GroupNorm
SiLU
residual skip
```

If input and output channels differ, use a `1x1` convolution for the residual skip.

### 14.4 FiLM

For each decoder block:

```python
gamma_beta = film_mlp(z_cond)
gamma, beta = gamma_beta.chunk(2, dim=1)

feature = feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
```

Use one FiLM MLP per decoder stage.

---

## 15. Output Heads

TRDM has two output heads:

```text
delta_logD_head
gate_head
```

### 15.1 Delta Log-Depth Head

```python
Conv2d(32, 32, kernel_size=3, padding=1)
SiLU
Conv2d(32, 1, kernel_size=3, padding=1)
```

Output:

```text
delta_logD: [B, 1, H, W]
```

Initialize the final convolution to zero:

```python
nn.init.zeros_(delta_logD_head[-1].weight)
nn.init.zeros_(delta_logD_head[-1].bias)
```

This makes the model start from the warped `t1` depth prior.

### 15.2 Gate Head

```python
Conv2d(32, 32, kernel_size=3, padding=1)
SiLU
Conv2d(32, 1, kernel_size=3, padding=1)
Sigmoid
```

Output:

```text
G: [B, 1, H, W]
```

Initialize the final gate bias to a negative value:

```python
nn.init.constant_(gate_head[-2].bias, -2.0)
```

This makes the initial gate approximately:

```text
sigmoid(-2.0) ≈ 0.12
```

So the model starts conservative.

---

## 16. Depth Prediction

Use:

```python
eps = 1e-4

base_logD_valid = torch.log(D1_to_t2.clamp_min(eps))
base_logD_prior = self.learned_log_depth_prior.expand_as(D1_to_t2)

base_logD = torch.where(
    M1_to_t2 > 0,
    base_logD_valid,
    base_logD_prior,
)

pred_logD = base_logD + tau[:, :, None, None] * G * delta_logD

D2_hat = torch.exp(pred_logD).clamp_min(eps)
```

The learned fallback prior is a parameter:

```python
self.learned_log_depth_prior = nn.Parameter(torch.tensor(0.0))
```

Initialize it from the mean log depth of the training set if available. If not available, initialize to `0.0`.

---

## 17. Forward Pass Pseudocode

```python
def forward(batch):
    D1 = batch["D1"]          # [B, 1, H, W]
    C1 = batch["C1"]          # [B, 1, H, W]
    D3 = batch["D3"]          # [B, V3, 1, H, W]
    C3 = batch["C3"]          # [B, V3, 1, H, W]

    K1 = batch["K1"]          # [B, 3, 3]
    K2 = batch["K2"]          # [B, 3, 3]
    K3 = batch["K3"]          # [B, V3, 3, 3]

    T1_c2w = batch["T1_c2w"]  # [B, 4, 4]
    T2_c2w = batch["T2_c2w"]  # [B, 4, 4]
    T3_c2w = batch["T3_c2w"]  # [B, V3, 4, 4]

    tau = batch["tau"]        # [B, 1]

    # 1. Warp t1 depth into target t2 view
    D1_to_t2, C1_to_t2, M1_to_t2 = warp_depth_to_target(
        depth=D1,
        confidence=C1,
        K_src=K1,
        T_src_c2w=T1_c2w,
        K_tgt=K2,
        T_tgt_c2w=T2_c2w,
    )

    # 2. Build target ray map and xy grid
    ray2 = build_ray_map(K2, H, W)     # [B, 3, H, W]
    xy = build_xy_grid(B, H, W)        # [B, 2, H, W]
    tau_map = tau[:, :, None, None].expand(B, 1, H, W)

    # 3. Build encoder input
    log_D1_to_t2 = torch.where(
        M1_to_t2 > 0,
        torch.log(D1_to_t2.clamp_min(eps)),
        torch.zeros_like(D1_to_t2),
    )

    x_in = torch.cat(
        [
            log_D1_to_t2,
            C1_to_t2,
            M1_to_t2,
            xy,
            ray2,
            tau_map,
        ],
        dim=1,
    )

    # 4. Encode t3 depth context
    Q3 = sample_t3_depth_context(
        D3=D3,
        C3=C3,
        K3=K3,
        T3_c2w=T3_c2w,
        T2_c2w=T2_c2w,
        num_samples=4096,
    )
    z3 = pointnet(Q3)

    # 5. Pose and time conditioning
    z_pose = pose_encoder(T1_c2w, T2_c2w, T3_c2w)
    z_time = time_encoder(batch["time_feat"])

    z_cond = fusion_mlp(torch.cat([z3, z_pose, z_time], dim=1))

    # 6. U-Net
    f = unet(x_in, z_cond)

    # 7. Output heads
    delta_logD = delta_logD_head(f)
    G = sigmoid(gate_head(f))

    # 8. Residual log-depth prediction
    base_logD = torch.where(
        M1_to_t2 > 0,
        torch.log(D1_to_t2.clamp_min(eps)),
        learned_log_depth_prior.expand_as(D1_to_t2),
    )

    pred_logD = base_logD + tau[:, :, None, None] * G * delta_logD
    D2_hat = torch.exp(pred_logD).clamp_min(eps)

    return {
        "D2_hat": D2_hat,
        "pred_logD": pred_logD,
        "delta_logD": delta_logD,
        "gate": G,
        "D1_to_t2": D1_to_t2,
        "C1_to_t2": C1_to_t2,
        "M1_to_t2": M1_to_t2,
    }
```

---

## 18. Loss Function

Use depth-native supervision.

The total loss is:

```text
L =
  λ_depth   * L_log_depth
+ λ_grad    * L_depth_gradient
+ λ_chamfer * L_depth_chamfer
+ λ_res     * L_residual
+ λ_gate    * L_gate
```

Use these weights:

```yaml
loss:
  log_depth_weight: 1.0
  depth_gradient_weight: 0.2
  depth_chamfer_weight: 0.01
  residual_weight: 0.01
  gate_weight: 0.001
  conf_threshold: 0.02
  depth_smooth_l1_beta: 0.05
```

---

## 19. Log-Depth Loss

Use the target depth and target confidence:

```python
valid = (
    torch.isfinite(D2)
    & torch.isfinite(D2_hat)
    & (D2 > eps)
    & (C2 > conf_threshold)
)
```

Compute:

```python
target_logD = torch.log(D2.clamp_min(eps))
pred_logD = torch.log(D2_hat.clamp_min(eps))

L_log_depth = smooth_l1(
    pred_logD[valid],
    target_logD[valid],
    beta=0.05,
)
```

This is the main training signal.

---

## 20. Depth Gradient Loss

The gradient loss helps preserve canopy edges and elevated regions.

Compute gradients on log-depth:

```python
def gradient_x(x):
    return x[..., :, 1:] - x[..., :, :-1]

def gradient_y(x):
    return x[..., 1:, :] - x[..., :-1, :]
```

Use masks for neighboring valid pixels.

```python
valid_x = valid[..., :, 1:] & valid[..., :, :-1]
valid_y = valid[..., 1:, :] & valid[..., :-1, :]

pred_gx = gradient_x(pred_logD)
target_gx = gradient_x(target_logD)

pred_gy = gradient_y(pred_logD)
target_gy = gradient_y(target_logD)

L_grad_x = smooth_l1(pred_gx[valid_x], target_gx[valid_x], beta=0.05)
L_grad_y = smooth_l1(pred_gy[valid_y], target_gy[valid_y], beta=0.05)

L_depth_gradient = L_grad_x + L_grad_y
```

---

## 21. Depth Chamfer Loss

Depth Chamfer compares unprojected predicted and target depth points in the target camera frame.

### 21.1 Unproject predicted and target depth

```python
P_pred_cam = unproject_depth(D2_hat, K2)
P_tgt_cam = unproject_depth(D2, K2)
```

Shapes:

```text
P_pred_cam: [B, H, W, 3]
P_tgt_cam:  [B, H, W, 3]
```

### 21.2 Sample valid pixels

Sample `4096` valid pixels per batch item.

Use the same target validity mask:

```text
valid = D2 > eps and C2 > conf_threshold
```

Also require predicted depth to be finite.

### 21.3 Chamfer

Compute symmetric Chamfer:

```python
dist_pred_to_tgt = nearest_neighbor_distance(P_pred_sample, P_tgt_sample)
dist_tgt_to_pred = nearest_neighbor_distance(P_tgt_sample, P_pred_sample)

L_depth_chamfer = dist_pred_to_tgt.mean() + dist_tgt_to_pred.mean()
```

Use camera-frame points for this loss.

---

## 22. Residual Regularization

Penalize unnecessary depth changes:

```python
L_residual = mean(abs(G * delta_logD))
```

This encourages the model to use small residuals unless the data requires change.

---

## 23. Gate Regularization

Penalize opening the gate everywhere:

```python
L_gate = mean(G)
```

This makes the model conservative and encourages localized updates.

---

## 24. Total Loss Pseudocode

```python
def trdm_loss(outputs, batch, cfg):
    D2_hat = outputs["D2_hat"]
    delta_logD = outputs["delta_logD"]
    G = outputs["gate"]

    D2 = batch["D2"]
    C2 = batch["C2"]

    eps = 1e-4
    conf_thr = cfg.conf_threshold

    valid = (
        torch.isfinite(D2)
        & torch.isfinite(D2_hat)
        & (D2 > eps)
        & (C2 > conf_thr)
    )

    pred_logD = torch.log(D2_hat.clamp_min(eps))
    target_logD = torch.log(D2.clamp_min(eps))

    L_log_depth = smooth_l1(
        pred_logD[valid],
        target_logD[valid],
        beta=cfg.depth_smooth_l1_beta,
    )

    L_grad = depth_gradient_loss(
        pred_logD=pred_logD,
        target_logD=target_logD,
        valid=valid,
        beta=cfg.depth_smooth_l1_beta,
    )

    L_chamfer = depth_chamfer_loss(
        pred_depth=D2_hat,
        target_depth=D2,
        K=batch["K2"],
        valid=valid,
        num_points=4096,
    )

    L_res = torch.mean(torch.abs(G * delta_logD))
    L_gate = torch.mean(G)

    total = (
        cfg.log_depth_weight * L_log_depth
        + cfg.depth_gradient_weight * L_grad
        + cfg.depth_chamfer_weight * L_chamfer
        + cfg.residual_weight * L_res
        + cfg.gate_weight * L_gate
    )

    return {
        "loss": total,
        "loss_log_depth": L_log_depth.detach(),
        "loss_depth_gradient": L_grad.detach(),
        "loss_depth_chamfer": L_chamfer.detach(),
        "loss_residual": L_res.detach(),
        "loss_gate": L_gate.detach(),
    }
```

---

## 25. Training Configuration

Use this starting config:

```yaml
model_class: trdm.model.TRDM

model_kwargs:
  input_channels: 9
  base_channels: 32
  context_dim: 128
  pose_dim: 128
  time_dim: 128
  cond_dim: 256
  t3_context_samples: 4096
  eps: 1.0e-4

optimizer:
  lr: 2.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]

scheduler:
  warmup_epochs: 5
  min_lr: 1.0e-6

training:
  epochs: 100
  batch_size: 8
  val_every: 1
  grad_clip: 1.0

early_stopping:
  patience: 15
  min_delta: 1.0e-4

loss:
  log_depth_weight: 1.0
  depth_gradient_weight: 0.2
  depth_chamfer_weight: 0.01
  residual_weight: 0.01
  gate_weight: 0.001
  conf_threshold: 0.02
  depth_smooth_l1_beta: 0.05
  chamfer_num_points: 4096
```

---

## 26. Dataset Output

The dataset should return:

```python
sample = {
    "D1": D1,
    "C1": C1,
    "D2": D2,
    "C2": C2,
    "D3": D3,
    "C3": C3,

    "K1": K1,
    "K2": K2,
    "K3": K3,

    "T1_c2w": T1_c2w,
    "T2_c2w": T2_c2w,
    "T3_c2w": T3_c2w,

    "tau": tau,
    "time_feat": time_feat,

    "metadata": metadata,
}
```

All depth maps should be resized or generated at the same resolution:

```text
[H, W]
```

The intrinsics must match this resolution.

---

## 27. Evaluation and Export

At evaluation time, generate:

```text
D2_hat
```

Then convert it to a world-space point cloud.

### 27.1 Unproject depth to target camera frame

```python
P2_cam_hat = unproject_depth(D2_hat, K2)
```

### 27.2 Transform to world

```python
P2_world_hat = T2_c2w @ P2_cam_hat
```

### 27.3 Filter

Use:

```text
finite depth
positive depth
predicted depth confidence if implemented later
target ROI mask if available
```

Since the first implementation does not predict confidence, use a geometric validity mask:

```text
D2_hat > eps
D2_hat finite
inside ROI
```

### 27.4 Save geometry asset

Save:

```text
point_cloud_clean.npz
```

with arrays:

```text
points_aligned
points_normalized
confidence
view_index
```

For confidence, use:

```text
ones_like(valid_points)
```

in the first implementation.

---

## 28. Implementation Files

Create:

```text
src/trdm/model.py
src/trdm/geometry.py
src/trdm/loss.py
src/trdm/dataset.py
src/trdm/train.py
src/trdm/evaluate.py
src/trdm/build_predicted_geometry.py
```

### `src/trdm/geometry.py`

Contains:

```text
unproject_depth
project_points
warp_depth_to_target
build_ray_map
build_xy_grid
sample_t3_depth_context
transform_points
```

### `src/trdm/model.py`

Contains:

```text
TRDM
ConvBlock
DownBlock
UpBlock
FiLM
PointNetContextEncoder
PoseEncoder
TimeEncoder
FusionMLP
```

### `src/trdm/loss.py`

Contains:

```text
trdm_loss
log_depth_loss
depth_gradient_loss
depth_chamfer_loss
```

### `src/trdm/build_predicted_geometry.py`

Contains:

```text
D2_hat + K2 + T2_c2w → world-space predicted cloud
```

---

## 29. Debug Checks

Before training seriously, run one batch and verify:

```text
D1_to_t2 shape is [B, 1, H, W]
C1_to_t2 shape is [B, 1, H, W]
M1_to_t2 shape is [B, 1, H, W]
x_in shape is [B, 9, H, W]
Q3 shape is [B, 4096, 13]
z3 shape is [B, 128]
z_pose shape is [B, 128]
z_time shape is [B, 128]
z_cond shape is [B, 256]
D2_hat shape is [B, 1, H, W]
G shape is [B, 1, H, W]
delta_logD shape is [B, 1, H, W]
```

Also log:

```text
mean D1_to_t2
mean D2
mean D2_hat
valid ratio of M1_to_t2
target valid ratio
mean gate
mean abs delta_logD
log-depth loss
gradient loss
```

Expected initial behavior:

```text
mean gate ≈ 0.12
delta_logD ≈ 0
D2_hat close to warped D1 where M1_to_t2 is valid
```

---

## 30. Important Failure Checks

### 30.1 If `D1_to_t2` is mostly empty

The t1-to-t2 warp is failing.

Check:

```text
camera poses
intrinsics scale
depth convention
image resolution
projection bounds
z-positive convention
```

### 30.2 If `D2_hat` becomes NaN

Check:

```text
depth values before log
clamp_min(eps)
invalid masks
learning rate
gradient clipping
```

### 30.3 If model predicts a smooth plane

Increase:

```yaml
depth_gradient_weight: 0.3
```

and check whether `D1_to_t2` contains canopy structure.

### 30.4 If gate opens everywhere

Increase:

```yaml
gate_weight: 0.005
```

### 30.5 If model does not change from `D1_to_t2`

Decrease:

```yaml
gate_weight: 0.0005
residual_weight: 0.005
```

and check that target valid masks are not too sparse.

---

## 31. Final Summary

TRDM should be implemented as a depth-native temporal model.

The model should:

```text
1. Warp t1 depth into the target t2 view.
2. Encode warped t1 depth, target rays, confidence, mask, xy, and time.
3. Encode all t3 depth views as sampled 3D context in the t2 camera frame.
4. Fuse t3 context, relative pose, and time.
5. Predict a gated log-depth residual.
6. Produce D2_hat.
7. Train using log-depth, depth-gradient, and depth-Chamfer losses.
8. Export predicted geometry by unprojecting D2_hat with the target camera.
```

The main advantage is that the network predicts only target-view depth, while the calibrated camera model reconstructs the final 3D geometry.
