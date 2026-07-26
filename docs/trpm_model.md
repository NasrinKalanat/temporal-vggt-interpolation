# TRPM Model Architecture

TRPM means **Temporal Residual Point-Map Network**. It predicts the intermediate
date `t2` geometry from endpoint dates `t1` and `t3`.

The central idea is residual temporal interpolation:

```text
P2_hat = P1 + tau * G * ΔP
```

where:

```text
P1      = t1 point map
P2_hat  = predicted t2 point map
tau     = temporal position of t2 between t1 and t3
G       = learned per-pixel gate in [0, 1]
ΔP      = learned per-pixel geometry update
```

So TRPM does not generate geometry from scratch. It starts from `t1` geometry
and learns a gated update toward `t2`, conditioned on `t3` and time.

---

## 1. Data Representation

TRPM uses VGGT output point maps as dense 2D images of 3D points:

```text
point_map.npy        [S, H, W, 3]
point_confidence.npy [S, H, W] or [S, H, W, 1]
```

For each variant:

```text
t1: endpoint source date
t2: target/intermediate date
t3: endpoint source date
```

The TRPM dataset loads all views per variant:

```text
P1: [V,  3, H, W]
C1: [V,  1, H, W]
P2: [V,  3, H, W]
C2: [V,  1, H, W]
P3: [V3, 3, H, W]
C3: [V3, 1, H, W]
tau: [1]
```

The training loop usually flattens the target views into the batch dimension:

```text
[B, V, ...] → [B * V, ...]
```

This means each `t1/t2` view pair is trained as one target sample, while all
available `t3` views are used as context.

---

## 2. Base Model: `TRPMSmall`

Implemented in:

```text
src/trpm/model.py
```

Inputs:

```text
P1:  [B, 3, H, W]   t1 point map
C1:  [B, 1, H, W]   t1 confidence
P3:  [B, 3, H, W]   t3 point map
C3:  [B, 1, H, W]   t3 confidence
tau: [B, 1]
```

The base model assumes one `t3` map aligned with the target sample. It builds an
8-channel encoder input:

```text
P1          3 channels
C1          1 channel
M1          1 channel, confidence mask
xy          2 channels, normalized pixel grid
tau_map     1 channel
------------------------------
total       8 channels
```

Forward path:

```text
t1 point-map image
    ↓
U-Net encoder
    ↓
FiLM-conditioned decoder
    ↓
residual head ΔP and gate head G
    ↓
P2_hat = P1 + tau * G * ΔP
```

The `t3` point map is not concatenated spatially. Instead, valid/confident `t3`
points are sampled and encoded with a small PointNet.

Each sampled `t3` point has:

```text
xyz         3
confidence 1
uv          2
----------------
total       6
```

PointNet max-pools sampled points into a global context vector:

```text
Q3: [B, K, 6] → z3: [B, 256]
```

`tau` is encoded with Fourier features:

```text
tau → ztau: [B, 256]
```

Then:

```text
concat(z3, ztau) → fusion MLP → z_cond
```

`z_cond` is injected into decoder features through FiLM:

```text
feature = feature * (1 + gamma(z_cond)) + beta(z_cond)
```

---

## 3. Camera-Aware Model: `TRPMSmallCam`

Implemented in:

```text
src/trpm/trpm_small_cam.py
```

This is the main geometry-aware version. It uses camera poses and intrinsics so
geometry is predicted in the **target t2 camera frame**.

Additional inputs:

```text
P3_world: [B, V3, 3, H, W]  all t3 views
C3:       [B, V3, 1, H, W]
T1_c2w:   [B, 4, 4]
T2_c2w:   [B, 4, 4]
T3_c2w:   [B, V3, 4, 4]
K2:       [B, 3, 3]
K3:       [B, V3, 3, 3]
```

### Target Frame Conversion

The model first converts `P1_world` into the target `t2` camera frame:

```text
P1_cam = R2ᵀ * (P1_world - c2)
```

This makes the residual update local to the target view, instead of predicting
directly in global/GPS coordinates.

### Target Ray Map

It computes a ray direction for every target pixel using `K2`:

```text
x = (u - cx) / fx
y = (v - cy) / fy
ray = normalize([x, y, 1])
```

The encoder input becomes 11 channels:

```text
P1_cam     3
C1         1
M1         1
xy         2
ray2       3
tau_map    1
----------------
total      11
```

### Multi-View t3 Context

Unlike the base model, `TRPMSmallCam` can use all `t3` views.

It converts every sampled `t3` point into the target `t2` camera frame and adds
view-geometry features.

Each sampled `t3` point has 12 features:

```text
xyz in target frame             3
confidence                      1
source uv                       2
source ray in target frame      3
source camera center in target  3
----------------------------------
total                          12
```

These are encoded by PointNet:

```text
Q3: [B, K_total, 12] → z3: [B, 128]
```

### Relative Pose Encoding

The model also encodes relative camera poses:

```text
t1 camera relative to t2
mean/std of t3 camera centers relative to t2
mean t3 relative rotations
```

This produces:

```text
z_pose: [B, 128]
```

Time encoding produces:

```text
ztau: [B, 128]
```

The conditioning vector is:

```text
concat(z3, z_pose, ztau) → fusion MLP → z_cond
```

Then the same U-Net decoder predicts:

```text
delta_P:    [B, 3, H, W]
G:          [B, 1, H, W]
P2_cam_hat: [B, 3, H, W]
```

with:

```text
P2_cam_hat = P1_cam + tau * G * ΔP
```

During training, the target `P2_world` is also converted into `t2` camera frame
before applying the point-map loss.

---

## 4. Depth Variant: `TRPMSmallCamDepth`

Implemented in:

```text
src/trpm/trpm_small_cam_depth.py
```

This extends `TRPMSmallCam` with one extra head:

```text
D2_hat: [B, 1, H, W]
```

The depth head is:

```text
Conv → SiLU → Conv → Softplus
```

`Softplus` keeps predicted depth positive.

The geometry output is still:

```text
P2_cam_hat = P1_cam + tau * G * ΔP
```

The depth output is supervised from:

```text
depth_map.npy
```

when the dataset is loaded with `load_depth=True`.

Training supports three depth modes in `train_ddp.py`:

```text
depth_only   supervise D2_hat; point target is detached
point_only   ignore D2_hat loss
point_depth  supervise both point map and depth
```

The depth loss is masked by valid target confidence and valid positive depth.
If Chamfer is enabled, the loss also unprojects sampled predicted/target depths
with `K2` and compares the resulting camera-frame point clouds.

---

## 5. Color Variant: `TRPMSmallCamColor`

Implemented in:

```text
src/trpm/trpm_small_cam_color.py
```

This extends the camera-aware model with RGB inputs and an RGB residual head.

Additional inputs:

```text
I1: [B, 3, H, W]
I2: [B, 3, H, W]       target supervision only
I3: [B, V3, 3, H, W]
```

The encoder input becomes 14 channels:

```text
P1_cam      3
C1          1
M1          1
I1          3
xy          2
ray2        3
tau_map     1
-----------------
total       14
```

The sampled `t3` context feature becomes 15-dimensional:

```text
xyz in target frame             3
confidence                      1
RGB3                            3
source uv                       2
source ray in target frame      3
source camera center in target  3
----------------------------------
total                          15
```

It predicts:

```text
P2_cam_hat
G
delta_P
RGB2_hat
delta_RGB
```

Color prediction is another residual:

```text
RGB2_hat = clamp(I1 + tau * delta_RGB, 0, 1)
```

---

## 6. U-Net Backbone

All variants use a small U-Net-like convolutional backbone:

```text
encoder:
  e0: ConvBlock → 32 channels
  e1: DownBlock → 64 channels
  e2: DownBlock → 128 channels
  e3: DownBlock → 192 channels

decoder:
  d2: UpBlock with e2 skip → 128 channels
  d1: UpBlock with e1 skip → 64 channels
  d0: UpBlock with e0 skip → 32 channels
```

Each `ConvBlock` is:

```text
Conv2d → GroupNorm → SiLU → Conv2d → GroupNorm → SiLU + residual skip
```

Each decoder `UpBlock`:

```text
upsample → concat skip → ConvBlock → FiLM(z_cond)
```

---

## 7. Losses

Implemented in:

```text
src/trpm/loss.py
```

Main loss:

```text
L = L_point
  + lambda_chamfer * L_chamfer
  + lambda_res * L_res
  + lambda_gate * L_gate
  + optional RGB/depth losses
```

### Point Loss

Masked SmoothL1:

```text
L_point = SmoothL1(P2_hat, P2) over pixels where C2 > conf_threshold
```

For camera-aware models, both `P2_hat` and `P2` are in target `t2` camera frame.

### Chamfer Loss

If enabled, randomly samples valid target pixels and computes symmetric Chamfer
between predicted and target point sets.

### Residual Regularization

```text
L_res = mean(abs(G * ΔP))
```

This discourages unnecessarily large residual updates.

### Gate Regularization

```text
L_gate = mean(G)
```

This discourages opening the gate everywhere.

### RGB Loss

Only for `TRPMSmallCamColor`:

```text
L_rgb = masked L1(RGB2_hat, I2)
```

### Depth Loss

Only for `TRPMSmallCamDepth`:

```text
L_depth = masked SmoothL1(D2_hat, D2)
```

Optionally, depth Chamfer unprojects `D2_hat` and `D2` using `K2`:

```text
x = (u - cx) / fx * d
y = (v - cy) / fy * d
z = d
```

and compares the resulting camera-frame point clouds.

---

## 8. Training Flow

Training entrypoints:

```text
src/trpm/train.py
src/trpm/train_ddp.py
```

The model class is selected from config:

```yaml
model_class: trpm.trpm_small_cam.TRPMSmallCam
```

Examples:

```text
configs/train_trpm_small.yaml
configs/train_trpm_small_cam.yaml
configs/train_trpm_small_cam_depth.yaml
configs/train_trpm_small_cam_point_depth.yaml
configs/train_trpm_small_cam_color.yaml
```

Dataset:

```text
src/trpm/dataset.py
```

The dataset:

1. Loads VGGT `point_map.npy` and `point_confidence.npy`.
2. Aligns VGGT world points to GPS/dataset world with Umeyama alignment.
3. Optionally loads RGB images for color models.
4. Optionally loads `depth_map.npy` for depth models.
5. Loads camera poses/intrinsics for camera-aware models.

Training dispatch:

```text
TRPMSmall            → _step_base
TRPMSmallCam         → _step_cam_point / _step_cam
TRPMSmallCamDepth    → _step_cam_depth
TRPMSmallCamColor    → _step_cam_color
```

For `TRPMSmallCam` and variants, `t3` views are broadcast to each target `t1/t2`
view:

```text
P3: [B, V3, 3, H, W]
→ repeated for each V target view
→ [B * V, V3, 3, H, W]
```

---

## 9. Evaluation Flow

Direct TRPM evaluation:

```text
src/trpm/evaluate.py
src/trpm/evaluate_ddp.py
```

It evaluates:

```text
B0_t1_date_copy
B1_t3_date_copy
B2_nearest_date_copy
B3_linear_point_map_interpolation
B4_temporal_weighted_point_map_union
trpm model prediction
```

Direct TRPM evaluation loads VGGT point maps directly from `vggt_output_root`.
It uses:

```text
point_map.npy
point_confidence.npy
extrinsic.npy
dataset_cameras.json
```

It aligns VGGT point maps to GPS/dataset world with Umeyama alignment.

For `TRPMSmallCam`, model outputs `P2_cam_hat`; evaluation transforms it back to
world space using `T2_c2w`:

```text
P2_world_hat = R2 * P2_cam_hat + c2
```

For `TRPMSmallCamDepth`, direct `src/trpm/evaluate.py` still evaluates
`P2_cam_hat`; the separate script `src/trpm/build_predicted_geometry.py` is the
path that unprojects predicted `D2_hat` into geometry assets.

---

## 10. Geometry-Assets Bridge

`src/trpm/build_predicted_geometry.py` converts TRPM predictions into the
`geometry_assets/` layout so they can be evaluated by the generic
`src/evaluate.py` pipeline.

For depth models:

```text
D2_hat + intrinsics + pose → unprojected predicted cloud
```

For camera-aware point models:

```text
P2_cam_hat + T2_c2w → predicted world cloud
```

For base models:

```text
P2_hat is already treated as world-space point map
```

This is the bridge between TRPM models and the depth-based geometry-assets
evaluation described in `docs/geometry_unprojection.md`.

---

## 11. Practical Notes

- `TRPMSmall` is simplest but assumes stronger view/point-map alignment.
- `TRPMSmallCam` is usually the more correct geometry model because it predicts
  in the target camera frame.
- `TRPMSmallCamDepth` adds depth supervision and can export predictions by
  unprojecting `D2_hat`.
- `TRPMSmallCamColor` adds RGB reconstruction as auxiliary supervision.
- The gate `G` controls how much residual update is applied per pixel.
- The residual formulation makes the model conservative: early behavior is close
  to copying `t1`, then learning temporal changes where useful.

---

## 12. Shape Summary

Base model:

```text
P1, P2, P3: [B, 3, H, W]
C1, C2, C3: [B, 1, H, W]
tau:        [B, 1]
P2_hat:     [B, 3, H, W]
G:          [B, 1, H, W]
```

Camera-aware model:

```text
P1_world:   [B, 3, H, W]
P3_world:   [B, V3, 3, H, W]
T1_c2w:     [B, 4, 4]
T2_c2w:     [B, 4, 4]
T3_c2w:     [B, V3, 4, 4]
K2:         [B, 3, 3]
K3:         [B, V3, 3, 3]
P2_cam_hat: [B, 3, H, W]
```

Dataset before view flattening:

```text
P1: [B, V,  3, H, W]
P2: [B, V,  3, H, W]
P3: [B, V3, 3, H, W]
```
