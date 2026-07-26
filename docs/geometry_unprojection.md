# Geometry Unprojection Path

This document explains how VGGT prediction files become evaluation point clouds
in this repository, especially the path that uses `depth_map.npy`.

There are two related evaluation/data paths:

1. `src/evaluate.py` with `geometry_assets/`
2. `src/trpm/evaluate.py` direct TRPM evaluation

Only the first path builds its geometry from `depth_map.npy`.

---

## 1. Input Files

VGGT inference writes per-date prediction folders like:

```text
vggt_outputs/.../{triplet_id}/{variant}/{t1,t2,t3}/
├── dataset_cameras.json
├── selected_images.json
└── predictions/
    ├── depth_map.npy
    ├── depth_confidence.npy
    ├── point_map.npy
    ├── point_confidence.npy
    ├── extrinsic.npy
    └── intrinsic.npy
```

The depth-unprojection path requires:

```text
predictions/depth_map.npy
predictions/depth_confidence.npy
predictions/extrinsic.npy
dataset_cameras.json
```

`point_map.npy` is only used as a fallback when `dataset_cameras.json` is
missing.

---

## 2. Main Depth-Based Geometry Builder

The depth-based path is implemented in:

```text
src/vggt_pipeline/build_geometry_assets.py
```

Its file header states the intended behavior: for each triplet/date/variant, it
loads `depth_map.npy` and `depth_confidence.npy`, unprojects them into dataset
world space, and writes `geometry_assets/.../point_cloud_clean.npz`.

The core function is:

```python
extract_depth_cloud(date_dir, dataset_cameras, ...)
```

It loads:

```python
depth_map = np.load(pred_dir / "depth_map.npy")
depth_conf = np.load(pred_dir / "depth_confidence.npy")
extrinsics = np.load(pred_dir / "extrinsic.npy")
```

Then it uses intrinsics from `dataset_cameras.json`:

```python
intrinsics = dataset_cameras["intrinsics"]
frames = dataset_cameras["frames"]
```

---

## 3. Intrinsic Scaling

VGGT predictions are at VGGT output resolution, not necessarily the original
image resolution. The code rescales the dataset camera intrinsics to the depth
map resolution.

For `pad` preprocessing:

```python
scale = min(vggt_w / w_orig, vggt_h / h_orig)
pad_left = (vggt_w - w_orig * scale) / 2
pad_top = (vggt_h - h_orig * scale) / 2

fx_s = fx * scale
fy_s = fy * scale
cx_s = cx * scale + pad_left
cy_s = cy * scale + pad_top
```

This matches the image padding used before VGGT inference.

For non-pad resizing:

```python
fx_s = fx * (W / w_orig)
fy_s = fy * (H / h_orig)
cx_s = cx * (W / w_orig)
cy_s = cy * (H / h_orig)
```

---

## 4. Depth Pixel to Camera-Space Point

For every retained pixel `(u, v)` and depth value `d`, the code uses the
standard pinhole unprojection:

```python
x_c = (u - cx_s) / fx_s * d
y_c = (v - cy_s) / fy_s * d
z_c = d
```

So each depth pixel becomes:

```text
[x_c, y_c, z_c, 1]
```

This is the geometry-unprojection step.

Important: this path trusts the VGGT `depth_map.npy` values and converts them
into 3D points using the scaled dataset intrinsics.

---

## 5. Camera Space to VGGT World Space

VGGT writes `extrinsic.npy` as world-to-camera matrices with shape:

```text
[S, 3, 4]
```

The builder inverts each extrinsic into camera-to-world:

```python
R = ext_3x4[:, :3]
t = ext_3x4[:, 3]

c2w[:3, :3] = R.T
c2w[:3, 3] = -R.T @ t
```

Then each camera-space point is transformed to VGGT world space:

```python
pts_vggt = (c2w @ pts_cam_h.T).T[:, :3]
```

At this stage the point cloud is in VGGT's predicted coordinate frame.

---

## 6. VGGT World to Dataset/GPS World

VGGT world coordinates are not automatically in the dataset/GPS world frame.
The code aligns them using camera centers.

VGGT camera centers are computed from `extrinsic.npy`:

```python
vggt_center_i = -R_i.T @ t_i
```

Dataset/GPS camera centers come from:

```python
dataset_cameras.json
frames[i]["transform_matrix"][:3, 3]
```

The code computes an Umeyama similarity transform:

```text
dataset_point ≈ scale * R_align @ vggt_point + t_align
```

Then applies it to every unprojected point:

```python
pts_ds = (scale_a * R_align @ pts_vggt.T + t_align[:, None]).T
```

Now the point cloud is in the NeRFStudio/dataset/GPS world coordinate system.

---

## 7. Confidence Filtering and Cleanup

The builder normalizes `depth_confidence.npy` and filters points:

```python
valid = isfinite(point) and isfinite(confidence) and confidence >= threshold
```

Then it applies cleanup:

```text
outlier filtering
voxel downsampling
max-point subsampling
optional crop-level normalization
optional ROI mask
```

The final saved file is:

```text
geometry_assets/{triplet_id}/{variant}/{t1,t2,t3}/point_cloud_clean.npz
```

Typical arrays inside:

```text
points_normalized   # normalized evaluation coordinates, if normalization enabled
points_aligned      # dataset/GPS-aligned world coordinates
points              # raw selected points, depending on save path/version
confidence
view_index          # source view id for each point
```

`src/evaluate.py` later loads this `.npz` file and evaluates baselines/model
against the `t2` geometry.

---

## 8. How Baselines Use These Geometries

When using `src/evaluate.py`, baselines do not directly read `depth_map.npy`.
Instead they read the already-built geometry assets:

```text
geometry_assets/.../t1/point_cloud_clean.npz
geometry_assets/.../t2/point_cloud_clean.npz
geometry_assets/.../t3/point_cloud_clean.npz
```

But if those assets were built by `build_geometry_assets.py` with
`dataset_cameras.json` available, then the clouds were originally generated from
`depth_map.npy`.

The baseline logic is:

```text
B0: use t1 cloud as prediction
B1: use t3 cloud as prediction
B2: use temporally nearest endpoint cloud
B3: random equal-size point interpolation, (1 - tau) * t1 + tau * t3
B4: union/subsample t1 and t3 with temporal weights
```

All baseline metrics compare against:

```text
geometry_assets/.../t2/point_cloud_clean.npz
```

So in this path the reference `t2` geometry is also depth-unprojected.

---

## 9. Direct TRPM Evaluation Path

`src/trpm/evaluate.py` is different.

It loads:

```text
point_map.npy
point_confidence.npy
extrinsic.npy
dataset_cameras.json
```

It converts VGGT `point_map.npy` points to GPS/world space using the same kind
of Umeyama camera-center alignment, but it does not load `depth_map.npy`.

So:

```text
src/trpm/evaluate.py              uses point_map.npy
src/vggt_pipeline/build_geometry_assets.py uses depth_map.npy
```

This is why it can look contradictory: TRPM has a direct point-map evaluator,
but the geometry-assets pipeline used by `src/evaluate.py` is depth-based.

---

## 10. TRPM Depth-Model Predicted Geometry

There is another depth-unprojection path for predicted TRPM depth models:

```text
src/trpm/build_predicted_geometry.py
```

For `TRPMSmallCamDepth`, the model predicts:

```text
D2_hat
```

The script unprojects `D2_hat`:

```python
x_cam = (u - cu) / fu * depth
y_cam = (v - cv) / fv * depth
z_cam = depth
pts_world = T_c2w @ [x_cam, y_cam, z_cam, 1]
```

Then it applies the GPS/world alignment before saving predicted geometry assets.

This path is for predicted model geometry, not for the ordinary VGGT baseline
clouds.

---

## 11. Summary

Use this rule of thumb:

```text
If evaluation loads geometry_assets/.../point_cloud_clean.npz:
    the geometry was usually built from depth_map.npy by build_geometry_assets.py.

If evaluation reads vggt_outputs/.../predictions/point_map.npy directly:
    it uses point_map.npy, not depth_map.npy.
```

The depth-based geometry pipeline is:

```text
depth_map.npy
  + depth_confidence.npy
  + scaled dataset intrinsics
  + VGGT extrinsic.npy
  + dataset_cameras.json
      ↓
camera-space points
      ↓
VGGT world points
      ↓
dataset/GPS-aligned points
      ↓
point_cloud_clean.npz
      ↓
src/evaluate.py baselines/model metrics
```
