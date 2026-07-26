# TRDM Training Guide

This document explains how to train the TRDM model implemented in `src/trpm/`.

TRDM predicts a target-view depth map `D2_hat` for the intermediate date `t2`.
During evaluation, `D2_hat` is unprojected with the target camera intrinsics and
pose to produce 3D geometry.

## Main Files

- Model: `src/trpm/trdm_model.py`
- Dataset: `src/trpm/trdm_dataset.py`
- Loss: `src/trpm/trdm_loss.py`
- Geometry utilities: `src/trpm/trdm_geometry.py`
- Training script: `src/trpm/trdm_train.py`
- Evaluation script: `src/trpm/trdm_evaluate.py`
- Training config: `configs/train_trdm.yaml`
- Evaluation config: `configs/eval_trdm.yaml`

## Required Inputs

TRDM expects VGGT outputs for each triplet in this layout:

```text
vggt_outputs/t1t2_paired_v16_o8/
  {t1_date}_{t2_date}_{t3_date}_{crop}/
    variant_XX/
      t1/
        predictions/
          depth_map.npy
          depth_confidence.npy
          extrinsic.npy
          intrinsic.npy
        dataset_cameras.json
      t2/
        predictions/
          depth_map.npy
          depth_confidence.npy
          extrinsic.npy
          intrinsic.npy
        dataset_cameras.json
      t3/
        predictions/
          depth_map.npy
          depth_confidence.npy
          extrinsic.npy
          intrinsic.npy
        dataset_cameras.json
```

`intrinsic.npy` is preferred when present. If it is missing, the dataset falls
back to `dataset_cameras.json` intrinsics scaled to the VGGT depth-map
resolution.

## Training Config

Use:

```text
configs/train_trdm.yaml
```

Important fields:

```yaml
vggt_output_root: vggt_outputs/t1t2_paired_v16_o8
triplets_path: prepared_data/subsets/benchmark_triplets.json
output_root: runs/trdm_t1t2_paired_v16_o8

protocols:
  - strict

crops:
  - corn

test_date: "20230831"
device: cuda:0

epochs: 100
batch_size: 8
num_workers: 4
```

When using distributed training, `batch_size` is per GPU. For example, with
8 GPUs and `batch_size: 8`, the effective global batch size is `64`.

## Single-GPU Training

From the repository root:

```bash
python src/trpm/trdm_train.py --config configs/train_trdm.yaml
```

If you want to force a specific GPU:

```bash
python src/trpm/trdm_train.py \
  --config configs/train_trdm.yaml \
  --device cuda:0
```

## Multi-GPU Training

Use `torchrun`:

```bash
torchrun --nproc_per_node=8 src/trpm/trdm_train.py \
  --config configs/train_trdm.yaml
```

The trainer automatically detects distributed mode from `WORLD_SIZE`,
`LOCAL_RANK`, and `RANK`.

In distributed mode:

- The model is wrapped with `DistributedDataParallel`.
- Training data uses `DistributedSampler`.
- Validation samples are split across ranks.
- Loss metrics are reduced across all GPUs.
- Only rank 0 writes checkpoints and logs.

## Outputs

For each fold, outputs are saved under:

```text
runs/trdm_t1t2_paired_v16_o8/{protocol}/{fold_id}/
```

Example:

```text
runs/trdm_t1t2_paired_v16_o8/strict/test_20230831_corn/
```

Main files:

```text
best_model.pt
last_checkpoint.pt
training_history.json
```

`best_model.pt` contains only the model state dict and is used by evaluation.

`last_checkpoint.pt` contains:

```text
epoch
model_state_dict
optimizer_state_dict
best_val_loss
best_epoch
history
```

## Evaluation After Training

Run:

```bash
python src/trpm/trdm_evaluate.py --config configs/eval_trdm.yaml
```

For 8 GPUs:

```bash
torchrun --nproc_per_node=8 src/trpm/trdm_evaluate.py \
  --config configs/eval_trdm.yaml
```

Evaluation compares:

- `trdm`: unprojected model depth prediction `D2_hat`
- `B0_t1_date_copy`
- `B1_t3_date_copy`
- `B2_nearest_date_copy`
- `B3_linear_depth_cloud_interpolation`
- `B4_temporal_weighted_depth_cloud_union`

The target geometry is always t2 depth geometry unprojected from
`depth_map.npy`.

Evaluation reports:

- `merged`: merge all views, then compute metrics
- `per_view_avg`: compute metrics per view, then average

## Useful Debug Checks

Before a long run, check that the dataset finds samples:

```bash
PYTHONPATH=src python - <<'PY'
from trpm.trdm_dataset import TRDMDepthDataset
d = TRDMDepthDataset("vggt_outputs/t1t2_paired_v16_o8")
print("samples:", len(d))
print(d.index[0] if len(d) else "empty")
PY
```

Run a short training test by temporarily setting:

```yaml
epochs: 1
batch_size: 1
num_workers: 0
```

Then launch:

```bash
python src/trpm/trdm_train.py --config configs/train_trdm.yaml
```

Expected initial behavior:

- `mean_gate` should start around `0.12`.
- `mean_abs_delta_logD` should start near zero.
- `D2_hat` should initially be close to warped t1 depth where the warp is valid.
- `warp_valid_ratio` should not be near zero; if it is, check camera convention,
  intrinsics, depth scale, and preprocessing mode.

## Common Issues

### Dataset Has Zero Samples

Check that each `t1`, `t2`, and `t3` folder contains:

```text
predictions/depth_map.npy
predictions/depth_confidence.npy
predictions/extrinsic.npy
dataset_cameras.json
```

Also check that `vggt_output_root`, `triplets_path`, `crops`, and `test_date`
match the data you generated.

### CUDA Out Of Memory

Reduce one or more of:

```yaml
batch_size: 4
model_kwargs:
  t3_context_samples: 2048
  base_channels: 24
```

### Validation Gets Worse Quickly

Try a smaller learning rate:

```yaml
optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4
```

or:

```yaml
optimizer:
  lr: 5.0e-5
  weight_decay: 1.0e-4
```

### Warp Is Mostly Empty

This usually means one of these is wrong:

- Camera pose convention
- Depth forward-axis convention
- Intrinsics scaling
- Image/depth resolution mismatch
- VGGT output preprocessing mode

The current implementation assumes standard pinhole depth:

```text
x = (u - cx) / fx * depth
y = (v - cy) / fy * depth
z = depth
```

and camera-to-world matrices `T_c2w`.

