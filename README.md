# Temporal-VGGT

Predict intermediate 3D crop scene geometry at date `t2` given multi-view UAV images from two endpoint dates `t1` and `t3`. Evaluated with leave-one-date-out (LOTO) cross-validation on corn and soybean UAV sequences.

## Setup

The project uses the `4d` conda environment.

```bash
conda activate 4d
```

Install dependencies if not already present:

```bash
pip install -r requirements.txt
```

All scripts should be run from the project root. The `src/` directory is added to the Python path automatically by each script.

## Data

Raw data is at `/data/nak168/learning_3d/canopy_data/`. Prepared manifests and temporal triplets are already in `prepared_data/` and do not need to be regenerated unless the dataset changes.

Key prepared files:

| File | Description |
|---|---|
| `prepared_data/manifests/dataset_manifest.json` | All scenes with paths, frame counts, and cleaned transform paths |
| `prepared_data/subsets/benchmark_subset.json` | Scenes selected for experiments |
| `prepared_data/subsets/benchmark_triplets.json` | Temporal date-triplets `(t1, t2, t3)` — 140 combinations, no view_batch |
| `prepared_data/subsets/available_triplets.json` | Expanded triplets built from completed vggt_output variants — one entry per `(date-triplet, view_batch)` |

The benchmark uses only the **matic** platform. Corn starts at `20230812`; soybean starts at `20230817`. Both end at `20230922`.

## Workflow

### Step 1 — Run VGGT inference

Runs VGGT on each scene with multiple view batches and saves raw point maps.

```bash
conda run -n 4d python src/vggt_pipeline/run_vggt_inference.py --config configs/vggt_inference.yaml
```

For each scene, all available frames are shuffled once (deterministic per scene), then sliced into overlapping batches of `n_views` using a sliding window with stride `n_views - max_overlap_views`. Each batch is saved under its own variant directory:

```
vggt_output/<scene_id>/views_00/predictions/
vggt_output/<scene_id>/views_01/predictions/
...
```

Each variant contains:
- `point_map.npy` — shape `(S, H, W, 3)`, world-space 3D points
- `point_confidence.npy` — shape `(S, H, W)`, per-pixel confidence
- `extrinsic.npy` — shape `(S, 3, 4)`, camera extrinsics
- `selected_images.json` — which frames were used

With `n_views=32` and `max_overlap_views=16` (stride=16), each scene with ~840 frames produces ~51 variants (corn) or ~42 variants (soybean).

Results are cached — rerunning skips already-completed variants.

Key config options (`configs/vggt_inference.yaml`):

```yaml
n_views: 32              # views per batch
max_overlap_views: 16    # overlap between consecutive batches (stride = n_views - max_overlap_views)
skip_existing: true      # skip variants already done
```

To run only specific scenes:

```bash
conda run -n 4d python src/vggt_pipeline/run_vggt_inference.py --scene-id nerfstudio_matic20230822_corn --scene-id nerfstudio_matic20230827_corn
```

### Step 2 — Build geometry assets

Fuses raw point maps into a cleaned, aligned, and normalized point cloud per scene variant.

```bash
conda run -n 4d python src/vggt_pipeline/build_geometry_assets.py \
    --vggt-root vggt_output \
    --output-root geometry_assets \
    --normalize
```

Automatically discovers all `views_NN/` variants under `vggt_output/` and processes each one.

**Processing order per crop:**
1. Load raw points for all scenes.
2. Estimate a centroid-translation alignment transform from each scene to the crop reference date.
3. Apply alignment to produce `points_aligned` (shared reference frame).
4. Compute normalization params (center XY, ground Z, scale, ROI bounds) from the aligned reference-date cloud only.
5. For each scene variant: apply ROI, normalize, save.

**Reference dates** used for normalization:

| Crop | Reference date | Note |
|---|---|---|
| corn | `20230812` | |
| soybean | `20230822` | 20230817 survey covers a smaller footprint; 20230822+ all share the same extent |

**Output per variant** — `geometry_assets/<scene_id>/views_NN/point_cloud_clean.npz`:

| Key | Shape | Description |
|---|---|---|
| `points_raw` | `(N, 3)` | VGGT-space points (used for RGB lookup) |
| `points_aligned` | `(N, 3)` | After centroid alignment to reference date |
| `points_normalized` | `(N, 3)` | After per-axis normalization (use this for training/eval) |
| `confidence` | `(N,)` | Per-point normalized confidence |

**Normalization** is per-axis: `x_norm = (x - center_x) / scale`, same for y and z (using `ground_z` as z offset). Scale is the 95th-percentile XY radius of the reference cloud.

Crop-level normalization params are saved to `geometry_assets/crop_normalization.json`.

Key options:

```
--conf-threshold 0.02   confidence cutoff (normalized 0–1)
--stride 2              pixel stride when flattening the point map
--voxel-size 0.02       voxel size for downsampling (scene units)
--max-points 500000     cap on final cloud size
--skip-existing         skip variants already built
```

### Step 3 — Build triplets from completed inference

After inference, scan `vggt_output/` to find which `(crop, date, batch_idx)` combinations are actually on disk and generate a triplets JSON covering only those.

```bash
conda run -n 4d python src/data_prep/build_available_triplets.py \
    --vggt-root vggt_output \
    --output prepared_data/subsets/available_triplets.json
```

Each triplet entry carries a `view_batch` integer — the same batch index is used for all three dates (`t1`, `t2`, `t3`). Only batch indices present for all three dates in a triplet are included.

With all scenes fully inferred (`n_views=32, max_overlap_views=16`):

| Crop | Date-triplets | Batches/triplet | Total |
|---|---|---|---|
| corn | 84 | 51 | 4,284 |
| soybean | 56 | 41–42 | 2,331 |
| **total** | 140 | — | **6,615** |

**Pre-inference estimate** (optional): to compute expected triplet counts before running inference, use `build_expanded_triplets.py` which derives batch counts from cleaned frame counts instead of disk state:

```bash
conda run -n 4d python src/data_prep/build_expanded_triplets.py \
    --inference-config configs/vggt_inference.yaml \
    --output prepared_data/subsets/expanded_triplets.json
```

### Step 3b — Build per-frame view triplets (optional)

These scripts build per-frame triplets across dates for each crop — one frame from t1, one from t2, one from t3 per entry. Both read from the subset manifest and require `selected_dates` and `selected_crops` in the config.

**All triplets** — randomly shuffles each date's frames and zips them together. Produces one entry per frame slot (limited by the shortest frame list in the group).

```bash
conda run -n 4d python src/vggt_pipeline/all_triplets.py --config configs/all_triplet.yaml
```

Output: `prepared_data/all_triplet.json` (configurable via `output_path`).

Key config options (`configs/all_triplet.yaml`):

```yaml
selected_dates: []          # list of date strings to include
selected_crops: []          # list of crop names to include
seed: 42                    # random seed for frame shuffling
output_path: prepared_data/all_triplet.json
```

**Camera-consistent triplets** — matches frames across dates that observe the same scene location (same camera position and view direction within thresholds). Uses t2's coordinate system as the reference.

```bash
conda run -n 4d python src/vggt_pipeline/camera_consistent_triplets.py --config configs/camera_consistent_triplet.yaml
```

Output: `prepared_data/camera_consistent_triplet.json` (configurable via `output_path`).

Key config options (`configs/camera_consistent_triplet.yaml`):

```yaml
selected_dates: []
selected_crops: []
max_position_distance_m: 0.1     # max camera-center distance in t2 world coords
max_view_angle_deg: 3.0          # max angle between viewing directions
max_tilt_difference_deg: 3.0     # max tilt-from-nadir difference
max_oblique_yaw_difference_deg: 5.0  # max horizontal yaw difference for oblique views
use_xy_position_only: false      # ignore Z when comparing positions
one_to_one: true                 # each frame used in at most one triplet
max_results: null                # cap total matched triplets (null = no limit)
output_path: prepared_data/camera_consistent_triplet.json
```

Each matched triplet is scored by a weighted sum of pairwise position distance and angle differences (lower is better) and selected greedily when `one_to_one: true`.

**t1-t2 paired triplets (free t3)** — matches frames only between t1 and t2; t3 views are unconstrained. Useful when you want camera-consistent t1/t2 pairs but want t3 to vary freely (e.g. to generate more training variants by sampling different t3 windows).

```bash
conda run -n 4d python src/vggt_pipeline/t1t2_paired_triplets.py --config configs/t1t2_paired_triplets.yaml
```

Output: `prepared_data/t1t2_paired_triplets.json` (configurable via `output_path`).

Each entry contains:
- `pairs` — list of camera-consistent `(v1, v2)` pairs for t1 and t2, sorted by match score
- `views_t3` — all t3 views in t2's coordinate system, no camera constraint

Key config options (`configs/t1t2_paired_triplets.yaml`):

```yaml
max_position_distance_m: 0.5     # max camera-center distance in t2 world coords
max_view_angle_deg: 3.0
max_tilt_difference_deg: 3.0
max_oblique_yaw_difference_deg: 3.0
one_to_one: true                 # each frame used in at most one pair
max_pairs: null                  # cap on pairs per (t1,t2) combination (null = no limit)
output_path: prepared_data/t1t2_paired_triplets.json
```

Then run VGGT inference using the paired triplets:

```bash
conda run -n 4d python src/vggt_pipeline/run_t1t2_paired_inference.py --config configs/t1t2_paired_inference.yaml
```

This script slides a window over the t1-t2 pairs and for each window samples multiple random t3 windows, producing more variants than the camera-consistent approach:

- **t1-t2 window**: same sliding window as `run_vggt_inference.py` (`n_views`, `max_overlap_views`)
- **t3 sampling**: `t3_variants_per_window` random windows of `n_views` drawn from `views_t3` per t1-t2 window
- **Variant naming**: `variant_{t1t2_idx:02d}_{t3_idx:02d}` — e.g. `variant_00_00`, `variant_00_01` share the same t1/t2 window but differ in t3

Key config options (`configs/t1t2_paired_inference.yaml`):

```yaml
paired_triplets_path: prepared_data/t1t2_paired_triplets.json
output_root: vggt_outputs/t1t2_paired_v16_o4
n_views: 16
max_overlap_views: 4
max_t1t2_windows: null          # cap on t1-t2 sliding windows per entry (null = all)
t3_variants_per_window: 3       # random t3 windows sampled per t1-t2 window
seed: 42                        # random seed for t3 sampling (reproducibility)
t2_only: true                   # only run VGGT for t2; write metadata for t1/t3
```

Output layout matches `run_vggt_inference.py`:

```
vggt_outputs/t1t2_paired_v16_o4/
└── {t1}_{t2}_{t3}_{crop}/
    └── variant_{t1t2_idx:02d}_{t3_idx:02d}/
        ├── t1/   (dataset_cameras.json, selected_images.json[, predictions/])
        ├── t2/   (same)
        └── t3/   (same)
```

**Non-view-consistent triplets** — keeps the same temporal logic (`t1 < t2 < t3`) but does not match camera views across dates. Each triplet stores all available views for t1, t2, and t3 independently; the inference script later builds variants from separate sliding windows for each date.

```bash
conda run -n 4d python src/vggt_pipeline/non_view_consistent_triplets.py \
    --config configs/non_view_consistent_triplets.yaml
```

Output: `prepared_data/non_view_consistent_triplets.json` (configurable via `output_path`).

Each entry contains:
- `views_t1` — all t1 views in t2's coordinate system
- `views_t2` — all t2 views
- `views_t3` — all t3 views in t2's coordinate system

Key config options (`configs/non_view_consistent_triplets.yaml`):

```yaml
selected_dates: []          # ordered date strings to include
selected_crops: []          # crop names to include
output_path: prepared_data/non_view_consistent_triplets.json
```

Then run VGGT inference using independent t1/t2/t3 view windows:

```bash
conda run -n 4d python src/vggt_pipeline/run_non_view_consistent_inference.py \
    --config configs/non_view_consistent_inference.yaml
```

Variant generation:
- **t1 window**: sliding window over `views_t1`
- **t2 window**: sliding window over `views_t2`
- **t3 window**: sliding window over `views_t3`
- **All variants**: every `(t1_window, t2_window, t3_window)` combination when `max_variants_per_triplet: null`
- **Limited variants**: deterministic random sample when `max_variants_per_triplet` is an integer
- **Variant naming**: `variant_{t1_idx:02d}_{t2_idx:02d}_{t3_idx:02d}`

Key config options (`configs/non_view_consistent_inference.yaml`):

```yaml
triplets_path: prepared_data/non_view_consistent_triplets.json
output_root: vggt_outputs/non_view_consistent_v16_o8
n_views: 16
max_overlap_views: 8
max_windows_per_date: null        # cap windows for each date (null = all)
max_variants_per_triplet: null    # cap final variants per triplet (null = all)
require_distinct_view_windows: true
seed: 42
prediction_outputs: none          # metadata-only; use "all" or a list to run VGGT
t2_cache_layers: [4, 11, 17, 23]
```

Dry-run variant counts without loading VGGT:

```bash
conda run -n 4d python src/vggt_pipeline/run_non_view_consistent_inference.py \
    --config configs/non_view_consistent_inference.yaml \
    --dry-run
```

Multi-GPU sharding uses the same rank arguments as the paired inference script:

```bash
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$r conda run -n 4d python src/vggt_pipeline/run_non_view_consistent_inference.py \
    --config configs/non_view_consistent_inference.yaml \
    --num-gpus 8 --gpu-rank $r --device cuda:0 &
done
wait
```

Output layout:

```
vggt_outputs/non_view_consistent_v16_o8/
└── {t1}_{t2}_{t3}_{crop}/
    └── variant_{t1_idx:02d}_{t2_idx:02d}_{t3_idx:02d}/
        ├── t1/   (dataset_cameras.json, selected_images.json[, predictions/])
        ├── t2/   (same)
        └── t3/   (same)
```

### Step 4 — Visualize (optional)

Interactive browser-based 3D viewer for geometry assets and model evaluation results.

```bash
conda run -n 4d python src/visualize.py --geometry-root geometry_assets --vggt-root vggt_output
```

Then open `http://localhost:8050`.

Four tabs are available:

| Tab | Description |
|---|---|
| **Single scene** | One scene; color by height (viridis) or RGB sampled from source images |
| **Compare** | 2–4 scenes side by side with synchronized camera — rotating one rotates all |
| **Overlay (red / blue)** | Two scenes overlaid in shared normalized space; scene A red, scene B blue |
| **Model vs GT** | Model prediction vs teacher t2 reference side by side (requires `--eval-root`) |

**Model vs GT tab** — requires eval output with saved clouds (run Step 6 with `--save-clouds` first):

```bash
conda run -n 4d python src/visualize.py \
    --geometry-root geometry_assets \
    --vggt-root vggt_output \
    --eval-root eval/model_v1
```

Select a fold and a test sample from the dropdowns. Both panels share a synchronized camera — rotating one rotates the other.

Options:

```
--geometry-root PATH   default: geometry_assets
--vggt-root PATH       default: vggt_output (needed for RGB color mode)
--eval-root PATH       default: none (enables Model vs GT tab when provided)
--port INT             default: 8050
--debug                enable Dash debug mode
```

### Step 5 — Train

See [`src/train.md`](src/train.md) for full training instructions and CLI options.

**V1** — patch-level time conditioning only:

```bash
conda run -n 4d python src/train.py --config configs/train_model_v1.yaml
```

**V2** — same as V1 plus block-level time conditioning injected before each LoRA-adapted aggregator block:

```bash
conda run -n 4d python src/train.py --config configs/train_model_v2.yaml
```

Trains one model per crop via LOTO cross-validation. Checkpoints go to `runs/<model>/<protocol>/<fold_id>/`.

### Step 6 — Evaluate

Evaluates the five non-training baselines (B0–B4) and the trained model (`best_model.pt` per fold) on all LOTO test folds. All inputs and references come from `vggt_output_root` — no separate geometry-assets build is required.

```bash
# Baselines + trained model:
conda run -n 4d python src/eval_model.py --config configs/train_model_v1.yaml \
    --runs-root runs/model_v1 --output-root eval/model_v1 --save-clouds

# Baselines only (no checkpoint needed):
conda run -n 4d python src/eval_model.py --config configs/train_model_v1.yaml \
    --output-root eval/baselines

# Single fold:
conda run -n 4d python src/eval_model.py --config configs/train_model_v1.yaml \
    --runs-root runs/model_v1 --crop corn --protocol strict --test-date 20230831
```

Add `--save-clouds` to save the predicted and reference point clouds per test sample for 3D visualization (see Step 7):

```bash
conda run -n 4d python src/eval_model.py --config configs/train_model_v1.yaml \
    --runs-root runs/model_v1 --output-root eval/model_v1 --save-clouds
```

Clouds are written to `<output-root>/<protocol>/<fold_id>/clouds/` as merged `{t1}_{t2}_{t3}_{crop}_{variant}_pred.npy` / `..._ref.npy`, plus per-view `..._pred_views.npz` / `..._ref_views.npz` when `--save-clouds` is used.

Per-fold results go to `<output-root>/<protocol>/<fold_id>/eval_result.json`.
Cross-fold summary is written to `<output-root>/eval_summary.json` and printed at the end.

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/train_model_v1.yaml` | YAML config (reuses train config) |
| `--runs-root` | — | `runs/model_v1/` from train.py; enables model evaluation |
| `--output-root` | `eval/model_v1` | Where to write results |
| `--crop corn` | (from config) | Evaluate only this crop; repeatable |
| `--protocol strict` | (from config) | Evaluate only this protocol; repeatable |
| `--test-date 20230831` | (from config) | Evaluate only this fold |
| `--device cuda:0` | (from config) | Compute device |
| `--save-clouds` | off | Save predicted + reference clouds for visualization |

**Baselines** (use VGGT t1/t3 point maps from `vggt_output_root` as inputs):

| Name | Description |
|---|---|
| `B0_t1_date_copy` | Copy t1 point cloud as prediction |
| `B1_t3_date_copy` | Copy t3 point cloud as prediction |
| `B2_nearest_date_copy` | Copy whichever endpoint is temporally closer to t2 |
| `B3_linear_interpolation` | Element-wise `(1−τ)·t1 + τ·t3` on matched-size random subsamples |
| `B4_temporal_weighted_union` | Sample from t1 and t3 proportional to `(1−τ)` and `τ` |

**Metrics** (computed against VGGT teacher t2 predictions as reference):

| Metric | Description |
|---|---|
| `accuracy` | Mean NN distance from predicted to reference points |
| `completeness` | Mean NN distance from reference to predicted points |
| `asymmetric_chamfer` | `eval_alpha × accuracy + eval_beta × completeness` (default 0.5 + 0.5) |
| `precision` | Fraction of predicted points within `distance_threshold` of reference |
| `recall` | Fraction of reference points within `distance_threshold` of predicted |
| `f1` | Harmonic mean of precision and recall |
| `normal_consistency` | Mean \|n_pred · n_ref_matched\| for nearest-neighbour pairs |
| `height_max_error` | Absolute canopy height error (PCA height axis) |
| `height_median_error` | Median height error (PCA height axis) |
| `voxel_iou` | Voxel occupancy IoU |
| `pointmap_l1` | Mean absolute pixel-aligned error (model only; NaN for baselines) |
| `pointmap_l2` | Mean L2 pixel-aligned error (model only; NaN for baselines) |

Results are broken down by: **overall**, **adjacent triplets** (t1 and t3 are immediate date neighbours of t2), and **multi-gap triplets** (one or more dates skipped between endpoints).

## Project structure

```
temporal_vggt/
  configs/
    prepare_data.yaml       dataset scanning settings (matic platform, 20230812–20230922)
    vggt_inference.yaml     VGGT inference settings (n_views, max_overlap_views)
    non_view_consistent_triplets.yaml   independent-view triplet generation
    non_view_consistent_inference.yaml  independent-view VGGT inference variants
    train_model_v1.yaml     v1 training config (patch-level time conditioning)
    train_model_v2.yaml     v2 training config (+ block-level time conditioning at LoRA layers)
  docs/                     project documentation
  prepared_data/            manifests, triplets, splits (pre-built)
  src/
    vggt_pipeline/
      execute_vggt.py           VGGT model loading and forward pass
      run_vggt_inference.py     scene-level inference; produces views_NN/ variants per scene
      non_view_consistent_triplets.py      build t1/t2/t3 triplets with independent view pools
      run_non_view_consistent_inference.py build/run independent t1/t2/t3 window variants
      build_geometry_assets.py  fuse point maps → aligned + normalized point clouds
    data_prep/
      build_available_triplets.py  scan vggt_output/ and build triplets from completed variants
      build_expanded_triplets.py   pre-inference estimate of triplet counts from frame counts
      ...                          dataset scanning and cleaning scripts
    dataset/
      triplet_dataset.py    PyTorch Dataset for (t1, t3, tau, t2) triplets
    losses/
      geometry.py           asymmetric Chamfer loss (Euclidean) + evaluation metrics
    models/
      base.py               abstract TemporalGeometryPredictor interface
      lora.py               LoRALinear wrapper
      time_encoding.py      TimeEncoder, BlockTimeEncoder, ResidualAdaLN, FiLM helpers
      camera_encoding.py    CameraEmbedding
      temporal_vggt_v1.py   V1 — patch-level time conditioning + LoRA + DPT head
      temporal_vggt_v2.py   V2 — V1 + block-level AdaLN before each LoRA aggregator block
    loto.py                 LOTO fold generation (target_date and strict)
    train.py                LOTO training loop
    train.md                training usage guide
    eval_model.py           baseline and model evaluation; --save-clouds for visualization
    visualize.py            interactive 3D viewer: geometry + Model vs GT (Dash)
```

## LOTO protocols

Two leave-one-date-out evaluation protocols are used, each run separately for corn and soybean:

**Target-date LOTO** (easier): hold out date `d` as the prediction target `t2`. The date `d` may still appear as `t1` or `t3` in training triplets.

**Strict LOTO** (main): hold out date `d` completely. The date `d` cannot appear as `t1`, `t2`, or `t3` in any training triplet.

Edge dates (the temporally earliest and latest dates in a crop's sequence) are excluded from the test date pool because they lack a bracket on one side and cannot serve as `t2`. The validation date is selected as the closest non-adjacent date to the test date from the same non-edge pool.

Results are reported per fold (per held-out date) and aggregated over all folds.

torchrun --nproc_per_node=4 src/eval_model.py \
  --config configs/train_residual_endpoint_adaln_freezed_head.yaml \
  --runs-root runs/residual_endpoint_t1t2_paired_v16_o8_adaln_1layer_freezed_head_cam \
  --output-root eval/residual_endpoint_freezed_head \
  --protocol strict \
  --crop corn \
  --test-date 20230831 \
  --save-clouds
