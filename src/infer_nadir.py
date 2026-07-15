"""Run VGGT inference on nadir-view images and visualize results.

Usage:
    # All corn images
    conda run -n 4d python src/infer_nadir.py --crop corn

    # All soy images, specific GPU
    conda run -n 4d python src/infer_nadir.py --crop soy --device cuda:1

    # Pick specific files
    conda run -n 4d python src/infer_nadir.py \
        --images corn_0812.tif corn_0831.tif corn_0922.tif

    # Custom nadir dir
    conda run -n 4d python src/infer_nadir.py --crop corn \
        --nadir-dir /data/nak168/learning_3d/canopy_data/Nadir_view_images

Outputs (written to --output-dir, default nadir_inference/<crop>_<timestamp>):
    depth_maps.png      — per-image depth heatmaps
    confidence.png      — per-image confidence maps
    point_cloud.html    — interactive 3D Plotly point cloud colored by image RGB
    predictions/        — raw .npy arrays (point_map, depth_map, etc.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


NADIR_DIR = Path("/data/nak168/learning_3d/canopy_data/Nadir_view_images")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ─── image selection ──────────────────────────────────────────────────────────

def resolve_images(args: argparse.Namespace) -> list[Path]:
    nadir_dir = Path(args.nadir_dir)
    if args.images:
        paths = [nadir_dir / f if not Path(f).is_absolute() else Path(f) for f in args.images]
    elif args.crop:
        paths = sorted(nadir_dir.glob(f"{args.crop}_*.tif"))
        if not paths:
            sys.exit(f"No images found for crop '{args.crop}' in {nadir_dir}")
    else:
        paths = sorted(nadir_dir.glob("*.tif"))

    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"Images not found: {missing}")

    log(f"Selected {len(paths)} images:")
    for p in paths:
        log(f"  {p.name}")
    return paths


# ─── inference ────────────────────────────────────────────────────────────────

def run_inference(image_paths: list[Path], device: str, model_id: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vggt_pipeline.execute_vggt import get_vggt_runner, run_vggt_inference_in_memory

    runner = get_vggt_runner(model_id=model_id, device=device)
    log("Running VGGT forward pass...")
    preds = run_vggt_inference_in_memory(
        [str(p) for p in image_paths],
        runner,
        image_preprocess_mode="pad",
    )
    log(f"  point_map:   {tuple(preds['point_map'].shape)}")
    log(f"  depth_map:   {tuple(preds['depth_map'].shape)}")
    log(f"  confidence:  {tuple(preds['point_confidence'].shape)}")
    return preds


def save_predictions(preds: dict, out_dir: Path) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key, tensor in preds.items():
        np.save(pred_dir / f"{key}.npy", tensor.numpy())
    log(f"Saved raw predictions to {pred_dir}")


# ─── visualization ────────────────────────────────────────────────────────────

def _load_image_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """Load image resized to (H, W), returns float32 [0,1] RGB."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))
    return np.array(img, dtype=np.float32) / 255.0


def plot_depth_maps(preds: dict, image_paths: list[Path], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    depth  = preds["depth_map"].numpy()       # [S, H, W] or [S, H, W, 1]
    conf   = preds["point_confidence"].numpy() # [S, H, W] or [S, H, W, 1]
    if depth.ndim == 4: depth = depth.squeeze(-1)
    if conf.ndim == 4:  conf  = conf.squeeze(-1)
    S = depth.shape[0]
    ncols = min(S, 4)
    nrows = (S + ncols - 1) // ncols

    for arr, fname, title in [
        (depth, "depth_maps.png",  "Depth"),
        (conf,  "confidence.png",  "Confidence"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        for idx in range(S):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            im = ax.imshow(arr[idx], vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(image_paths[idx].stem, fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for idx in range(S, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        save_path = out_dir / fname
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved {save_path}")


def plot_point_cloud(preds: dict, image_paths: list[Path], out_dir: Path,
                     conf_threshold: float, max_points: int) -> None:
    import plotly.graph_objects as go

    point_map = preds["point_map"].numpy()         # [S, H, W, 3]
    conf      = preds["point_confidence"].numpy()  # [S, H, W] or [S, H, W, 1]
    if conf.ndim == 4: conf = conf.squeeze(-1)
    S, H, W, _ = point_map.shape

    # Sample one color per image (load + resize to match VGGT output H×W)
    all_pts, all_colors = [], []
    for i, path in enumerate(image_paths):
        rgb = _load_image_rgb(path, (H, W))        # [H, W, 3] float32
        mask = conf[i] > conf_threshold
        pts  = point_map[i][mask]                  # [N, 3]
        col  = rgb[mask]                           # [N, 3]
        all_pts.append(pts)
        all_colors.append(col)

    pts    = np.concatenate(all_pts,    axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Subsample if needed
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, colors = pts[idx], colors[idx]

    def to_hex(c: np.ndarray) -> list[str]:
        c = np.clip(c, 0, 1)
        r, g, b = (c * 255).astype(np.uint8).T
        return [f"#{ri:02x}{gi:02x}{bi:02x}" for ri, gi, bi in zip(r, g, b)]

    scatter = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=to_hex(colors), opacity=0.85),
        name="point cloud",
    )

    # Camera positions from extrinsic [S, 3, 4]: camera center = -R^T @ t
    ext = preds["extrinsic"].numpy()   # [S, 3, 4]
    cam_centers = -ext[:, :3, :3].transpose(0, 2, 1) @ ext[:, :3, 3:]
    cam_centers = cam_centers.squeeze(-1)            # [S, 3]
    cam_scatter = go.Scatter3d(
        x=cam_centers[:, 0], y=cam_centers[:, 1], z=cam_centers[:, 2],
        mode="markers+text",
        marker=dict(size=6, color="red", symbol="diamond"),
        text=[p.stem for p in image_paths],
        textposition="top center",
        name="cameras",
    )

    fig = go.Figure(data=[scatter, cam_scatter])
    fig.update_layout(
        title=f"Nadir VGGT — {S} images",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    save_path = out_dir / "point_cloud.html"
    fig.write_html(str(save_path))
    log(f"Saved interactive point cloud to {save_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT inference + visualization on nadir images.")
    p.add_argument("--nadir-dir", default=str(NADIR_DIR))
    p.add_argument("--crop", choices=["corn", "soy"], default=None,
                   help="Filter images by crop name.")
    p.add_argument("--images", nargs="+", default=None,
                   help="Explicit image filenames (relative to --nadir-dir or absolute).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-id", default="facebook/VGGT-1B")
    p.add_argument("--conf-threshold", type=float, default=0.1,
                   help="Min point confidence for point cloud visualization.")
    p.add_argument("--max-points", type=int, default=200_000,
                   help="Max points to render in the 3D plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    image_paths = resolve_images(args)

    tag = args.crop or "custom"
    ts  = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"nadir_inference/{tag}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output dir: {out_dir}")

    # get_vggt_runner caches the model so it's only loaded once across the loop.
    for i, path in enumerate(image_paths):
        log(f"[{i+1}/{len(image_paths)}] Processing {path.name}")
        date_dir = out_dir / path.stem
        date_dir.mkdir(parents=True, exist_ok=True)

        preds = run_inference([path], args.device, args.model_id)
        save_predictions(preds, date_dir)
        plot_depth_maps(preds, [path], date_dir)
        plot_point_cloud(preds, [path], date_dir, args.conf_threshold, args.max_points)

    log("Done.")


if __name__ == "__main__":
    main()

