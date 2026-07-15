"""Visualize pre-computed StreamVGGT outputs using the StreamVGGT Gradio viewer.

Loads saved point maps and launches an interactive 3D viewer.

Usage:
    conda run -n 4d python src/visualize_streamvggt.py \
        --triplet-id 20230812_20230831_20230922_corn \
        --variant variant_00_00 --date t2

    # All dates combined:
    conda run -n 4d python src/visualize_streamvggt.py \
        --triplet-id 20230812_20230831_20230922_corn \
        --variant variant_00_00

    # Export GLB file without launching viewer:
    conda run -n 4d python src/visualize_streamvggt.py \
        --triplet-id 20230812_20230831_20230922_corn \
        --variant variant_00_00 --export-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

sys.path.insert(0, "/mnt/data/StreamVGGT/src")
from visual_util import predictions_to_glb


def load_predictions(variant_dir: Path, dates: list[str]) -> dict:
    """Load and merge predictions across dates into format expected by visual_util."""
    all_points, all_conf, all_extrinsic, all_images = [], [], [], []

    for date in dates:
        pred_dir = variant_dir / date / "predictions"
        if not pred_dir.exists():
            print(f"  skip {date}: no predictions")
            continue

        point_map = np.load(pred_dir / "point_map.npy")         # [S, H, W, 3]
        point_conf = np.load(pred_dir / "point_confidence.npy") # [S, H, W]
        extrinsic = np.load(pred_dir / "extrinsic.npy")         # [S, 3, 4]

        H, W = point_map.shape[1], point_map.shape[2]

        # Load source images resized to match point map
        sel_path = variant_dir / date / "selected_images.json"
        if sel_path.exists():
            selected = json.loads(sel_path.read_text())
            imgs = []
            for entry in selected:
                img = Image.open(entry["image_path"]).convert("RGB")
                img = img.resize((W, H), Image.Resampling.BICUBIC)
                imgs.append(TF.ToTensor()(img).numpy())  # [3, H, W]
            all_images.append(np.stack(imgs))  # [S, 3, H, W]

        all_points.append(point_map)
        all_conf.append(point_conf)
        all_extrinsic.append(extrinsic)

    return {
        "world_points": np.concatenate(all_points, axis=0),
        "world_points_conf": np.concatenate(all_conf, axis=0),
        "world_points_from_depth": np.concatenate(all_points, axis=0),
        "extrinsic": np.concatenate(all_extrinsic, axis=0),
        "images": np.concatenate(all_images, axis=0) if all_images else None,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path,
                   default=Path("vggt_outputs/streamvggt_v16_o8"))
    p.add_argument("--gt-root", type=Path,
                   default=Path("vggt_outputs/t1t2_paired_v16_o8"),
                   help="Ground truth VGGT outputs for comparison")
    p.add_argument("--triplet-id", required=True)
    p.add_argument("--variant", default="variant_00_00")
    p.add_argument("--date", choices=["t1", "t2", "t3", "all"], default="t2")
    p.add_argument("--conf-thres", type=float, default=50.0,
                   help="Confidence percentile threshold (0-100)")
    p.add_argument("--export-only", action="store_true",
                   help="Export GLB and exit (no Gradio)")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main():
    args = parse_args()
    variant_dir = args.output_root / args.triplet_id / args.variant
    gt_variant_dir = args.gt_root / args.triplet_id / args.variant
    dates = ["t1", "t2", "t3"] if args.date == "all" else [args.date]

    Path("visualization").mkdir(exist_ok=True)

    # StreamVGGT prediction
    print(f"Loading StreamVGGT: {args.triplet_id}/{args.variant} dates={dates}")
    pred = load_predictions(variant_dir, dates)
    print(f"  StreamVGGT points: {pred['world_points'].shape}")

    pred_glb = f"visualization/{args.triplet_id}_{args.variant}_{args.date}_streamvggt.glb"
    scene = predictions_to_glb(pred, conf_thres=args.conf_thres, show_cam=True,
                               prediction_mode="Pointmap Branch")
    scene.export(file_obj=pred_glb)
    print(f"Exported: {pred_glb}")

    # Ground truth (VGGT)
    gt_glb = None
    if gt_variant_dir.exists():
        print(f"Loading GT (VGGT): {args.triplet_id}/{args.variant} dates={dates}")
        gt = load_predictions(gt_variant_dir, dates)
        print(f"  VGGT points: {gt['world_points'].shape}")

        gt_glb = f"visualization/{args.triplet_id}_{args.variant}_{args.date}_vggt_gt.glb"
        scene_gt = predictions_to_glb(gt, conf_thres=args.conf_thres, show_cam=True,
                                      prediction_mode="Pointmap Branch")
        scene_gt.export(file_obj=gt_glb)
        print(f"Exported: {gt_glb}")
    else:
        print(f"  No GT found at {gt_variant_dir}")

    if args.export_only:
        return

    # Launch side-by-side Gradio viewer
    import gradio as gr

    with gr.Blocks() as demo:
        gr.Markdown(f"## {args.triplet_id} / {args.variant} / {args.date}")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### StreamVGGT (prediction)")
                gr.Model3D(value=pred_glb, height=500)
            with gr.Column():
                gr.Markdown("### VGGT (ground truth)")
                if gt_glb:
                    gr.Model3D(value=gt_glb, height=500)
                else:
                    gr.Markdown("*No GT available*")

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)


if __name__ == "__main__":
    main()


