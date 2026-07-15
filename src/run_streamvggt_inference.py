"""Run StreamVGGT on existing VGGT variant image selections.

Reads the existing variant structure (selected_images.json per date) from
vggt_outputs/t1t2_paired_v16_o8/ and re-runs inference using StreamVGGT,
saving outputs in the same format to a new output directory.

Usage:
    # All variants:
    conda run -n 4d python src/run_streamvggt_inference.py

    # Single triplet:
    conda run -n 4d python src/run_streamvggt_inference.py \
        --triplet-id 20230812_20230831_20230922_corn

    # Limit variants per triplet:
    conda run -n 4d python src/run_streamvggt_inference.py \
        --max-variants-per-triplet 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/mnt/data/StreamVGGT/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def run_streamvggt_inference(
    model: StreamVGGT, image_paths: list[str], device: str
) -> dict[str, np.ndarray]:
    """Run StreamVGGT streaming inference on images."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    images = load_and_preprocess_images(image_paths, mode="pad")  # [S, 3, H, W]
    S, _, H, W = images.shape

    frames = [{"img": images[i].unsqueeze(0).to(device)} for i in range(S)]

    with torch.amp.autocast("cuda", dtype=dtype):
        output = model.inference(frames)

    all_pts3d, all_conf, all_pose = [], [], []
    for res in output.ress:
        all_pts3d.append(res["pts3d_in_other_view"].squeeze(0).float().cpu().numpy())
        all_conf.append(res["conf"].squeeze(0).float().cpu().numpy())
        all_pose.append(res["camera_pose"].squeeze(0).float().cpu().numpy())

    point_map = np.stack(all_pts3d, axis=0)   # [S, H, W, 3]
    point_conf = np.stack(all_conf, axis=0)   # [S, H, W]
    pose_enc = np.stack(all_pose, axis=0)     # [S, 9]

    pose_t = torch.from_numpy(pose_enc).unsqueeze(0)
    extrinsic, _ = pose_encoding_to_extri_intri(pose_t, image_size_hw=(H, W))
    extrinsic = extrinsic.squeeze(0).numpy()  # [S, 3, 4]

    return {"point_map": point_map, "point_confidence": point_conf, "extrinsic": extrinsic}


def get_image_paths_from_variant(source_date_dir: Path) -> list[str] | None:
    """Read image paths from an existing variant's selected_images.json."""
    sel_path = source_date_dir / "selected_images.json"
    if not sel_path.exists():
        return None
    selected = json.loads(sel_path.read_text())
    return [entry["image_path"] for entry in selected]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run StreamVGGT on existing VGGT variants")
    p.add_argument("--source-root", type=Path,
                   default=Path("vggt_outputs/t1t2_paired_v16_o8"),
                   help="Existing VGGT output root with variants")
    p.add_argument("--output-root", type=Path,
                   default=Path("vggt_outputs/streamvggt_v16_o8"),
                   help="Where to write StreamVGGT outputs")
    p.add_argument("--ckpt", type=Path,
                   default=Path("/mnt/data/StreamVGGT/ckpt/checkpoints.pth"))
    p.add_argument("--triplet-id", action="append", default=None,
                   help="Process only these triplet(s)")
    p.add_argument("--test-date", default=None,
                   help="Process only triplets where middle_date == this value")
    p.add_argument("--max-variants-per-triplet", type=int, default=None,
                   help="Cap variants per triplet")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Total number of parallel workers")
    p.add_argument("--gpu-rank", type=int, default=0,
                   help="This worker's rank (0-indexed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Discover triplets and variants from source
    triplet_dirs = sorted(d for d in args.source_root.iterdir() if d.is_dir())
    if args.triplet_id:
        id_set = set(args.triplet_id)
        triplet_dirs = [d for d in triplet_dirs if d.name in id_set]
    if args.test_date:
        # Keep triplets where middle_date matches (format: left_middle_right_crop)
        triplet_dirs = [d for d in triplet_dirs if d.name.split("_")[1] == args.test_date]

    device = args.device or f"cuda:{args.gpu_rank}"

    log(f"Source: {args.source_root} ({len(triplet_dirs)} triplets)")
    log(f"Output: {args.output_root}")

    # Load model
    log(f"Loading StreamVGGT from {args.ckpt} -> {device}")
    model = StreamVGGT()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    model.eval().to(device)
    log("Model loaded")

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Build flat work list
    work_items = []
    for triplet_dir in triplet_dirs:
        variants = sorted(d for d in triplet_dir.iterdir() if d.is_dir())
        if args.max_variants_per_triplet:
            variants = variants[:args.max_variants_per_triplet]
        for variant_dir in variants:
            work_items.append((triplet_dir.name, variant_dir))

    # Shard across GPUs
    my_items = work_items[args.gpu_rank::args.num_gpus]
    log(f"rank={args.gpu_rank}/{args.num_gpus} total={len(work_items)} mine={len(my_items)}")

    for triplet_id, variant_dir in tqdm(my_items, desc=f"GPU{args.gpu_rank}"):
        variant_name = variant_dir.name
        out_variant = args.output_root / triplet_id / variant_name

        if args.skip_existing and all(
            (out_variant / t / "predictions" / "point_map.npy").exists()
            for t in ("t1", "t2", "t3")
        ):
            continue

        for t_label in ("t1", "t2", "t3"):
            out_date = out_variant / t_label
            if args.skip_existing and (out_date / "predictions" / "point_map.npy").exists():
                continue

            source_date = variant_dir / t_label
            image_paths = get_image_paths_from_variant(source_date)
            if image_paths is None:
                continue

            predictions = run_streamvggt_inference(model, image_paths, device)

            pred_dir = out_date / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.save(pred_dir / "point_map.npy", predictions["point_map"])
            np.save(pred_dir / "point_confidence.npy", predictions["point_confidence"])
            np.save(pred_dir / "extrinsic.npy", predictions["extrinsic"])

            for meta_file in ("selected_images.json", "dataset_cameras.json"):
                src = source_date / meta_file
                if src.exists():
                    (out_date / meta_file).write_text(src.read_text())

            torch.cuda.empty_cache()

    log(f"Done. Output: {args.output_root}")


if __name__ == "__main__":
    main()

