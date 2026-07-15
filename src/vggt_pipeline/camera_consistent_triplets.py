import argparse
import json
from pathlib import Path
from typing import Any
import math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_manifest": "prepared_data/manifests/dataset_manifest.json",
    "subset_manifest": "prepared_data/subsets/benchmark_subset.json",
    "output_path": "prepared_data/camera_consistent_triplet.json",
    "selected_dates": [],
    "selected_crops": [],
    "max_position_distance_m": 0.1,
    "max_view_angle_deg": 3.0,
    "max_tilt_difference_deg": 3.0,
    "max_oblique_yaw_difference_deg": 5.0,
    "use_xy_position_only": False,
    "one_to_one": True,
    "max_results": None,
}



def _parse_offset(offset):
    if isinstance(offset, str):
        return np.fromstring(offset, sep=" ", dtype=float)
    return np.asarray(offset, dtype=float)


def _normalize_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def _angle_between_deg(a, b):
    a = _normalize_vector(a)
    b = _normalize_vector(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _horizontal_angle_deg(a, b):
    a_xy = np.asarray([a[0], a[1]], dtype=float)
    b_xy = np.asarray([b[0], b[1]], dtype=float)

    if np.linalg.norm(a_xy) < 1e-8 or np.linalg.norm(b_xy) < 1e-8:
        return 0.0

    return _angle_between_deg(a_xy, b_xy)


def _load_json(path):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def _load_views_using_t2_world(
    transforms_path,
    date_label,
    t2_scale,
    t2_offset,
):
    """
    Load one transforms.json file.

    Camera centers from this file are converted into the t2 world coordinate
    convention using:

        position_t2_world = normalized_position * t2_scale + t2_offset

    This intentionally ignores this file's own scale and offset.
    """
    path = Path(transforms_path)
    data = _load_json(path)

    views = []

    for frame_index, frame in enumerate(data["frames"]):
        M = np.asarray(frame["transform_matrix"], dtype=float)

        if M.shape != (4, 4):
            raise ValueError(
                f"Bad transform_matrix shape in {path}, frame {frame_index}: {M.shape}"
            )

        R = M[:3, :3]
        t_normalized = M[:3, 3]

        # Force all dates into t2's world coordinate convention.
        position_t2_world = t_normalized * t2_scale + t2_offset

        # Nerfstudio/OpenGL convention: camera looks along local -Z.
        view_direction = -R[:, 2]
        view_direction = _normalize_vector(view_direction)

        # 0 degrees means straight down/nadir.
        tilt_from_nadir_deg = _angle_between_deg(
            view_direction,
            np.array([0.0, 0.0, -1.0], dtype=float),
        )

        views.append({
            "date": date_label,
            "source_transforms_path": str(path),
            "frame_index": frame_index,
            "colmap_im_id": frame.get("colmap_im_id"),
            "image_path": frame.get("file_path"),

            # Original matrix from that date.
            "transform_matrix": M.tolist(),

            # Position converted using t2 scale/offset.
            "position_t2_world": position_t2_world.tolist(),

            "view_direction": view_direction.tolist(),
            "tilt_from_nadir_deg": float(tilt_from_nadir_deg),
            "is_oblique": bool(tilt_from_nadir_deg >= 30.0),
        })

    return views


def _pair_metrics(v_a, v_b, use_xy_position_only=False):
    p_a = np.asarray(v_a["position_t2_world"], dtype=float)
    p_b = np.asarray(v_b["position_t2_world"], dtype=float)

    if use_xy_position_only:
        position_distance = float(np.linalg.norm(p_a[:2] - p_b[:2]))
    else:
        position_distance = float(np.linalg.norm(p_a - p_b))

    d_a = np.asarray(v_a["view_direction"], dtype=float)
    d_b = np.asarray(v_b["view_direction"], dtype=float)

    view_angle = float(_angle_between_deg(d_a, d_b))
    tilt_difference = float(
        abs(v_a["tilt_from_nadir_deg"] - v_b["tilt_from_nadir_deg"])
    )
    yaw_difference = float(_horizontal_angle_deg(d_a, d_b))

    return {
        "position_distance_m": position_distance,
        "view_angle_difference_deg": view_angle,
        "tilt_difference_deg": tilt_difference,
        "yaw_difference_deg": yaw_difference,
    }


def _passes_thresholds(
    metrics,
    v_a,
    v_b,
    max_position_distance_m,
    max_view_angle_deg,
    max_tilt_difference_deg,
    max_oblique_yaw_difference_deg,
):
    if metrics["position_distance_m"] > max_position_distance_m:
        return False

    if metrics["view_angle_difference_deg"] > max_view_angle_deg:
        return False

    if metrics["tilt_difference_deg"] > max_tilt_difference_deg:
        return False

    # Yaw is critical for oblique views.
    # For near-nadir views, yaw is often less meaningful.
    if v_a["is_oblique"] or v_b["is_oblique"]:
        if metrics["yaw_difference_deg"] > max_oblique_yaw_difference_deg:
            return False

    return True


def _triplet_score(m12, m23, m13):
    """
    Lower score is better.

    Position is in meters.
    Angles are in degrees.
    """
    return float(
        1.00 * (
            m12["position_distance_m"]
            + m23["position_distance_m"]
            + m13["position_distance_m"]
        )
        + 0.35 * (
            m12["view_angle_difference_deg"]
            + m23["view_angle_difference_deg"]
            + m13["view_angle_difference_deg"]
        )
        + 0.15 * (
            m12["tilt_difference_deg"]
            + m23["tilt_difference_deg"]
            + m13["tilt_difference_deg"]
        )
        + 0.10 * (
            m12["yaw_difference_deg"]
            + m23["yaw_difference_deg"]
            + m13["yaw_difference_deg"]
        )
    )


def _compact_view(v):
    return {
        "date": v["date"],
        "frame_index": v["frame_index"],
        "colmap_im_id": v["colmap_im_id"],
        "image_path": v["image_path"],
        "source_transforms_path": v["source_transforms_path"],
        "transform_matrix": v["transform_matrix"],
        "position_t2_world": v["position_t2_world"],
        "view_direction": v["view_direction"],
        "tilt_from_nadir_deg": v["tilt_from_nadir_deg"],
        "is_oblique": v["is_oblique"],
    }


def _build_pair_candidates(
    views_a,
    views_b,
    max_position_distance_m,
    max_view_angle_deg,
    max_tilt_difference_deg,
    max_oblique_yaw_difference_deg,
    use_xy_position_only=False,
):
    """
    Return candidate pairs between two dates.

    Uses KD-tree for position prefiltering, then checks view-angle thresholds.
    """
    if len(views_a) == 0 or len(views_b) == 0:
        return []

    if use_xy_position_only:
        b_positions = np.asarray(
            [v["position_t2_world"][:2] for v in views_b],
            dtype=float,
        )
    else:
        b_positions = np.asarray(
            [v["position_t2_world"] for v in views_b],
            dtype=float,
        )

    tree = cKDTree(b_positions)

    pairs = []

    for i, va in enumerate(views_a):
        if use_xy_position_only:
            pa = np.asarray(va["position_t2_world"][:2], dtype=float)
        else:
            pa = np.asarray(va["position_t2_world"], dtype=float)

        candidate_indices = tree.query_ball_point(
            pa,
            r=max_position_distance_m,
        )

        for j in candidate_indices:
            vb = views_b[j]
            metrics = _pair_metrics(
                va,
                vb,
                use_xy_position_only=use_xy_position_only,
            )

            if not _passes_thresholds(
                metrics,
                va,
                vb,
                max_position_distance_m,
                max_view_angle_deg,
                max_tilt_difference_deg,
                max_oblique_yaw_difference_deg,
            ):
                continue

            pairs.append({
                "index_a": i,
                "index_b": j,
                "metrics": metrics,
            })

    return pairs


def find_same_view_triplets_t2_reference(
    t1_transforms_path,
    t2_transforms_path,
    t3_transforms_path,
    date_labels=("t1", "t2", "t3"),

    max_position_distance_m=2.0,
    max_view_angle_deg=5.0,
    max_tilt_difference_deg=5.0,
    max_oblique_yaw_difference_deg=10.0,

    use_xy_position_only=False,
    one_to_one=True,
    max_results=None,
):
    """
    Find same-view triplets across three dates using t2 as the reference
    coordinate system.

    All camera centers are converted using t2's scale and offset:

        position_t2_world = transform_matrix[:3, 3] * t2_scale + t2_offset

    Parameters
    ----------
    t1_transforms_path, t2_transforms_path, t3_transforms_path : str or Path
        Paths to the three transforms.json files.

    date_labels : tuple[str, str, str]
        Labels for the three dates.

    max_position_distance_m : float
        Max camera-center distance in t2 world coordinates.

    max_view_angle_deg : float
        Max angle between viewing directions.

    max_tilt_difference_deg : float
        Max difference in tilt from nadir.

    max_oblique_yaw_difference_deg : float
        Max horizontal direction difference for oblique views.

    use_xy_position_only : bool
        If True, compare only X/Y camera position, ignoring Z.
        For drone crop data, this is often safer if altitude/Z is noisy.

    one_to_one : bool
        If True, each frame from each date can be used in only one triplet.

    max_results : None or int
        If set, return only the top N triplets.

    Returns
    -------
    list[dict]
        List of matched triplets. No DataFrame is returned.

    Each item contains:
        - v1
        - v2
        - v3
        - canonical transform from t2
        - pairwise metrics
        - score
    """

    if len(date_labels) != 3:
        raise ValueError("date_labels must contain exactly three values.")

    # Use t2's scale and offset as the reference.
    t2_data = _load_json(t2_transforms_path)
    t2_scale = float(t2_data.get("scale", 1.0))
    t2_offset = _parse_offset(t2_data.get("offset", [0.0, 0.0, 0.0]))

    views1 = _load_views_using_t2_world(
        t1_transforms_path,
        date_label=date_labels[0],
        t2_scale=t2_scale,
        t2_offset=t2_offset,
    )

    views2 = _load_views_using_t2_world(
        t2_transforms_path,
        date_label=date_labels[1],
        t2_scale=t2_scale,
        t2_offset=t2_offset,
    )

    views3 = _load_views_using_t2_world(
        t3_transforms_path,
        date_label=date_labels[2],
        t2_scale=t2_scale,
        t2_offset=t2_offset,
    )

    # Candidate pairs t1-t2 and t2-t3.
    pairs12 = _build_pair_candidates(
        views1,
        views2,
        max_position_distance_m=max_position_distance_m,
        max_view_angle_deg=max_view_angle_deg,
        max_tilt_difference_deg=max_tilt_difference_deg,
        max_oblique_yaw_difference_deg=max_oblique_yaw_difference_deg,
        use_xy_position_only=use_xy_position_only,
    )

    pairs23 = _build_pair_candidates(
        views2,
        views3,
        max_position_distance_m=max_position_distance_m,
        max_view_angle_deg=max_view_angle_deg,
        max_tilt_difference_deg=max_tilt_difference_deg,
        max_oblique_yaw_difference_deg=max_oblique_yaw_difference_deg,
        use_xy_position_only=use_xy_position_only,
    )

    # Group t1-t2 pairs by t2 index.
    t2_to_t1 = defaultdict(list)
    for p in pairs12:
        t2_to_t1[p["index_b"]].append(p)

    # Group t2-t3 pairs by t2 index.
    t2_to_t3 = defaultdict(list)
    for p in pairs23:
        t2_to_t3[p["index_a"]].append(p)

    candidates = []

    # Join through the same t2 view.
    # This makes t2 the anchor view for the whole triplet.
    common_t2_indices = set(t2_to_t1.keys()) & set(t2_to_t3.keys())

    for idx2 in common_t2_indices:
        for p12 in t2_to_t1[idx2]:
            for p23 in t2_to_t3[idx2]:
                idx1 = p12["index_a"]
                idx3 = p23["index_b"]

                v1 = views1[idx1]
                v2 = views2[idx2]
                v3 = views3[idx3]

                # Final safety check: v1 and v3 must also match each other.
                m13 = _pair_metrics(
                    v1,
                    v3,
                    use_xy_position_only=use_xy_position_only,
                )

                if not _passes_thresholds(
                    m13,
                    v1,
                    v3,
                    max_position_distance_m,
                    max_view_angle_deg,
                    max_tilt_difference_deg,
                    max_oblique_yaw_difference_deg,
                ):
                    continue

                m12 = p12["metrics"]
                m23 = p23["metrics"]
                score = _triplet_score(m12, m23, m13)

                candidates.append({
                    "v1": _compact_view(v1),
                    "v2": _compact_view(v2),
                    "v3": _compact_view(v3),

                    # t2 represents the triplet transform.
                    "canonical_date": date_labels[1],
                    "canonical_frame_index": v2["frame_index"],
                    "canonical_image_path": v2["image_path"],
                    "canonical_transform_matrix": v2["transform_matrix"],

                    "pairwise_metrics": {
                        "v1_v2": m12,
                        "v2_v3": m23,
                        "v1_v3": m13,
                    },

                    "score": score,

                    "reference_coordinate_system": {
                        "reference_date": date_labels[1],
                        "reference_transforms_path": str(t2_transforms_path),
                        "scale_used": float(t2_scale),
                        "offset_used": t2_offset.tolist(),
                        "position_field": "position_t2_world",
                    },
                })

    candidates.sort(key=lambda x: x["score"])

    if not one_to_one:
        if max_results is not None:
            return candidates[:max_results]
        return candidates

    # Greedy one-to-one selection.
    selected = []
    used_t1 = set()
    used_t2 = set()
    used_t3 = set()

    for triplet in candidates:
        k1 = triplet["v1"]["frame_index"]
        k2 = triplet["v2"]["frame_index"]
        k3 = triplet["v3"]["frame_index"]

        if k1 in used_t1 or k2 in used_t2 or k3 in used_t3:
            continue

        selected.append(triplet)
        used_t1.add(k1)
        used_t2.add(k2)
        used_t3.add(k3)

        if max_results is not None and len(selected) >= max_results:
            break

    return selected


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for config files.") from exc
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must be a key-value mapping.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT inference for benchmark scenes."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/camera_consistent_triplet.yaml")
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = dict(DEFAULT_CONFIG)
    config.update(load_yaml_config(args.config))
    
    return config


def main(config: dict[str, Any]):
    selected_dates = sorted(config["selected_dates"])

    dataset = read_json(Path(config["subset_manifest"]))
    scenes = dataset["scenes"]
    by_crop = {crop: {} for crop in config["selected_crops"]}

    for scene in scenes:
        by_crop[scene["crop"]][scene["date"]] = scene
    all_triplets = []
    for crop in config["selected_crops"]:
        by_date = by_crop[crop]
        for i in range(len(selected_dates)):
            t1 = selected_dates[i]
            if t1 not in by_date:
                continue
            for j in range(i + 1, len(selected_dates)):
                t2 = selected_dates[j]
                if t2 not in by_date:
                    continue
                for k in range(j + 1, len(selected_dates)):
                    t3 = selected_dates[k]
                    if t3 not in by_date:
                        continue
                    triplets = find_same_view_triplets_t2_reference(
                        t1_transforms_path=by_date[t1]["transforms_path"],
                        t2_transforms_path=by_date[t2]["transforms_path"],
                        t3_transforms_path=by_date[t3]["transforms_path"],
                        max_position_distance_m=config["max_position_distance_m"],
                        max_view_angle_deg=config["max_view_angle_deg"],
                        max_tilt_difference_deg=config["max_tilt_difference_deg"],
                        max_oblique_yaw_difference_deg=config["max_oblique_yaw_difference_deg"],
                        use_xy_position_only=config["use_xy_position_only"],
                        one_to_one=config["one_to_one"],
                        max_results=config["max_results"]
                    )
                    for triplet in triplets:
                        del triplet["v1"]["frame_index"]
                        del triplet["v2"]["frame_index"]
                        del triplet["v3"]["frame_index"]
                    print(f"Triplet {t1}_{t2}_{t3}_{crop}: {len(triplets)} views")
                    all_triplets.append({
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "crop": crop,
                        "triplets": triplets
                    })
    for crop in config["selected_crops"]:
        print(f"{crop}: {len(list(filter(lambda x: x['crop'] == crop, all_triplets)))} triplets")
    write_json(Path(config["output_path"]), all_triplets)


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    main(config)

