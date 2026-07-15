"""Interactive 3D point cloud viewer for temporal-VGGT geometry assets.

Three modes
-----------
single  – one scene, height colormap or RGB sampled from source images
compare – 2-4 scenes side by side with synchronized camera
overlay – two scenes overlaid in shared normalized space (red vs blue)

Usage
-----
python src/visualize.py
python src/visualize.py --geometry-root geometry_assets --vggt-root vggt_output
python src/visualize.py --port 8051 --debug
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from dash import Dash, Input, Output, State, Patch, ctx, dcc, html, no_update
import plotly.graph_objects as go


# ─── constants ───────────────────────────────────────────────────────────────

MAX_COMPARE = 4
MAX_POINTS = 80_000


# ─── geometry helpers ────────────────────────────────────────────────────────

# ─── eval cloud helpers ───────────────────────────────────────────────────────

def discover_eval_clouds(eval_root: Path) -> dict[str, dict]:
    """Scan eval_root for fold dirs that have saved clouds from --save-clouds.

    Returns {fold_id: {"clouds_dir": Path, "samples": [key, ...]}} where each
    key is like '20230812_20230831_20230922_corn_views_00'.
    """
    folds: dict[str, dict] = {}
    for clouds_dir in sorted(eval_root.rglob("clouds")):
        if not clouds_dir.is_dir():
            continue
        fold_id = clouds_dir.parent.name
        samples = sorted(f.stem[:-5] for f in clouds_dir.glob("*_pred.npy"))  # strip "_pred"
        if samples:
            folds[fold_id] = {"clouds_dir": clouds_dir, "samples": samples}
    return folds


def _parse_sample_label(key: str) -> str:
    """'20230812_20230831_20230922_corn_views_00' → readable label."""
    parts = key.split("_")
    if len(parts) >= 5:
        t1, t2, t3, crop = parts[0], parts[1], parts[2], parts[3]
        variant = "_".join(parts[4:])
        fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        return f"{crop} · {fmt(t1)} → {fmt(t2)} ← {fmt(t3)} · {variant}"
    return key


# ─── data loading ────────────────────────────────────────────────────────────

def discover_scenes(geometry_root: Path) -> list[str]:
    if not geometry_root.exists():
        return []
    return sorted(
        d.name for d in geometry_root.iterdir()
        if d.is_dir() and (d / "vggt_cameras" / "point_cloud_clean.npz").exists()
    )


def load_geo(
    scene_id: str,
    geometry_root: Path,
    max_points: int = MAX_POINTS,
    z_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load points_normalized from geometry_assets. Returns (pts [N,3], z_color [N] in [0,1])."""
    data = np.load(geometry_root / scene_id / "vggt_cameras" / "point_cloud_clean.npz")
    key = "points_normalized" if "points_normalized" in data else "points"
    pts = data[key].astype(np.float32)
    if max_points > 0 and len(pts) > max_points:
        idx = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    z = pts[:, 2]
    z_col = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)
    pts = pts.copy()
    pts[:, 2] *= z_scale
    return pts, z_col


def load_rgb(
    scene_id: str,
    vggt_root: Path,
    geometry_root: Path,
    max_points: int = MAX_POINTS,
    z_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load same points as load_geo, colored by nearest source image pixel.

    Uses points_raw (original VGGT coords saved alongside points_normalized) to
    look up each point's color in the raw point_map + source images via KDTree.
    This guarantees RGB mode shows exactly the same points the model uses.
    """
    data = np.load(geometry_root / scene_id / "vggt_cameras" / "point_cloud_clean.npz")
    key = "points_normalized" if "points_normalized" in data else "points"
    pts = data[key].astype(np.float32)
    pts_raw = data["points_raw"].astype(np.float32) if "points_raw" in data else None

    if max_points > 0 and len(pts) > max_points:
        idx = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if pts_raw is not None:
            pts_raw = pts_raw[idx]

    pm_path = vggt_root / scene_id / "vggt_cameras" / "predictions" / "point_map.npy"
    if pts_raw is None or not pm_path.exists():
        return pts, np.full((len(pts), 3), 128, dtype=np.uint8)

    pm = np.load(pm_path)   # (S, H, W, 3)
    S, H, W = pm.shape[:3]

    images = np.zeros((S, H, W, 3), dtype=np.uint8)
    sel_path = vggt_root / scene_id / "vggt_cameras" / "selected_images.json"
    if sel_path.exists():
        for i, item in enumerate(json.loads(sel_path.read_text())[:S]):
            p = item.get("image_path", "")
            if p and Path(p).exists():
                img = Image.open(p).convert("RGB")
                images[i] = np.array(img.resize((W, H), Image.BILINEAR))

    # Use the same stride as the build step so pts_raw coords appear exactly in pm_flat,
    # making nearest-neighbor lookup exact (distance ≈ 0).
    meta_path = geometry_root / scene_id / "vggt_cameras" / "geometry_metadata.json"
    build_stride = 2
    if meta_path.exists():
        build_stride = json.loads(meta_path.read_text()).get("stride", 2)
    s = max(1, build_stride)

    pm_flat = pm[:, ::s, ::s, :].reshape(-1, 3).astype(np.float32)
    img_flat = images[:, ::s, ::s, :].reshape(-1, 3)
    finite = np.isfinite(pm_flat).all(1)
    pm_valid = pm_flat[finite]
    img_valid = img_flat[finite]

    try:
        from scipy.spatial import KDTree
        _, nn_idx = KDTree(pm_valid).query(pts_raw, workers=-1)
        rgb = img_valid[nn_idx]
    except Exception:
        rgb = np.full((len(pts), 3), 128, dtype=np.uint8)

    pts = pts.copy()
    pts[:, 2] *= z_scale
    return pts, rgb


# ─── plotly traces ───────────────────────────────────────────────────────────

def _trace_height(pts: np.ndarray, z_col: np.ndarray, name: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1.5, color=z_col, colorscale="Viridis", showscale=False, opacity=0.85),
        name=name, hoverinfo="none",
    )


def _trace_rgb(pts: np.ndarray, rgb: np.ndarray, name: str) -> go.Scatter3d:
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb.tolist()]
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1.5, color=colors, opacity=0.85),
        name=name, hoverinfo="none",
    )


def _trace_flat(pts: np.ndarray, color: str, name: str, opacity: float = 0.55) -> go.Scatter3d:
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1.5, color=color, opacity=opacity),
        name=name, hoverinfo="none",
    )


def _layout(title: str = "", uirevision: str = "shared") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=11, color="#aaa")),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0d0d0d",
        ),
        paper_bgcolor="#1c1c1c",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=25, b=0),
        uirevision=uirevision,
        legend=dict(font=dict(size=10), x=0, y=1),
    )


def _empty(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_layout())
    if msg:
        fig.add_annotation(text=msg, showarrow=False, font=dict(color="#555", size=13))
    return fig


def _label(scene_id: str) -> str:
    return scene_id.removeprefix("nerfstudio_").replace("_", " ")


# ─── app ─────────────────────────────────────────────────────────────────────

_DD = {"width": "300px", "display": "inline-block", "marginRight": "8px"}
_LB = {"color": "#aaa", "fontSize": "12px", "marginRight": "4px", "verticalAlign": "middle"}
_ZSCALE_OPTS = [{"label": f"Z×{v}", "value": v} for v in [1, 2, 3, 4, 5]]


def _zscale_dd(id: str) -> list:
    return [
        html.Span("Z scale:", style=_LB),
        dcc.Dropdown(
            id=id, options=_ZSCALE_OPTS, value=1, clearable=False,
            style={"width": "110px", "display": "inline-block", "marginRight": "8px"},
        ),
    ]


def build_app(geometry_root: Path, vggt_root: Path, eval_root: Path | None = None) -> Dash:
    scenes = discover_scenes(geometry_root)
    opts = [{"label": _label(s), "value": s} for s in scenes]

    eval_clouds = discover_eval_clouds(eval_root) if eval_root is not None else {}

    app = Dash(__name__)

    # Fixed compare graph slots (always in DOM; hidden when unused)
    compare_slots = [
        html.Div(
            id=f"compare-slot-{i}",
            style={"display": "none"},
            children=[
                dcc.Graph(id=f"compare-graph-{i}", figure=_empty(), style={"height": "76vh"}),
            ],
        )
        for i in range(MAX_COMPARE)
    ]

    app.layout = html.Div(
        style={"backgroundColor": "#141414", "minHeight": "100vh", "padding": "10px"},
        children=[
            dcc.Store(id="camera-store"),
            dcc.Store(id="eval-camera-store"),
            html.H2(
                "temporal-VGGT · point cloud viewer",
                style={"color": "#ccc", "fontFamily": "monospace", "fontSize": "15px",
                       "margin": "0 0 10px 0"},
            ),
            dcc.Tabs(
                value="single",
                colors={"border": "#333", "primary": "#4a90d9", "background": "#1c1c1c"},
                children=[
                    # ── Single scene ──────────────────────────────────
                    dcc.Tab(label="Single scene", value="single", children=[
                        html.Div(style={"padding": "8px"}, children=[
                            html.Div([
                                html.Span("Scene:", style=_LB),
                                dcc.Dropdown(
                                    id="single-scene", options=opts,
                                    value=scenes[0] if scenes else None,
                                    style={**_DD, "width": "360px"},
                                ),
                                html.Span("Color:", style=_LB),
                                dcc.Dropdown(
                                    id="single-color",
                                    options=[
                                        {"label": "Height (viridis)", "value": "height"},
                                        {"label": "RGB from source images", "value": "rgb"},
                                    ],
                                    value="height", clearable=False, style=_DD,
                                ),
                                *_zscale_dd("single-zscale"),
                            ], style={"marginBottom": "8px"}),
                            dcc.Loading(dcc.Graph(
                                id="single-graph",
                                figure=_empty("Select a scene above."),
                                style={"height": "80vh"},
                            )),
                        ]),
                    ]),

                    # ── Compare ───────────────────────────────────────
                    dcc.Tab(label="Compare", value="compare", children=[
                        html.Div(style={"padding": "8px"}, children=[
                            html.Div([
                                html.Span("Scenes (up to 4):", style=_LB),
                                dcc.Dropdown(
                                    id="compare-scenes", options=opts,
                                    value=scenes[:2] if len(scenes) >= 2 else scenes,
                                    multi=True,
                                    style={"width": "700px", "display": "inline-block",
                                           "marginRight": "8px"},
                                ),
                                html.Span("Color:", style=_LB),
                                dcc.Dropdown(
                                    id="compare-color",
                                    options=[
                                        {"label": "Height (viridis)", "value": "height"},
                                        {"label": "RGB from source images", "value": "rgb"},
                                    ],
                                    value="height", clearable=False, style=_DD,
                                ),
                                *_zscale_dd("compare-zscale"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(compare_slots),
                        ]),
                    ]),

                    # ── Overlay ───────────────────────────────────────
                    dcc.Tab(label="Overlay (red / blue)", value="overlay", children=[
                        html.Div(style={"padding": "8px"}, children=[
                            html.Div([
                                html.Span("Scene A (red):", style=_LB),
                                dcc.Dropdown(
                                    id="overlay-a", options=opts,
                                    value=scenes[0] if scenes else None,
                                    style=_DD,
                                ),
                                html.Span("Scene B (blue):", style=_LB),
                                dcc.Dropdown(
                                    id="overlay-b", options=opts,
                                    value=scenes[1] if len(scenes) >= 2 else None,
                                    style=_DD,
                                ),
                                *_zscale_dd("overlay-zscale"),
                            ], style={"marginBottom": "8px"}),
                            dcc.Loading(dcc.Graph(
                                id="overlay-graph",
                                figure=_empty("Select two scenes above."),
                                style={"height": "80vh"},
                            )),
                        ]),
                    ]),

                    # ── Model vs GT ───────────────────────────────────
                    dcc.Tab(label="Model vs GT", value="eval", children=[
                        html.Div(style={"padding": "8px"}, children=[
                            html.Div([
                                html.Span("Fold:", style=_LB),
                                dcc.Dropdown(
                                    id="eval-fold",
                                    options=[{"label": k, "value": k} for k in eval_clouds],
                                    value=next(iter(eval_clouds), None),
                                    style={"width": "300px", "display": "inline-block",
                                           "marginRight": "8px"},
                                ),
                                html.Span("Sample:", style=_LB),
                                dcc.Dropdown(
                                    id="eval-sample",
                                    options=[],
                                    value=None,
                                    style={"width": "620px", "display": "inline-block",
                                           "marginRight": "8px"},
                                ),
                                *_zscale_dd("eval-zscale"),
                            ], style={"marginBottom": "8px"}),
                            html.Div([
                                html.Div([
                                    html.Div("Model prediction",
                                             style={"color": "#7ec8e3", "fontFamily": "monospace",
                                                    "fontSize": "11px", "marginBottom": "2px"}),
                                    dcc.Graph(id="eval-pred-graph",
                                              figure=_empty("Select a fold and sample."),
                                              style={"height": "76vh"}),
                                ], style={"width": "50%", "display": "inline-block",
                                          "verticalAlign": "top"}),
                                html.Div([
                                    html.Div("Teacher t2 reference",
                                             style={"color": "#f4a261", "fontFamily": "monospace",
                                                    "fontSize": "11px", "marginBottom": "2px"}),
                                    dcc.Graph(id="eval-ref-graph",
                                              figure=_empty("Select a fold and sample."),
                                              style={"height": "76vh"}),
                                ], style={"width": "50%", "display": "inline-block",
                                          "verticalAlign": "top"}),
                            ]),
                        ]),
                    ]),
                ],
            ),
        ],
    )

    # ── callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output("single-graph", "figure"),
        [Input("single-scene", "value"), Input("single-color", "value"),
         Input("single-zscale", "value")],
    )
    def update_single(scene_id: str | None, color_mode: str, z_scale: int) -> go.Figure:
        if not scene_id:
            return _empty("Select a scene.")
        zs = float(z_scale or 1)
        label = _label(scene_id)
        try:
            if color_mode == "rgb":
                pts, rgb = load_rgb(scene_id, vggt_root, geometry_root, z_scale=zs)
                trace = _trace_rgb(pts, rgb, label)
            else:
                pts, z_col = load_geo(scene_id, geometry_root, z_scale=zs)
                trace = _trace_height(pts, z_col, label)
        except FileNotFoundError as exc:
            return _empty(f"File not found: {exc}")
        except Exception as exc:
            return _empty(f"Error loading scene: {exc}")
        return go.Figure(data=[trace], layout=go.Layout(**_layout(label)))

    @app.callback(
        [Output(f"compare-graph-{i}", "figure") for i in range(MAX_COMPARE)]
        + [Output(f"compare-slot-{i}", "style") for i in range(MAX_COMPARE)],
        [Input("compare-scenes", "value"), Input("compare-color", "value"),
         Input("compare-zscale", "value")],
    )
    def update_compare(selected: list[str] | None, color_mode: str, z_scale: int) -> list:
        n = min(len(selected or []), MAX_COMPARE)
        zs = float(z_scale or 1)
        col_w = "50%" if n > 1 else "100%"
        figs, styles = [], []
        for i in range(MAX_COMPARE):
            if i < n:
                sid = selected[i]
                lbl = _label(sid)
                try:
                    if color_mode == "rgb":
                        pts, rgb = load_rgb(sid, vggt_root, geometry_root, z_scale=zs)
                        trace = _trace_rgb(pts, rgb, lbl)
                    else:
                        pts, z_col = load_geo(sid, geometry_root, z_scale=zs)
                        trace = _trace_height(pts, z_col, lbl)
                    fig = go.Figure(
                        data=[trace],
                        layout=go.Layout(**_layout(lbl, uirevision="compare")),
                    )
                except Exception as exc:
                    fig = _empty(str(exc))
                figs.append(fig)
                styles.append({"width": col_w, "display": "inline-block", "verticalAlign": "top"})
            else:
                figs.append(_empty())
                styles.append({"display": "none"})
        return figs + styles

    # Camera capture: any compare graph user interaction → store
    @app.callback(
        Output("camera-store", "data"),
        [Input(f"compare-graph-{i}", "relayoutData") for i in range(MAX_COMPARE)],
        State("camera-store", "data"),
        prevent_initial_call=True,
    )
    def capture_camera(*args: object) -> object:
        relayouts, current = args[:MAX_COMPARE], args[MAX_COMPARE]
        if not ctx.triggered:
            return no_update
        val = ctx.triggered[0].get("value")
        if not val or "scene.camera" not in val:
            return no_update
        cam = val["scene.camera"]
        if current and current.get("cam") == cam:
            return no_update
        return {"cam": cam, "ts": time.time()}

    # Camera sync: store change → patch all compare graphs
    @app.callback(
        [Output(f"compare-graph-{i}", "figure", allow_duplicate=True) for i in range(MAX_COMPARE)],
        Input("camera-store", "data"),
        prevent_initial_call=True,
    )
    def sync_cameras(store: dict | None) -> list:
        if not store or "cam" not in store:
            return [no_update] * MAX_COMPARE
        cam = store["cam"]
        patches = []
        for _ in range(MAX_COMPARE):
            p = Patch()
            p["layout"]["scene"]["camera"] = cam
            patches.append(p)
        return patches

    @app.callback(
        Output("overlay-graph", "figure"),
        [Input("overlay-a", "value"), Input("overlay-b", "value"),
         Input("overlay-zscale", "value")],
    )
    def update_overlay(sid_a: str | None, sid_b: str | None, z_scale: int) -> go.Figure:
        zs = float(z_scale or 1)
        traces = []
        for sid, color in ((sid_a, "red"), (sid_b, "dodgerblue")):
            if not sid:
                continue
            try:
                pts, _ = load_geo(sid, geometry_root, z_scale=zs)
                traces.append(_trace_flat(pts, color, _label(sid)))
            except Exception:
                pass
        if not traces:
            return _empty("Select two scenes.")
        title = f"{_label(sid_a) if sid_a else '?'}  vs  {_label(sid_b) if sid_b else '?'}"
        return go.Figure(data=traces, layout=go.Layout(**_layout(title)))

    # ── Model vs GT callbacks ─────────────────────────────────────────────────

    @app.callback(
        [Output("eval-sample", "options"), Output("eval-sample", "value")],
        Input("eval-fold", "value"),
    )
    def update_eval_samples(fold_id: str | None) -> tuple:
        if not fold_id or fold_id not in eval_clouds:
            return [], None
        samples = eval_clouds[fold_id]["samples"]
        opts = [{"label": _parse_sample_label(s), "value": s} for s in samples]
        return opts, (samples[0] if samples else None)

    @app.callback(
        [Output("eval-pred-graph", "figure"), Output("eval-ref-graph", "figure")],
        [Input("eval-fold", "value"), Input("eval-sample", "value"),
         Input("eval-zscale", "value")],
    )
    def update_eval_graphs(fold_id: str | None, sample_key: str | None,
                           z_scale: int) -> tuple:
        if not fold_id or not sample_key or fold_id not in eval_clouds:
            empty = _empty("Select a fold and sample.")
            return empty, empty
        clouds_dir = eval_clouds[fold_id]["clouds_dir"]
        zs = float(z_scale or 1)
        try:
            pred_pts = np.load(clouds_dir / f"{sample_key}_pred.npy").astype(np.float32)
            ref_pts  = np.load(clouds_dir / f"{sample_key}_ref.npy").astype(np.float32)
        except FileNotFoundError as exc:
            empty = _empty(f"Cloud file not found: {exc}")
            return empty, empty
        except Exception as exc:
            empty = _empty(f"Error loading clouds: {exc}")
            return empty, empty

        # Eval clouds are in NeRFStudio GPS world space (Z = height) after
        # Umeyama alignment in eval_model.py. Normalize for display using ref
        # as anchor (same formula as build_geometry_assets.py).
        cx = float(ref_pts[:, 0].mean())
        cy = float(ref_pts[:, 1].mean())
        ground_z = float(np.percentile(ref_pts[:, 2], 2))
        xy_radii = np.sqrt((ref_pts[:, 0] - cx) ** 2 + (ref_pts[:, 1] - cy) ** 2)
        norm_scale = max(float(np.percentile(xy_radii, 95)), 1e-6)

        def _norm_display(pts: np.ndarray) -> np.ndarray:
            p = pts.copy()
            p[:, 0] = (p[:, 0] - cx) / norm_scale
            p[:, 1] = (p[:, 1] - cy) / norm_scale
            p[:, 2] = (p[:, 2] - ground_z) / norm_scale
            return p

        pred_pts = _norm_display(pred_pts)
        ref_pts  = _norm_display(ref_pts)

        def _height_fig(pts: np.ndarray, title: str, uirev: str) -> go.Figure:
            z = pts[:, 2]
            z_col = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)
            pts = pts.copy()
            pts[:, 2] *= zs
            trace = _trace_height(pts, z_col, title)
            return go.Figure(data=[trace], layout=go.Layout(**_layout(title, uirev)))

        label = _parse_sample_label(sample_key)
        pred_fig = _height_fig(pred_pts, f"Prediction · {label}", "eval-pred")
        ref_fig  = _height_fig(ref_pts,  f"Reference  · {label}", "eval-ref")
        return pred_fig, ref_fig

    @app.callback(
        Output("eval-camera-store", "data"),
        [Input("eval-pred-graph", "relayoutData"), Input("eval-ref-graph", "relayoutData")],
        State("eval-camera-store", "data"),
        prevent_initial_call=True,
    )
    def capture_eval_camera(*args: object) -> object:
        _, current = args[:2], args[2]
        if not ctx.triggered:
            return no_update
        val = ctx.triggered[0].get("value")
        if not val or "scene.camera" not in val:
            return no_update
        cam = val["scene.camera"]
        if current and current.get("cam") == cam:
            return no_update
        return {"cam": cam, "ts": time.time()}

    @app.callback(
        [Output("eval-pred-graph", "figure", allow_duplicate=True),
         Output("eval-ref-graph", "figure", allow_duplicate=True)],
        Input("eval-camera-store", "data"),
        prevent_initial_call=True,
    )
    def sync_eval_cameras(store: dict | None) -> tuple:
        if not store or "cam" not in store:
            return no_update, no_update
        cam = store["cam"]
        p1, p2 = Patch(), Patch()
        p1["layout"]["scene"]["camera"] = cam
        p2["layout"]["scene"]["camera"] = cam
        return p1, p2

    return app


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal-VGGT point cloud viewer")
    p.add_argument("--geometry-root", type=Path, default=Path("geometry_assets"),
                   help="Root of geometry_assets (point_cloud_clean.npz files)")
    p.add_argument("--vggt-root", type=Path, default=Path("vggt_output"),
                   help="Root of vggt_output (needed for RGB color mode)")
    p.add_argument("--eval-root", type=Path, default=None,
                   help="Root of eval output dir (enables Model vs GT tab)")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"geometry_root : {args.geometry_root}", flush=True)
    print(f"vggt_root     : {args.vggt_root}", flush=True)
    scenes = discover_scenes(args.geometry_root)
    print(f"scenes found  : {len(scenes)}", flush=True)
    if scenes:
        print(f"  {scenes[:4]}{'...' if len(scenes) > 4 else ''}", flush=True)
    app = build_app(args.geometry_root, args.vggt_root, eval_root=args.eval_root)
    print(f"starting server on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

