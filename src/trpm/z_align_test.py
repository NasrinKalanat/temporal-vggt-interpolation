"""Compare model pred metrics before and after z-aligning pred to GT ground level.

Side-by-side 3D viewer: left=after z-align, right=before. Each panel overlays
pred (blue) and GT (orange) with per-cloud visibility toggles.

Usage:
    conda run -n 4d python src/trpm/z_align_test.py \
        --config configs/train_trpm_small_cam.yaml \
        --checkpoint runs/trpm_small_cam/strict/strict_20230831_corn/best_model.pt \
        --triplet-id 20230812_20230831_20230906_corn \
        --variant variant_00_00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trpm.evaluate import (
    _list_variants, predict_trpm_variant, _load_variant_clouds,
    _z_align, build_config, choose_device, _load_model_class,
)
from losses.geometry import compute_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="path to best_model.pt")
    p.add_argument("--triplet-id", required=True)
    p.add_argument("--variant",    default=None, help="default: first variant")
    p.add_argument("--port",       type=int, default=8054)
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--no-viz",     action="store_true", help="print metrics only, no browser")
    args = p.parse_args()

    # reuse build_config by faking the namespace it expects
    class _FakeArgs:
        config        = args.config
        runs_root     = None
        output_root   = None
        device        = None
        protocol      = None
        crop          = None
        test_date     = None
        save_clouds   = False
        baselines_only = False
        baseline_cache = None
    cfg    = build_config(_FakeArgs())
    device = choose_device(cfg["device"])

    vggt_root = cfg["vggt_output_root"]
    triplet_id = args.triplet_id

    variant = args.variant or _list_variants(vggt_root, triplet_id)[0]
    print(f"triplet : {triplet_id}")
    print(f"variant : {variant}")

    # ── load checkpoint ───────────────────────────────────────────────────────
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    model_cls = _load_model_class(cfg.get("model_class", "trpm.model.TRPMSmall"))
    model = model_cls(**cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"checkpoint: {checkpoint}")

    # ── infer tau from triplet_id ─────────────────────────────────────────────
    from trpm.dataset import _parse_date
    parts = triplet_id.split("_")
    tau = (_parse_date(parts[1]) - _parse_date(parts[0])) / max(_parse_date(parts[2]) - _parse_date(parts[0]), 1)
    print(f"tau     : {tau:.4f}")

    # ── load GT t2 cloud ──────────────────────────────────────────────────────
    v_clouds = _load_variant_clouds(
        vggt_root, triplet_id, variant,
        cfg["conf_threshold"], cfg["n_points"], cfg["seed"],
    )
    if v_clouds is None:
        raise RuntimeError("Could not load variant clouds")
    _, pts_t2, _ = v_clouds

    # ── run model inference ───────────────────────────────────────────────────
    result = predict_trpm_variant(
        model, vggt_root, triplet_id, variant, tau, device,
        cfg["pred_conf_threshold"], cfg["n_points"], cfg["seed"],
    )
    if result is None:
        raise RuntimeError("predict_trpm_variant returned None")
    pred_cloud, _ = result

    # ── metrics before z-align ────────────────────────────────────────────────
    kw = dict(
        threshold  = cfg["distance_threshold"],
        voxel_size = cfg["voxel_size"],
        alpha      = cfg["eval_alpha"],
        beta       = cfg["eval_beta"],
    )
    m_before = compute_metrics(pred_cloud[:, :3], pts_t2[:, :3], **kw)

    # ── z-align pred to GT ground level ──────────────────────────────────────
    z_ref = np.percentile(pts_t2[:, 2], 1.0)
    pred_aligned = _z_align(pred_cloud, z_ref)
    m_after = compute_metrics(pred_aligned[:, :3], pts_t2[:, :3], **kw)

    # ── print results ─────────────────────────────────────────────────────────
    metric_keys = ["f1", "precision", "recall", "asymmetric_chamfer",
                   "voxel_iou", "normal_consistency", "height_median_error"]
    w = max(len(k) for k in metric_keys)
    print(f"{'metric':<{w}}   {'before':>10}   {'after':>10}   {'delta':>10}")
    print("-" * (w + 36))
    for k in metric_keys:
        b = m_before.get(k, float("nan"))
        a = m_after.get(k, float("nan"))
        print(f"{k:<{w}}   {b:>10.4f}   {a:>10.4f}   {a-b:>+10.4f}")

    z_shift = z_ref - np.percentile(pred_cloud[:, 2], 1.0)
    print(f"\nz-shift applied: {z_shift:+.4f} m")

    if args.no_viz:
        return

    # ── build viz ─────────────────────────────────────────────────────────────
    _launch_viz(
        pts_gt      = pts_t2[:, :3],
        pts_before  = pred_cloud[:, :3],
        pts_after   = pred_aligned[:, :3],
        m_before    = m_before,
        m_after     = m_after,
        z_shift     = z_shift,
        host        = args.host,
        port        = args.port,
    )


def _scene_layout(title: str) -> dict:
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
        margin=dict(l=0, r=0, t=28, b=0),
        uirevision="shared",
        legend=dict(font=dict(size=10), x=0, y=1),
    )


def _cloud_trace(pts: np.ndarray, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1, color=color, opacity=0.7),
        name=name, hoverinfo="none",
    )


def _metric_str(m: dict) -> str:
    keys = ["f1", "precision", "recall", "asymmetric_chamfer", "voxel_iou"]
    return "  ".join(f"{k}={m.get(k, float('nan')):.3f}" for k in keys)


def _launch_viz(
    pts_gt: np.ndarray,
    pts_before: np.ndarray,
    pts_after: np.ndarray,
    m_before: dict,
    m_after: dict,
    z_shift: float,
    host: str,
    port: int,
) -> None:
    _LB = {"color": "#aaa", "fontSize": "11px", "marginRight": "6px", "verticalAlign": "middle"}
    _DD = {"display": "inline-block", "marginRight": "10px"}

    visibility_options = [
        {"label": "Pred", "value": "pred"},
        {"label": "GT",   "value": "gt"},
    ]

    def _make_fig(pts_pred: np.ndarray, title: str) -> go.Figure:
        return go.Figure(
            data=[
                _cloud_trace(pts_pred, "Pred", "#5bc8ef"),
                _cloud_trace(pts_gt,   "GT",   "#f4a261"),
            ],
            layout=go.Layout(**_scene_layout(title)),
        )

    fig_before = _make_fig(pts_before, f"Before  |  {_metric_str(m_before)}")
    fig_after  = _make_fig(pts_after,  f"After z-shift={z_shift:+.3f}m  |  {_metric_str(m_after)}")

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#141414", "minHeight": "100vh", "padding": "10px"},
        children=[
            html.H2("Z-Align Test · Pred vs GT",
                    style={"color": "#ccc", "fontFamily": "monospace",
                           "fontSize": "15px", "margin": "0 0 8px 0"}),
            html.Div([
                html.Span("Show:", style=_LB),
                dcc.Checklist(
                    id="visibility",
                    options=visibility_options,
                    value=["pred", "gt"],
                    inline=True,
                    style={"color": "#ccc", "fontSize": "12px", "display": "inline-block"},
                    labelStyle={"marginRight": "12px"},
                ),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Div([
                    html.Div("Before z-align",
                             style={"color": "#f4a261", "fontFamily": "monospace",
                                    "fontSize": "11px", "marginBottom": "2px"}),
                    dcc.Graph(id="graph-before", figure=fig_before, style={"height": "82vh"}),
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    html.Div("After z-align",
                             style={"color": "#7ec8e3", "fontFamily": "monospace",
                                    "fontSize": "11px", "marginBottom": "2px"}),
                    dcc.Graph(id="graph-after", figure=fig_after, style={"height": "82vh"}),
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
            ]),
        ],
    )

    @app.callback(
        [Output("graph-before", "figure"), Output("graph-after", "figure")],
        Input("visibility", "value"),
    )
    def update_visibility(visible: list[str]):
        show_pred = "pred" in visible
        show_gt   = "gt"   in visible

        def _fig(pts_pred: np.ndarray, title: str) -> go.Figure:
            traces = []
            if show_pred:
                traces.append(_cloud_trace(pts_pred, "Pred", "#5bc8ef"))
            if show_gt:
                traces.append(_cloud_trace(pts_gt, "GT", "#f4a261"))
            return go.Figure(data=traces, layout=go.Layout(**_scene_layout(title)))

        return (
            _fig(pts_before, f"Before  |  {_metric_str(m_before)}"),
            _fig(pts_after,  f"After z-shift={z_shift:+.3f}m  |  {_metric_str(m_after)}"),
        )

    print(f"http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

