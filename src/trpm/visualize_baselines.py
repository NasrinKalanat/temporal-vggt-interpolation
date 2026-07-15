"""Baseline vs GT cloud viewer.

Side-by-side height-colored point clouds: selected baseline prediction vs GT t2.
Cameras are synchronized between the two panels.

Usage:
    conda run -n 4d python src/trpm/visualize_baselines.py \
        --eval-root evaluation/trpm_small_cam \
        --vggt-root vggt_outputs/t1t2_paired_v16_o8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from dash import Dash, Input, Output, State, Patch, ctx, dcc, html, no_update
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trpm.evaluate import (
    BASELINES, _list_variants, _load_variant_clouds, apply_baseline,
    _load_date_cloud,
)


# ── data discovery ────────────────────────────────────────────────────────────

def discover(eval_root: Path) -> dict[str, dict]:
    """Return {fold_id: {fold_dir, samples: [triplet_id, ...]}}."""
    folds: dict[str, dict] = {}
    for result_file in sorted(eval_root.rglob("eval_result.json")):
        fold_dir = result_file.parent
        fold_id  = fold_dir.name
        import json
        data = json.loads(result_file.read_text())
        samples = [r["triplet_id"] for r in data.get("triplet_rows", [])]
        if samples:
            folds[fold_id] = {"fold_dir": fold_dir, "samples": samples}
    return folds


def _label(triplet_id: str) -> str:
    parts = triplet_id.split("_")
    if len(parts) >= 4:
        fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        return f"{parts[3]} · {fmt(parts[0])} → {fmt(parts[1])} ← {fmt(parts[2])}"
    return triplet_id


def _tau_for(eval_root: Path, fold_id: str, triplet_id: str) -> float:
    """Look up tau from the saved eval_result.json."""
    import json
    for result_file in eval_root.rglob(f"{fold_id}/eval_result.json"):
        data = json.loads(result_file.read_text())
        for row in data.get("triplet_rows", []):
            if row["triplet_id"] == triplet_id:
                return float(row["tau"])
    return 0.5


# ── cloud helpers ─────────────────────────────────────────────────────────────

def _merge_views(views: dict, selected: list[str], threshold: float, max_pts: int = 80_000) -> np.ndarray:
    parts = []
    for k in selected:
        if k not in views:
            continue
        pts = views[k]
        if pts.shape[1] == 4:
            pts = pts[pts[:, 3] >= threshold]
        parts.append(pts[:, :3])
    if not parts:
        return np.zeros((0, 3), np.float32)
    merged = np.concatenate(parts, axis=0)
    if len(merged) > max_pts:
        rng = np.random.default_rng(0)
        merged = merged[rng.choice(len(merged), max_pts, replace=False)]
    return merged


def _height_trace(pts: np.ndarray, name: str) -> go.Scatter3d:
    z = pts[:, 2]
    z_col = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1, color=z_col, colorscale="Viridis",
                    showscale=False, opacity=0.8),
        name=name, hoverinfo="none",
    )


def _layout(title: str, uirev: str) -> dict:
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
        uirevision=uirev,
        legend=dict(font=dict(size=10), x=0, y=1),
    )


def _empty(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_layout("", "empty"))
    if msg:
        fig.add_annotation(text=msg, showarrow=False, font=dict(color="#555", size=13))
    return fig


# ── app ───────────────────────────────────────────────────────────────────────

_DD = {"display": "inline-block", "marginRight": "8px"}
_LB = {"color": "#aaa", "fontSize": "12px", "marginRight": "4px", "verticalAlign": "middle"}


def build_app(
    eval_root: Path,
    vggt_root: Path,
    conf_threshold: float = 1.0,
    n_points: int = 50_000,
    seed: int = 42,
) -> Dash:
    folds = discover(eval_root)
    if not folds:
        raise SystemExit(f"No eval_result.json found under {eval_root}")

    first_fold   = next(iter(folds))
    first_sample = folds[first_fold]["samples"][0]
    first_baseline = BASELINES[0]

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#141414", "minHeight": "100vh", "padding": "10px"},
        children=[
            dcc.Store(id="cam-store"),
            html.H2("TRPM · Baseline vs GT",
                    style={"color": "#ccc", "fontFamily": "monospace",
                           "fontSize": "15px", "margin": "0 0 10px 0"}),
            html.Div([
                html.Span("Fold:", style=_LB),
                dcc.Dropdown(
                    id="fold-dd",
                    options=[{"label": k, "value": k} for k in folds],
                    value=first_fold,
                    style={**_DD, "width": "320px"},
                    clearable=False,
                ),
                html.Span("Triplet:", style=_LB),
                dcc.Dropdown(
                    id="sample-dd",
                    options=[{"label": _label(s), "value": s}
                             for s in folds[first_fold]["samples"]],
                    value=first_sample,
                    style={**_DD, "width": "480px"},
                    clearable=False,
                ),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Span("Baseline:", style=_LB),
                dcc.Dropdown(
                    id="baseline-dd",
                    options=[{"label": b, "value": b} for b in BASELINES],
                    value=first_baseline,
                    style={**_DD, "width": "380px"},
                    clearable=False,
                ),
                html.Span("Variant:", style=_LB),
                dcc.Dropdown(
                    id="variant-dd",
                    style={**_DD, "width": "200px"},
                    clearable=False,
                ),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Span("GT conf:", style=_LB),
                dcc.Input(
                    id="conf-ref", type="number",
                    value=conf_threshold, min=1.0, max=10.0, step=0.000000001,
                    style={"width": "100px", "marginRight": "8px",
                           "backgroundColor": "#2a2a2a", "color": "white",
                           "border": "1px solid #555"},
                ),
                html.Button("Apply", id="apply-btn", n_clicks=0,
                            style={"marginRight": "16px", "backgroundColor": "#2a2a2a",
                                   "color": "#ccc", "border": "1px solid #555",
                                   "cursor": "pointer", "padding": "2px 10px"}),
                dcc.Dropdown(
                    id="zscale-dd",
                    options=[{"label": f"Z×{v}", "value": v} for v in [1, 2, 3, 5]],
                    value=1, clearable=False,
                    style={**_DD, "width": "100px"},
                ),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Div([
                    html.Div("Baseline",
                             style={"color": "#7ec8e3", "fontFamily": "monospace",
                                    "fontSize": "11px", "marginBottom": "2px"}),
                    dcc.Graph(id="baseline-graph", figure=_empty("Loading..."),
                              style={"height": "80vh"}),
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    html.Div("GT (t2)",
                             style={"color": "#f4a261", "fontFamily": "monospace",
                                    "fontSize": "11px", "marginBottom": "2px"}),
                    dcc.Graph(id="ref-graph", figure=_empty("Loading..."),
                              style={"height": "80vh"}),
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
            ]),
        ],
    )

    @app.callback(
        [Output("sample-dd", "options"), Output("sample-dd", "value")],
        Input("fold-dd", "value"),
    )
    def update_samples(fold_id: str):
        samples = folds[fold_id]["samples"]
        return [{"label": _label(s), "value": s} for s in samples], samples[0]

    @app.callback(
        [Output("variant-dd", "options"), Output("variant-dd", "value")],
        [Input("fold-dd", "value"), Input("sample-dd", "value")],
    )
    def update_variants(fold_id: str, triplet_id: str):
        if not fold_id or not triplet_id:
            return [], None
        variants = _list_variants(vggt_root, triplet_id)
        if not variants:
            return [], None
        opts = [{"label": v, "value": v} for v in variants]
        return opts, variants[0]

    @app.callback(
        [Output("baseline-graph", "figure"), Output("ref-graph", "figure")],
        [Input("fold-dd", "value"), Input("sample-dd", "value"),
         Input("baseline-dd", "value"), Input("variant-dd", "value"),
         Input("zscale-dd", "value"), Input("apply-btn", "n_clicks")],
        State("conf-ref", "value"),
    )
    def update_graphs(fold_id, triplet_id, baseline, variant, z_scale, _btn, conf_ref):
        if not all([fold_id, triplet_id, baseline, variant]):
            e = _empty("Select all options.")
            return e, e

        zs       = float(z_scale or 1)
        conf_thr = float(conf_ref or 1.0)

        tau = _tau_for(eval_root, fold_id, triplet_id)

        try:
            v_clouds = _load_variant_clouds(vggt_root, triplet_id, variant,
                                            conf_thr, n_points, seed)
            if v_clouds is None:
                e = _empty(f"Could not load clouds for {triplet_id}/{variant}")
                return e, e
            pts_t1, pts_t2, pts_t3 = v_clouds

            baseline_pts = apply_baseline(baseline, pts_t1, pts_t2, pts_t3, tau, n_points, seed)
        except Exception as exc:
            e = _empty(str(exc))
            return e, e

        b_xyz = baseline_pts[:, :3].copy()
        r_xyz = pts_t2[:, :3].copy()
        b_xyz[:, 2] *= zs
        r_xyz[:, 2] *= zs

        lbl = _label(triplet_id)
        b_fig = go.Figure(
            data=[_height_trace(b_xyz, baseline)],
            layout=go.Layout(**_layout(f"{baseline} · {lbl} · {variant}", "bl-shared")),
        )
        r_fig = go.Figure(
            data=[_height_trace(r_xyz, "GT t2")],
            layout=go.Layout(**_layout(f"GT t2 · {lbl} · {variant}", "bl-shared")),
        )
        return b_fig, r_fig

    @app.callback(
        Output("cam-store", "data"),
        [Input("baseline-graph", "relayoutData"), Input("ref-graph", "relayoutData")],
        State("cam-store", "data"),
        prevent_initial_call=True,
    )
    def capture_cam(*args):
        current = args[2]
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
        [Output("baseline-graph", "figure", allow_duplicate=True),
         Output("ref-graph",      "figure", allow_duplicate=True)],
        Input("cam-store", "data"),
        prevent_initial_call=True,
    )
    def sync_cam(store):
        if not store or "cam" not in store:
            return no_update, no_update
        cam = store["cam"]
        p1, p2 = Patch(), Patch()
        p1["layout"]["scene"]["camera"] = cam
        p2["layout"]["scene"]["camera"] = cam
        return p1, p2

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root",  type=Path, default=Path("evaluation/trpm_small_cam"))
    p.add_argument("--vggt-root",  type=Path, default=None)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--port",       type=int,  default=8053)
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--debug",      action="store_true")
    args = p.parse_args()

    conf_threshold = 1.0
    vggt_root      = args.vggt_root
    n_points       = 50_000
    seed           = 42

    if args.config and args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text()) or {}
        conf_threshold = cfg.get("conf_threshold", conf_threshold)
        n_points       = cfg.get("n_points", n_points)
        seed           = cfg.get("seed", seed)
        if vggt_root is None:
            vggt_root = Path(cfg["vggt_output_root"]) if "vggt_output_root" in cfg else None

    if vggt_root is None:
        p.error("--vggt-root is required (or set vggt_output_root in --config)")

    app = build_app(args.eval_root, vggt_root, conf_threshold, n_points, seed)
    print(f"http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()

