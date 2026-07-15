"""TRPM prediction vs GT cloud viewer.

Side-by-side height-colored point clouds for each saved triplet.
Cameras are synchronized between the two panels.

Usage:
    conda run -n 4d python src/trpm/visualize_clouds.py \
        --eval-root evaluation/trpm_small
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from dash import Dash, Input, Output, State, Patch, ctx, dcc, html, no_update
import plotly.graph_objects as go


# ── data discovery ────────────────────────────────────────────────────────────

def discover(eval_root: Path) -> dict[str, dict]:
    """Return {fold_id: {clouds_dir, samples: [triplet_id, ...]}}."""
    folds: dict[str, dict] = {}
    for clouds_dir in sorted(eval_root.rglob("clouds")):
        if not clouds_dir.is_dir():
            continue
        fold_id = clouds_dir.parent.name
        samples = sorted(
            f.stem.removesuffix("_pred")
            for f in sorted(clouds_dir.glob("*_pred.npy")) + sorted(clouds_dir.glob("*_pred.npz"))
        )
        if samples:
            folds[fold_id] = {"clouds_dir": clouds_dir, "samples": samples}
    return folds


def _label(triplet_id: str) -> str:
    parts = triplet_id.split("_")
    if len(parts) >= 4:
        fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        return f"{parts[3]} · {fmt(parts[0])} → {fmt(parts[1])} ← {fmt(parts[2])}"
    return triplet_id


# ── cloud helpers ─────────────────────────────────────────────────────────────

def _load(clouds_dir: Path, triplet_id: str) -> tuple[dict, dict]:
    """Load per-view npz dicts. Falls back to legacy .npy as single view."""
    def _load_one(path_npz: Path, path_npy: Path) -> dict:
        if path_npz.exists():
            d = np.load(path_npz)
            return {k: d[k].astype(np.float32) for k in sorted(d.files)}
        arr = np.load(path_npy).astype(np.float32)
        return {"v00": arr}
    pred = _load_one(clouds_dir / f"{triplet_id}_pred.npz",
                     clouds_dir / f"{triplet_id}_pred.npy")
    ref  = _load_one(clouds_dir / f"{triplet_id}_ref.npz",
                     clouds_dir / f"{triplet_id}_ref.npy")
    return pred, ref


def _merge_views(views: dict, selected: list[str], threshold: float, max_pts: int = 80_000) -> np.ndarray:
    """Merge selected view arrays, apply conf filter, return xyz [N,3]."""
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


def _normalize(pts: np.ndarray, cx: float, cy: float, gz: float, scale: float) -> np.ndarray:
    p = pts.copy()
    p[:, 0] = (p[:, 0] - cx) / scale
    p[:, 1] = (p[:, 1] - cy) / scale
    p[:, 2] = (p[:, 2] - gz) / scale
    return p


def _height_trace(pts: np.ndarray, name: str) -> go.Scattergl:
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

_DD  = {"display": "inline-block", "marginRight": "8px"}
_LB  = {"color": "#aaa", "fontSize": "12px", "marginRight": "4px", "verticalAlign": "middle"}


def build_app(eval_root: Path, conf_threshold: float = 1.0, pred_conf_threshold: float = 0.0) -> Dash:
    folds = discover(eval_root)
    if not folds:
        raise SystemExit(f"No saved clouds found under {eval_root}")

    first_fold = next(iter(folds))
    first_sample = folds[first_fold]["samples"][0]

    def _initial_figures():
        try:
            clouds_dir = folds[first_fold]["clouds_dir"]
            pred_views, ref_views = _load(clouds_dir, first_sample)
            pred = _merge_views(pred_views, list(pred_views.keys()), 0.0)
            ref  = _merge_views(ref_views,  list(ref_views.keys()),  0.0)
            lbl  = _label(first_sample)
            return (
                go.Figure(data=[_height_trace(pred, "prediction")],
                          layout=go.Layout(**_layout(f"Prediction · {lbl}", "trpm-shared"))),
                go.Figure(data=[_height_trace(ref, "GT t2")],
                          layout=go.Layout(**_layout(f"GT t2 · {lbl}", "trpm-shared"))),
            )
        except Exception:
            return _empty("Loading..."), _empty("Loading...")

    init_pred_fig, init_ref_fig = _initial_figures()

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#141414", "minHeight": "100vh", "padding": "10px"},
        children=[
            dcc.Store(id="cam-store"),
            html.H2("TRPM · Prediction vs GT",
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
                html.Span("Pred views:", style=_LB),
                dcc.Dropdown(id="pred-views-dd", multi=True,
                             style={**_DD, "width": "340px"}),
                html.Span("GT views:", style=_LB),
                dcc.Dropdown(id="ref-views-dd", multi=True,
                             style={**_DD, "width": "340px"}),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Span("Pred conf:", style=_LB),
                dcc.Input(
                    id="conf-pred", type="number",
                    value=pred_conf_threshold, min=0.0, max=1.0, step=0.01,
                    style={"width": "70px", "marginRight": "8px",
                           "backgroundColor": "#2a2a2a", "color": "white",
                           "border": "1px solid #555"},
                ),
                html.Span("GT conf:", style=_LB),
                dcc.Input(
                    id="conf-ref", type="number",
                    value=conf_threshold, min=1.0, max=10.0, step=0.000000001,
                    style={"width": "70px", "marginRight": "8px",
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
                    html.Div("Prediction",
                             style={"color": "#7ec8e3", "fontFamily": "monospace",
                                    "fontSize": "11px", "marginBottom": "2px"}),
                    dcc.Graph(id="pred-graph", figure=init_pred_fig,
                              style={"height": "80vh"}),
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    html.Div("GT (t2 teacher)",
                             style={"color": "#f4a261", "fontFamily": "monospace",
                                    "fontSize": "11px", "marginBottom": "2px"}),
                    dcc.Graph(id="ref-graph", figure=init_ref_fig,
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
        opts = [{"label": _label(s), "value": s} for s in samples]
        return opts, samples[0]

    @app.callback(
        [Output("pred-views-dd", "options"), Output("pred-views-dd", "value"),
         Output("ref-views-dd",  "options"), Output("ref-views-dd",  "value")],
        [Input("fold-dd", "value"), Input("sample-dd", "value")],
    )
    def update_view_options(fold_id: str, triplet_id: str):
        if not fold_id or not triplet_id:
            return [], [], [], []
        clouds_dir = folds[fold_id]["clouds_dir"]
        try:
            pred_views, ref_views = _load(clouds_dir, triplet_id)
        except Exception:
            return [], [], [], []
        pred_opts = [{"label": k, "value": k} for k in pred_views]
        ref_opts  = [{"label": k, "value": k} for k in ref_views]
        return pred_opts, list(pred_views.keys()), ref_opts, list(ref_views.keys())

    @app.callback(
        [Output("pred-graph", "figure"), Output("ref-graph", "figure")],
        [Input("fold-dd", "value"), Input("sample-dd", "value"),
         Input("zscale-dd", "value"), Input("apply-btn", "n_clicks"),
         Input("pred-views-dd", "value"), Input("ref-views-dd", "value")],
        [State("conf-pred", "value"), State("conf-ref", "value")],
    )
    def update_graphs(fold_id: str, triplet_id: str, z_scale: int, _btn,
                      pred_sel: list[str], ref_sel: list[str],
                      conf_pred: float, conf_ref: float):
        if not fold_id or not triplet_id:
            e = _empty("Select a fold and triplet.")
            return e, e
        zs        = float(z_scale or 1)
        conf_pred = float(conf_pred or 0.0)
        conf_ref  = float(conf_ref  or 1.0)
        clouds_dir = folds[fold_id]["clouds_dir"]
        try:
            pred_views, ref_views = _load(clouds_dir, triplet_id)
        except Exception as exc:
            e = _empty(str(exc))
            return e, e

        pred = _merge_views(pred_views, pred_sel or list(pred_views.keys()), conf_pred)
        ref  = _merge_views(ref_views,  ref_sel  or list(ref_views.keys()),  conf_ref)

        pred_n = pred.copy()
        ref_n  = ref.copy()
        pred_n[:, 2] *= zs
        ref_n[:, 2]  *= zs

        lbl = _label(triplet_id)
        pred_fig = go.Figure(
            data=[_height_trace(pred_n, "prediction")],
            layout=go.Layout(**_layout(f"Prediction · {lbl}", "trpm-shared")),
        )
        ref_fig = go.Figure(
            data=[_height_trace(ref_n, "GT t2")],
            layout=go.Layout(**_layout(f"GT t2 · {lbl}", "trpm-shared")),
        )
        return pred_fig, ref_fig

    # Sync cameras between the two panels.
    @app.callback(
        Output("cam-store", "data"),
        [Input("pred-graph", "relayoutData"), Input("ref-graph", "relayoutData")],
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
        [Output("pred-graph", "figure", allow_duplicate=True),
         Output("ref-graph",  "figure", allow_duplicate=True)],
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
    p.add_argument("--eval-root", type=Path, default=Path("evaluation/trpm_small"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--port", type=int, default=8052)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    conf_threshold = 1.0
    pred_conf_threshold = 0.0
    if args.config and args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text()) or {}
        conf_threshold = cfg.get("conf_threshold", conf_threshold)
        pred_conf_threshold = cfg.get("pred_conf_threshold", pred_conf_threshold)
    app = build_app(args.eval_root, conf_threshold, pred_conf_threshold)
    print(f"http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()

