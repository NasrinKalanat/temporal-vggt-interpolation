"""TRPM GT confidence distribution viewer.

Shows GT (VGGT raw) confidence histogram and 3D point cloud
filtered by a threshold slider.

Usage:
    conda run -n 4d python src/trpm/visualize_conf.py \
        --eval-root evaluation/trpm_small_cam --port 8283
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dash import Dash, Input, Output, State, dcc, html, no_update
import plotly.graph_objects as go


# ── data helpers ──────────────────────────────────────────────────────────────

def discover(eval_root: Path) -> dict[str, dict]:
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


def _load_ref(clouds_dir: Path, triplet_id: str) -> np.ndarray:
    path_npz = clouds_dir / f"{triplet_id}_ref.npz"
    path_npy = clouds_dir / f"{triplet_id}_ref.npy"
    if path_npz.exists():
        d = np.load(path_npz)
        return np.concatenate([d[k].astype(np.float32) for k in sorted(d.files)], axis=0)
    return np.load(path_npy).astype(np.float32)


def _subsample(pts: np.ndarray, max_pts: int = 80_000) -> np.ndarray:
    if len(pts) <= max_pts:
        return pts
    idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
    return pts[idx]


def _label(triplet_id: str) -> str:
    parts = triplet_id.split("_")
    if len(parts) >= 4:
        fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        return f"{parts[3]} · {fmt(parts[0])} → {fmt(parts[1])} ← {fmt(parts[2])}"
    return triplet_id


# ── app ───────────────────────────────────────────────────────────────────────

_DD = {"display": "inline-block", "marginRight": "8px"}
_LB = {"color": "#aaa", "fontSize": "12px", "marginRight": "4px", "verticalAlign": "middle"}
_INPUT_STYLE = {
    "width": "140px", "marginRight": "8px",
    "backgroundColor": "#2a2a2a", "color": "white", "border": "1px solid #555",
}


def build_app(eval_root: Path, conf_threshold: float = 1.0) -> Dash:
    folds = discover(eval_root)
    if not folds:
        raise SystemExit(f"No saved clouds found under {eval_root}")

    first_fold   = next(iter(folds))
    first_sample = folds[first_fold]["samples"][0]

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#141414", "minHeight": "100vh", "padding": "10px"},
        children=[
            html.H2("TRPM · GT Confidence Distribution",
                    style={"color": "#ccc", "fontFamily": "monospace",
                           "fontSize": "15px", "margin": "0 0 10px 0"}),
            html.Div([
                html.Span("Fold:", style=_LB),
                dcc.Dropdown(
                    id="fold-dd",
                    options=[{"label": k, "value": k} for k in folds],
                    value=first_fold, clearable=False,
                    style={**_DD, "width": "320px"},
                ),
                html.Span("Triplet:", style=_LB),
                dcc.Dropdown(
                    id="sample-dd",
                    options=[{"label": _label(s), "value": s} for s in folds[first_fold]["samples"]],
                    value=first_sample, clearable=False,
                    style={**_DD, "width": "480px"},
                ),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Span("GT conf threshold (VGGT raw):", style=_LB),
                dcc.Input(id="thr-ref", type="text",
                          value=str(conf_threshold), debounce=True, style=_INPUT_STYLE),
            ], style={"marginBottom": "12px"}),
            html.Div([
                html.Span("Remove N lowest conf points:", style=_LB),
                dcc.Input(id="n-remove", type="number", value=0, min=0, step=1,
                          style=_INPUT_STYLE),
                html.Button("Compute & Apply", id="n-btn", n_clicks=0,
                            style={"backgroundColor": "#2a2a2a", "color": "#ccc",
                                   "border": "1px solid #555", "cursor": "pointer",
                                   "padding": "2px 10px"}),
                html.Span(id="n-result", style={"color": "#aaa", "fontSize": "12px",
                                               "marginLeft": "12px"}),
            ], style={"marginBottom": "12px"}),
            html.Div("GT confidence (VGGT)", style={"color": "#f4a261", "fontFamily": "monospace", "fontSize": "11px"}),
            dcc.Graph(id="hist-ref", style={"height": "35vh"}),
            html.Div("GT (t2)", style={"color": "#f4a261", "fontFamily": "monospace",
                                       "fontSize": "11px", "marginTop": "8px"}),
            dcc.Graph(id="cloud-ref", style={"height": "55vh"}),
        ],
    )

    @app.callback(
        [Output("thr-ref", "value"), Output("n-result", "children")],
        Input("n-btn", "n_clicks"),
        [State("fold-dd", "value"), State("sample-dd", "value"), State("n-remove", "value")],
        prevent_initial_call=True,
    )
    def compute_n_threshold(_, fold_id, triplet_id, n_remove):
        if not fold_id or not triplet_id or n_remove is None:
            return no_update, ""
        n = int(n_remove)
        clouds_dir = folds[fold_id]["clouds_dir"]
        ref_conf = _load_ref(clouds_dir, triplet_id)[:, 3]
        total = len(ref_conf)
        if n >= total:
            return no_update, f"N={n} exceeds total points ({total})"
        threshold = float(np.sort(ref_conf)[n])
        return repr(threshold), f"threshold={repr(threshold)}  (removes {n}/{total} points)"

    @app.callback(
        [Output("sample-dd", "options"), Output("sample-dd", "value")],
        Input("fold-dd", "value"),
    )
    def update_samples(fold_id: str):
        samples = folds[fold_id]["samples"]
        return [{"label": _label(s), "value": s} for s in samples], samples[0]

    @app.callback(
        [Output("hist-ref", "figure"), Output("cloud-ref", "figure")],
        [Input("fold-dd", "value"), Input("sample-dd", "value"), Input("thr-ref", "value")],
    )
    def update(fold_id: str, triplet_id: str, thr_ref):
        try:
            thr_ref = float(thr_ref or 1.0)
        except (ValueError, TypeError):
            thr_ref = conf_threshold
        clouds_dir = folds[fold_id]["clouds_dir"]
        ref_all  = _load_ref(clouds_dir, triplet_id)
        ref_conf = ref_all[:, 3]

        hist_mask = (ref_conf >= 1.0) & (ref_conf <= 1.001)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=ref_conf[hist_mask], nbinsx=100000, name="VGGT conf",
            marker_color="#f4a261", opacity=0.75,
            hovertemplate="%{x:.6f}<br>count=%{y}<extra></extra>",
        ))
        hist_fig.add_vline(x=thr_ref, line_color="red", line_dash="dash", line_width=1.5)
        hist_fig.update_layout(
            paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
            font=dict(color="white"), margin=dict(l=40, r=10, t=10, b=30),
            xaxis=dict(color="white", range=[1.0, 1.001]), yaxis=dict(color="white"),
            showlegend=False,
        )

        ref_filt = _subsample(ref_all[ref_conf >= thr_ref])
        if len(ref_filt):
            z = ref_filt[:, 2]
            z_col = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)
            trace = go.Scatter3d(
                x=ref_filt[:, 0], y=ref_filt[:, 1], z=ref_filt[:, 2], mode="markers",
                marker=dict(size=1, color=z_col, colorscale="Viridis", showscale=False, opacity=0.8),
                name="GT t2", hoverinfo="none",
            )
        else:
            trace = go.Scatter3d(x=[], y=[], z=[], mode="markers", name="GT t2")

        cloud_fig = go.Figure(data=[trace])
        cloud_fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                bgcolor="#0d0d0d",
            ),
            paper_bgcolor="#1c1c1c", font=dict(color="white"),
            margin=dict(l=0, r=0, t=25, b=0),
        )
        return hist_fig, cloud_fig

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=Path("evaluation/trpm_small"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--port", type=int, default=8053)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    conf_threshold = 1.0
    if args.config and args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text()) or {}
        conf_threshold = cfg.get("conf_threshold", conf_threshold)
    app = build_app(args.eval_root, conf_threshold)
    print(f"http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()

