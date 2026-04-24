"""Dash app: interactive visualization of backtest and platform logs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from tools.viz.metrics import (
    align_position_to_activities,
    empty_ob_timestamps,
    enrich_activities,
    position_limit,
    position_timeline,
)
from tools.viz.parser import LoadedLog

BID_COLOR = "34,139,34"       # forestgreen
ASK_COLOR = "220,20,60"        # crimson
DEPTH_STYLES = {
    1: {"alpha": 1.0, "width": 1.6},
    2: {"alpha": 0.55, "width": 1.2},
    3: {"alpha": 0.30, "width": 1.0},
}

MID_COLOR = "#111111"
OB_VWAP_COLOR = "#1f77b4"
WALL_MID_COLOR = "#ff7f0e"
SPREAD_COLOR = "#1f77b4"
EMPTY_OB_COLOR = "rgba(150,150,150,0.35)"

OWN_BUY_COLOR = "#0b7a0b"
OWN_SELL_COLOR = "#b00020"
MARKET_COLOR = "#6c6c6c"


@dataclass
class Precomputed:
    activities_by_product: dict[str, pd.DataFrame]
    trades_by_product: dict[str, pd.DataFrame]
    position_by_product: dict[str, pd.Series]
    position_timeline_by_product: dict[str, pd.DataFrame]
    products: list[str]
    days: list[int]
    run_id: str
    source: str
    meta: dict


def precompute(log: LoadedLog) -> Precomputed:
    enriched = enrich_activities(log.activities)
    activities_by_product: dict[str, pd.DataFrame] = {}
    trades_by_product: dict[str, pd.DataFrame] = {}
    position_by_product: dict[str, pd.Series] = {}
    position_timeline_by_product: dict[str, pd.DataFrame] = {}

    for product in log.products:
        p = enriched[enriched["product"] == product].sort_values("global_ts").reset_index(drop=True)
        t = log.trades[log.trades["symbol"] == product].sort_values("global_ts").reset_index(drop=True)
        activities_by_product[product] = p
        trades_by_product[product] = t
        position_by_product[product] = align_position_to_activities(p, t)
        position_timeline_by_product[product] = position_timeline(t)

    return Precomputed(
        activities_by_product=activities_by_product,
        trades_by_product=trades_by_product,
        position_by_product=position_by_product,
        position_timeline_by_product=position_timeline_by_product,
        products=log.products,
        days=log.days,
        run_id=log.run_id,
        source=log.source,
        meta=log.meta,
    )


def build_app(log: LoadedLog) -> Dash:
    pre = precompute(log)
    app = Dash(__name__, title=f"viz — {pre.run_id}")

    header = _header_text(pre)
    app.layout = html.Div(
        style={"fontFamily": "system-ui, sans-serif", "padding": "12px"},
        children=[
            html.Div(header, style={"fontWeight": "bold", "marginBottom": "8px"}),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "alignItems": "flex-start"},
                children=[
                    _control_group(
                        "Layout",
                        dcc.RadioItems(
                            id="layout-mode",
                            options=[
                                {"label": " single", "value": "single"},
                                {"label": " stacked", "value": "stacked"},
                            ],
                            value="single",
                            inline=True,
                        ),
                    ),
                    _control_group(
                        "Product (single mode)",
                        dcc.Dropdown(
                            id="product-dropdown",
                            options=[{"label": p, "value": p} for p in pre.products],
                            value=pre.products[0] if pre.products else None,
                            clearable=False,
                            style={"minWidth": "260px"},
                        ),
                    ),
                    _control_group(
                        "Days",
                        dcc.Checklist(
                            id="day-filter",
                            options=[{"label": f" {d} ", "value": d} for d in pre.days],
                            value=list(pre.days),
                            inline=True,
                        ),
                    ),
                    _control_group(
                        "Overlays",
                        dcc.Checklist(
                            id="overlay-toggles",
                            options=[
                                {"label": " mid", "value": "mid"},
                                {"label": " ob_vwap", "value": "ob_vwap"},
                                {"label": " wall_mid", "value": "wall_mid"},
                                {"label": " empty_ob", "value": "empty_ob"},
                            ],
                            value=["mid"],
                            inline=True,
                        ),
                    ),
                    _control_group(
                        "Trade layers",
                        dcc.Checklist(
                            id="trade-toggles",
                            options=[
                                {"label": " own", "value": "own"},
                                {"label": " market", "value": "market"},
                            ],
                            value=["own"],
                            inline=True,
                        ),
                    ),
                    _control_group(
                        "Depth levels",
                        dcc.Checklist(
                            id="depth-toggles",
                            options=[
                                {"label": " L1", "value": 1},
                                {"label": " L2", "value": 2},
                                {"label": " L3", "value": 3},
                            ],
                            value=[1, 2, 3],
                            inline=True,
                        ),
                    ),
                ],
            ),
            dcc.Graph(
                id="main-graph",
                style={"height": "92vh"},
                config={"displaylogo": False, "scrollZoom": True},
            ),
        ],
    )

    @app.callback(
        Output("main-graph", "figure"),
        Input("layout-mode", "value"),
        Input("product-dropdown", "value"),
        Input("day-filter", "value"),
        Input("overlay-toggles", "value"),
        Input("trade-toggles", "value"),
        Input("depth-toggles", "value"),
    )
    def _update(layout_mode, product, days, overlays, trade_layers, depth_levels):
        days_sel = set(days or [])
        overlays_sel = set(overlays or [])
        trade_sel = set(trade_layers or [])
        depth_sel = set(depth_levels or [])

        if layout_mode == "stacked":
            products_to_render = pre.products
        else:
            products_to_render = [product] if product else []

        return build_figure(
            pre,
            products_to_render,
            days_sel,
            overlays_sel,
            trade_sel,
            depth_sel,
        )

    @app.callback(
        Output("main-graph", "figure", allow_duplicate=True),
        Input("main-graph", "relayoutData"),
        State("layout-mode", "value"),
        State("product-dropdown", "value"),
        State("day-filter", "value"),
        State("overlay-toggles", "value"),
        State("trade-toggles", "value"),
        State("depth-toggles", "value"),
        prevent_initial_call=True,
    )
    def _autoscale_y(relayout, layout_mode, product, days, overlays, trade_layers, depth_levels):
        kind = _classify_relayout(relayout or {})
        if kind is None:
            raise PreventUpdate

        if layout_mode == "stacked":
            products_to_render = pre.products
        else:
            products_to_render = [product] if product else []
        if not products_to_render:
            raise PreventUpdate

        patch = Patch()

        if kind == "reset":
            for idx in range(len(products_to_render)):
                base = 4 * idx
                for r in range(1, 5):
                    patch["layout"][_yaxis_key(base + r)]["autorange"] = True
            return patch

        xr = _extract_x_range(relayout)
        if xr is None:
            raise PreventUpdate
        x_lo, x_hi = xr

        days_sel = set(days or [])
        overlays_sel = set(overlays or [])
        trade_sel = set(trade_layers or [])
        depth_sel = set(depth_levels or [])

        wrote_anything = False
        for idx, p in enumerate(products_to_render):
            base = 4 * idx

            acts = pre.activities_by_product.get(p, pd.DataFrame())
            trades = pre.trades_by_product.get(p, pd.DataFrame())
            pos_tl = pre.position_timeline_by_product.get(p, pd.DataFrame())

            if not acts.empty:
                acts_f = acts[acts["day"].isin(days_sel)] if days_sel else acts.iloc[0:0]
            else:
                acts_f = acts
            if not trades.empty:
                trades_f = trades[trades["day"].isin(days_sel)] if days_sel else trades.iloc[0:0]
            else:
                trades_f = trades

            if not acts_f.empty:
                acts_f = acts_f[(acts_f["global_ts"] >= x_lo) & (acts_f["global_ts"] <= x_hi)]
            if not trades_f.empty:
                trades_f = trades_f[(trades_f["global_ts"] >= x_lo) & (trades_f["global_ts"] <= x_hi)]
            if not pos_tl.empty:
                pos_tl_f = pos_tl[(pos_tl["global_ts"] >= x_lo) & (pos_tl["global_ts"] <= x_hi)]
            else:
                pos_tl_f = pos_tl

            panels = (
                (1, _price_y_range(acts_f, trades_f, overlays_sel, depth_sel, trade_sel)),
                (2, _spread_y_range(acts_f)),
                (3, _position_y_range(pos_tl_f)),
                (4, _pnl_y_range(acts_f)),
            )
            for row, rng in panels:
                if rng is None:
                    continue
                lo, hi = _pad_range(*rng)
                key = _yaxis_key(base + row)
                patch["layout"][key]["range"] = [lo, hi]
                patch["layout"][key]["autorange"] = False
                wrote_anything = True

        if not wrote_anything:
            raise PreventUpdate

        return patch

    return app


def _control_group(label: str, component) -> html.Div:
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
        children=[
            html.Label(label, style={"fontSize": "11px", "color": "#555", "textTransform": "uppercase"}),
            component,
        ],
    )


def _header_text(pre: Precomputed) -> str:
    bits = [f"run={pre.run_id}", f"source={pre.source}"]
    if pre.source == "platform":
        if pre.meta.get("profit") is not None:
            bits.append(f"profit={pre.meta['profit']:.2f}")
        if pre.meta.get("status"):
            bits.append(f"status={pre.meta['status']}")
    bits.append(f"products={','.join(pre.products)}")
    bits.append(f"days={pre.days}")
    return "  |  ".join(bits)


def build_figure(
    pre: Precomputed,
    products: list[str],
    days: set[int],
    overlays: set[str],
    trade_layers: set[str],
    depth_levels: set[int],
) -> go.Figure:
    if not products:
        return go.Figure()

    rows_per_product = 4
    total_rows = rows_per_product * len(products)
    row_heights = []
    for _ in products:
        row_heights.extend([0.50, 0.15, 0.20, 0.15])
    row_heights = [h / sum(row_heights) for h in row_heights]

    subplot_titles = []
    for p in products:
        subplot_titles.extend([
            f"{p} — price & trades",
            f"{p} — spread",
            f"{p} — position",
            f"{p} — PnL",
        ])

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    render_empty_ob = "empty_ob" in overlays

    for idx, product in enumerate(products):
        price_row = rows_per_product * idx + 1
        spread_row = price_row + 1
        pos_row = price_row + 2
        pnl_row = price_row + 3

        acts = pre.activities_by_product.get(product, pd.DataFrame())
        trades = pre.trades_by_product.get(product, pd.DataFrame())
        pos_aligned = pre.position_by_product.get(product, pd.Series(dtype=float))
        pos_tl = pre.position_timeline_by_product.get(product, pd.DataFrame())

        if not acts.empty:
            acts_f = acts[acts["day"].isin(days)] if days else acts.iloc[0:0]
        else:
            acts_f = acts

        if not trades.empty:
            trades_f = trades[trades["day"].isin(days)] if days else trades.iloc[0:0]
        else:
            trades_f = trades

        if not pos_tl.empty and not acts_f.empty:
            lo, hi = acts_f["global_ts"].min(), acts_f["global_ts"].max()
            pos_tl_f = pos_tl[(pos_tl["global_ts"] >= lo - 1) & (pos_tl["global_ts"] <= hi)]
        else:
            pos_tl_f = pos_tl

        _add_depth_lines(fig, acts_f, price_row, depth_levels)
        _add_overlays(fig, acts_f, price_row, overlays)
        _add_trade_markers(fig, trades_f, price_row, trade_layers, pos_aligned, acts_f)
        _add_spread(fig, acts_f, spread_row)
        _add_position_panel(fig, pos_tl_f, price_row=pos_row, product=product)
        _add_pnl_panel(fig, acts_f, pnl_row)

        if render_empty_ob:
            _add_empty_ob_markers(
                fig, acts_f, first_row=price_row, rows=rows_per_product,
            )

        fig.update_yaxes(title_text="price", row=price_row, col=1)
        fig.update_yaxes(title_text="spread", row=spread_row, col=1)
        fig.update_yaxes(title_text="position", row=pos_row, col=1)
        fig.update_yaxes(title_text="PnL", row=pnl_row, col=1)

    fig.update_xaxes(title_text="global_ts", row=total_rows, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=total_rows, col=1)

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode="x unified",
        template="plotly_white",
    )

    _add_day_dividers(fig, pre, days, total_rows)

    return fig


def _add_depth_lines(fig: go.Figure, acts: pd.DataFrame, row: int, levels: set[int]) -> None:
    if acts.empty:
        return

    x = acts["global_ts"]

    for level in sorted(levels):
        style = DEPTH_STYLES[level]
        alpha = style["alpha"]
        width = style["width"]
        bid_price = acts[f"bid_price_{level}"]
        bid_vol = acts[f"bid_volume_{level}"]
        ask_price = acts[f"ask_price_{level}"]
        ask_vol = acts[f"ask_volume_{level}"]

        _add_depth_side(
            fig, x, bid_price, bid_vol, row,
            name=f"bid L{level}", level=level, side="bid",
            color=f"rgba({BID_COLOR},{alpha})", width=width,
            legendgroup=f"bid_L{level}",
        )
        _add_depth_side(
            fig, x, ask_price, ask_vol, row,
            name=f"ask L{level}", level=level, side="ask",
            color=f"rgba({ASK_COLOR},{alpha})", width=width,
            legendgroup=f"ask_L{level}",
        )


def _add_depth_side(
    fig: go.Figure,
    x: pd.Series,
    price: pd.Series,
    qty: pd.Series,
    row: int,
    *,
    name: str,
    level: int,
    side: str,
    color: str,
    width: float,
    legendgroup: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x, y=price,
            name=name,
            mode="lines",
            line=dict(color=color, width=width),
            connectgaps=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        ),
        row=row, col=1,
    )

    hover_y = price.ffill().bfill()
    if hover_y.isna().all():
        return

    customdata = _price_qty_customdata(price, qty)
    fig.add_trace(
        go.Scatter(
            x=x, y=hover_y,
            name=name,
            mode="markers",
            marker=dict(size=1, opacity=0, color=color),
            customdata=customdata,
            legendgroup=legendgroup,
            showlegend=False,
            hovertemplate=(
                f"L{level} {side} px: %{{customdata[0]}} / qty: %{{customdata[1]}}"
                "<extra></extra>"
            ),
        ),
        row=row, col=1,
    )


def _price_qty_customdata(price: pd.Series, qty: pd.Series) -> np.ndarray:
    px_str = np.where(price.isna(), "N/A", price.map(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"))
    qty_str = np.where(qty.isna(), "N/A", qty.map(lambda v: f"{int(v)}" if pd.notna(v) else "N/A"))
    return np.column_stack([px_str, qty_str])


def _add_spread(fig: go.Figure, acts: pd.DataFrame, row: int) -> None:
    if acts.empty or "spread" not in acts.columns:
        return
    fig.add_trace(
        go.Scatter(
            x=acts["global_ts"],
            y=acts["spread"],
            name="spread",
            mode="lines",
            line=dict(color=SPREAD_COLOR, width=1.0),
            connectgaps=False,
            showlegend=False,
            hovertemplate="spread=%{y:.2f}<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.add_hline(y=0, line=dict(color="#999", width=0.6, dash="dot"), row=row, col=1)


def _add_empty_ob_markers(
    fig: go.Figure, acts: pd.DataFrame, first_row: int, rows: int,
) -> None:
    """Draw grey dotted vertical markers at ticks where the orderbook is
    fully empty on both sides. Scoped to the subplot rows of a single
    product so markers don't bleed across products in stacked mode.
    """
    if acts.empty:
        return
    ts_arr = empty_ob_timestamps(acts)
    if ts_arr.size == 0:
        return
    for ts in ts_arr:
        for offset in range(rows):
            fig.add_vline(
                x=int(ts),
                line=dict(color=EMPTY_OB_COLOR, width=0.6, dash="dot"),
                row=first_row + offset, col=1,
            )


def _add_overlays(fig: go.Figure, acts: pd.DataFrame, row: int, overlays: set[str]) -> None:
    if acts.empty:
        return

    x = acts["global_ts"]
    specs = [
        ("mid", acts["mid"], MID_COLOR, "solid", 1.6),
        ("ob_vwap", acts.get("ob_vwap"), OB_VWAP_COLOR, "solid", 1.2),
        ("wall_mid", acts.get("wall_mid"), WALL_MID_COLOR, "dash", 1.2),
    ]

    for name, y, color, dash, width in specs:
        if name not in overlays or y is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                name=name,
                mode="lines",
                line=dict(color=color, dash=dash, width=width),
                connectgaps=False,
                legendgroup=name,
                hovertemplate=f"{name}: " "%{y:.2f}<extra></extra>",
            ),
            row=row, col=1,
        )


def _add_trade_markers(
    fig: go.Figure,
    trades: pd.DataFrame,
    row: int,
    layers: set[str],
    pos_aligned: pd.Series,
    acts: pd.DataFrame,
) -> None:
    if trades.empty:
        return

    if "own" in layers:
        own = trades[trades["is_own"]]
        buys = own[own["side"] == "BUY"]
        sells = own[own["side"] == "SELL"]
        _scatter_trades(fig, buys, row, "own BUY", "triangle-up", OWN_BUY_COLOR)
        _scatter_trades(fig, sells, row, "own SELL", "triangle-down", OWN_SELL_COLOR)

    if "market" in layers:
        mkt = trades[~trades["is_own"]]
        _scatter_trades(fig, mkt, row, "market", "x", MARKET_COLOR)


def _scatter_trades(
    fig: go.Figure,
    trades: pd.DataFrame,
    row: int,
    name: str,
    symbol: str,
    color: str,
) -> None:
    if trades.empty:
        return
    sizes = np.clip(8 + 3 * np.log1p(trades["quantity"].to_numpy()), 8, 24)
    hover = (
        "ts=%{x}<br>"
        "px=%{y}<br>"
        "qty=%{customdata[0]}<br>"
        "side=%{customdata[1]}<extra></extra>"
    )
    customdata = np.stack(
        [trades["quantity"].to_numpy(), trades["side"].to_numpy()],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=trades["global_ts"],
            y=trades["price"],
            mode="markers",
            name=name,
            marker=dict(
                symbol=symbol,
                color=color,
                size=sizes,
                line=dict(color="white", width=0.5),
            ),
            customdata=customdata,
            hovertemplate=hover,
            legendgroup=name,
        ),
        row=row, col=1,
    )


def _add_position_panel(fig: go.Figure, pos_tl: pd.DataFrame, price_row: int, product: str) -> None:
    if pos_tl.empty:
        return
    limit = position_limit(product)

    fig.add_trace(
        go.Scatter(
            x=pos_tl["global_ts"],
            y=pos_tl["position"],
            name=f"position {product}",
            mode="lines",
            line=dict(color="#2b5cbf", width=1.5, shape="hv"),
            showlegend=False,
            hovertemplate="pos=%{y}<extra></extra>",
        ),
        row=price_row, col=1,
    )
    for sign in (1, -1):
        fig.add_hline(
            y=sign * limit,
            line=dict(color="#b00020", width=1, dash="dash"),
            row=price_row, col=1,
        )
    fig.add_hline(y=0, line=dict(color="#999", width=0.6, dash="dot"), row=price_row, col=1)


def _add_pnl_panel(fig: go.Figure, acts: pd.DataFrame, row: int) -> None:
    if acts.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=acts["global_ts"],
            y=acts["profit_and_loss"],
            name="PnL",
            mode="lines",
            line=dict(color="#1f7a1f", width=1.4),
            showlegend=False,
            hovertemplate="PnL=%{y:.2f}<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.add_hline(y=0, line=dict(color="#999", width=0.6, dash="dot"), row=row, col=1)


def _add_day_dividers(fig: go.Figure, pre: Precomputed, days: set[int], total_rows: int) -> None:
    selected = sorted(d for d in pre.days if d in days)
    if len(selected) < 2:
        return
    boundaries: list[int] = []
    for product_acts in pre.activities_by_product.values():
        if product_acts.empty:
            continue
        grouped = product_acts.groupby("day")["global_ts"].min().sort_index()
        for d in selected[1:]:
            if d in grouped.index:
                boundaries.append(int(grouped.loc[d]))
    for x in sorted(set(boundaries)):
        fig.add_vline(
            x=x,
            line=dict(color="#bbbbbb", width=0.8, dash="dot"),
            row="all", col=1,
        )


def _yaxis_key(n: int) -> str:
    return "yaxis" if n == 1 else f"yaxis{n}"


def _classify_relayout(relayout: dict) -> str | None:
    """Return 'zoom' / 'reset' / None.

    - 'zoom': x-axis range changed; no explicit user y-range set in the same event.
    - 'reset': x-axis autorange was toggled on (e.g. double-click).
    - None: event is irrelevant (hover spikes, y-only echoes of our own patch, etc.).
    """
    if not relayout:
        return None

    has_x_range = False
    has_x_reset = False
    has_y_range = False

    for k, v in relayout.items():
        if k.startswith("xaxis"):
            if k.endswith(".autorange") and v:
                has_x_reset = True
            elif ".range" in k:
                has_x_range = True
        elif k.startswith("yaxis"):
            if ".range" in k and not k.endswith(".autorange"):
                has_y_range = True

    if has_y_range:
        return None
    if has_x_reset:
        return "reset"
    if has_x_range:
        return "zoom"
    return None


def _extract_x_range(relayout: dict) -> tuple[float, float] | None:
    """Pull (lo, hi) from relayoutData regardless of which key form Plotly used."""
    if not relayout:
        return None

    for k, v in relayout.items():
        if k.startswith("xaxis") and k.endswith(".range") and isinstance(v, list) and len(v) == 2:
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                return None

    lo = hi = None
    for k, v in relayout.items():
        if not k.startswith("xaxis"):
            continue
        if k.endswith(".range[0]"):
            lo = v
        elif k.endswith(".range[1]"):
            hi = v
    if lo is None or hi is None:
        return None
    try:
        return float(lo), float(hi)
    except (TypeError, ValueError):
        return None


def _pad_range(lo: float, hi: float, pad: float = 0.05) -> tuple[float, float]:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return lo, hi
    if hi == lo:
        delta = max(abs(lo) * 0.05, 0.5)
        return lo - delta, hi + delta
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _price_y_range(
    acts_f: pd.DataFrame,
    trades_f: pd.DataFrame,
    overlays: set[str],
    depth_levels: set[int],
    trade_layers: set[str],
) -> tuple[float, float] | None:
    pieces: list[pd.Series] = []
    if not acts_f.empty:
        for level in depth_levels:
            for side in ("bid", "ask"):
                col = f"{side}_price_{level}"
                if col in acts_f.columns:
                    pieces.append(acts_f[col])
        for name in ("mid", "ob_vwap", "wall_mid"):
            if name in overlays and name in acts_f.columns:
                pieces.append(acts_f[name])
    if not trades_f.empty and "price" in trades_f.columns:
        is_own = trades_f["is_own"].astype(bool) if "is_own" in trades_f.columns else pd.Series(False, index=trades_f.index)
        mask = pd.Series(False, index=trades_f.index)
        if "own" in trade_layers:
            mask = mask | is_own
        if "market" in trade_layers:
            mask = mask | ~is_own
        if mask.any():
            pieces.append(trades_f.loc[mask, "price"])

    if not pieces:
        return None
    combined = pd.concat(pieces, ignore_index=True).dropna()
    if combined.empty:
        return None
    return float(combined.min()), float(combined.max())


def _spread_y_range(acts_f: pd.DataFrame) -> tuple[float, float] | None:
    if acts_f.empty or "spread" not in acts_f.columns:
        return None
    s = acts_f["spread"].dropna()
    if s.empty:
        return None
    return float(s.min()), float(s.max())


def _position_y_range(pos_tl_f: pd.DataFrame) -> tuple[float, float] | None:
    if pos_tl_f.empty or "position" not in pos_tl_f.columns:
        return None
    s = pos_tl_f["position"].dropna()
    if s.empty:
        return None
    return float(s.min()), float(s.max())


def _pnl_y_range(acts_f: pd.DataFrame) -> tuple[float, float] | None:
    if acts_f.empty or "profit_and_loss" not in acts_f.columns:
        return None
    s = acts_f["profit_and_loss"].dropna()
    if s.empty:
        return None
    return float(s.min()), float(s.max())
