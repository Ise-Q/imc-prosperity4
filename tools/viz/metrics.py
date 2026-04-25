"""Per-product market metrics derived from the activities dataframe.

Formulas for position, edge stats, and missed-opp analysis mirror
`round1/notebooks/analyze_results.ipynb`. Microprice and wall-mid are new.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.viz.black_scholes import (
    BlackScholes,
    DEFAULT_IV,
    IV_EMA_ALPHA,
    TICKS_PER_DAY,
    TRADING_DAYS_PER_YEAR,
    TTE_DAYS_AT_ROUND_START,
    UNDERLYING,
    VOUCHER_STRIKE,
)

POSITION_LIMITS: dict[str, int] = {
    # Round 1 / 2
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    # Round 3 — d1
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    # Round 3 — options (VEV_* calls on VELVETFRUIT_EXTRACT)
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
    "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
    "VEV_5400": 300, "VEV_5500": 300, "VEV_6000": 300,
    "VEV_6500": 300,
}
DEFAULT_POSITION_LIMIT = 80

LEVELS = (1, 2, 3)


def position_limit(product: str) -> int:
    return POSITION_LIMITS.get(product, DEFAULT_POSITION_LIMIT)


def enrich_activities(activities: pd.DataFrame) -> pd.DataFrame:
    """Return activities with derived per-row metrics attached.

    Adds: `best_bid`, `best_ask`, `mid`, `spread`, `microprice`, `wall_mid`,
    `ob_empty`.

    Conventions (see memory `project_ob_empty_definition.md`):
    - `best_bid` / `best_ask`: first non-NaN price across L1→L3 per side.
    - `mid`: `(best_bid + best_ask) / 2` normally; if one side is fully
      empty, mid falls back to the other side's best; if both sides are
      fully empty, mid is NaN.
    - `spread`: `best_ask - best_bid` (NaN if either is NaN).
    - `ob_empty`: True only when both sides are fully empty across L1-L3.
    """
    if activities.empty:
        return activities

    df = activities.copy()

    best_bid, best_ask = _best_prices(df)
    df["best_bid"] = best_bid
    df["best_ask"] = best_ask

    df["mid"] = pd.concat([best_bid, best_ask], axis=1).mean(axis=1, skipna=True)
    df["spread"] = best_ask - best_bid
    df["ob_empty"] = best_bid.isna() & best_ask.isna()

    df["wall_mid"] = _compute_wall_mid(df)
    df["microprice"] = _compute_microprice(df)

    if "profit_and_loss" in df.columns:
        # The Rust backtester emits PnL using mid_price=0.0 on fully-empty-OB
        # rows, which yields large spurious swings. Blank those out and
        # forward-fill within (product, day) so the PnL line stays continuous
        # through empty-OB ticks.
        df.loc[df["ob_empty"], "profit_and_loss"] = np.nan
        df["profit_and_loss"] = (
            df.groupby(["product", "day"])["profit_and_loss"]
              .transform(lambda s: s.ffill())
        )

    return df


def _best_prices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    bid_cols = [f"bid_price_{i}" for i in LEVELS]
    ask_cols = [f"ask_price_{i}" for i in LEVELS]
    best_bid = df[bid_cols].bfill(axis=1).iloc[:, 0]
    best_ask = df[ask_cols].bfill(axis=1).iloc[:, 0]
    return best_bid, best_ask


def _compute_microprice(df: pd.DataFrame) -> pd.Series:
    """L1 inside microprice = (best_bid * ask_vol_1 + best_ask * bid_vol_1) /
    (bid_vol_1 + ask_vol_1). Uses `best_bid`/`best_ask` (L1→L3 fallback) for
    consistency with `mid`/`spread`; volumes from L1 only.
    """
    bb = df["best_bid"]
    ba = df["best_ask"]
    bv = df["bid_volume_1"]
    av = df["ask_volume_1"]
    num = bb * av + ba * bv
    den = bv + av
    out = num.where(den > 0) / den.where(den > 0)
    return out.where(bb.notna() & ba.notna())


def _compute_wall_mid(df: pd.DataFrame) -> pd.Series:
    bid_prices = np.column_stack([df[f"bid_price_{i}"].to_numpy() for i in LEVELS])
    bid_vols = np.column_stack([df[f"bid_volume_{i}"].to_numpy() for i in LEVELS])
    ask_prices = np.column_stack([df[f"ask_price_{i}"].to_numpy() for i in LEVELS])
    ask_vols = np.column_stack([df[f"ask_volume_{i}"].to_numpy() for i in LEVELS])

    bid_vols_clean = np.where(np.isnan(bid_vols), -np.inf, bid_vols)
    ask_vols_clean = np.where(np.isnan(ask_vols), -np.inf, ask_vols)

    bid_argmax = np.argmax(bid_vols_clean, axis=1)
    ask_argmax = np.argmax(ask_vols_clean, axis=1)

    rows = np.arange(bid_prices.shape[0])
    bid_wall_px = bid_prices[rows, bid_argmax]
    bid_wall_vol = bid_vols[rows, bid_argmax]
    ask_wall_px = ask_prices[rows, ask_argmax]
    ask_wall_vol = ask_vols[rows, ask_argmax]

    num = bid_wall_px * ask_wall_vol + ask_wall_px * bid_wall_vol
    den = bid_wall_vol + ask_wall_vol
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return pd.Series(out, index=df.index)


def position_timeline(trades_for_product: pd.DataFrame) -> pd.DataFrame:
    """Cumulative signed position over time, continuous across days."""
    if trades_for_product.empty:
        return pd.DataFrame({"global_ts": [0], "position": [0], "day": [0]})

    t = trades_for_product.sort_values(["day", "global_ts"]).copy()
    t["position"] = t["signed_qty"].cumsum()

    first = t.iloc[0]
    baseline = pd.DataFrame({
        "global_ts": [int(first["global_ts"]) - 1],
        "position": [0],
        "day": [int(first["day"])],
    })
    return pd.concat(
        [baseline, t[["global_ts", "position", "day"]]],
        ignore_index=True,
    )


def align_position_to_activities(
    activities_for_product: pd.DataFrame,
    trades_for_product: pd.DataFrame,
) -> pd.Series:
    """Return position at each activities row; continuous across days."""
    if activities_for_product.empty:
        return pd.Series(dtype=float)

    if trades_for_product.empty:
        return pd.Series(0, index=activities_for_product.index)

    t = trades_for_product.sort_values(["day", "global_ts"]).copy()
    t["position"] = t["signed_qty"].cumsum()

    acts_sorted = activities_for_product.sort_values("global_ts")
    lookup = t[["global_ts", "position"]].drop_duplicates("global_ts", keep="last")
    merged = pd.merge_asof(
        acts_sorted[["global_ts"]],
        lookup.sort_values("global_ts"),
        on="global_ts",
        direction="backward",
    )
    merged["position"] = merged["position"].fillna(0).astype(int)
    merged.index = acts_sorted.index
    return merged["position"].reindex(activities_for_product.index)


def edge_stats(trades_for_product: pd.DataFrame, fair_value: float) -> dict:
    if trades_for_product.empty:
        return {}
    buys = trades_for_product[trades_for_product["side"] == "BUY"]
    sells = trades_for_product[trades_for_product["side"] == "SELL"]
    out = {}
    if len(buys):
        out["avg_buy_edge"] = float(fair_value - buys["price"].mean())
    if len(sells):
        out["avg_sell_edge"] = float(sells["price"].mean() - fair_value)
    return out


def missed_opps(
    activities_for_product: pd.DataFrame,
    position_series: pd.Series,
    fair_value: float,
    limit: int,
) -> pd.DataFrame:
    df = activities_for_product.copy()
    df["position"] = position_series
    df["missed_buy"] = (df["position"] >= limit) & (df["ask_price_1"] < fair_value)
    df["missed_sell"] = (df["position"] <= -limit) & (df["bid_price_1"] > fair_value)
    return df


def attach_bs_fair_value(
    activities_by_product: dict[str, pd.DataFrame],
    days: list[int],
) -> dict[str, pd.DataFrame]:
    """For each VEV_* call voucher in `activities_by_product`, attach a
    `bs_fair_value` column computed from BlackScholes.call_price using:

    - S = VELVETFRUIT_EXTRACT mid aligned by `global_ts`,
    - K = strike from VOUCHER_STRIKE,
    - T = TTE in years (TTE_DAYS_AT_ROUND_START - day_offset - ts/TICKS_PER_DAY,
      then / TRADING_DAYS_PER_YEAR; clamped to 1e-6),
    - sigma = live IV (bisection on option mid) → EMA → DEFAULT_IV fallback,
      mirroring `OptionTrader` in `round3/strats/trader2.py`.

    Mutates and returns the same dict (with replaced frames where applicable).
    Products without a strike mapping or without a usable underlying mid are
    left untouched.
    """
    underlying_acts = activities_by_product.get(UNDERLYING)
    if underlying_acts is None or underlying_acts.empty:
        return activities_by_product

    s_lookup = (
        underlying_acts[["global_ts", "mid"]]
        .dropna(subset=["mid"])
        .drop_duplicates("global_ts", keep="last")
        .set_index("global_ts")["mid"]
    )
    if s_lookup.empty:
        return activities_by_product

    day_zero = min(days) if days else 0

    for product, acts in list(activities_by_product.items()):
        K = VOUCHER_STRIKE.get(product)
        if K is None or acts.empty:
            continue

        df = acts.copy()
        S_series = df["global_ts"].map(s_lookup).to_numpy(dtype=float)
        day_offset = (df["day"].astype(int) - int(day_zero)).clip(lower=0).to_numpy()
        ts = df["timestamp"].astype(int).to_numpy()

        tte_days = TTE_DAYS_AT_ROUND_START - day_offset - ts / TICKS_PER_DAY
        tte_days = np.clip(tte_days, 1e-6, None)
        T_arr = tte_days / TRADING_DAYS_PER_YEAR
        C_arr = df["mid"].to_numpy(dtype=float)

        n = len(df)
        fair = np.full(n, np.nan)
        ema = None
        # Time order — within a single product, (day, timestamp) defines time.
        order = np.lexsort((ts, df["day"].to_numpy()))

        for i in order:
            S = S_series[i]
            T = T_arr[i]
            C = C_arr[i]
            if not (np.isfinite(S) and np.isfinite(T)) or S <= 0 or T <= 0:
                continue
            live = None
            if np.isfinite(C):
                live = BlackScholes.implied_vol(float(C), float(S), float(K), float(T))
            if live is not None:
                ema = live if ema is None else IV_EMA_ALPHA * live + (1 - IV_EMA_ALPHA) * ema
            iv_used = live if live is not None else (ema if ema is not None else DEFAULT_IV)
            fair[i] = BlackScholes.call_price(float(S), float(K), float(T), float(iv_used))

        df["bs_fair_value"] = fair
        activities_by_product[product] = df

    return activities_by_product
