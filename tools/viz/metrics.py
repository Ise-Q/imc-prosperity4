"""Per-product market metrics derived from the activities dataframe.

Formulas for position, edge stats, and missed-opp analysis mirror
`round1/notebooks/analyze_results.ipynb`. VWAP and wall-mid are new.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITION_LIMITS: dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}
DEFAULT_POSITION_LIMIT = 80

LEVELS = (1, 2, 3)


def position_limit(product: str) -> int:
    return POSITION_LIMITS.get(product, DEFAULT_POSITION_LIMIT)


def enrich_activities(activities: pd.DataFrame) -> pd.DataFrame:
    """Return activities with derived per-row metrics attached.

    Adds: `best_bid`, `best_ask`, `mid`, `spread`, `ob_vwap`, `wall_mid`,
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

    df["ob_vwap"] = _compute_ob_vwap(df)
    df["wall_mid"] = _compute_wall_mid(df)

    return df


def _best_prices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    bid_cols = [f"bid_price_{i}" for i in LEVELS]
    ask_cols = [f"ask_price_{i}" for i in LEVELS]
    best_bid = df[bid_cols].bfill(axis=1).iloc[:, 0]
    best_ask = df[ask_cols].bfill(axis=1).iloc[:, 0]
    return best_bid, best_ask


def empty_ob_timestamps(activities_for_product: pd.DataFrame) -> np.ndarray:
    """Return sorted `global_ts` values where the orderbook is fully empty."""
    if activities_for_product.empty or "ob_empty" not in activities_for_product.columns:
        return np.array([], dtype=int)
    mask = activities_for_product["ob_empty"].fillna(False).astype(bool)
    return np.sort(activities_for_product.loc[mask, "global_ts"].to_numpy())


def _compute_ob_vwap(df: pd.DataFrame) -> pd.Series:
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for i in LEVELS:
        bp = df[f"bid_price_{i}"].fillna(0)
        bv = df[f"bid_volume_{i}"].fillna(0)
        ap = df[f"ask_price_{i}"].fillna(0)
        av = df[f"ask_volume_{i}"].fillna(0)
        num = num + bp * bv + ap * av
        den = den + bv + av
    return num.where(den > 0) / den.where(den > 0)


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
    """Cumulative signed position over time, reset at each day boundary."""
    if trades_for_product.empty:
        return pd.DataFrame({"global_ts": [0], "position": [0], "day": [0]})

    t = trades_for_product.sort_values(["day", "global_ts"]).copy()
    t["position"] = t.groupby("day")["signed_qty"].cumsum()

    pieces = []
    for day, grp in t.groupby("day", sort=True):
        baseline_ts = grp["global_ts"].iloc[0] - 1
        pieces.append(pd.DataFrame({"global_ts": [baseline_ts], "position": [0], "day": [day]}))
        pieces.append(grp[["global_ts", "position", "day"]])
    return pd.concat(pieces, ignore_index=True)


def align_position_to_activities(
    activities_for_product: pd.DataFrame,
    trades_for_product: pd.DataFrame,
) -> pd.Series:
    """Return position at each activities row; resets per day."""
    if activities_for_product.empty:
        return pd.Series(dtype=float)

    if trades_for_product.empty:
        return pd.Series(0, index=activities_for_product.index)

    t = trades_for_product.sort_values(["day", "global_ts"]).copy()
    t["position"] = t.groupby("day")["signed_qty"].cumsum()

    out = pd.Series(0, index=activities_for_product.index, dtype=int)
    for day, acts_day in activities_for_product.groupby("day", sort=False):
        trades_day = t[t["day"] == day]
        if trades_day.empty:
            continue
        lookup = trades_day[["global_ts", "position"]].drop_duplicates("global_ts", keep="last")
        merged = pd.merge_asof(
            acts_day[["global_ts"]].sort_values("global_ts"),
            lookup.sort_values("global_ts"),
            on="global_ts",
            direction="backward",
        )
        merged["position"] = merged["position"].fillna(0).astype(int)
        merged.index = acts_day.sort_values("global_ts").index
        out.loc[acts_day.index] = merged["position"].reindex(acts_day.index).to_numpy()
    return out


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
