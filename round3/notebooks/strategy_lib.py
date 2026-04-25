"""Helpers for round3/notebooks/strategy.ipynb.

Heavy logic lives here so the notebook stays a thin orchestrator. The public
surface used by the notebook:

  load_prices(round_dir, days)       -> long-form prices_df
  load_trades(round_dir, days)       -> long-form trades_df
  best_prices(df)                    -> (best_bid, best_ask) Series with L1->L3 fallback
  enrich(df)                         -> adds best_bid, best_ask, mid, spread, ob_empty
  pivot_wide(prices, col)            -> (timestamp_index, product) wide frame
  universe_stats(prices)             -> per-product sanity table
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"

LIQUID_VOUCHERS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
DEEP_ITM_VOUCHERS = ["VEV_4000", "VEV_4500"]
PINNED_VOUCHERS = ["VEV_6000", "VEV_6500"]
ALL_VOUCHERS = DEEP_ITM_VOUCHERS + LIQUID_VOUCHERS + PINNED_VOUCHERS
HYDROGEL = "HYDROGEL_PACK"
VEE = "VELVETFRUIT_EXTRACT"
ALL_PRODUCTS = [HYDROGEL, VEE] + ALL_VOUCHERS


def load_prices(round_dir: str | Path = "round3/data", days: Iterable[int] = (0, 1, 2)) -> pd.DataFrame:
    """Load + concatenate prices_round_3_day_{D}.csv for given days. Long-form, semicolon delim."""
    round_dir = Path(round_dir)
    frames = []
    for d in days:
        f = round_dir / f"prices_round_3_day_{d}.csv"
        df = pd.read_csv(f, sep=";")
        if "day" not in df.columns:
            df["day"] = d
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def load_trades(round_dir: str | Path = "round3/data", days: Iterable[int] = (0, 1, 2)) -> pd.DataFrame:
    round_dir = Path(round_dir)
    frames = []
    for d in days:
        f = round_dir / f"trades_round_3_day_{d}.csv"
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def best_prices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """L1->L3 fallback. Per MEMORY.md project_ob_empty_definition: 'best' bid/ask is
    first non-NaN across L1..L3, not hardcoded L1."""
    bid_cols = ["bid_price_1", "bid_price_2", "bid_price_3"]
    ask_cols = ["ask_price_1", "ask_price_2", "ask_price_3"]
    # bfill across columns row-wise: pick first non-NaN
    best_bid = df[bid_cols].bfill(axis=1).iloc[:, 0]
    best_ask = df[ask_cols].bfill(axis=1).iloc[:, 0]
    return best_bid, best_ask


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add best_bid, best_ask, mid_proper, spread, ob_empty."""
    out = df.copy()
    bb, ba = best_prices(out)
    out["best_bid"] = bb
    out["best_ask"] = ba
    out["mid_proper"] = pd.concat([bb, ba], axis=1).mean(axis=1, skipna=True)
    out["spread"] = ba - bb
    bid_all_nan = out[["bid_price_1", "bid_price_2", "bid_price_3"]].isna().all(axis=1)
    ask_all_nan = out[["ask_price_1", "ask_price_2", "ask_price_3"]].isna().all(axis=1)
    out["ob_empty"] = bid_all_nan & ask_all_nan
    return out


def pivot_wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot to (day, timestamp) -> product wide frame."""
    return prices.pivot_table(index=["day", "timestamp"], columns="product", values=col, aggfunc="first")


def universe_stats(prices: pd.DataFrame, trades: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """Per-product sanity table."""
    rows = []
    for p in products:
        sub = prices[prices["product"] == p]
        if sub.empty:
            continue
        mid = sub["mid_proper"]
        sp = sub["spread"]
        n_trades = int((trades["symbol"] == p).sum()) if "symbol" in trades.columns else 0
        rows.append({
            "product": p,
            "n_ticks": len(sub),
            "min": mid.min(),
            "max": mid.max(),
            "mean": mid.mean(),
            "std": mid.std(),
            "p1": mid.quantile(0.01),
            "p5": mid.quantile(0.05),
            "p50": mid.quantile(0.5),
            "p95": mid.quantile(0.95),
            "p99": mid.quantile(0.99),
            "mean_spread": sp.mean(),
            "p99_spread": sp.quantile(0.99),
            "n_trades": n_trades,
        })
    return pd.DataFrame(rows)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# -------------------- HYDROGEL grid backtester --------------------

def hydrogel_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Filter, sort, return HYDROGEL_PACK ticks with [day, timestamp, best_bid, best_ask, mid_proper]."""
    h = prices[prices["product"] == HYDROGEL].copy()
    h = h.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return h


def make_channel(mids: np.ndarray, mode: str = "static", ema_n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) arrays per tick.

    mode == 'static':  mu, sigma are scalars broadcast to length(mids).
    mode == 'ema':     mu = EMA over ema_n; sigma = global std (constant).
    """
    n = len(mids)
    if mode == "static":
        mu = np.full(n, np.nanmean(mids))
        sigma = np.full(n, np.nanstd(mids))
        return mu, sigma
    elif mode == "ema":
        alpha = 2.0 / (ema_n + 1)
        mu = np.empty(n)
        v = mids[0]
        for i in range(n):
            v = alpha * mids[i] + (1 - alpha) * v
            mu[i] = v
        sigma = np.full(n, np.nanstd(mids))
        return mu, sigma
    else:
        raise ValueError(f"Unknown channel mode: {mode}")


# Exit-rule encoding (small-int for fast dispatch)
EXIT_MEAN = 0
EXIT_MEAN_PLUS_EPS = 1
EXIT_OPPBAND_MINUS_DELTA = 2

# Backtester returns dict; use jitted numpy core for speed.
def grid_backtest(
    days: np.ndarray,           # int array, length N
    timestamps: np.ndarray,     # int array
    bid: np.ndarray,            # best_bid
    ask: np.ndarray,            # best_ask
    mid: np.ndarray,            # best mid
    mu: np.ndarray,             # channel mean per tick
    sigma: np.ndarray,          # channel sigma per tick
    k_l: float,
    k_u: float,
    exit_rule: int,             # EXIT_*
    eps: float,                 # for EXIT_MEAN_PLUS_EPS (sigma multiples)
    delta: float,               # for EXIT_OPPBAND_MINUS_DELTA (sigma multiples)
    k_stop: float,              # 0 means no stop
    entry_size: int,
    pos_cap: int,
    cooldown_ticks: int,
    pessimistic_exit: bool,     # True: exit also crosses spread; False: exit at limit (mid)
) -> dict:
    """Per-tick HYDROGEL grid backtester.

    Long entries at L(k_l) = mu - k_l*sigma, crossing the ask.
    Short entries at U(k_u) = mu + k_u*sigma, crossing the bid.
    Exits per `exit_rule`. Stop-loss at mu ± k_stop*sigma if k_stop > 0.

    Returns dict with: pnl, fills, avg_edge, per_day_pnl, max_dd, pos_pin_pct, n_long, n_short.
    """
    n = len(mid)
    pos = 0
    cash = 0.0
    cooldown = 0
    entry_price = np.nan
    fills = 0
    n_long = 0
    n_short = 0
    edges = []
    realised_pnls_per_day: dict = {}
    pos_pin_count = 0

    realised_total = 0.0
    pnl_curve = np.empty(n)
    last_day = days[0]
    realised_per_day = {int(d): 0.0 for d in np.unique(days)}

    PIN_THRESH = int(0.95 * pos_cap)

    for i in range(n):
        d = int(days[i])
        b = bid[i]
        a = ask[i]
        m = mid[i]
        mu_i = mu[i]
        sg_i = sigma[i]
        if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(m)):
            pnl_curve[i] = realised_total + (pos * (m if np.isfinite(m) else 0.0))
            continue

        L = mu_i - k_l * sg_i
        U = mu_i + k_u * sg_i

        # Exit price target
        if pos > 0:
            if exit_rule == EXIT_MEAN:
                tp = mu_i
            elif exit_rule == EXIT_MEAN_PLUS_EPS:
                tp = mu_i + eps * sg_i
            else:
                tp = U - delta * sg_i
            stop = mu_i - k_stop * sg_i if k_stop > 0 else -np.inf
            # Exit condition: mid >= tp (take-profit) or mid <= stop
            if m >= tp:
                exit_px = b if pessimistic_exit else m
                cash += pos * exit_px
                realised_per_day[d] += pos * (exit_px - entry_price)
                edges.append((exit_px - entry_price))
                pos = 0
                cooldown = cooldown_ticks
            elif k_stop > 0 and m <= stop:
                exit_px = b
                cash += pos * exit_px
                realised_per_day[d] += pos * (exit_px - entry_price)
                edges.append((exit_px - entry_price))
                pos = 0
                cooldown = cooldown_ticks
        elif pos < 0:
            if exit_rule == EXIT_MEAN:
                tp = mu_i
            elif exit_rule == EXIT_MEAN_PLUS_EPS:
                tp = mu_i - eps * sg_i
            else:
                tp = L + delta * sg_i
            stop = mu_i + k_stop * sg_i if k_stop > 0 else np.inf
            if m <= tp:
                exit_px = a if pessimistic_exit else m
                cash += pos * exit_px
                realised_per_day[d] += -pos * (entry_price - exit_px)
                edges.append((entry_price - exit_px))
                pos = 0
                cooldown = cooldown_ticks
            elif k_stop > 0 and m >= stop:
                exit_px = a
                cash += pos * exit_px
                realised_per_day[d] += -pos * (entry_price - exit_px)
                edges.append((entry_price - exit_px))
                pos = 0
                cooldown = cooldown_ticks

        # Entry (only if flat and not in cooldown)
        if pos == 0 and cooldown == 0:
            if m <= L:  # long entry
                size = min(entry_size, pos_cap)
                pos = size
                entry_price = a  # crossed
                cash -= pos * a
                fills += 1
                n_long += 1
            elif m >= U:  # short entry
                size = min(entry_size, pos_cap)
                pos = -size
                entry_price = b
                cash += size * b
                fills += 1
                n_short += 1

        if cooldown > 0:
            cooldown -= 1

        if abs(pos) >= PIN_THRESH:
            pos_pin_count += 1

        # Mark-to-market PnL curve
        realised_total = sum(realised_per_day.values())
        unrealised = 0.0
        if pos != 0:
            unrealised = pos * (m - entry_price)
        pnl_curve[i] = realised_total + unrealised

    # Force-flatten any open position at end (mark at last mid)
    if pos != 0:
        last_m = mid[-1] if np.isfinite(mid[-1]) else (bid[-1] + ask[-1]) / 2
        if pos > 0:
            cash += pos * last_m
            realised_per_day[int(days[-1])] += pos * (last_m - entry_price)
        else:
            cash += pos * last_m
            realised_per_day[int(days[-1])] += -pos * (entry_price - last_m)
        pos = 0

    realised_total = sum(realised_per_day.values())
    # Drawdown
    peak = np.maximum.accumulate(pnl_curve)
    dd = peak - pnl_curve
    max_dd = float(dd.max()) if len(dd) else 0.0

    return {
        "pnl": float(realised_total),
        "per_day_pnl": {k: float(v) for k, v in realised_per_day.items()},
        "fills": fills,
        "n_long": n_long,
        "n_short": n_short,
        "avg_edge": float(np.mean(edges)) if edges else 0.0,
        "max_dd": max_dd,
        "pos_pin_pct": float(pos_pin_count / n),
        "pnl_curve": pnl_curve,
    }


# -------------------- HYDROGEL grid sweep --------------------

EXIT_FAMILY_LABELS = {
    EXIT_MEAN: "MEAN",
    EXIT_MEAN_PLUS_EPS: "MEAN+EPS",
    EXIT_OPPBAND_MINUS_DELTA: "OPP-DELTA",
}


def grid_sweep(
    h: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    k_l_grid: list[float],
    k_u_grid: list[float],
    exit_combos: list[tuple[int, float, float]],  # (rule, eps, delta)
    k_stop_grid: list[float],
    entry_size_grid: list[int],
    pos_cap: int = 160,
    cooldown_ticks: int = 5,
    pessimistic_exit: bool = True,
) -> pd.DataFrame:
    """Run cartesian sweep. Returns DataFrame with one row per config."""
    days = h["day"].values.astype(np.int64)
    ts = h["timestamp"].values.astype(np.int64)
    bid = h["best_bid"].values.astype(np.float64)
    ask = h["best_ask"].values.astype(np.float64)
    mid = h["mid_proper"].values.astype(np.float64)

    rows = []
    for k_l in k_l_grid:
        for k_u in k_u_grid:
            for (rule, eps, delta) in exit_combos:
                for k_stop in k_stop_grid:
                    for entry_size in entry_size_grid:
                        r = grid_backtest(
                            days, ts, bid, ask, mid, mu, sigma,
                            k_l=k_l, k_u=k_u,
                            exit_rule=rule, eps=eps, delta=delta,
                            k_stop=k_stop, entry_size=entry_size,
                            pos_cap=pos_cap, cooldown_ticks=cooldown_ticks,
                            pessimistic_exit=pessimistic_exit,
                        )
                        per_day = r["per_day_pnl"]
                        rows.append({
                            "k_l": k_l, "k_u": k_u,
                            "exit_rule": rule, "eps": eps, "delta": delta,
                            "k_stop": k_stop, "entry_size": entry_size,
                            "pnl": r["pnl"],
                            "pnl_d0": per_day.get(0, 0.0),
                            "pnl_d1": per_day.get(1, 0.0),
                            "pnl_d2": per_day.get(2, 0.0),
                            "fills": r["fills"],
                            "avg_edge": r["avg_edge"],
                            "max_dd": r["max_dd"],
                            "pos_pin_pct": r["pos_pin_pct"],
                            "exit_label": EXIT_FAMILY_LABELS[rule] + (f"+{eps}" if eps else "") + (f"-{delta}" if delta else ""),
                        })
    return pd.DataFrame(rows)


def smoothness_score(df: pd.DataFrame, axes: list[str], threshold_frac: float = 0.7) -> pd.Series:
    """For each row, find Manhattan-1 neighbours along each axis (one step in each direction).
    Smoothness = #(neighbours with PnL >= threshold_frac * row PnL) / #neighbours.
    """
    # Discretise each axis into rank-ordered indices
    axis_values = {ax: sorted(df[ax].unique().tolist()) for ax in axes}
    axis_idx = {ax: {v: i for i, v in enumerate(vals)} for ax, vals in axis_values.items()}

    # Index lookup: tuple(axis values) -> df row
    keys = list(zip(*[df[ax] for ax in axes]))
    # For string-valued axes (exit_label) make sure they're hashable
    lookup: dict[tuple, int] = {}
    for i, k in enumerate(keys):
        lookup[k] = i

    pnl = df["pnl"].values
    smooth = np.zeros(len(df))
    for i in range(len(df)):
        center_pnl = pnl[i]
        if center_pnl <= 0:
            smooth[i] = 0.0
            continue
        center_key = list(keys[i])
        neighbours = 0
        good = 0
        for a, ax in enumerate(axes):
            vals = axis_values[ax]
            cur_idx = vals.index(df[ax].iloc[i])
            for delta_i in (-1, 1):
                ni = cur_idx + delta_i
                if 0 <= ni < len(vals):
                    nb_key = list(center_key)
                    nb_key[a] = vals[ni]
                    nb_idx = lookup.get(tuple(nb_key))
                    if nb_idx is not None:
                        neighbours += 1
                        if pnl[nb_idx] >= threshold_frac * center_pnl:
                            good += 1
        smooth[i] = good / neighbours if neighbours > 0 else 0.0
    return pd.Series(smooth, index=df.index, name="smoothness")


def walk_forward_cv(sweep_df: pd.DataFrame, chosen_idx: int) -> dict:
    """For each LOO fold, compute the chosen config's combined test-days PnL rank vs all configs.
    Returns per-fold rank percentile (lower=better; <=25 means top quartile)."""
    folds = {0: ["pnl_d1", "pnl_d2"], 1: ["pnl_d0", "pnl_d2"], 2: ["pnl_d0", "pnl_d1"]}
    out = {}
    for train_day, test_cols in folds.items():
        test_pnl = sweep_df[test_cols].sum(axis=1)
        chosen_pnl = test_pnl.iloc[chosen_idx]
        # Percentile rank: fraction of configs with strictly lower test PnL → high = good.
        # We invert to "rank percentile from top": 100 - percentile_from_bottom
        pct_from_bottom = (test_pnl < chosen_pnl).mean() * 100
        rank_from_top = 100 - pct_from_bottom
        out[f"fold_train_{train_day}"] = {
            "chosen_test_pnl": float(chosen_pnl),
            "rank_from_top_pct": float(rank_from_top),  # smaller = better; <=25 = top quartile
            "best_test_pnl": float(test_pnl.max()),
            "median_test_pnl": float(test_pnl.median()),
        }
    return out


# -------------------- VEV event extraction --------------------

def infer_trade_side(price: float, best_bid: float, best_ask: float) -> str:
    """Trade-side inference from price vs prevailing book.
       'B' = buyer-aggressor (lifted ask), 'S' = seller-aggressor (hit bid),
       'M' = midpoint / ambiguous."""
    if not (np.isfinite(best_bid) and np.isfinite(best_ask)):
        return "M"
    mid = (best_bid + best_ask) / 2
    if price >= best_ask - 1e-9:
        return "B"
    if price <= best_bid + 1e-9:
        return "S"
    if price > mid:
        return "B"
    if price < mid:
        return "S"
    return "M"


def extract_vev_events(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    strikes: list[int] | None = None,
    forward_horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 50),
    include_l1_shifts: bool = True,
    l1_min_spread_change: int = 2,
) -> pd.DataFrame:
    """Build per-event DataFrame for VEV bot-flow analysis (vectorised).

    TRADE events: trade prints from trades CSV with side inferred from price vs prevailing book.
    L1SHIFT events (optional): ticks where spread changed by >= l1_min_spread_change ticks.
    """
    if strikes is None:
        strikes = [5000, 5100, 5200, 5300, 5400, 5500]

    # Wide pivots
    mid_w = pivot_wide(prices, "mid_proper").sort_index()
    spread_w = pivot_wide(prices, "spread").sort_index()
    bb_w = pivot_wide(prices, "best_bid").sort_index()
    ba_w = pivot_wide(prices, "best_ask").sort_index()
    bid1_w = pivot_wide(prices, "bid_price_1").sort_index()
    ask1_w = pivot_wide(prices, "ask_price_1").sort_index()

    # Per-day numpy arrays for fast forward lookup
    days_unique = sorted(mid_w.index.get_level_values("day").unique())
    per_day = {}
    for d in days_unique:
        sub = mid_w.loc[d]
        per_day[d] = {
            "ts": sub.index.values,
            "mid": {sym: sub[sym].values for sym in sub.columns},
            "spread": {sym: spread_w.loc[d, sym].values if sym in spread_w.columns else None for sym in sub.columns},
            "bb": {sym: bb_w.loc[d, sym].values if sym in bb_w.columns else None for sym in sub.columns},
            "ba": {sym: ba_w.loc[d, sym].values if sym in ba_w.columns else None for sym in sub.columns},
            "bid1": {sym: bid1_w.loc[d, sym].values if sym in bid1_w.columns else None for sym in sub.columns},
            "ask1": {sym: ask1_w.loc[d, sym].values if sym in ask1_w.columns else None for sym in sub.columns},
        }

    rows = []

    # ---- TRADE events ----
    trades_v = trades[trades["symbol"].isin([f"VEV_{k}" for k in strikes])].copy()
    if not trades_v.empty:
        trades_v["K"] = trades_v["symbol"].str.replace("VEV_", "").astype(int)
        for _, tr in trades_v.iterrows():
            d = int(tr["day"])
            if d not in per_day:
                continue
            day = per_day[d]
            ts = int(tr["timestamp"])
            sym = tr["symbol"]
            K = int(tr["K"])
            pos = int(np.searchsorted(day["ts"], ts))
            if pos >= len(day["ts"]) or day["ts"][pos] != ts:
                continue
            bb_arr = day["bb"][sym]
            ba_arr = day["ba"][sym]
            if bb_arr is None or ba_arr is None:
                continue
            side = infer_trade_side(float(tr["price"]), float(bb_arr[pos]), float(ba_arr[pos]))
            rows.append((d, ts, K, "TRADE", side, float(tr["quantity"]), pos, sym))

    # ---- L1SHIFT events (filter by |Δspread| >= l1_min_spread_change) ----
    if include_l1_shifts:
        for K in strikes:
            sym = f"VEV_{K}"
            for d in days_unique:
                day = per_day[d]
                bid1 = day["bid1"].get(sym)
                ask1 = day["ask1"].get(sym)
                if bid1 is None or ask1 is None or len(bid1) < 2:
                    continue
                spread = ask1 - bid1
                d_spread = np.diff(spread, prepend=spread[0])
                d_bid = np.diff(bid1, prepend=bid1[0])
                d_ask = np.diff(ask1, prepend=ask1[0])
                meaningful_idx = np.where(np.abs(d_spread) >= l1_min_spread_change)[0]
                for i in meaningful_idx:
                    if i == 0:
                        continue
                    side = "M"
                    if d_ask[i] < 0 or d_bid[i] > 0:
                        side = "B"
                    elif d_ask[i] > 0 or d_bid[i] < 0:
                        side = "S"
                    rows.append((d, int(day["ts"][i]), K, "L1SHIFT", side, np.nan, int(i), sym))

    if not rows:
        return pd.DataFrame()

    # Vectorised annotation per row using per_day arrays
    out = []
    for d, ts, K, ev_type, side, qty, pos, sym in rows:
        day = per_day[d]
        sym_5200 = "VEV_5200"
        sym_5300 = "VEV_5300"
        mid_K = float(day["mid"][sym][pos]) if sym in day["mid"] else np.nan
        sp_K = float(day["spread"][sym][pos]) if (sym in day["spread"] and day["spread"][sym] is not None) else np.nan
        sp_5200 = float(day["spread"][sym_5200][pos]) if (sym_5200 in day["spread"] and day["spread"][sym_5200] is not None) else np.nan
        sp_5300 = float(day["spread"][sym_5300][pos]) if (sym_5300 in day["spread"] and day["spread"][sym_5300] is not None) else np.nan
        vee_mid = float(day["mid"][VEE][pos]) if VEE in day["mid"] else np.nan
        # Forward mids
        fwd = {}
        n = len(day["ts"])
        for h in forward_horizons:
            j = pos + h
            fwd[f"mid_K_t{h}"] = float(day["mid"][sym][j]) if j < n else np.nan
        vee_mid_t10 = float(day["mid"][VEE][pos + 10]) if pos + 10 < n else np.nan
        # Prior trend over last 20 ticks (linear slope)
        if pos >= 20:
            past = day["mid"][sym][pos - 20:pos]
            past = past[np.isfinite(past)]
            if len(past) >= 5 and np.std(past) > 0:
                prior_trend = float(np.polyfit(np.arange(len(past)), past, 1)[0])
            else:
                prior_trend = 0.0
        else:
            prior_trend = np.nan
        row = {
            "t_event": ts, "day": d, "K": K,
            "event_type": ev_type, "side": side, "qty": qty,
            "mid_K": mid_K, "spread_K": sp_K,
            "spread_5200": sp_5200, "spread_5300": sp_5300,
            "vee_mid": vee_mid, "vee_mid_t10": vee_mid_t10,
            "prior_trend_20": prior_trend,
        }
        row.update(fwd)
        out.append(row)
    return pd.DataFrame(out)


# -------------------- Rule C: spread-compression aggressor-snipe --------------------

# Exit rules for Rule C
EXIT_C_NEXT = 0          # always exit at opposite-side L1 next tick
EXIT_C_HOLD_IF_LOSS = 1  # exit at opposite next tick if profitable, else hold one more (max 2-tick horizon)
EXIT_C_NEXT_MID = 2      # exit at next-tick mid


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    """Standard exponential moving average."""
    alpha = 2.0 / (n + 1)
    out = np.empty_like(arr, dtype=float)
    v = arr[0]
    for i in range(len(arr)):
        v = alpha * arr[i] + (1 - alpha) * v
        out[i] = v
    return out


def rule_c_events(
    prices: pd.DataFrame,
    K: int,
    *,
    compression_threshold: float = 0.5,
    ref_kind: str = "ema_2000",   # "full_mean", "ema_500", "ema_1000", "ema_2000", "ema_5000"
    min_aggressor_ticks: int = 2,
    exit_rule: int = EXIT_C_NEXT,
) -> pd.DataFrame:
    """Detect Rule C events for one strike.

    Returns DataFrame with: t, day, K, side, entry_px, exit_px, raw_pnl, ref_K, spread_at_t,
    spread_at_t-1, aggressor_delta, prior_trend_20.
    """
    sym = f"VEV_{K}"
    sub = prices[prices["product"] == sym].copy().sort_values(["day", "timestamp"]).reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for d in sorted(sub["day"].unique()):
        ds = sub[sub["day"] == d].reset_index(drop=True)
        ts = ds["timestamp"].values
        bid1 = ds["bid_price_1"].values.astype(float)
        ask1 = ds["ask_price_1"].values.astype(float)
        spread = ask1 - bid1
        # Reference spread
        if ref_kind == "full_mean":
            ref = np.full_like(spread, np.nanmean(spread))
        elif ref_kind.startswith("ema_"):
            n_ref = int(ref_kind.split("_")[1])
            valid_spread = np.where(np.isfinite(spread), spread, np.nanmean(spread))
            ref = _ema(valid_spread, n_ref)
        else:
            ref = np.full_like(spread, np.nanmean(spread))

        # mid for prior_trend
        mid = (bid1 + ask1) / 2

        for i in range(20, len(spread) - 1):
            if not (np.isfinite(spread[i]) and np.isfinite(spread[i-1])
                    and np.isfinite(bid1[i]) and np.isfinite(ask1[i])
                    and np.isfinite(bid1[i+1]) and np.isfinite(ask1[i+1])):
                continue
            if spread[i] >= compression_threshold * ref[i]:
                continue
            d_bid = bid1[i] - bid1[i-1]
            d_ask = ask1[i] - ask1[i-1]
            # Aggressor: side that compressed more (moved INTO the spread)
            #   d_ask < 0  -> ask stepped down (aggressor undercut ask)  => BUY at new ask
            #   d_bid > 0  -> bid stepped up   (aggressor overbid bid)   => SELL at new bid
            agg_buy_amt = -d_ask if d_ask < 0 else 0.0
            agg_sell_amt = d_bid if d_bid > 0 else 0.0
            if max(agg_buy_amt, agg_sell_amt) < min_aggressor_ticks:
                continue
            if agg_buy_amt > agg_sell_amt:
                side = "B"
                entry_px = ask1[i]
                aggressor_delta = agg_buy_amt
            elif agg_sell_amt > agg_buy_amt:
                side = "S"
                entry_px = bid1[i]
                aggressor_delta = agg_sell_amt
            else:
                continue  # tie

            # Exit
            if exit_rule == EXIT_C_NEXT_MID:
                exit_px = (bid1[i+1] + ask1[i+1]) / 2
            elif exit_rule == EXIT_C_HOLD_IF_LOSS:
                # Try next-tick opposite first; if loss and we have i+2, hold one more.
                if side == "B":
                    nx_opp = ask1[i+1]
                    raw_now = nx_opp - entry_px
                else:
                    nx_opp = bid1[i+1]
                    raw_now = entry_px - nx_opp
                if raw_now > 0 or (i + 2 >= len(spread)):
                    exit_px = nx_opp
                else:
                    exit_px = ask1[i+2] if side == "B" else bid1[i+2]
            else:  # EXIT_C_NEXT
                exit_px = ask1[i+1] if side == "B" else bid1[i+1]

            if side == "B":
                raw_pnl = exit_px - entry_px
            else:
                raw_pnl = entry_px - exit_px

            # Prior trend over last 20 ticks
            past = mid[max(0, i-20):i]
            past = past[np.isfinite(past)]
            if len(past) >= 5 and np.std(past) > 0:
                prior_trend = float(np.polyfit(np.arange(len(past)), past, 1)[0])
            else:
                prior_trend = 0.0

            rows.append({
                "t": int(ts[i]), "day": int(d), "K": int(K), "side": side,
                "entry_px": float(entry_px), "exit_px": float(exit_px),
                "raw_pnl": float(raw_pnl),
                "ref_K": float(ref[i]),
                "spread_at_t": float(spread[i]), "spread_at_t-1": float(spread[i-1]),
                "aggressor_delta": float(aggressor_delta),
                "prior_trend_20": prior_trend,
            })
    return pd.DataFrame(rows)


def rule_c_sweep(
    prices: pd.DataFrame,
    strikes: list[int] | None = None,
    threshold_grid: list[float] | None = None,
    ref_grid: list[str] | None = None,
    min_agg_grid: list[int] | None = None,
    exit_grid: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Rule C across the parameter grid.

    Returns (events_all, sweep_summary). events_all carries per-event detail; sweep_summary
    aggregates per (threshold, ref, min_agg, exit_rule)."""
    if strikes is None:
        strikes = [5000, 5100, 5200, 5300, 5400, 5500]
    if threshold_grid is None:
        threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    if ref_grid is None:
        ref_grid = ["full_mean", "ema_500", "ema_1000", "ema_2000", "ema_5000"]
    if min_agg_grid is None:
        min_agg_grid = [1, 2, 3, 5]
    if exit_grid is None:
        exit_grid = [EXIT_C_NEXT, EXIT_C_HOLD_IF_LOSS, EXIT_C_NEXT_MID]

    all_events = []
    summary_rows = []
    for thresh in threshold_grid:
        for ref in ref_grid:
            for agg in min_agg_grid:
                for ex in exit_grid:
                    pooled = []
                    for K in strikes:
                        ev = rule_c_events(
                            prices, K,
                            compression_threshold=thresh,
                            ref_kind=ref, min_aggressor_ticks=agg, exit_rule=ex,
                        )
                        if not ev.empty:
                            ev = ev.assign(threshold=thresh, ref=ref, min_agg=agg, exit_rule=ex)
                            pooled.append(ev)
                            all_events.append(ev)
                    if not pooled:
                        n = 0; m = 0; s = 0; t = 0; hit = 0; per_day = {}
                    else:
                        df = pd.concat(pooled, ignore_index=True)
                        n = len(df)
                        r = df["raw_pnl"].values
                        m = float(r.mean())
                        s = float(r.std(ddof=1)) if n > 1 else 0
                        t = m / (s / np.sqrt(n)) if s > 0 else 0
                        hit = float((r > 0).mean())
                        per_day = {int(d): float(df[df.day == d]["raw_pnl"].mean()) if (df.day == d).any() else 0
                                   for d in [0, 1, 2]}
                    summary_rows.append({
                        "threshold": thresh, "ref": ref, "min_agg": agg, "exit_rule": ex,
                        "n": n, "mean_pnl": m, "std_pnl": s, "tstat": t, "hit_rate": hit,
                        "d0_mean_pnl": per_day.get(0, 0),
                        "d1_mean_pnl": per_day.get(1, 0),
                        "d2_mean_pnl": per_day.get(2, 0),
                    })
    if all_events:
        events_all = pd.concat(all_events, ignore_index=True)
    else:
        events_all = pd.DataFrame()
    return events_all, pd.DataFrame(summary_rows)


# -------------------- iteration log --------------------

def append_iteration(strategy: str, action: str, result: str, metrics: dict | None = None) -> None:
    """Append one row to strategy_99_iterations.csv."""
    ensure_output_dir()
    path = OUTPUT_DIR / "strategy_99_iterations.csv"
    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "action": action,
        "result": result,
        "metrics_json": "" if metrics is None else __import__("json").dumps(metrics, default=float),
    }
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
