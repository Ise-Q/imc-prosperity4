#!/usr/bin/env python3
"""
IMC Prosperity 4 — Backtester
==============================
Simulates the competition environment tick-by-tick using the historical
price and trade CSVs, then produces a PnL chart comparable to the
Prosperity website performance tab.

Directory layout expected:
  ./data/
      prices_round_4_day_1.csv
      prices_round_4_day_2.csv
      prices_round_4_day_3.csv
      trades_round_4_day_1.csv
      trades_round_4_day_2.csv
      trades_round_4_day_3.csv
  ./Trader.py          ← your trader
  ./datamodel.py       ← mock datamodel (provided alongside this file)
  ./backtester.py      ← this file

Run:
  python backtester.py

Output saved to ./backtest_output/
"""

import os
import sys
import importlib.util
import traceback
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "./data"
TRADER_PATH = "./Trader.py"
OUTPUT_DIR  = "./backtest_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_FILES = {
    0: f"{DATA_DIR}/prices_round_4_day_1.csv",
    1: f"{DATA_DIR}/prices_round_4_day_2.csv",
    2: f"{DATA_DIR}/prices_round_4_day_3.csv",
}
TRADE_FILES = {
    0: f"{DATA_DIR}/trades_round_4_day_1.csv",
    1: f"{DATA_DIR}/trades_round_4_day_2.csv",
    2: f"{DATA_DIR}/trades_round_4_day_3.csv",
}

# ── Position limits (must mirror Trader.py) ───────────────────────────────────
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
POS_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK"      : 200,
    **{f"VEV_{k}": 300 for k in STRIKES},
}
ALL_PRODUCTS = list(POS_LIMITS.keys())


# ── Dynamic Trader import (avoids circular import issues) ─────────────────────
def load_trader():
    spec   = importlib.util.spec_from_file_location("Trader", TRADER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["Trader"] = module
    spec.loader.exec_module(module)
    return module.Trader()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    price_dfs, trade_dfs = [], []
    for day_idx in range(3):
        p = pd.read_csv(PRICE_FILES[day_idx], sep=";")
        p["_day"] = day_idx
        t = pd.read_csv(TRADE_FILES[day_idx], sep=";")
        t["_day"] = day_idx
        price_dfs.append(p)
        trade_dfs.append(t)

    prices = pd.concat(price_dfs, ignore_index=True).sort_values(["_day", "timestamp"])
    trades = pd.concat(trade_dfs, ignore_index=True).sort_values(["_day", "timestamp"])
    return prices, trades


# ── Order book construction ───────────────────────────────────────────────────
def build_order_depth(row: pd.Series):
    """Build an OrderDepth from a single row of the prices CSV."""
    from datamodel import OrderDepth
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bp > 0:
            od.buy_orders[int(bp)] = int(bv)

        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and ap > 0:
            od.sell_orders[int(ap)] = -int(av)   # negative per IMC convention
    return od


# ── Order filling ─────────────────────────────────────────────────────────────
def fill_orders(
    orders_dict: Dict[str, list],
    order_depths: Dict,
    positions: Dict[str, int],
    cash: List[float],   # single-element list for mutability
    own_trades_out: Dict[str, list],
    timestamp: int,
) -> Dict[str, List]:
    """
    Match trader orders against the reconstructed order book.

    Fill rules (matching competition behaviour):
      BUY  order at price P:  filled at ask levels where ask_price <= P.
                               Fill price = ask_price (best available).
      SELL order at price P:  filled at bid levels where bid_price >= P.
                               Fill price = bid_price (best available).
    Multiple price levels consumed if needed (partial fills supported).
    Returns per-product list of fill records for summary stats.
    """
    from datamodel import Trade

    fill_log: Dict[str, list] = defaultdict(list)

    for symbol, orders in orders_dict.items():
        if not orders:
            continue
        od  = order_depths.get(symbol)
        if od is None:
            continue

        pos_limit = POS_LIMITS.get(symbol, 999999)

        for order in orders:
            qty = order.quantity

            if qty > 0:    # ── BUY ──────────────────────────────────────────
                room = pos_limit - positions.get(symbol, 0)
                qty  = min(qty, max(room, 0))
                if qty == 0:
                    continue

                remaining = qty
                for ask_px in sorted(od.sell_orders.keys()):
                    if ask_px > order.price or remaining <= 0:
                        break
                    available = abs(od.sell_orders[ask_px])
                    filled    = min(remaining, available)
                    cash[0]  -= filled * ask_px
                    positions[symbol] = positions.get(symbol, 0) + filled
                    remaining -= filled
                    fill_log[symbol].append(
                        Trade(symbol, ask_px, filled, buyer="SELF",
                              seller="MARKET", timestamp=timestamp)
                    )
                    own_trades_out[symbol].append(
                        Trade(symbol, ask_px, filled, buyer="SELF",
                              seller="MARKET", timestamp=timestamp)
                    )

            elif qty < 0:  # ── SELL ─────────────────────────────────────────
                qty_abs = abs(qty)
                room    = pos_limit + positions.get(symbol, 0)
                qty_abs = min(qty_abs, max(room, 0))
                if qty_abs == 0:
                    continue

                remaining = qty_abs
                for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_px < order.price or remaining <= 0:
                        break
                    available = od.buy_orders[bid_px]
                    filled    = min(remaining, available)
                    cash[0]  += filled * bid_px
                    positions[symbol] = positions.get(symbol, 0) - filled
                    remaining -= filled
                    fill_log[symbol].append(
                        Trade(symbol, bid_px, -filled, buyer="MARKET",
                              seller="SELF", timestamp=timestamp)
                    )
                    own_trades_out[symbol].append(
                        Trade(symbol, bid_px, -filled, buyer="MARKET",
                              seller="SELF", timestamp=timestamp)
                    )

    return fill_log


# ── PnL computation ───────────────────────────────────────────────────────────
def compute_pnl(cash: float, positions: Dict, mid_prices: Dict) -> float:
    """cash + mark-to-market of all open positions."""
    mtm = sum(positions.get(s, 0) * mid_prices.get(s, 0.0) for s in positions)
    return cash + mtm


def compute_per_product_pnl(
    cash_per_product: Dict, positions: Dict, mid_prices: Dict
) -> Dict[str, float]:
    """Per-product realised cash + unrealised MTM."""
    result = {}
    for s in ALL_PRODUCTS:
        realised   = cash_per_product.get(s, 0.0)
        unrealised = positions.get(s, 0) * mid_prices.get(s, 0.0)
        result[s]  = realised + unrealised
    return result


# ── Main backtester ───────────────────────────────────────────────────────────
def run_backtest():
    print("=" * 65)
    print("  IMC Prosperity 4 — Backtester")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    prices, trades = load_data()

    # Unique ticks in chronological order
    ticks = (prices[["_day", "timestamp"]]
             .drop_duplicates()
             .sort_values(["_day", "timestamp"])
             .values.tolist())   # list of [day_idx, timestamp]
    print(f"  Ticks      : {len(ticks):,}")
    print(f"  Price rows : {len(prices):,}")
    print(f"  Trade rows : {len(trades):,}")

    # Pre-group for speed (group by (_day, timestamp) once)
    prices_g = prices.groupby(["_day", "timestamp"])
    trades_g = trades.groupby(["_day", "timestamp"])

    # ── Load trader ───────────────────────────────────────────────────────────
    print("\n[2] Loading Trader ...")
    try:
        trader = load_trader()
        print("  Trader loaded successfully.")
    except Exception as e:
        print(f"  ERROR loading Trader: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── State ─────────────────────────────────────────────────────────────────
    positions:       Dict[str, int]   = defaultdict(int)
    cash:            List[float]      = [0.0]     # mutable wrapper
    cash_per_product:Dict[str, float] = defaultdict(float)
    trader_data:     str              = ""
    prev_own_trades: Dict[str, list]  = defaultdict(list)

    # ── History records ───────────────────────────────────────────────────────
    pnl_history:      List[dict] = []
    position_history: List[dict] = []
    fill_history:     List[dict] = []
    error_count:      int        = 0

    print(f"\n[3] Running backtest over {len(ticks):,} ticks ...")
    report_interval = max(len(ticks) // 20, 1)

    for tick_idx, (day_idx, timestamp) in enumerate(ticks):

        # Progress
        if tick_idx % report_interval == 0:
            pct = tick_idx / len(ticks) * 100
            total_pnl = pnl_history[-1]["pnl"] if pnl_history else 0
            print(f"  {pct:5.1f}%  day={day_idx}  t={timestamp:>7}  "
                  f"PnL={total_pnl:>10.0f}")

        # ── Build order depths and mid prices for this tick ───────────────────
        order_depths: Dict = {}
        mid_prices:   Dict = {}

        try:
            tick_price_rows = prices_g.get_group((day_idx, timestamp))
        except KeyError:
            continue

        for _, row in tick_price_rows.iterrows():
            product = row["product"]
            order_depths[product] = build_order_depth(row)
            mp = row.get("mid_price")
            if pd.notna(mp):
                mid_prices[product] = float(mp)

        # ── Build market trades for this tick ─────────────────────────────────
        from datamodel import Trade as DmTrade
        market_trades: Dict[str, list] = defaultdict(list)
        try:
            tick_trade_rows = trades_g.get_group((day_idx, timestamp))
            for _, row in tick_trade_rows.iterrows():
                t = DmTrade(
                    symbol    = row["symbol"],
                    price     = int(row["price"]),
                    quantity  = int(row["quantity"]),
                    buyer     = row.get("buyer"),
                    seller    = row.get("seller"),
                    timestamp = int(timestamp),
                )
                market_trades[row["symbol"]].append(t)
        except KeyError:
            pass   # no market trades at this tick — that's fine

        # ── Construct TradingState ────────────────────────────────────────────
        from datamodel import TradingState
        state = TradingState(
            traderData    = trader_data,
            timestamp     = int(timestamp),
            listings      = {},
            order_depths  = order_depths,
            own_trades    = dict(prev_own_trades),
            market_trades = dict(market_trades),
            position      = dict(positions),
            observations  = {},
        )

        # ── Run trader ────────────────────────────────────────────────────────
        try:
            result = trader.run(state)
            # Unpack — IMC returns (orders_dict, conversions, traderData)
            if isinstance(result, tuple) and len(result) == 3:
                orders_dict, _conversions, trader_data = result
            else:
                orders_dict = result or {}
                trader_data = ""
        except Exception as e:
            error_count += 1
            if error_count <= 5:   # show first 5 errors only
                print(f"\n  [WARN] Trader error at day={day_idx} "
                      f"t={timestamp}: {type(e).__name__}: {e}")
            orders_dict = {}

        # ── Fill orders ───────────────────────────────────────────────────────
        prev_own_trades = defaultdict(list)
        fill_log = fill_orders(
            orders_dict   = orders_dict,
            order_depths  = order_depths,
            positions     = positions,
            cash          = cash,
            own_trades_out= prev_own_trades,
            timestamp     = int(timestamp),
        )

        # Track per-product cash flow
        for symbol, fills in fill_log.items():
            for fill in fills:
                # fill.quantity < 0 means we sold (received cash)
                cash_per_product[symbol] -= fill.price * fill.quantity

        # Aggregate fill history (sample every 100th tick to keep memory low)
        if tick_idx % 100 == 0:
            for symbol, fills in fill_log.items():
                for fill in fills:
                    fill_history.append({
                        "day_idx"  : day_idx,
                        "timestamp": timestamp,
                        "symbol"   : symbol,
                        "price"    : fill.price,
                        "quantity" : fill.quantity,
                    })

        # ── Record PnL ────────────────────────────────────────────────────────
        total_pnl   = compute_pnl(cash[0], positions, mid_prices)
        per_pnl     = compute_per_product_pnl(cash_per_product, positions, mid_prices)
        elapsed     = day_idx + timestamp / 1_000_000

        pnl_record  = {
            "elapsed"   : elapsed,
            "day_idx"   : day_idx,
            "timestamp" : timestamp,
            "pnl"       : total_pnl,
            "cash"      : cash[0],
        }
        pnl_record.update({f"pnl_{s}": per_pnl.get(s, 0.0) for s in ALL_PRODUCTS})
        pnl_history.append(pnl_record)

        pos_record = {"elapsed": elapsed, "day_idx": day_idx, "timestamp": timestamp}
        pos_record.update({s: positions.get(s, 0) for s in ALL_PRODUCTS})
        position_history.append(pos_record)

    print(f"\n  Done. {error_count} trader errors across {len(ticks):,} ticks.")

    # ── Convert to DataFrames ─────────────────────────────────────────────────
    pnl_df = pd.DataFrame(pnl_history)
    pos_df = pd.DataFrame(position_history)

    return pnl_df, pos_df, fill_history, positions, mid_prices


# ── Summary statistics ────────────────────────────────────────────────────────
def print_summary(pnl_df: pd.DataFrame, pos_df: pd.DataFrame,
                  fill_history: list, final_positions: dict,
                  final_mids: dict):
    print("\n" + "=" * 65)
    print("  BACKTEST SUMMARY")
    print("=" * 65)

    final_pnl = pnl_df["pnl"].iloc[-1]
    peak      = pnl_df["pnl"].max()
    trough    = pnl_df["pnl"].min()
    drawdown  = peak - trough

    pnl_diff  = pnl_df["pnl"].diff().dropna()
    sharpe    = (pnl_diff.mean() / pnl_diff.std() * math.sqrt(len(pnl_diff))
                 if pnl_diff.std() > 0 else 0)

    print(f"\n  Final PnL      : {final_pnl:>12,.0f}")
    print(f"  Peak PnL       : {peak:>12,.0f}")
    print(f"  Trough PnL     : {trough:>12,.0f}")
    print(f"  Max drawdown   : {drawdown:>12,.0f}")
    print(f"  Sharpe (proxy) : {sharpe:>12.3f}")

    print("\n  Per-product final PnL:")
    product_pnls = []
    for s in ALL_PRODUCTS:
        col = f"pnl_{s}"
        if col in pnl_df.columns:
            v = pnl_df[col].iloc[-1]
            product_pnls.append((s, v))
    product_pnls.sort(key=lambda x: x[1], reverse=True)
    for s, v in product_pnls:
        if abs(v) > 0.5:
            print(f"    {s:<30} {v:>10,.0f}")

    print("\n  Final positions:")
    for s, pos in final_positions.items():
        if pos != 0:
            mid = final_mids.get(s, 0)
            print(f"    {s:<30} {pos:>+6d}   mid={mid:.1f}   MTM={pos*mid:>8,.0f}")

    fills_df = pd.DataFrame(fill_history)
    if len(fills_df):
        print(f"\n  Total fill records sampled : {len(fills_df):,}")
        print(f"  Products with fills        : "
              f"{fills_df['symbol'].nunique()}")


# ── Plotting ──────────────────────────────────────────────────────────────────
PRODUCT_COLORS = {
    "VELVETFRUIT_EXTRACT": "#378ADD",
    "HYDROGEL_PACK"      : "#1D9E75",
    **{f"VEV_{k}": f"#{c}" for k, c in zip(
        STRIKES,
        ["E24B4A","D85A30","BA7517","639922","3C3489",
         "0F6E56","993C1D","533AAA","A32D2D","185FA5"]
    )},
}


def plot_results(pnl_df: pd.DataFrame, pos_df: pd.DataFrame):
    x = pnl_df["elapsed"].values

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: Total PnL (full width, matches Prosperity website chart) ─────
    ax1 = fig.add_subplot(gs[0, :])
    pnl = pnl_df["pnl"].values
    color = "#1D9E75" if pnl[-1] >= 0 else "#E24B4A"
    ax1.plot(x, pnl, lw=1.2, color=color)
    ax1.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax1.fill_between(x, pnl, 0, alpha=0.12, color=color)
    for d in [1.0, 2.0]:
        ax1.axvline(d, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax1.text(d + 0.01, ax1.get_ylim()[0], f"Day {int(d)+1}",
                 fontsize=8, color="gray", va="bottom")
    ax1.set_title("Total PnL over time", fontsize=12, fontweight="500")
    ax1.set_xlabel("Elapsed days")
    ax1.set_ylabel("PnL")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Per-product PnL (excluding near-zero products) ──────────────
    ax2 = fig.add_subplot(gs[1, :])
    for s in ALL_PRODUCTS:
        col = f"pnl_{s}"
        if col not in pnl_df.columns:
            continue
        series = pnl_df[col].values
        if max(abs(series)) < 100:    # skip products with negligible PnL
            continue
        ax2.plot(x, series, lw=0.8, label=s,
                 color=PRODUCT_COLORS.get(s, "gray"), alpha=0.85)
    ax2.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax2.set_title("PnL by product", fontsize=12, fontweight="500")
    ax2.set_xlabel("Elapsed days")
    ax2.set_ylabel("PnL")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    ax2.legend(fontsize=7, ncol=4, loc="upper left")
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: HYDROGEL position ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    hyd_pos = pos_df["HYDROGEL_PACK"].values if "HYDROGEL_PACK" in pos_df.columns else np.zeros(len(x))
    ax3.plot(x, hyd_pos, lw=0.8, color=PRODUCT_COLORS["HYDROGEL_PACK"])
    ax3.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax3.axhline( 200, color="red", lw=0.5, ls=":", alpha=0.5, label="+limit")
    ax3.axhline(-200, color="red", lw=0.5, ls=":", alpha=0.5, label="-limit")
    ax3.set_title("HYDROGEL_PACK position", fontsize=11, fontweight="500")
    ax3.set_xlabel("Elapsed days")
    ax3.set_ylabel("Position")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: VEV option positions (active strikes only) ───────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    for k in [5200, 5300, 5400, 5500]:
        col = f"VEV_{k}"
        if col not in pos_df.columns:
            continue
        series = pos_df[col].values
        if max(abs(series)) < 5:
            continue
        ax4.plot(x, series, lw=0.8, label=f"VEV_{k}",
                 color=PRODUCT_COLORS.get(col, "gray"))
    ax4.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax4.set_title("VEV option positions (active strikes)", fontsize=11, fontweight="500")
    ax4.set_xlabel("Elapsed days")
    ax4.set_ylabel("Position")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.2)

    fig.suptitle("IMC Prosperity 4 — Backtest Results", fontsize=14, fontweight="500", y=0.98)

    out = f"{OUTPUT_DIR}/backtest_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {out}")

    # ── Second figure: PnL drawdown decomposition ─────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Rolling drawdown
    ax = axes[0]
    cummax = np.maximum.accumulate(pnl)
    dd     = pnl - cummax
    ax.fill_between(x, dd, 0, color="#E24B4A", alpha=0.5)
    ax.plot(x, dd, lw=0.6, color="#E24B4A")
    ax.set_title("Drawdown from peak", fontsize=11, fontweight="500")
    ax.set_xlabel("Elapsed days")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, alpha=0.2)

    # PnL histogram
    ax = axes[1]
    pnl_changes = np.diff(pnl)
    ax.hist(pnl_changes, bins=80, color="#378ADD", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(np.mean(pnl_changes), color="#1D9E75", lw=1.2, ls="--",
               label=f"mean={np.mean(pnl_changes):.1f}")
    ax.set_title("Tick-by-tick PnL change distribution", fontsize=11, fontweight="500")
    ax.set_xlabel("PnL change per tick")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out2 = f"{OUTPUT_DIR}/backtest_diagnostics.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out2}")


# ── Save CSVs ─────────────────────────────────────────────────────────────────
def save_csvs(pnl_df: pd.DataFrame, pos_df: pd.DataFrame):
    pnl_df.round(2).to_csv(f"{OUTPUT_DIR}/pnl_history.csv", index=False)
    pos_df.to_csv(f"{OUTPUT_DIR}/position_history.csv", index=False)
    print(f"  CSVs saved to {OUTPUT_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pnl_df, pos_df, fill_history, final_positions, final_mids = run_backtest()
    print_summary(pnl_df, pos_df, fill_history, final_positions, final_mids)
    print("\n[4] Generating plots ...")
    plot_results(pnl_df, pos_df)
    print("\n[5] Saving CSVs ...")
    save_csvs(pnl_df, pos_df)
    print("\nDone. Check ./backtest_output/ for results.")