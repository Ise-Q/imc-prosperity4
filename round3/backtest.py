"""
IMC Prosperity 4 — Round 3 Backtester (fast version)
======================================================
Runs all 30,000 ticks in ~30s by pre-indexing all data into dicts
so every tick lookup is O(1), and silencing the trader's print() calls.

Usage:   uv run python backtest.py
Output:  backtest_output/ folder with 5 charts + printed summary table.

Order matching rules
--------------------
  BUY  order at price P  →  fill against asks ≤ P  (cheapest first, then mkt trades)
  SELL order at price P  →  fill against bids ≥ P  (most expensive first, then mkt trades)

Position limit enforcement
--------------------------
  If total_buy  orders (assuming full fill) would push pos above +limit, or
  if total_sell orders would push pos below  -limit,  ALL orders for that
  product are cancelled. This mirrors the official Prosperity rule.

PnL accounting
--------------
  PnL(product) = realised_cash_flow + current_position × current_mid_price
  Total PnL = sum over all products.
"""

import sys, os, importlib.util, types, math
from pathlib import Path
from collections import defaultdict
from io import StringIO
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("./data")
OUT_DIR     = Path("./backtest_output")
TRADER_FILE = Path("./trader.py")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILES = {
    0: DATA_DIR / "prices_round_3_day_0.csv",
    1: DATA_DIR / "prices_round_3_day_1.csv",
    2: DATA_DIR / "prices_round_3_day_2.csv",
}
TRADE_FILES = {
    0: DATA_DIR / "trades_round_3_day_0.csv",
    1: DATA_DIR / "trades_round_3_day_1.csv",
    2: DATA_DIR / "trades_round_3_day_2.csv",
}

POS_LIMITS = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK":        200,
    **{f"VEV_{k}": 300 for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]},
}

# ── Datamodel stubs ───────────────────────────────────────────────────────────
class Order:
    __slots__ = ("symbol","price","quantity")
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol; self.price = int(price); self.quantity = int(quantity)

class Trade:
    __slots__ = ("symbol","price","quantity","buyer","seller","timestamp")
    def __init__(self, symbol, price, quantity, buyer="", seller="", timestamp=0):
        self.symbol=symbol; self.price=price; self.quantity=quantity
        self.buyer=buyer; self.seller=seller; self.timestamp=timestamp

class OrderDepth:
    __slots__ = ("buy_orders","sell_orders")
    def __init__(self): self.buy_orders={}; self.sell_orders={}

class Observation:
    def __init__(self): self.plainValueObservations={}; self.conversionObservations={}

class TradingState:
    __slots__ = ("traderData","timestamp","listings","order_depths",
                 "own_trades","market_trades","position","observations")
    def __init__(self, traderData, timestamp, order_depths,
                 own_trades, market_trades, position, observations):
        self.traderData=traderData; self.timestamp=timestamp; self.listings={}
        self.order_depths=order_depths; self.own_trades=own_trades
        self.market_trades=market_trades; self.position=position
        self.observations=observations

_dm = types.ModuleType("datamodel")
for _cls in (Order, Trade, OrderDepth, TradingState, Observation):
    setattr(_dm, _cls.__name__, _cls)
sys.modules["datamodel"] = _dm

# ── Load trader ───────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("trader_module", TRADER_FILE)
trader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trader_mod)
TRADER = trader_mod.Trader()
print("Trader loaded OK.")

# ── 1. Load CSVs ──────────────────────────────────────────────────────────────
print("Loading CSVs ...")
price_dfs, trade_dfs = [], []
for day in range(3):
    p = pd.read_csv(PRICE_FILES[day], sep=";"); p["day"] = day
    t = pd.read_csv(TRADE_FILES[day], sep=";"); t["day"] = day
    price_dfs.append(p); trade_dfs.append(t)
prices = pd.concat(price_dfs, ignore_index=True)
trades = pd.concat(trade_dfs, ignore_index=True)

prices["global_t"] = prices["day"] * 1_000_000 + prices["timestamp"]
trades["global_t"] = trades["day"] * 1_000_000 + trades["timestamp"]
trades["price"]    = pd.to_numeric(trades["price"],    errors="coerce")
trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
trades = trades.dropna(subset=["price","quantity"])

for col in ["bid_price_1","bid_volume_1","bid_price_2","bid_volume_2",
            "bid_price_3","bid_volume_3","ask_price_1","ask_volume_1",
            "ask_price_2","ask_volume_2","ask_price_3","ask_volume_3","mid_price"]:
    prices[col] = pd.to_numeric(prices[col], errors="coerce")

print(f"  {len(prices):,} price rows  |  {len(trades):,} trade rows")

# ── 2. Pre-index everything for O(1) per-tick lookup ─────────────────────────
print("Pre-indexing ...")

# depths_index[global_t][product] = OrderDepth
depths_index: dict[int, dict[str, OrderDepth]] = defaultdict(dict)
mid_index:    dict[int, dict[str, float]]       = defaultdict(dict)

for row in prices.itertuples(index=False):
    gt = int(row.global_t)
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = getattr(row, f"bid_price_{i}"); bv = getattr(row, f"bid_volume_{i}")
        ap = getattr(row, f"ask_price_{i}"); av = getattr(row, f"ask_volume_{i}")
        if not (math.isnan(bp) if isinstance(bp, float) else False) and bv > 0:
            od.buy_orders[int(bp)] =  int(bv)
        if not (math.isnan(ap) if isinstance(ap, float) else False) and av > 0:
            od.sell_orders[int(ap)] = -int(av)
    depths_index[gt][row.product] = od
    if not (isinstance(row.mid_price, float) and math.isnan(row.mid_price)):
        mid_index[gt][row.product] = float(row.mid_price)

# trades_index[global_t][symbol] = [Trade, ...]
trades_index: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
for row in trades.itertuples(index=False):
    gt  = int(row.global_t)
    sym = row.symbol
    trades_index[gt][sym].append(
        Trade(sym, float(row.price), int(row.quantity),
              str(row.buyer) if hasattr(row,"buyer") else "",
              str(row.seller) if hasattr(row,"seller") else "", gt))

all_timestamps = sorted(depths_index.keys())
all_products   = sorted(set(prices["product"].unique()))
print(f"  {len(all_timestamps):,} unique timestamps  |  {len(all_products)} products")

# ── 3. Helper functions ───────────────────────────────────────────────────────
def enforce_limits(orders_dict: dict, position: dict) -> dict:
    clean = {}
    cancelled = []
    for sym, orders in orders_dict.items():
        lim    = POS_LIMITS.get(sym, 0)
        pos    = position.get(sym, 0)
        t_buy  = sum(o.quantity       for o in orders if o.quantity > 0)
        t_sell = sum(abs(o.quantity)  for o in orders if o.quantity < 0)
        if pos + t_buy > lim or pos - t_sell < -lim:
            cancelled.append(f"{sym}(pos={pos} +{t_buy}/-{t_sell} lim={lim})")
            clean[sym] = []
        else:
            clean[sym] = orders
    if cancelled:
        pass  # silent — uncomment to debug: print(f"  [LIMIT] cancelled: {cancelled}")
    return clean

def match_orders(orders_dict: dict, depths: dict, mkt_trades: dict) -> list:
    fills = []
    for sym, orders in orders_dict.items():
        od   = depths.get(sym, OrderDepth())
        asks = sorted([(p, abs(v)) for p, v in od.sell_orders.items()])
        bids = sorted([(p, abs(v)) for p, v in od.buy_orders.items()], reverse=True)
        mkt  = mkt_trades.get(sym, [])

        for order in orders:
            rem = abs(order.quantity)
            if rem == 0:
                continue
            if order.quantity > 0:           # BUY
                for ap, av in asks:
                    if ap > order.price or rem == 0: break
                    f = min(rem, av); fills.append((sym, ap, f, "BUY")); rem -= f
                for tr in mkt:
                    if rem == 0: break
                    if tr.price <= order.price:
                        f = min(rem, tr.quantity); fills.append((sym, tr.price, f, "BUY")); rem -= f
            else:                            # SELL
                for bp, bv in bids:
                    if bp < order.price or rem == 0: break
                    f = min(rem, bv); fills.append((sym, bp, f, "SELL")); rem -= f
                for tr in mkt:
                    if rem == 0: break
                    if tr.price >= order.price:
                        f = min(rem, tr.quantity); fills.append((sym, tr.price, f, "SELL")); rem -= f
    return fills

# ── 4. Simulation loop ────────────────────────────────────────────────────────
print("\nRunning backtest ...")
position     = {}
cash         = defaultdict(float)
trader_data  = ""
own_trades   = defaultdict(list)

# history[sym] = list of (global_t, pnl, position)
pnl_history  = defaultdict(list)
mid_hist     = defaultdict(list)    # for final plots only (sparse)

N = len(all_timestamps)
report_at = set(range(0, N, N // 10))
_sink = StringIO(); _sink.close = lambda: None

for i, gt in enumerate(all_timestamps):
    if i in report_at:
        total = sum(
            cash[s] + position.get(s, 0) * (mid_hist[s][-1] if mid_hist[s] else 0)
            for s in all_products
        )
        print(f"  {i/N*100:4.0f}%  gt={gt:>9,}  total PnL ≈ {total:+,.0f}")

    depths   = depths_index[gt]
    mkt_t    = trades_index.get(gt, {})
    local_ts = gt % 1_000_000

    # Build TradingState
    state = TradingState(
        traderData    = trader_data,
        timestamp     = local_ts,
        order_depths  = depths,
        own_trades    = dict(own_trades),
        market_trades = dict(mkt_t),
        position      = dict(position),
        observations  = Observation(),
    )

    # Run trader (silence prints)
    try:
        with redirect_stdout(_sink):
            orders_dict, _, trader_data = TRADER.run(state)
    except Exception as e:
        orders_dict = {}
        trader_data = trader_data or ""

    # Enforce limits then match
    orders_dict = enforce_limits(orders_dict, position)
    fills       = match_orders(orders_dict, depths, mkt_t)

    # Settle fills
    own_trades = defaultdict(list)
    for sym, price, qty, side in fills:
        if side == "BUY":
            position[sym] = position.get(sym, 0) + qty
            cash[sym]    -= price * qty
            own_trades[sym].append(Trade(sym, price, qty, "SUBMISSION", "", gt))
        else:
            position[sym] = position.get(sym, 0) - qty
            cash[sym]    += price * qty
            own_trades[sym].append(Trade(sym, price, qty, "", "SUBMISSION", gt))

    # Record PnL (mark-to-market)
    mids = mid_index.get(gt, {})
    for sym in all_products:
        mid = mids.get(sym, (mid_hist[sym][-1] if mid_hist[sym] else 0))
        mid_hist[sym].append(mid)
        pos = position.get(sym, 0)
        pnl_history[sym].append((gt, cash[sym] + pos * mid, pos))

print("\nDone.\n")

# ── 5. Summary ────────────────────────────────────────────────────────────────
rows = []
for sym in sorted(all_products):
    h = pnl_history[sym]
    if not h: continue
    final_pnl = h[-1][1]
    final_pos = h[-1][2]
    max_pos   = max(abs(r[2]) for r in h)
    rows.append((sym, final_pnl, final_pos, max_pos))

total_pnl = sum(r[1] for r in rows)
print(f"{'Product':<30} {'Final PnL':>13} {'Final Pos':>10} {'Max Pos':>10}")
print("-" * 68)
for sym, pnl, pos, mp in rows:
    print(f"  {sym:<28} {pnl:>+13,.1f} {pos:>+10} {mp:>10}")
print("-" * 68)
print(f"  {'TOTAL':<28} {total_pnl:>+13,.1f}")

# ── 6. Charts ─────────────────────────────────────────────────────────────────
print("\nGenerating charts ...")

vev   = [r[0] for r in rows if r[0].startswith("VEV_")]
d1    = [r[0] for r in rows if not r[0].startswith("VEV_")]
colors_map = {s: f"C{i}" for i, s in enumerate(all_products)}

# A. Cumulative total PnL
ts_all   = [r[0] for r in pnl_history[all_products[0]]]
total_ts = [sum(pnl_history[s][i][1] for s in all_products if pnl_history[s]) for i in range(len(ts_all))]

fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(ts_all, total_ts, lw=1.0, color="steelblue")
ax.fill_between(ts_all, total_ts, 0, alpha=0.12, color="steelblue")
ax.axhline(0, color="black", lw=0.6)
for d in [1,2]: ax.axvline(d*1_000_000, color="grey", lw=1, ls="--", alpha=0.5)
ax.set_title(f"Cumulative Total PnL  (Final: {total_pnl:+,.0f} XIRECs)")
ax.set_xlabel("Global timestamp"); ax.set_ylabel("PnL (XIRECs)")
plt.tight_layout(); fig.savefig(OUT_DIR/"A_total_pnl.png", dpi=150); plt.close()

# B. PnL per delta-1 product
fig, axes = plt.subplots(1, max(len(d1),1), figsize=(7*len(d1), 4), squeeze=False)
for ax, sym in zip(axes[0], d1):
    h = pnl_history[sym]; ts=[r[0] for r in h]; pnl=[r[1] for r in h]
    ax.plot(ts, pnl, lw=0.8); ax.axhline(0, color="black", lw=0.5, ls="--")
    for d in [1,2]: ax.axvline(d*1_000_000, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"{sym}\nFinal PnL: {pnl[-1]:+,.0f}")
    ax.set_xlabel("Global timestamp"); ax.set_ylabel("PnL")
plt.tight_layout(); fig.savefig(OUT_DIR/"B_pnl_delta1.png", dpi=150); plt.close()

# C. PnL per VEV option (2×5 grid)
fig, axes = plt.subplots(2, 5, figsize=(22, 7))
for i, sym in enumerate(vev):
    ax = axes[i//5][i%5]
    h  = pnl_history.get(sym, [])
    if not h: ax.set_title(sym); continue
    ts  = [r[0] for r in h]; pnl = [r[1] for r in h]
    ax.plot(ts, pnl, lw=0.6)
    ax.axhline(0, color="black", lw=0.4, ls="--")
    ax.set_title(f"{sym}  (PnL:{pnl[-1]:+,.0f})", fontsize=9)
    ax.tick_params(labelsize=7)
plt.suptitle("PnL over time — VEV Options", fontsize=13)
plt.tight_layout(); fig.savefig(OUT_DIR/"C_pnl_vev.png", dpi=150); plt.close()

# D. Position over time — VEV options (2×5 grid)
fig, axes = plt.subplots(2, 5, figsize=(22, 7))
for i, sym in enumerate(vev):
    ax = axes[i//5][i%5]
    h  = pnl_history.get(sym, [])
    if not h: ax.set_title(sym); continue
    ts  = [r[0] for r in h]; pos = [r[2] for r in h]
    ax.plot(ts, pos, lw=0.6, color="darkorchid")
    ax.axhline(0, color="black", lw=0.4, ls="--")
    lim = POS_LIMITS.get(sym, 300)
    ax.axhline( lim, color="red", lw=0.4, ls=":", alpha=0.6)
    ax.axhline(-lim, color="red", lw=0.4, ls=":", alpha=0.6)
    ax.set_title(f"{sym}  (pos:{pos[-1]:+d})", fontsize=9)
    ax.tick_params(labelsize=7)
plt.suptitle("Position over time — VEV Options", fontsize=13)
plt.tight_layout(); fig.savefig(OUT_DIR/"D_positions_vev.png", dpi=150); plt.close()

# E. Final PnL bar chart (all products)
fig, ax = plt.subplots(figsize=(14, 5))
syms = [r[0] for r in rows]; pnls = [r[1] for r in rows]
cols = ["steelblue" if p >= 0 else "tomato" for p in pnls]
ax.bar(range(len(syms)), pnls, color=cols)
ax.set_xticks(range(len(syms)))
ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Final PnL per Product  (Total: {total_pnl:+,.0f} XIRECs)")
plt.tight_layout(); fig.savefig(OUT_DIR/"E_final_pnl_bars.png", dpi=150); plt.close()

print(f"Charts saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == ".png" and f.name[0] in "ABCDE":
        print(f"  {f.name}")