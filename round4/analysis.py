"""
IMC Prosperity 4 — Round 4 Analysis
=====================================
Products:
  VELVETFRUIT_EXTRACT        — underlying, ~5250
  VEV_4000 … VEV_6500        — call options (10 strikes)
  HYDROGEL_PACK              — mean-reverting, ~9991

TTE (time-to-expiry) mapping — confirmed from Round 4 rules:
  Historical data covers 3 days of the PREVIOUS rounds.
  Options launched at start of the game (TTE=8 at timestamp 0 of day 0).

  Formula for historical data:
      TTE = 8 - day - timestamp / 1_000_000

  Formula inside the Round 4 trader:
      TTE = 5 - (day_offset + timestamp / 1_000_000)
      (starts at 5 when Round 4 begins, counts down to 0 at expiry)

Calibration result (from historical data with correct TTE):
  sigma ≈ 0.01255  per competition-day  (1 "year" = 1 comp-day)
  Extremely stable: IV in [0.01252, 0.01266] across ATM strikes.

Position limits (official Round 4 rules):
  HYDROGEL_PACK              → 200
  VELVETFRUIT_EXTRACT        → 200
  VEV_XXXX (each voucher)    → 300

Key counterparty findings:
  Mark 67  — pure buyer of VEV (net +1510, never sells) → informed long signal
  Mark 01  — systematic buyer of OTM call options from Mark 22
  Mark 22  — options dealer (net short all strikes), VEV market participant
  Mark 55  — neutral VEV market maker (net ~0 over thousands of trades)
  Mark 49  — persistent net seller of VEV (net -956) → likely uninformed
  Mark 14/38 — two-sided HYDROGEL market makers, modest VEV option buyers
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
OUTPUT_DIR = "./analysis_output"
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

STRIKES       = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_COLS      = [f"VEV_{k}" for k in STRIKES]
ALL_PRODUCTS  = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_COLS
TRADERS       = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]
TICKS_PER_DAY = 1_000_000


# ── Black-Scholes helpers (no scipy) ─────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price under Black-Scholes (r=0)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(S * norm_cdf(d1) - K * norm_cdf(d2))


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS delta of a European call."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return float(norm_cdf(d1))


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """BS gamma of a European call."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    return math.exp(-0.5 * d1 ** 2) / (S * sigma * sqrtT * math.sqrt(2 * math.pi))


def implied_vol(mkt: float, S: float, K: float, T: float,
                lo: float = 1e-4, hi: float = 3.0) -> float:
    """Bisection implied vol. Returns nan if no solution exists."""
    intrinsic = max(S - K, 0.0)
    if mkt <= intrinsic + 0.01 or T <= 0:
        return np.nan
    if bs_call(S, K, T, hi) < mkt:
        return np.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_call(S, K, T, mid) < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  IMC Prosperity 4 — Round 4 Analysis")
print("=" * 65)
print("\n[1] Loading data ...")

price_dfs, trade_dfs = [], []
for day in range(3):
    p = pd.read_csv(PRICE_FILES[day], sep=";")
    p["day"] = day
    t = pd.read_csv(TRADE_FILES[day], sep=";")
    t["day"] = day
    price_dfs.append(p)
    trade_dfs.append(t)

prices = pd.concat(price_dfs, ignore_index=True)
trades = pd.concat(trade_dfs, ignore_index=True)

# Elapsed time and correct TTE
prices["elapsed_days"] = prices["day"] + prices["timestamp"] / TICKS_PER_DAY
prices["TTE"]          = 8.0 - prices["elapsed_days"]

print(f"  Price rows  : {len(prices):,}")
print(f"  Trade rows  : {len(trades):,}")
print(f"  TTE range   : [{prices['TTE'].min():.3f}, {prices['TTE'].max():.3f}] days")


# ── 2. Pivot price table to wide format ───────────────────────────────────────
print("\n[2] Pivoting prices to wide format ...")
mid = (
    prices[prices["product"].isin(ALL_PRODUCTS)]
    .pivot_table(
        index=["day", "timestamp", "elapsed_days", "TTE"],
        columns="product",
        values="mid_price",
        aggfunc="first",
    )
    .reset_index()
    .sort_values("elapsed_days")
)
mid.columns.name = None


# ── 3. VELVETFRUIT_EXTRACT stats ──────────────────────────────────────────────
print("\n[3] VELVETFRUIT_EXTRACT statistics")
print("-" * 45)

und = mid[["elapsed_days", "day", "TTE", "VELVETFRUIT_EXTRACT"]].dropna()
desc = und["VELVETFRUIT_EXTRACT"].describe()
print(desc.round(2).to_string())

S_series = und.set_index("elapsed_days")["VELVETFRUIT_EXTRACT"]
log_ret  = np.log(S_series / S_series.shift(1)).dropna()
sigma_realized = log_ret.std() * np.sqrt(TICKS_PER_DAY)
print(f"\n  Realized sigma per comp-day : {sigma_realized:.5f}")
print(f"  Mean                        : {und['VELVETFRUIT_EXTRACT'].mean():.2f}")
print(f"  Std                         : {und['VELVETFRUIT_EXTRACT'].std():.2f}")


# ── 4. Implied vol calibration ────────────────────────────────────────────────
print("\n[4] Implied volatility calibration (correct TTE)")
print("-" * 45)

sample = mid.iloc[::100].copy()
iv_rows = []

for _, row in sample.iterrows():
    S   = row.get("VELVETFRUIT_EXTRACT", np.nan)
    TTE = row.get("TTE", 0.0)
    if np.isnan(S) or TTE <= 0:
        continue
    for K in STRIKES:
        col = f"VEV_{K}"
        mkt = row.get(col, np.nan)
        if np.isnan(mkt):
            continue
        iv  = implied_vol(mkt, S, K, TTE)
        dlt = bs_delta(S, K, TTE, iv) if not np.isnan(iv) else np.nan
        iv_rows.append({
            "elapsed_days": row["elapsed_days"],
            "day"         : row["day"],
            "TTE"         : TTE,
            "strike"      : K,
            "S"           : S,
            "mkt_price"   : mkt,
            "IV"          : iv,
            "delta"       : dlt,
            "moneyness"   : S / K,
        })

iv_df = pd.DataFrame(iv_rows)

# ATM-only calibration (moneyness 0.90–1.10) for reliable sigma estimate
atm = iv_df[
    (iv_df["moneyness"] > 0.90) &
    (iv_df["moneyness"] < 1.10) &
    iv_df["IV"].notna()
]
sigma_atm_mean = atm["IV"].mean()
sigma_atm_std  = atm["IV"].std()

print(f"  ATM IV mean  : {sigma_atm_mean:.5f}")
print(f"  ATM IV std   : {sigma_atm_std:.5f}  (stability check — lower is better)")
print(f"  Realized σ   : {sigma_realized:.5f}")

print("\n  Per-strike avg implied vol:")
per_strike = iv_df.groupby("strike")["IV"].mean()
for K, iv in per_strike.items():
    label = "  ← ATM" if abs(K - 5250) < 300 else ""
    print(f"    VEV_{K:5d}: {iv:.5f}{label}")


# ── 5. Round 4 fair values ────────────────────────────────────────────────────
print(f"\n[5] Round 4 fair values  (TTE=5, S=5250, sigma={sigma_atm_mean:.5f})")
print("-" * 45)

S0, T_r4 = 5250.0, 5.0
print(f"  {'Strike':<10} {'BS fair':>10} {'Delta':>8} {'Gamma':>10} {'Moneyness':>11}")
print(f"  {'-'*49}")
for K in STRIKES:
    fv  = bs_call(S0, K, T_r4, sigma_atm_mean)
    dlt = bs_delta(S0, K, T_r4, sigma_atm_mean)
    gam = bs_gamma(S0, K, T_r4, sigma_atm_mean)
    mon = S0 / K
    print(f"  VEV_{K:<6} {fv:>10.2f} {dlt:>8.4f} {gam:>10.6f} {mon:>11.4f}")


# ── 6. HYDROGEL_PACK stats ────────────────────────────────────────────────────
print("\n[6] HYDROGEL_PACK statistics")
print("-" * 45)

hyd     = mid[["elapsed_days", "HYDROGEL_PACK"]].dropna()
hyd_mean = hyd["HYDROGEL_PACK"].mean()
hyd_std  = hyd["HYDROGEL_PACK"].std()
print(f"  Mean : {hyd_mean:.2f}")
print(f"  Std  : {hyd_std:.2f}")
print(f"  Min  : {hyd['HYDROGEL_PACK'].min():.0f}")
print(f"  Max  : {hyd['HYDROGEL_PACK'].max():.0f}")

hyd_rows = prices[
    (prices["product"] == "HYDROGEL_PACK") &
    prices["bid_price_1"].notna() &
    prices["ask_price_1"].notna()
].copy()
if len(hyd_rows):
    hyd_rows["spread"] = hyd_rows["ask_price_1"] - hyd_rows["bid_price_1"]
    print(f"  Avg bid-ask spread : {hyd_rows['spread'].mean():.2f}")
    print(f"  Typical entry band : [{hyd_mean - hyd_std:.0f}, {hyd_mean + hyd_std:.0f}]  (±1σ)")


# ── 7. Counterparty analysis ──────────────────────────────────────────────────
print("\n[7] Counterparty analysis")
print("-" * 45)

def net_position(df: pd.DataFrame, trader: str, symbol: str):
    """Returns (net, bought, sold) quantities for a trader/symbol pair."""
    bought = int(df[(df["buyer"]  == trader) & (df["symbol"] == symbol)]["quantity"].sum())
    sold   = int(df[(df["seller"] == trader) & (df["symbol"] == symbol)]["quantity"].sum())
    return bought - sold, bought, sold

# VEV spot net positions
print("\n  VELVETFRUIT_EXTRACT net positions:")
print(f"  {'Trader':<12} {'Net':>7} {'Bought':>8} {'Sold':>8}  Role")
print(f"  {'-'*48}")
for tr in TRADERS:
    net, b, s = net_position(trades, tr, "VELVETFRUIT_EXTRACT")
    if b + s == 0:
        continue
    role = {
        "Mark 67": "pure buyer — informed long signal",
        "Mark 01": "two-sided, trades exclusively with Mark 55",
        "Mark 55": "neutral market maker",
        "Mark 49": "persistent net seller",
        "Mark 22": "liquidity provider / options dealer",
        "Mark 14": "market maker",
        "Mark 38": "market maker",
    }.get(tr, "")
    print(f"  {tr:<12} {net:>+7d} {b:>8d} {s:>8d}  {role}")

# VEV option net positions
print("\n  VEV option net positions (key traders):")
print(f"  {'Symbol':<12} {'Mark 01':>8} {'Mark 22':>8} {'Mark 14':>8}")
print(f"  {'-'*42}")
for K in STRIKES:
    sym = f"VEV_{K}"
    n01, _, _ = net_position(trades, "Mark 01", sym)
    n22, _, _ = net_position(trades, "Mark 22", sym)
    n14, _, _ = net_position(trades, "Mark 14", sym)
    if abs(n01) + abs(n22) + abs(n14) == 0:
        continue
    print(f"  {sym:<12} {n01:>+8d} {n22:>+8d} {n14:>+8d}")

# Mark 67 signal: forward return after each buy
m67_buys = trades[
    (trades["buyer"]  == "Mark 67") &
    (trades["symbol"] == "VELVETFRUIT_EXTRACT")
].copy()
m67_buys["elapsed_days"] = m67_buys["day"] + m67_buys["timestamp"] / TICKS_PER_DAY

print(f"\n  Mark 67 buy events: {len(m67_buys)}")
print(f"  Avg buy price     : {m67_buys['price'].mean():.2f}")
print(f"  Buy qty per event : {m67_buys['quantity'].mean():.1f} avg")

und_indexed = und.set_index("elapsed_days")["VELVETFRUIT_EXTRACT"]
FORWARD_LAGS = [0.001, 0.005, 0.01]  # elapsed-days forward: ~1k, 5k, 10k ticks

print(f"\n  Forward returns after a Mark 67 buy event:")
print(f"  {'Lag (ticks)':>14} {'Avg return (bps)':>18} {'Hit rate':>10}")
print(f"  {'-'*44}")
for lag in FORWARD_LAGS:
    fwd = []
    for _, row in m67_buys.iterrows():
        t0     = row["elapsed_days"]
        idx    = und_indexed.index
        before = und_indexed[idx <= t0]
        after  = und_indexed[idx >  t0 + lag]
        if len(before) == 0 or len(after) == 0:
            continue
        ret = (after.iloc[0] - before.iloc[-1]) / before.iloc[-1]
        fwd.append(ret)
    if fwd:
        avg_bps  = np.mean(fwd) * 1e4
        hit_rate = np.mean(np.array(fwd) > 0)
        lag_ticks = int(lag * TICKS_PER_DAY)
        print(f"  {lag_ticks:>14,} {avg_bps:>+18.2f} {hit_rate:>9.1%}")

# Mark 49 signal: forward return when selling
m49_sells = trades[
    (trades["seller"] == "Mark 49") &
    (trades["symbol"] == "VELVETFRUIT_EXTRACT")
].copy()
m49_sells["elapsed_days"] = m49_sells["day"] + m49_sells["timestamp"] / TICKS_PER_DAY
print(f"\n  Mark 49 sell events: {len(m49_sells)}")
print(f"  Avg sell price     : {m49_sells['price'].mean():.2f}")

fwd_49 = []
lag = 0.005
for _, row in m49_sells.iterrows():
    t0     = row["elapsed_days"]
    idx    = und_indexed.index
    before = und_indexed[idx <= t0]
    after  = und_indexed[idx >  t0 + lag]
    if len(before) == 0 or len(after) == 0:
        continue
    ret = (after.iloc[0] - before.iloc[-1]) / before.iloc[-1]
    fwd_49.append(ret)
if fwd_49:
    avg_bps = np.mean(fwd_49) * 1e4
    hit_rate = np.mean(np.array(fwd_49) > 0)
    print(f"  Avg VEV return after Mark 49 sell (+5k ticks): {avg_bps:+.2f} bps  hit={hit_rate:.1%}")
    print("  (positive = price goes UP after Mark 49 sells → he is uninformed)")


# ── 8. Save IV summary CSV ────────────────────────────────────────────────────
print("\n[8] Saving IV summary CSV ...")
iv_df.groupby("strike")[["IV", "delta", "mkt_price"]].mean().round(5).to_csv(
    f"{OUTPUT_DIR}/iv_summary_corrected.csv"
)
print(f"  Saved to {OUTPUT_DIR}/iv_summary_corrected.csv")


# ── 9. Plots ──────────────────────────────────────────────────────────────────
print("\n[9] Generating plots ...")
sns.set_theme(style="darkgrid")
PALETTE = sns.color_palette("tab10")

# A. IV smile per day
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for d, ax in enumerate(axes):
    sub = iv_df[iv_df["day"] == d].groupby("strike")["IV"].mean()
    ax.plot(sub.index, sub.values, marker="o", lw=1.5,
            label=f"Day {d}  (TTE≈{8 - d}d)")
    ax.axhline(sigma_atm_mean, ls="--", color="red", lw=1,
               label=f"ATM mean σ={sigma_atm_mean:.4f}")
    ax.set_title(f"Day {d} — TTE ≈ {8 - d} days")
    ax.set_xlabel("Strike")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Implied Volatility")
fig.suptitle("Implied Volatility Smile per Day (correct TTE)", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/A_iv_smile.png", dpi=150)
plt.close()

# B. IV stability over time (near-ATM strikes)
fig, ax = plt.subplots(figsize=(14, 4))
for i, K in enumerate([5000, 5200, 5300, 5400, 5500]):
    sub = iv_df[iv_df["strike"] == K].sort_values("elapsed_days")
    ax.plot(sub["elapsed_days"], sub["IV"], lw=0.7,
            label=f"K={K}", color=PALETTE[i])
ax.axhline(sigma_atm_mean, ls="--", color="black", lw=1,
           label=f"ATM mean σ={sigma_atm_mean:.4f}")
ax.set_title("Implied Volatility over Time — Near-ATM Strikes (correct TTE)")
ax.set_xlabel("Elapsed days")
ax.set_ylabel("Implied Volatility")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/B_iv_over_time.png", dpi=150)
plt.close()

# C. TTE countdown
fig, ax = plt.subplots(figsize=(10, 3))
t = np.linspace(0, 3, 300)
ax.plot(t, 8 - t, color="steelblue", lw=2)
ax.axvline(3, color="red",   ls="--", lw=1.5, label="Round 4 starts (TTE=5)")
ax.axvline(0, color="green", ls="--", lw=1.5, label="Historical data start (TTE=8)")
ax.fill_betweenx([0, 8], 0, 3, alpha=0.08, color="blue",
                 label="Historical data window")
ax.set_xlabel("Elapsed days from options launch")
ax.set_ylabel("TTE (days)")
ax.set_title("Time-to-Expiry Countdown")
ax.legend(fontsize=9)
ax.set_ylim(0, 9)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/C_tte_timeline.png", dpi=150)
plt.close()

# D. HYDROGEL mean-reversion bands
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(hyd["elapsed_days"], hyd["HYDROGEL_PACK"], lw=0.5, color="teal")
for n, c, ls in [(1, "red", "--"), (2, "darkred", ":")]:
    ax.axhline(hyd_mean + n * hyd_std, color=c, lw=0.8, ls=ls, label=f"+{n}σ")
    ax.axhline(hyd_mean - n * hyd_std, color=c, lw=0.8, ls=ls, label=f"-{n}σ")
ax.axhline(hyd_mean, color="black", lw=1, label=f"Mean={hyd_mean:.0f}")
ax.set_title("HYDROGEL_PACK — Mean Reversion Bands")
ax.set_xlabel("Elapsed days")
ax.set_ylabel("Mid price")
ax.legend(fontsize=8, ncol=3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/D_hydrogel.png", dpi=150)
plt.close()

# E. Mark 67 buy signals overlaid on VEV price
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(und["elapsed_days"], und["VELVETFRUIT_EXTRACT"],
        lw=0.5, color="steelblue", label="VEV mid price")
ax.scatter(m67_buys["elapsed_days"], m67_buys["price"],
           color="green", s=8, alpha=0.6, zorder=3, label="Mark 67 buy")
ax.set_title("VELVETFRUIT_EXTRACT — Mark 67 buy signals overlaid")
ax.set_xlabel("Elapsed days")
ax.set_ylabel("Price")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/E_mark67_signal.png", dpi=150)
plt.close()

# F. Net positions bar chart (all traders, VEV spot)
trader_nets = {}
for tr in TRADERS:
    net, _, _ = net_position(trades, tr, "VELVETFRUIT_EXTRACT")
    trader_nets[tr] = net
trader_nets = {k: v for k, v in trader_nets.items() if v != 0}

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#3B9922" if v > 0 else "#C23030" for v in trader_nets.values()]
ax.barh(list(trader_nets.keys()), list(trader_nets.values()), color=colors, height=0.55)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Net VELVETFRUIT_EXTRACT position by counterparty")
ax.set_xlabel("Net quantity (bought − sold)")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/F_net_positions.png", dpi=150)
plt.close()

# G. Option net positions heatmap (Mark 01 vs Mark 22)
opt_nets = {"Mark 01": [], "Mark 22": [], "Mark 14": []}
for K in STRIKES:
    sym = f"VEV_{K}"
    for tr in opt_nets:
        net, _, _ = net_position(trades, tr, sym)
        opt_nets[tr].append(net)

opt_df = pd.DataFrame(opt_nets, index=[f"VEV_{k}" for k in STRIKES])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(opt_df, annot=True, fmt="+.0f", center=0,
            cmap="RdYlGn", linewidths=0.5, ax=ax)
ax.set_title("VEV option net positions by trader")
ax.set_xlabel("")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/G_option_net_heatmap.png", dpi=150)
plt.close()

# H. Round 4 BS fair values vs market prices at end of historical data
last_row = mid.dropna(subset=["VELVETFRUIT_EXTRACT"]).iloc[-1]
S_last   = last_row["VELVETFRUIT_EXTRACT"]
T_last   = last_row["TTE"]

fair_vals = [bs_call(S_last, K, T_last, sigma_atm_mean) for K in STRIKES]
mkt_vals  = [last_row.get(f"VEV_{K}", np.nan) for K in STRIKES]

fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(STRIKES))
w = 0.35
ax.bar(x - w/2, fair_vals, width=w, label=f"BS fair (S={S_last:.0f}, TTE={T_last:.2f}d)",
       color="steelblue", alpha=0.8)
ax.bar(x + w/2, mkt_vals,  width=w, label="Market mid price",
       color="coral", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"VEV_{k}" for k in STRIKES], rotation=30, ha="right")
ax.set_title("BS fair value vs market mid — end of historical data")
ax.set_ylabel("Price")
ax.legend()
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/H_fair_vs_market.png", dpi=150)
plt.close()

print(f"  8 plots saved to {OUTPUT_DIR}/")


# ── 10. Ready-to-paste config ─────────────────────────────────────────────────
print(f"""
{'=' * 65}
  READY-TO-PASTE PARAMS for trader.py
{'=' * 65}

SIGMA_FALLBACK = {sigma_atm_mean:.5f}
# ATM implied vol from historical data (correct TTE, very stable)

HYDROGEL_MEAN  = {hyd_mean:.2f}
HYDROGEL_STD   = {hyd_std:.2f}

# TTE inside the Round 4 trader:
#   TTE = 5.0 - (day_offset + timestamp / 1_000_000)
#   starts at 5.0 when Round 4 begins, hits 0 at expiry

POS_LIMITS = {{
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK"      : 200,
    # VEV_XXXX (each)    : 300
}}

# ── Counterparty signal thresholds ──────────────────────────────
#
# Mark 67 seen as BUYER in market_trades["VELVETFRUIT_EXTRACT"]:
#   → informed long signal
#   → raise target inventory by +30, tighten ask by 1 tick
#
# Mark 49 seen as SELLER in market_trades["VELVETFRUIT_EXTRACT"]:
#   → likely uninformed — take their liquidity aggressively
#   → place buy order at their price for +5 to +10 units
#
# Mark 22 on ASK side of VEV_5300 / VEV_5400 / VEV_5500:
#   → buy up to 50 units (keep delta-neutral with VEV hedge)
#   → Mark 01 does this systematically — follow their lead
#
# Mark 55 on BID or ASK of VELVETFRUIT_EXTRACT:
#   → neutral market maker — safe to cross for inventory mgmt
{'=' * 65}
""")