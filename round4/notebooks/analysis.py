"""
IMC Prosperity Round 3 — Analysis
===================================
Products:
  VELVETFRUIT_EXTRACT   -> underlying asset (~5250)
  VEV_XXXX              -> call options on the underlying (strikes 4000-6500)
  HYDROGEL_PACK         -> stable mean-reverting product (~10000)

Key calibration insight
-----------------------
In competition time, 1 "year" = 1 competition day (10,000 ticks of 100 each).
Options expire at END of Day 2 (end of the round), so:

    T_remaining = 3.0 - (day + timestamp / 1_000_000)   [competition-days]
    sigma_annual = sigma_per_tick * sqrt(10_000)         [daily vol as "annual" vol]

Calibration check (day 0, tick 0):
    VEV_5400 market ~23 -> BS ~25  (close!)
    VEV_5500 market ~8.5 -> BS ~8  (close!)

Output: analysis_output/ with 9 plots + 2 CSV summaries.
Also prints ready-to-paste PARAMS for Jay's ProductTrader template.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
OUTPUT_DIR = "./analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_FILES = {
    0: f"{DATA_DIR}/prices_round_3_day_0.csv",
    1: f"{DATA_DIR}/prices_round_3_day_1.csv",
    2: f"{DATA_DIR}/prices_round_3_day_2.csv",
}
TRADE_FILES = {
    0: f"{DATA_DIR}/trades_round_3_day_0.csv",
    1: f"{DATA_DIR}/trades_round_3_day_1.csv",
    2: f"{DATA_DIR}/trades_round_3_day_2.csv",
}

STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_COLS     = [f"VEV_{k}" for k in STRIKES]
ALL_PRODUCTS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_COLS

# Competition time: 1 day = 10,000 ticks; options expire end of Day 2
TICKS_PER_DAY    = 10_000
TOTAL_ROUND_DAYS = 3.0
RISK_FREE        = 0.0

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data ...")
price_dfs, trade_dfs = [], []
for day in range(3):
    p = pd.read_csv(PRICE_FILES[day], sep=";"); p["day"] = day
    t = pd.read_csv(TRADE_FILES[day], sep=";"); t["day"] = day
    price_dfs.append(p); trade_dfs.append(t)

prices = pd.concat(price_dfs, ignore_index=True)
trades = pd.concat(trade_dfs, ignore_index=True)

prices["elapsed_days"] = prices["day"] + prices["timestamp"] / 1_000_000
prices["T_remaining"]  = TOTAL_ROUND_DAYS - prices["elapsed_days"]

print(f"  Price rows : {len(prices):,}")
print(f"  Trade rows : {len(trades):,}")

# ── 2. Wide pivot ─────────────────────────────────────────────────────────────
mid = (
    prices[prices["product"].isin(ALL_PRODUCTS)]
    .pivot_table(
        index=["day","timestamp","elapsed_days","T_remaining"],
        columns="product", values="mid_price", aggfunc="first")
    .reset_index().sort_values("elapsed_days")
)
mid.columns.name = None

# ── 3. Underlying ─────────────────────────────────────────────────────────────
print("\n-- VELVETFRUIT_EXTRACT -----------------------------------------------")
und = mid[["elapsed_days","day","timestamp","T_remaining",
           "VELVETFRUIT_EXTRACT"]].dropna()
print(und["VELVETFRUIT_EXTRACT"].describe().round(2).to_string())

# ── 4. Realized vol (competition-calibrated) ──────────────────────────────────
print("\n-- Realized Volatility -----------------------------------------------")
S_series   = und.set_index("elapsed_days")["VELVETFRUIT_EXTRACT"]
log_ret    = np.log(S_series / S_series.shift(1)).dropna()
sigma_tick = log_ret.std()
# In comp-BS: 1 year = 1 competition day = TICKS_PER_DAY ticks
sigma_annual = sigma_tick * np.sqrt(TICKS_PER_DAY)

print(f"  sigma per tick        : {sigma_tick:.6f}")
print(f"  sigma annual (comp)   : {sigma_annual:.5f}  ({sigma_annual*100:.3f}%)")
print(f"  Underlying mean price : {S_series.mean():.2f}")

rolling_sigma = log_ret.rolling(TICKS_PER_DAY).std() * np.sqrt(TICKS_PER_DAY)
print(f"  Rolling sigma range   : [{rolling_sigma.min():.5f}, {rolling_sigma.max():.5f}]")

# ── 5. BS helpers ─────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(float(S - K), 0.0)
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

def bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return float(norm.cdf(d1))

def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return float(norm.pdf(d1) / (S*sigma*np.sqrt(T)))

def implied_vol(mkt, S, K, T, r=0.0):
    intrinsic = max(float(S - K), 0.0)
    if mkt <= intrinsic + 0.01 or T <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - mkt, 1e-4, 5.0, xtol=1e-6)
    except Exception:
        return np.nan

# ── 6. Calibration check ──────────────────────────────────────────────────────
print("\n-- Calibration check (day 0, tick 0) -----------------------------------")
S0  = und[und["day"]==0]["VELVETFRUIT_EXTRACT"].iloc[0]
row0 = mid[(mid["day"]==0) & (mid["timestamp"]==0)].iloc[0] if len(mid[(mid["day"]==0) & (mid["timestamp"]==0)]) else None
print(f"  S = {S0},  T = 3.0 days,  sigma = {sigma_annual:.5f}")
for K in [5000, 5200, 5300, 5400, 5500, 6000]:
    col = f"VEV_{K}"
    mkt_val = row0[col] if row0 is not None else np.nan
    mkt_str = f"{mkt_val:.1f}" if not np.isnan(mkt_val) else "n/a"
    theo    = bs_call(S0, K, 3.0, 0, sigma_annual)
    print(f"    K={K:5d}: market={mkt_str:>7}  BS={theo:>7.2f}")

# ── 7. Full options analysis ──────────────────────────────────────────────────
print("\nRunning options analysis (sampling every 200 rows) ...")
sample = mid.iloc[::200].copy()

opt_rows = []
for _, row in sample.iterrows():
    S = row.get("VELVETFRUIT_EXTRACT", np.nan)
    T = row.get("T_remaining", 0.0)
    if np.isnan(S) or T <= 0:
        continue
    for K in STRIKES:
        col = f"VEV_{K}"
        mkt = row.get(col, np.nan)
        if np.isnan(mkt):
            continue
        theo  = bs_call(S, K, T, RISK_FREE, sigma_annual)
        dlt   = bs_delta(S, K, T, RISK_FREE, sigma_annual)
        gam   = bs_gamma(S, K, T, RISK_FREE, sigma_annual)
        iv    = implied_vol(mkt, S, K, T, RISK_FREE)
        opt_rows.append({
            "elapsed_days": row["elapsed_days"],
            "day"         : row["day"],
            "T_remaining" : T,
            "strike"      : K,
            "S"           : S,
            "mkt_price"   : mkt,
            "bs_price"    : theo,
            "mispricing"  : mkt - theo,
            "implied_vol" : iv,
            "bs_delta"    : dlt,
            "bs_gamma"    : gam,
            "moneyness"   : S / K,
        })

opts = pd.DataFrame(opt_rows)

# ── 8. HYDROGEL analysis ──────────────────────────────────────────────────────
print("\n-- HYDROGEL_PACK -------------------------------------------------------")
hyd = mid[["elapsed_days","day","timestamp","HYDROGEL_PACK"]].dropna()
hyd_mean = hyd["HYDROGEL_PACK"].mean()
hyd_std  = hyd["HYDROGEL_PACK"].std()
hyd_log  = np.log(hyd["HYDROGEL_PACK"] / hyd["HYDROGEL_PACK"].shift(1)).dropna()
hyd_sigma = hyd_log.std() * np.sqrt(TICKS_PER_DAY)
hyd["z"] = (hyd["HYDROGEL_PACK"] - hyd_mean) / hyd_std

print(f"  Mean={hyd_mean:.2f}  Std={hyd_std:.2f}  "
      f"Min={hyd['HYDROGEL_PACK'].min():.0f}  Max={hyd['HYDROGEL_PACK'].max():.0f}")
print(f"  Daily sigma : {hyd_sigma:.5f}")
print(f"  |z|>1 : {(hyd['z'].abs()>1).mean()*100:.1f}%   "
      f"|z|>2 : {(hyd['z'].abs()>2).mean()*100:.1f}%")

hyd_rows = prices[(prices["product"]=="HYDROGEL_PACK") & prices["bid_price_1"].notna()].copy()
if len(hyd_rows):
    hyd_rows["spread"] = hyd_rows["ask_price_1"] - hyd_rows["bid_price_1"]
    print(f"  Avg bid-ask spread : {hyd_rows['spread'].mean():.2f}")

und_rows = prices[(prices["product"]=="VELVETFRUIT_EXTRACT") & prices["bid_price_1"].notna()].copy()
if len(und_rows):
    und_rows["spread"] = und_rows["ask_price_1"] - und_rows["bid_price_1"]
    print(f"\n-- VELVETFRUIT_EXTRACT avg spread: {und_rows['spread'].mean():.2f}")

# ── 9. Plots ──────────────────────────────────────────────────────────────────
print("\nGenerating plots ...")
sns.set_theme(style="darkgrid")
WIDE = (14, 4)

# A. Underlying per day
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
for d, ax in enumerate(axes):
    sub = und[und["day"]==d]
    ax.plot(sub["timestamp"], sub["VELVETFRUIT_EXTRACT"], lw=0.7)
    ax.set_title(f"Day {d}"); ax.set_xlabel("Timestamp")
axes[0].set_ylabel("VELVETFRUIT_EXTRACT mid price")
fig.suptitle("Underlying Mid Price per Day", fontsize=13)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/01_underlying_per_day.png", dpi=150); plt.close()

# B. Log-return distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(log_ret, bins=120, color="steelblue", edgecolor="none", density=True, alpha=0.8)
x = np.linspace(log_ret.min(), log_ret.max(), 300)
ax.plot(x, norm.pdf(x, log_ret.mean(), log_ret.std()), "r-", lw=1.5, label="Normal fit")
ax.set_title("Log-Return Distribution (VELVETFRUIT_EXTRACT)")
ax.set_xlabel("Log return per tick"); ax.legend()
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/02_log_return_dist.png", dpi=150); plt.close()

# C. Rolling realized vol
fig, ax = plt.subplots(figsize=WIDE)
ax.plot(rolling_sigma.index, rolling_sigma, lw=0.8, color="steelblue")
ax.axhline(sigma_annual, ls="--", color="tomato", lw=1.2,
           label=f"Full-sample sigma = {sigma_annual:.5f}")
ax.set_title("Rolling Daily Realized Volatility (1 competition-day window)")
ax.set_ylabel("sigma (comp-annual)"); ax.legend()
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/03_rolling_vol.png", dpi=150); plt.close()

# D. Market vs BS price per strike
fig, axes = plt.subplots(2, 5, figsize=(22, 7))
for i, K in enumerate(STRIKES):
    ax  = axes[i//5][i%5]
    sub = opts[opts["strike"]==K].sort_values("elapsed_days")
    ax.plot(sub["elapsed_days"], sub["mkt_price"], lw=0.7, label="Market", color="steelblue")
    ax.plot(sub["elapsed_days"], sub["bs_price"],  lw=0.7, label="BS",     color="tomato", ls="--")
    ax.set_title(f"VEV_{K}"); ax.legend(fontsize=7); ax.set_xlabel("Days elapsed")
fig.suptitle(f"Market vs BS Price  (sigma={sigma_annual:.4f} per comp-day)", fontsize=13)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/04_mkt_vs_bs.png", dpi=150); plt.close()

# E. Mispricing over time
fig, ax = plt.subplots(figsize=WIDE)
for K in STRIKES:
    sub = opts[opts["strike"]==K].sort_values("elapsed_days")
    ax.plot(sub["elapsed_days"], sub["mispricing"], lw=0.6, label=f"K={K}")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Mispricing = Market - BS  (>0: market overpriced vs BS -> consider SELL)")
ax.set_xlabel("Days elapsed"); ax.set_ylabel("Ticks"); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/05_mispricing.png", dpi=150); plt.close()

# F. IV smile
iv_avg = opts.groupby("strike")["implied_vol"].mean()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iv_avg.index, iv_avg.values, marker="o", color="darkorchid", label="Avg IV")
ax.axhline(sigma_annual, ls="--", color="tomato", lw=1.2,
           label=f"Realized sigma={sigma_annual:.4f}")
ax.set_title("Implied Volatility Smile (averaged across all timestamps)")
ax.set_xlabel("Strike"); ax.set_ylabel("IV (comp-annual)"); ax.legend()
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/06_iv_smile.png", dpi=150); plt.close()

# G. IV over time per strike
fig, ax = plt.subplots(figsize=WIDE)
for K in STRIKES:
    sub = opts[opts["strike"]==K].dropna(subset=["implied_vol"]).sort_values("elapsed_days")
    ax.plot(sub["elapsed_days"], sub["implied_vol"], lw=0.5, label=f"K={K}")
ax.axhline(sigma_annual, ls="--", color="black", lw=1.0, label="Realized sigma")
ax.set_title("Implied Volatility over Time per Strike")
ax.set_xlabel("Days elapsed"); ax.set_ylabel("IV"); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/07_iv_over_time.png", dpi=150); plt.close()

# H. Delta
fig, ax = plt.subplots(figsize=WIDE)
for K in STRIKES:
    sub = opts[opts["strike"]==K].sort_values("elapsed_days")
    ax.plot(sub["elapsed_days"], sub["bs_delta"], lw=0.6, label=f"K={K}")
ax.set_title("BS Delta per Strike  (= hedge ratio: underlying units per option)")
ax.set_xlabel("Days elapsed"); ax.set_ylabel("Delta [0, 1]"); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/08_delta.png", dpi=150); plt.close()

# I. HYDROGEL bands
fig, ax = plt.subplots(figsize=WIDE)
ax.plot(hyd["elapsed_days"], hyd["HYDROGEL_PACK"], lw=0.5, color="teal", label="HYDROGEL")
ax.axhline(hyd_mean,           color="black",   lw=1.0, ls="-",  label=f"Mean={hyd_mean:.0f}")
ax.axhline(hyd_mean+hyd_std,   color="red",     lw=0.8, ls="--", label="+-1 sigma")
ax.axhline(hyd_mean-hyd_std,   color="red",     lw=0.8, ls="--")
ax.axhline(hyd_mean+2*hyd_std, color="darkred", lw=0.6, ls=":",  label="+-2 sigma")
ax.axhline(hyd_mean-2*hyd_std, color="darkred", lw=0.6, ls=":")
ax.set_title("HYDROGEL_PACK — Mean-Reversion Bands")
ax.set_xlabel("Days elapsed"); ax.set_ylabel("Price"); ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/09_hydrogel.png", dpi=150); plt.close()

# ── 10. Save summary tables ───────────────────────────────────────────────────
summary = opts.groupby("strike").agg(
    avg_mkt      = ("mkt_price",   "mean"),
    avg_bs       = ("bs_price",    "mean"),
    avg_misprice = ("mispricing",  "mean"),
    std_misprice = ("mispricing",  "std"),
    avg_iv       = ("implied_vol", "mean"),
    avg_delta    = ("bs_delta",    "mean"),
    avg_gamma    = ("bs_gamma",    "mean"),
).round(4)
summary.to_csv(f"{OUTPUT_DIR}/options_summary.csv")

hyd_stats = pd.Series({
    "mean": round(hyd_mean, 2), "std": round(hyd_std, 2),
    "min": hyd["HYDROGEL_PACK"].min(), "max": hyd["HYDROGEL_PACK"].max(),
    "daily_sigma": round(hyd_sigma, 5),
})
hyd_stats.to_csv(f"{OUTPUT_DIR}/hydrogel_summary.csv", header=False)

# ── 11. Print results ─────────────────────────────────────────────────────────
print("\n-- Options summary table -----------------------------------------------")
print(summary.to_string())

print(f"""
================================================================
  SUGGESTED PARAMS for Jay's ProductTrader template
================================================================

SIGMA        = {sigma_annual:.5f}   # realized comp-daily vol (= "annual" in BS)
EXPIRY_DAYS  = 3.0               # options expire at end of round Day 2

# HYDROGEL_PACK  ->  static fair-value market-making
"HYDROGEL_PACK": {{
    "static_fv"         : {int(round(hyd_mean))},
    "fv_method_weights" : [1, 0],      # pure static (no EMA drift)
    "take_margin"       : 2,           # snipe within 2 ticks of fair
    "clear_margin"      : 4,
    "make_margin"       : 3,
    "ema_configs"       : {{"mid_ema": {{"alpha": 0.03, "source": "mid"}}}},
}}

# VEV_XXXX options  ->  Black-Scholes fair value, compute each tick:
#   T_remaining = EXPIRY_DAYS - (day_offset + timestamp / 1_000_000)
#   fair_value  = bs_call(S, K, T_remaining, 0, SIGMA)
# where S = VELVETFRUIT_EXTRACT mid price.
VEV_TAKE_MARGIN = 3   # ticks — only trade when |mkt - BS| > this
VEV_MAKE_MARGIN = 5   # ticks — quote this wide around BS fair value
================================================================
""")

print(f"All outputs saved to: {OUTPUT_DIR}")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {fn}")