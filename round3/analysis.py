"""
IMC Prosperity 4 — Round 3 Analysis  (corrected TTE)
======================================================
Products:
  VELVETFRUIT_EXTRACT        — underlying, ~5250
  VEV_4000 … VEV_6500        — call options (10 strikes)
  HYDROGEL_PACK              — mean-reverting, ~9991

TTE (time-to-expiry) mapping — confirmed from Round 3 rules:
  Historical Day 0  →  TTE = 8 days  (tutorial round)
  Historical Day 1  →  TTE = 7 days  (Round 1)
  Historical Day 2  →  TTE = 6 days  (Round 2)
  Round 3 start     →  TTE = 5 days  ← what we submit into

  Formula for historical data:
      TTE = 8 - day - timestamp / 1_000_000

  Formula inside the Round 3 trader:
      TTE = 5 - (day_offset + timestamp / 1_000_000)

Calibration result (from historical data with correct TTE):
  sigma ≈ 0.01255  per competition-day  (1 "year" = 1 comp-day)
  This is extremely stable: IV ∈ [0.01252, 0.01266] across ATM strikes.

Position limits (from official Round 3 rules):
  HYDROGEL_PACK              → 200
  VELVETFRUIT_EXTRACT        → 200
  VEV_XXXX (each voucher)    → 300
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

TICKS_PER_DAY = 10_000

# ── BS helpers (pure math, no scipy) ─────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
    return float(S*norm_cdf(d1) - K*norm_cdf(d2 := d1 - sigma*math.sqrt(T)))

def bs_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    d1 = (math.log(S/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
    return float(norm_cdf(d1))

def implied_vol(mkt, S, K, T, lo=1e-4, hi=3.0):
    intrinsic = max(S - K, 0.0)
    if mkt <= intrinsic + 0.01 or T <= 0: return np.nan
    if bs_call(S, K, T, hi) < mkt: return np.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_call(S, K, T, mid) < mkt: lo = mid
        else: hi = mid
    return (lo + hi) / 2

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
price_dfs, trade_dfs = [], []
for day in range(3):
    p = pd.read_csv(PRICE_FILES[day], sep=";"); p["day"] = day
    t = pd.read_csv(TRADE_FILES[day], sep=";"); t["day"] = day
    price_dfs.append(p); trade_dfs.append(t)
prices = pd.concat(price_dfs, ignore_index=True)
trades = pd.concat(trade_dfs, ignore_index=True)

# CORRECT TTE formula
prices["elapsed_days"] = prices["day"] + prices["timestamp"] / 1_000_000
prices["TTE"]          = 8.0 - prices["elapsed_days"]   # 8 at day-0 start, 5 at day-2 end
print(f"  TTE range in historical data: [{prices['TTE'].min():.2f}, {prices['TTE'].max():.2f}]")

# ── 2. Pivot to wide ──────────────────────────────────────────────────────────
mid = (
    prices[prices["product"].isin(ALL_PRODUCTS)]
    .pivot_table(index=["day","timestamp","elapsed_days","TTE"],
                 columns="product", values="mid_price", aggfunc="first")
    .reset_index().sort_values("elapsed_days")
)
mid.columns.name = None

# ── 3. Underlying stats ───────────────────────────────────────────────────────
print("\n── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────")
und = mid[["elapsed_days","day","TTE","VELVETFRUIT_EXTRACT"]].dropna()
print(und["VELVETFRUIT_EXTRACT"].describe().round(2).to_string())

S_series  = und.set_index("elapsed_days")["VELVETFRUIT_EXTRACT"]
log_ret   = np.log(S_series / S_series.shift(1)).dropna()
sigma_realized_per_day = log_ret.std() * np.sqrt(TICKS_PER_DAY)
print(f"\n  Realized sigma per comp-day : {sigma_realized_per_day:.5f}")

# ── 4. Implied vol calibration with correct TTE ───────────────────────────────
print("\n── Implied Vol Calibration (correct TTE) ─────────────────────────────")
sample = mid.iloc[::100].copy()

iv_rows = []
for _, row in sample.iterrows():
    S   = row.get("VELVETFRUIT_EXTRACT", np.nan)
    TTE = row.get("TTE", 0.0)
    if np.isnan(S) or TTE <= 0: continue
    for K in STRIKES:
        col = f"VEV_{K}"
        mkt = row.get(col, np.nan)
        if np.isnan(mkt): continue
        iv   = implied_vol(mkt, S, K, TTE)
        dlt  = bs_delta(S, K, TTE, iv) if not np.isnan(iv) else np.nan
        mono = S / K
        iv_rows.append({
            "elapsed_days": row["elapsed_days"],
            "day"         : row["day"],
            "TTE"         : TTE,
            "strike"      : K,
            "S"           : S,
            "mkt_price"   : mkt,
            "IV"          : iv,
            "delta"       : dlt,
            "moneyness"   : mono,
        })

iv_df = pd.DataFrame(iv_rows)

# Focus on near-ATM strikes for reliable calibration (moneyness 0.9–1.1)
atm = iv_df[(iv_df["moneyness"] > 0.9) & (iv_df["moneyness"] < 1.1) & iv_df["IV"].notna()]
sigma_atm_mean = atm["IV"].mean()
sigma_atm_std  = atm["IV"].std()

print(f"  ATM IV mean : {sigma_atm_mean:.5f}")
print(f"  ATM IV std  : {sigma_atm_std:.5f}  (extremely stable → fixed sigma is reliable)")
print(f"  Realized σ  : {sigma_realized_per_day:.5f}")

print("\n  Per-strike avg IV:")
print(iv_df.groupby("strike")["IV"].mean().round(5).to_string())

# ── 5. BS fair value check for Round 3 (TTE=5) ───────────────────────────────
print(f"\n── Round 3 Fair Values (TTE=5, S=5250, sigma={sigma_atm_mean:.5f}) ──")
S0, T_r3 = 5250.0, 5.0
for K in STRIKES:
    fv = bs_call(S0, K, T_r3, sigma_atm_mean)
    d  = bs_delta(S0, K, T_r3, sigma_atm_mean)
    print(f"  VEV_{K:5d}: BS fair = {fv:8.2f}   delta = {d:.4f}")

# ── 6. HYDROGEL stats ─────────────────────────────────────────────────────────
print("\n── HYDROGEL_PACK ─────────────────────────────────────────────────────")
hyd = mid[["elapsed_days","HYDROGEL_PACK"]].dropna()
hyd_mean = hyd["HYDROGEL_PACK"].mean()
hyd_std  = hyd["HYDROGEL_PACK"].std()
print(f"  Mean={hyd_mean:.2f}  Std={hyd_std:.2f}  "
      f"Min={hyd['HYDROGEL_PACK'].min():.0f}  Max={hyd['HYDROGEL_PACK'].max():.0f}")
hyd_rows = prices[(prices["product"]=="HYDROGEL_PACK") & prices["bid_price_1"].notna()].copy()
if len(hyd_rows):
    hyd_rows["spread"] = hyd_rows["ask_price_1"] - hyd_rows["bid_price_1"]
    print(f"  Avg bid-ask spread : {hyd_rows['spread'].mean():.2f}")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
print("\nGenerating plots ...")
sns.set_theme(style="darkgrid")

# A. IV smile per day (now with correct TTE)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for d, ax in enumerate(axes):
    sub = iv_df[iv_df["day"]==d].groupby("strike")["IV"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=f"Day {d}  (TTE≈{8-d}d)")
    ax.axhline(sigma_atm_mean, ls="--", color="red", lw=1, label=f"ATM mean={sigma_atm_mean:.4f}")
    ax.set_title(f"Day {d} — TTE≈{8-d} days")
    ax.set_xlabel("Strike"); ax.legend(fontsize=8)
axes[0].set_ylabel("Implied Vol")
fig.suptitle("Implied Volatility Smile per Day (correct TTE)", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/A_iv_smile_corrected.png", dpi=150); plt.close()

# B. IV of ATM options over time (stability check)
fig, ax = plt.subplots(figsize=(14, 4))
for K in [5000, 5200, 5300, 5400, 5500]:
    sub = iv_df[iv_df["strike"]==K].sort_values("elapsed_days")
    ax.plot(sub["elapsed_days"], sub["IV"], lw=0.6, label=f"K={K}")
ax.axhline(sigma_atm_mean, ls="--", color="black", lw=1, label=f"ATM mean σ={sigma_atm_mean:.4f}")
ax.set_title("Implied Volatility over Time — Near-ATM Strikes  (correct TTE)")
ax.set_xlabel("Elapsed days (0=start of historical data, 3=start of Round 3)")
ax.set_ylabel("IV"); ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/B_iv_over_time_corrected.png", dpi=150); plt.close()

# C. TTE decay illustration
fig, ax = plt.subplots(figsize=(10, 3))
t = np.linspace(0, 3, 300)
ax.plot(t, 8 - t, color="steelblue", lw=2)
ax.axvline(3, color="red", ls="--", lw=1.5, label="Round 3 starts (TTE=5)")
ax.axvline(0, color="green", ls="--", lw=1.5, label="Historical data start (TTE=8)")
ax.fill_betweenx([0, 8], 0, 3, alpha=0.08, color="blue", label="Historical data window")
ax.set_xlabel("Elapsed days from options launch"); ax.set_ylabel("TTE (days)")
ax.set_title("Time-to-Expiry Countdown")
ax.legend(fontsize=9); ax.set_ylim(0, 9)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/C_tte_timeline.png", dpi=150); plt.close()

# D. HYDROGEL mean-reversion
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(hyd["elapsed_days"], hyd["HYDROGEL_PACK"], lw=0.5, color="teal")
for n, c, ls in [(1,"red","--"),(2,"darkred",":")]:
    ax.axhline(hyd_mean + n*hyd_std, color=c, lw=0.8, ls=ls, label=f"+{n}σ")
    ax.axhline(hyd_mean - n*hyd_std, color=c, lw=0.8, ls=ls, label=f"-{n}σ")
ax.axhline(hyd_mean, color="black", lw=1, label=f"Mean={hyd_mean:.0f}")
ax.set_title("HYDROGEL_PACK — Mean Reversion Bands"); ax.legend(fontsize=8, ncol=3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/D_hydrogel.png", dpi=150); plt.close()

# ── 8. Save CSVs ──────────────────────────────────────────────────────────────
iv_df.groupby("strike")[["IV","delta","mkt_price"]].mean().round(5).to_csv(
    f"{OUTPUT_DIR}/iv_summary_corrected.csv")

# ── 9. Print ready-to-paste config ───────────────────────────────────────────
print(f"""
================================================================
  CORRECTED PARAMS for trader.py
================================================================

SIGMA_FALLBACK = {sigma_atm_mean:.5f}  # ATM implied vol from historical data
                                        # (with correct TTE — very stable)

# In the Round 3 TRADER, time-to-expiry is:
#   TTE = 5.0 - (day_offset + timestamp / 1_000_000)
#   (starts at 5 when Round 3 begins, counts down to 0 at expiry)

# HYDROGEL_PACK
HYDROGEL_MEAN  = {hyd_mean:.2f}
HYDROGEL_STD   = {hyd_std:.2f}

# Position limits (confirmed from Round 3 rules)
POS_LIMITS = {{
    "VELVETFRUIT_EXTRACT" : 200,
    "HYDROGEL_PACK"       : 200,
    # each VEV_XXXX        : 300
}}
================================================================
""")

print("Plots saved:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png") and f[0] in "ABCD":
        print(f"  {f}")