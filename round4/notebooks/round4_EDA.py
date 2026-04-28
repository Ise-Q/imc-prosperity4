"""
Round 4 EDA — IMC Prosperity 4
================================
Products: VELVETFRUIT_EXTRACT, VEV_4000-6500 (call options), HYDROGEL_PACK
Round 4 new feature: buyer/seller IDs on market trades.

Usage:
    python round4_EDA.py               # full analysis + plots
    python round4_EDA.py --no-plots    # stats only
"""

import argparse
import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "../data"
OUT_DIR       = "../eda_output_r4"
DAYS          = [1, 2, 3]   # Round 4 uses days 1,2,3 (vs 0,1,2 in R3)
STRIKES       = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_COLS      = [f"VEV_{k}" for k in STRIKES]
PRODUCTS      = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_COLS
TICKS_PER_DAY = 10_000
TOTAL_DAYS    = 3.0
PARTICIPANTS  = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]


# ── Math helpers ──────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(float(S - K), 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        p = pd.read_csv(f"{DATA_DIR}/prices_round_4_day_{d}.csv", sep=";")
        p["day"] = d
        dfs.append(p)
    prices = pd.concat(dfs, ignore_index=True)
    # elapsed: day 1 tick 0 → elapsed=1.0, day 3 tick 1e6 → elapsed=4.0
    # T_remaining: options expire at T=0, which is elapsed=3.0 (end of day 2)
    prices["elapsed"]     = prices["day"] + prices["timestamp"] / 1_000_000
    prices["T_remaining"] = TOTAL_DAYS - (prices["day"] - 1 + prices["timestamp"] / 1_000_000)
    return prices


def load_trades() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        t = pd.read_csv(f"{DATA_DIR}/trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        dfs.append(t)
    trades = pd.concat(dfs, ignore_index=True)
    trades["elapsed"] = trades["day"] + trades["timestamp"] / 1_000_000
    return trades


def pivot_mid(prices: pd.DataFrame) -> pd.DataFrame:
    mid = (
        prices[prices["product"].isin(PRODUCTS)]
        .pivot_table(
            index=["day", "timestamp", "elapsed", "T_remaining"],
            columns="product",
            values="mid_price",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("elapsed")
    )
    mid.columns.name = None
    return mid


# ── Counterparty analysis ─────────────────────────────────────────────────────

def counterparty_price_impact(trades: pd.DataFrame, prices: pd.DataFrame,
                               symbol: str, lag: int = 1000) -> pd.DataFrame:
    """
    Compute price impact at +lag ticks for each participant trading symbol.
    Positive = profitable direction (bought before price rose, or sold before fell).
    """
    prod_t = trades[trades["symbol"] == symbol].copy()
    prod_p = prices[prices["product"] == symbol][["day", "timestamp", "mid_price"]].copy()

    prod_t = prod_t.merge(prod_p.rename(columns={"mid_price": "mid_at"}),
                          on=["day", "timestamp"], how="left")

    future = prod_p.copy()
    future["timestamp"] = future["timestamp"] - lag
    future = future.rename(columns={"mid_price": f"mid_{lag}"})
    prod_t = prod_t.merge(future, on=["day", "timestamp"], how="left")

    rows = []
    for p in PARTICIPANTS:
        b = prod_t[prod_t["buyer"] == p]
        s = prod_t[prod_t["seller"] == p]
        b_impact = (b[f"mid_{lag}"] - b["price"]).dropna()
        s_impact = (s["price"] - s[f"mid_{lag}"]).dropna()
        combined = pd.concat([b_impact, s_impact])
        if combined.empty:
            continue
        rows.append({
            "participant": p,
            "n_trades":   len(combined),
            "avg_impact": combined.mean(),
            "win_rate":   (combined > 0).mean(),
            "n_buys":     len(b),
            "n_sells":    len(s),
            "avg_buy_price":  b["price"].mean() if len(b) else float("nan"),
            "avg_sell_price": s["price"].mean() if len(s) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("participant")


def print_counterparty_summary(trades: pd.DataFrame, prices: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  COUNTERPARTY ANALYSIS — Round 4")
    print("=" * 70)

    # Overall role
    print("\n=== Role summary (all products) ===")
    print(f"{'':12} {'buy_trades':>11} {'sell_trades':>12} {'qty_bought':>12} {'qty_sold':>10}")
    for p in PARTICIPANTS:
        b = trades[trades["buyer"]  == p]
        s = trades[trades["seller"] == p]
        print(f"  {p:<10} {len(b):>11} {len(s):>12} {b['quantity'].sum():>12.0f} {s['quantity'].sum():>10.0f}")

    # Price impact per product
    for prod, lag in [("VELVETFRUIT_EXTRACT", 1000), ("HYDROGEL_PACK", 1000)]:
        df = counterparty_price_impact(trades, prices, prod, lag)
        if df.empty:
            continue
        print(f"\n=== {prod} price impact at +{lag} ticks ===")
        print(f"{'':12} {'n_trades':>9} {'avg_impact':>11} {'win_rate':>10}")
        for p, row in df.iterrows():
            print(f"  {p:<10} {row['n_trades']:>9.0f} {row['avg_impact']:>+11.2f} {row['win_rate']*100:>9.0f}%")

    # HYDROGEL pair analysis
    hyd = trades[trades["symbol"] == "HYDROGEL_PACK"]
    print("\n=== HYDROGEL — who trades with whom? ===")
    pairs = hyd.groupby(["buyer", "seller"])["quantity"].sum().reset_index()
    print(pairs.sort_values("quantity", ascending=False).to_string(index=False))

    print("\n=== Counterparty behavioral archetypes ===")
    print("  Mark 01 : INFORMED BUYER — VEV underlying (83% win), systematic call buyer")
    print("  Mark 14 : PROFITABLE MM  — HYDROGEL buys FV-8/sells FV+8 (90% win)")
    print("  Mark 22 : OPTION SELLER  — systematic call seller (our competitor)")
    print("  Mark 38 : SYSTEMATIC LOSER — HYDROGEL buys FV+8/sells FV-8 (9% win) → FADE")
    print("  Mark 49 : NOISE TRADER   — VEV, 33% win rate → FADE")
    print("  Mark 55 : LIQUIDITY      — VEV, ~50/50, slight adverse to informed")
    print("  Mark 67 : DIRECTIONAL    — pure buyer VEV (never sells), bullish momentum")

    # Options flow
    print("\n=== Option call buyers (who is net long calls?) ===")
    opt_t = trades[trades["symbol"].str.startswith("VEV_")]
    opt_buyers = opt_t.groupby(["buyer", "symbol"])["quantity"].sum().unstack(fill_value=0)
    print(opt_buyers.to_string())


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_sigma(mid: pd.DataFrame) -> float:
    S = mid["VELVETFRUIT_EXTRACT"].dropna()
    log_ret = np.log(S / S.shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(TICKS_PER_DAY))


def compute_mispricing(mid: pd.DataFrame, sigma: float, sample_every: int = 100) -> pd.DataFrame:
    sample = mid.iloc[::sample_every].copy()
    rows = []
    for _, row in sample.iterrows():
        S = row.get("VELVETFRUIT_EXTRACT", float("nan"))
        T = row.get("T_remaining", 0.0)
        if math.isnan(S) or T <= 0:
            continue
        for K in STRIKES:
            col = f"VEV_{K}"
            mkt = row.get(col, float("nan"))
            if math.isnan(mkt):
                continue
            fair  = bs_call(S, K, T, sigma)
            delta = bs_delta(S, K, T, sigma)
            rows.append({
                "elapsed":     row["elapsed"],
                "day":         row["day"],
                "T_remaining": T,
                "strike":      K,
                "S":           S,
                "mkt":         mkt,
                "bs":          fair,
                "misprice":    mkt - fair,
                "delta":       delta,
            })
    return pd.DataFrame(rows)


def print_mispricing_summary(opts: pd.DataFrame) -> None:
    print("\n=== Options mispricing (mkt - BS) by strike and day ===")
    print(f"{'Strike':>8}  {'Day':>4}  {'Mean':>8}  {'P25':>7}  {'P75':>7}")
    for K in STRIKES:
        sub = opts[opts["strike"] == K]
        if sub.empty:
            continue
        for d in DAYS:
            s = sub[sub["day"] == d]["misprice"]
            if s.empty:
                continue
            print(f"  K={K:5d}  day={d}  mean={s.mean():+7.2f}"
                  f"  p25={s.quantile(.25):+6.2f}  p75={s.quantile(.75):+6.2f}")


def print_hydrogel_stats(prices: pd.DataFrame) -> None:
    hyd = prices[prices["product"] == "HYDROGEL_PACK"]["mid_price"]
    print("\n=== HYDROGEL_PACK statistics ===")
    print(f"  Mean = {hyd.mean():.2f}  Std = {hyd.std():.2f}")
    print(f"  Range = [{hyd.min():.0f}, {hyd.max():.0f}]")

    vals = hyd.dropna().values
    dX   = np.diff(vals)
    X    = vals[:-1]
    A    = np.column_stack([np.ones(len(X)), X])
    from numpy.linalg import lstsq
    c, _, _, _ = lstsq(A, dX, rcond=None)
    theta = -c[1]
    mu    = c[0] / theta if theta > 0 else hyd.mean()
    hl    = math.log(2) / theta if theta > 0 else None
    print(f"  OU: mu={mu:.2f}, theta={theta:.5f}, half-life={hl:.1f} ticks" if hl else "  No OU fit")

    sub = prices[(prices["product"] == "HYDROGEL_PACK") & prices["bid_price_1"].notna()].copy()
    sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
    print(f"  Avg spread = {sub['spread'].mean():.2f}  (mode: {sub['spread'].mode()[0]:.0f})")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_counterparty_impact(trades: pd.DataFrame, prices: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (prod, lag) in zip(axes, [("VELVETFRUIT_EXTRACT", 1000), ("HYDROGEL_PACK", 1000)]):
        df = counterparty_price_impact(trades, prices, prod, lag)
        if df.empty:
            continue
        df_plot = df.sort_values("avg_impact", ascending=False)
        colors = ["steelblue" if x >= 0 else "tomato" for x in df_plot["avg_impact"]]
        ax.barh(df_plot.index, df_plot["avg_impact"], color=colors, edgecolor="none")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{prod}\nPrice impact at +{lag} ticks")
        ax.set_xlabel("Avg profit per trade (ticks)")

    fig.suptitle("Counterparty Price Impact — Round 4", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_mispricing(opts: pd.DataFrame, sigma: float, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    key_strikes = [5200, 5300, 5400, 5500]
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(key_strikes)))
    for d, ax in zip(DAYS, axes):
        sub = opts[opts["day"] == d]
        for i, K in enumerate(key_strikes):
            s = sub[sub["strike"] == K].sort_values("elapsed")
            if s.empty:
                continue
            ax.plot(s["elapsed"], s["misprice"], lw=0.7, label=f"K={K}", color=colors[i])
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"Day {d}")
        ax.set_xlabel("Elapsed")
    axes[0].set_ylabel("Mkt - BS (ticks)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Options mispricing  sigma={sigma:.5f}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_hydrogel(mid: pd.DataFrame, out: str) -> None:
    h = mid[["elapsed", "HYDROGEL_PACK"]].dropna()
    mu  = h["HYDROGEL_PACK"].mean()
    std = h["HYDROGEL_PACK"].std()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(h["elapsed"], h["HYDROGEL_PACK"], lw=0.4, color="teal")
    for mult, c in [(1, "red"), (2, "darkred")]:
        ax.axhline(mu + mult * std, ls="--", color=c, lw=0.8)
        ax.axhline(mu - mult * std, ls="--", color=c, lw=0.8)
    ax.axhline(mu, ls="-", color="black", lw=1.0, label=f"Mean={mu:.0f}")
    ax.set_title("HYDROGEL_PACK mean-reversion bands")
    ax.set_xlabel("Elapsed"); ax.set_ylabel("Price"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Round 4 data...")
    prices = load_prices()
    mid    = pivot_mid(prices)
    trades = load_trades()
    print(f"  Price rows : {len(prices):,}   Trade rows : {len(trades):,}")

    sigma = calibrate_sigma(mid)
    print(f"\n=== Sigma calibration ===")
    print(f"  Round 4 sigma = {sigma:.5f}  (Round 3 was 0.02155)")
    print(f"  VEV mean = {mid['VELVETFRUIT_EXTRACT'].mean():.2f}")

    print_hydrogel_stats(prices)

    print_counterparty_summary(trades, prices)

    print("\nComputing BS mispricing...")
    opts = compute_mispricing(mid, sigma, sample_every=100)
    print_mispricing_summary(opts)

    if not args.no_plots:
        print("\nGenerating plots...")
        plot_counterparty_impact(trades, prices, f"{OUT_DIR}/01_counterparty_impact.png")
        plot_mispricing(opts, sigma, f"{OUT_DIR}/02_mispricing.png")
        plot_hydrogel(mid, f"{OUT_DIR}/03_hydrogel.png")
        print(f"  Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
