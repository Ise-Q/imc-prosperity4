"""
Round 3 EDA — IMC Prosperity 4
================================
Products: VELVETFRUIT_EXTRACT, VEV_4000-6500 (call options), HYDROGEL_PACK
Focus: voucher (options) mispricing, HYDROGEL mean-reversion, signal extraction.

Usage:
    python round3_EDA.py               # generates plots + prints stats
    python round3_EDA.py --no-plots    # stats only (faster)
"""

import argparse
import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "./data"
OUT_DIR   = "./eda_output"
DAYS      = [0, 1, 2]
STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_COLS  = [f"VEV_{k}" for k in STRIKES]
PRODUCTS  = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_COLS
TICKS_PER_DAY = 10_000
TOTAL_DAYS    = 3.0


# ── Math helpers ──────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(float(S - K), 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        p = pd.read_csv(f"{DATA_DIR}/prices_round_3_day_{d}.csv", sep=";")
        p["day"] = d
        dfs.append(p)
    prices = pd.concat(dfs, ignore_index=True)
    prices["elapsed"] = prices["day"] + prices["timestamp"] / 1_000_000
    prices["T_remaining"] = TOTAL_DAYS - prices["elapsed"]
    return prices


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


def load_trades() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        t = pd.read_csv(f"{DATA_DIR}/trades_round_3_day_{d}.csv", sep=";")
        t["day"] = d
        dfs.append(t)
    return pd.concat(dfs, ignore_index=True)


# ── Analysis functions ────────────────────────────────────────────────────────

def calibrate_sigma(mid: pd.DataFrame) -> float:
    S = mid["VELVETFRUIT_EXTRACT"].dropna()
    log_ret = np.log(S / S.shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(TICKS_PER_DAY))


def compute_bs_mispricing(mid: pd.DataFrame, sigma: float, sample_every: int = 100) -> pd.DataFrame:
    """Compute BS fair value and mispricing for all strikes at sampled timestamps."""
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
            bs = bs_call(S, K, T, sigma)
            delta = bs_delta(S, K, T, sigma)
            rows.append({
                "elapsed": row["elapsed"],
                "day": row["day"],
                "T_remaining": T,
                "strike": K,
                "S": S,
                "mkt": mkt,
                "bs": bs,
                "misprice": mkt - bs,
                "moneyness": S / K,
                "delta": delta,
            })
    return pd.DataFrame(rows)


def print_mispricing_summary(opts: pd.DataFrame) -> None:
    print("\n=== Mispricing (mkt - BS) by strike and day ===")
    print(f"{'Strike':>8}  {'Day':>4}  {'Mean':>8}  {'P25':>7}  {'P75':>7}  {'Min':>8}  {'Max':>8}")
    for K in STRIKES:
        sub = opts[opts["strike"] == K]
        if sub.empty:
            continue
        for d in DAYS:
            s = sub[sub["day"] == d]["misprice"]
            if s.empty:
                continue
            print(f"  K={K:5d}  day={d}  mean={s.mean():+7.2f}"
                  f"  p25={s.quantile(.25):+6.2f}  p75={s.quantile(.75):+6.2f}"
                  f"  min={s.min():+7.2f}  max={s.max():+7.2f}")


def print_hydrogel_stats(prices: pd.DataFrame) -> None:
    hyd = prices[prices["product"] == "HYDROGEL_PACK"].copy()
    hyd = hyd.sort_values("elapsed")
    s = hyd["mid_price"]
    print("\n=== HYDROGEL_PACK statistics ===")
    print(f"  Mean  = {s.mean():.2f}  Std = {s.std():.2f}")
    print(f"  Min   = {s.min():.0f}   Max = {s.max():.0f}")

    # OU half-life
    vals = s.dropna().values
    dX   = np.diff(vals)
    X    = vals[:-1]
    A    = np.column_stack([np.ones(len(X)), X])
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(A, dX, rcond=None)
    theta = -coeffs[1]
    mu    = coeffs[0] / theta if theta > 0 else s.mean()
    hl    = math.log(2) / theta if theta > 0 else None
    print(f"  OU mu = {mu:.2f}  theta = {theta:.5f}  half-life = {hl:.1f} ticks" if hl else "  No OU fit")

    # Spread
    sub = prices[(prices["product"] == "HYDROGEL_PACK") & prices["bid_price_1"].notna()].copy()
    sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
    print(f"  Avg bid-ask spread = {sub['spread'].mean():.2f}  (most common: {sub['spread'].mode()[0]:.0f})")

    mean, std = s.mean(), s.std()
    print(f"  1-sigma bands: ({mean-std:.0f}, {mean+std:.0f})")
    print(f"  2-sigma bands: ({mean-2*std:.0f}, {mean+2*std:.0f})")
    print(f"  Pct outside 1σ: {((s > mean+std) | (s < mean-std)).mean()*100:.1f}%")
    print(f"  Pct outside 2σ: {((s > mean+2*std) | (s < mean-2*std)).mean()*100:.1f}%")


def print_deepitm_stats(mid: pd.DataFrame) -> None:
    print("\n=== Deep-ITM options: VEV_4000, VEV_4500 vs intrinsic ===")
    for K in [4000, 4500]:
        col = f"VEV_{K}"
        sub = mid[["elapsed", "VELVETFRUIT_EXTRACT", col]].dropna()
        intrinsic = sub["VELVETFRUIT_EXTRACT"] - K
        diff = sub[col] - intrinsic
        print(f"  VEV_{K}: price - (S-K)  mean={diff.mean():.3f}  std={diff.std():.3f}")
        print(f"    Corr(VEV_{K}, S) = {sub['VELVETFRUIT_EXTRACT'].corr(sub[col]):.6f}")


def print_strategy_thresholds(opts: pd.DataFrame, sigma: float) -> None:
    """Compute optimal sell-threshold candidates per strike and day."""
    print("\n=== Suggested SELL thresholds (mkt - BS must exceed X to sell) ===")
    print(f"{'Strike':>8}  {'Day 0':>8}  {'Day 1':>8}  {'Day 2':>8}  "
          f"{'p10 D0':>8}  {'p10 D1':>8}  {'p10 D2':>8}")
    for K in [5000, 5100, 5200, 5300, 5400, 5500]:
        col = f"VEV_{K}"
        sub = opts[opts["strike"] == K]
        line = f"  K={K:5d} "
        p10_vals = []
        mean_vals = []
        for d in DAYS:
            s = sub[sub["day"] == d]["misprice"]
            if s.empty:
                line += f"  {'n/a':>7}"
                p10_vals.append(float("nan"))
                mean_vals.append(float("nan"))
            else:
                line += f"  {s.mean():+7.2f}"
                p10_vals.append(s.quantile(0.10))
                mean_vals.append(s.mean())
        for v in p10_vals:
            line += f"  {v:+7.2f}" if not math.isnan(v) else f"  {'n/a':>7}"
        print(line)
    print("\nRecommendation:")
    print("  Day 0: take_margin=10 (misprice too small/noisy to trade reliably)")
    print("  Day 1: take_margin=5  (consistent +12-13 tick misprice for K=5200-5300)")
    print("  Day 2: take_margin=5  (consistent +27-30 tick misprice — sell aggressively)")
    print("  make_margin: DISABLED (passive quoting causes immediate adverse fills)")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_underlying(mid: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    for d, ax in enumerate(axes):
        sub = mid[mid["day"] == d]
        ax.plot(sub["timestamp"], sub["VELVETFRUIT_EXTRACT"], lw=0.6, color="steelblue")
        ax.set_title(f"Day {d}")
        ax.set_xlabel("Timestamp")
    axes[0].set_ylabel("VELVETFRUIT_EXTRACT mid")
    fig.suptitle("Underlying mid price per day", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_mispricing_by_day(opts: pd.DataFrame, sigma: float, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    key_strikes = [5000, 5100, 5200, 5300, 5400, 5500]
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(key_strikes)))
    for d, ax in enumerate(axes):
        sub = opts[opts["day"] == d]
        for i, K in enumerate(key_strikes):
            s = sub[sub["strike"] == K].sort_values("elapsed")
            if s.empty:
                continue
            ax.plot(s["elapsed"], s["misprice"], lw=0.7, label=f"K={K}", color=colors[i])
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"Day {d}")
        ax.set_xlabel("Elapsed days")
    axes[0].set_ylabel("Mkt - BS price (ticks)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Options mispricing (market - BS)  sigma={sigma:.5f}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_hydrogel(mid: pd.DataFrame, out: str) -> None:
    h = mid[["elapsed", "HYDROGEL_PACK"]].dropna()
    mu  = h["HYDROGEL_PACK"].mean()
    std = h["HYDROGEL_PACK"].std()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(h["elapsed"], h["HYDROGEL_PACK"], lw=0.4, color="teal", label="HYDROGEL")
    for mult, label, c in [(1, "±1σ", "red"), (2, "±2σ", "darkred")]:
        ax.axhline(mu + mult * std, ls="--", color=c, lw=0.8, label=label if mult == 1 else "")
        ax.axhline(mu - mult * std, ls="--", color=c, lw=0.8)
    ax.axhline(mu, ls="-", color="black", lw=1.0, label=f"Mean={mu:.0f}")
    ax.set_title("HYDROGEL_PACK mean-reversion bands")
    ax.set_xlabel("Elapsed days")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_iv_smile(opts: pd.DataFrame, sigma: float, out: str) -> None:
    avg_iv = opts.groupby(["day", "strike"])["misprice"].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for d, ax in enumerate(axes):
        sub = avg_iv[avg_iv["day"] == d]
        ax.bar(sub["strike"], sub["misprice"], color="steelblue", edgecolor="none", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Day {d} — avg misprice by strike")
        ax.set_xlabel("Strike")
    axes[0].set_ylabel("Avg misprice (mkt - BS)")
    fig.suptitle("Options average mispricing per day and strike", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_delta_profile(opts: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    for K in STRIKES:
        sub = opts[opts["strike"] == K].sort_values("elapsed")
        ax.plot(sub["elapsed"], sub["delta"], lw=0.6, label=f"K={K}")
    ax.set_title("BS Delta by strike over time")
    ax.set_xlabel("Elapsed days")
    ax.set_ylabel("Delta (0–1)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


def plot_volatility(mid: pd.DataFrame, sigma: float, out: str) -> None:
    S = mid.set_index("elapsed")["VELVETFRUIT_EXTRACT"].dropna()
    log_ret = np.log(S / S.shift(1)).dropna()
    rolling = log_ret.rolling(TICKS_PER_DAY).std() * np.sqrt(TICKS_PER_DAY)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling.index, rolling, lw=0.7, color="steelblue", label="Rolling sigma")
    ax.axhline(sigma, ls="--", color="tomato", lw=1.2, label=f"Full-sample sigma={sigma:.5f}")
    ax.set_title("Rolling realized volatility (1-day window)")
    ax.set_ylabel("Sigma (comp-annual)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    prices = load_prices()
    mid    = pivot_mid(prices)
    trades = load_trades()
    print(f"  Price rows : {len(prices):,}   Trade rows : {len(trades):,}")

    # ── Sigma calibration ─────────────────────────────────────────────────────
    sigma = calibrate_sigma(mid)
    print(f"\n=== Sigma calibration ===")
    print(f"  Realized sigma (comp-annual) = {sigma:.5f}")
    print(f"  Underlying mean  = {mid['VELVETFRUIT_EXTRACT'].mean():.2f}")
    print(f"  Underlying range = [{mid['VELVETFRUIT_EXTRACT'].min():.0f}, {mid['VELVETFRUIT_EXTRACT'].max():.0f}]")

    # ── HYDROGEL ──────────────────────────────────────────────────────────────
    print_hydrogel_stats(prices)

    # ── Deep-ITM options ──────────────────────────────────────────────────────
    print_deepitm_stats(mid)

    # ── Options mispricing ────────────────────────────────────────────────────
    print("\nComputing BS mispricing (sampling every 100 rows)...")
    opts = compute_bs_mispricing(mid, sigma, sample_every=100)
    print_mispricing_summary(opts)
    print_strategy_thresholds(opts, sigma)

    # ── Hypothesis summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  HYPOTHESIS SUMMARY")
    print("=" * 65)
    print("""
Pattern 1 — Options Systematic Overpricing (HIGH CONFIDENCE)
  Type: Options mispricing / mean reversion to intrinsic at expiry
  Signal: market bid - BS fair > threshold
  Day 0: noisy, threshold=10 (trade sparingly)
  Day 1: consistent +12-13 ticks for K=5200-5300 → threshold=5
  Day 2: consistent +27-30 ticks for K=5200-5300 → threshold=5
  Action: SELL calls when overpriced, hold to expiry (T→0 → intrinsic)
  Risk: underlying directional move; mitigate via delta hedge

Pattern 2 — HYDROGEL Mean Reversion (HIGH CONFIDENCE)
  Type: Mean reversion / Ornstein-Uhlenbeck
  Signal: price deviation from mu=9991 (1-sigma=31.94)
  Half-life ≈ 301 ticks; spread = 16 ticks (cost to cross)
  Action: buy below fair-2, sell above fair+2, clear at fair+4
  Current strategy already profitable (+15,633 baseline)

Pattern 3 — Deep-ITM Options as Underlying Proxies (MEDIUM)
  Type: Delta-1 arbitrage / market-making
  VEV_4000, VEV_4500: correlation(price, S-K) > 0.998
  Mean deviation from (S-K) = 0.01 ticks
  Action: market-make same way as underlying

Pattern 4 — Delta Hedging of Short Option Book (STRUCTURAL)
  Type: Risk management (not alpha source)
  When short calls, buy underlying to neutralize delta exposure
  Net portfolio delta should stay near 0
""")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating plots...")
        plot_underlying(mid, f"{OUT_DIR}/01_underlying.png")
        plot_volatility(mid, sigma, f"{OUT_DIR}/02_volatility.png")
        plot_mispricing_by_day(opts, sigma, f"{OUT_DIR}/03_mispricing_by_day.png")
        plot_hydrogel(mid, f"{OUT_DIR}/04_hydrogel_bands.png")
        plot_iv_smile(opts, sigma, f"{OUT_DIR}/05_mispricing_bar.png")
        plot_delta_profile(opts, f"{OUT_DIR}/06_delta_profile.png")
        print(f"  Plots saved to: {OUT_DIR}/")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary = opts.groupby(["day", "strike"]).agg(
        avg_misprice=("misprice", "mean"),
        std_misprice=("misprice", "std"),
        p10_misprice=("misprice", lambda x: x.quantile(0.10)),
        p90_misprice=("misprice", lambda x: x.quantile(0.90)),
        avg_delta=("delta", "mean"),
        n_obs=("misprice", "count"),
    ).round(3)
    csv_path = f"{OUT_DIR}/mispricing_summary.csv"
    summary.to_csv(csv_path)
    print(f"\nSummary saved to: {csv_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
