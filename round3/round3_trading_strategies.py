"""
Round 3 Trading Strategies
===========================
EDA-grounded modular strategy classes for IMC Prosperity Round 3.

Architecture:
  ProductTrader (base) — defined in Trader.py
  ↳ HydrogelTrader    — mean-reversion, take/clear/make around mu=9991
  ↳ VEVOptionSeller   — TAKE-ONLY options seller for overpriced strikes
  ↳ DeltaHedger       — buys/sells underlying to maintain delta-neutral book
  ↳ DeepITMTrader     — disabled (delta-1 instruments, no option premium)

Key findings from EDA (see round3_EDA.py for full analysis):
─────────────────────────────────────────────────────────────
Pattern 1 — Options Systematic Overpricing
  Market bids consistently exceed BS fair value from day 1 onwards:
    K=5200: +12 ticks day 1, +27 ticks day 2
    K=5300: +13 ticks day 1, +30 ticks day 2
  SELL when market_bid > BS_fair + take_margin (TAKE-ONLY, no passive making).
  Passive making caused immediate adverse fills at sub-market prices; disabled.

Pattern 2 — Delta Risk (CRITICAL FIX from baseline)
  Baseline Trader.py: lost -179k on VELVETFRUIT_EXTRACT + -30k on options.
  Cause: short calls without delta hedge = effectively short the underlying.
  Fix: buy underlying = sum(short_qty × delta) per option position.

Pattern 3 — HYDROGEL Mean Reversion
  OU half-life = 301 ticks, mu = 9991, sigma = 32.
  Current take/clear/make strategy earns +15,633. Kept unchanged.

Strikes to trade:
  VEV_5200, VEV_5300 — primary (large, consistent mispricing)
  VEV_5400, VEV_5500 — secondary (smaller mispricing, less directional risk)
  VEV_4000, VEV_4500 — SKIP (delta-1, no real option premium, pure underlying exposure)
  VEV_5000, VEV_5100 — SKIP (delta ~0.97, small misprice doesn't justify risk)
  VEV_6000, VEV_6500 — SKIP (market price = 0.5 minimum, no tradeable premium)
"""

import math
from typing import Dict, List, Optional, Tuple


# ── Black-Scholes helpers (no scipy dependency) ──────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price with competition calibration."""
    if T <= 0 or sigma <= 0:
        return max(float(S - K), 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS delta = N(d1), the hedge ratio for one call unit."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


# ── Strategy parameters ───────────────────────────────────────────────────────

SIGMA        = 0.02155   # calibrated realized volatility (comp-annual)
EXPIRY_DAYS  = 3.0       # options expire at end of Day 2

# Strikes we will actively trade (sell)
SELL_STRIKES = [5200, 5300, 5400, 5500]

# Per-option position limits (conservative to manage delta exposure)
# Deep-in-money strikes (5200-5300) have higher edge but more delta risk.
OPTION_MAX_SHORT: Dict[int, int] = {
    5200: 50,
    5300: 50,
    5400: 75,
    5500: 100,
}

# Take threshold by time regime (mkt_bid - BS_fair > threshold to SELL)
# Day 0 (T > 2.0): mispricing is small/noisy — conservative threshold
# Day 1+ (T ≤ 2.0): mispricing is large and consistent — aggressive threshold
def take_threshold(T_remaining: float, strike: int) -> float:
    if T_remaining > 2.0:
        return 10.0   # day 0: don't trade unless very obvious
    return 5.0        # day 1+: sell when misprice > 5 ticks


# ── VEVOptionSeller ───────────────────────────────────────────────────────────

class VEVOptionSeller:
    """
    TAKE-ONLY options seller. No passive market-making.

    For each tick:
      SELL: when market bid > BS_fair + threshold (option overpriced)
      BUY:  when market ask < BS_fair - threshold (option underpriced / cover)

    No passive MAKE orders — baseline showed immediate adverse fills from
    placing make orders at BS ± 5 when BS is below market mid.
    """

    def __init__(
        self,
        name: str,
        strike: int,
        state,
        last_td: dict,
        new_td: dict,
        underlying_mid: Optional[float],
        T_remaining: float,
        position_limit: int,
    ):
        self.name     = name
        self.strike   = strike
        self.lim      = position_limit
        self.T        = T_remaining
        self.S        = underlying_mid
        self.orders: List = []

        self.pos = state.position.get(name, 0)
        self.max_buy  = self.lim - self.pos
        self.max_sell = self.lim + self.pos   # positive = how much we can still short

        od = state.order_depths.get(name)
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}
        if od:
            self.bids = {p: abs(v) for p, v in
                         sorted(od.buy_orders.items(),  key=lambda x: -x[0])
                         if v != 0}
            self.asks = {p: abs(v) for p, v in
                         sorted(od.sell_orders.items(), key=lambda x:  x[0])
                         if v != 0}

        # BS fair value
        self.fair: Optional[float] = None
        self.delta: float = 0.0
        if self.S is not None and self.T > 0:
            self.fair  = bs_call(self.S, self.strike, self.T, SIGMA)
            self.delta = bs_delta(self.S, self.strike, self.T, SIGMA)
        elif self.S is not None:
            # At expiry: intrinsic
            self.fair  = max(self.S - self.strike, 0.0)
            self.delta = 1.0 if self.S > self.strike else 0.0

        self.threshold = take_threshold(T_remaining, strike)
        self.max_option_short = OPTION_MAX_SHORT.get(strike, 50)

    def _sell(self, price: int, volume: int) -> None:
        from datamodel import Order
        vol = min(volume, self.max_sell, max(0, self.max_option_short + self.pos))
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, -vol))
        self.max_sell -= vol
        self.pos -= vol

    def _buy(self, price: int, volume: int) -> None:
        from datamodel import Order
        # Only buy to cover existing short positions — never initiate a long
        cover_room = max(0, -self.pos)
        vol = min(volume, self.max_buy, cover_room)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, vol))
        self.max_buy -= vol
        self.pos += vol

    def get_orders(self) -> Dict[str, List]:
        if self.fair is None:
            return {self.name: []}

        fv = self.fair

        # SELL: market bid > BS + threshold (option is overpriced)
        for bp, bv in self.bids.items():
            if bp < fv + self.threshold:
                break
            self._sell(bp, bv)

        # BUY back: market ask < BS - threshold (option is underpriced / cover shorts)
        for sp, sv in self.asks.items():
            if sp > fv - self.threshold:
                break
            self._buy(sp, sv)

        return {self.name: self.orders}


# ── DeltaHedger ───────────────────────────────────────────────────────────────

class DeltaHedger:
    """
    Buys/sells VELVETFRUIT_EXTRACT to maintain a delta-neutral options book.

    Target position = round(sum over all option positions of: pos_i * delta_i)
    where pos_i < 0 for short calls (negative delta from short calls).

    We BUY underlying when target > current (short calls increase underlying need).
    We SELL underlying when target < current.

    Rebalancing is aggressive (TAKE orders against the best available quote)
    so the hedge stays close to target even as S and T change each tick.
    """

    def __init__(
        self,
        state,
        last_td: dict,
        new_td: dict,
        option_positions: Dict[int, int],   # {strike: position}
        option_deltas:    Dict[int, float],  # {strike: delta}
        position_limit: int = 600,
        rebalance_threshold: int = 5,       # min delta units before rebalancing
    ):
        self.name = "VELVETFRUIT_EXTRACT"
        self.lim  = position_limit
        self.orders: List = []
        self.rebalance_threshold = rebalance_threshold

        self.pos = state.position.get(self.name, 0)

        od = state.order_depths.get(self.name)
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}
        if od:
            self.bids = {p: abs(v) for p, v in
                         sorted(od.buy_orders.items(),  key=lambda x: -x[0])
                         if v != 0}
            self.asks = {p: abs(v) for p, v in
                         sorted(od.sell_orders.items(), key=lambda x:  x[0])
                         if v != 0}

        # Compute target position from options delta exposure
        net_delta = sum(pos * option_deltas.get(k, 0.0)
                        for k, pos in option_positions.items())
        # net_delta is negative (short calls = short delta)
        # to hedge: hold +|net_delta| underlying
        self.target_pos = round(-net_delta)

    def _buy(self, price: int, volume: int) -> None:
        from datamodel import Order
        vol = min(volume, self.lim - self.pos)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, vol))
        self.pos += vol

    def _sell(self, price: int, volume: int) -> None:
        from datamodel import Order
        vol = min(volume, self.lim + self.pos)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, -vol))
        self.pos -= vol

    def get_orders(self) -> Dict[str, List]:
        diff = self.target_pos - self.pos

        # Only rebalance if off by more than threshold
        if abs(diff) < self.rebalance_threshold:
            return {self.name: []}

        if diff > 0:
            # Need to buy underlying
            remaining = diff
            for sp, sv in self.asks.items():
                if remaining <= 0:
                    break
                self._buy(sp, min(sv, remaining))
                remaining -= sv
        else:
            # Need to sell underlying
            remaining = -diff
            for bp, bv in self.bids.items():
                if remaining <= 0:
                    break
                self._sell(bp, min(bv, remaining))
                remaining -= bv

        return {self.name: self.orders}


# ── Strategy summary & parameter access ──────────────────────────────────────

def get_option_sellers(state, last_td, new_td, underlying_mid, T_remaining,
                       position_limits: Dict[str, int]) -> List[VEVOptionSeller]:
    """Create one VEVOptionSeller per tradeable strike."""
    sellers = []
    for K in SELL_STRIKES:
        name = f"VEV_{K}"
        if name not in state.order_depths:
            continue
        lim = position_limits.get(name, 200)
        sellers.append(VEVOptionSeller(
            name=name, strike=K, state=state,
            last_td=last_td, new_td=new_td,
            underlying_mid=underlying_mid,
            T_remaining=T_remaining,
            position_limit=lim,
        ))
    return sellers


def get_delta_hedger(state, last_td, new_td,
                     sellers: List[VEVOptionSeller],
                     position_limits: Dict[str, int]) -> DeltaHedger:
    """Build delta hedger from the current sellers' positions and deltas."""
    option_positions = {s.strike: state.position.get(s.name, 0) for s in sellers}
    option_deltas    = {s.strike: s.delta for s in sellers}
    return DeltaHedger(
        state=state, last_td=last_td, new_td=new_td,
        option_positions=option_positions,
        option_deltas=option_deltas,
        position_limit=position_limits.get("VELVETFRUIT_EXTRACT", 600),
    )
