from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ROUND 3 — "GLOVES OFF"
#  Products: VELVETFRUIT_EXTRACT, HYDROGEL_PACK, VEV_4000 … VEV_6500
#
#  TTE convention (confirmed from Round 3 rules):
#    Options have a 7-day expiry starting from Round 1.
#    At the START of Round 3, TTE = 5 days.
#    Each competition day = 1,000,000 ticks (timestamps 0 → 999900, step 100).
#    Inside Round 3:
#        TTE = 5.0 - (day_offset + timestamp / 1_000_000)
#
#  Sigma convention:
#    1 "year" in Black-Scholes = 1 competition day.
#    SIGMA_FALLBACK = 0.01262 (ATM implied vol calibrated from historical data
#    with correct TTE values). The trader updates sigma live using a rolling
#    window of implied vols computed from market prices (more robust than fixed).
#
#  Position limits (from official Round 3 rules):
#    HYDROGEL_PACK       → 200
#    VELVETFRUIT_EXTRACT → 200
#    VEV_XXXX (each)     → 300
# ─────────────────────────────────────────────────────────────────────────────

STRIKES     = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_NAMES   = [f"VEV_{k}" for k in STRIKES]
ALL_PRODUCTS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_NAMES

POS_LIMITS = {
    "VELVETFRUIT_EXTRACT" : 200,
    "HYDROGEL_PACK"       : 200,
    **{f"VEV_{k}": 300 for k in STRIKES},
}

SIGMA_FALLBACK = 0.01262   # ATM IV from historical data; used until rolling IV warms up
TTE_AT_R3_START = 5.0      # TTE when Round 3 begins
IV_WINDOW       = 100      # rolling window length (ticks) for implied vol smoothing

PARAMS = {
    # Underlying: market-make around a slow EMA of mid
    "VELVETFRUIT_EXTRACT": {
        "ema_alpha"   : 0.05,
        "take_margin" : 2,
        "clear_margin": 3,
        "make_margin" : 3,
    },
    # HYDROGEL: static fair-value market-making around long-run mean
    "HYDROGEL_PACK": {
        "static_fv"   : 9991,
        "ema_alpha"   : 0.03,
        "take_margin" : 2,
        "clear_margin": 4,
        "make_margin" : 3,
    },
    # Each VEV: BS-priced, margins around fair value
    **{
        f"VEV_{k}": {
            "strike"      : k,
            "take_margin" : 3,
            "clear_margin": 2,
            "make_margin" : 5,
        }
        for k in STRIKES
    },
}


def default_traderData():
    return {p: {} for p in ALL_PRODUCTS + ["_meta"]}


# ─────────────────────────────────────────────────────────────────────────────
#  BLACK-SCHOLES  (pure math — safe for competition, no scipy needed)
# ─────────────────────────────────────────────────────────────────────────────
def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price. T and sigma both in competition-day units."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * _ncdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS call delta (dC/dS). Used for optional delta-hedging of underlying."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)

def implied_vol(mkt: float, S: float, K: float, T: float) -> Optional[float]:
    """
    Implied vol via bisection (50 steps → ~1e-14 precision).
    Returns None when market is at intrinsic or unsolvable.
    """
    intrinsic = max(S - K, 0.0)
    if mkt <= intrinsic + 0.01 or T <= 0:
        return None
    lo, hi = 1e-4, 3.0
    if bs_call(S, K, T, hi) < mkt:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if bs_call(S, K, T, mid) < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
#  BASE PRODUCT TRADER  (Jay's architecture)
# ─────────────────────────────────────────────────────────────────────────────
class ProductTrader:
    def __init__(self, name: str, state: TradingState,
                 last_td: dict, new_td: dict):
        self.orders    = []
        self.name      = name
        self.state     = state
        self.timestamp = state.timestamp
        self.new_td    = new_td.setdefault(name, {})
        self.last_td   = last_td.get(name, {})
        self.params    = PARAMS.get(name, {})
        self.pos_limit = POS_LIMITS.get(name, 0)

        self.position  = state.position.get(name, 0)
        self.market_trades = state.market_trades.get(name, [])

        self.bids, self.asks = self._parse_book()
        self.max_buy  = self.pos_limit - self.position
        self.max_sell = self.pos_limit + self.position

    def _parse_book(self):
        bids, asks = {}, {}
        try:
            od   = self.state.order_depths[self.name]
            bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  reverse=True)}
            asks = {p: abs(v) for p, v in sorted(od.sell_orders.items())}
        except Exception:
            pass
        return bids, asks

    def best_bid(self): return next(iter(self.bids), None)
    def best_ask(self): return next(iter(self.asks), None)
    def mid(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    def buy(self, price: float, volume: float):
        vol = min(round(abs(volume)), self.max_buy)
        if vol <= 0: return
        self.max_buy -= vol           # only reduce buy capacity; don't touch sell
        self.orders.append(Order(self.name, round(price), vol))

    def sell(self, price: float, volume: float):
        vol = min(round(abs(volume)), self.max_sell)
        if vol <= 0: return
        self.max_sell -= vol          # only reduce sell capacity; don't touch buy
        self.orders.append(Order(self.name, round(price), -vol))

    def ema(self, key: str, signal: Optional[float], alpha: float) -> Optional[float]:
        """Generic EMA with state stored in new_td."""
        prev = self.last_td.get(key)
        if signal is None and prev is None: return None
        value = signal if prev is None else (prev if signal is None
                else alpha * signal + (1 - alpha) * prev)
        self.new_td[key] = value
        return value

    def save_timestamp(self):
        self.new_td["last_timestamp"] = self.timestamp

    def result(self): return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC FAIR-VALUE TRADER  (HYDROGEL_PACK + VELVETFRUIT_EXTRACT)
# ─────────────────────────────────────────────────────────────────────────────
class StaticFVTrader(ProductTrader):
    """
    Take/Clear/Make around a fair value that is either:
      - a static constant weighted with an EMA of mid  (HYDROGEL)
      - pure EMA of mid                                (underlying)
    """
    def __init__(self, name, state, last_td, new_td):
        super().__init__(name, state, last_td, new_td)
        alpha      = self.params["ema_alpha"]
        static_fv  = self.params.get("static_fv")
        mid_ema    = self.ema("mid_ema", self.mid(), alpha)

        if static_fv is not None:
            # HYDROGEL: weight 100% static (mean is very stable)
            self.fv = static_fv if mid_ema is None else static_fv
        else:
            # Underlying: pure EMA
            self.fv = mid_ema

        self.take_w  = self.params["take_margin"]
        self.clear_w = self.params["clear_margin"]
        self.make_w  = self.params["make_margin"]

    def get_orders(self):
        if self.fv is None:
            return self.result()

        fv = self.fv
        # ── TAKE ─────────────────────────────────────────────────────────────
        # Hit bids that are too high (> fv + take_w → sell)
        for bp, bv in self.bids.items():
            edge = bp - fv
            if edge < self.take_w: break
            self.sell(bp, bv)
        # Lift asks that are too low (< fv - take_w → buy)
        for ap, av in self.asks.items():
            edge = fv - ap
            if edge < self.take_w : break
            self.buy(ap, av)

        # ── CLEAR: work position back toward zero ─────────────────────────────
        pos_after = self.position + sum(
            o.quantity for o in self.orders)
        if pos_after > 0:
            self.sell(round(fv) + self.clear_w, pos_after)
        elif pos_after < 0:
            self.buy(round(fv) - self.clear_w, -pos_after)

        # ── MAKE ─────────────────────────────────────────────────────────────
        # Quote ask: undercut best ask if it's above fair_ask, else post at fair_ask
        fair_ask = round(fv) + self.make_w
        fair_bid = round(fv) - self.make_w
        ba = self.best_ask()
        bb = self.best_bid()
        ask_price = (ba - 1) if (ba is not None and ba > fair_ask) else fair_ask
        bid_price = (bb + 1) if (bb is not None and bb < fair_bid) else fair_bid

        self.sell(ask_price, self.max_sell)
        self.buy(bid_price,  self.max_buy)

        return self.result()


# ─────────────────────────────────────────────────────────────────────────────
#  VEV OPTION TRADER  (Black-Scholes fair value with rolling IV)
# ─────────────────────────────────────────────────────────────────────────────
class VEVTrader(ProductTrader):
    """
    Each tick:
      1. Compute TTE from day_offset + timestamp.
      2. Get implied vol from this option's market price → update rolling IV EMA.
      3. Use rolling IV (or SIGMA_FALLBACK) as sigma.
      4. BS fair value = bs_call(S, K, TTE, sigma).
      5. Take/Clear/Make around BS fair.

    The rolling IV makes the strategy adaptive:
      - If the market is pricing vol higher (e.g. near expiry spike), we adapt.
      - Prevents the "dead model on submission day" problem from P3 top teams.
    """
    def __init__(self, name: str, state: TradingState,
                 last_td: dict, new_td: dict,
                 underlying_mid: Optional[float],
                 day_offset: int):
        super().__init__(name, state, last_td, new_td)

        self.K   = self.params["strike"]
        self.S   = underlying_mid
        self.TTE = max(TTE_AT_R3_START - (day_offset + self.timestamp / 1_000_000), 0.0)

        self.take_w  = self.params["take_margin"]
        self.clear_w = self.params["clear_margin"]
        self.make_w  = self.params["make_margin"]

        # ── Update rolling implied vol ────────────────────────────────────────
        mkt_mid = self.mid()
        live_iv = None
        if mkt_mid is not None and self.S is not None and self.TTE > 0:
            live_iv = implied_vol(mkt_mid, self.S, self.K, self.TTE)

        # EMA-smooth the IV; alpha=0.05 → ~20-tick memory
        self.sigma = self.ema("rolling_iv", live_iv, alpha=0.05)
        if self.sigma is None or self.sigma <= 0:
            self.sigma = SIGMA_FALLBACK

        # ── BS fair value ─────────────────────────────────────────────────────
        if self.S is not None and self.TTE > 0:
            self.fv = bs_call(self.S, self.K, self.TTE, self.sigma)
        elif self.S is not None:
            self.fv = max(self.S - self.K, 0.0)   # at expiry: intrinsic only
        else:
            self.fv = None

    def get_orders(self):
        if self.fv is None:
            return self.result()

        fv = max(self.fv, 1.0)   # ensure fair value is positive (for options, and just in case)

        if fv < 5:
            take_w = 1
            clear_w = 1
            make_w = 1
        
        else:
            take_w  = self.params["take_margin"]
            clear_w = self.params["clear_margin"]
            make_w  = self.params["make_margin"]

        # ── TAKE ─────────────────────────────────────────────────────────────
        # Option looks cheap → BUY (lift ask below fair - take_w)
        for ap, av in self.asks.items():
            if ap > fv*0.8: break
            self.buy(ap, av)
        # Option looks expensive → SELL (hit bid above fair + take_w)
        for bp, bv in self.bids.items():
            if bp < fv * 1.2: break
            self.sell(bp, bv)

        # ── CLEAR ─────────────────────────────────────────────────────────────
        pos_after = self.position + sum(o.quantity for o in self.orders)
        if pos_after > 0:
            self.sell(round(fv) + clear_w, pos_after)
        elif pos_after < 0:
            self.buy(round(fv) - clear_w, -pos_after)

        # ── MAKE ─────────────────────────────────────────────────────────────
        fair_ask  = round(fv) + make_w
        fair_bid  = round(fv) - make_w
        ba = self.best_ask(); bb = self.best_bid()
        ask_price = (ba - 1) if (ba is not None and ba > fair_ask) else fair_ask
        bid_price = (bb + 1) if (bb is not None and bb < fair_bid) else fair_bid

        self.sell(ask_price, self.max_sell)
        self.buy(bid_price,  self.max_buy)

        return self.result()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRADER
# ─────────────────────────────────────────────────────────────────────────────
class Trader:
    """
    Execution order each tick:
      1. Load traderData; determine day_offset (0,1,2 across the 3 days).
      2. Read VELVETFRUIT_EXTRACT mid → passed to all VEV traders.
      3. Run StaticFVTrader for VELVETFRUIT_EXTRACT and HYDROGEL_PACK.
      4. Run VEVTrader for each of the 10 option strikes.
      5. Encode and return traderData.

    day_offset: increments each time timestamp wraps back to 0,
    i.e. day_offset=0 on Day 1 of Round 3, =1 on Day 2, =2 on Day 3.
    """

    def _day_offset(self, state: TradingState, last_td: dict) -> int:
        last_ts    = last_td.get("_meta", {}).get("last_timestamp", state.timestamp)
        day_offset = last_td.get("_meta", {}).get("day_offset", 0)
        if state.timestamp < last_ts:   # timestamp wrapped → new day
            day_offset += 1
        return day_offset

    def _underlying_mid(self, state: TradingState) -> Optional[float]:
        try:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            return (bb + ba) / 2.0
        except Exception:
            return None

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # ── Load state ────────────────────────────────────────────────────────
        last_td = jsonpickle.decode(state.traderData) if state.traderData else default_traderData()
        new_td  = default_traderData()

        # ── Day tracking ──────────────────────────────────────────────────────
        day_offset = self._day_offset(state, last_td)
        new_td["_meta"]["day_offset"]    = day_offset
        new_td["_meta"]["last_timestamp"] = state.timestamp

        tte = max(TTE_AT_R3_START - (day_offset + state.timestamp / 1_000_000), 0.0)
        print(f"[t={state.timestamp}] day_offset={day_offset}  TTE={tte:.4f}")

        # ── Underlying mid price (needed by all VEV traders) ──────────────────
        S = self._underlying_mid(state)

        # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
        # Disabled because it is strongly negative in backtest
        new_td["VELVETFRUIT_EXTRACT"] = last_td.get("VELVETFRUIT_EXTRACT", {})

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        hyd = StaticFVTrader("HYDROGEL_PACK", state, last_td, new_td)
        result.update(hyd.get_orders())
        hyd.save_timestamp()

        # ── VEV options (10 strikes) ──────────────────────────────────────────
        for k in STRIKES:
            vev = VEVTrader(f"VEV_{k}", state, last_td, new_td,
                            underlying_mid=S, day_offset=day_offset)
            result.update(vev.get_orders())
            vev.save_timestamp()

        # ── Return ────────────────────────────────────────────────────────────
        return result, 0, jsonpickle.encode(new_td)