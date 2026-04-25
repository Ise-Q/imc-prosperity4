from datamodel import TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ROUND 3 PRODUCTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#
#  VELVETFRUIT_EXTRACT  — underlying asset, mean ~5250, tight spread ~5
#  VEV_XXXX             — call options on the underlying (strikes 4000–6500)
#  HYDROGEL_PACK        — mean-reverting product, mean ~9991, spread ~16
#
#  Options pricing convention (calibrated from data):
#    1 "year" in Black-Scholes = 1 competition day
#    T_remaining = 3.0 - (day_offset + timestamp / 1_000_000)
#    sigma = 0.02155  (daily realized vol, used as "annual" vol in BS)
#    Options expire at end of Day 2 (= T=0)
#
#  Strategy overview (v3 — fixes platform import failure diagnosed in log 386056):
#    ROOT CAUSE of 386056: round3_trading_strategies.py was a separate file;
#    the platform only accepts one file, so _STRATEGIES_LOADED was always False.
#    FIX: all option logic is now inline in this single file.
#
#    HYDROGEL:      position-aware take/clear/make around mu=9991
#    VEV_5200-5500: TAKE-ONLY seller, intrinsic-floor safety trigger
#      sell_trigger = intrinsic + 5  when BS_fair/market_mid > 1.10 (T inflated)
#      sell_trigger = BS_fair + threshold  otherwise (T correct)
#    VEV_4000-5100, 6000-6500: NOT TRADED (delta-1 or no tradeable premium)
#    VELVETFRUIT_EXTRACT: NOT TRADED

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_PRODUCTS = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_PRODUCTS

POS_LIMITS = {
    "VELVETFRUIT_EXTRACT": 600,
    "HYDROGEL_PACK":        50,
    **{f"VEV_{k}": 200 for k in STRIKES},
}

# ── Options parameters ────────────────────────────────────────────────────────

SIGMA       = 0.02155
EXPIRY_DAYS = 3.0

# Strikes actively traded (sell only)
SELL_STRIKES = [5200, 5300, 5400, 5500]

# Conservative per-strike short limits (managed delta risk)
OPTION_MAX_SHORT: Dict[int, int] = {
    5200: 50,
    5300: 50,
    5400: 75,
    5500: 100,
}

# ─────────────────────────────────────────────────────────────────────────────
#  BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(float(S - K), 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


# ─────────────────────────────────────────────────────────────────────────────
#  BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    "HYDROGEL_PACK": {
        "static_fv"  : 9991,
        "take_margin": 2,
        "clear_margin": 4,
        "make_margin": 3,
    },
}


def default_traderData():
    return {product: {} for product in PRODUCTS}


class ProductTrader:
    def __init__(self, name, state, last_traderData, new_traderData):
        self.orders   = []
        self.name     = name
        self.state    = state
        self.timestamp = state.timestamp
        self.new_traderData  = new_traderData.setdefault(name, {})
        self.params          = PARAMS.get(name, {})
        self.last_traderData = last_traderData.get(name, {})
        self.position_limit  = POS_LIMITS.get(name, 0)

        self.starting_position = state.position.get(name, 0)
        self.expected_position = self.starting_position

        self.quoted_buy_orders, self.quoted_sell_orders = self._get_order_depth()
        self.max_allowed_buy_volume  = self.position_limit - self.starting_position
        self.max_allowed_sell_volume = self.position_limit + self.starting_position

    def _get_order_depth(self):
        od = self.state.order_depths.get(self.name)
        if od is None:
            return {}, {}
        buy_orders  = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  key=lambda x: x[0], reverse=True) if v}
        sell_orders = {p: abs(v) for p, v in sorted(od.sell_orders.items(), key=lambda x: x[0]) if v}
        return buy_orders, sell_orders

    def get_best_bid(self):
        return next(iter(self.quoted_buy_orders),  None)

    def get_best_ask(self):
        return next(iter(self.quoted_sell_orders), None)

    def buy(self, price, volume):
        vol = min(round(abs(volume)), self.max_allowed_buy_volume)
        if vol <= 0:
            return
        self.max_allowed_buy_volume -= vol
        self.orders.append(Order(self.name, round(price), vol))
        self.expected_position += vol

    def sell(self, price, volume):
        vol = min(round(abs(volume)), self.max_allowed_sell_volume)
        if vol <= 0:
            return
        self.max_allowed_sell_volume -= vol
        self.orders.append(Order(self.name, round(price), -vol))
        self.expected_position -= vol

    def update_traderData(self):
        self.new_traderData["last_timestamp"] = self.timestamp


# ─────────────────────────────────────────────────────────────────────────────
#  HYDROGEL_PACK — position-aware mean-reversion
# ─────────────────────────────────────────────────────────────────────────────

class HydrogelTrader(ProductTrader):
    """
    Take/Clear/Make around static fair value 9991.

    Improvement over v2: MAKE orders are position-aware.
    When position is significantly one-sided, we suppress the MAKE order
    in the direction that would worsen inventory.  This prevents runaway
    accumulation on trending days (which caused -516 in log 386056).

    Rule: only post MAKE bid when pos < +make_gate (default +25),
          only post MAKE ask when pos > -make_gate (default -25).
    """

    MAKE_GATE = 25  # suppress MAKE when abs(pos) > this threshold

    def __init__(self, name, state, last_traderData, new_traderData):
        super().__init__(name, state, last_traderData, new_traderData)
        self.static_fv    = self.params["static_fv"]
        self.take_margin  = self.params["take_margin"]
        self.clear_margin = self.params["clear_margin"]
        self.make_margin  = self.params["make_margin"]
        self.fair_value   = self.static_fv

    def _make_ask_price(self):
        fair_ask = self.fair_value + self.make_margin
        ba = self.get_best_ask()
        if ba is None:
            return fair_ask
        return ba - 1 if ba > fair_ask else fair_ask

    def _make_bid_price(self):
        fair_bid = self.fair_value - self.make_margin
        bb = self.get_best_bid()
        if bb is None:
            return fair_bid
        return bb + 1 if bb < fair_bid else fair_bid

    def get_orders(self):
        fv = self.fair_value

        # TAKE: hit overpriced bids, lift underpriced asks
        for bp, bv in self.quoted_buy_orders.items():
            if bp < fv + self.take_margin:
                break
            self.sell(bp, bv)

        for sp, sv in self.quoted_sell_orders.items():
            if sp > fv - self.take_margin:
                break
            self.buy(sp, sv)

        # CLEAR: nudge toward zero
        pos = self.expected_position
        if pos > 0:
            self.sell(fv + self.clear_margin, pos)
        elif pos < 0:
            self.buy(fv - self.clear_margin, -pos)

        # MAKE: passive quotes — suppressed when inventory is already skewed
        pos = self.expected_position
        if self.max_allowed_sell_volume > 0 and pos > -self.MAKE_GATE:
            self.sell(self._make_ask_price(), self.max_allowed_sell_volume)
        if self.max_allowed_buy_volume > 0 and pos < self.MAKE_GATE:
            self.buy(self._make_bid_price(), self.max_allowed_buy_volume)

        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  VEV OPTIONS — TAKE-ONLY seller (intrinsic-floor + T-sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def _take_threshold(T_remaining: float) -> float:
    if T_remaining > 2.0:
        return 10.0   # day 0: conservative
    return 5.0        # day 1+: aggressive


class VEVOptionSeller:
    """
    TAKE-ONLY options seller for overpriced calls.

    Two-regime sell trigger (T-robust):
      Normal T  → sell when bid > BS_fair + threshold
      Inflated T (BS_fair / market_mid > 1.10) → sell when bid > intrinsic + 5

    The intrinsic-floor trigger guarantees profit at settlement regardless of
    T_remaining accuracy.  It saved us from the fresh-start disaster (log 385103).

    No passive MAKE orders — they cause immediate adverse fills when T is wrong.
    Cover logic: buy back only at ask ≤ intrinsic (never pay time value).
    """

    def __init__(
        self,
        name: str,
        strike: int,
        state,
        underlying_mid: Optional[float],
        T_remaining: float,
        position_limit: int,
    ):
        self.name   = name
        self.strike = strike
        self.lim    = position_limit
        self.T      = T_remaining
        self.S      = underlying_mid
        self.orders: List = []

        self.pos      = state.position.get(name, 0)
        self.max_sell = self.lim + self.pos
        self.max_buy  = self.lim - self.pos

        od = state.order_depths.get(name)
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}
        if od:
            self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  key=lambda x: -x[0]) if v}
            self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items(), key=lambda x:  x[0]) if v}

        # BS fair value
        self.fair: Optional[float] = None
        if self.S is not None:
            if T_remaining > 0:
                self.fair = _bs_call(self.S, self.strike, T_remaining, SIGMA)
            else:
                self.fair = max(float(self.S - self.strike), 0.0)

        self.threshold       = _take_threshold(T_remaining)
        self.max_option_short = OPTION_MAX_SHORT.get(strike, 50)

    def _sell(self, price: int, volume: int) -> None:
        cover_cap = max(0, self.max_option_short + self.pos)
        vol = min(volume, self.max_sell, cover_cap)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, -vol))
        self.max_sell -= vol
        self.pos -= vol

    def _buy(self, price: int, volume: int) -> None:
        cover_room = max(0, -self.pos)
        vol = min(volume, self.max_buy, cover_room)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, vol))
        self.max_buy -= vol
        self.pos += vol

    def get_orders(self) -> Dict[str, List]:
        if self.S is None:
            return {self.name: []}

        intrinsic   = max(float(self.S - self.strike), 0.0)
        MIN_PREMIUM = 5.0
        sell_floor  = intrinsic + MIN_PREMIUM

        market_mid = None
        if self.bids and self.asks:
            market_mid = (next(iter(self.bids)) + next(iter(self.asks))) / 2.0

        # T-sanity: if BS_fair >> market_mid, day_offset is likely wrong
        t_inflated = (
            self.fair is not None
            and market_mid is not None
            and self.fair > market_mid * 1.10
        )

        if t_inflated:
            sell_trigger = sell_floor
        elif self.fair is not None:
            sell_trigger = max(self.fair + self.threshold, sell_floor)
        else:
            sell_trigger = sell_floor

        # SELL: hit market bids above trigger
        for bp, bv in self.bids.items():
            if bp < sell_trigger:
                break
            self._sell(bp, bv)

        # COVER: only at ask ≤ intrinsic (never pay time value)
        for sp, sv in self.asks.items():
            if sp > intrinsic:
                break
            self._buy(sp, sv)

        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRADER
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    """
    v3 — single-file fix for platform import failure (386056 root cause).

    Execution order each tick:
      1. Decode traderData
      2. Track day_offset for T_remaining
      3. Read VELVETFRUIT_EXTRACT mid price
      4. HYDROGEL_PACK — position-aware mean-reversion
      5. VEV_5200-5500 — TAKE-ONLY selling with intrinsic-floor safety
      6. Encode state
    """

    def _get_day_offset(self, state: TradingState, last_td: dict) -> int:
        last_ts    = last_td.get("_meta", {}).get("last_timestamp", state.timestamp)
        day_offset = last_td.get("_meta", {}).get("day_offset", 0)
        if state.timestamp < last_ts:
            day_offset += 1
        return day_offset

    def _get_underlying_mid(self, state: TradingState) -> Optional[float]:
        try:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            return (bb + ba) / 2.0
        except Exception:
            return None

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # 1. Load state
        last_td = (jsonpickle.decode(state.traderData)
                   if state.traderData else default_traderData())
        new_td  = default_traderData()
        new_td.setdefault("_meta", {})

        # 2. Time tracking
        day_offset = self._get_day_offset(state, last_td)
        new_td["_meta"]["day_offset"]     = day_offset
        new_td["_meta"]["last_timestamp"] = state.timestamp
        T_remaining = max(EXPIRY_DAYS - (day_offset + state.timestamp / 1_000_000), 0.0)

        # 3. Underlying mid
        underlying_mid = self._get_underlying_mid(state)

        # 4. HYDROGEL_PACK
        hg = HydrogelTrader("HYDROGEL_PACK", state, last_td, new_td)
        result.update(hg.get_orders())
        hg.update_traderData()

        # 5. VEV options — TAKE-ONLY selling
        if underlying_mid is not None:
            for K in SELL_STRIKES:
                name = f"VEV_{K}"
                if name not in state.order_depths:
                    continue
                seller = VEVOptionSeller(
                    name=name,
                    strike=K,
                    state=state,
                    underlying_mid=underlying_mid,
                    T_remaining=T_remaining,
                    position_limit=POS_LIMITS.get(name, 200),
                )
                result.update(seller.get_orders())

        # 6. Save state
        return result, 0, jsonpickle.encode(new_td)
