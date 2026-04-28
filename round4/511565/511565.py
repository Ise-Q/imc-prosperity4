from datamodel import TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ROUND 4 — rebuilt after log 509764 forensic analysis
# ─────────────────────────────────────────────────────────────────────────────
#
#  Root causes fixed vs 509764:
#    BUG 1 — ProductTrader cross-update: buy() was adding to max_allowed_sell_volume
#             and sell() was adding to max_allowed_buy_volume. This caused total orders
#             to exceed position limits every tick. Platform rejected ALL HYDROGEL orders.
#             Fix: each side's allowance only decreases, never cross-increments.
#
#    BUG 2 — Delta hedge destruction: buying 181 VEV at 5299 with wrong T=3.0 deltas
#             into a -42 tick fall. Cost -8405. Options only +5664. Net -2741.
#             Fix: DISABLE delta hedge entirely. Options profit from theta, not delta.
#             On day 3 (T→0), OTM delta < 0.1 → hedge spread cost > hedge benefit.
#
#    BUG 3 — HYDROGEL FV wrong: static 9995 vs actual day-3 mean 10033.5.
#             Fix: dynamic FV from EMA of order book mid. Always stays near market.
#
#    BUG 4 — Wrong T_remaining: fresh start on day 3 gave T=3.0 instead of 1.0.
#             Fix: detect day from option market prices as sanity check.
#
#  Products (Round 4):
#    VELVETFRUIT_EXTRACT  — underlying, spread ~5 ticks
#    VEV_XXXX             — call options (strikes 4000–6500), expire T=0
#    HYDROGEL_PACK        — OU mean-reversion, but FV drifts between days
#
#  Position limits (Round 4):
#    VELVETFRUIT_EXTRACT : 200
#    HYDROGEL_PACK       : 200
#    VEV_*               : 300
#
#  Counterparty profiles (from EDA + log analysis):
#    Mark 01  — systematic call buyer (funds our option premium), VEV informed seller
#    Mark 14  — HYDROGEL profitable MM (buys FV-8, sells FV+8), VEV informed seller
#    Mark 22  — competing option seller
#    Mark 38  — HYDROGEL systematic loser (buys FV+8, sells FV-8, 9% win)
#    Mark 49  — VEV noise (33% win, fade)
#    Mark 55  — VEV liquidity/noise
#    Mark 67  — VEV directional buyer (never sells)

STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_PRODUCTS = [f"VEV_{k}" for k in STRIKES]
PRODUCTS     = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_PRODUCTS

POS_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK":        200,
    **{f"VEV_{k}": 300 for k in STRIKES},
}

# ── Options ───────────────────────────────────────────────────────────────────

SIGMA        = 0.02168
EXPIRY_DAYS  = 3.0

SELL_STRIKES = [5200, 5300, 5400, 5500]

# Per-strike short caps. Conservative — options are the clean profit source.
OPTION_MAX_SHORT: Dict[int, int] = {
    5200: 75,
    5300: 100,
    5400: 150,
    5500: 200,
}

# ── Counterparty IDs ──────────────────────────────────────────────────────────

# HYDROGEL: Mark 38 is the systematic loser (buys FV+8, sells FV-8) — fade their flow.
FADE_HYD = frozenset({"Mark 38"})


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(float(S - K), 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def _detect_day_from_options(state: TradingState, underlying_mid: float) -> Optional[int]:
    """
    Estimate day_offset from option market prices when traderData is unavailable.
    At T=3.0: BS gives high values. At T=1.0: much lower.
    Compare market price of VEV_5300 to BS at T=3,2,1 → pick closest.
    Returns estimated day_offset (0, 1, or 2) or None if unable.
    """
    if underlying_mid is None:
        return None
    od = state.order_depths.get("VEV_5300")
    if od is None:
        return None
    if not od.buy_orders or not od.sell_orders:
        return None
    mkt_mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
    if mkt_mid <= 0:
        return None

    best_offset = 0
    best_err = float("inf")
    for offset in [0, 1, 2]:
        T_guess = max(EXPIRY_DAYS - offset, 0.01)
        bs_guess = _bs_call(underlying_mid, 5300, T_guess, SIGMA)
        err = abs(bs_guess - mkt_mid)
        if err < best_err:
            best_err = err
            best_offset = offset
    return best_offset


def default_traderData() -> dict:
    return {product: {} for product in PRODUCTS}


# ─────────────────────────────────────────────────────────────────────────────
#  BASE CLASS — BUG FIX: no cross-update between buy/sell allowances
# ─────────────────────────────────────────────────────────────────────────────
#
#  Previous code in buy():  self.max_allowed_sell_volume += vol  (WRONG)
#  Previous code in sell(): self.max_allowed_buy_volume  += vol  (WRONG)
#
#  Why it was wrong: each side's allowance is independently bounded by the
#  position limit. After submitting sell orders, remaining sell capacity should
#  ONLY decrease. Cross-incrementing the buy side is meaningless to the platform.
#  The platform checks: sum(sell_orders) ≤ limit + current_pos per side.
#  Cross-incrementing caused sum(sell_orders) > limit, rejected every tick.

PARAMS = {
    "HYDROGEL_PACK": {
        "take_margin" : 2,
        "clear_margin": 4,
        "make_margin" : 3,
        "ema_alpha"   : 0.05,   # slow EMA for adaptive FV
    },
}


class ProductTrader:
    def __init__(self, name: str, state: TradingState, last_td: dict, new_td: dict):
        self.orders    = []
        self.name      = name
        self.state     = state
        self.timestamp = state.timestamp
        self.new_traderData  = new_td.setdefault(name, {})
        self.params          = PARAMS.get(name, {})
        self.last_traderData = last_td.get(name, {})
        self.position_limit  = POS_LIMITS.get(name, 0)

        self.starting_position = state.position.get(name, 0)
        self.expected_position = self.starting_position

        self.quoted_buy_orders, self.quoted_sell_orders = self._get_order_depth()
        # Each side is bounded independently by position limit.
        self.max_allowed_buy_volume  = self.position_limit - self.starting_position
        self.max_allowed_sell_volume = self.position_limit + self.starting_position

    def _get_order_depth(self):
        od = self.state.order_depths.get(self.name)
        if od is None:
            return {}, {}
        buy_orders  = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  key=lambda x: x[0], reverse=True) if v}
        sell_orders = {p: abs(v) for p, v in sorted(od.sell_orders.items(), key=lambda x: x[0]) if v}
        return buy_orders, sell_orders

    def get_best_bid(self) -> Optional[int]:
        return next(iter(self.quoted_buy_orders), None)

    def get_best_ask(self) -> Optional[int]:
        return next(iter(self.quoted_sell_orders), None)

    def buy(self, price: float, volume: float) -> None:
        vol = min(round(abs(volume)), self.max_allowed_buy_volume)
        if vol <= 0:
            return
        self.max_allowed_buy_volume -= vol
        # FIX: do NOT touch max_allowed_sell_volume here.
        self.orders.append(Order(self.name, round(price), vol))
        self.expected_position += vol

    def sell(self, price: float, volume: float) -> None:
        vol = min(round(abs(volume)), self.max_allowed_sell_volume)
        if vol <= 0:
            return
        self.max_allowed_sell_volume -= vol
        # FIX: do NOT touch max_allowed_buy_volume here.
        self.orders.append(Order(self.name, round(price), -vol))
        self.expected_position -= vol

    def update_traderData(self) -> None:
        self.new_traderData["last_timestamp"] = self.timestamp


# ─────────────────────────────────────────────────────────────────────────────
#  HYDROGEL_PACK — adaptive FV + counterparty bias
# ─────────────────────────────────────────────────────────────────────────────
#
#  FIX: FV is now an EMA of market mid, not static 9995.
#  Root cause of failure: day 3 actual mean was 10033.5, our FV was 9995.
#  All sell orders at 9998 never filled; all buy orders far below market.
#
#  Counterparty layer:
#  Mark 38 consistently buys at FV+8 and sells at FV-8 (9% win rate).
#  When Mark 38 is buying, price is at a local HIGH — suppress our buys.
#  When Mark 38 is selling, price is at a local LOW — suppress our sells.

class HydrogelTrader(ProductTrader):

    TAKE_LONG_GATE = 20
    MAKE_GATE      = 80

    def __init__(self, name: str, state: TradingState, last_td: dict, new_td: dict):
        super().__init__(name, state, last_td, new_td)
        p = self.params

        # Adaptive FV: EMA of market mid, initialized from order book on first tick.
        ob_mid = None
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is not None and ba is not None:
            ob_mid = (bb + ba) / 2.0

        prev_fv  = self.last_traderData.get("fv_ema", ob_mid)
        alpha    = p.get("ema_alpha", 0.05)
        if prev_fv is None:
            self.fair_value = ob_mid if ob_mid else 10000.0
        elif ob_mid is None:
            self.fair_value = prev_fv
        else:
            self.fair_value = (1 - alpha) * prev_fv + alpha * ob_mid

        self.new_traderData["fv_ema"] = self.fair_value

        self.take_margin  = p.get("take_margin",  2)
        self.clear_margin = p.get("clear_margin", 4)
        self.make_margin  = p.get("make_margin",  3)

        # Counterparty signal: detect Mark 38 direction to fade them.
        hyd_trades = state.market_trades.get(name, [])
        self.cp_signal = 0
        for t in hyd_trades:
            qty    = getattr(t, 'quantity', 0)
            buyer  = getattr(t, 'buyer',   '')
            seller = getattr(t, 'seller',  '')
            if buyer in FADE_HYD:
                # Mark 38 buying = price at local HIGH → fade = we should SELL
                self.cp_signal -= qty
            if seller in FADE_HYD:
                # Mark 38 selling = price at local LOW → fade = we should BUY
                self.cp_signal += qty

    def _make_ask_price(self) -> int:
        fair_ask = self.fair_value + self.make_margin
        ba = self.get_best_ask()
        if ba is None:
            return round(fair_ask)
        return ba - 1 if ba > fair_ask else round(fair_ask)

    def _make_bid_price(self) -> int:
        fair_bid = self.fair_value - self.make_margin
        bb = self.get_best_bid()
        if bb is None:
            return round(fair_bid)
        return bb + 1 if bb < fair_bid else round(fair_bid)

    def get_orders(self) -> Dict[str, List]:
        fv = self.fair_value

        # Adjust TAKE_LONG_GATE: if Mark 38 is buying (price HIGH), reduce buy willingness
        gate_adj      = min(15, max(-15, self.cp_signal // 2))
        effective_gate = self.TAKE_LONG_GATE + gate_adj

        # TAKE: hit overpriced bids
        for bp, bv in self.quoted_buy_orders.items():
            if bp < fv + self.take_margin:
                break
            self.sell(bp, bv)

        # TAKE: lift cheap asks — gated
        for sp, sv in self.quoted_sell_orders.items():
            if sp > fv - self.take_margin:
                break
            if self.expected_position >= effective_gate:
                break
            self.buy(sp, sv)

        # CLEAR: nudge toward zero
        pos = self.expected_position
        if pos > 0:
            self.sell(fv + self.clear_margin, pos)
        elif pos < 0:
            self.buy(fv - self.clear_margin, -pos)

        # MAKE: passive quotes — suppressed when inventory skewed
        pos = self.expected_position
        if self.max_allowed_sell_volume > 0 and pos > -self.MAKE_GATE:
            self.sell(self._make_ask_price(), self.max_allowed_sell_volume)
        if self.max_allowed_buy_volume > 0 and pos < self.MAKE_GATE:
            self.buy(self._make_bid_price(), self.max_allowed_buy_volume)

        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  VEV OPTIONS — TAKE-ONLY seller (intrinsic-floor + T-sanity check)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Round 4 mispricing pattern (from EDA):
#    Day 1 (T: 3→2): market BELOW BS by 3-5 ticks → threshold=10 blocks selling cheap
#    Day 2 (T: 2→1): market ABOVE BS by +6-8 ticks → sell when bid > BS+5
#    Day 3 (T: 1→0): market ABOVE BS by +20-22 ticks → sell aggressively, bid > BS+5
#
#  Intrinsic floor always active: sell only if bid > intrinsic + 5.
#  This guarantees profit at settlement regardless of T_remaining accuracy.
#
#  NO delta hedge: log 509764 showed the hedge cost -8405, options earned +5664.
#  On day 3 with T→0, OTM delta < 0.1, hedge spread cost > hedge benefit.
#  Delta-neutral is only valuable if you can earn theta. Here we collect premium
#  via the sell-at-expiry structure — no ongoing theta capture needed.

def _take_threshold(T_remaining: float) -> float:
    if T_remaining > 2.0:
        return 10.0   # day 1: market below BS, high threshold blocks selling cheap
    return 5.0         # days 2-3: market above BS, aggressive


class VEVOptionSeller:
    """
    TAKE-ONLY options seller. No delta hedge (removed after log 509764 analysis).
    """

    def __init__(
        self,
        name: str,
        strike: int,
        state: TradingState,
        underlying_mid: Optional[float],
        T_remaining: float,
        position_limit: int,
    ):
        self.name   = name
        self.strike = strike
        self.lim    = position_limit
        self.T      = T_remaining
        self.S      = underlying_mid
        self.orders: List[Order] = []

        self.pos      = state.position.get(name, 0)
        self.max_sell = self.lim + self.pos
        self.max_buy  = self.lim - self.pos

        od = state.order_depths.get(name)
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}
        if od:
            self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  key=lambda x: -x[0]) if v}
            self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items(), key=lambda x:  x[0]) if v}

        self.fair: Optional[float] = None
        if self.S is not None:
            if T_remaining > 0:
                self.fair = _bs_call(self.S, self.strike, T_remaining, SIGMA)
            else:
                self.fair = max(float(self.S - self.strike), 0.0)

        self.threshold        = _take_threshold(T_remaining)
        self.max_option_short = OPTION_MAX_SHORT.get(strike, 75)

    def _sell(self, price: int, volume: int) -> None:
        cover_cap = max(0, self.max_option_short + self.pos)
        vol = min(volume, self.max_sell, cover_cap)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, -vol))
        self.max_sell -= vol
        self.pos      -= vol

    def _buy(self, price: int, volume: int) -> None:
        cover_room = max(0, -self.pos)
        vol = min(volume, self.max_buy, cover_room)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, vol))
        self.max_buy -= vol
        self.pos     += vol

    def get_orders(self) -> Dict[str, List]:
        if self.S is None:
            return {self.name: []}

        intrinsic   = max(float(self.S - self.strike), 0.0)
        MIN_PREMIUM = 5.0
        sell_floor  = intrinsic + MIN_PREMIUM

        market_mid = None
        if self.bids and self.asks:
            market_mid = (next(iter(self.bids)) + next(iter(self.asks))) / 2.0

        # T-sanity: if BS(T) >> market_mid, T_remaining is likely wrong (day_offset error).
        # Fall back to intrinsic floor — guaranteed profitable at settlement.
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

        # SELL: hit bids above trigger
        for bp, bv in self.bids.items():
            if bp < sell_trigger:
                break
            self._sell(bp, bv)

        # COVER: buy back only at intrinsic (never pay time value)
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
    Round 4 — rebuilt after log 509764 forensic analysis.

    Changes from previous version:
      - ProductTrader cross-update bug FIXED (no more limit exceeded errors)
      - HYDROGEL FV now adaptive (EMA of market mid, not static 9995)
      - Delta hedge REMOVED (cost -8405 vs options +5664 in log 509764)
      - Day detection improved: uses option price sanity check as fallback
      - VEVUnderlyingTrader class removed entirely

    Execution order each tick:
      1. Decode state, track day_offset
      2. Sanity-check T_remaining against option market prices
      3. HYDROGEL — adaptive mean-reversion (adaptive FV, fixed limits)
      4. VEV options — TAKE-ONLY selling (intrinsic-floor + BS trigger)
    """

    def _get_day_offset(self, state: TradingState, last_td: dict,
                         underlying_mid: Optional[float]) -> int:
        last_ts    = last_td.get("_meta", {}).get("last_timestamp", state.timestamp)
        day_offset = last_td.get("_meta", {}).get("day_offset", 0)

        if state.timestamp < last_ts:
            day_offset += 1

        # Sanity-check: if T_remaining implied by day_offset seems inconsistent
        # with option market prices, override with estimate from option prices.
        # This handles the "fresh start on day 3" failure from log 509764.
        T_implied = max(EXPIRY_DAYS - (day_offset + state.timestamp / 1_000_000), 0.0)
        if T_implied > 2.0 and underlying_mid is not None:
            # Our T estimate says day 1, but let's verify against VEV_5300 price
            est_offset = _detect_day_from_options(state, underlying_mid)
            if est_offset is not None and est_offset > day_offset:
                day_offset = est_offset

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
        new_td = default_traderData()
        new_td.setdefault("_meta", {})

        # 2. Underlying mid (needed for day detection)
        underlying_mid = self._get_underlying_mid(state)

        # 3. Time tracking with option-price sanity check
        day_offset = self._get_day_offset(state, last_td, underlying_mid)
        new_td["_meta"]["day_offset"]     = day_offset
        new_td["_meta"]["last_timestamp"] = state.timestamp
        T_remaining = max(EXPIRY_DAYS - (day_offset + state.timestamp / 1_000_000), 0.0)

        # 4. HYDROGEL_PACK — adaptive mean-reversion
        hg = HydrogelTrader("HYDROGEL_PACK", state, last_td, new_td)
        result.update(hg.get_orders())
        hg.update_traderData()

        # 5. VEV options — TAKE-ONLY selling, no delta hedge
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
                    position_limit=POS_LIMITS.get(name, 300),
                )
                result.update(seller.get_orders())

        # 6. Save state
        return result, 0, jsonpickle.encode(new_td)