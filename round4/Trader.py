from datamodel import TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ROUND 4 — single-file Trader
# ─────────────────────────────────────────────────────────────────────────────
#
#  Products:
#    VELVETFRUIT_EXTRACT  — underlying, mean ~5248, spread ~5
#    VEV_XXXX             — call options (strikes 4000–6500)
#    HYDROGEL_PACK        — OU mean-reversion, mu~9995, spread=16, HL=350 ticks
#
#  Round 4 calibration (from EDA on rounds_4_day_1/2/3):
#    sigma           = 0.02168   (up from 0.02155 in R3)
#    HYDROGEL mu     = 9995      (up from 9991 in R3)
#    T_remaining     = 3.0 − (day_offset + timestamp / 1_000_000)
#    Options expiry  = T=0 at end of day 3 (3-day round; days labeled 1/2/3)
#
#  Round 4 options mispricing pattern (DIFFERS from R3):
#    Day 1 (T: 3→2): market is BELOW BS by 3–5 ticks → DO NOT SELL (threshold=10 protects)
#    Day 2 (T: 2→1): market is ABOVE BS by +6–8 ticks → sell when bid > BS+5
#    Day 3 (T: 1→0): market is ABOVE BS by +20–22 ticks → sell very aggressively
#
#  Round 4 position limits (doubled/quadrupled vs R3):
#    VELVETFRUIT_EXTRACT : 200   (was 600 — tighter)
#    HYDROGEL_PACK       : 200   (was 50  — 4× increase)
#    VEV_*               : 300   (was 200 — 50% increase)
#
#  Round 4 new feature: counterparty IDs on market trades.
#  Counterparty profiles (from EDA on 4,281 trades):
#    Mark 01  — informed buyer VEV, 83% win rate, systematic call buyer → FOLLOW
#    Mark 14  — profitable MM hydrogel (buys FV-8, sells FV+8, 90% win) → FOLLOW
#    Mark 22  — systematic option seller (competitor) → IGNORE
#    Mark 38  — systematic loser hydrogel (buys FV+8, sells FV-8, 9% win) → FADE
#    Mark 49  — noise buyer VEV, 33% win rate → FADE
#    Mark 55  — liquidity/noise VEV → IGNORE
#    Mark 67  — directional buyer VEV (never sells) → mild FOLLOW

STRIKES     = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_PRODUCTS = [f"VEV_{k}" for k in STRIKES]
PRODUCTS    = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_PRODUCTS

POS_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK":        200,
    **{f"VEV_{k}": 300 for k in STRIKES},
}

# ── Options parameters ────────────────────────────────────────────────────────

SIGMA        = 0.02168   # recalibrated from Round 4 realized vol
EXPIRY_DAYS  = 3.0       # T_remaining = 3.0 - (day_offset + ts/1e6)

SELL_STRIKES = [5200, 5300, 5400, 5500]

# Conservative per-strike short caps (increased vs R3, delta hedge now active)
OPTION_MAX_SHORT: Dict[int, int] = {
    5200: 75,    # ITM, delta~0.8 — keep tight
    5300: 100,   # near-ATM, delta~0.5
    5400: 150,   # OTM, delta~0.3
    5500: 200,   # deep OTM, delta~0.15
}

# ── Counterparty signal config ────────────────────────────────────────────────

# VEV underlying: follow informed buyers, fade noise
FOLLOW_VEV = frozenset({"Mark 01", "Mark 14", "Mark 67"})
FADE_VEV   = frozenset({"Mark 49"})

# HYDROGEL: Mark 14 = informed MM (follow), Mark 38 = systematic loser (fade)
FOLLOW_HYD = frozenset({"Mark 14"})
FADE_HYD   = frozenset({"Mark 38"})

# Max directional units to add from counterparty signal on VEV underlying
CP_VEV_LIMIT = 40   # conservative: only 40/200 of position limit is signal-driven


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


def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def _counterparty_signal(market_trades: dict, symbol: str,
                          follow: frozenset, fade: frozenset) -> int:
    """
    Net signed volume from predictive counterparties in the last tick.
    +N = informed net buy pressure of N units (bullish signal).
    -N = informed net sell pressure (bearish signal).
    """
    trades = market_trades.get(symbol, [])
    signal = 0
    for t in trades:
        qty    = getattr(t, 'quantity', 0)
        buyer  = getattr(t, 'buyer',   '')
        seller = getattr(t, 'seller',  '')
        if buyer in follow or seller in fade:
            signal += qty
        if seller in follow or buyer in fade:
            signal -= qty
    return signal


def default_traderData() -> dict:
    return {product: {} for product in PRODUCTS}


# ─────────────────────────────────────────────────────────────────────────────
#  BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    "HYDROGEL_PACK": {
        "static_fv"   : 9995,   # Round 4 OU mu (recalibrated)
        "take_margin" : 2,
        "clear_margin": 4,
        "make_margin" : 3,
    },
}


class ProductTrader:
    def __init__(self, name: str, state: TradingState, last_td: dict, new_td: dict):
        self.orders   = []
        self.name     = name
        self.state    = state
        self.timestamp = state.timestamp
        self.new_traderData  = new_td.setdefault(name, {})
        self.params          = PARAMS.get(name, {})
        self.last_traderData = last_td.get(name, {})
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

    def get_best_bid(self) -> Optional[int]:
        return next(iter(self.quoted_buy_orders), None)

    def get_best_ask(self) -> Optional[int]:
        return next(iter(self.quoted_sell_orders), None)

    def buy(self, price: float, volume: float) -> None:
        vol = min(round(abs(volume)), self.max_allowed_buy_volume)
        if vol <= 0:
            return
        self.max_allowed_buy_volume  -= vol
        self.max_allowed_sell_volume += vol
        self.orders.append(Order(self.name, round(price), vol))
        self.expected_position += vol

    def sell(self, price: float, volume: float) -> None:
        vol = min(round(abs(volume)), self.max_allowed_sell_volume)
        if vol <= 0:
            return
        self.max_allowed_sell_volume -= vol
        self.max_allowed_buy_volume  += vol
        self.orders.append(Order(self.name, round(price), -vol))
        self.expected_position -= vol

    def update_traderData(self) -> None:
        self.new_traderData["last_timestamp"] = self.timestamp


# ─────────────────────────────────────────────────────────────────────────────
#  HYDROGEL_PACK — position-aware mean-reversion + counterparty bias
# ─────────────────────────────────────────────────────────────────────────────

class HydrogelTrader(ProductTrader):
    """
    Take/Clear/Make around static FV=9995, enhanced with counterparty signal.

    Counterparty layer (Round 4 insight):
      Mark 38 is systematically wrong (buys FV+8, sells FV-8, 9% win rate).
      Mark 14 is systematically right (buys FV-8, sells FV+8, 90% win rate).

      cp_signal > 0: informed flow is NET BUYING → price is currently CHEAP
        → be more aggressive on BUY side (lower TAKE_LONG_GATE threshold)
      cp_signal < 0: informed flow is NET SELLING → price is currently RICH
        → be more aggressive on SELL side, suppress buy MAKE orders

    Three-layer inventory control (inherited from R3 fix):
      TAKE_LONG_GATE = 20  (was 10 — scaled for 200-unit limit)
      MAKE_GATE      = 80  (was 25 — scaled for 200-unit limit)
    """

    TAKE_LONG_GATE = 20
    MAKE_GATE      = 80

    def __init__(self, name: str, state: TradingState, last_td: dict, new_td: dict):
        super().__init__(name, state, last_td, new_td)
        self.static_fv    = self.params["static_fv"]
        self.take_margin  = self.params["take_margin"]
        self.clear_margin = self.params["clear_margin"]
        self.make_margin  = self.params["make_margin"]
        self.fair_value   = self.static_fv

        # Counterparty signal: >0 = informed net buying (price cheap), <0 = selling (rich)
        self.cp_signal = _counterparty_signal(
            state.market_trades, name, FOLLOW_HYD, FADE_HYD
        )

    def _make_ask_price(self) -> int:
        fair_ask = self.fair_value + self.make_margin
        ba = self.get_best_ask()
        if ba is None:
            return fair_ask
        return ba - 1 if ba > fair_ask else fair_ask

    def _make_bid_price(self) -> int:
        fair_bid = self.fair_value - self.make_margin
        bb = self.get_best_bid()
        if bb is None:
            return fair_bid
        return bb + 1 if bb < fair_bid else fair_bid

    def get_orders(self) -> Dict[str, List]:
        fv = self.fair_value

        # Adjust TAKE_LONG_GATE based on counterparty signal:
        # If informed traders are net buying, we can hold more long inventory
        gate_adj      = min(20, max(-20, self.cp_signal // 2))
        effective_gate = self.TAKE_LONG_GATE + gate_adj

        # TAKE: hit overpriced bids (no gate on short side)
        for bp, bv in self.quoted_buy_orders.items():
            if bp < fv + self.take_margin:
                break
            self.sell(bp, bv)

        # TAKE: lift cheap asks — gated by adjusted inventory threshold
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
        # Suppress make ask when counterparty signal says price is cheap (strong buy signal)
        suppress_ask = (self.cp_signal > 10 and pos > 0)
        # Suppress make bid when counterparty signal says price is rich (strong sell signal)
        suppress_bid = (self.cp_signal < -10 and pos < 0)

        if self.max_allowed_sell_volume > 0 and pos > -self.MAKE_GATE and not suppress_ask:
            self.sell(self._make_ask_price(), self.max_allowed_sell_volume)
        if self.max_allowed_buy_volume > 0 and pos < self.MAKE_GATE and not suppress_bid:
            self.buy(self._make_bid_price(), self.max_allowed_buy_volume)

        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  VEV UNDERLYING — delta hedge + counterparty signal tilt
# ─────────────────────────────────────────────────────────────────────────────

class VEVUnderlyingTrader:
    """
    Trades VELVETFRUIT_EXTRACT for two purposes:
      1. Delta hedge the short option book (primary)
      2. Directional tilt from counterparty signal (secondary, bounded)

    Delta hedge:
      net_delta = sum(option_pos_i × BS_delta_i)   [neg for short calls]
      hedge_target = round(-net_delta)              [need long underlying]

    Counterparty signal (Round 4 insight):
      Mark 01 wins 83% at +1000 ticks — follow their direction
      Mark 14 wins 77% — follow
      Mark 49 wins 33% — fade
      Signal is additive to hedge_target, capped at ±CP_VEV_LIMIT units

    Execution: TAKE-only (aggressive), no passive quoting.
    Rebalance threshold: 3 units (avoid excessive churn).
    """

    REBALANCE_THRESHOLD = 3

    def __init__(
        self,
        state: TradingState,
        option_positions: Dict[int, int],
        option_deltas:    Dict[int, float],
    ):
        self.name     = "VELVETFRUIT_EXTRACT"
        self.lim      = POS_LIMITS["VELVETFRUIT_EXTRACT"]
        self.orders: List[Order] = []

        self.pos = state.position.get(self.name, 0)
        self.max_buy  = self.lim - self.pos
        self.max_sell = self.lim + self.pos

        od = state.order_depths.get(self.name)
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}
        if od:
            self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  key=lambda x: -x[0]) if v}
            self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items(), key=lambda x:  x[0]) if v}

        # 1. Delta hedge target
        net_delta = sum(pos * option_deltas.get(k, 0.0) for k, pos in option_positions.items())
        hedge_target = round(-net_delta)   # negative delta from short calls → need long underlying

        # 2. Counterparty signal tilt
        cp_signal = _counterparty_signal(state.market_trades, self.name, FOLLOW_VEV, FADE_VEV)
        # Scale: each unit of signed volume = 0.3 directional units, capped
        cp_tilt = max(-CP_VEV_LIMIT, min(CP_VEV_LIMIT, round(cp_signal * 0.3)))

        # Total target (capped to position limit)
        self.target_pos = max(-self.lim, min(self.lim, hedge_target + cp_tilt))

    def _buy(self, price: int, volume: int) -> None:
        vol = min(volume, self.max_buy)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, vol))
        self.max_buy -= vol
        self.pos     += vol

    def _sell(self, price: int, volume: int) -> None:
        vol = min(volume, self.max_sell)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, price, -vol))
        self.max_sell -= vol
        self.pos      -= vol

    def get_orders(self) -> Dict[str, List]:
        diff = self.target_pos - self.pos

        if abs(diff) < self.REBALANCE_THRESHOLD:
            return {self.name: []}

        if diff > 0:
            remaining = diff
            for sp, sv in self.asks.items():
                if remaining <= 0:
                    break
                take = min(sv, remaining)
                self._buy(sp, take)
                remaining -= take
        else:
            remaining = -diff
            for bp, bv in self.bids.items():
                if remaining <= 0:
                    break
                take = min(bv, remaining)
                self._sell(bp, take)
                remaining -= take

        return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  VEV OPTIONS — TAKE-ONLY seller (intrinsic-floor + T-sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def _take_threshold(T_remaining: float) -> float:
    # T>2.0 = day 1: market is below BS → high threshold prevents selling cheap options
    # T≤2.0 = days 2-3: market is above BS → sell when overpriced by ≥5 ticks
    if T_remaining > 2.0:
        return 10.0
    return 5.0


class VEVOptionSeller:
    """
    TAKE-ONLY options seller for overpriced calls.

    Two-regime sell trigger (T-robust):
      Normal T  → sell when bid > BS_fair + threshold
      Inflated T (BS_fair / market_mid > 1.10) → sell when bid > intrinsic + 5

    Round 4 enhancements:
      - Higher position limits (300) → higher OPTION_MAX_SHORT caps
      - Option_max_short values increased proportionally vs R3
      - No change to logic (pattern identical to R3: +12/+27 mispricing holds)
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
        self.delta: float = 0.0
        if self.S is not None:
            if T_remaining > 0:
                self.fair  = _bs_call(self.S, self.strike, T_remaining, SIGMA)
                self.delta = _bs_delta(self.S, self.strike, T_remaining, SIGMA)
            else:
                self.fair  = max(float(self.S - self.strike), 0.0)
                self.delta = 1.0 if self.S > self.strike else 0.0

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

        for bp, bv in self.bids.items():
            if bp < sell_trigger:
                break
            self._sell(bp, bv)

        # Cover: only at ask ≤ intrinsic (never pay time value)
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
    Round 4 — single-file Trader (platform requires one file).

    Execution order each tick:
      1. Decode traderData
      2. Track day_offset for T_remaining
      3. Read VELVETFRUIT_EXTRACT mid price
      4. HYDROGEL_PACK — position-aware mean-reversion + counterparty bias
      5. VEV_5200-5500 — TAKE-ONLY option selling (same as R3, updated limits)
      6. VELVETFRUIT_EXTRACT — delta hedge + counterparty tilt (NEW in R4)
      7. Encode state
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
        new_td = default_traderData()
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

        # 5. VEV options — TAKE-ONLY selling, collect deltas for hedge
        option_positions: Dict[int, int]   = {}
        option_deltas:    Dict[int, float] = {}

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
                option_positions[K] = state.position.get(name, 0)
                option_deltas[K]    = seller.delta

        # 6. VELVETFRUIT_EXTRACT — delta hedge + counterparty signal (NEW)
        if underlying_mid is not None:
            vev_trader = VEVUnderlyingTrader(
                state=state,
                option_positions=option_positions,
                option_deltas=option_deltas,
            )
            result.update(vev_trader.get_orders())

        # 7. Save state
        return result, 0, jsonpickle.encode(new_td)
