from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
from functools import cached_property
import jsonpickle
import numpy as np
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
#  Strategy overview (v2 — improved from forensic analysis of log 385103):
#    HYDROGEL:   static mean-reversion (take/clear/make around mu=9991) ← kept
#    VEV_5200-5500: TAKE-ONLY seller with intrinsic-floor safety trigger
#      sell_trigger = intrinsic + 5  when BS_fair/market_mid > 1.10 (T inflated)
#      sell_trigger = BS_fair + threshold  otherwise (T correct)
#    VEV_4000-5100, 6000-6500: NOT TRADED (delta-1 / no premium)
#    VELVETFRUIT_EXTRACT: NOT TRADED (removed standalone MM that caused -179k)

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_PRODUCTS = [f"VEV_{k}" for k in STRIKES]

PRODUCTS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_PRODUCTS

# !! UPDATE THESE once you see the official position limits for Round 3 !!
POS_LIMITS = {
    "VELVETFRUIT_EXTRACT": 600,
    "HYDROGEL_PACK":        50,
    **{f"VEV_{k}": 200 for k in STRIKES},
}

# Options BS parameters (from analysis.py)
SIGMA        = 0.02155   # calibrated realized vol
EXPIRY_DAYS  = 3.0       # 3 data days (0,1,2); offset 0→2; T=EXPIRY_DAYS-(offset+ts/1e6)
RISK_FREE    = 0.0

PARAMS = {
    # ── Underlying: simple market-making around mid ──────────────────────────
    "VELVETFRUIT_EXTRACT": {
        "ema_configs"       : {"mid_ema": {"alpha": 0.05, "source": "mid"}},
        "fv_method_weights" : [0, 1],   # pure EMA (underlying has no fixed fair value)
        "take_margin"       : 2,
        "clear_margin"      : 3,
        "make_margin"       : 3,
    },
    # ── HYDROGEL: static fair-value market-making ────────────────────────────
    "HYDROGEL_PACK": {
        "ema_configs"       : {"mid_ema": {"alpha": 0.03, "source": "mid"}},
        "static_fv"         : 9991,     # long-run mean from data
        "fv_method_weights" : [1, 0],   # pure static
        "take_margin"       : 2,
        "clear_margin"      : 4,
        "make_margin"       : 3,
    },
    # ── VEV options: BS fair value (computed dynamically, not in PARAMS) ─────
    **{
        f"VEV_{k}": {
            "strike"      : k,
            "take_margin" : 3,   # only trade if |market - BS| > 3 ticks
            "clear_margin": 2,
            "make_margin" : 5,   # quote 5 ticks either side of BS fair
        }
        for k in STRIKES
    },
}


def default_traderData():
    return {product: {} for product in PRODUCTS}


# ─────────────────────────────────────────────────────────────────────────────
#  BASE CLASS  (copied from Jay's template, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class ProductTrader:
    def __init__(self, name, state, last_traderData, new_traderData):
        self.orders   = []
        self.name     = name
        self.state    = state
        self.timestamp = self.state.timestamp
        self.new_traderData  = new_traderData.setdefault(self.name, {})
        self.params          = PARAMS.get(self.name, {})
        self.last_traderData = last_traderData.get(self.name, {})
        self.position_limit  = POS_LIMITS.get(name, 0)

        self.starting_position = self._get_current_position()
        self.expected_position = self.starting_position

        self.market_trades = self._get_market_trades()
        self.own_trades    = self._get_own_trades()

        self.quoted_buy_orders, self.quoted_sell_orders = self._get_order_depth()
        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self._get_max_allowed_volume()

        self.take_margin  = self._get_take_margin()
        self.clear_margin = self._get_clear_margin()
        self.make_margin  = self._get_make_margin()

    _EMA_SOURCES = {
        "mid"        : "compute_mid_price",
        "vwap"       : "compute_vwap",
        "microprice" : "compute_microprice",
        "imbalance"  : "compute_imbalance_price",
        "wall_mid"   : "compute_wall_mid",
    }

    def _get_current_position(self):
        return self.state.position.get(self.name, 0)

    def _get_market_trades(self):
        return self.state.market_trades.get(self.name, [])

    def _get_own_trades(self):
        return self.state.own_trades.get(self.name, [])

    def _get_order_depth(self):
        order_depth = None
        try:
            order_depth: OrderDepth = self.state.order_depths[self.name]
        except Exception:
            pass
        buy_orders, sell_orders = {}, {}
        try:
            buy_orders  = {bp: abs(bv) for bp, bv in
                           sorted(order_depth.buy_orders.items(),  key=lambda x: x[0], reverse=True)}
            sell_orders = {sp: abs(sv) for sp, sv in
                           sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except Exception:
            pass
        return buy_orders, sell_orders

    def _get_max_allowed_volume(self):
        return (self.position_limit - self.starting_position,
                self.position_limit + self.starting_position)

    def _get_take_margin(self):   return self.params.get("take_margin",  1)
    def _get_clear_margin(self):  return self.params.get("clear_margin", 0)
    def _get_make_margin(self):   return self.params.get("make_margin",  1)

    def get_best_bid(self):
        return next(iter(self.quoted_buy_orders),  None)
    def get_best_ask(self):
        return next(iter(self.quoted_sell_orders), None)

    def buy(self, price, volume):
        vol = min(round(abs(volume)), self.max_allowed_buy_volume)
        self.max_allowed_buy_volume -= vol
        self.orders.append(Order(self.name, round(price), vol))
        self.expected_position += vol

    def sell(self, price, volume):
        vol = min(round(abs(volume)), self.max_allowed_sell_volume)
        self.max_allowed_sell_volume -= vol
        self.orders.append(Order(self.name, round(price), -vol))
        self.expected_position -= vol

    def update_traderData(self):
        self.new_traderData["last_timestamp"] = self.timestamp

    def compute_mid_price(self):
        bb, ba = self.get_best_bid(), self.get_best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0

    def compute_wall_mid(self):
        def wall(book):
            return max(book.items(), key=lambda x: x[1])[0] if book else None
        wb = wall(self.quoted_buy_orders)
        wa = wall(self.quoted_sell_orders)
        if wb is not None and wa is not None:
            return (wb + wa) / 2.0

    def compute_microprice(self):
        if self.quoted_sell_orders and self.quoted_buy_orders:
            ba, bav = next(iter(self.quoted_sell_orders)), 0
            bb, bbv = next(iter(self.quoted_buy_orders)),  0
            bav = self.quoted_sell_orders[ba]
            bbv = self.quoted_buy_orders[bb]
            if bav + bbv == 0:
                return None
            return (ba * bbv + bb * bav) / (bav + bbv)

    def compute_imbalance_price(self):
        bb, ba = self.get_best_bid(), self.get_best_ask()
        if bb is None or ba is None:
            return self.compute_mid_price()
        total_buy  = sum(self.quoted_buy_orders.values())
        total_sell = sum(self.quoted_sell_orders.values())
        total = total_buy + total_sell
        if not total:
            return self.compute_mid_price()
        mid       = (ba + bb) / 2
        spread    = ba - bb
        imbalance = (total_buy - total_sell) / total
        return mid + imbalance * (spread / 2)

    def compute_vwap(self):
        if not self.market_trades:
            return None
        num = sum(t.price * t.quantity for t in self.market_trades)
        den = sum(t.quantity              for t in self.market_trades)
        return num / den if den > 0 else None

    def compute_ema(self, config_name):
        cfg    = self.params["ema_configs"][config_name]
        alpha  = cfg["alpha"]
        source = cfg["source"]
        signal = getattr(self, self._EMA_SOURCES[source])()
        prev   = self.last_traderData.get(config_name)
        if signal is None and prev is None:
            return None
        ema = signal if prev is None else (prev if signal is None
              else alpha * signal + (1 - alpha) * prev)
        self.new_traderData[config_name] = ema
        return ema

    def compute_make_ask_price(self):
        fair_ask = self.fair_value + self.make_margin
        if not self.quoted_sell_orders:
            return fair_ask
        best_ask = next(iter(self.quoted_sell_orders))
        return best_ask - 1 if best_ask > fair_ask else fair_ask

    def compute_make_bid_price(self):
        fair_bid = self.fair_value - self.make_margin
        if not self.quoted_buy_orders:
            return fair_bid
        best_bid = next(iter(self.quoted_buy_orders))
        return best_bid + 1 if best_bid < fair_bid else fair_bid

    def get_orders(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  PRODUCT TRADERS
# ─────────────────────────────────────────────────────────────────────────────

class StaticFairValueTrader(ProductTrader):
    """
    Take/Clear/Make around a weighted (static_fv, EMA) fair value.
    Used for HYDROGEL_PACK and VELVETFRUIT_EXTRACT.
    """
    def __init__(self, name, state, last_traderData, new_traderData):
        super().__init__(name, state, last_traderData, new_traderData)
        self.static_fv = self.params.get("static_fv")
        weights = self.params.get("fv_method_weights", [1, 0])
        total   = sum(weights)
        self.w_static, self.w_ema = (weights[0]/total, weights[1]/total)
        self.fair_value = self._compute_fair_value()

    def _compute_fair_value(self):
        ema = self.compute_ema("mid_ema")
        if self.static_fv is None:
            return round(ema) if ema else None
        if ema is None:
            return self.static_fv
        return round(self.w_static * self.static_fv + self.w_ema * ema)

    def get_orders(self):
        if self.fair_value is None:
            return {self.name: []}

        # ── TAKE: hit bids above fair, lift asks below fair ──────────────────
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.fair_value + self.take_margin:
                break
            self.sell(bp, bv)

        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.fair_value - self.take_margin:
                break
            self.buy(sp, sv)

        # ── CLEAR: nudge position back toward zero at fair ───────────────────
        pos_after = self.expected_position
        if pos_after > 0:
            self.sell(self.fair_value + self.clear_margin, pos_after)
        elif pos_after < 0:
            self.buy(self.fair_value - self.clear_margin, -pos_after)

        # ── MAKE: passive quotes either side of fair ─────────────────────────
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()
        if self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume)
        if self.max_allowed_buy_volume > 0:
            self.buy(bid_price,  self.max_allowed_buy_volume)

        return {self.name: self.orders}


class VEVTrader(ProductTrader):
    """
    Call-option trader for VEV_XXXX products.

    Fair value = Black-Scholes call price using:
      - S  = current VELVETFRUIT_EXTRACT mid price   (passed in from Trader.run)
      - K  = option strike (from PARAMS)
      - T  = time remaining to expiry in competition-days
      - σ  = SIGMA  (calibrated from historical data)
      - r  = 0

    Strategy:
      TAKE  — immediately snipe options mispriced by > take_margin ticks vs BS
      CLEAR — reduce position at fair if we've accumulated inventory
      MAKE  — passive quotes MAKE_MARGIN either side of BS fair value
    """
    def __init__(self, name, state, last_traderData, new_traderData,
                 underlying_mid: Optional[float], day_offset: int):
        super().__init__(name, state, last_traderData, new_traderData)

        self.strike          = self.params["strike"]
        self.underlying_mid  = underlying_mid
        self.day_offset      = day_offset

        # Time remaining to expiry in competition-days
        self.T_remaining = max(
            EXPIRY_DAYS - (day_offset + self.timestamp / 1_000_000),
            0.0
        )

        # Compute BS fair value
        self.fair_value = self._bs_fair_value()

    # ── Black-Scholes (uses only math/numpy — safe for competition) ──────────
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    def _bs_call(self, S, K, T, sigma) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _bs_delta(self, S, K, T, sigma) -> float:
        """Delta = dC/dS.  Useful for hedging the underlying."""
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        return self._norm_cdf(d1)

    def _bs_fair_value(self) -> Optional[float]:
        if self.underlying_mid is None or self.T_remaining <= 0:
            # At expiry: intrinsic value only
            if self.underlying_mid is not None:
                return max(self.underlying_mid - self.strike, 0.0)
            return None
        return self._bs_call(self.underlying_mid, self.strike,
                              self.T_remaining, SIGMA)

    def get_orders(self):
        if self.fair_value is None:
            return {self.name: []}

        fv = self.fair_value

        # ── TAKE ─────────────────────────────────────────────────────────────
        # Lift cheap asks (market price < BS fair - margin → option is cheap, BUY)
        for sp, sv in self.quoted_sell_orders.items():
            if sp > fv - self.take_margin:
                break
            self.buy(sp, sv)

        # Hit expensive bids (market price > BS fair + margin → option is dear, SELL)
        for bp, bv in self.quoted_buy_orders.items():
            if bp < fv + self.take_margin:
                break
            self.sell(bp, bv)

        # ── CLEAR ─────────────────────────────────────────────────────────────
        pos_after = self.expected_position
        if pos_after > 0:
            self.sell(round(fv) + self.clear_margin, pos_after)
        elif pos_after < 0:
            self.buy(round(fv)  - self.clear_margin, -pos_after)

        # ── MAKE ─────────────────────────────────────────────────────────────
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()
        if self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume)
        if self.max_allowed_buy_volume > 0:
            self.buy(bid_price,  self.max_allowed_buy_volume)

        return {self.name: self.orders}

    # Override so compute_make_xxx use self.fair_value (already set)
    def compute_make_ask_price(self):
        fv = self.fair_value
        fair_ask = fv + self.make_margin
        if not self.quoted_sell_orders:
            return fair_ask
        best_ask = next(iter(self.quoted_sell_orders))
        return best_ask - 1 if best_ask > fair_ask else fair_ask

    def compute_make_bid_price(self):
        fv = self.fair_value
        fair_bid = fv - self.make_margin
        if not self.quoted_buy_orders:
            return fair_bid
        best_bid = next(iter(self.quoted_buy_orders))
        return best_bid + 1 if best_bid < fair_bid else fair_bid


# ── Import improved strategy classes ─────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from round3_trading_strategies import (
        VEVOptionSeller,
        get_option_sellers,
        SELL_STRIKES,
    )
    _STRATEGIES_LOADED = True
except Exception as _e:
    _STRATEGIES_LOADED = False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRADER
# ─────────────────────────────────────────────────────────────────────────────
class Trader:
    """
    Orchestrates all product traders each tick.

    v2 execution order:
      1. Decode traderData from last tick
      2. Compute T_remaining (day_offset tracks resets; if state is fresh, BS sanity
         check in VEVOptionSeller catches inflated T via fair/market_mid > 1.10)
      3. Read VELVETFRUIT_EXTRACT mid price
      4. HYDROGEL_PACK  — unchanged mean-reversion take/clear/make
      5. VEV options    — TAKE-ONLY with intrinsic-floor fallback (T-robust)
         sell when bid > max(BS_fair+threshold, intrinsic+5), or just intrinsic+5
         when T looks wrong. No delta hedge, no passive MAKE orders.
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

        # ── 1. Load state ─────────────────────────────────────────────────────
        last_traderData = (jsonpickle.decode(state.traderData)
                           if state.traderData else default_traderData())
        new_traderData  = default_traderData()
        new_traderData.setdefault("_meta", {})

        # ── 2. Time tracking ──────────────────────────────────────────────────
        day_offset = self._get_day_offset(state, last_traderData)
        new_traderData["_meta"]["day_offset"]     = day_offset
        new_traderData["_meta"]["last_timestamp"] = state.timestamp

        T_remaining = max(EXPIRY_DAYS - (day_offset + state.timestamp / 1_000_000), 0.0)

        # ── 3. Underlying mid ─────────────────────────────────────────────────
        underlying_mid = self._get_underlying_mid(state)

        # ── 4. HYDROGEL_PACK — mean-reversion (unchanged from v1) ─────────────
        hyd_trader = StaticFairValueTrader(
            "HYDROGEL_PACK", state, last_traderData, new_traderData
        )
        result.update(hyd_trader.get_orders())
        hyd_trader.update_traderData()

        # ── 5. VEV options — TAKE-ONLY selling (no delta hedge) ──────────────
        if _STRATEGIES_LOADED:
            # Sell overpriced options (K=5200-5500 only)
            sellers = get_option_sellers(
                state, last_traderData, new_traderData,
                underlying_mid, T_remaining, POS_LIMITS,
            )
            for seller in sellers:
                result.update(seller.get_orders())
        else:
            # strategies module failed to load — skip options entirely.
            # DO NOT fall back to VEVTrader: its MAKE orders cause immediate
            # adverse fills when T_remaining is wrong (forensic: log 385103).
            pass

        # ── 6. Save state ─────────────────────────────────────────────────────
        traderData = jsonpickle.encode(new_traderData)
        return result, 0, traderData