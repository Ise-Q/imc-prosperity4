from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# trader3 — incremental upgrade over trader2:
#   * HYDROGEL : full-size passive MM + take leg (was 10% sizing, no take)
#   * Vouchers : ALWAYS-on BS-fair MM on liquid strikes (5000-5500); take leg
#                vs BS fair beyond a threshold; deep-ITM (4000/4500) keeps the
#                no-arb intrinsic take-only behaviour from trader2
#   * VEE      : same hedge-then-residual-MM as trader2 (slightly looser
#                hedge threshold to cut churn cost)
#
# Motivation (from round3 EDA):
#   - HYDROGEL spread mean=16 ticks → wide MM edge underused at 10% size
#   - RV(~0.41) >> mean IV(~0.25) → vouchers underpriced → tilt long vol
#   - Liquid strikes 5200/5300/5400/5500 had 18/121/225/267 trades and
#     trader2 quoted ZERO of them


# =============================================================================
# Product universe
# =============================================================================

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{K}" for K in STRIKES]
VOUCHER_STRIKE = {f"VEV_{K}": K for K in STRIKES}

# Pinned at mid=0.5 across all historical days — illiquid, skip.
DEAD_VOUCHERS = {"VEV_6000", "VEV_6500"}
TRADED_VOUCHERS = [v for v in VOUCHERS if v not in DEAD_VOUCHERS]

# Deep-ITM strikes where BS fair ≈ intrinsic and IV inversion is unreliable —
# keep trader2's no-arb-only take leg, no MM.
DEEP_ITM_VOUCHERS = {"VEV_4000", "VEV_4500"}
MM_VOUCHERS = [v for v in TRADED_VOUCHERS if v not in DEEP_ITM_VOUCHERS]

ALL_PRODUCTS = [HYDROGEL, UNDERLYING] + VOUCHERS

POS_LIMITS = {
    HYDROGEL: 200,
    UNDERLYING: 200,
    **{v: 300 for v in VOUCHERS},
}

# Per-product MM params
PARAMS = {
    HYDROGEL: {
        "ema_alpha": 0.05,
        "static_fv": 9990,
        "fv_method_weights": [0.0, 1.0],
        "take_margin": 3,    # lift asks <= fair - 3, hit bids >= fair + 3
        "clear_margin": 4,
        "make_margin": 5,
        "make_size_frac": 1.0,   # full available volume per side
        "clear_pos_frac": 0.4,   # start clearing when |pos| > 40% limit
        "clear_qty_frac": 0.25,  # clear 25% of position toward flat
    },
}

# Hedge / VEE residual-MM params
HEDGE_PARAMS = {
    "hedge_every_ticks": 1,
    "delta_threshold": 8,           # was 5; loosen slightly to cut churn
    "make_margin": 2,
    "max_make_size": 30,
    "vee_ema_alpha": 0.05,
}

# Shared option-trader params
OPTION_PARAMS = {
    "iv_ema_alpha": 0.1,
    "spread_ema_alpha": 0.05,
    "default_iv": 0.23,
    "make_skew_per_unit": 1.0,      # tighter inventory skew (was 0.5)
    "min_make_half_spread": 1,
    "take_edge": 2.0,               # take vs BS fair when |edge| > this (price units)
    "vol_tilt": 0.5,                # bias bids up / asks up by 0.5 (long-vol tilt)
    "make_size_frac": 0.5,          # 50% of available volume per quote
}

# TTE tracking
TTE_DAYS_AT_ROUND_START = 5
TICKS_PER_DAY = 1_000_000
TRADING_DAYS_PER_YEAR = 365.0


def default_traderData():
    d = {product: {} for product in ALL_PRODUCTS}
    d["_meta"] = {}
    return d


# =============================================================================
# Black-Scholes utilities
# =============================================================================

class BlackScholes:
    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, sigma: float):
        v = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
        d2 = d1 - v
        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        d1, d2 = BlackScholes._d1_d2(S, K, T, sigma)
        return S * BlackScholes.norm_cdf(d1) - K * BlackScholes.norm_cdf(d2)

    @staticmethod
    def call_delta(S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        d1, _ = BlackScholes._d1_d2(S, K, T, sigma)
        return BlackScholes.norm_cdf(d1)

    @staticmethod
    def implied_vol(C_obs: float, S: float, K: float, T: float,
                    lo: float = 1e-4, hi: float = 5.0,
                    tol: float = 1e-5, maxit: int = 60) -> Optional[float]:
        if T <= 0 or S <= 0 or K <= 0:
            return None
        intrinsic = max(S - K, 0.0)
        if C_obs < intrinsic - 1e-6 or C_obs > S + 1e-6:
            return None
        if C_obs <= intrinsic + 1e-6:
            return None
        f_lo = BlackScholes.call_price(S, K, T, lo) - C_obs
        f_hi = BlackScholes.call_price(S, K, T, hi) - C_obs
        if f_lo * f_hi > 0:
            return None
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            f_mid = BlackScholes.call_price(S, K, T, mid) - C_obs
            if abs(f_mid) < tol:
                return mid
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)


# =============================================================================
# Base class
# =============================================================================

class ProductTrader:
    def __init__(self, name, state, new_traderData, last_traderData):
        self.orders: List[Order] = []
        self.name = name
        self.state = state
        self.timestamp = state.timestamp
        self.new_traderData = new_traderData
        self.new_traderData.setdefault(name, {})
        self.last_traderData = last_traderData

        self.position_limit = POS_LIMITS.get(name, 0)
        self.starting_position = state.position.get(name, 0)
        self.expected_position = self.starting_position

        self.quoted_buy_orders, self.quoted_sell_orders = self._get_order_depth()

        self.max_allowed_buy_volume = self.position_limit - self.starting_position
        self.max_allowed_sell_volume = self.position_limit + self.starting_position

        p = PARAMS.get(name, {})
        self.take_margin = p.get("take_margin", 1)
        self.clear_margin = p.get("clear_margin", 0)
        self.make_margin = p.get("make_margin", 1)

    def _get_order_depth(self):
        buy_orders, sell_orders = {}, {}
        od = self.state.order_depths.get(self.name)
        if od is not None:
            buy_orders = {bp: abs(bv) for bp, bv in sorted(od.buy_orders.items(), reverse=True)}
            sell_orders = {sp: abs(sv) for sp, sv in sorted(od.sell_orders.items())}
        return buy_orders, sell_orders

    def get_best_bid(self):
        return next(iter(self.quoted_buy_orders), None)

    def get_best_ask(self):
        return next(iter(self.quoted_sell_orders), None)

    def compute_mid_price(self):
        bb, ba = self.get_best_bid(), self.get_best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    def compute_spread(self):
        bb, ba = self.get_best_bid(), self.get_best_ask()
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def buy(self, price, volume):
        abs_volume = min(int(abs(volume)), self.max_allowed_buy_volume)
        if abs_volume <= 0:
            return
        self.max_allowed_buy_volume -= abs_volume
        self.expected_position += abs_volume
        self.orders.append(Order(self.name, int(price), abs_volume))

    def sell(self, price, volume):
        abs_volume = min(int(abs(volume)), self.max_allowed_sell_volume)
        if abs_volume <= 0:
            return
        self.max_allowed_sell_volume -= abs_volume
        self.expected_position -= abs_volume
        self.orders.append(Order(self.name, int(price), -abs_volume))

    def compute_make_ask_price(self, fair_value):
        fair_ask = fair_value + self.make_margin
        best_ask = self.get_best_ask()
        if best_ask is None:
            return fair_ask
        return best_ask - 1 if best_ask > fair_ask else fair_ask

    def compute_make_bid_price(self, fair_value):
        fair_bid = fair_value - self.make_margin
        best_bid = self.get_best_bid()
        if best_bid is None:
            return fair_bid
        return best_bid + 1 if best_bid < fair_bid else fair_bid

    def update_traderData(self):
        self.new_traderData[self.name]["last_timestamp"] = self.timestamp


# =============================================================================
# MeanReversionTrader — HYDROGEL_PACK : take + clear + make at full size
# =============================================================================

class MeanReversionTrader(ProductTrader):
    def __init__(self, name, state, new_traderData, last_traderData):
        super().__init__(name, state, new_traderData, last_traderData)
        p = PARAMS[name]
        self.ema_alpha = p["ema_alpha"]
        self.static_fv = p["static_fv"]
        self.fv_method_weights = self._normalize_weights(p["fv_method_weights"])
        self.make_size_frac = p.get("make_size_frac", 1.0)
        self.clear_pos_frac = p.get("clear_pos_frac", 0.5)
        self.clear_qty_frac = p.get("clear_qty_frac", 0.1)
        self.fair_value = self.compute_fair_value()

    @staticmethod
    def _normalize_weights(weights):
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else [1.0, 0.0]

    def compute_fair_value(self):
        mid = self.compute_mid_price()
        prev_ema = self.last_traderData.get(self.name, {}).get("ema")
        if mid is None:
            ema = prev_ema if prev_ema is not None else self.static_fv
        elif prev_ema is None:
            ema = mid
        else:
            ema = self.ema_alpha * mid + (1 - self.ema_alpha) * prev_ema
        w_static, w_ema = self.fv_method_weights
        fair = w_static * self.static_fv + w_ema * ema
        self.new_traderData[self.name]["ema"] = ema
        return round(fair)

    def get_orders(self):
        # ---- TAKE leg : lift cheap asks, hit rich bids vs fair ----
        for sp, sv in self.quoted_sell_orders.items():
            if sp <= self.fair_value - self.take_margin:
                self.buy(sp, sv)
            else:
                break
        for bp, bv in self.quoted_buy_orders.items():
            if bp >= self.fair_value + self.take_margin:
                self.sell(bp, bv)
            else:
                break

        # ---- CLEAR leg : reduce inventory toward flat at fair ----
        threshold = self.clear_pos_frac * self.position_limit
        if self.expected_position > threshold:
            qty = max(1, int(self.expected_position * self.clear_qty_frac))
            self.sell(self.fair_value + self.clear_margin, qty)
        elif self.expected_position < -threshold:
            qty = max(1, int(-self.expected_position * self.clear_qty_frac))
            self.buy(self.fair_value - self.clear_margin, qty)

        # ---- MAKE leg : full-size two-sided quotes ----
        ask_price = self.compute_make_ask_price(self.fair_value)
        bid_price = self.compute_make_bid_price(self.fair_value)
        if self.max_allowed_sell_volume > 0:
            self.sell(ask_price, int(self.max_allowed_sell_volume * self.make_size_frac))
        if self.max_allowed_buy_volume > 0:
            self.buy(bid_price, int(self.max_allowed_buy_volume * self.make_size_frac))

        return {self.name: self.orders}


# =============================================================================
# OptionTrader — three modes:
#   DEEP_ITM (4000/4500)  : intrinsic-violation take-only (trader2 behaviour)
#   MM strikes (5000-5500): BS-fair take + always-on BS-fair MM with vol-tilt
# =============================================================================

class OptionTrader(ProductTrader):
    def __init__(self, name, state, new_traderData, last_traderData,
                 underlying_mid: Optional[float], tte_years: float):
        super().__init__(name, state, new_traderData, last_traderData)
        self.S = underlying_mid
        self.T = tte_years
        self.K = VOUCHER_STRIKE[name]
        self.is_deep_itm = name in DEEP_ITM_VOUCHERS
        self.mid = self.compute_mid_price()
        self.spread = self.compute_spread()

        prev = self.last_traderData.get(self.name, {})
        self.prev_iv_ema: Optional[float] = prev.get("iv_ema")
        self.prev_spread_ema: Optional[float] = prev.get("spread_ema")
        self.prev_mid: Optional[float] = prev.get("last_mid")

        self.live_iv: Optional[float] = None
        if self.mid is not None and self.S is not None:
            self.live_iv = BlackScholes.implied_vol(self.mid, self.S, self.K, self.T)

        a_iv = OPTION_PARAMS["iv_ema_alpha"]
        if self.live_iv is not None:
            self.iv_ema = (a_iv * self.live_iv + (1 - a_iv) * self.prev_iv_ema
                           if self.prev_iv_ema is not None else self.live_iv)
        else:
            self.iv_ema = self.prev_iv_ema

        a_sp = OPTION_PARAMS["spread_ema_alpha"]
        if self.spread is not None:
            self.spread_ema = (a_sp * self.spread + (1 - a_sp) * self.prev_spread_ema
                               if self.prev_spread_ema is not None else float(self.spread))
        else:
            self.spread_ema = self.prev_spread_ema

        self.iv_used = (self.live_iv if self.live_iv is not None
                        else (self.iv_ema if self.iv_ema is not None
                              else OPTION_PARAMS["default_iv"]))

        self._post_take_position: Optional[int] = None

        td = self.new_traderData[self.name]
        if self.iv_ema is not None:
            td["iv_ema"] = self.iv_ema
        if self.spread_ema is not None:
            td["spread_ema"] = self.spread_ema
        if self.mid is not None:
            td["last_mid"] = self.mid

    @property
    def position_delta(self) -> float:
        if self.S is None:
            return 0.0
        pos = self._post_take_position if self._post_take_position is not None else self.expected_position
        d = BlackScholes.call_delta(self.S, self.K, self.T, self.iv_used)
        return pos * d

    def get_orders(self):
        self._post_take_position = self.starting_position
        if self.S is None or not (self.quoted_buy_orders or self.quoted_sell_orders):
            return {self.name: self.orders}

        intrinsic = max(self.S - self.K, 0.0)

        if self.is_deep_itm:
            # ---- Deep-ITM : intrinsic-violation take-only (trader2) ----
            for sp, sv in self.quoted_sell_orders.items():
                if sp >= intrinsic:
                    break
                self.buy(sp, sv)
            for bp, bv in self.quoted_buy_orders.items():
                if bp <= self.S:
                    break
                self.sell(bp, bv)
            self._post_take_position = self.expected_position
            return {self.name: self.orders}

        # ---- Liquid MM strikes : BS-fair take + always-on MM ----
        bs_fair = BlackScholes.call_price(self.S, self.K, self.T, self.iv_used)
        if bs_fair <= 0:
            self._post_take_position = self.expected_position
            return {self.name: self.orders}

        take_edge = OPTION_PARAMS["take_edge"]
        # Take : lift asks below bs_fair - take_edge (long vol bias built in via vol_tilt later)
        for sp, sv in self.quoted_sell_orders.items():
            if sp <= bs_fair - take_edge:
                self.buy(sp, sv)
            else:
                break
        # Always also take obvious no-arb violations
        for sp, sv in self.quoted_sell_orders.items():
            if sp >= intrinsic:
                break
            self.buy(sp, sv)
        for bp, bv in self.quoted_buy_orders.items():
            if bp >= bs_fair + take_edge:
                self.sell(bp, bv)
            else:
                break

        self._post_take_position = self.expected_position

        # ---- Make leg around BS fair with long-vol tilt + inventory skew ----
        half = max(0.5 * (self.spread_ema or 2.0),
                   OPTION_PARAMS["min_make_half_spread"])
        skew = (OPTION_PARAMS["make_skew_per_unit"]
                * self.expected_position / max(self.position_limit, 1))
        vol_tilt = OPTION_PARAMS["vol_tilt"]
        bid_price = round(bs_fair - half - skew + vol_tilt)
        ask_price = round(bs_fair + half - skew + vol_tilt)
        bid_price = self.compute_make_bid_price(bid_price)
        ask_price = self.compute_make_ask_price(ask_price)

        size_frac = OPTION_PARAMS["make_size_frac"]
        if self.max_allowed_buy_volume > 0:
            self.buy(bid_price, int(self.max_allowed_buy_volume * size_frac))
        if self.max_allowed_sell_volume > 0:
            self.sell(ask_price, int(self.max_allowed_sell_volume * size_frac))

        return {self.name: self.orders}


# =============================================================================
# HedgeTrader — VEE: priority delta hedge + residual passive MM
# =============================================================================

class HedgeTrader(ProductTrader):
    def __init__(self, name, state, new_traderData, last_traderData, option_traders):
        super().__init__(name, state, new_traderData, last_traderData)
        self.option_traders = option_traders

        prev = self.last_traderData.get(self.name, {})
        self.last_hedge_ts: Optional[int] = prev.get("last_hedge_ts")
        self.prev_vee_ema: Optional[float] = prev.get("vee_ema")

        mid = self.compute_mid_price()
        a = HEDGE_PARAMS["vee_ema_alpha"]
        if mid is None:
            self.vee_ema = self.prev_vee_ema
        elif self.prev_vee_ema is None:
            self.vee_ema = mid
        else:
            self.vee_ema = a * mid + (1 - a) * self.prev_vee_ema
        if self.vee_ema is not None:
            self.new_traderData[self.name]["vee_ema"] = self.vee_ema

    def _hedge_due(self, target_delta_trade: int) -> bool:
        every_ticks = HEDGE_PARAMS["hedge_every_ticks"]
        if self.last_hedge_ts is None:
            return True
        elapsed = self.timestamp - self.last_hedge_ts
        if elapsed >= every_ticks * 100:
            return True
        if abs(target_delta_trade) > HEDGE_PARAMS["delta_threshold"]:
            return True
        return False

    def get_orders(self):
        opt_delta = sum(t.position_delta for t in self.option_traders)
        target = -opt_delta
        target_int = max(-self.position_limit, min(self.position_limit, int(round(target))))
        delta_trade = target_int - self.starting_position

        if self._hedge_due(delta_trade):
            if delta_trade > 0:
                remaining = delta_trade
                for sp, sv in self.quoted_sell_orders.items():
                    if remaining <= 0:
                        break
                    qty = min(sv, remaining)
                    self.buy(sp, qty)
                    remaining -= qty
            elif delta_trade < 0:
                remaining = -delta_trade
                for bp, bv in self.quoted_buy_orders.items():
                    if remaining <= 0:
                        break
                    qty = min(bv, remaining)
                    self.sell(bp, qty)
                    remaining -= qty
            self.new_traderData[self.name]["last_hedge_ts"] = self.timestamp

        return {self.name: self.orders}


# =============================================================================
# Orchestrator
# =============================================================================

def _decode_traderData(raw: str):
    if raw:
        try:
            return jsonpickle.decode(raw)
        except Exception:
            return default_traderData()
    return default_traderData()


def _resolve_day_offset(last_meta: dict, timestamp: int) -> int:
    day_offset = last_meta.get("day_offset", 0)
    last_ts = last_meta.get("last_timestamp")
    if last_ts is not None and timestamp < last_ts:
        day_offset += 1
    return day_offset


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        last_traderData = _decode_traderData(state.traderData)
        new_traderData = default_traderData()

        last_meta = last_traderData.get("_meta", {})
        day_offset = _resolve_day_offset(last_meta, state.timestamp)
        tte_days = TTE_DAYS_AT_ROUND_START - day_offset - state.timestamp / TICKS_PER_DAY
        tte_days = max(tte_days, 1e-6)
        tte_years = tte_days / TRADING_DAYS_PER_YEAR

        hydro = MeanReversionTrader(HYDROGEL, state, new_traderData, last_traderData)
        result[HYDROGEL] = hydro.get_orders()[HYDROGEL]

        S = None
        vee_od = state.order_depths.get(UNDERLYING)
        if vee_od is not None and vee_od.buy_orders and vee_od.sell_orders:
            best_bid = max(vee_od.buy_orders.keys())
            best_ask = min(vee_od.sell_orders.keys())
            S = (best_bid + best_ask) / 2.0

        option_traders = []
        for v in TRADED_VOUCHERS:
            ot = OptionTrader(v, state, new_traderData, last_traderData, S, tte_years)
            result[v] = ot.get_orders()[v]
            option_traders.append(ot)

        hedger = HedgeTrader(UNDERLYING, state, new_traderData, last_traderData, option_traders)
        result[UNDERLYING] = hedger.get_orders()[UNDERLYING]

        for t in [hydro, hedger] + option_traders:
            t.update_traderData()
        new_traderData["_meta"]["day_offset"] = day_offset
        new_traderData["_meta"]["last_timestamp"] = state.timestamp

        return result, 0, jsonpickle.encode(new_traderData)

    def bid(self):
        return 0
