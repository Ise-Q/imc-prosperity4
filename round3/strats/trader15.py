from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# trader8 — trader7 + direct VEE passive MM
# Hedge stays disabled (saves ~11k spread cost). With VEE budget free,
# post passive symmetric quotes around the slow VEE EMA. Tight inventory
# clearing keeps total exposure bounded so the unhedged option delta
# does not get amplified.
#
# trader7 — trader6 with VEE hedger DISABLED
# Reasoning: trader6 spent -11k on hedge spread cost market-walking VEE.
# With ~417 long delta from VEV_4000+VEV_4500 (300 each) and VEE drifting only
# +9 ticks net across the 3 historical days, unhedged exposure is roughly
# zero-EV in expectation but saves the ~11k spread tax.
#
# trader6 — trader5 + HYDROGEL MM size 0.1 → 0.25
# Hypothesis: top-of-book fills should mostly survive 2.5x sizing.
#
# trader5 — trader2 baseline preserved + smarter VEE side
#
# Key insight from trader3/trader4 failures:
#   trader2 HYDROGEL is a +27k profit centre. ANY take-leg there bleeds because
#   the EMA fair value lags the trending mid — adding "take" actually buys
#   into downtrends. Don't touch HYDROGEL.
#
# The real bleed is on VEE:
#   VEV_4000 (300 long × Δ=0.73) + VEV_4500 (300 long × Δ=0.66) = ~417 long
#   delta. Hedger maxes VEE at -200 (limit), leaving 217 unhedged. The hedger
#   chases delta with MARKET orders, paying 4-5 ticks of spread per trade →
#   -12k VEE bleed in trader2.
#
# trader5 changes (all confined to VEE / hedger):
#   1. Hedge primarily with LIMIT (passive) orders — only walk the book if
#      |unhedged delta| exceeds a hard threshold. Saves spread cost.
#   2. Add VEE range overlay (user-seeded): lean residual target toward the
#      mean-reverting centre when mid is far from EMA. EDA: VEE std=15.6, range
#      5198-5300, no momentum after detrending → range trading is plausible.


# =============================================================================
# Universe (identical to trader2)
# =============================================================================

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{K}" for K in STRIKES]
VOUCHER_STRIKE = {f"VEV_{K}": K for K in STRIKES}

DEAD_VOUCHERS = {"VEV_6000", "VEV_6500"}
TRADED_VOUCHERS = [v for v in VOUCHERS if v not in DEAD_VOUCHERS]

ALL_PRODUCTS = [HYDROGEL, UNDERLYING] + VOUCHERS

POS_LIMITS = {
    HYDROGEL: 200,
    UNDERLYING: 200,
    **{v: 300 for v in VOUCHERS},
}

PARAMS = {
    HYDROGEL: {
        "ema_alpha": 0.05,
        "static_fv": 9990,
        "fv_method_weights": [0.0, 1.0],
        "take_margin": 2,
        "clear_margin": 6,
        "make_margin": 6,
    },
}

HEDGE_PARAMS = {
    "vee_ema_alpha": 0.005,          # very slow EMA → range center
    "max_make_size": 30,
    # ---- Passive hedge band — DISABLED in trader7+ ----
    "hedge_passive_thresh": 10_000,
    "hedge_market_thresh": 10_000,
    "passive_step_in_ticks": 1,
    # ---- Direct VEE MM (trader8/9/10) ----
    "mm_enabled": True,
    "mm_make_margin": 1,             # tighter than trader8/9 (was 2)
    "mm_size_frac": 0.35,
    "mm_clear_pos": 80,
    # ---- VEE range overlay ----
    "range_enabled": True,
    "range_std": 15.0,
    "range_entry_k": 0.7,            # enter when |dev| > 0.7σ
    "range_max_lean": 60,            # max additional inventory leaned toward mean
}

OPTION_PARAMS = {
    "iv_ema_alpha": 0.1,
    "spread_ema_alpha": 0.05,
    "default_iv": 0.23,
    "make_skew_per_unit": 0.5,
    "min_make_half_spread": 1,
    # ---- Liquid voucher MM (trader9) ----
    "liquid_mm_enabled": True,
    "liquid_mm_strikes": {5200, 5300, 5400, 5500},
    "liquid_mm_size_frac": 0.10,
    "liquid_mm_min_half_spread": 1,
    "liquid_mm_skew": 1.0,
    # ---- Liquid voucher TAKE (trader10) ----
    "liquid_take_enabled": True,
    "liquid_take_edge": 1.0,         # lift ask if best_ask <= bs_fair - edge
}

TTE_DAYS_AT_ROUND_START = 5
TICKS_PER_DAY = 1_000_000
TRADING_DAYS_PER_YEAR = 365.0


def default_traderData():
    d = {product: {} for product in ALL_PRODUCTS}
    d["_meta"] = {}
    return d


# =============================================================================
# Black-Scholes
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
# Base
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
# HYDROGEL trader — IDENTICAL to trader2's MeanReversionTrader
# =============================================================================

class MeanReversionTrader(ProductTrader):
    def __init__(self, name, state, new_traderData, last_traderData):
        super().__init__(name, state, new_traderData, last_traderData)
        p = PARAMS[name]
        self.ema_alpha = p["ema_alpha"]
        self.static_fv = p["static_fv"]
        self.fv_method_weights = self._normalize_weights(p["fv_method_weights"])
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
        ask_price = self.compute_make_ask_price(self.fair_value)
        bid_price = self.compute_make_bid_price(self.fair_value)
        if self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume * 0.25)
        if self.max_allowed_buy_volume > 0:
            self.buy(bid_price, self.max_allowed_buy_volume * 0.25)

        half = 0.5 * self.position_limit
        if self.expected_position > half:
            qty = max(int(self.expected_position * 0.1), int(self.position_limit * 0.05))
            self.sell(self.fair_value, qty)
        elif self.expected_position < -half:
            qty = max(int(-self.expected_position * 0.1), int(self.position_limit * 0.05))
            self.buy(self.fair_value, qty)

        return {self.name: self.orders}


# =============================================================================
# OptionTrader — IDENTICAL to trader2 (no MM addition; deep-ITM intrinsic take)
# =============================================================================

class OptionTrader(ProductTrader):
    def __init__(self, name, state, new_traderData, last_traderData,
                 underlying_mid: Optional[float], tte_years: float):
        super().__init__(name, state, new_traderData, last_traderData)
        self.S = underlying_mid
        self.T = tte_years
        self.K = VOUCHER_STRIKE[name]
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

        for sp, sv in self.quoted_sell_orders.items():
            if sp >= intrinsic:
                break
            self.buy(sp, sv)
        for bp, bv in self.quoted_buy_orders.items():
            if bp <= self.S:
                break
            self.sell(bp, bv)

        self._post_take_position = self.expected_position

        if self.live_iv is None:
            mid_ref = self.mid if self.mid is not None else self.prev_mid
            if mid_ref is not None:
                fair = BlackScholes.call_price(self.S, self.K, self.T, self.iv_used)
                if fair > 0:
                    half = max(0.5 * (self.spread_ema or 2.0),
                               OPTION_PARAMS["min_make_half_spread"])
                    skew = (OPTION_PARAMS["make_skew_per_unit"]
                            * self.expected_position / max(self.position_limit, 1))
                    bid_price = round(fair - half - skew)
                    ask_price = round(fair + half - skew)
                    bid_price = self.compute_make_bid_price(bid_price)
                    ask_price = self.compute_make_ask_price(ask_price)
                    if self.max_allowed_buy_volume > 0:
                        self.buy(bid_price, self.max_allowed_buy_volume)
                    if self.max_allowed_sell_volume > 0:
                        self.sell(ask_price, self.max_allowed_sell_volume)

        # ---- trader10: voucher TAKE leg on liquid strikes ----
        if (OPTION_PARAMS.get("liquid_take_enabled")
                and self.K in OPTION_PARAMS["liquid_mm_strikes"]
                and self.iv_ema is not None):
            bs_fair = BlackScholes.call_price(self.S, self.K, self.T, self.iv_ema)
            edge = OPTION_PARAMS["liquid_take_edge"]
            for sp, sv in self.quoted_sell_orders.items():
                if sp <= bs_fair - edge:
                    self.buy(sp, sv)
                else:
                    break
            for bp, bv in self.quoted_buy_orders.items():
                if bp >= bs_fair + edge:
                    self.sell(bp, bv)
                else:
                    break
            self._post_take_position = self.expected_position

        # ---- trader9: small symmetric MM on liquid strikes (always-on) ----
        if (OPTION_PARAMS.get("liquid_mm_enabled")
                and self.K in OPTION_PARAMS["liquid_mm_strikes"]
                and self.live_iv is not None):
            fair = BlackScholes.call_price(self.S, self.K, self.T, self.iv_used)
            if fair > 0:
                half = max(0.5 * (self.spread_ema or 2.0),
                           OPTION_PARAMS["liquid_mm_min_half_spread"])
                skew = (OPTION_PARAMS["liquid_mm_skew"]
                        * self.expected_position / max(self.position_limit, 1))
                bid_price = round(fair - half - skew)
                ask_price = round(fair + half - skew)
                bid_price = self.compute_make_bid_price(bid_price)
                ask_price = self.compute_make_ask_price(ask_price)
                size_frac = OPTION_PARAMS["liquid_mm_size_frac"]
                if self.max_allowed_buy_volume > 0:
                    self.buy(bid_price, int(self.max_allowed_buy_volume * size_frac))
                if self.max_allowed_sell_volume > 0:
                    self.sell(ask_price, int(self.max_allowed_sell_volume * size_frac))

        return {self.name: self.orders}


# =============================================================================
# HedgeTrader — passive limit hedging + range overlay
# =============================================================================

class HedgeTrader(ProductTrader):
    def __init__(self, name, state, new_traderData, last_traderData, option_traders):
        super().__init__(name, state, new_traderData, last_traderData)
        self.option_traders = option_traders

        prev = self.last_traderData.get(self.name, {})
        self.prev_vee_ema: Optional[float] = prev.get("vee_ema")

        self.mid = self.compute_mid_price()
        a = HEDGE_PARAMS["vee_ema_alpha"]
        if self.mid is None:
            self.vee_ema = self.prev_vee_ema
        elif self.prev_vee_ema is None:
            self.vee_ema = self.mid
        else:
            self.vee_ema = a * self.mid + (1 - a) * self.prev_vee_ema
        if self.vee_ema is not None:
            self.new_traderData[self.name]["vee_ema"] = self.vee_ema

    def _range_lean(self) -> int:
        if not HEDGE_PARAMS["range_enabled"]:
            return 0
        if self.mid is None or self.vee_ema is None:
            return 0
        dev = self.mid - self.vee_ema
        thresh = HEDGE_PARAMS["range_entry_k"] * HEDGE_PARAMS["range_std"]
        if abs(dev) < thresh:
            return 0
        scale = min(1.0, (abs(dev) - thresh) / thresh)
        sign = -1 if dev > 0 else +1   # mid above center → want short
        return int(round(sign * scale * HEDGE_PARAMS["range_max_lean"]))

    def _do_hedge(self):
        """Hedge legs (disabled when thresholds are 10_000)."""
        opt_delta = sum(t.position_delta for t in self.option_traders)
        target = -opt_delta + self._range_lean()
        target_int = max(-self.position_limit, min(self.position_limit, int(round(target))))
        delta_trade = target_int - self.starting_position
        if delta_trade == 0:
            return
        passive_thresh = HEDGE_PARAMS["hedge_passive_thresh"]
        market_thresh = HEDGE_PARAMS["hedge_market_thresh"]
        step = HEDGE_PARAMS["passive_step_in_ticks"]
        bb, ba = self.get_best_bid(), self.get_best_ask()
        if bb is None or ba is None:
            return
        abs_trade = abs(delta_trade)
        if abs_trade <= passive_thresh:
            return
        if abs_trade <= market_thresh:
            if delta_trade > 0:
                self.buy(bb + step, delta_trade)
            else:
                self.sell(ba - step, -delta_trade)
            return
        if delta_trade > 0:
            remaining = delta_trade
            for sp, sv in self.quoted_sell_orders.items():
                if remaining <= 0:
                    break
                qty = min(sv, remaining)
                self.buy(sp, qty)
                remaining -= qty
        else:
            remaining = -delta_trade
            for bp, bv in self.quoted_buy_orders.items():
                if remaining <= 0:
                    break
                qty = min(bv, remaining)
                self.sell(bp, qty)
                remaining -= qty

    def _post_mm(self):
        """Symmetric passive MM around vee_ema."""
        if not HEDGE_PARAMS.get("mm_enabled"):
            return
        if self.vee_ema is None:
            return
        fair = round(self.vee_ema)
        margin = HEDGE_PARAMS["mm_make_margin"]
        clear_pos = HEDGE_PARAMS["mm_clear_pos"]
        size_frac = HEDGE_PARAMS["mm_size_frac"]

        # Inventory clearing : take liquidity to flatten when over band
        if self.expected_position > clear_pos:
            qty = min(self.expected_position - clear_pos, self.expected_position // 2)
            self.sell(fair, qty)
        elif self.expected_position < -clear_pos:
            qty = min(-self.expected_position - clear_pos, -self.expected_position // 2)
            self.buy(fair, qty)

        # 4-level ladder: capture progressively wider deviation tiers
        bb, ba = self.get_best_bid(), self.get_best_ask()
        levels = [
            (margin,      size_frac * 0.35),
            (margin + 2,  size_frac * 0.40),
            (margin + 5,  size_frac * 0.50),
            (margin + 9,  size_frac * 0.60),
        ]
        for m, frac in levels:
            fair_bid = fair - m
            fair_ask = fair + m
            bid_price = (bb + 1) if (bb is not None and bb < fair_bid) else fair_bid
            ask_price = (ba - 1) if (ba is not None and ba > fair_ask) else fair_ask
            if self.max_allowed_buy_volume > 0:
                self.buy(bid_price, int(self.max_allowed_buy_volume * frac))
            if self.max_allowed_sell_volume > 0:
                self.sell(ask_price, int(self.max_allowed_sell_volume * frac))

    def get_orders(self):
        self._do_hedge()
        self._post_mm()
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
