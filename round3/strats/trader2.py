from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle


# =============================================================================
# Product universe
# =============================================================================

DELTA1_PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
UNDERLYING = "VELVETFRUIT_EXTRACT"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{K}" for K in STRIKES]
VOUCHER_STRIKE = {f"VEV_{K}": K for K in STRIKES}

# Empirically pinned at mid=0.5 across all historical days — illiquid/dead.
DEAD_VOUCHERS = {"VEV_6000", "VEV_6500"}
TRADED_VOUCHERS = [v for v in VOUCHERS if v not in DEAD_VOUCHERS]

ALL_PRODUCTS = DELTA1_PRODUCTS + VOUCHERS

POS_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{v: 300 for v in VOUCHERS},
}

PARAMS = {
    # Wider spread (~16 ticks), price anchored near 9990 → treat as near-static.
    "HYDROGEL_PACK": {
        "ema_alpha": 0.05,
        "static_fv": 9990,
        "fv_method_weights": [1.0, 0.0],   # [static, ema]
        "take_margin": 2,
        "clear_margin": 6,
        "make_margin": 4,
    },
    # Tight spread (~5 ticks), drifts across days → pure EMA.
    "VELVETFRUIT_EXTRACT": {
        "ema_alpha": 0.05,
        "static_fv": 5250,
        "fv_method_weights": [0.0, 1.0],   # [static, ema]
        "take_margin": 1,
        "clear_margin": 3,
        "make_margin": 2,
    },
}


def default_traderData():
    return {product: {} for product in ALL_PRODUCTS}


# =============================================================================
# Base class
# =============================================================================

class ProductTrader:
    def __init__(self, name, state, new_traderData):
        self.orders: List[Order] = []
        self.name = name
        self.state = state
        self.timestamp = state.timestamp
        self.new_traderData = new_traderData
        self.new_traderData.setdefault(name, {})

        self.last_traderData = self._get_last_traderData()
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

    def _get_last_traderData(self):
        if self.state.traderData:
            try:
                return jsonpickle.decode(self.state.traderData)
            except Exception:
                return default_traderData()
        return default_traderData()

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

    def compute_make_ask_price(self):
        fair_ask = self.fair_value + self.make_margin
        best_ask = self.get_best_ask()
        if best_ask is None:
            return fair_ask
        return best_ask - 1 if best_ask > fair_ask else fair_ask

    def compute_make_bid_price(self):
        fair_bid = self.fair_value - self.make_margin
        best_bid = self.get_best_bid()
        if best_bid is None:
            return fair_bid
        return best_bid + 1 if best_bid < fair_bid else fair_bid

    def update_traderData(self):
        self.new_traderData[self.name]["last_timestamp"] = self.timestamp


# =============================================================================
# MeanReversionTrader — used for BOTH delta-1 products
# =============================================================================

class MeanReversionTrader(ProductTrader):
    """
    Fair value = w_static * static_fv + w_ema * ema(mid, alpha).
    Take any book level better than fair ± take_margin, then post two-sided
    quotes at fair ± make_margin (or one-tick-in-front of the book if tighter).
    """

    def __init__(self, name, state, new_traderData):
        super().__init__(name, state, new_traderData)
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
        # Take mispriced book levels
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.fair_value + self.take_margin:
                break
            self.sell(bp, bv)

        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.fair_value - self.take_margin:
                break
            self.buy(sp, sv)

        # Post both sides
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()
        if self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume)
        if self.max_allowed_buy_volume > 0:
            self.buy(bid_price, self.max_allowed_buy_volume)

        return {self.name: self.orders}


# =============================================================================
# OptionArbitrageTrader — lifts asks < max(S-K,0), hits bids > S
# =============================================================================

class OptionArbitrageTrader(ProductTrader):
    """
    European-call no-arb bounds with r=0, q=0:
        max(S - K, 0)  <=  C  <=  S
    Any book level outside these bounds is free edge — take it. No passive quotes.
    """

    def __init__(self, name, state, new_traderData, underlying_mid):
        super().__init__(name, state, new_traderData)
        self.underlying_mid = underlying_mid
        self.strike = VOUCHER_STRIKE[name]

    def get_orders(self):
        if self.underlying_mid is None:
            return {self.name: self.orders}

        S = self.underlying_mid
        intrinsic = max(S - self.strike, 0.0)

        # Buy asks strictly below intrinsic
        for sp, sv in self.quoted_sell_orders.items():
            if sp >= intrinsic:
                break
            self.buy(sp, sv)

        # Sell bids strictly above upper bound S
        for bp, bv in self.quoted_buy_orders.items():
            if bp <= S:
                break
            self.sell(bp, bv)

        return {self.name: self.orders}


# =============================================================================
# Orchestrator
# =============================================================================

class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_traderData = default_traderData()

        traders = [MeanReversionTrader(name, state, new_traderData)
                   for name in DELTA1_PRODUCTS]

        # VEE mid drives voucher no-arb bounds. Use the VEE trader's mid so
        # all logic agrees on the same reference.
        vee_trader = next(t for t in traders if t.name == UNDERLYING)
        underlying_mid = vee_trader.compute_mid_price()

        for v in TRADED_VOUCHERS:
            traders.append(OptionArbitrageTrader(v, state, new_traderData, underlying_mid))

        for t in traders:
            result.update(t.get_orders())
            t.update_traderData()

        return result, 0, jsonpickle.encode(new_traderData)

    def bid(self):
        # Stub for compatibility; round 3 does not use the Market Access Fee auction.
        return 0
