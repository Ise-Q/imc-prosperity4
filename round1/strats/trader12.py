from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle


# ASH: DynamicTrader (configurable fair value methods) | IPR: buy-and-hold

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
POS_LIMITS = {
    'ASH_COATED_OSMIUM': 80,
    "INTARIAN_PEPPER_ROOT": 80
}
PARAMS = {
    "ASH_COATED_OSMIUM": {
        "take_margin": 1, "clear_margin": 5, "make_margin": 2,
        "fv_default": 10000,
        # each entry: (method_name, weight, method_params)
        "fv_methods": [
            ("static",   0.5, {"value": 10000}),
            ("ema",      0.3, {"alpha": 0.05}),
            ("vwap",     0.1, {"window": 500}),
            ("wall_mid", 0.1, {}),
        ],
    },
    "INTARIAN_PEPPER_ROOT": {
        "slope": 0.001, "intercept": 12000.0,
        "take_margin": 1, "clear_margin": 2,
        "make_margin": 2
    }
}


def default_traderData():
    return {product: {} for product in PRODUCTS}


class ProductTrader:
    def __init__(self, name, state, new_traderData):
        self.orders = []
        self.name = name
        self.state = state
        self.timestamp = self.state.timestamp
        self.new_traderData = new_traderData

        self.last_traderData = self._get_last_traderData()
        self.position_limit = POS_LIMITS.get(name, 0)

        self.starting_position = self._get_current_position()
        self.expected_position = self.starting_position

        self.quoted_buy_orders, self.quoted_sell_orders = self._get_order_depth()
        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self._get_max_allowed_volume()

        self.take_margin = self._get_take_margin()
        self.clear_margin = self._get_clear_margin()
        self.make_margin = self._get_make_margin()

    def _get_last_traderData(self):
        if self.state.traderData and self.state.traderData != "":
            try:
                last_traderData = jsonpickle.decode(self.state.traderData)
            except Exception:
                last_traderData = default_traderData()
            return last_traderData
        else:
            return default_traderData()

    def _get_current_position(self):
        return self.state.position.get(self.name, 0)

    def _get_order_depth(self):
        order_depth = None
        try:
            order_depth: OrderDepth = self.state.order_depths[self.name]
        except Exception:
            pass

        buy_orders, sell_orders = {}, {}
        try:
            buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
            sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except Exception:
            pass

        return buy_orders, sell_orders

    def _get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.starting_position
        max_allowed_sell_volume = self.position_limit + self.starting_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def _get_take_margin(self):
        return PARAMS.get(self.name, {}).get("take_margin", 1)

    def _get_clear_margin(self):
        return PARAMS.get(self.name, {}).get("clear_margin", 0)

    def _get_make_margin(self):
        return PARAMS.get(self.name, {}).get("make_margin", 1)

    def get_best_bid(self):
        if self.quoted_buy_orders:
            return next(iter(self.quoted_buy_orders))

    def get_best_ask(self):
        if self.quoted_sell_orders:
            return next(iter(self.quoted_sell_orders))

    def buy(self, price, volume):
        abs_volume = min(int(abs(volume)), self.max_allowed_buy_volume)
        self.max_allowed_buy_volume -= abs_volume
        order = Order(self.name, int(price), abs_volume)
        self.orders.append(order)
        self.expected_position += abs_volume

    def sell(self, price, volume):
        abs_volume = min(int(abs(volume)), self.max_allowed_sell_volume)
        self.max_allowed_sell_volume -= abs_volume
        order = Order(self.name, int(price), -abs_volume)
        self.orders.append(order)
        self.expected_position -= abs_volume

    def update_traderData(self):
        self.new_traderData[self.name]["last_timestamp"] = self.timestamp

    def compute_mid_price(self):
        bb = self.get_best_bid()
        bo = self.get_best_ask()
        if bb and bo:
            return (bb + bo) / 2.0

    def compute_make_ask_price(self):
        fair_ask_price = self.fair_value + self.make_margin
        if not self.quoted_sell_orders:
            return fair_ask_price
        best_ask_price = next(iter(self.quoted_sell_orders))
        if best_ask_price > fair_ask_price:
            return best_ask_price - 1
        else:
            return fair_ask_price

    def compute_make_bid_price(self):
        fair_bid_price = self.fair_value - self.make_margin
        if not self.quoted_buy_orders:
            return fair_bid_price
        best_bid_price = next(iter(self.quoted_buy_orders))
        if best_bid_price < fair_bid_price:
            return best_bid_price + 1
        else:
            return fair_bid_price

    def get_orders(self):
        # STEP 1: take orders
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.fair_value + self.take_margin:
                break
            self.sell(bp, bv)
        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.fair_value - self.take_margin:
                break
            self.buy(sp, sv)

        # STEP 2: clear inventory
        if self.expected_position > 0:
            clear_volume = min(self.expected_position, self.max_allowed_sell_volume)
            self.sell(self.fair_value + self.clear_margin, clear_volume)
        elif self.expected_position < 0:
            clear_volume = min(-self.expected_position, self.max_allowed_buy_volume)
            self.buy(self.fair_value - self.clear_margin, clear_volume)

        # STEP 3: make orders
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()
        if ask_price is not None and self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume)
        if bid_price is not None and self.max_allowed_buy_volume > 0:
            self.buy(bid_price, self.max_allowed_buy_volume)

        return {self.name: self.orders}


class DynamicTrader(ProductTrader):
    def __init__(self, name, state, new_traderData):
        super().__init__(name, state, new_traderData)
        self.fv_methods = PARAMS[self.name]["fv_methods"]
        self.fv_default = PARAMS[self.name].get("fv_default", 10000)
        self._method_states = {}
        self.fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        dispatch = {
            "static":   self._compute_static,
            "ema":      self._compute_ema,
            "vwap":     self._compute_vwap,
            "wall_mid": self._compute_wall_mid,
            "sma":      self._compute_sma,
        }

        values = []
        weights = []

        for method_name, weight, method_params in self.fv_methods:
            prev_state = self.last_traderData[self.name].get(method_name, {})
            fv, new_state = dispatch[method_name](method_params, prev_state)
            self._method_states[method_name] = new_state

            if fv is not None:
                values.append(fv)
                weights.append(weight)

        if not values:
            return self.fv_default

        total_w = sum(weights)
        return round(sum(v * w / total_w for v, w in zip(values, weights)))

    def _compute_static(self, params, prev_state):
        return (params["value"], {})

    def _compute_ema(self, params, prev_state):
        alpha = params.get("alpha", 0.05)
        mid = self.compute_mid_price()
        prev_ema = prev_state.get("value", None)

        if mid is None:
            ema = prev_ema
        elif prev_ema is None:
            ema = mid
        else:
            ema = alpha * mid + (1 - alpha) * prev_ema

        if ema is None:
            return (None, {})
        return (ema, {"value": ema})

    def _compute_vwap(self, params, prev_state):
        window = params.get("window", 500)
        trades = self.state.market_trades.get(self.name, [])
        tick_pv = sum(t.price * t.quantity for t in trades)
        tick_v = sum(t.quantity for t in trades)

        history = prev_state.get("history", [])
        if tick_v > 0:
            history.append([tick_pv, tick_v])
        history = history[-window:]

        total_pv = sum(pv for pv, v in history)
        total_v = sum(v for pv, v in history)

        fv = total_pv / total_v if total_v > 0 else None
        return (fv, {"history": history})

    def _compute_wall_mid(self, params, prev_state):
        if not self.quoted_buy_orders or not self.quoted_sell_orders:
            return (None, {})
        wall_bid = max(self.quoted_buy_orders.items(), key=lambda x: x[1])[0]
        wall_ask = max(self.quoted_sell_orders.items(), key=lambda x: x[1])[0]
        return ((wall_bid + wall_ask) / 2.0, {})

    def _compute_sma(self, params, prev_state):
        window = params.get("window", 200)
        warmup = params.get("warmup_ticks", 0)
        mid = self.compute_mid_price()

        history = prev_state.get("history", [])
        if mid is not None:
            history.append(mid)
        history = history[-window:]

        if self.timestamp < warmup or len(history) == 0:
            return (None, {"history": history})

        return (sum(history) / len(history), {"history": history})

    def update_traderData(self):
        super().update_traderData()
        for method_name, new_state in self._method_states.items():
            self.new_traderData[self.name][method_name] = new_state


class LinearTrendTrader(ProductTrader):
    def __init__(self, name, state, new_traderData):
        super().__init__(name, state, new_traderData)
        self.alpha = PARAMS[self.name]["intercept"]
        self.beta = PARAMS[self.name]["slope"]
        self.fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        day_offset = self.last_traderData[self.name].get("day_offset", None)
        if day_offset is None:
            day_offset = round((self.compute_mid_price() - self.alpha) / (self.beta * 1_000_000))
            g_time_index = day_offset * 1_000_000 + self.timestamp
            fair = self.alpha + self.beta * g_time_index
            self.new_traderData[self.name]["day_offset"] = day_offset
        else:
            if self.timestamp < self.last_traderData[self.name]["last_timestamp"]:
                day_offset += 1
            g_time_index = day_offset * 1_000_000 + self.timestamp
            fair = self.alpha + self.beta * g_time_index
            self.new_traderData[self.name]["day_offset"] = day_offset
        return round(fair)

    def get_orders(self):
        agg_sp = list(self.quoted_sell_orders.keys())[-1]
        self.buy(agg_sp, self.position_limit)
        return {self.name: self.orders}


PRODUCT_TRADERS = {
    "ASH_COATED_OSMIUM": DynamicTrader,
    "INTARIAN_PEPPER_ROOT": LinearTrendTrader,
}


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_traderData = default_traderData()

        traders = []
        for product in PRODUCTS:
            trader_class = PRODUCT_TRADERS[product]
            traders.append(trader_class(product, state, new_traderData))

        for t in traders:
            result.update(t.get_orders())
            t.update_traderData()

        traderData = jsonpickle.encode(new_traderData)
        conversions = 0
        return result, conversions, traderData
