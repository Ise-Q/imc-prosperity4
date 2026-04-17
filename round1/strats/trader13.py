from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle
import statistics


# ASH: DynamicTrader with adaptive margins + inventory/momentum skew + informed-flow
# IPR: buy-and-hold (unchanged from trader12)

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
POS_LIMITS = {
    'ASH_COATED_OSMIUM': 80,
    "INTARIAN_PEPPER_ROOT": 80
}
PARAMS = {
    "ASH_COATED_OSMIUM": {
        # Legacy fallback margins (used when margin_mode == "fixed")
        "take_margin": 1, "clear_margin": 5, "make_margin": 2,
        "fv_default": 10000,

        # Feature 2: informed_flow enters fv_methods
        "fv_methods": [
            ("static",        0.45, {"value": 10000}),
            ("ema",           0.25, {"alpha": 0.05}),
            ("vwap",          0.10, {"window": 500}),
            ("wall_mid",      0.10, {}),
            ("informed_flow", 0.10, {"z": 2.0, "alpha": 0.2}),
        ],

        # Feature 1: adaptive margins
        "margin_mode": "fixed",     # "fixed" restores trader12 behavior
        "vol_window": 100,
        "vol_min": 1.0,                # floor on rolling std
        "k_take":  0.3, "take_floor":  1, "take_cap":  3,
        "k_clear": 1.0, "clear_floor": 3, "clear_cap": 8,
        "k_make":  0.5, "make_floor":  2, "make_cap":  5,

        # Feature 3: skews (both ON)
        "gamma_inv": 1.5,              # ticks of inventory shift at |pos|==limit
        "gamma_mom": 0.5,              # ticks per 1-sigma residual
        "skew_cap":  2,                # max |eff_fair - fair_value|
    },
    "INTARIAN_PEPPER_ROOT": {
        "slope": 0.001, "intercept": 12000.0,
        "take_margin": 1, "clear_margin": 2,
        "make_margin": 2
    }
}


def default_traderData():
    return {product: {} for product in PRODUCTS}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


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

        # fair_value + eff_fair are set by subclasses after compute_fair_value().
        # Default eff_fair = fair_value for any subclass that doesn't skew.
        self.fair_value = None
        self.eff_fair = None
        self.rolling_std = PARAMS.get(self.name, {}).get("vol_min", 1.0)

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

    # --- Feature 1 / shared: rolling std over a dedicated mid_history buffer ---
    def compute_rolling_std(self):
        """Appends current mid to mid_history (persisted in traderData), trims
        to vol_window, returns max(pstdev, vol_min). Warm-up returns vol_min."""
        params = PARAMS.get(self.name, {})
        window = params.get("vol_window", 100)
        vol_min = params.get("vol_min", 1.0)

        history = list(self.last_traderData[self.name].get("mid_history", []))
        mid = self.compute_mid_price()
        if mid is not None:
            history.append(mid)
        history = history[-window:]

        self.new_traderData[self.name]["mid_history"] = history

        if len(history) < 2:
            return vol_min
        return max(statistics.pstdev(history), vol_min)

    # --- Feature 1: adaptive margins ---
    def compute_adaptive_margins(self, rolling_std):
        """When margin_mode=='adaptive', overwrites self.{take,clear,make}_margin
        with int(clip(k * rolling_std, floor, cap))."""
        params = PARAMS.get(self.name, {})
        if params.get("margin_mode", "fixed") != "adaptive":
            return

        def adapt(k_key, floor_key, cap_key):
            k = params.get(k_key, 0.5)
            floor = params.get(floor_key, 1)
            cap = params.get(cap_key, 10)
            return int(_clip(round(k * rolling_std), floor, cap))

        self.take_margin  = adapt("k_take",  "take_floor",  "take_cap")
        self.clear_margin = adapt("k_clear", "clear_floor", "clear_cap")
        self.make_margin  = adapt("k_make",  "make_floor",  "make_cap")

    # --- Feature 3: skews ---
    def compute_skew(self, rolling_std):
        """Returns (skew_inv, skew_mom). Uses starting_position (pre-trade)."""
        params = PARAMS.get(self.name, {})
        gamma_inv = params.get("gamma_inv", 0.0)
        gamma_mom = params.get("gamma_mom", 0.0)
        vol_min = params.get("vol_min", 1.0)

        # Inventory skew: positive when long -> drops eff_fair -> easier to sell
        if self.position_limit > 0:
            skew_inv = gamma_inv * (self.starting_position / self.position_limit)
        else:
            skew_inv = 0.0

        # Momentum skew: positive residual (mid > fair) -> raise eff_fair -> lean up
        skew_mom = 0.0
        mid = self.compute_mid_price()
        if mid is not None and self.fair_value is not None:
            residual = mid - self.fair_value
            skew_mom = gamma_mom * residual / max(rolling_std, vol_min)

        return (skew_inv, skew_mom)

    def apply_skews(self, fair_value, skew_inv, skew_mom):
        """Returns round(fair - skew_inv + skew_mom), clamped to ±skew_cap."""
        params = PARAMS.get(self.name, {})
        skew_cap = params.get("skew_cap", 0)

        raw_shift = -skew_inv + skew_mom
        if skew_cap > 0:
            raw_shift = _clip(raw_shift, -skew_cap, skew_cap)
        return int(round(fair_value + raw_shift))

    # --- Make-price overrides: quote off eff_fair (not raw fair_value) ---
    def compute_make_ask_price(self):
        fair_ask_price = self.eff_fair + self.make_margin
        if not self.quoted_sell_orders:
            return fair_ask_price
        best_ask_price = next(iter(self.quoted_sell_orders))
        if best_ask_price > fair_ask_price:
            return best_ask_price - 1
        else:
            return fair_ask_price

    def compute_make_bid_price(self):
        fair_bid_price = self.eff_fair - self.make_margin
        if not self.quoted_buy_orders:
            return fair_bid_price
        best_bid_price = next(iter(self.quoted_buy_orders))
        if best_bid_price < fair_bid_price:
            return best_bid_price + 1
        else:
            return fair_bid_price

    def get_orders(self):
        # STEP 1: take — uses RAW fair_value (only act on true mispricing)
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.fair_value + self.take_margin:
                break
            self.sell(bp, bv)
        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.fair_value - self.take_margin:
                break
            self.buy(sp, sv)

        # STEP 2: clear inventory — uses eff_fair (leans in direction of skew)
        if self.expected_position > 0:
            clear_volume = min(self.expected_position, self.max_allowed_sell_volume)
            self.sell(self.eff_fair + self.clear_margin, clear_volume)
        elif self.expected_position < 0:
            clear_volume = min(-self.expected_position, self.max_allowed_buy_volume)
            self.buy(self.eff_fair - self.clear_margin, clear_volume)

        # STEP 3: make — uses eff_fair via compute_make_*_price
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

        # 1) Compute fair value (weighted hybrid of fv_methods).
        self.fair_value = self.compute_fair_value()

        # 2) Rolling std from dedicated mid_history buffer.
        self.rolling_std = self.compute_rolling_std()

        # 3) Adaptive margins (mutates self.{take,clear,make}_margin if enabled).
        self.compute_adaptive_margins(self.rolling_std)

        # 4) Skews -> eff_fair for clear + make steps.
        skew_inv, skew_mom = self.compute_skew(self.rolling_std)
        self.eff_fair = self.apply_skews(self.fair_value, skew_inv, skew_mom)

    def compute_fair_value(self):
        dispatch = {
            "static":        self._compute_static,
            "ema":           self._compute_ema,
            "vwap":          self._compute_vwap,
            "wall_mid":      self._compute_wall_mid,
            "sma":           self._compute_sma,
            "informed_flow": self._compute_informed_flow,
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

    def _compute_informed_flow(self, params, prev_state):
        """EWMA of trade prices that deviate from reference fair by > z * std.

        Uses a locally-computed std from last_traderData['mid_history'] to avoid
        ordering coupling with compute_rolling_std() (which runs AFTER
        compute_fair_value()).
        """
        z = params.get("z", 2.0)
        alpha = params.get("alpha", 0.2)

        ewma = prev_state.get("value", None)

        trades = self.state.market_trades.get(self.name, []) or []
        if not trades:
            # No new trades: pass through prior ewma (or None if never seen).
            if ewma is None:
                return (None, {})
            return (ewma, {"value": ewma})

        # Local std from persisted mid_history (not the buffer we're writing this tick).
        pparams = PARAMS.get(self.name, {})
        vol_min = pparams.get("vol_min", 1.0)
        hist = list(self.last_traderData[self.name].get("mid_history", []))
        if len(hist) >= 2:
            local_std = max(statistics.pstdev(hist), vol_min)
        else:
            local_std = vol_min

        # Reference fair: prior ewma if present, else mid, else skip.
        ref = ewma
        if ref is None:
            ref = self.compute_mid_price()
        if ref is None:
            if ewma is None:
                return (None, {})
            return (ewma, {"value": ewma})

        threshold = z * local_std
        for t in trades:
            if abs(t.price - ref) > threshold:
                if ewma is None:
                    ewma = float(t.price)
                else:
                    ewma = alpha * t.price + (1 - alpha) * ewma

        if ewma is None:
            return (None, {})
        return (ewma, {"value": ewma})

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
        self.eff_fair = self.fair_value  # no skew; keeps base-class compute_make_* happy

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
