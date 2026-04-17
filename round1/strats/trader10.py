from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle
import numpy as np


# ASH: Wall-mid fair value + take/clear/make | IPR: buy-and-hold

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
POS_LIMITS = {
    'ASH_COATED_OSMIUM': 80,
    "INTARIAN_PEPPER_ROOT": 80
}
PARAMS = {
    "ASH_COATED_OSMIUM": {
      "take_margin": 2,
      "clear_margin": 5, "make_margin": 2
    },
    "INTARIAN_PEPPER_ROOT":{
        "slope": 0.001, "intercept": 12000.0,
        "take_margin": 1, "clear_margin": 2,
        "make_margin": 2
    }
}

# define default traderData
def default_traderData():
    return {product: {} for product in PRODUCTS}

# base ProductTrader
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
        """
        unravel order_depths into {price:order} object for each side of the orderbook.
        We normalize so that order quantity is always POSITIVE.
        """
        order_depth = None
        try:
            order_depth : OrderDepth = self.state.order_depths[self.name]
        except Exception:
            pass

        buy_orders, sell_orders = {}, {}
        try:
            buy_orders = {bp : abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key = lambda x: x[0], reverse=True)}
            sell_orders ={sp : abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key = lambda x: x[0])}
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
        return PARAMS.get(self.name, {}).get("clear_margin",0)

    def _get_make_margin(self):
        return PARAMS.get(self.name, {}).get("make_margin",1)

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


class WallMidTrader(ProductTrader):
    def __init__(self, state, new_traderData):
        super().__init__("ASH_COATED_OSMIUM", state, new_traderData)
        self.fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        """
        Wall-mid: average the price with the largest volume on each side.
        The "wall" is the level where the most liquidity sits — a better
        anchor than simple mid when book depth is asymmetric.
        """
        if not self.quoted_buy_orders or not self.quoted_sell_orders:
            mid = self.compute_mid_price()
            return round(mid) if mid else 10000

        # price with max volume on bid side
        wall_bid = max(self.quoted_buy_orders.items(), key=lambda x: x[1])[0]
        # price with max volume on ask side
        wall_ask = max(self.quoted_sell_orders.items(), key=lambda x: x[1])[0]

        return round((wall_bid + wall_ask) / 2.0)

    def get_orders(self):
        # STEP 1: take orders first
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.fair_value + self.take_margin:
                break
            self.sell(bp, bv)
        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.fair_value - self.take_margin:
                break
            self.buy(sp, sv)

        # STEP 2: clear
        if self.expected_position > 0:
            clear_volume = min(self.expected_position, self.max_allowed_sell_volume)
            clear_price = self.fair_value + self.clear_margin
            self.sell(clear_price, clear_volume)
        elif self.expected_position < 0:
            clear_volume = min(-self.expected_position, self.max_allowed_buy_volume)
            clear_price = self.fair_value - self.clear_margin
            self.buy(clear_price, clear_volume)

        # STEP 3: make
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()

        if ask_price is not None and self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume)
        if bid_price is not None and self.max_allowed_buy_volume > 0:
            self.buy(bid_price, self.max_allowed_buy_volume)

        return {self.name : self.orders}


class LinearTrendTrader(ProductTrader):
    def __init__(self, state, new_traderData):
        super().__init__("INTARIAN_PEPPER_ROOT", state, new_traderData)
        self.alpha = PARAMS[self.name]["intercept"]
        self.beta = PARAMS[self.name]["slope"]
        self.fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        """
        fair value is a linear relationship with respect global time index
        fair = alpha + beta * g_time_index
        where
            g_time_index = day_offset * 1e6 + timestamp

        we reverse engineer day_offset at first iteration and save it in traderData
        """
        day_offset = self.last_traderData[self.name].get("day_offset", None)
        if (day_offset is None):
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
        # buy at aggressive prices
        agg_sp = list(self.quoted_sell_orders.keys())[-1]
        self.buy(agg_sp, self.position_limit)

        return {self.name : self.orders}

class Trader:
    def run(self, state: TradingState):
        result : Dict[str, List[Order]] = {}
        new_traderData = default_traderData()

        wallmid_trader = WallMidTrader(state, new_traderData)
        lin_trend_trader = LinearTrendTrader(state, new_traderData)

        result.update(wallmid_trader.get_orders())
        result.update(lin_trend_trader.get_orders())

        wallmid_trader.update_traderData()
        lin_trend_trader.update_traderData()

        new_traderData.update(wallmid_trader.new_traderData)
        new_traderData.update(lin_trend_trader.new_traderData)

        traderData = jsonpickle.encode(new_traderData)
        conversions = 0
        return result, conversions, traderData
