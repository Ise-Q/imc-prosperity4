from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle
import numpy as np


# ASH: Rolling VWAP(window=20) fair value + take/clear/make | IPR: buy-and-hold

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
POS_LIMITS = {
    'ASH_COATED_OSMIUM': 80,
    "INTARIAN_PEPPER_ROOT": 80
}
PARAMS = {
    "ASH_COATED_OSMIUM": {
      "vwap_window": 10_000,
      "take_margin": 1,
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
    # need to fillin
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
        self.position_limit = POS_LIMITS.get(name, 0) # default to limit = 0 if prod not found in dict

        self.starting_position = self._get_current_position()
        self.expected_position = self.starting_position # to be updated (mutatable)

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
                # corrupt data -- reset cleanly
                last_traderData = default_traderData() # TODO
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

    # define generic lookups for margins, but we can override them by redefining them in inherited classes
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
        """
        Take in intended buy order (price and volume).
        Append a buy order that respects exchange's requirements, mainly:
            1. respect position limits
            2. is an Order object (price and volume must be ints)
        """
        abs_volume = min(int(abs(volume)), self.max_allowed_buy_volume)
        # update allowed buy volume
        self.max_allowed_buy_volume -= abs_volume
        order = Order(self.name, int(price), abs_volume)
        self.orders.append(order)

        # update expected position
        self.expected_position += abs_volume

    def sell(self, price, volume):
        """
        Take in intended sell order (price and volume).
        Append a sell order that respects exchange's requirements, mainly:
            1. respect position limits
            2. is an Order object (price and volume must be ints)
        """
        abs_volume = min(int(abs(volume)), self.max_allowed_sell_volume)
        # update allowed sell volume
        self.max_allowed_sell_volume -= abs_volume
        order = Order(self.name, int(price), -abs_volume)
        self.orders.append(order)

        # updated expected position
        self.expected_position -= abs_volume

    def update_traderData(self):
        self.new_traderData[self.name]["last_timestamp"] = self.timestamp


    def compute_mid_price(self):
        bb = self.get_best_bid()
        bo = self.get_best_ask()

        if bb and bo:
            return (bb + bo) / 2.0

    def compute_wall_mid(self):
        pass

    def compute_VWAP(self):
        pass

    def compute_make_ask_price(self):
        fair_ask_price = self.fair_value + self.make_margin
        if not self.quoted_sell_orders:
            return fair_ask_price
        best_ask_price = next(iter(self.quoted_sell_orders)) # take lowest sell price with active volume (>1)

        if best_ask_price > fair_ask_price:
            # if ba > fair_ask, we undercut it by 1 price tick
            return best_ask_price - 1
        else:
            return fair_ask_price

    def compute_make_bid_price(self):
        fair_bid_price = self.fair_value - self.make_margin
        if not self.quoted_buy_orders:
            return fair_bid_price
        best_bid_price = next(iter(self.quoted_buy_orders)) # take highest bid price with active volume (>1)

        if best_bid_price < fair_bid_price:
            # if bb < fair_bid, we overbid bb by 1 price tick
            return best_bid_price + 1
        else:
            return fair_bid_price


class VWAPTrader(ProductTrader):
    def __init__(self, state, new_traderData):
        super().__init__("ASH_COATED_OSMIUM", state, new_traderData)
        self.vwap_window = PARAMS[self.name]["vwap_window"]
        self.fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        """
        Rolling VWAP from market trades as dynamic fair value.
        Accumulates (price*volume, volume) per tick in a rolling window.
        Falls back to mid-price if no trade history yet.
        """
        # get market trades that occurred in previous tick
        trades = self.state.market_trades.get(self.name, [])
        tick_pv = sum(t.price * t.quantity for t in trades)
        tick_v = sum(t.quantity for t in trades)

        # retrieve rolling history from last traderData
        history = self.last_traderData[self.name].get("vwap_history", [])

        # only append if there were trades this tick
        if tick_v > 0:
            history.append([tick_pv, tick_v])

        # keep only last N entries
        history = history[-self.vwap_window:]

        # persist to new traderData
        self.new_traderData[self.name]["vwap_history"] = history

        # compute VWAP across window
        total_pv = sum(pv for pv, v in history)
        total_v = sum(v for pv, v in history)

        if total_v > 0:
            return round(total_pv / total_v)

        # fallback: use mid price, then 10000
        mid = self.compute_mid_price()
        return round(mid) if mid else 10000

    def get_orders(self):
        """
        Same take-clear-make pipeline as StaticTrader in trader6,
        but using VWAP-based fair value.
        """
        # STEP 1: take orders first
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.fair_value + self.take_margin:
                break # buy_orders are in decreasing order. If first bp is smaller, rest are all smaller. break for faster run
            # hit bid
            self.sell(bp, bv)
        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.fair_value - self.take_margin:
                break
            # lift offer
            self.buy(sp, sv)

        # STEP 2: clear orders to have enough bullets for next iteration if +EV traders emerge
        if self.expected_position > 0:
            # if we sell too much, then orders will get cancelled
            clear_volume = min(self.expected_position, self.max_allowed_sell_volume)
            clear_price = self.fair_value + self.clear_margin # default is zero
            self.sell(clear_price, clear_volume) # we are okay with going market neutral at the end of each iteration, if possible

        elif self.expected_position < 0:
            clear_volume = min(-self.expected_position, self.max_allowed_buy_volume)
            clear_price = self.fair_value - self.clear_margin
            self.buy(clear_price, clear_volume)

        else: # when expected_pos == 0
            pass

        # STEP 3: Make orders
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()

        # default: don't care about inventory as of now, change later
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
            # if day offset is not included in trader data, we reverse engineer by using mid price
            day_offset = round((self.compute_mid_price() - self.alpha) / (self.beta * 1_000_000))

            g_time_index = day_offset * 1_000_000 + self.timestamp
            fair = self.alpha + self.beta * g_time_index

            # update trader data
            self.new_traderData[self.name]["day_offset"] = day_offset

        else:
            # if day offset is in traderData, use it to compute fair
            # first check if we should use a new day_offset
            if self.timestamp < self.last_traderData[self.name]["last_timestamp"]:
                day_offset += 1 # increment day_offset by 1


            g_time_index = day_offset * 1_000_000 + self.timestamp
            fair = self.alpha + self.beta * g_time_index
            # update day offset
            self.new_traderData[self.name]["day_offset"] = day_offset
        return round(fair)


    def get_orders(self):
        """
        don't clear when position > 0.
        """

        # buy at aggressive prices
        agg_sp = list(self.quoted_sell_orders.keys())[-1] # highest offer
        self.buy(agg_sp, self.position_limit)

        # STEP 2: clear orders when expected pos is negative



        # STEP 3: Make orders
        # Since it is an upward trending product, we only make bids
#        bid_price = self.compute_make_bid_price()

 ##          self.buy(bid_price, self.max_allowed_buy_volume)


        return {self.name : self.orders}

class Trader:
    def run(self, state: TradingState):
        result : Dict[str, List[Order]] = {}
        # STEP 1:intiialize new traderdata
        new_traderData = default_traderData()

        # STEP 2: intialize trader classes
        vwap_trader = VWAPTrader(state, new_traderData)
        lin_trend_trader = LinearTrendTrader(state, new_traderData)

        # STEP 3: get orders
        result.update(vwap_trader.get_orders())
        result.update(lin_trend_trader.get_orders())

        # STEP 4: Update TraderData
        # save timestamp
        vwap_trader.update_traderData()
        lin_trend_trader.update_traderData()

        new_traderData.update(vwap_trader.new_traderData)
        new_traderData.update(lin_trend_trader.new_traderData)


        # STEP 5:ENCODE TraderData
        traderData = jsonpickle.encode(new_traderData)
        conversions = 0
        return result, conversions, traderData


