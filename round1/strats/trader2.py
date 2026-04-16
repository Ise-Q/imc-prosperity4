from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
from collections import deque
import jsonpickle
import numpy as np
import pandas as pd

# params that can be optimized
# take_margin, clear_margin

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
POS_LIMITS = {
    'ASH_COATED_OSMIUM': 80,
    "INTARIAN_PEPPER_ROOT": 80
}
PARAMS = {
    "ASH_COATED_OSMIUM": {
      "static_fair_value": 10_000, "take_margin": 1,
      "clear_margin": 0, "make_margin": 1
    },
    "INTARIAN_PEPPER_ROOT":{
        "slope": 0.001, "intercept": 12000.0,
        "take_margin": 1, "clear_margin": 0,
        "make_margin": 1
    }
}

# define default traderData
def default_traderData():
    # need to fillin
    return {}

# base ProductTrader 
class ProductTrader:
    def __init__(self, name, state, new_traderData):
        self.orders = []
        self.name = name
        self.state = state
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
        pass

    def get_best_ask(self):
        pass

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
        

    def compute_mid_price(self):
        pass

    def compute_wall_mid(self):
        pass

    def compute_VWAP(self):
        pass

    def compute_make_bid_price(self):
        # OVERRIDE WITHIN EACH INHERITED CLASS
        pass

    def compute_make_ask_price(self):
        # OVERRIDE WITHIN EACH INHERITED CLASS
        pass

    def compute_fair_value():
        # OVERRIDE WITHIN EACH INHERITED CLASS
        pass
    
class StaticTrader(ProductTrader):
    def __init__(self, state, new_traderData):
        super().__init__("ASH_COATED_OSMIUM", state, new_traderData)
        
        self.static_fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        if "static_fair_value" not in PARAMS[self.name].keys():
            raise KeyError(f"static_fair_value is not a key of PARAMS{self.name}!")
        else:
            return PARAMS[self.name]["static_fair_value"]
    
    def compute_make_ask_price(self):
        if not self.quoted_sell_orders:
            return None
        self.quoted_sell_orders.keys()[0] # take lowest sell price with active volume (>1)
        self.q
        ask_price = self.static_fair_value + self.make_margin
    
    def compute_make_bid_price(self):
        if not self.quoted_buy_orders:
            return None
        self.quoted_buy_orders.keys()[0]

    def get_orders(self):
        """

        """
        # STEP 1: take orders first
        for bp, bv in self.quoted_buy_orders.items():
            if bp < self.static_fair_value + self.take_margin:
                break # buy_orders are in decreasing order. If first bp is smaller, rest are all smaller. break for faster run
            # hit bid
            self.sell(bp, -bv)
        for sp, sv in self.quoted_sell_orders.items():
            if sp > self.static_fair_value - self.take_margin:
                break
            # lift offer
            self.buy(sp, sv)

        # STEP 2: clear orders to have enough bullets for next iteration if +EV traders emerge
        if self.expected_position > 0:
            # if we sell too much, then orders will get cancelled
            clear_volume = min(self.expected_position, self.max_allowed_sell_volume)
            clear_price = self.static_fair_value + self.clear_margin # default is zero
            self.sell(clear_price, -clear_volume) # we are okay with going market neutral at the end of each iteration, if possible
        
        elif self.expected_position < 0:
            clear_volume = min(-self.expected_position, self.max_allowed_buy_volume)
            clear_price = self.static_fair_value - self.clear_margin
            self.buy(clear_price, clear_volume)
        
        else: # when expected_pos == 0
            pass

        # STEP 3: Make orders
        
        

        return {self.name : self.orders}    
    
        

class LinearTrendTrader(ProductTrader):
    def __init__(self, state, new_traderData):
        super().__init__("INTARIAN_PEPPER_ROOT", state, new_traderData)
    
    def compute_fair_value(self):
        pass

    def get_orders(self):
        pass

class Trader:
    def run(self, state: TradingState):
        # STEP 1:DECODE TraderData

        # STEP 2:UPDATE STATE

        # STEP 3:TRADING LOGIC

        # take

        # clear

        # make
        
        # STEP 4:ENCODE TraderData
        pass

        