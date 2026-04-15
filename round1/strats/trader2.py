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
      "fair_value": 10_000, "take_margin": 1,
      "clear_margin": None
    },
    "INTARIAN_PEPPER_ROOT":{
        "slope": 0.001, "intercept": 12000.0,
        "take_margin": 1, "clear_margin": None
        
    }
}

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
    def take(self, product, od, fair, width, position, limit):
        """
        For OSMIUM, we assume static fair value of 10000
        For ROOT, we do do a linear regression against TIME, since it is quite clear
        there is a deterministic trend. 

        We make markets around the fair value with width = take_margin
        """
        orders, bv, sv = [], 0, 0
        for ask, amt in sorted(od.sell_orders.items()): 
            if ask > fair - width:
                break # no trade
            qty = min(-amt, limit - position - bv) # amt is negative
            if qty > 0:
                orders.append(Order(product, ask, qty))
                bv += qty
        for bid, amt in sorted(od.buy_ordres.items(), reverse = True):
            if bid < fair + width: 
                break
            qty = min(amt, limit + position - sv)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                sv+= qty
        return orders, bv, sv
    
    def clear(self):
        """
        clear orders to reserve more spot
        """