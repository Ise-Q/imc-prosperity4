from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
from functools import cached_property
import jsonpickle
import numpy as np
import statistics
import math


# ASH: EMA(alpha = 0.05) fair value + take/clear/make | IPR: buy-and-hold

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
POS_LIMITS = {
    'ASH_COATED_OSMIUM': 80,
    "INTARIAN_PEPPER_ROOT": 80
}
PARAMS = {
    "ASH_COATED_OSMIUM": {
      "ema_alpha": 0.05,
      "static_fv" : 10_000,
      "fv_method_weights" : [1, 0], # [static, ema]
      "take_margin": 1,
      "clear_margin": 6, "make_margin": 4,
      "vwap_window": 500
    },
    "INTARIAN_PEPPER_ROOT":{
        "slope": 0.001, "intercept": 12000.0,
        "ema_alpha": 0.05,
        "take_margin": 1, "clear_margin": 2,
        "make_margin": 2,
        "vwap_window": 500
    }
}

# define default traderData
def default_traderData():
    # need to fillin
    return {product: {} for product in PRODUCTS}

# base ProductTrader
class ProductTrader:
    def __init__(self, name, state, last_traderData, new_traderData):
        self.orders = []
        self.name = name
        self.state = state
        self.timestamp = self.state.timestamp
        self.new_traderData = new_traderData.setdefault(self.name, {}) # product specific trader data. Note that we use .setdefault() instead of .get(). 
        # While in this case, the returned dictionary references to the same inner dictionary in new_traderData (because default_traderData() guarantees self.name is in the keys),  
        # originally, it is a fragile method. So we instead use .setdefault(self.name, {}), which inserts the key (if not already in the original dict) and returns the value

        self.params = PARAMS.get(self.name, {})

        self.last_traderData = last_traderData.get(self.name, {})
        self.position_limit = POS_LIMITS.get(name, 0) # default to limit = 0 if prod not found in dict

        self.starting_position = self._get_current_position()
        self.expected_position = self.starting_position # to be updated (mutatable)

        self.market_trades = self._get_market_trades()
        self.own_trades = self._get_own_trades()

        self.quoted_buy_orders, self.quoted_sell_orders = self._get_order_depth()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self._get_max_allowed_volume()

        self.take_margin = self._get_take_margin()
        self.clear_margin = self._get_clear_margin()
        self.make_margin = self._get_make_margin()

        # Parms
        self.ema_alpha = self.params.get("ema_alpha", 0.05)

    @cached_property
    def vwap(self):
        return self.compute_vwap()

    def _get_last_traderData(self):
        if self.state.traderData and self.state.traderData != "":
            try:
                last_traderData = jsonpickle.decode(self.state.traderData)
                product_last_traderData = last_traderData.setdefault(self.name, {})
            except Exception:
                # corrupt data -- reset cleanly
                last_traderData = default_traderData()
                product_last_traderData = last_traderData.setdefault(self.name, {})
            return product_last_traderData
        else:
            last_traderData = default_traderData()
            product_last_traderData = last_traderData.setdefault(self.name, {})
            return product_last_traderData

    def _get_current_position(self):
        return self.state.position.get(self.name, 0)

    def _get_market_trades(self):
        return self.state.market_trades.get(self.name, [])

    def _get_own_trades(self):
        return self.state.own_trades.get(self.name, [])

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
        return self.params.get("take_margin", 1)

    def _get_clear_margin(self):
        return self.params.get("clear_margin",0)

    def _get_make_margin(self):
        return self.params.get("make_margin",1)
    
    def _get_wall_level(self, side):
        """Returns (price, volume) of the level with the largest resting size
        on the given side, or (None, None) if empty.
        side ∈ {'buy', 'sell'}."""
        book = self.quoted_buy_orders if side == 'buy' else self.quoted_sell_orders
        if not book:
            return (None, None)
        price, volume = max(book.items(), key=lambda x: x[1])
        return (price, volume)

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

        A useful corollary of this structure: One need not care about rounding
        price and volume calculations in other features since as long as orders
        flow through this pipeline, prices will be automatically adjusted
        """
        abs_volume = min(round(abs(volume)), self.max_allowed_buy_volume)
        # update allowed buy volume
        self.max_allowed_buy_volume -= abs_volume
        order = Order(self.name, round(price), abs_volume)
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
        abs_volume = min(round(abs(volume)), self.max_allowed_sell_volume)
        # update allowed sell volume
        self.max_allowed_sell_volume -= abs_volume
        order = Order(self.name, round(price), -abs_volume)
        self.orders.append(order)

        # updated expected position
        self.expected_position -= abs_volume

    def update_traderData(self):
        self.new_traderData["last_timestamp"] = self.timestamp


    def compute_mid_price(self):
        bb = self.get_best_bid()
        bo = self.get_best_ask()

        if (bb is not None) and (bo is not None):
            return (bb + bo) / 2.0

    def compute_wall_mid(self, weighted=False):
        """Midpoint anchored at the deepest (highest-volume) resting level on
        each side. If weighted=True, weights each wall price by the that side's
        wall volume. Returns None if either side empty. Later, ffill() wall price 
        at iterations when either side is empty."""
        bid_p, bid_v = self._get_wall_level('buy')
        ask_p, ask_v = self._get_wall_level('sell')
        if bid_p is None or ask_p is None:
            return None
        if not weighted:
            return (bid_p + ask_p) / 2.0
        if bid_v + ask_v == 0:
            return (bid_p + ask_p) / 2.0
        return (ask_p * ask_v + bid_p * bid_v) / (bid_v + ask_v)
        
    def compute_liquidity_price(self):
        """
        This assumes mms post quotes that reflect most the fair value of the security.
        Consequentially, higher volume at a lower bid price reflects that fair value is
        dragged downwards, and vice versa for high volume at a high ask price.

        This is a competitng framework of the microprice framework. The foundations of microprice
        are built on the premise that quotes reprsent mms' will to *absorb* orderflow.
        For instance, higher volume at a certain bid price reflects mm's will to absorb any sell orders
        coming through that level; therefore, fair value should be higher.
        """
        agg_ask_p = 0
        agg_ask_v = 0
        agg_bid_p = 0
        agg_bid_v = 0

        if self.quoted_sell_orders and self.quoted_buy_orders:
            for sp, sv in self.quoted_sell_orders.items():
                agg_ask_p += sp * sv
                agg_ask_v += sv
            for bp, bv in self.quoted_buy_orders.items():
                agg_bid_p += bp * bv
                agg_bid_v += bv
            
            if not (agg_bid_v or agg_ask_v):
                return None  # if both volumes are zero return None
            
            liq_price = (agg_bid_p + agg_ask_p) / (agg_bid_v + agg_ask_v)
                  
            return liq_price
    
    def compute_imbalance_price(self):
        bb = self.get_best_bid()
        ba = self.get_best_ask()

        if (bb is None) or (ba is None):
            return self.compute_mid_price()

 
        total_buy_volume = sum(self.quoted_buy_orders.values())
        total_sell_volume = sum(self.quoted_sell_orders.values())
        
        total_volume = total_buy_volume + total_sell_volume
        
        if not total_volume: 
            return self.compute_mid_price()
        
        mid = (ba + bb) / 2
        spread = ba - bb
        imbalance = (total_buy_volume - total_sell_volume) / total_volume
        imbalance_price = mid + imbalance * (spread / 2)

        return imbalance_price
        
            
        
    def compute_microprice(self):
        """
        microprice from stoikov(2017).
        """

        if self.quoted_sell_orders and self.quoted_buy_orders:
            best_ask_p = next(iter(self.quoted_sell_orders))
            best_ask_v = self.quoted_sell_orders.get(best_ask_p, 0)
                        
            best_bid_p = next(iter(self.quoted_buy_orders))
            best_bid_v = self.quoted_buy_orders.get(best_bid_p, 0)

            if not (best_ask_v or best_bid_v): # both volumes are zero
                return None
        
            microprice = (best_ask_p * best_bid_v + best_bid_p * best_ask_v) / (best_ask_v + best_bid_v)
            return microprice
    
    def compute_vwap(self):
        """
        state-dependent. Make sure you only run the method
        ONCE per product per time iteration, so that you don't store duplicate values. 

        Use self.vwap when vwap is needed as an input to other methods.
        """
        vwap_window = self.params["vwap_window"]
        vwap_history = list(self.last_traderData.get("vwap_history", [])) # return a new object instead of the pointer to last_traderData["vwap_history"]
        if self.market_trades:
            sum_product = sum([trade.price * trade.quantity for trade in self.market_trades])
            tick_volume = sum([trade.quantity for trade in self.market_trades])
            if tick_volume > 0:
                tick_vwap = sum_product / tick_volume
                vwap_history.append((tick_vwap, tick_volume))
        
        vwap_history = vwap_history[-vwap_window:]
      
        total_sumproduct = sum([p*v for p, v in vwap_history])
        total_volume = sum([v for p, v in vwap_history])
        vwap = total_sumproduct / total_volume if total_volume > 0 else None

        # update new traderData
        self.new_traderData["vwap_history"] = vwap_history

        return vwap
    
    def compute_ema(self):
        """
        state dependent
        """

        mid = self.compute_mid_price()
        prev_ema = self.last_traderData.get("ema", None)

        if mid is None and prev_ema is None:
            return #  if prev_ema, mid are None, then return None
            # no order book - carry forward last EMA or default
        
        elif mid is None and prev_ema is not None:
            ema = prev_ema
    
        elif mid is not None and prev_ema is None:
            # first tick - initialize EMA with mid
            ema = mid
        else: # both mid and prev_ema are not None
            ema = self.ema_alpha * mid + (1 - self.ema_alpha) * prev_ema
    

        # update new_traderData with ema computed in current iteration 
        self.new_traderData["ema"] = ema
        return ema  


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


class MeanReversionTrader(ProductTrader):
    def __init__(self, state, last_traderData, new_traderData):
        super().__init__("ASH_COATED_OSMIUM", state, last_traderData, new_traderData)
        self.static_fv = self._get_static_fv()
        self.fv_method_weights = self._get_fv_method_weights()

        
        self.fair_value = self.compute_fair_value()
        

    def compute_fair_value(self):
        """
        fair value is weighted sum of ema and static fair value. 
        Persists EMA state across ticks via traderData.
        Falls back to 10000 if no order book available.
        """
       
        # weight between static fair value and ema
        ema = self.compute_ema() # new ema is also updated to new_traderData
        if ema is None:
            return self.static_fv
        fair = self.fv_method_weights[0] * self.static_fv + self.fv_method_weights[1] * ema

        return round(fair)
    
    def _get_fv_method_weights(self):
        weights = self.params.get("fv_method_weights", [])
        # make sure weights are normalized to sum 1
        if weights:
            weights = [w / sum(weights) for w in weights]

        return weights


    def _get_static_fv(self):
        return self.params.get("static_fv", 10_000) # default to 10_000
        

    def get_orders(self):
        """
        Take and Make.
        """
        # STEP 1: take mispriced orders first
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

        # STEP 2: Make orders
        ask_price = self.compute_make_ask_price()
        bid_price = self.compute_make_bid_price()   

        # default: don't care about inventory as of now, change later
        if ask_price is not None and self.max_allowed_sell_volume > 0:
            self.sell(ask_price, self.max_allowed_sell_volume)
        if bid_price is not None and self.max_allowed_buy_volume > 0:
            self.buy(bid_price, self.max_allowed_buy_volume)


        return {self.name : self.orders}



class LinearTrendTrader(ProductTrader):
    def __init__(self, state, last_traderData, new_traderData):
        super().__init__("INTARIAN_PEPPER_ROOT", state, last_traderData, new_traderData)
        self.alpha = self.params["intercept"]
        self.beta = self.params["slope"]
        self.fair_value = self.compute_fair_value()

    def compute_fair_value(self):
        """
        fair value is a linear relationship with respect global time index
        fair = alpha + beta * g_time_index
        where
            g_time_index = day_offset * 1e6 + timestamp

        we reverse engineer day_offset at first iteration and save it in traderData
        """
        day_offset = self.last_traderData.get("day_offset", None)
        if (day_offset is None):
            mid = self.compute_mid_price()
            if mid is not None:
            
                # if day offset is not included in trader data, we reverse engineer by using mid price
                day_offset = round((self.compute_mid_price() - self.alpha) / (self.beta * 1_000_000))

                g_time_index = day_offset * 1_000_000 + self.timestamp
                fair = self.alpha + self.beta * g_time_index
            else:
                day_offset = 0 # delay reverse engineering to next step
                g_time_index = day_offset * 1_000_000 + self.timestamp
                fair = self.alpha + self.beta * g_time_index

            # update trader data
            self.new_traderData["day_offset"] = day_offset

        else:
            # if day offset is in traderData, use it to compute fair
            # first check if we should use a new day_offset
            if self.timestamp < self.last_traderData.get("last_timestamp", self.timestamp):
                day_offset += 1 # increment day_offset by 1


            g_time_index = day_offset * 1_000_000 + self.timestamp
            fair = self.alpha + self.beta * g_time_index
            # update day offset
            self.new_traderData["day_offset"] = day_offset
        return round(fair)


    def get_orders(self):
        """
        buy-hold
        """

        # buy at aggressive prices
        if self.quoted_sell_orders:
            agg_sp = list(self.quoted_sell_orders.keys())[-1] # highest offer
            self.buy(agg_sp, self.position_limit)


        return {self.name : self.orders}


class Trader:
    def run(self, state: TradingState):
        result : Dict[str, List[Order]] = {}
        # STEP 1:intiialize new traderdata and get last_traderData
        new_traderData = default_traderData()
        last_traderData = jsonpickle.decode(state.traderData) if state.traderData else default_traderData()

        # STEP 2: intialize trader classes
        mr_trader = MeanReversionTrader(state, last_traderData, new_traderData)
        lin_trend_trader = LinearTrendTrader(state, last_traderData, new_traderData)

        # STEP 3: get orders
        result.update(mr_trader.get_orders())
        result.update(lin_trend_trader.get_orders())

        # STEP 4: Update TraderData
        # save timestamp
        mr_trader.update_traderData()
        lin_trend_trader.update_traderData()

        # STEP 5:ENCODE TraderData
        traderData = jsonpickle.encode(new_traderData)
        conversions = 0
        return result, conversions, traderData


