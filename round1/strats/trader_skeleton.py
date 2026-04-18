from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
from collections import deque
import jsonpickle
import numpy as np

LIMITS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "CROISSANTS": 250,
}

PARAMS = {
    "RAINFOREST_RESIN": {
        "fair_value": 10000, "take_width": 1,
        "clear_width": 0, "default_edge": 2,
    },
    "KELP": {
        "fair_value": None, "take_width": 1,
        "clear_width": 0, "default_edge": 1,
        "ema_alpha": 0.15, "adverse_volume": 15,
    },
}

# ── Default state factory ──
# Called on the very first timestep (or if traderData is empty/corrupt)
def default_state():
    return {
        # Per-product rolling data
        "kelp_ema": None,
        "kelp_prices": [],          # capped at N entries
        "squid_ink_prices": [],
        "squid_ink_mean": None,
        "squid_ink_std": None,
        # Cross-product signals
        "olivia_signals": {},        # {"CROISSANTS": "BUY", ...}
        "olivia_last_seen": {},      # {"CROISSANTS": 45000, ...}
        # Basket spread tracking
        "basket1_spread_history": [],
        "basket1_spread_ema": None,
    }


class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        # ════════════════════════════════════════════
        # STEP 1: DECODE traderData (or initialize)
        # ════════════════════════════════════════════
        if state.traderData and state.traderData != "":
            try:
                data = jsonpickle.decode(state.traderData)
            except Exception:
                # Corrupt data — reset cleanly rather than crash
                data = default_state()
        else:
            # First timestep — no prior state exists
            data = default_state()

        result = {}
        conversions = 0

        # ════════════════════════════════════════════
        # STEP 2: UPDATE STATE (before trading logic)
        #   Compute fair values, update EMAs, detect
        #   signals — anything that reads market data
        #   and writes to `data`.
        # ════════════════════════════════════════════

        # -- Update Kelp EMA --
        if "KELP" in state.order_depths:
            od = state.order_depths["KELP"]
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders.keys())
                best_ask = min(od.sell_orders.keys())
                mid = (best_bid + best_ask) / 2

                # Append to rolling price list (capped)
                data["kelp_prices"].append(mid)
                if len(data["kelp_prices"]) > 50:
                    data["kelp_prices"] = data["kelp_prices"][-50:]

                # Update EMA
                alpha = PARAMS["KELP"]["ema_alpha"]
                if data["kelp_ema"] is None:
                    data["kelp_ema"] = mid
                else:
                    data["kelp_ema"] = alpha * mid + (1 - alpha) * data["kelp_ema"]

        # -- Detect Olivia signals --
        for product in ["SQUID_INK", "KELP", "CROISSANTS"]:
            for trade in state.market_trades.get(product, []):
                if abs(trade.timestamp - state.timestamp) <= 100:
                    if trade.buyer == "Olivia":
                        data["olivia_signals"][product] = "BUY"
                        data["olivia_last_seen"][product] = state.timestamp
                    elif trade.seller == "Olivia":
                        data["olivia_signals"][product] = "SELL"
                        data["olivia_last_seen"][product] = state.timestamp
            # Also check own_trades (Olivia may have traded against us)
            for trade in state.own_trades.get(product, []):
                if abs(trade.timestamp - state.timestamp) <= 100:
                    if trade.buyer == "Olivia":
                        data["olivia_signals"][product] = "BUY"
                        data["olivia_last_seen"][product] = state.timestamp
                    elif trade.seller == "Olivia":
                        data["olivia_signals"][product] = "SELL"
                        data["olivia_last_seen"][product] = state.timestamp

        # -- Update Squid Ink rolling stats --
        if "SQUID_INK" in state.order_depths:
            od = state.order_depths["SQUID_INK"]
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
                data["squid_ink_prices"].append(mid)
                if len(data["squid_ink_prices"]) > 30:
                    data["squid_ink_prices"] = data["squid_ink_prices"][-30:]
                if len(data["squid_ink_prices"]) >= 10:
                    data["squid_ink_mean"] = np.mean(data["squid_ink_prices"])
                    data["squid_ink_std"] = np.std(data["squid_ink_prices"])

        # ════════════════════════════════════════════
        # STEP 3: TRADING LOGIC (reads from `data`)
        #   Take → Clear → Make pipeline per product.
        #   Uses fair values and signals computed above.
        # ════════════════════════════════════════════

        for product in state.order_depths:
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = LIMITS.get(product, 20)
            params = PARAMS.get(product, {})

            # Compute fair value using state
            if params.get("fair_value"):
                fair = params["fair_value"]
            elif product == "KELP" and data["kelp_ema"] is not None:
                fair = data["kelp_ema"]
            elif product == "SQUID_INK" and data["squid_ink_mean"] is not None:
                fair = data["squid_ink_mean"]
            else:
                if od.buy_orders and od.sell_orders:
                    fair = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
                else:
                    result[product] = []
                    continue

            # ── TAKE ──
            take_orders, bv, sv = self.take(
                product, od, fair,
                params.get("take_width", 1),
                position, limit,
                params.get("adverse_volume"),
            )

            # ── CLEAR ──
            clear_orders, bv, sv = self.clear(
                product, od, fair,
                params.get("clear_width", 0),
                position, limit, bv, sv,
            )

            # ── MAKE ──
            make_orders = self.make(
                product, fair,
                params.get("default_edge", 2),
                position, limit, bv, sv,
            )

            result[product] = take_orders + clear_orders + make_orders

        # ════════════════════════════════════════════
        # STEP 4: ENCODE traderData (last thing before return)
        # ════════════════════════════════════════════
        traderData = jsonpickle.encode(data)

        # Safety check: if serialized state exceeds the 50K limit,
        # trim the largest rolling lists and re-encode
        if len(traderData) > 48000:
            for key in ["kelp_prices", "squid_ink_prices", "basket1_spread_history"]:
                if key in data and isinstance(data[key], list):
                    data[key] = data[key][-20:]  # aggressive trim
            traderData = jsonpickle.encode(data)

        return result, conversions, traderData

    # ── Pipeline methods (same as before) ──

    def take(self, product, od, fair, width, position, limit, adverse_vol=None):
        orders, bv, sv = [], 0, 0
        for ask, amt in sorted(od.sell_orders.items()):
            if ask > fair - width:
                break
            if adverse_vol and -amt >= adverse_vol:
                continue
            qty = min(-amt, limit - position - bv)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                bv += qty
        for bid, amt in sorted(od.buy_orders.items(), reverse=True):
            if bid < fair + width:
                break
            if adverse_vol and amt >= adverse_vol:
                continue
            qty = min(amt, limit + position - sv)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                sv += qty
        return orders, bv, sv

    def clear(self, product, od, fair, width, position, limit, bv, sv):
        orders = []
        pos_after = position + bv - sv
        if pos_after > 0:
            price = round(fair) + width
            qty = min(pos_after, limit + position - sv)
            if qty > 0:
                orders.append(Order(product, int(price), -qty))
                sv += qty
        elif pos_after < 0:
            price = round(fair) - width
            qty = min(-pos_after, limit - position - bv)
            if qty > 0:
                orders.append(Order(product, int(price), qty))
                bv += qty
        return orders, bv, sv

    def make(self, product, fair, edge, position, limit, bv, sv):
        orders = []
        bid_qty = limit - position - bv
        if bid_qty > 0:
            orders.append(Order(product, round(fair) - edge, bid_qty))
        ask_qty = limit + position - sv
        if ask_qty > 0:
            orders.append(Order(product, round(fair) + edge, -ask_qty))
        return orders