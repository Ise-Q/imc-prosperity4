# Submission UUID: (save after uploading to Prosperity)
#
# Strategy summary:
#   ASH_COATED_OSMIUM    — static fair value 10000 (CV=0.054%, extremely stable)
#   INTARIAN_PEPPER_ROOT — adaptive fair value via 10-tick rolling mid price
#
#   Both products:
#     1. Market-take: hit bot sell orders below FV (buy cheap)
#                     hit bot buy  orders above FV (sell dear)
#     2. Market-make: post buy at FV-1, sell at FV+1 (inside the ~13-16 tick bot spread)
#
# !! VERIFY position limits on Prosperity platform before submitting !!

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

# Per-product absolute position limits.  Verify these on the Prosperity platform.
POSITION_LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 20,
    "INTARIAN_PEPPER_ROOT": 20,
}

# ASH has an essentially fixed fair value (price never left 9977-10023 across 3 days).
ASH_FAIR_VALUE = 10000

# Rolling window for IPR fair value (short window tracks the within-day random walk).
IPR_WINDOW = 10

# Max units to post on each side of the market when market-making.
MM_SIZE = 10


class Trader:

    def bid(self):
        """Required stub for Round 2; harmless in all other rounds."""
        return 15

    def run(self, state: TradingState):
        saved = (
            jsonpickle.decode(state.traderData)
            if state.traderData
            else {"price_history": {}}
        )

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            pos_limit = POSITION_LIMITS.get(product, 20)
            mid = _compute_mid(od)

            # ── Fair value ──────────────────────────────────────────────────
            if product == "ASH_COATED_OSMIUM":
                fair_value = ASH_FAIR_VALUE
            else:
                hist = saved["price_history"].get(product, [])
                if mid is not None:
                    hist.append(mid)
                    hist = hist[-IPR_WINDOW:]
                saved["price_history"][product] = hist
                fair_value = sum(hist) / len(hist) if hist else mid

            if fair_value is None:
                result[product] = []
                continue

            fair_int = int(round(fair_value))
            orders: List[Order] = []

            # Remaining position headroom (updated as we size orders below)
            buy_cap = pos_limit - position   # how many units we can still buy
            sell_cap = pos_limit + position  # how many units we can still sell

            # ── Market-take: buy cheap, sell dear ───────────────────────────
            # Buy bot sell orders priced strictly below our fair value
            for ask in sorted(od.sell_orders):
                if ask >= fair_int or buy_cap <= 0:
                    break
                qty = min(-od.sell_orders[ask], buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty
                position += qty

            # Sell bot buy orders priced strictly above our fair value
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= fair_int or sell_cap <= 0:
                    break
                qty = min(od.buy_orders[bid], sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty
                position -= qty

            # ── Market-make: post inside the bot spread ─────────────────────
            if buy_cap > 0:
                orders.append(Order(product, fair_int - 1, min(buy_cap, MM_SIZE)))
            if sell_cap > 0:
                orders.append(Order(product, fair_int + 1, -min(sell_cap, MM_SIZE)))

            result[product] = orders

        return result, 0, jsonpickle.encode(saved)


def _compute_mid(od: OrderDepth):
    """Best-bid/ask midpoint; falls back to whichever side is available."""
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    if od.buy_orders:
        return float(max(od.buy_orders))
    if od.sell_orders:
        return float(min(od.sell_orders))
    return None