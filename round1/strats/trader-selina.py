from datamodel import TradingState, Order
from typing import List
import json

PRODUCT_CONFIG = {
    "ASH_COATED_OSMIUM": {
        "strategy" : "rolling_mm",
        "window"   : 20,
        "alpha"    : 0.917,
        "beta"     : 3,
    },
    "INTARIAN_PEPPER_ROOT": {
        "strategy" : "fv_mm",
        "window"   : 10,
        "mm_size"  : 10,
    },
}

LIMITS = {
    "ASH_COATED_OSMIUM"    : 80,
    "INTARIAN_PEPPER_ROOT" : 20,
}


class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result      = {}
        conversions = 0

        for product, od in state.order_depths.items():

            cfg   = PRODUCT_CONFIG.get(product)
            limit = LIMITS.get(product)

            if cfg is None or limit is None or cfg["strategy"] == "ignore":
                result[product] = []
                continue

            bids = od.buy_orders
            asks = od.sell_orders

            pos      = state.position.get(product, 0)
            buy_cap  = limit - pos
            sell_cap = limit + pos

            if cfg["strategy"] == "fixed_fv":
                if not bids or not asks:
                    result[product] = []
                    continue
                best_bid = max(bids)
                best_ask = min(asks)
                orders = self._fixed_fv(
                    product, pos, bids, asks,
                    best_bid, best_ask, buy_cap, sell_cap,
                    cfg["fair_value"], cfg["edge"]
                )

            elif cfg["strategy"] == "rolling_mm":
                if not bids or not asks:
                    result[product] = []
                    continue
                best_bid = max(bids)
                best_ask = min(asks)
                history_key = f"h_{product}"
                history     = td.get(history_key, [])
                orders, history = self._rolling_mm(
                    product, pos, bids, asks,
                    best_bid, best_ask, buy_cap, sell_cap,
                    history,
                    cfg["window"], cfg["alpha"], cfg["beta"]
                )
                td[history_key] = history

            elif cfg["strategy"] == "fv_mm":
                history_key = f"h_{product}"
                history     = td.get(history_key, [])
                orders, history = self._fv_mm(
                    product, pos, bids, asks,
                    buy_cap, sell_cap,
                    history, cfg["window"], cfg["mm_size"]
                )
                td[history_key] = history

            else:
                orders = []

            result[product] = orders

        return result, conversions, json.dumps(td)

    def _fixed_fv(self, product, pos, bids, asks, best_bid, best_ask,
                  buy_cap, sell_cap, fair, edge) -> List[Order]:
        orders = []

        aggressive_buy = min(
            sum(-v for p, v in asks.items() if p <= fair),
            buy_cap
        )
        aggressive_sell = min(
            sum(v for p, v in bids.items() if p >= fair),
            sell_cap
        )

        if aggressive_buy > 0:
            orders.append(Order(product, fair, aggressive_buy))
        if aggressive_sell > 0:
            orders.append(Order(product, fair, -aggressive_sell))

        passive_buy  = buy_cap  - aggressive_buy
        passive_sell = sell_cap - aggressive_sell

        if passive_buy > 0:
            orders.append(Order(product, fair - edge, passive_buy))
        if passive_sell > 0:
            orders.append(Order(product, fair + edge, -passive_sell))

        return orders

    def _fv_mm(self, product, pos, bids, asks, buy_cap, sell_cap,
               history, window, mm_size):
        orders = []

        mid = None
        if bids and asks:
            mid = (max(bids) + min(asks)) / 2.0
        elif bids:
            mid = float(max(bids))
        elif asks:
            mid = float(min(asks))

        if mid is None:
            return orders, history

        history.append(mid)
        if len(history) > window:
            history = history[-window:]

        fair = int(round(sum(history) / len(history)))

        # market-take: sweep mispriced orders
        for ask in sorted(asks):
            if ask >= fair or buy_cap <= 0:
                break
            qty = min(-asks[ask], buy_cap)
            orders.append(Order(product, ask, qty))
            buy_cap  -= qty
            pos      += qty

        for bid in sorted(bids, reverse=True):
            if bid <= fair or sell_cap <= 0:
                break
            qty = min(bids[bid], sell_cap)
            orders.append(Order(product, bid, -qty))
            sell_cap -= qty
            pos      -= qty

        # market-make: passive quotes at fair±1, capped at mm_size
        if buy_cap > 0:
            orders.append(Order(product, fair - 1, min(buy_cap, mm_size)))
        if sell_cap > 0:
            orders.append(Order(product, fair + 1, -min(sell_cap, mm_size)))

        return orders, history

    def _rolling_mm(self, product, pos, bids, asks, best_bid, best_ask,
                    buy_cap, sell_cap, history, window, alpha, beta):
        limit = LIMITS[product]

        mid = (best_bid + best_ask) / 2.0
        history.append(mid)
        if len(history) > window:
            history = history[-window:]

        orders   = []
        base_bid = best_bid + 1
        base_ask = best_ask - 1

        if len(history) >= max(10, window // 2):
            rolling_mean   = sum(history) / len(history)
            dev            = mid - rolling_mean
            signal_skew    = -dev * alpha
            inventory_skew = -(pos / limit) * beta
            net_skew       = signal_skew + inventory_skew
            bid_px = int(round(base_bid + net_skew))
            ask_px = int(round(base_ask + net_skew))
        else:
            bid_px = base_bid
            ask_px = base_ask

        fair_est = int(sum(history) / len(history)) if history else int(mid)
        bid_px = min(bid_px, fair_est - 1)
        ask_px = max(ask_px, fair_est + 1)
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        if buy_cap > 0:
            orders.append(Order(product, bid_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -sell_cap))

        return orders, history
