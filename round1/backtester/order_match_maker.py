"""
OrderMatchMaker
===============
Simulates Prosperity exchange order matching.

Key rules:
  1. Aggressive orders (buy >= best_ask, sell <= best_bid) fill immediately.
  2. Passive orders do not fill this tick.
  3. Position limits: if net filled position would exceed ±limit, the entire
     product's orders are cancelled (Prosperity all-or-nothing behaviour).
  4. Fills happen at the book price, not the order price.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Fill:
    product: str
    price: int
    quantity: int   # + = buy, - = sell


class OrderMatchMaker:
    VERSION = "v2_simulation"

    def match(
        self,
        orders_by_product: Dict[str, list],
        order_depths: Dict[str, object],
        market_trade_rows: list,
        positions: Dict[str, int],
        limits: Dict[str, int],
        default_limit: int = 80,
    ) -> Tuple[Dict[str, List[Fill]], Dict[str, int]]:

        fills_by_product: Dict[str, List[Fill]] = {}

        for product, orders in orders_by_product.items():
            if not orders:
                continue

            limit = limits.get(product, default_limit)
            od    = order_depths.get(product)
            pos   = positions.get(product, 0)

            book_bids = dict(od.buy_orders)  if od else {}
            book_asks = dict(od.sell_orders) if od else {}

            best_ask = min(book_asks.keys()) if book_asks else None
            best_bid = max(book_bids.keys()) if book_bids else None

            # simulate fills without committing
            candidate_fills: List[Fill] = []
            sim_pos = pos
            sim_bids = dict(book_bids)
            sim_asks = dict(book_asks)

            for order in orders:
                qty_signed = order.quantity
                price      = order.price

                if qty_signed > 0:
                    if best_ask is None or price < best_ask:
                        continue
                    remaining = qty_signed
                    for ask in sorted(sim_asks.keys()):
                        if ask > price or remaining == 0:
                            break
                        available = -sim_asks[ask]
                        qty = max(0, min(remaining, available, limit - sim_pos))
                        if qty <= 0:
                            break
                        candidate_fills.append(Fill(product, ask, qty))
                        sim_pos    += qty
                        remaining  -= qty
                        sim_asks[ask] += qty
                        if sim_asks[ask] == 0:
                            del sim_asks[ask]

                elif qty_signed < 0:
                    if best_bid is None or price > best_bid:
                        continue
                    sell_qty = -qty_signed
                    for bid in sorted(sim_bids.keys(), reverse=True):
                        if bid < price or sell_qty == 0:
                            break
                        available = sim_bids[bid]
                        qty = max(0, min(sell_qty, available, sim_pos + limit))
                        if qty <= 0:
                            break
                        candidate_fills.append(Fill(product, bid, -qty))
                        sim_pos   -= qty
                        sell_qty  -= qty
                        sim_bids[bid] -= qty
                        if sim_bids[bid] == 0:
                            del sim_bids[bid]

            # enforce limit — if sim_pos breaches, cancel all
            if abs(sim_pos) > limit:
                fills_by_product[product] = []
                continue

            positions[product] = sim_pos
            fills_by_product[product] = candidate_fills

        return fills_by_product, positions
