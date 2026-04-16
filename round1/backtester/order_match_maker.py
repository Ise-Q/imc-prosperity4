"""
OrderMatchMaker
===============
Simulates exchange order matching:
  1. Hit order book (order depths) first
  2. Fall back to market trades if book volume insufficient
Position limits are enforced — orders that would exceed the limit are cancelled.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Fill:
    product: str
    price: int
    quantity: int   # + = buy, - = sell


class OrderMatchMaker:

    def match(
        self,
        orders_by_product: Dict[str, list],   # product -> [Order]
        order_depths: Dict[str, object],       # product -> OrderDepth
        market_trade_rows: list,               # [TradeRow]
        positions: Dict[str, int],
        limits: Dict[str, int],
        default_limit: int = 20,
    ) -> Tuple[Dict[str, List[Fill]], Dict[str, int]]:
        """
        Returns (fills_by_product, updated_positions).
        Positions dict is mutated in place and also returned.
        """
        fills_by_product: Dict[str, List[Fill]] = {}

        # Pre-index market trades by symbol for fast lookup
        mkt_by_symbol: Dict[str, list] = {}
        for tr in market_trade_rows:
            mkt_by_symbol.setdefault(tr.symbol, []).append(tr)

        for product, orders in orders_by_product.items():
            if not orders:
                continue

            limit = limits.get(product, default_limit)
            od    = order_depths.get(product)
            mkt   = mkt_by_symbol.get(product, [])

            # Check if ANY order would breach the limit — if so, cancel ALL for this product
            pos = positions.get(product, 0)
            projected = pos
            for o in orders:
                projected += o.quantity
            if projected > limit or projected < -limit:
                # cancel all — Prosperity behaviour
                continue

            product_fills: List[Fill] = []
            book_bids = dict(od.buy_orders)  if od else {}
            book_asks = dict(od.sell_orders) if od else {}  # values are negative

            for order in orders:
                remaining = order.quantity  # + buy, - sell
                price     = order.price

                if remaining > 0:
                    # BUY — match against ask side
                    for ask in sorted(book_asks.keys()):
                        if ask > price or remaining == 0:
                            break
                        available = -book_asks[ask]
                        qty = min(remaining, available, limit - pos)
                        if qty <= 0:
                            break
                        product_fills.append(Fill(product, ask, qty))
                        pos       += qty
                        remaining -= qty
                        book_asks[ask] += qty
                        if book_asks[ask] == 0:
                            del book_asks[ask]

                    # Fallback: market trades
                    for tr in mkt:
                        if remaining == 0:
                            break
                        if tr.price > price:
                            continue
                        qty = min(remaining, tr.quantity, limit - pos)
                        if qty <= 0:
                            continue
                        product_fills.append(Fill(product, price, qty))
                        pos       += qty
                        remaining -= qty

                elif remaining < 0:
                    # SELL — match against bid side
                    sell_qty = -remaining
                    for bid in sorted(book_bids.keys(), reverse=True):
                        if bid < price or sell_qty == 0:
                            break
                        available = book_bids[bid]
                        qty = min(sell_qty, available, pos + limit)
                        if qty <= 0:
                            break
                        product_fills.append(Fill(product, bid, -qty))
                        pos      -= qty
                        sell_qty -= qty
                        book_bids[bid] -= qty
                        if book_bids[bid] == 0:
                            del book_bids[bid]

                    # Fallback: market trades
                    for tr in mkt:
                        if sell_qty == 0:
                            break
                        if tr.price < price:
                            continue
                        qty = min(sell_qty, tr.quantity, pos + limit)
                        if qty <= 0:
                            continue
                        product_fills.append(Fill(product, price, -qty))
                        pos      -= qty
                        sell_qty -= qty

            positions[product] = pos
            fills_by_product[product] = product_fills

        return fills_by_product, positions
