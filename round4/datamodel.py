"""
Mock IMC Prosperity datamodel — drop-in replacement for the real datamodel.py.
Provides the same classes and signatures used in Trader.py.
"""
from typing import Dict, List, Optional, Any

Symbol = str
UserId = str
Product = str
Position = int
Cash = float


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol   = symbol
        self.price    = int(price)
        self.quantity = int(quantity)

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({self.symbol} {side} {abs(self.quantity)}@{self.price})"


class OrderDepth:
    def __init__(self) -> None:
        # buy_orders:  price -> quantity  (positive quantities)
        # sell_orders: price -> quantity  (negative quantities, IMC convention)
        self.buy_orders:  Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int,
                 buyer: Optional[UserId] = None,
                 seller: Optional[UserId] = None,
                 timestamp: int = 0) -> None:
        self.symbol    = symbol
        self.price     = price
        self.quantity  = quantity
        self.buyer     = buyer
        self.seller    = seller
        self.timestamp = timestamp

    def __repr__(self):
        return (f"Trade({self.symbol} {self.quantity}@{self.price} "
                f"buyer={self.buyer} seller={self.seller})")


class TradingState:
    def __init__(self,
                 traderData:    str,
                 timestamp:     int,
                 listings:      Dict,
                 order_depths:  Dict[Symbol, OrderDepth],
                 own_trades:    Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position:      Dict[Symbol, Position],
                 observations:  Any) -> None:
        self.traderData    = traderData
        self.timestamp     = timestamp
        self.listings      = listings
        self.order_depths  = order_depths
        self.own_trades    = own_trades
        self.market_trades = market_trades
        self.position      = position
        self.observations  = observations

    def __repr__(self):
        return f"TradingState(t={self.timestamp}, pos={self.position})"


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: str):
        self.symbol      = symbol
        self.product     = product
        self.denomination = denomination


class Observation:
    def __init__(self, plainValueObservations=None, conversionObservations=None):
        self.plainValueObservations   = plainValueObservations or {}
        self.conversionObservations   = conversionObservations or {}