"""
Input data models — hold raw market data parsed from CSV files.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PriceRow:
    """One row from prices_round_X_day_Y.csv (one product, one timestamp)."""
    day: int
    timestamp: int
    product: str
    bid_prices: List[Optional[float]]   # up to 3 levels
    bid_volumes: List[Optional[int]]
    ask_prices: List[Optional[float]]
    ask_volumes: List[Optional[int]]
    mid_price: float
    profit_and_loss: float


@dataclass
class TradeRow:
    """One row from trades_round_X_day_Y.csv."""
    timestamp: int
    buyer: str
    seller: str
    symbol: str
    currency: str
    price: float
    quantity: int


@dataclass
class BacktestData:
    """All market data for a single day, indexed by timestamp."""
    day: int
    # timestamp -> {product -> PriceRow}
    price_by_ts: Dict[int, Dict[str, PriceRow]] = field(default_factory=dict)
    # timestamp -> [TradeRow]
    trades_by_ts: Dict[int, List[TradeRow]] = field(default_factory=dict)

    @property
    def timestamps(self) -> List[int]:
        return sorted(self.price_by_ts.keys())