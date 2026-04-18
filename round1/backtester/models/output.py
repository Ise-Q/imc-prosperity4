"""
Output data models — hold simulation results at each stage.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class TimestampResult:
    """Result for a single timestamp (Stage 1 → 3)."""
    timestamp: int
    day: int
    # Stage 1 — raw trader output
    orders: Dict[str, List[Any]]        # product -> [Order]
    trader_data_out: str
    lambda_log: str                     # captured stdout from trader
    # Stage 2 — activity log line
    activity_log: str
    # Stage 3 — after order matching
    own_trades: Dict[str, List[Any]]    # product -> [Trade]
    positions: Dict[str, int]
    pnl: Dict[str, float]               # product -> mark-to-market PnL


@dataclass
class DayResult:
    """Aggregated result for one day (Stage 4)."""
    round: int
    day: int
    timestamp_results: List[TimestampResult] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Final merged result across all rounds/days."""
    day_results: List[DayResult] = field(default_factory=list)