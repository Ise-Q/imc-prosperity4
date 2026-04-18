"""
ActivityLogCreator
==================
Produces activity log lines in the format expected by the Prosperity platform
and compatible with jmerle's visualizer.

Activity log format (semicolon-separated):
  day;timestamp;product;bid_price_1;bid_volume_1;...;ask_price_1;ask_volume_1;...;mid_price;pnl
"""
from typing import Dict, List, Optional
from .models.input import PriceRow
from .order_match_maker import Fill


class ActivityLogCreator:

    def create_header(self) -> str:
        cols = ["day", "timestamp", "product",
                "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
                "bid_price_3", "bid_volume_3",
                "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
                "ask_price_3", "ask_volume_3",
                "mid_price", "profit_and_loss"]
        return ";".join(cols)

    def create_row(
        self,
        price_row: PriceRow,
        pnl: float,
    ) -> str:
        def fmt_opt(v: Optional[float]) -> str:
            if v is None:
                return ""
            return str(int(v)) if v == int(v) else str(v)

        parts = [
            str(price_row.day),
            str(price_row.timestamp),
            price_row.product,
        ]
        for i in range(3):
            parts.append(fmt_opt(price_row.bid_prices[i]))
            parts.append(fmt_opt(price_row.bid_volumes[i]))
        for i in range(3):
            parts.append(fmt_opt(price_row.ask_prices[i]))
            parts.append(fmt_opt(price_row.ask_volumes[i]))
        parts.append(str(price_row.mid_price))
        parts.append(str(round(pnl, 2)))

        return ";".join(parts)
