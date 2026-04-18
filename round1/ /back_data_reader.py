"""
BackDataReader
==============
Parses price and trade CSV files into BacktestData objects.
"""
import csv
import os
from typing import Optional
from .models.input import PriceRow, TradeRow, BacktestData


def _opt_float(val: str) -> Optional[float]:
    v = val.strip()
    return float(v) if v else None


def _opt_int(val: str) -> Optional[int]:
    v = val.strip()
    return int(float(v)) if v else None


class BackDataReader:
    """Reads and parses one day's worth of CSV market data."""

    def read(self, prices_path: str, trades_path: str, day: int) -> BacktestData:
        data = BacktestData(day=day)
        self._read_prices(prices_path, data)
        if os.path.exists(trades_path):
            self._read_trades(trades_path, data)
        return data

    # ── private ─────────────────────────────────────────────────────────────

    def _read_prices(self, path: str, data: BacktestData) -> None:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                ts      = int(row["timestamp"])
                product = row["product"]
                pr = PriceRow(
                    day        = int(row["day"]),
                    timestamp  = ts,
                    product    = product,
                    bid_prices = [_opt_float(row.get(f"bid_price_{i}", "")) for i in range(1, 4)],
                    bid_volumes= [_opt_int  (row.get(f"bid_volume_{i}", "")) for i in range(1, 4)],
                    ask_prices = [_opt_float(row.get(f"ask_price_{i}", "")) for i in range(1, 4)],
                    ask_volumes= [_opt_int  (row.get(f"ask_volume_{i}", "")) for i in range(1, 4)],
                    mid_price  = float(row["mid_price"]) if row.get("mid_price","").strip() else 0.0,
                    profit_and_loss = float(row["profit_and_loss"]) if row.get("profit_and_loss","").strip() else 0.0,
                )
                data.price_by_ts.setdefault(ts, {})[product] = pr

    def _read_trades(self, path: str, data: BacktestData) -> None:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                ts = int(row["timestamp"])
                tr = TradeRow(
                    timestamp = ts,
                    buyer     = row.get("buyer", "").strip(),
                    seller    = row.get("seller", "").strip(),
                    symbol    = row.get("symbol", row.get("product", "")).strip(),
                    currency  = row.get("currency", "").strip(),
                    price     = float(row["price"]),
                    quantity  = int(row["quantity"]),
                )
                data.trades_by_ts.setdefault(ts, []).append(tr)
