"""
BackTester
==========
Top-level controller. Iterates over rounds and days, delegates to TestRunner,
merges results, and writes the output log file.

Output log format
-----------------
The file is split into two sections separated by blank lines, matching
jmerle's Prosperity visualizer format:

  [Activity log section]
  day;timestamp;product;bid_price_1;...;mid_price;profit_and_loss
  <rows>

  [Sandbox logs section]
  {"sandboxLog": "", "lambdaLog": "<trader stdout>", "timestamp": <ts>}
  ...

Usage
-----
  from prosperity4bt.back_tester import BackTester
  bt = BackTester(data_dir=".", round_days={1: [-2, -1, 0]})
  bt.run("trader.py", out="results.log")
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .back_data_reader import BackDataReader
from .test_runner import TestRunner
from .activity_log_creator import ActivityLogCreator
from .models.output import BacktestResult, DayResult


class BackTester:
    """
    Parameters
    ----------
    data_dir   : directory containing the CSV files
    round_days : {round_num: [day, ...]}  e.g. {1: [-2, -1, 0]}
    """

    def __init__(
        self,
        data_dir: str = ".",
        round_days: Optional[Dict[int, List[int]]] = None,
    ):
        self.data_dir   = data_dir
        self.round_days = round_days or {1: [-2, -1, 0]}
        self._reader    = BackDataReader()
        self._runner    = TestRunner()
        self._log_creator = ActivityLogCreator()

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        trader_path: str,
        out: Optional[str] = None,
        merge_pnl: bool = True,
    ) -> BacktestResult:
        """
        Run the full backtest.

        Parameters
        ----------
        trader_path : path to the trader .py file
        out         : output .log file path (default: timestamped filename)
        merge_pnl   : if True, carry positions/cash across days

        Returns
        -------
        BacktestResult
        """
        trader = self._load_trader(trader_path)
        result = BacktestResult()

        # Persistent state across days (when merge_pnl=True)
        trader_data = ""
        positions: Dict[str, int]   = defaultdict(int)
        cash: Dict[str, float]      = defaultdict(float)

        print(f"BackTester starting...")
        print(f"  Trader : {trader_path}")
        print(f"  Data   : {self.data_dir}")
        print(f"  Rounds : {self.round_days}")

        for round_num, days in sorted(self.round_days.items()):
            for day in days:
                prices_file = f"prices_round_{round_num}_day_{day}.csv"
                trades_file = f"trades_round_{round_num}_day_{day}.csv"
                prices_path = os.path.join(self.data_dir, prices_file)
                trades_path = os.path.join(self.data_dir, trades_file)

                if not os.path.exists(prices_path):
                    print(f"  [SKIP] {prices_file} not found")
                    continue

                print(f"  Running round {round_num} day {day}...")

                # Stage 0: read data
                data = self._reader.read(prices_path, trades_path, day)

                # Run simulation
                day_result, trader_data, positions, cash = self._runner.run(
                    trader      = trader,
                    data        = data,
                    round_num   = round_num,
                    initial_trader_data = trader_data if merge_pnl else "",
                    initial_positions   = positions   if merge_pnl else {},
                    initial_cash        = cash        if merge_pnl else {},
                )
                result.day_results.append(day_result)

        # Merge results & write log
        self._print_summary(result, cash, positions)
        out_path = out or self._default_out_path()
        self._write_log(result, out_path)
        print(f"\n✓ Log written to: {out_path}")
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_trader(self, path: str):
        """Dynamically load the Trader class from a .py file."""
        # Ensure datamodel is importable from the trader's directory
        trader_dir = os.path.dirname(os.path.abspath(path))
        if trader_dir not in sys.path:
            sys.path.insert(0, trader_dir)

        spec = importlib.util.spec_from_file_location("trader_module", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Trader()

    def _default_out_path(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}.log"

    def _print_summary(self, result: BacktestResult, cash: Dict, positions: Dict):
        """Print final PnL summary to console."""
        # Gather last pnl per product from last timestamp of last day
        if not result.day_results:
            return
        last_day = result.day_results[-1]
        if not last_day.timestamp_results:
            return
        last_ts = last_day.timestamp_results[-1]
        total = sum(last_ts.pnl.values())
        print(f"\n  Final positions : {dict(positions)}")
        print(f"  Final PnL       : {total:,.2f}")
        for product, pnl in last_ts.pnl.items():
            print(f"    {product:<30} {pnl:>12,.2f}")

    def _write_log(self, result: BacktestResult, out_path: str) -> None:
        """
        Write output log in Prosperity visualizer JSON format:
        {
            "submissionId": "local",
            "activitiesLog": "day;timestamp;...\n...",
            "logs": "sandbox log lines",
            "tradeHistory": [...]
        }
        """
        activity_header = self._log_creator.create_header()
        activity_rows: List[str] = [activity_header]
        sandbox_lines: List[str] = []
        trade_history: List[dict] = []

        for day_result in result.day_results:
            for ts_result in day_result.timestamp_results:
                if ts_result.activity_log:
                    activity_rows.append(ts_result.activity_log)

                sandbox_lines.append(json.dumps({
                    "sandboxLog": "",
                    "lambdaLog" : ts_result.lambda_log.rstrip("\n"),
                    "timestamp" : ts_result.timestamp,
                }, separators=(",", ":")))

                for product, trades in ts_result.own_trades.items():
                    for trade in trades:
                        trade_history.append({
                            "timestamp" : ts_result.timestamp,
                            "buyer"     : trade.buyer,
                            "seller"    : trade.seller,
                            "symbol"    : trade.symbol,
                            "currency"  : "XIREC",
                            "price"     : trade.price,
                            "quantity"  : trade.quantity,
                        })

        seen = set()
        deduped_trades = []
        for t in trade_history:
            key = (t["symbol"], t["timestamp"], t["price"], t["quantity"], t["buyer"], t["seller"])
            if key not in seen:
                seen.add(key)
                deduped_trades.append(t)

        output = {
            "submissionId" : "local",
            "activitiesLog": "\n".join(activity_rows) + "\n",
            "logs"         : "\n".join(sandbox_lines) + "\n",
            "tradeHistory" : deduped_trades,
        }

        with open(out_path, "w") as f:
            json.dump(output, f)