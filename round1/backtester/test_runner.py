"""
TestRunner
==========
Runs the backtest simulation for a single (round, day).

Execution flow per timestamp:
  Stage 0 — data loaded by BackDataReader (passed in)
  Stage 1 — call trader.run(), capture stdout (lambda_log)
  Stage 2 — create activity log row
  Stage 3 — match orders, update positions/PnL
  Stage 4 — aggregate into DayResult
"""
import io
import sys
from collections import defaultdict
from typing import Dict

from .back_data_reader import BackDataReader
from .order_match_maker import OrderMatchMaker
from .activity_log_creator import ActivityLogCreator
from .models.input import BacktestData
from .models.output import DayResult, TimestampResult

# Import Prosperity datamodel types
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from datamodel import OrderDepth, TradingState, Trade, Listing, Observation


POSITION_LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}
DEFAULT_LIMIT = 80


class TestRunner:
    """Simulates one full day for a given trader algorithm."""

    def __init__(self):
        self._matcher  = OrderMatchMaker()
        self._log_creator = ActivityLogCreator()

    def run(
        self,
        trader,
        data: BacktestData,
        round_num: int,
        initial_trader_data: str = "",
        initial_positions: Dict[str, int] = None,
        initial_cash: Dict[str, float] = None,
    ) -> DayResult:
        """
        Parameters
        ----------
        trader            : Trader instance (has .run(state) method)
        data              : BacktestData for this day (Stage 0)
        round_num         : competition round number
        initial_trader_data : traderData string carried from previous day
        initial_positions   : position dict carried from previous day
        initial_cash        : cumulative cash flow dict from previous day

        Returns
        -------
        DayResult with all TimestampResults
        """
        result = DayResult(round=round_num, day=data.day)

        trader_data = initial_trader_data
        positions: Dict[str, int] = dict(initial_positions or {})
        cash: Dict[str, float]    = dict(initial_cash or {})
        last_pnl: Dict[str, float] = {}

        # Track own_trades per product (fills since last tick — reset each ts)
        own_trades: Dict[str, list] = defaultdict(list)

        for ts in data.timestamps:
            price_rows   = data.price_by_ts[ts]
            trade_rows   = data.trades_by_ts.get(ts, [])

            # ── Build TradingState ────────────────────────────────────────
            order_depths: Dict[str, OrderDepth] = {}
            listings: Dict[str, Listing] = {}

            for product, pr in price_rows.items():
                od = OrderDepth()
                for i in range(3):
                    bp, bv = pr.bid_prices[i], pr.bid_volumes[i]
                    ap, av = pr.ask_prices[i], pr.ask_volumes[i]
                    if bp is not None and bv is not None:
                        od.buy_orders[int(bp)] = int(bv)
                    if ap is not None and av is not None:
                        od.sell_orders[int(ap)] = -int(av)
                order_depths[product] = od
                listings[product]     = Listing(product, product, product)

            market_trades_state: Dict[str, list] = {
                p: [
                    Trade(
                        symbol   = tr.symbol,
                        price    = int(tr.price),
                        quantity = tr.quantity,
                        buyer    = tr.buyer,
                        seller   = tr.seller,
                        timestamp= ts,
                    )
                    for tr in trade_rows if tr.symbol == p
                ]
                for p in order_depths
            }

            state = TradingState(
                traderData   = trader_data,
                timestamp    = ts,
                listings     = listings,
                order_depths = order_depths,
                own_trades   = {p: list(own_trades.get(p, [])) for p in order_depths},
                market_trades= market_trades_state,
                position     = dict(positions),
                observations = Observation(
                    plainValueObservations={},
                    conversionObservations={},
                ),
            )

            # ── Stage 1: Run trader, capture stdout ───────────────────────
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                result_tuple = trader.run(state)
            except Exception as exc:
                result_tuple = ({}, 0, trader_data)
                print(f"[ERROR ts={ts}] {exc}", file=sys.stderr)
            finally:
                sys.stdout = old_stdout

            lambda_log = captured.getvalue()
            orders_by_product, _conversions, trader_data = result_tuple

            # ── Stage 3: Match orders ─────────────────────────────────────
            fills_by_product, positions = self._matcher.match(
                orders_by_product = orders_by_product,
                order_depths      = order_depths,
                market_trade_rows = trade_rows,
                positions         = positions,
                limits            = POSITION_LIMITS,
                default_limit     = DEFAULT_LIMIT,
            )

            # Update own_trades for next tick
            own_trades = defaultdict(list)
            for product, fills in fills_by_product.items():
                for fill in fills:
                    cash[product] = cash.get(product, 0.0) + (-fill.price * fill.quantity)
                    own_trades[product].append(
                        Trade(
                            symbol   = product,
                            price    = fill.price,
                            quantity = abs(fill.quantity),
                            buyer    = "SUBMISSION" if fill.quantity > 0 else "",
                            seller   = "" if fill.quantity > 0 else "SUBMISSION",
                            timestamp= ts,
                        )
                    )

            # ── Stage 2: Activity log + PnL ───────────────────────────────
            pnl_this_ts: Dict[str, float] = {}
            activity_lines: list[str] = []

            for product, pr in price_rows.items():
                pos     = positions.get(product, 0)
                c       = cash.get(product, 0.0)
                if pr.mid_price and pr.mid_price > 0:
                    mtm_pnl = c + pos * pr.mid_price
                else:
                    mtm_pnl = last_pnl.get(product, 0.0)
                pnl_this_ts[product] = mtm_pnl
                last_pnl[product] = mtm_pnl
                activity_lines.append(
                    self._log_creator.create_row(pr, mtm_pnl)
                )

            ts_result = TimestampResult(
                timestamp       = ts,
                day             = data.day,
                orders          = orders_by_product,
                trader_data_out = trader_data,
                lambda_log      = lambda_log,
                activity_log    = "\n".join(activity_lines),
                own_trades      = {p: list(own_trades.get(p, [])) for p in order_depths},
                positions       = dict(positions),
                pnl             = pnl_this_ts,
            )
            result.timestamp_results.append(ts_result)

        return result, trader_data, dict(positions), dict(cash)
