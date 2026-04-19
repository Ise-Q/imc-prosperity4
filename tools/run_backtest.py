"""In-process continuous multi-day backtest runner.

Drives the `prosperity4bt` package's tick helpers directly, threading state
(position, traderData, own/market trades, PnL) across day boundaries so the
trader sees a single continuous timeline instead of three independent days.

The Python `prosperity4btest` CLI runs each day as an independent simulation
with `state.position={}`, `traderData=""`, and `data.profit_loss={}` reset
at the start of every day. That makes position limits unenforceable across
day boundaries — the trader thinks it's flat on day N+1 even though it
carried ±80 inventory from day N. This wrapper fixes that by running the
entire multi-day range inside one Python process with shared state.

Output still lands at `round{N}/logs/local-<timestamp>/backtest.log` in the
same three-section format the visualizer parses.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional

from prosperity4bt.__main__ import (
    parse_algorithm,
    parse_days,
    write_output,
)
from prosperity4bt.data import read_day_data
from prosperity4bt.datamodel import Observation, Trade, TradingState
from prosperity4bt.file_reader import FileReader, FileSystemReader, PackageResourcesReader
from prosperity4bt.models import BacktestResult, SandboxLogRow, TradeMatchingMode
from prosperity4bt.runner import (
    create_activity_logs,
    enforce_limits,
    match_orders,
    prepare_state,
    type_check_orders,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent

DAY_TIMESTAMP_SPAN = 1_000_000

POSITION_LIMITS: dict[int, dict[str, int]] = {
    0: {"EMERALDS": 80, "TOMATOES": 80},
    1: {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80},
}


def _rebase_day_data(data, offset: int) -> None:
    """Rewrite data.prices/trades/observations keys to global timestamps.

    Day-local timestamps (0..999900) become globally increasing so
    state.timestamp is monotonic across the whole run. Trade objects inside
    data.trades get new `timestamp` fields too so market trades written to
    the output log show global timestamps.
    """
    if offset == 0:
        return

    data.prices = {ts + offset: rows for ts, rows in data.prices.items()}

    # data.trades is a defaultdict(lambda: defaultdict(list)) in prosperity4bt.data —
    # match_orders() accesses missing timestamps directly (data.trades[ts]) and relies
    # on the default_factory to return an empty dict. Preserve that here.
    rebased_trades = defaultdict(lambda: defaultdict(list))
    for ts, product_map in data.trades.items():
        global_ts = ts + offset
        inner: dict[str, list[Trade]] = defaultdict(list)
        for product, trade_list in product_map.items():
            inner[product] = [
                Trade(t.symbol, t.price, t.quantity, t.buyer, t.seller, global_ts)
                for t in trade_list
            ]
        rebased_trades[global_ts] = inner
    data.trades = rebased_trades

    data.observations = {ts + offset: row for ts, row in data.observations.items()}


def run_continuous(
    algorithm_path: Path,
    day_strs: list[str],
    out_path: Optional[Path],
    file_reader: Optional[FileReader] = None,
    match_mode: TradeMatchingMode = TradeMatchingMode.all,
    limits_override: Optional[dict[str, int]] = None,
    carry: bool = True,
    print_output: bool = False,
    show_progress: bool = True,
) -> BacktestResult:
    """Run a multi-day backtest with optional cross-day state continuity.

    Parameters
    ----------
    algorithm_path
        Trader .py file with a `Trader` class.
    day_strs
        List of day specifiers accepted by prosperity4bt — e.g. ["1"] for
        all days in round 1, or ["1-0"] for round 1 day 0.
    out_path
        Destination for the three-section log. None skips writing.
    carry
        When True (default), position / traderData / own_trades /
        market_trades / PnL are threaded across day boundaries. When
        False, each day starts fresh — useful for reproducing the
        pre-fix per-day-reset behavior.

    Returns the merged BacktestResult (all days, timestamps already
    globally offset).
    """
    file_reader = file_reader or PackageResourcesReader()

    trader_module = parse_algorithm(algorithm_path)
    if not hasattr(trader_module, "Trader"):
        raise SystemExit(f"{algorithm_path}: no Trader class exported")
    trader = trader_module.Trader()

    parsed_days = parse_days(file_reader, day_strs)

    trader_data = ""
    position: dict[str, int] = {}
    own_trades: dict[str, list] = {}
    market_trades: dict[str, list] = {}
    profit_loss: dict[str, float] = {}

    merged = BacktestResult(
        round_num=parsed_days[0][0],
        day_num=parsed_days[0][1],
        sandbox_logs=[],
        activity_logs=[],
        trades=[],
    )

    for day_idx, (round_num, day_num) in enumerate(parsed_days):
        data = read_day_data(file_reader, round_num, day_num, no_names=True)
        offset = day_idx * DAY_TIMESTAMP_SPAN
        _rebase_day_data(data, offset)

        if carry:
            for product in data.products:
                profit_loss.setdefault(product, 0.0)
            data.profit_loss = profit_loss
            state = TradingState(
                traderData=trader_data,
                timestamp=0,
                listings={},
                order_depths={},
                own_trades=own_trades,
                market_trades=market_trades,
                position=position,
                observations=Observation({}, {}),
            )
        else:
            trader_data = ""
            position = {}
            own_trades = {}
            market_trades = {}
            state = TradingState(
                traderData="",
                timestamp=0,
                listings={},
                order_depths={},
                own_trades=own_trades,
                market_trades=market_trades,
                position=position,
                observations=Observation({}, {}),
            )

        day_result = BacktestResult(
            round_num=round_num,
            day_num=day_num,
            sandbox_logs=[],
            activity_logs=[],
            trades=[],
        )

        timestamps = sorted(data.prices.keys())
        iterator = timestamps
        if show_progress and tqdm is not None:
            iterator = tqdm(timestamps, ascii=True, desc=f"day {day_num}")

        for timestamp in iterator:
            state.timestamp = timestamp
            state.traderData = trader_data
            prepare_state(state, data)

            stdout = StringIO()
            stdout.close = lambda: None  # type: ignore[method-assign]
            if print_output:
                print(f"-- tick {timestamp} --")
                orders, conversions, trader_data = trader.run(state)
            else:
                with redirect_stdout(stdout):
                    orders, conversions, trader_data = trader.run(state)

            sandbox_row = SandboxLogRow(
                timestamp=timestamp,
                sandbox_log="",
                lambda_log=stdout.getvalue().rstrip(),
            )
            day_result.sandbox_logs.append(sandbox_row)

            type_check_orders(orders)
            create_activity_logs(state, data, day_result)
            enforce_limits(state, data, orders, sandbox_row, limits_override)
            match_orders(state, data, orders, day_result, match_mode, limits_override)

        merged.sandbox_logs.extend(day_result.sandbox_logs)
        merged.activity_logs.extend(day_result.activity_logs)
        merged.trades.extend(day_result.trades)

    if out_path is not None:
        write_output(out_path, merged)

    return merged


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a continuous multi-day backtest (state carries across days by default).",
    )
    p.add_argument("--round", dest="round_num", type=int, required=True,
                   help="Round number (e.g. 1).")
    p.add_argument("--trader", type=Path, default=None,
                   help="Path to trader file. Defaults to round{N}/trader.py.")
    p.add_argument("--day", type=str, default=None,
                   help="Specific day as N-D (e.g. 1-0). Omit to run all days in the round.")
    p.add_argument("--no-carry", action="store_true",
                   help="Disable cross-day state continuity (reproduces pre-fix per-day-reset).")
    p.add_argument("--merge-pnl", action="store_true",
                   help="(Deprecated, no-op with carry=True — continuous runs are already continuous.)")
    p.add_argument("--print", dest="print_output", action="store_true")
    p.add_argument("--match-trades", choices=["all", "worse", "none"], default="all")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--limit", dest="limit_overrides", action="append", default=[],
                   help="Override position limit for a product (PRODUCT:LIMIT). Repeat for multiple.")
    p.add_argument("--data", type=Path, default=None,
                   help="Optional path to a data directory (defaults to the bundled package data).")
    return p


def resolve_trader(args: argparse.Namespace) -> Path:
    if args.trader is not None:
        return args.trader.resolve()
    default = REPO_ROOT / f"round{args.round_num}" / "trader.py"
    if not default.exists():
        sys.exit(f"[run_backtest] trader not found: {default}. Pass --trader explicitly.")
    return default


def resolve_day_strs(args: argparse.Namespace) -> list[str]:
    if args.day is None:
        return [str(args.round_num)]
    if "-" not in args.day:
        sys.exit(f"[run_backtest] --day must be in N-D form (e.g. 1-0), got: {args.day}")
    return [args.day]


def parse_limit_overrides(items: list[str], round_num: int) -> dict[str, int]:
    merged: dict[str, int] = dict(POSITION_LIMITS.get(round_num, {}))
    for item in items:
        if ":" not in item:
            sys.exit(f"[run_backtest] --limit must be PRODUCT:NUMBER, got {item!r}")
        sym, num = item.split(":", 1)
        try:
            merged[sym.strip()] = int(num.strip())
        except ValueError:
            sys.exit(f"[run_backtest] invalid limit number in {item!r}")
    return merged


def main() -> int:
    args = build_parser().parse_args()

    trader = resolve_trader(args)
    day_strs = resolve_day_strs(args)
    limits_override = parse_limit_overrides(args.limit_overrides, args.round_num)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"local-{timestamp}"
    log_dir = REPO_ROOT / f"round{args.round_num}" / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "backtest.log"

    file_reader: FileReader
    if args.data is not None:
        file_reader = FileSystemReader(args.data)
    else:
        file_reader = PackageResourcesReader()

    print(f"[run_backtest] trader : {trader}")
    print(f"[run_backtest] days   : {day_strs}")
    print(f"[run_backtest] carry  : {not args.no_carry}")
    print(f"[run_backtest] limits : {limits_override}")
    print(f"[run_backtest] log    : {out_path}")

    if args.merge_pnl and not args.no_carry:
        print("[run_backtest] note   : --merge-pnl is a no-op with continuous runs.")

    match_mode = TradeMatchingMode(args.match_trades)

    run_continuous(
        algorithm_path=trader,
        day_strs=day_strs,
        out_path=out_path,
        file_reader=file_reader,
        match_mode=match_mode,
        limits_override=limits_override,
        carry=not args.no_carry,
        print_output=args.print_output,
        show_progress=not args.no_progress,
    )

    print(f"[run_backtest] wrote  : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
