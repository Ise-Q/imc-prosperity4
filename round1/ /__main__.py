"""
CLI entry point for the backtester.

Usage:
  python -m prosperity4bt <trader.py> <round> [day_spec] [options]

Examples:
  python -m prosperity4bt trader.py 1              # all days of round 1
  python -m prosperity4bt trader.py 1-0            # round 1 day 0 only
  python -m prosperity4bt trader.py 1--2 1--1 1-0  # specific days
  python -m prosperity4bt trader.py 1 --out my.log
  python -m prosperity4bt trader.py 1 --no-merge   # don't carry PnL across days
  python -m prosperity4bt trader.py 1 --data /path/to/csvs
"""
import sys
import os
import argparse
import re


def parse_day_spec(spec: str):
    """
    Parse a day specification string.
    '1'    -> (round=1, day=None)  meaning all days
    '1-0'  -> (round=1, day=0)
    '1--2' -> (round=1, day=-2)
    """
    m = re.fullmatch(r"(\d+)(-(-?\d+))?", spec)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid spec: {spec!r}. Use '1', '1-0', or '1--2'.")
    round_num = int(m.group(1))
    day       = int(m.group(3)) if m.group(3) is not None else None
    return round_num, day


# Default days available per round
DEFAULT_DAYS = {
    0: [0],
    1: [-2, -1, 0],
    2: [-2, -1, 0],
    3: [-2, -1, 0],
    4: [-2, -1, 0],
    5: [-2, -1, 0],
}


def main():
    parser = argparse.ArgumentParser(
        description="IMC Prosperity 4 Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("trader",    help="Path to trader .py file")
    parser.add_argument("specs",     nargs="+", help="Round/day specs e.g. 1  1-0  1--2")
    parser.add_argument("--out",     default=None, help="Output .log file path")
    parser.add_argument("--data",    default=".",  help="Directory containing CSV files (default: .)")
    parser.add_argument("--no-merge", action="store_true", help="Don't carry PnL/positions across days")
    args = parser.parse_args()

    # Build round_days dict from specs
    round_days: dict[int, list[int]] = {}
    for spec in args.specs:
        round_num, day = parse_day_spec(spec)
        if day is not None:
            round_days.setdefault(round_num, []).append(day)
        else:
            # All known days for this round
            round_days[round_num] = DEFAULT_DAYS.get(round_num, [-2, -1, 0])

    # Ensure datamodel.py is findable
    trader_dir = os.path.dirname(os.path.abspath(args.trader))
    if trader_dir not in sys.path:
        sys.path.insert(0, trader_dir)
    # Also add current working dir
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    from backtester.backtester import BackTester

    bt = BackTester(data_dir=args.data, round_days=round_days)
    bt.run(
        trader_path = args.trader,
        out         = args.out,
        merge_pnl   = not args.no_merge,
    )


if __name__ == "__main__":
    main()