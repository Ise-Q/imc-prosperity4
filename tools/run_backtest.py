"""Thin wrapper around `prosperity4btest` that standardizes log output location
and per-round position-limit defaults for this repo.

Local runs land in `round{N}/logs/local-<timestamp>/backtest.log` so they are
visually distinct from run-id folders downloaded from the Prosperity website
(which are numeric, e.g. `round1/logs/242328/`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-round position limits, sourced from docs/ROUND{N}.md.
# The backtester defaults unknown products to 50, so we pass these explicitly.
POSITION_LIMITS: dict[int, dict[str, int]] = {
    0: {"EMERALDS": 80, "TOMATOES": 80},
    1: {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run prosperity4btest on a trader with repo-standard log paths."
    )
    p.add_argument("--round", dest="round_num", type=int, required=True,
                   help="Round number (e.g. 1).")
    p.add_argument("--trader", type=Path, default=None,
                   help="Path to trader file. Defaults to round{N}/trader.py.")
    p.add_argument("--day", type=str, default=None,
                   help="Specific day as N-D (e.g. 1-0). Omit to run all days in the round.")

    # Pass-through flags
    p.add_argument("--merge-pnl", action="store_true")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--print", dest="print_output", action="store_true")
    p.add_argument("--match-trades", choices=["all", "worse", "none"], default=None)
    p.add_argument("--no-progress", action="store_true")

    # Escape hatch: extra args passed verbatim to prosperity4btest, e.g. --limit FOO:50
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded verbatim to prosperity4btest (put after --).")
    return p


def resolve_trader(args: argparse.Namespace) -> Path:
    if args.trader is not None:
        return args.trader.resolve()
    default = REPO_ROOT / f"round{args.round_num}" / "trader.py"
    if not default.exists():
        sys.exit(f"[run_backtest] trader not found: {default}. Pass --trader explicitly.")
    return default


def resolve_day_arg(args: argparse.Namespace) -> str:
    if args.day is None:
        return str(args.round_num)
    # Basic sanity: "N-D" form
    if "-" not in args.day:
        sys.exit(f"[run_backtest] --day must be in N-D form (e.g. 1-0), got: {args.day}")
    return args.day


def build_limit_args(round_num: int) -> list[str]:
    limits = POSITION_LIMITS.get(round_num, {})
    out: list[str] = []
    for product, cap in limits.items():
        out += ["--limit", f"{product}:{cap}"]
    return out


def main() -> int:
    args = build_parser().parse_args()

    trader = resolve_trader(args)
    day_arg = resolve_day_arg(args)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"local-{timestamp}"
    log_dir = REPO_ROOT / f"round{args.round_num}" / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "backtest.log"

    cmd: list[str] = [
        "uv", "run", "prosperity4btest",
        str(trader),
        day_arg,
        "--out", str(out_path),
    ]
    cmd += build_limit_args(args.round_num)

    if args.merge_pnl:
        cmd.append("--merge-pnl")
    if args.vis:
        cmd.append("--vis")
    if args.print_output:
        cmd.append("--print")
    if args.match_trades is not None:
        cmd += ["--match-trades", args.match_trades]
    if args.no_progress:
        cmd.append("--no-progress")
    cmd += list(args.extra)

    print(f"[run_backtest] trader : {trader}")
    print(f"[run_backtest] days   : {day_arg}")
    print(f"[run_backtest] log    : {out_path}")
    print(f"[run_backtest] cmd    : {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
