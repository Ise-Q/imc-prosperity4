# tools/

Project-specific utilities that sit on top of third-party tools. The current
contents are a thin wrapper around the
[`prosperity4btest`](https://github.com/nabayansaha/imc-prosperity-4-backtester)
backtester.

## Setup

The backtester is installed from PyPI as a regular project dependency, so new
teammates only need:

```sh
uv sync
```

No cloning, no submodules.

## Running a backtest

```sh
# All days of round 1, using round1/trader.py
uv run python tools/run_backtest.py --round 1

# A specific day
uv run python tools/run_backtest.py --round 1 --day 1-0

# Backtest a strategy variant and merge PnL across days
uv run python tools/run_backtest.py --round 1 \
    --trader round1/strats/trader8-2.py --merge-pnl

# Open the web visualizer when done
uv run python tools/run_backtest.py --round 1 --vis
```

The wrapper forwards the common flags (`--merge-pnl`, `--vis`, `--print`,
`--match-trades {all|worse|none}`, `--no-progress`). Anything else can be
passed through verbatim after `--extra`, e.g. overriding a position limit:

```sh
uv run python tools/run_backtest.py --round 1 --extra --limit ASH_COATED_OSMIUM:60
```

## Where logs go

Local backtest output lands at:

```
round{N}/logs/local-<YYYYMMDD-HHMMSS>/backtest.log
```

The `local-` prefix distinguishes wrapper-generated runs from numeric
run-id folders downloaded from the Prosperity website. The
`analyze_results.ipynb` notebook accepts either — just point `LOG_DIR` at
the folder you want to inspect.

## Upgrading the backtester

```sh
uv lock --upgrade-package prosperity4btest
uv sync
```

## Adding a new round

Edit `POSITION_LIMITS` in `run_backtest.py` with the products and caps from
`docs/ROUND{N}.md`. Unlisted products fall back to the backtester's internal
default of 50, which is usually wrong for Prosperity.
