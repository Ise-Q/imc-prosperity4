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

## Interactive visualizer (`viz.py`)

Launch a Dash app in the browser to inspect any run — either the Rust local
backtester output or a Prosperity website submission.

```sh
# Local backtester run
uv run python tools/viz.py round1/logs/local-20260418-203355

# Website submission run
uv run python tools/viz.py round1/logs/263711

# Single file also works
uv run python tools/viz.py round1/logs/local-20260418-203355/backtest.log
```

Flags: `--port 8050`, `--host 127.0.0.1`, `--no-open` (skip auto-open), `--debug`.

Per product the UI shows four stacked panels with a shared x-axis: price +
trades, spread, position (±limit bands, resets per day), PnL. Toggles along
the top:

- Layout: `single` (product dropdown) or `stacked` (all products).
- Days: checkbox filter for multi-day runs.
- Overlays: `mid`, `ob_vwap` (orderbook VWAP across all 3 levels),
  `wall_mid` (cross-weighted max-volume-level price), `empty_ob` (grey
  dotted verticals at ticks where *both* sides of the orderbook are fully
  empty — no bid or ask price at any level).
- Trade layers: `own` (triangles, green BUY / red SELL, size ∝ qty) vs
  `market` (grey x). Hover shows price, quantity, side.
- Depth levels: toggle L1/L2/L3 independently. One line per level per
  side; vivid at L1 and fading at L3.

Mid, spread, and the overlays use **best bid / best ask** (first non-NaN
price across L1→L3), not hardcoded L1. If one side is fully empty, mid
falls back to the other side's best; only when both sides are fully empty
does mid become NaN (and `empty_ob` marks those ticks).

## Upgrading the backtester

```sh
uv lock --upgrade-package prosperity4btest
uv sync
```

## Adding a new round

Edit `POSITION_LIMITS` in `run_backtest.py` with the products and caps from
`docs/ROUND{N}.md`. Unlisted products fall back to the backtester's internal
default of 50, which is usually wrong for Prosperity.
