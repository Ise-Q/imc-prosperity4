# tools/

Project-specific utilities built on top of the
[`prosperity4btest`](https://github.com/nabayansaha/imc-prosperity-4-backtester)
backtester. `run_backtest.py` drives the backtester's tick helpers in-process
so a multi-day run is one continuous simulation instead of N independent days
stitched together — see "Continuous runs" below.

## Setup

The backtester is installed from PyPI as a regular project dependency, so new
teammates only need:

```sh
uv sync
```

No cloning, no submodules.

## Running a backtest

```sh
# All days of round 1, using round1/trader.py — continuous across days
uv run python tools/run_backtest.py --round 1

# A specific day
uv run python tools/run_backtest.py --round 1 --day 1-0

# A strategy variant
uv run python tools/run_backtest.py --round 1 --trader round1/strats/trader8-2.py

# Override a position limit
uv run python tools/run_backtest.py --round 1 --limit ASH_COATED_OSMIUM:60

# Reproduce the old per-day-reset behavior (debug only)
uv run python tools/run_backtest.py --round 1 --no-carry
```

Other flags: `--print`, `--match-trades {all|worse|none}`, `--no-progress`,
`--data <dir>` (point at a non-default data directory).

## Continuous runs

By default, a multi-day run shares **position, traderData, own/market
trades, and cumulative PnL** across day boundaries. One `Trader()` instance
handles every tick in order, and timestamps are globally monotonic
(day -2 → 0..999900, day -1 → 1_000_000..1_999_900, etc.). Position limits
are enforced against the carried inventory — an algo that ends day -2 long
80 can only sell on day -1 until flat, never add long exposure.

Use `--no-carry` to opt out: each day starts with `state.position={}`,
`traderData=""`, and PnL reset to 0. Useful only for A/B comparisons
against the pre-fix behavior — a trader that carries real inventory will
appear to violate position limits across day boundaries because its side
of the invariant (position) is being reset under it.

`--merge-pnl` is accepted for backwards compatibility but is a no-op under
`--carry` — continuous runs are already PnL-continuous by construction.

## Invariant test

```sh
uv run python tools/test_run_backtest.py
```

Runs a full round 1 continuous backtest and checks: position never
exceeds ±limit; position is continuous at day boundaries; PnL is continuous
at day boundaries. Also runs a `--no-carry` control to confirm the fix is
meaningful.

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

Launch a Dash app in the browser to inspect any run — either a local
backtester run or a Prosperity website submission.

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
trades, spread, position (±limit bands, continuous across days), PnL. Toggles along
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
