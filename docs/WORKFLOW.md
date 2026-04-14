# Team Workflow Guide

This document explains how we work together as a team: how the project is organized, how to analyze data, how to write algo bots, and how to submit.

Read `SETUP-INSTRUCTIONS.md` first if you haven't set up your environment yet.

---

## Project Structure

```
imc-prosperity4/
├── docs/                              # Include .md files you want your agentic tool (Claude Code) to read in here
│   └── Writing-an-algo-in-python.md   # Official Prosperity algo documentation
│   └── SETUP-INSTRUCTIONS.md
│   └── WORKFLOW.md
├── tutorial/
│   └── data/                          # Sample CSV data from the tutorial round
├── round1/                            # Round 1 work (to be created)
│   ├── data/                          # CSV data files for round 1
│   ├── notebooks/                     # EDA notebooks for round 1
│   └── trader.py                      # The algo bot submitted for round 1
├── round2/                            # Round 2 work (same structure)
│   └── ...
├── pyproject.toml                     # Project dependencies
├── uv.lock                            # Locked dependency versions (commit this!)
```

Each round gets its own folder. We follow the pattern `roundN/` where N is the round number.

---

## Git Workflow

We use a branch per round to keep work isolated.

```bash
# Start of a new round — create a branch
git checkout -b round1

# Work on your notebook or trader.py...

# Stage and commit your changes
git add round1/
git commit -m "round1: add EDA notebook for EMERALDS"

# Push to GitHub
git push -u origin round1
```

After a round is complete, open a Pull Request on GitHub to merge `round1` into `main`.

**Never commit directly to `main`.**

---

## Step 1 — Receive Round Data

When a new round starts, Prosperity provides CSV data files. Save them into the round's data folder:

```
roundN/data/prices_round_N_day_X.csv
roundN/data/trades_round_N_day_X.csv
```

Create the folder if it doesn't exist:
```bash
mkdir -p round1/data
```

### CSV Format

The data files are **semicolon-delimited** (not comma). There are two types:

**Prices file** (`prices_round_N_day_X.csv`):
```
day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
```
- Up to 3 price levels on each side of the order book
- `mid_price` is the midpoint between best bid and best ask
- `profit_and_loss` is the cumulative PnL from that product

**Trades file** (`trades_round_N_day_X.csv`):
```
timestamp;buyer;seller;symbol;currency;price;quantity
```
- Records every trade that happened on the exchange
- `buyer`/`seller` are empty strings for bot trades

---

## Step 2 — Exploratory Data Analysis (EDA) in Jupyter

Create a notebook per product (or per theme) in `roundN/notebooks/`.

**Naming convention:** `eda_<product_name>.ipynb`, e.g. `eda_emeralds.ipynb`

### Standard Notebook Template

Start every EDA notebook with this imports block:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['figure.figsize'] = (14, 5)

# Load prices data — note sep=';'
prices = pd.read_csv('../data/prices_round_1_day_-1.csv', sep=';')

# Load trades data
trades = pd.read_csv('../data/trades_round_1_day_-1.csv', sep=';')

# Filter to the product you care about
product = 'EMERALDS'
p = prices[prices['product'] == product].copy()
t = trades[trades['symbol'] == product].copy()

print(p.shape, t.shape)
p.head()
```

### Suggested Analysis Sections

1. **Price Overview** — plot `mid_price` over time; check for trends, mean-reversion, or momentum
2. **Spread Analysis** — plot `ask_price_1 - bid_price_1` (the bid-ask spread); a tight, stable spread suggests market-making opportunities
3. **Order Book Depth** — compare volumes at each price level; large imbalances hint at price direction
4. **Trade Flow** — from the trades file, look at trade frequency, average size, and price impact
5. **Position Limit Context** — check the Prosperity platform for each product's position limit; note it in the notebook so you don't forget when writing the bot
6. **Fair Value Estimate** — derive your estimate of the product's "true" price (e.g., rolling mid-price, VWAP); this will become the `acceptable_price` in the algo

---

## Step 3 — Write the Algo Bot

Each round's bot lives in `roundN/trader.py`. This file is what you upload to Prosperity.

### Trader Class Structure

The Prosperity platform calls `Trader.run(state)` once per simulation iteration (~1,000 times during testing, ~10,000 in the final run). Your job is to return a list of orders.

```python
from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle

class Trader:

    def bid(self):
        """Required stub for Round 2. Safe to leave as-is for other rounds."""
        return 15

    def run(self, state: TradingState):
        """
        Main entry point. Called every iteration with the current market state.

        Args:
            state: TradingState object containing order books, positions, trades, etc.

        Returns:
            Tuple of (orders_dict, conversions, traderData)
            - orders_dict: {product: [Order, ...]}
            - conversions: int (usually 0 unless doing conversion arbitrage)
            - traderData: str (persisted state, max 50,000 characters)
        """

        # --- Restore persistent state ---
        if state.traderData:
            saved = jsonpickle.decode(state.traderData)
        else:
            saved = {}  # First iteration — initialize state here

        # --- Trading logic ---
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Your fair value estimate for this product
            fair_value = 10_000  # TODO: replace with your calculated value

            # Current position (capped by position limit — check Prosperity docs)
            position = state.position.get(product, 0)
            position_limit = 20  # TODO: set per-product from Prosperity docs

            # --- Buy side: hit sell orders below fair value ---
            for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                if ask_price < fair_value:
                    # ask_volume is negative in sell_orders — negate to get qty
                    qty = min(-ask_volume, position_limit - position)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        position += qty

            # --- Sell side: hit buy orders above fair value ---
            for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > fair_value:
                    qty = min(bid_volume, position_limit + position)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        position -= qty

            result[product] = orders

        # --- Save persistent state ---
        # Only store what you truly need across iterations.
        # The string is cut to 50,000 characters — keep it lean.
        saved['last_timestamp'] = state.timestamp
        traderData = jsonpickle.encode(saved)

        conversions = 0
        return result, conversions, traderData
```

### Key Rules

**Position limits** — Each product has an absolute position limit (e.g., ±20). If your aggregated buy orders would push you past the limit, the exchange rejects ALL your orders for that product in that iteration. Always check `state.position.get(product, 0)` before sizing orders.

**Order signs** — In `order_depth.sell_orders`, volumes are **negative**. In `order_depth.buy_orders`, volumes are **positive**. When you send an `Order`:
- Positive quantity = BUY
- Negative quantity = SELL

**Persistent state** — Lambda is stateless. Class variables and global variables are reset every call. Use `state.traderData` (a string you return from `run()`) to persist data. Serialize with `jsonpickle`:
```python
# Save
traderData = jsonpickle.encode({'prices': [10, 11, 12], 'position': 3})

# Load next iteration
data = jsonpickle.decode(state.traderData)
```

**Timeout** — Your `run()` must complete within ~900ms. If you're doing heavy computation, profile it locally first.

**Allowed imports** — Only standard library modules and the packages in `pyproject.toml` are available on the platform. Do not use libraries not listed there. If you need something new, run `uv add <package>` and commit `pyproject.toml` and `uv.lock` (then confirm with the team that it's allowed by Prosperity).

---

## Step 4 — Test Locally

Before submitting, test your bot locally using a mock `TradingState`. You'll need `datamodel.py` from the Prosperity platform (download it and place it in the project root).

Create a test script at `roundN/test_trader.py`:

```python
from datamodel import Listing, OrderDepth, Trade, TradingState
from trader import Trader

# Build a mock state
order_depths = {
    "EMERALDS": OrderDepth(
        buy_orders={9990: 5, 9989: 10},
        sell_orders={10010: -5, 10011: -10}
    ),
}
state = TradingState(
    traderData="",
    timestamp=1000,
    listings={
        "EMERALDS": Listing(symbol="EMERALDS", product="EMERALDS", denomination="XIRECS")
    },
    order_depths=order_depths,
    own_trades={"EMERALDS": []},
    market_trades={"EMERALDS": []},
    position={"EMERALDS": 0},
    observations={}
)

trader = Trader()
orders, conversions, traderData = trader.run(state)

print("Orders:", orders)
print("Conversions:", conversions)
print("traderData length:", len(traderData))
```

Run it:
```bash
uv run python round1/test_trader.py
```

Check that:
- No exceptions are thrown
- Orders are correctly sized (within position limits)
- `traderData` length stays well under 50,000 characters

---

## Step 5 — Submit to Prosperity

1. Go to the Prosperity platform and navigate to the current round's submission page
2. Upload `roundN/trader.py` — only this single file
3. Note the **UUID submission identifier** shown after upload (e.g., `59f81e67-f6c6-4254-b61e-39661eac6141`) — save it in a comment at the top of `trader.py` or in a team chat message in case you need to reference it with Prosperity staff
4. The testing simulation runs 1,000 iterations and returns a results log — use it to refine your strategy
5. The final simulation runs 10,000 iterations and determines your PnL for the round

---

## Quick Reference

| Task | Command |
|---|---|
| Sync environment after `git pull` | `uv sync` |
| Open notebooks | `uv run jupyter notebook` |
| Run a Python script | `uv run python script.py` |
| Add a new package | `uv add <package>` |
| Start a new round branch | `git checkout -b roundN` |

---
