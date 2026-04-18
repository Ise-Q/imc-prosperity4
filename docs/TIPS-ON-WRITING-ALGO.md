
## 0. What this is

A compact, practitioner-focused reference for writing competitive trading algorithms. Built from the official Prosperity docs and battle-tested patterns from top-placing Prosperity 3 teams: **Frankfurt Hedgehogs** (2nd), **CMU Physics** (7th / 1st USA), **Alpha Animals** (9th / 2nd USA), **Linear Utility** (2nd in Prosperity 2), and others.

---

## 1. The Skeleton: Minimum Viable Trader

Every submission is a single-file Python module containing a `Trader` class with a `run()` method. The simulation calls `run()` once per timestep (1,000 iterations in testing; 10,000 in scoring), passing a `TradingState` and expecting back a tuple of `(orders_dict, conversions, traderData)`.

```python
from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle

class Trader:

    def bid(self):
        # Only used in Algo Round 2; ignored in all other rounds
        return 15

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # <-- Replace with your fair value logic

            # TAKE: Lift underpriced asks
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    orders.append(Order(product, best_ask, -best_ask_amount))

            # TAKE: Hit overpriced bids
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        traderData = ""  # Serialize state here
        conversions = 0
        return result, conversions, traderData
```

**What this does**: loops over every product, checks if the best ask is below or the best bid is above a hardcoded "fair value", and sends a single take order. It works, but it's the bare minimum.

**What it's missing** (and what top teams add): dynamic fair value calculation, multi-level sweeping, position-aware sizing, the Clear and Make phases, and state persistence.

---

## 2. Core Data Model — What You Receive

### `TradingState` Properties

|Property|Type|What It Contains|
|---|---|---|
|`traderData`|`str`|Your serialized state from the previous timestep|
|`timestamp`|`int`|Current simulation time (increments by 100)|
|`order_depths`|`Dict[Symbol, OrderDepth]`|Bot quotes you can trade against|
|`own_trades`|`Dict[Symbol, List[Trade]]`|Your fills since last timestep|
|`market_trades`|`Dict[Symbol, List[Trade]]`|Other participants' fills since last timestep|
|`position`|`Dict[Product, int]`|Your signed position per product|
|`observations`|`Observation`|External data (sunlight, tariffs, etc.)|

### `OrderDepth` — Reading the Book

```python
class OrderDepth:
    buy_orders: Dict[int, int] = {}   # {price: +qty}
    sell_orders: Dict[int, int] = {}  # {price: -qty}  ← quantities are NEGATIVE
```

**Critical detail**: `sell_orders` quantities are negative. When you send a buy `Order` to match an ask, use `-best_ask_amount` (negating the negative) to get a positive buy quantity.

**Example**: `buy_orders = {10: 7, 9: 5}` means 7 units bid at 10, 5 at 9. `sell_orders = {12: -3, 11: -2}` means 2 offered at 11, 3 at 12.

### `Order` — What You Send

```python
Order(symbol: str, price: int, quantity: int)
# quantity > 0 → BUY order
# quantity < 0 → SELL order
```

### `Trade` — What You Inspect

The `buyer`/`seller` fields are only non-empty when your algo is one of the counterparties (`"SUBMISSION"`). In Round 5, bot trader IDs become visible, enabling signal-following strategies.

---

## 3. Return Values — What You Send Back

```python
return result, conversions, traderData
```

|Value|Type|Purpose|
|---|---|---|
|`result`|`Dict[str, List[Order]]`|Orders keyed by product symbol|
|`conversions`|`int`|Signed conversion request (relevant for cross-island products)|
|`traderData`|`str`|Serialized state string (max 50,000 chars)|

---

## 4. Position Limits — The #1 Disqualifier

Position limits are absolute (long AND short). If _the sum of all your buy orders for a product_ could push you past the limit assuming full execution, **ALL orders for that product are rejected**.

**Example**: Limit = 10, current position = 3. You can send buy orders totaling at most 7 quantity (10 − 3). If you send 8, every order — buys AND sells — for that product is cancelled.

### Safe Position-Aware Sizing (from Linear Utility / Frankfurt Hedgehogs)

```python
position = state.position.get(product, 0)
limit = POSITION_LIMITS[product]

max_buy_qty  = limit - position      # room to go long
max_sell_qty = limit + position      # room to go short

# Then cap every order against these
buy_qty  = min(desired_buy_qty, max_buy_qty)
sell_qty = min(desired_sell_qty, max_sell_qty)
```

Track cumulative volume across all orders in the same timestep. Frankfurt Hedgehogs and CMU Physics both maintain running `buy_order_volume` / `sell_order_volume` counters throughout the Take → Clear → Make pipeline to never exceed limits.

---

## 5. State Persistence with `traderData`

The simulation runs on AWS Lambda — **class variables and globals reset between invocations**. All state must flow through `traderData`.

### Pattern: jsonpickle Serialization (Used by All Top Teams)

```python
import jsonpickle

class Trader:
    def run(self, state: TradingState):
        # LOAD
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {"price_history": [], "ema": None}

        # UPDATE
        mid = (best_bid + best_ask) / 2
        data["price_history"].append(mid)
        if len(data["price_history"]) > 100:
            data["price_history"] = data["price_history"][-100:]

        # ... trading logic ...

        # SAVE
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData
```

**Pitfall**: The 50,000-character limit. Storing full price histories for every product will exceed this quickly. Use fixed-size `deque`s or rolling aggregates (EMA, running mean/std) instead of raw lists.

**CMU Physics pattern**: Use a flat dict with product-keyed entries for clean multi-product state.

```python
data = {
    "KELP_prices": deque(maxlen=50),
    "KELP_vwap": deque(maxlen=50),
    "SQUID_INK_ema": None,
    "olivia_signal": {}
}
```

---

## 6. The Take → Clear → Make Pipeline

This is the universal execution architecture shared by nearly every top team. It runs sequentially for each product, threading position and volume state through each phase.

### Phase 1: TAKE — Snipe Mispriced Orders

Walk through the book and aggressively fill every level that's better than your fair value minus a configurable `take_width`.

```python
def take_orders(product, order_depth, fair_value, take_width, position, limit):
    orders = []
    buy_volume = 0
    sell_volume = 0

    # Buy: sweep asks below fair - width
    for ask, ask_amt in sorted(order_depth.sell_orders.items()):
        if ask > fair_value - take_width:
            break
        can_buy = limit - position - buy_volume
        qty = min(-ask_amt, can_buy)
        if qty > 0:
            orders.append(Order(product, ask, qty))
            buy_volume += qty

    # Sell: sweep bids above fair + width
    for bid, bid_amt in sorted(order_depth.buy_orders.items(), reverse=True):
        if bid < fair_value + take_width:
            break
        can_sell = limit + position - sell_volume
        qty = min(bid_amt, can_sell)
        if qty > 0:
            orders.append(Order(product, bid, -qty))
            sell_volume += qty

    return orders, buy_volume, sell_volume
```

**Linear Utility insight**: They added an `adverse_volume` filter — if the volume on a level exceeds a threshold, it might be a sophisticated participant (adverse selection risk), so they skip it. This was critical for Starfruit and Kelp.

### Phase 2: CLEAR — Reduce Position at Fair

After taking, you may have accumulated a position. Clear trades at approximately fair value to flatten exposure. This is a "0 EV trade" that frees up capacity for more positive-EV takes later.

```python
def clear_orders(product, order_depth, fair_value, clear_width, position,
                 buy_volume, sell_volume):
    orders = []
    pos_after_take = position + buy_volume - sell_volume

    if pos_after_take > 0:
        # Sell at fair (or fair + clear_width) to reduce long
        clear_price = round(fair_value) + clear_width
        clear_qty = min(pos_after_take, limit + position - sell_volume)
        if clear_qty > 0:
            orders.append(Order(product, clear_price, -clear_qty))
            sell_volume += clear_qty

    elif pos_after_take < 0:
        # Buy at fair (or fair - clear_width) to reduce short
        clear_price = round(fair_value) - clear_width
        clear_qty = min(-pos_after_take, limit - position - buy_volume)
        if clear_qty > 0:
            orders.append(Order(product, clear_price, clear_qty))
            buy_volume += clear_qty

    return orders, buy_volume, sell_volume
```

**Linear Utility found**: Adding 0 EV clearing boosted PnL by ~3% because it kept the algo under position limits, enabling more profitable takes on subsequent timesteps.

### Phase 3: MAKE — Quote Passively for Spread

Post resting limit orders on both sides of fair to earn spread when bots cross. Your quotes expire at the end of the timestep if unfilled.

```python
def make_orders(product, order_depth, fair_value, position,
                buy_volume, sell_volume, default_edge, limit):
    orders = []

    # Bid below fair
    bid_price = round(fair_value) - default_edge
    bid_qty = limit - position - buy_volume
    if bid_qty > 0:
        orders.append(Order(product, bid_price, bid_qty))

    # Ask above fair
    ask_price = round(fair_value) + default_edge
    ask_qty = limit + position - sell_volume
    if ask_qty > 0:
        orders.append(Order(product, ask_price, -ask_qty))

    return orders
```

**Frankfurt Hedgehogs' advanced making**: They check existing book levels before quoting. If a bot's quote is within a `join_edge` of fair, they _join_ at the same price (piggyback on the bot's level). If it's within `disregard_edge`, they skip quoting on that side. Otherwise they quote at `default_edge`.

**Alpha Animals' aggressive execution**: When following Olivia signals, they placed layered limit orders across multiple book levels instead of only hitting the best price. This "breadth-aware" execution captured significantly more volume.

---

## 7. Computing Fair Value — The Core Alpha

The `acceptable_price = 10` in the example is a placeholder. Real edge comes from computing fair value intelligently.

### Approach A: Hardcoded Fair (Stable Products)

For products with a known fixed value (e.g., Rainforest Resin at 10,000), hardcode it.

```python
PARAMS = {
    "RAINFOREST_RESIN": {"fair_value": 10000, "take_width": 1, "default_edge": 4}
}
```

### Approach B: Mid Price / VWAP (Trending Products)

```python
# Simple mid
fair = (best_bid + best_ask) / 2

# VWAP across the full book (from Alpha Animals)
def book_vwap(order_depth):
    total_value, total_qty = 0, 0
    for prc, amt in order_depth.buy_orders.items():
        total_value += prc * amt
        total_qty += amt
    for prc, amt in order_depth.sell_orders.items():
        total_value += prc * abs(amt)
        total_qty += abs(amt)
    return total_value / total_qty if total_qty else None
```

### Approach C: Wall Mid (Frankfurt Hedgehogs' Key Innovation)

Instead of using the raw best bid/ask mid, find the **largest-volume** (wall) quotes on each side. These tend to come from stable market-making bots and produce a less noisy fair value.

```python
def wall_mid(order_depth):
    # Find the price level with the most volume on each side
    wall_bid = max(order_depth.buy_orders.items(), key=lambda x: x[1])[0]
    wall_ask = min(order_depth.sell_orders.items(), key=lambda x: -x[1])[0]
    return (wall_bid + wall_ask) / 2
```

Linear Utility independently discovered this in Prosperity 2 for Starfruit — they called it the "market maker mid" and found the website was actually marking PnL to this value rather than the raw mid.

### Approach D: Rolling EMA (For Mean-Reverting Products)

```python
def update_ema(prev_ema, new_price, alpha=0.1):
    if prev_ema is None:
        return new_price
    return alpha * new_price + (1 - alpha) * prev_ema
```

### Approach E: Synthetic Value (ETF/Basket Arbitrage)

For basket products, compute the theoretical value from constituents:

```python
# Picnic Basket 1 = 6×Croissants + 3×Jams + 1×Djembe
def synthetic_basket1(state):
    c_mid = mid_price(state.order_depths["CROISSANTS"])
    j_mid = mid_price(state.order_depths["JAMS"])
    d_mid = mid_price(state.order_depths["DJEMBES"])
    return 6 * c_mid + 3 * j_mid + 1 * d_mid
```

Trade when `basket_price - synthetic_price` exceeds a threshold. Frankfurt Hedgehogs tracked the z-score of this spread over a rolling window to time entries.

---

## 8. Conversions — Cross-Island Arbitrage

For products tradable on a foreign exchange (Orchids in P2, Macarons in P3), `observations.conversionObservations` provides foreign prices plus fees.

```python
conv = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
# conv.askPrice, conv.bidPrice, conv.transportFees, conv.importTariff, conv.exportTariff

# Cost to import (buy foreign, sell local):
import_cost = conv.askPrice + conv.transportFees + conv.importTariff

# Revenue from export (buy local, sell foreign):
export_revenue = conv.bidPrice - conv.transportFees - conv.exportTariff
```

Return a signed integer as `conversions`: positive to import (convert short → flat), negative to export.

### Frankfurt Hedgehogs' Hidden Taker Bot Exploit

In Prosperity 3 Round 4, a hidden taker bot would fill limit sell orders placed at `int(conv.bidPrice + 0.5)` approximately 60% of the time — even when no visible orderbook participant should have filled them. By quoting sell orders at this price and converting inventory each timestep, teams could extract 130,000–160,000 SeaShells.

**The key insight**: Normalizing local trade prices against the external ask (after costs) revealed anomalous fills above any visible bid. This is the kind of edge that only emerges from careful visualization and data analysis.

---

## 9. Signal Following — Olivia Copy Trading

In Round 5, trader IDs became visible. The bot "Olivia" consistently bought at daily lows and sold at daily highs across Squid Ink, Croissants, and Kelp.

### Detection Pattern (from CMU Physics / Alpha Animals)

```python
def check_olivia_trades(state):
    signals = {}
    for product in ["SQUID_INK", "KELP", "CROISSANTS"]:
        for trade in state.market_trades.get(product, []):
            if abs(trade.timestamp - state.timestamp) <= 100:
                if trade.buyer == "Olivia":
                    signals[product] = "BUY"
                elif trade.seller == "Olivia":
                    signals[product] = "SELL"
        # Also check own_trades (Olivia might have traded against you)
        for trade in state.own_trades.get(product, []):
            if abs(trade.timestamp - state.timestamp) <= 100:
                if trade.buyer == "Olivia":
                    signals[product] = "BUY"
                elif trade.seller == "Olivia":
                    signals[product] = "SELL"
    return signals
```

**CMU Physics YOLO strategy**: When Olivia signaled BUY on Croissants, they went max-long not just Croissants (250) but also both baskets, getting effective long exposure of ~1,050 Croissants. Hedged residual Jams/Djembes exposure with offsetting positions.

**Alpha Animals' execution edge**: Layered limit orders across multiple book levels instead of only lifting the best ask — this "breadth-aware" approach turned a widely-known signal into differentiated alpha.

---

## 10. Practical Gotchas & Pro Tips

### Timing Constraint

Your `run()` must return within **900ms** (average ≤ 100ms). Avoid heavy computation. Precompute where possible; use NumPy for bulk math.

### Supported Libraries

Python 3.12 stdlib plus: `pandas`, `numpy`, `statistics`, `math`, `typing`, `jsonpickle`. No external packages.

### Debugging

`print()` statements appear in log files after backtesting. Use them liberally during development.

```python
print(f"[{state.timestamp}] {product} pos={position} fair={fair:.1f} bid={best_bid} ask={best_ask}")
```

### Order Sorting Matters

`buy_orders` and `sell_orders` are dicts — **not guaranteed sorted**. Always sort explicitly:

```python
asks_sorted = sorted(order_depth.sell_orders.items())           # ascending price
bids_sorted = sorted(order_depth.buy_orders.items(), reverse=True)  # descending price
```

### The Position Trap

A common bug: sending both buys and sells in the same timestep without checking that the **aggregate** of each side independently respects limits. The exchange checks buys and sells separately — if your total buy volume could breach the long limit, ALL orders (including sells) are rejected.

### Backtesting

Use jmerle's open-source backtester (`imc-prosperity-3-backtester`) locally. Top teams forked and extended it with custom PnL marking, conversion support, and visualization. Frankfurt Hedgehogs built a full Dash/Plotly dashboard for orderbook scatter-plot analysis.

### Parameter Optimization

All top teams grid-searched their parameters (`take_width`, `clear_width`, `default_edge`, `adverse_volume`, EMA periods, z-score thresholds) against historical data. Re-optimize every round as new data becomes available.

---

## 11. Putting It All Together — Production Template

```python
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List
import jsonpickle
import numpy as np
from collections import deque

LIMITS = {"PRODUCT_A": 50, "PRODUCT_B": 75}

PARAMS = {
    "PRODUCT_A": {
        "fair_value": 10000,   # or None if dynamic
        "take_width": 1,
        "clear_width": 0,
        "default_edge": 2,
        "adverse_volume": 15,
    },
}

class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        # ── Load State ──
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        result = {}
        conversions = 0

        for product in state.order_depths:
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = LIMITS.get(product, 20)
            params = PARAMS.get(product, {})

            # ── Compute Fair Value ──
            if params.get("fair_value"):
                fair = params["fair_value"]
            else:
                fair = self.compute_fair(od, data, product)

            if fair is None:
                result[product] = []
                continue

            # ── Phase 1: TAKE ──
            take_orders, buy_vol, sell_vol = self.take(
                product, od, fair, params.get("take_width", 1),
                position, limit, params.get("adverse_volume")
            )

            # ── Phase 2: CLEAR ──
            clear_orders, buy_vol, sell_vol = self.clear(
                product, od, fair, params.get("clear_width", 0),
                position, limit, buy_vol, sell_vol
            )

            # ── Phase 3: MAKE ──
            make_orders = self.make(
                product, fair, params.get("default_edge", 2),
                position, limit, buy_vol, sell_vol
            )

            result[product] = take_orders + clear_orders + make_orders

        # ── Save State ──
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def compute_fair(self, od, data, product):
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def take(self, product, od, fair, width, position, limit, adverse_vol=None):
        orders, bv, sv = [], 0, 0
        for ask, amt in sorted(od.sell_orders.items()):
            if ask > fair - width:
                break
            if adverse_vol and -amt >= adverse_vol:
                continue
            qty = min(-amt, limit - position - bv)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                bv += qty
        for bid, amt in sorted(od.buy_orders.items(), reverse=True):
            if bid < fair + width:
                break
            if adverse_vol and amt >= adverse_vol:
                continue
            qty = min(amt, limit + position - sv)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                sv += qty
        return orders, bv, sv

    def clear(self, product, od, fair, width, position, limit, bv, sv):
        orders = []
        pos_after = position + bv - sv
        if pos_after > 0:
            price = round(fair) + width
            qty = min(pos_after, limit + position - sv)
            if qty > 0:
                orders.append(Order(product, price, -qty))
                sv += qty
        elif pos_after < 0:
            price = round(fair) - width
            qty = min(-pos_after, limit - position - bv)
            if qty > 0:
                orders.append(Order(product, price, qty))
                bv += qty
        return orders, bv, sv

    def make(self, product, fair, edge, position, limit, bv, sv):
        orders = []
        bid_qty = limit - position - bv
        if bid_qty > 0:
            orders.append(Order(product, round(fair) - edge, bid_qty))
        ask_qty = limit + position - sv
        if ask_qty > 0:
            orders.append(Order(product, round(fair) + edge, -ask_qty))
        return orders
```

---

## 12. Strategy Playbook by Round Type

|Round Type|Core Strategy|Key Technique|Top Team Reference|
|---|---|---|---|
|**Stable product** (Resin/Amethysts)|Market make around fixed fair|Hardcoded fair + Take/Clear/Make|Linear Utility, Frankfurt Hedgehogs|
|**Trending product** (Kelp/Starfruit)|Market make around dynamic fair|Wall Mid / MM-bot mid + EMA|Linear Utility, Frankfurt Hedgehogs|
|**Mean-reverting product** (Squid Ink)|Spike detection + revert|Z-score over rolling window|CMU Physics, Alpha Animals|
|**ETF/Basket**|Statistical arbitrage|Synthetic value spread + z-score entry|Frankfurt Hedgehogs, CMU Physics|
|**Options/Vouchers**|Volatility trading|IV z-score vs historical mean vol|Alpha Animals, Frankfurt Hedgehogs|
|**Cross-island conversion**|Arbitrage + hidden taker exploit|Sell at `int(foreignBid + 0.5)` + convert|Frankfurt Hedgehogs|
|**Trader ID reveal**|Olivia copy trading|Signal detection → max position YOLO|CMU Physics, Alpha Animals|

---

## 13. Prosperity-Specific Edges

These edges exist because of the competition's unique mechanics and would NOT work in real markets:

1. **Guaranteed sniping**: Execution is instantaneous — no latency. If you see a mispriced order, you WILL get it.
2. **Non-adaptive bots**: Bots follow fixed patterns. Once you identify a behavior, it won't change to exploit you back.
3. **Full book transparency**: You see every outstanding order. In real markets, dark pools and hidden orders exist.
4. **No adverse selection on passive quotes**: Bot market-makers don't "pick you off" when they know something. Spread farming is viable.
5. **Riskless multi-leg arb**: You can simultaneously send orders across all legs of a basket arb in one timestep — no leg risk.

Use these structural advantages aggressively.
