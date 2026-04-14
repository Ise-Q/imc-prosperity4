"""
Local test harness for round1/trader.py.

Usage:
    uv run python round1/test_trader.py

Runs the Trader through several mock iterations covering edge cases:
  - Normal two-sided market (both ASH and IPR)
  - One-sided market (ask only, no bids)
  - Bot orders outside fair value (should trigger market-take)
  - Position-limit pressure (near ±20)
  - Multi-iteration traderData round-trip
"""

import sys
import os

# Allow running from repo root OR from round1/ directory
sys.path.insert(0, os.path.dirname(__file__))

from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from trader import Trader, POSITION_LIMITS

import jsonpickle

ASH = "ASH_COATED_OSMIUM"
IPR = "INTARIAN_PEPPER_ROOT"


def make_od(buys: dict, sells: dict) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders = buys
    # Prosperity convention: sell_orders values are NEGATIVE
    od.sell_orders = {p: -v for p, v in sells.items()}
    return od


def make_state(order_depths, position, trader_data="") -> TradingState:
    listings = {
        ASH: Listing(ASH, ASH, "XIRECS"),
        IPR: Listing(IPR, IPR, "XIRECS"),
    }
    return TradingState(
        traderData=trader_data,
        timestamp=100,
        listings=listings,
        order_depths=order_depths,
        own_trades={ASH: [], IPR: []},
        market_trades={ASH: [], IPR: []},
        position=position,
        observations={},
    )


def check_limits(product, orders, position):
    """Assert that the aggregate order quantity keeps us within position limits."""
    limit = POSITION_LIMITS.get(product, 20)
    net = position + sum(o.quantity for o in orders)
    assert -limit <= net <= limit, (
        f"{product}: position {position} + orders {[o.quantity for o in orders]} "
        f"= {net}, exceeds ±{limit}"
    )


def run_scenario(name, state, expected_keys=None):
    trader = Trader()
    result, conversions, trader_data = trader.run(state)

    print(f"\n── {name} ──")
    print(f"  conversions  : {conversions}")
    print(f"  traderData   : {trader_data[:120]}")
    for product, orders in result.items():
        pos = state.position.get(product, 0)
        print(f"  {product}: {[(o.price, o.quantity) for o in orders]}")
        check_limits(product, orders, pos)

    if expected_keys:
        assert set(result.keys()) == set(expected_keys), (
            f"Expected products {expected_keys}, got {list(result.keys())}"
        )

    assert conversions == 0
    assert isinstance(trader_data, str)
    assert len(trader_data) < 50_000, f"traderData too long: {len(trader_data)}"
    return trader_data


# ── Scenario 1: Normal two-sided market, position 0 ──────────────────────────
state1 = make_state(
    order_depths={
        ASH: make_od(buys={9992: 15, 9990: 10}, sells={10008: 15, 10010: 10}),
        IPR: make_od(buys={11991: 19}, sells={12006: 10, 12009: 19}),
    },
    position={ASH: 0, IPR: 0},
)
td1 = run_scenario("Normal two-sided market (position=0)", state1, expected_keys=[ASH, IPR])

# ── Scenario 2: ASH bot sell below FV → market-take should trigger ───────────
state2 = make_state(
    order_depths={
        ASH: make_od(buys={9985: 5}, sells={9995: 8}),  # ask 9995 < FV 10000
        IPR: make_od(buys={12490: 10}, sells={12510: 10}),
    },
    position={ASH: 0, IPR: 0},
)
td2 = run_scenario("ASH market-take (ask 9995 < FV 10000)", state2)
# Verify at least one buy order at 9995 for ASH
ash_orders = {o.price: o.quantity for o in Trader().run(state2)[0].get(ASH, [])}
assert any(p == 9995 and q > 0 for p, q in ash_orders.items()), \
    f"Expected buy at 9995 in {ash_orders}"
print("  ✓ Market-take at 9995 confirmed")

# ── Scenario 3: One-sided market (ask only, no bids) ─────────────────────────
state3 = make_state(
    order_depths={
        ASH: make_od(buys={}, sells={10013: 30}),
        IPR: make_od(buys={}, sells={12006: 10}),
    },
    position={ASH: 0, IPR: 0},
)
run_scenario("One-sided market (no bids)", state3)

# ── Scenario 4: Near position limit (long 18), buy cap = 2 ───────────────────
state4 = make_state(
    order_depths={
        ASH: make_od(buys={9992: 15}, sells={10008: 15}),
        IPR: make_od(buys={11991: 10}, sells={12010: 10}),
    },
    position={ASH: 18, IPR: 18},
)
run_scenario("Near long limit (position=18)", state4)

# ── Scenario 5: Near short limit (short 18), sell cap = 2 ────────────────────
state5 = make_state(
    order_depths={
        ASH: make_od(buys={9992: 15}, sells={10008: 15}),
        IPR: make_od(buys={11991: 10}, sells={12010: 10}),
    },
    position={ASH: -18, IPR: -18},
)
run_scenario("Near short limit (position=-18)", state5)

# ── Scenario 6: Multi-iteration traderData round-trip (5 iterations) ─────────
print("\n── Multi-iteration round-trip (5 ticks) ──")
trader = Trader()
td = ""
for i in range(5):
    state = make_state(
        order_depths={
            ASH: make_od(buys={9990 + i: 10}, sells={10010 + i: 10}),
            IPR: make_od(buys={12000 + i * 5: 10}, sells={12015 + i * 5: 10}),
        },
        position={ASH: 0, IPR: i * 2},
        trader_data=td,
    )
    result, _, td = trader.run(state)
    decoded = jsonpickle.decode(td)
    ipr_hist = decoded.get("price_history", {}).get(IPR, [])
    print(f"  iter {i+1}: IPR history len={len(ipr_hist)}, traderData len={len(td)}")
    assert len(td) < 50_000

print("\n✓ All scenarios passed — trader.py is ready to submit.")
