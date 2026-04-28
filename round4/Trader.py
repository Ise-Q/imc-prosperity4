from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ROUND 4 — COUNTERPARTY EDITION  (v2 — fixed)
#  Products: VELVETFRUIT_EXTRACT, HYDROGEL_PACK, VEV_4000 … VEV_6500
#
#  TTE convention (same as Round 3):
#    At the START of Round 4, TTE = 5 days.
#    Each competition day = 1,000,000 ticks.
#        TTE = 5.0 - (day_offset + timestamp / 1_000_000)
#
#  Sigma: 1 "year" = 1 competition day.
#    SIGMA_FALLBACK = 0.01255  (ATM IV calibrated from Round 4 historical data).
#    Per-strike smile correction for liquid strikes only (5000-5500).
#    Deep OTM/ITM use SIGMA_FALLBACK to avoid overvaluing near-zero options.
#
#  Fixes vs v1:
#    [FIX 1] VEV spot DISABLED — forward return signal (+1.8 bps) does not
#            cover the spread when crossing aggressively (~-22K bleed in v1).
#    [FIX 2] Option take threshold uses uncapped self.fv, not capped fv.
#            Capping to 1.0 caused buying VEV_6000/6500 at 4-8x fair value.
#    [FIX 3] "Follow Mark 01" block removed — 50 units OTM calls too large.
#    [FIX 4] Skip all orders when true BS fair < MIN_OPTION_FV (near-zero
#            options — no edge in providing liquidity at essentially zero).
#    [FIX 5] SIGMA_PER_STRIKE restricted to liquid 5000-5500 band only.
#    [NEW]   Subtle sigma skew when Mark 67 active: IV bumped +5% for OTM
#            calls — we demand more premium, a conservative risk adjustment.
#
#  Position limits:
#    HYDROGEL_PACK       -> 200
#    VELVETFRUIT_EXTRACT -> 200
#    VEV_XXXX (each)     -> 300
# ─────────────────────────────────────────────────────────────────────────────

STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_NAMES    = [f"VEV_{k}" for k in STRIKES]
ALL_PRODUCTS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"] + VEV_NAMES

POS_LIMITS = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK"      : 200,
    **{f"VEV_{k}": 300 for k in STRIKES},
}

SIGMA_FALLBACK  = 0.01255
TTE_AT_R4_START = 5.0

# Per-strike smile — ONLY for the liquid 5000-5500 trading range.
# Deep ITM (4000, 4500) and deep OTM (6000, 6500) use SIGMA_FALLBACK
# to avoid overvaluing near-zero options and accumulating bad inventory.
SIGMA_PER_STRIKE = {
    5000: 0.0122,
    5100: 0.0120,
    5200: 0.0117,
    5300: 0.0116,
    5400: 0.0113,
    5500: 0.0120,
}

# Skip all orders for options whose BS fair value is below this threshold.
# Providing two-sided liquidity on near-zero options creates losses when
# the capped fv pushes bid/ask prices above the true fair value. [FIX 4]
MIN_OPTION_FV = 1.0

MARK_67 = "Mark 67"
MARK_49 = "Mark 49"
MARK_22 = "Mark 22"

PARAMS = {
    "VELVETFRUIT_EXTRACT": {
        "ema_alpha"   : 0.05,
        "take_margin" : 2,
        "clear_margin": 3,
        "make_margin" : 3,
    },
    "HYDROGEL_PACK": {
        "static_fv"   : 9995,
        "ema_alpha"   : 0.03,
        "take_margin" : 2,
        "clear_margin": 4,
        "make_margin" : 3,
    },
    **{
        f"VEV_{k}": {
            "strike"      : k,
            "take_margin" : 3,
            "clear_margin": 2,
            "make_margin" : 5,
        }
        for k in STRIKES
    },
}


def default_traderData():
    return {p: {} for p in ALL_PRODUCTS + ["_meta"]}


# ─────────────────────────────────────────────────────────────────────────────
#  BLACK-SCHOLES  (pure math — no scipy)
# ─────────────────────────────────────────────────────────────────────────────
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _ncdf(d1) - K * _ncdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


def implied_vol(mkt: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if mkt <= intrinsic + 0.01 or T <= 0:
        return None
    lo, hi = 1e-4, 3.0
    if bs_call(S, K, T, hi) < mkt:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if bs_call(S, K, T, mid) < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
#  COUNTERPARTY SIGNAL SCANNER
# ─────────────────────────────────────────────────────────────────────────────
def scan_counterparty_signals(state: TradingState) -> dict:
    """
    Scan market_trades for this tick.
    Returns sigma_skew = 1.05 when Mark 67 is buying VEV spot (conservative
    IV bump for OTM calls — we demand more premium to sell into a rising market).
    """
    signals = {
        "mark67_buying" : False,
        "mark49_selling": False,
        "mark22_on_ask" : set(),
        "sigma_skew"    : 1.0,
    }

    for trade in state.market_trades.get("VELVETFRUIT_EXTRACT", []):
        if trade.buyer  == MARK_67: signals["mark67_buying"]  = True
        if trade.seller == MARK_49: signals["mark49_selling"] = True

    for name in VEV_NAMES:
        for trade in state.market_trades.get(name, []):
            if trade.seller == MARK_22:
                signals["mark22_on_ask"].add(name)

    if signals["mark67_buying"]:
        signals["sigma_skew"] = 1.05

    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  BASE PRODUCT TRADER
# ─────────────────────────────────────────────────────────────────────────────
class ProductTrader:
    def __init__(self, name: str, state: TradingState,
                 last_td: dict, new_td: dict):
        self.orders    = []
        self.name      = name
        self.state     = state
        self.timestamp = state.timestamp
        self.new_td    = new_td.setdefault(name, {})
        self.last_td   = last_td.get(name, {})
        self.params    = PARAMS.get(name, {})
        self.pos_limit = POS_LIMITS.get(name, 0)

        self.position      = state.position.get(name, 0)
        self.market_trades = state.market_trades.get(name, [])

        self.bids, self.asks = self._parse_book()
        self.max_buy  = self.pos_limit - self.position
        self.max_sell = self.pos_limit + self.position

    def _parse_book(self):
        bids, asks = {}, {}
        try:
            od   = self.state.order_depths[self.name]
            bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  reverse=True)}
            asks = {p: abs(v) for p, v in sorted(od.sell_orders.items())}
        except Exception:
            pass
        return bids, asks

    def best_bid(self): return next(iter(self.bids), None)
    def best_ask(self): return next(iter(self.asks), None)

    def mid(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    def buy(self, price: float, volume: float):
        vol = min(round(abs(volume)), self.max_buy)
        if vol <= 0: return
        self.max_buy -= vol
        self.orders.append(Order(self.name, round(price), vol))

    def sell(self, price: float, volume: float):
        vol = min(round(abs(volume)), self.max_sell)
        if vol <= 0: return
        self.max_sell -= vol
        self.orders.append(Order(self.name, round(price), -vol))

    def ema(self, key: str, signal: Optional[float], alpha: float) -> Optional[float]:
        prev = self.last_td.get(key)
        if signal is None and prev is None: return None
        value = (signal if prev is None
                 else (prev if signal is None
                       else alpha * signal + (1 - alpha) * prev))
        self.new_td[key] = value
        return value

    def save_timestamp(self):
        self.new_td["last_timestamp"] = self.timestamp

    def result(self): return {self.name: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
#  HYDROGEL TRADER  (static mean-reversion — proven to work)
# ─────────────────────────────────────────────────────────────────────────────
class StaticFVTrader(ProductTrader):
    def __init__(self, name, state, last_td, new_td):
        super().__init__(name, state, last_td, new_td)
        alpha     = self.params["ema_alpha"]
        static_fv = self.params.get("static_fv")
        mid       = self.mid()
        ema_mid   = self.ema("ema_mid", mid, alpha)

        if static_fv is not None and ema_mid is not None:
            self.fv = 0.7 * static_fv + 0.3 * ema_mid
        elif static_fv is not None:
            self.fv = static_fv
        else:
            self.fv = ema_mid

        self.take_w  = self.params["take_margin"]
        self.clear_w = self.params["clear_margin"]
        self.make_w  = self.params["make_margin"]

    def get_orders(self):
        if self.fv is None:
            return self.result()
        fv = self.fv

        for bp, bv in self.bids.items():
            if bp - fv < self.take_w: break
            self.sell(bp, bv)
        for ap, av in self.asks.items():
            if fv - ap < self.take_w: break
            self.buy(ap, av)

        pos_after = self.position + sum(o.quantity for o in self.orders)
        if pos_after > 0:
            self.sell(round(fv) + self.clear_w, pos_after)
        elif pos_after < 0:
            self.buy(round(fv) - self.clear_w, -pos_after)

        fair_ask = round(fv) + self.make_w
        fair_bid = round(fv) - self.make_w
        ba = self.best_ask(); bb = self.best_bid()
        ask_price = (ba - 1) if (ba is not None and ba > fair_ask) else fair_ask
        bid_price = (bb + 1) if (bb is not None and bb < fair_bid) else fair_bid
        self.sell(ask_price, self.max_sell)
        self.buy(bid_price,  self.max_buy)

        return self.result()


# ─────────────────────────────────────────────────────────────────────────────
#  VEV OPTION TRADER  (BS + rolling IV + counterparty skew — fixed)
# ─────────────────────────────────────────────────────────────────────────────
class VEVTrader(ProductTrader):
    def __init__(self, name: str, state: TradingState,
                 last_td: dict, new_td: dict,
                 underlying_mid: Optional[float],
                 day_offset: int,
                 signals: dict):
        super().__init__(name, state, last_td, new_td)

        self.K       = self.params["strike"]
        self.S       = underlying_mid
        self.TTE     = max(TTE_AT_R4_START - (day_offset + self.timestamp / 1_000_000), 0.0)
        self.signals = signals

        self.take_w  = self.params["take_margin"]
        self.clear_w = self.params["clear_margin"]
        self.make_w  = self.params["make_margin"]

        # ── Rolling IV ────────────────────────────────────────────────────────
        mkt_mid = self.mid()
        live_iv = None
        if mkt_mid is not None and self.S is not None and self.TTE > 0:
            live_iv = implied_vol(mkt_mid, self.S, self.K, self.TTE)

        rolling = self.ema("rolling_iv", live_iv, alpha=0.05)

        if rolling is not None and rolling > 0:
            self.sigma = rolling
        else:
            # [FIX 5] Smile fallback only for liquid 5000-5500 range.
            # Deep OTM/ITM use flat ATM sigma to prevent overvaluing them.
            self.sigma = SIGMA_PER_STRIKE.get(self.K, SIGMA_FALLBACK)

        # Counterparty skew: Mark 67 buying -> raise IV for OTM calls.
        # Conservative risk adjustment — not a directional bet.
        if self.K > 5250 and signals["sigma_skew"] != 1.0:
            self.sigma *= signals["sigma_skew"]

        # ── True BS fair value (uncapped) ─────────────────────────────────────
        if self.S is not None and self.TTE > 0:
            self.fv = bs_call(self.S, self.K, self.TTE, self.sigma)
        elif self.S is not None:
            self.fv = max(self.S - self.K, 0.0)
        else:
            self.fv = None

    def get_orders(self):
        # [FIX 4] Gate: skip entirely if option is near-worthless.
        if self.fv is None or self.fv < MIN_OPTION_FV:
            return self.result()

        true_fv = self.fv          # uncapped — used for take thresholds [FIX 2]
        fv      = max(true_fv, 1.0)  # price-level floor for make/clear orders

        if fv < 5:
            take_w, clear_w, make_w = 1, 1, 2
        else:
            take_w  = self.take_w
            clear_w = self.clear_w
            make_w  = self.make_w

        # ── TAKE using uncapped true_fv [FIX 2] ──────────────────────────────
        # Buy only if ask < 80% of TRUE fair (not capped fv).
        # Without this fix: VEV_6000 fv=0.1 → capped=1.0 → we'd buy at 0.8
        # paying 8x fair value. Now we only buy at 0.08 (never fires for ~0 opts).
        for ap, av in self.asks.items():
            if ap > true_fv * 0.80: break
            self.buy(ap, av)
        # Sell only if bid > 120% of TRUE fair.
        for bp, bv in self.bids.items():
            if bp < true_fv * 1.20: break
            self.sell(bp, bv)

        # ── CLEAR ─────────────────────────────────────────────────────────────
        pos_after = self.position + sum(o.quantity for o in self.orders)
        if pos_after > 0:
            self.sell(round(fv) + clear_w, pos_after)
        elif pos_after < 0:
            self.buy(round(fv) - clear_w, -pos_after)

        # ── MAKE ─────────────────────────────────────────────────────────────
        fair_ask  = round(fv) + make_w
        fair_bid  = round(fv) - make_w
        ba = self.best_ask(); bb = self.best_bid()
        ask_price = (ba - 1) if (ba is not None and ba > fair_ask) else fair_ask
        bid_price = (bb + 1) if (bb is not None and bb < fair_bid) else fair_bid

        # Never post a bid at zero or below for options
        if bid_price > 0:
            self.buy(bid_price, self.max_buy)
        self.sell(ask_price, self.max_sell)

        return self.result()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRADER
# ─────────────────────────────────────────────────────────────────────────────
class Trader:
    """
    Each tick:
      1. Load traderData; track day_offset.
      2. Scan market_trades for counterparty signals.
      3. Get VELVETFRUIT_EXTRACT mid for BS pricing.
      4. VELVETFRUIT_EXTRACT spot: DISABLED (no orders, carry state).
      5. HYDROGEL_PACK: StaticFVTrader around mean 9995.
      6. VEV options: VEVTrader (BS + rolling IV + signals) for each strike.
    """

    def _day_offset(self, state: TradingState, last_td: dict) -> int:
        last_ts    = last_td.get("_meta", {}).get("last_timestamp", state.timestamp)
        day_offset = last_td.get("_meta", {}).get("day_offset", 0)
        if state.timestamp < last_ts:
            day_offset += 1
        return day_offset

    def _underlying_mid(self, state: TradingState) -> Optional[float]:
        try:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            return (bb + ba) / 2.0
        except Exception:
            return None

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        last_td = (jsonpickle.decode(state.traderData)
                   if state.traderData else default_traderData())
        new_td  = default_traderData()

        day_offset = self._day_offset(state, last_td)
        new_td["_meta"]["day_offset"]     = day_offset
        new_td["_meta"]["last_timestamp"] = state.timestamp

        tte = max(TTE_AT_R4_START - (day_offset + state.timestamp / 1_000_000), 0.0)
        print(f"[t={state.timestamp}] day_offset={day_offset}  TTE={tte:.4f}")

        signals = scan_counterparty_signals(state)
        print(
            f"  mark67={signals['mark67_buying']}  "
            f"mark49={signals['mark49_selling']}  "
            f"skew={signals['sigma_skew']:.2f}"
        )

        S = self._underlying_mid(state)

        # [FIX 1] VEV spot disabled — carry state, no orders
        new_td["VELVETFRUIT_EXTRACT"] = last_td.get("VELVETFRUIT_EXTRACT", {})

        # HYDROGEL mean-reversion
        hyd = StaticFVTrader("HYDROGEL_PACK", state, last_td, new_td)
        result.update(hyd.get_orders())
        hyd.save_timestamp()

        # VEV options — BS priced, rolling IV, sigma skew from Mark 67 signal
        for k in STRIKES:
            vev = VEVTrader(
                f"VEV_{k}", state, last_td, new_td,
                underlying_mid=S, day_offset=day_offset, signals=signals,
            )
            result.update(vev.get_orders())
            vev.save_timestamp()

        return result, 0, jsonpickle.encode(new_td)