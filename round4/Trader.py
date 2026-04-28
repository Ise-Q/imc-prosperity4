from datamodel import TradingState, Order
from typing import Dict, List
import jsonpickle
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ROUND 4 — Hybrid strategy (best elements of A + B)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Decision basis (forensic analysis of logs 511565 vs 538153):
#
#  Strategy A (511565) earned +73 total:
#    - HYDROGEL: -1761 (adaptive EMA fills from Mark 14/01 — adverse selection)
#    - Options: +1834 (correct, but missed VEV_5200 due to day detection bug)
#    - Drawdown: -4075 (HYDROGEL position squeeze)
#
#  Strategy B (538153) earned +4432 total:
#    - HYDROGEL: +228 (KF fair value, selectively fills Mark 38 only)
#    - VEV_5300 SELL_BIAS: +3555 (200 shorts @57 avg, settled OTM at S=5253)
#    - VEF MM: +646 (spread capture)
#    - VEV_5000/5100 BUY_BIAS: -97 (paid 9+ ticks time value, lost at expiry)
#    - Drawdown: -683
#
#  Hybrid fixes:
#    1. Keep Strategy B's KF HYDROGEL (selective Mark 38 fills)
#    2. Remove BUY_BIAS on VEV_5000/5100 (structural loser — time value drain)
#    3. Keep VEV_5300 SELL_BIAS (verified alpha: OTM at expiry, full premium)
#    4. Add VEV_5400/5500 to SELL_BIAS (verified in Strategy A: +1678 combined)
#    5. Add VEV_5200 to SELL_BIAS (confirmed overpriced when S>5200 at expiry)
#    6. Keep VEF MM (stable +646, low risk)
#    7. Add OPTION_MAX_SHORT caps to prevent over-exposure
#
#  Counterparty profiles (confirmed EDA + log analysis):
#    Mark 01  — buys options aggressively (fills our sells), VEV informed seller
#    Mark 14  — HYDROGEL profitable MM (buys FV-8, sells FV+8) — DANGEROUS fill source
#    Mark 22  — competing option seller
#    Mark 38  — HYDROGEL systematic loser (buys FV+8) — SAFE fill source
#    Mark 49  — VEV noise trader (33% win rate)
#    Mark 55  — VEV liquidity provider
#    Mark 67  — VEV directional buyer

# ── Universe ──────────────────────────────────────────────────────────────────
UNDERLYING   = "VELVETFRUIT_EXTRACT"
HYDROGEL     = "HYDROGEL_PACK"
STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS     = [f"VEV_{K}" for K in STRIKES]
ALL_PRODUCTS = [HYDROGEL, UNDERLYING] + VOUCHERS
POS_LIMITS   = {HYDROGEL: 200, UNDERLYING: 200, **{v: 300 for v in VOUCHERS}}

# ── Parameters ────────────────────────────────────────────────────────────────
HYDROGEL_PARAMS = {"make_margin": 4, "inv_beta": 8, "imb_scale": 4}

VEF_PARAMS = {
    "make_margin": 1,
    "range_std":   15.0,
    "range_k":     0.7,
    "range_lean":  30,       # reduced from 40 — less directional exposure
    "mm_size_frac": 0.20,    # reduced from 0.25 — tighter size
    "clear_pos":   60,       # tighter clearing threshold
}

# Implied vol smile fitted from day 1 data (calibrated for T in calendar years)
SMILE_A = -0.0808
SMILE_B = -0.0298
SMILE_C = 0.2003

# Options: only trade strikes 5200-5500, all as SELL_BIAS
# Removed VEV_5000, VEV_5100 from TRADED_VOUCHERS (BUY_BIAS loses time value)
VOUCHER_PARAMS = {
    "VEV_5200": {"take_margin": 6, "make_margin": 1},
    "VEV_5300": {"take_margin": 6, "make_margin": 1},
    "VEV_5400": {"take_margin": 6, "make_margin": 1},
    "VEV_5500": {"take_margin": 6, "make_margin": 1},
}

# Per-strike max net short (prevents over-exposure on any single strike)
# Based on observed market depth and delta risk
OPTION_MAX_SHORT = {
    "VEV_5200": 75,   # ITM at start, highest delta risk
    "VEV_5300": 200,  # Confirmed profitable in both logs
    "VEV_5400": 150,  # Profitable in log 511565
    "VEV_5500": 200,  # Profitable in log 511565
}

# All traded strikes have SELL_BIAS: structural overpricing at expiry confirmed
# VEV_5000/5100 REMOVED — paying time value that expires worthless is a loser
SELL_BIAS      = {"VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}
BUY_BIAS       = set()   # empty — no buy bias anywhere
TRADED_VOUCHERS = list(VOUCHER_PARAMS.keys())

# T-remaining in calendar-year units (matches SMILE calibration)
INITIAL_T     = 7.0 / 365.0
STEP_SIZE_YR  = 1.0 / (250 * 10_000)
TICKS_PER_DAY = 1_000_000
TTE_START     = 5.0
DAYS_PER_YEAR = 365.0


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    v  = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / v
    return S * _ncdf(d1) - K * _ncdf(d1 - v)


def smile_iv(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return SMILE_C
    m = math.log(K / S) / math.sqrt(T)
    return max(0.01, SMILE_A * m * m + SMILE_B * m + SMILE_C)


# ── Base ProductTrader ────────────────────────────────────────────────────────
# NOTE: buy/sell do NOT cross-increment each other's capacity.
# Each side decrements only its own cap. This avoids the position-limit
# rejection bug from log 509764 (where cross-incrementing caused orders > limit).

class ProductTrader:
    def __init__(self, name: str, state: TradingState, td: dict, ntd: dict, new_day: bool):
        self.name      = name
        self.state     = state
        self.td        = td.get(name, {})
        self.ntd       = ntd
        self.new_day   = new_day
        self.orders: List[Order] = []
        self.pos_limit = POS_LIMITS.get(name, 0)
        self.pos       = state.position.get(name, 0)
        self.buy_cap   = self.pos_limit - self.pos
        self.sell_cap  = self.pos_limit + self.pos
        od             = state.order_depths.get(name)
        self.od        = od
        self.bb        = max(od.buy_orders)  if od and od.buy_orders  else None
        self.ba        = min(od.sell_orders) if od and od.sell_orders else None
        self.mid       = (self.bb + self.ba) / 2.0 if self.bb and self.ba else None

    def buy(self, price: float, qty: float) -> None:
        qty = min(int(abs(qty)), self.buy_cap)
        if qty > 0:
            self.orders.append(Order(self.name, int(price), qty))
            self.buy_cap -= qty

    def sell(self, price: float, qty: float) -> None:
        qty = min(int(abs(qty)), self.sell_cap)
        if qty > 0:
            self.orders.append(Order(self.name, int(price), -qty))
            self.sell_cap -= qty

    def save(self, key: str, val) -> None:
        self.ntd[self.name][key] = val

    def load(self, key: str, default=None):
        return self.td.get(key, default)

    def get_orders(self) -> List[Order]:
        raise NotImplementedError


# ── HYDROGEL Trader ───────────────────────────────────────────────────────────
# Uses a Kalman filter for fair value estimation (adapts to drifting HYDROGEL level).
# Key improvement over Strategy A: KF quotes inside Mark 38's range (FV±4),
# so Mark 38 (systematic loser, buys FV+8/sells FV-8) fills us preferentially.
# Mark 14 (smart MM) also sees our quotes but trades at FV±8 — our tighter quotes
# should get filled by Mark 38 before Mark 14 can react.

class HydrogelTrader(ProductTrader):
    def __init__(self, state: TradingState, td: dict, ntd: dict, new_day: bool):
        super().__init__(HYDROGEL, state, td, ntd, new_day)
        kf_x = self.load("kf_x")
        kf_p = self.load("kf_p", 1.0)
        if new_day or kf_x is None:
            kf_x = self.mid if self.mid else 9990
            kf_p = 1.0
        elif self.mid:
            Q, R = 0.5, 4.0
            kf_p = kf_p + Q
            K    = kf_p / (kf_p + R)
            kf_x = kf_x + K * (self.mid - kf_x)
            kf_p = (1 - K) * kf_p
        self.save("kf_x", kf_x)
        self.save("kf_p", kf_p)
        self.fv = round(kf_x) if kf_x else 9990

        # Order book imbalance skew
        self.imb = 0.0
        if self.od and self.od.buy_orders and self.od.sell_orders:
            bv = abs(list(self.od.buy_orders.values())[0])
            av = abs(list(self.od.sell_orders.values())[0])
            if bv + av > 0:
                self.imb = (bv - av) / (bv + av)

    def get_orders(self) -> List[Order]:
        if not self.od or not self.bb or not self.ba:
            return []
        mm       = HYDROGEL_PARAMS["make_margin"]
        beta     = HYDROGEL_PARAMS["inv_beta"]
        fv       = self.fv
        imb_skew = round(self.imb * HYDROGEL_PARAMS["imb_scale"])
        sig_conf = min(1.0, abs(self.imb) / 0.3)
        inv_skew = -round((self.pos / self.pos_limit) * beta * (1 - sig_conf))

        bid_px = max(self.bb + 1, fv - mm + inv_skew + imb_skew)
        ask_px = min(self.ba - 1, fv + mm + inv_skew + imb_skew)
        bid_px = min(bid_px, self.ba - 1)
        ask_px = max(ask_px, self.bb + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1
        bid_px = max(bid_px, 1)

        if self.imb > -0.2:
            self.buy(bid_px, self.buy_cap)
        if self.imb < 0.2:
            self.sell(ask_px, self.sell_cap)
        return self.orders


# ── VEF Underlying Trader ─────────────────────────────────────────────────────
# EMA-based mean reversion + passive MM.
# Size reduced vs Strategy B to limit directional exposure.
# Clear threshold tightened to prevent large inventory buildup.

class VEFTrader(ProductTrader):
    def __init__(self, state: TradingState, td: dict, ntd: dict, new_day: bool):
        super().__init__(UNDERLYING, state, td, ntd, new_day)
        ema         = self.load("ema")
        ticks_today = self.load("ticks_today", 0)
        if new_day or ema is None:
            ema = self.mid
            ticks_today = 0
        elif self.mid:
            alpha       = max(0.005, 0.1 * (1 - ticks_today / 500))
            ema         = alpha * self.mid + (1 - alpha) * ema
            ticks_today += 1
        self.save("ema", ema)
        self.save("ticks_today", ticks_today)
        self.ema = ema
        self.fv  = round(ema) if ema else None

    def get_orders(self) -> List[Order]:
        if not self.od or self.mid is None or self.fv is None:
            return []
        mm    = VEF_PARAMS["make_margin"]
        clear = VEF_PARAMS["clear_pos"]
        frac  = VEF_PARAMS["mm_size_frac"]
        fv    = self.fv

        # Range-reversion lean when price far from EMA
        range_lean = 0
        if self.ema:
            dev    = self.mid - self.ema
            thresh = VEF_PARAMS["range_k"] * VEF_PARAMS["range_std"]
            if abs(dev) > thresh:
                scale      = min(1.0, (abs(dev) - thresh) / thresh)
                sign       = -1 if dev > 0 else 1
                range_lean = int(round(sign * scale * VEF_PARAMS["range_lean"]))

        # Inventory clearing
        if self.pos > clear and self.ba:
            self.sell(fv, min(self.pos - clear, self.sell_cap // 2))
        elif self.pos < -clear and self.bb:
            self.buy(fv, min(-self.pos - clear, self.buy_cap // 2))

        # Directional lean
        if range_lean > 0 and self.bb and self.ba:
            self.buy(max(self.bb + 1, fv - mm), range_lean)
        elif range_lean < 0 and self.bb and self.ba:
            self.sell(min(self.ba - 1, fv + mm), -range_lean)

        # Passive MM layers
        if self.bb and self.ba:
            for margin in [mm, mm + 2]:
                bid2 = max(self.bb + 1, fv - margin)
                ask2 = min(self.ba - 1, fv + margin)
                if bid2 < ask2:
                    sz = max(1, int(self.pos_limit * frac * 0.5))
                    self.buy(bid2, sz)
                    self.sell(ask2, sz)
        return self.orders


# ── VEV Option Trader ─────────────────────────────────────────────────────────
# All traded strikes (5200-5500) use SELL_BIAS: structural overpricing confirmed
# across both logs and EDA. At expiry, market bids > BS fair → collect premium.
#
# Removed VEV_5000/5100 from BUY_BIAS (log 538153: paid 9.3 ticks time value
# never recovered at expiry → -97 combined loss).
#
# OPTION_MAX_SHORT caps per strike prevent unbounded gamma risk.
# SELL_BIAS mode: sweep bids above BS fair aggressively (fills Mark 01/14 who
# systematically buy calls), then passive ask at FV+make_margin.

class VEVTrader(ProductTrader):
    def __init__(self, name: str, state: TradingState, td: dict, ntd: dict,
                 new_day: bool, S: float, T: float):
        super().__init__(name, state, td, ntd, new_day)
        self.strike      = int(name.split("_")[1])
        self.S           = S
        self.T           = T
        self.soft_cap    = OPTION_MAX_SHORT.get(name, 0)
        vp               = VOUCHER_PARAMS.get(name, {"take_margin": 10, "make_margin": 3})
        self.take_margin = vp["take_margin"]
        self.make_margin = vp["make_margin"]
        self.iv          = smile_iv(S, self.strike, T) if S else SMILE_C
        self.fair        = bs_call(S, self.strike, T, self.iv) if S else None

        # Order book imbalance skew for passive quotes
        self.imb_skew = 0
        if self.od and self.od.buy_orders and self.od.sell_orders:
            bv = abs(list(self.od.buy_orders.values())[0])
            av = abs(list(self.od.sell_orders.values())[0])
            if bv + av > 0:
                self.imb_skew = round((bv - av) / (bv + av) * 1.5)

    def get_orders(self) -> List[Order]:
        if self.S is None or self.fair is None or self.soft_cap == 0:
            return []
        if not self.od:
            return []

        fv        = self.fair
        intrinsic = max(self.S - self.strike, 0.0)
        is_sell   = self.name in SELL_BIAS

        if is_sell and self.od.buy_orders:
            # Aggressively sweep all bids at or above BS fair value
            # net short cap = OPTION_MAX_SHORT[strike]
            for bp in sorted(self.od.buy_orders.keys(), reverse=True):
                if bp < fv:
                    break
                # Remaining short room: soft_cap + current_pos (pos is negative when short)
                short_room = self.soft_cap + self.pos
                qty = min(abs(self.od.buy_orders[bp]), short_room, self.sell_cap)
                if qty > 0:
                    self.sell(bp, qty)
        else:
            # Neutral MM: only take on significant dislocation
            if self.od.sell_orders:
                for sp in sorted(self.od.sell_orders.keys()):
                    if sp > fv - self.take_margin:
                        break
                    qty = min(abs(self.od.sell_orders[sp]),
                              self.soft_cap - self.pos, self.buy_cap)
                    if qty > 0:
                        self.buy(sp, qty)
            if self.od.buy_orders:
                for bp in sorted(self.od.buy_orders.keys(), reverse=True):
                    if bp < fv + self.take_margin:
                        break
                    short_room = self.soft_cap + self.pos
                    qty = min(abs(self.od.buy_orders[bp]), short_room, self.sell_cap)
                    if qty > 0:
                        self.sell(bp, qty)

        # Passive MM — sell bias: sell 3× more than buy
        if self.bb and self.ba:
            bid_px = max(self.bb + 1, round(fv - self.make_margin + self.imb_skew))
            ask_px = min(self.ba - 1, round(fv + self.make_margin + self.imb_skew))
            bid_px = max(bid_px, round(intrinsic) + 1)
            ask_px = min(ask_px, round(self.S))
            if ask_px > bid_px:
                short_room = self.soft_cap + self.pos
                sell_sz = min(self.soft_cap, self.sell_cap, short_room)
                buy_sz  = min(self.soft_cap // 3, self.buy_cap)  # 3× sell bias
                if sell_sz > 0:
                    self.sell(ask_px, sell_sz)
                if buy_sz > 0:
                    self.buy(bid_px, buy_sz)
        return self.orders


# ── Main Trader ───────────────────────────────────────────────────────────────

class Trader:
    def run(self, state: TradingState):
        try:
            td = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        if not isinstance(td, dict):
            td = {}
        for p in ALL_PRODUCTS + ["_meta"]:
            td.setdefault(p, {})
        ntd = {p: {} for p in ALL_PRODUCTS + ["_meta"]}
        ts  = state.timestamp

        meta    = td.get("_meta", {})
        day_off = meta.get("day_offset", 0)
        last_ts = meta.get("last_timestamp")
        new_day = last_ts is not None and ts < last_ts
        if new_day:
            day_off += 1

        # T-remaining in calendar-year units (calibrated to SMILE parameters)
        step_count  = meta.get("step_count", 0)
        if new_day:
            step_count = 0
        T_remaining = max(INITIAL_T - step_count * STEP_SIZE_YR, 1e-6)
        step_count += 1
        ntd["_meta"]["step_count"] = step_count

        res: Dict[str, List[Order]] = {}

        # HYDROGEL: KF market-making (fills Mark 38 selectively)
        hyd = HydrogelTrader(state, td, ntd, new_day)
        res[HYDROGEL] = hyd.get_orders()

        # VEF: EMA mean-reversion MM
        vef = VEFTrader(state, td, ntd, new_day)
        S   = vef.mid
        res[UNDERLYING] = vef.get_orders()

        # VEV options: SELL_BIAS on all 5200-5500
        for strike in STRIKES:
            prod = f"VEV_{strike}"
            if prod in TRADED_VOUCHERS and S is not None:
                trader = VEVTrader(prod, state, td, ntd, new_day, S, T_remaining)
                res[prod] = trader.get_orders()
            else:
                res[prod] = []

        ntd["_meta"]["day_offset"]     = day_off
        ntd["_meta"]["last_timestamp"] = ts
        return res, 0, jsonpickle.encode(ntd)
