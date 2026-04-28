from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import jsonpickle
import math

# ── Universe ──────────────────────────────────────────────────────────────────
UNDERLYING   = "VELVETFRUIT_EXTRACT"
HYDROGEL     = "HYDROGEL_PACK"
STRIKES      = [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
VOUCHERS     = [f"VEV_{K}" for K in STRIKES]
ALL_PRODUCTS = [HYDROGEL, UNDERLYING] + VOUCHERS
POS_LIMITS   = {HYDROGEL:200, UNDERLYING:200, **{v:300 for v in VOUCHERS}}

# ── Parameters ────────────────────────────────────────────────────────────────
HYDROGEL_PARAMS = {"make_margin":4, "inv_beta":8, "imb_scale":4}
VEF_PARAMS = {
    "make_margin":1, "range_std":15.0, "range_k":0.7,
    "range_lean":40, "mm_size_frac":0.25, "clear_pos":80,
}

# Round 4 smile: fitted from day 1 data
SMILE_A = -0.0808; SMILE_B = -0.0298; SMILE_C = 0.2003
# Back to friend5 original half-lives — faster adaptation hurt in round 3
VOUCHER_PARAMS  = {
    "VEV_5000": {"take_margin":6,  "make_margin":1},
    "VEV_5100": {"take_margin":6,  "make_margin":1},
    "VEV_5200": {"take_margin":6,  "make_margin":1},
    "VEV_5300": {"take_margin":6,  "make_margin":1},  # +1.79 edge, tight quotes
    "VEV_5400": {"take_margin":6,  "make_margin":1},
}
SOFT_CAPS       = {"VEV_5000":200,"VEV_5100":200,"VEV_5200":200,"VEV_5300":200,"VEV_5400":200}
# Structural mispricing (confirmed across all 3 historical days):
# VEV_5300: +1.80 ticks overpriced → SELL bias (lean net short)
# VEV_5100: -1.28 ticks underpriced → BUY bias (lean net long)
# All others: MM only
SELL_BIAS = {"VEV_5300"}   # sell more than buy → net short at expiry
BUY_BIAS  = {"VEV_5100"}   # buy more than sell → net long at expiry
TRADED_VOUCHERS = list(SOFT_CAPS.keys())
STEPS_PER_DAY   = 10_000
INITIAL_T       = 7.0/365.0
STEP_SIZE_YR    = 1.0/(250*STEPS_PER_DAY)

# Round 4: days labeled 1,2,3 — TTE_START same as round 3
TTE_START     = 5.0
TICKS_PER_DAY = 1_000_000
DAYS_PER_YEAR = 365.0

# ── Black-Scholes ─────────────────────────────────────────────────────────────
def _ncdf(x):
    return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    if T<=0 or sigma<=0 or S<=0: return max(S-K, 0.0)
    v  = sigma*math.sqrt(T)
    d1 = (math.log(S/K)+0.5*sigma**2*T)/v
    return S*_ncdf(d1) - K*_ncdf(d1-v)

def bs_delta(S, K, T, sigma):
    if T<=0 or sigma<=0: return 1.0 if S>K else 0.0
    v  = sigma*math.sqrt(T)
    d1 = (math.log(S/K)+0.5*sigma**2*T)/v
    return _ncdf(d1)

def smile_iv(S, K, T):
    if S<=0 or K<=0 or T<=0: return SMILE_C
    m = math.log(K/S)/math.sqrt(T)
    return max(0.01, SMILE_A*m*m + SMILE_B*m + SMILE_C)

# ── Base ProductTrader ────────────────────────────────────────────────────────
class ProductTrader:
    def __init__(self, name, state, td, ntd, new_day):
        self.name      = name
        self.state     = state
        self.td        = td.get(name, {})
        self.ntd       = ntd
        self.new_day   = new_day
        self.orders    = []
        self.pos_limit = POS_LIMITS.get(name, 0)
        self.pos       = state.position.get(name, 0)
        self.buy_cap   = self.pos_limit - self.pos
        self.sell_cap  = self.pos_limit + self.pos
        od             = state.order_depths.get(name)
        self.od        = od
        self.bb        = max(od.buy_orders)  if od and od.buy_orders  else None
        self.ba        = min(od.sell_orders) if od and od.sell_orders else None
        self.mid       = (self.bb+self.ba)/2.0 if self.bb and self.ba else None

    def buy(self, price, qty):
        qty = min(int(abs(qty)), self.buy_cap)
        if qty > 0:
            self.orders.append(Order(self.name, int(price), qty))
            self.buy_cap -= qty

    def sell(self, price, qty):
        qty = min(int(abs(qty)), self.sell_cap)
        if qty > 0:
            self.orders.append(Order(self.name, int(price), -qty))
            self.sell_cap -= qty

    def save(self, key, val): self.ntd[self.name][key] = val
    def load(self, key, default=None): return self.td.get(key, default)
    def get_orders(self): raise NotImplementedError

# ── HYDROGEL Trader ───────────────────────────────────────────────────────────
class HydrogelTrader(ProductTrader):
    def __init__(self, state, td, ntd, new_day):
        super().__init__(HYDROGEL, state, td, ntd, new_day)
        kf_x = self.load("kf_x")
        kf_p = self.load("kf_p", 1.0)
        if new_day or kf_x is None:
            kf_x = self.mid if self.mid else 9990
            kf_p = 1.0
        elif self.mid:
            Q, R = 0.5, 4.0
            kf_p = kf_p + Q
            K    = kf_p/(kf_p+R)
            kf_x = kf_x + K*(self.mid-kf_x)
            kf_p = (1-K)*kf_p
        self.save("kf_x", kf_x)
        self.save("kf_p", kf_p)
        self.fv = round(kf_x) if kf_x else 9990
        self.imb = 0.0
        if self.od and self.od.buy_orders and self.od.sell_orders:
            bv = abs(list(self.od.buy_orders.values())[0])
            av = abs(list(self.od.sell_orders.values())[0])
            if bv+av > 0: self.imb = (bv-av)/(bv+av)

    def get_orders(self):
        if not self.od or not self.bb or not self.ba: return []
        mm   = HYDROGEL_PARAMS["make_margin"]
        beta = HYDROGEL_PARAMS["inv_beta"]
        fv   = self.fv
        imb_skew = round(self.imb * HYDROGEL_PARAMS["imb_scale"])
        sig_conf = min(1.0, abs(self.imb)/0.3)
        inv_skew = -round((self.pos/self.pos_limit)*beta*(1-sig_conf))
        bid_px = max(self.bb+1, fv-mm+inv_skew+imb_skew)
        ask_px = min(self.ba-1, fv+mm+inv_skew+imb_skew)
        bid_px = min(bid_px, self.ba-1)
        ask_px = max(ask_px, self.bb+1)
        if bid_px >= ask_px: bid_px = ask_px-1
        bid_px = max(bid_px, 1)
        if self.imb > -0.2: self.buy(bid_px, self.buy_cap)
        if self.imb < 0.2:  self.sell(ask_px, self.sell_cap)
        return self.orders

# ── VEF Trader ────────────────────────────────────────────────────────────────
class VEFTrader(ProductTrader):
    def __init__(self, state, td, ntd, new_day):
        super().__init__(UNDERLYING, state, td, ntd, new_day)
        ema         = self.load("ema")
        ticks_today = self.load("ticks_today", 0)
        if new_day or ema is None:
            ema = self.mid; ticks_today = 0
        elif self.mid:
            alpha = max(0.005, 0.1*(1-ticks_today/500))
            ema   = alpha*self.mid + (1-alpha)*ema
            ticks_today += 1
        self.save("ema", ema)
        self.save("ticks_today", ticks_today)
        self.ema = ema
        self.fv  = round(ema) if ema else None
        self.imbalance = 0.0
        if self.od and self.od.buy_orders and self.od.sell_orders:
            bv = sum(abs(v) for v in self.od.buy_orders.values())
            av = sum(abs(v) for v in self.od.sell_orders.values())
            if bv+av > 0: self.imbalance = (bv-av)/(bv+av)

    def get_orders(self):
        if not self.od or self.mid is None or self.fv is None: return []
        mm    = VEF_PARAMS["make_margin"]
        clear = VEF_PARAMS["clear_pos"]
        frac  = VEF_PARAMS["mm_size_frac"]
        fv    = self.fv
        range_lean = 0
        if self.ema:
            dev    = self.mid - self.ema
            thresh = VEF_PARAMS["range_k"] * VEF_PARAMS["range_std"]
            if abs(dev) > thresh:
                scale      = min(1.0, (abs(dev)-thresh)/thresh)
                sign       = -1 if dev>0 else 1
                range_lean = int(round(sign*scale*VEF_PARAMS["range_lean"]))
        if self.pos > clear and self.ba:
            self.sell(fv, min(self.pos-clear, self.sell_cap//2))
        elif self.pos < -clear and self.bb:
            self.buy(fv, min(-self.pos-clear, self.buy_cap//2))
        if range_lean > 0 and self.bb and self.ba:
            self.buy(max(self.bb+1, fv-mm), range_lean)
        elif range_lean < 0 and self.bb and self.ba:
            self.sell(min(self.ba-1, fv+mm), -range_lean)
        if self.bb and self.ba:
            for margin in [mm, mm+2]:
                bid2 = max(self.bb+1, fv-margin)
                ask2 = min(self.ba-1, fv+margin)
                if bid2 < ask2:
                    sz = max(1, int(self.pos_limit*frac*0.5))
                    self.buy(bid2, sz)
                    self.sell(ask2, sz)
        return self.orders

# ── VEV Option Trader ─────────────────────────────────────────────────────────
class VEVTrader(ProductTrader):
    def __init__(self, name, state, td, ntd, new_day, S, T):
        super().__init__(name, state, td, ntd, new_day)
        self.strike      = int(name.split("_")[1])
        self.S           = S
        self.T           = T
        self.soft_cap    = SOFT_CAPS.get(name, 0)
        vp               = VOUCHER_PARAMS.get(name, {"take_margin":10,"make_margin":3})
        self.take_margin = vp["take_margin"]
        self.make_margin = vp["make_margin"]
        self.iv          = smile_iv(S, self.strike, T) if S else SMILE_C
        self.model_price = bs_call(S, self.strike, T, self.iv) if S else None

        # Pure BS fair value — no EWMA adaptation
        # EWMA would follow market price and erode our selling edge at expiry
        self.fair = self.model_price

        # Option L1 imbalance skew
        self.imb_skew = 0
        if self.od and self.od.buy_orders and self.od.sell_orders:
            bv = abs(list(self.od.buy_orders.values())[0])
            av = abs(list(self.od.sell_orders.values())[0])
            if bv+av > 0:
                opt_imb = (bv-av)/(bv+av)
                self.imb_skew = round(opt_imb * 1.5)

    def get_orders(self):
        if self.S is None or self.fair is None or self.soft_cap == 0: return []
        if not self.od: return []
        fv        = self.fair
        intrinsic = max(self.S - self.strike, 0.0)

        # Take on large dislocation
        if self.od.sell_orders:
            for sp in sorted(self.od.sell_orders.keys()):
                if sp > fv - self.take_margin: break
                qty = min(abs(self.od.sell_orders[sp]),
                          self.soft_cap - self.pos, self.buy_cap)
                if qty > 0: self.buy(sp, qty)
        if self.od.buy_orders:
            for bp in sorted(self.od.buy_orders.keys(), reverse=True):
                if bp < fv + self.take_margin: break
                qty = min(abs(self.od.buy_orders[bp]),
                          self.soft_cap + self.pos, self.sell_cap)
                if qty > 0: self.sell(bp, qty)

        # Directional bias based on structural BS mispricing:
        # VEV_5300: overpriced +1.80 → lean net short (sell more, buy less)
        # VEV_5100: underpriced -1.28 → lean net long (buy more, sell less)
        # Others: neutral MM
        is_sell_bias = self.name in SELL_BIAS
        is_buy_bias  = self.name in BUY_BIAS

        # Passive MM with imbalance skew
        if self.bb and self.ba:
            bid_px = max(self.bb+1, round(fv - self.make_margin + self.imb_skew))
            ask_px = min(self.ba-1, round(fv + self.make_margin + self.imb_skew))
            bid_px = max(bid_px, round(intrinsic)+1)
            ask_px = min(ask_px, round(self.S))
            if ask_px > bid_px:
                # Sell bias: full sell size, half buy size → net short
                # Buy bias: full buy size, half sell size → net long
                # Neutral: equal sizes
                sell_sz = self.soft_cap
                buy_sz  = self.soft_cap
                if is_sell_bias:
                    buy_sz  = self.soft_cap // 3   # sell 3x more than buy
                elif is_buy_bias:
                    sell_sz = self.soft_cap // 3   # buy 3x more than sell
                self.sell(ask_px, min(sell_sz, self.sell_cap))
                self.buy(bid_px,  min(buy_sz,  self.buy_cap))
        return self.orders

# ── Main Trader ───────────────────────────────────────────────────────────────
class Trader:
    def bid(self): return 0

    def run(self, state: TradingState):
        try:
            td = jsonpickle.decode(state.traderData) if state.traderData else {}
        except: td = {}
        if not isinstance(td, dict): td = {}
        for p in ALL_PRODUCTS+["_meta"]: td.setdefault(p, {})
        ntd = {p:{} for p in ALL_PRODUCTS+["_meta"]}
        ts  = state.timestamp

        meta    = td.get("_meta", {})
        day_off = meta.get("day_offset", 0)
        last_ts = meta.get("last_timestamp")
        new_day = last_ts is not None and ts < last_ts
        if new_day: day_off += 1
        T_tte = max(TTE_START - day_off - ts/TICKS_PER_DAY, 1e-6) / DAYS_PER_YEAR

        # T-bookkeeping
        step_count  = meta.get("step_count", 0)
        if new_day: step_count = 0
        T_remaining = max(INITIAL_T - step_count*STEP_SIZE_YR, 1e-6)
        step_count += 1
        ntd["_meta"]["step_count"] = step_count

        res = {}

        # HYDROGEL
        hyd = HydrogelTrader(state, td, ntd, new_day)
        res[HYDROGEL] = hyd.get_orders()

        # VEF
        vef_trader = VEFTrader(state, td, ntd, new_day)
        S = vef_trader.mid
        res[UNDERLYING] = vef_trader.get_orders()

        # VEV options
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