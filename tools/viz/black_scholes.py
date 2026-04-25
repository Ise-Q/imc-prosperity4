"""Black-Scholes pricing + IV inversion ported from
`round3/strats/trader2.py` so the viz can compute model fair values for the
VEV_* call vouchers.

Stdlib-only (no scipy) so behaviour matches the trader exactly.
"""
from __future__ import annotations

import math
from typing import Optional

UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_STRIKE: dict[str, int] = {f"VEV_{K}": K for K in STRIKES}

TTE_DAYS_AT_ROUND_START = 5
TICKS_PER_DAY = 1_000_000
TRADING_DAYS_PER_YEAR = 365.0

DEFAULT_IV = 0.23
IV_EMA_ALPHA = 0.1


class BlackScholes:
    """European-call pricing under r = q = 0."""

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, sigma: float):
        v = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
        d2 = d1 - v
        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        d1, d2 = BlackScholes._d1_d2(S, K, T, sigma)
        return S * BlackScholes.norm_cdf(d1) - K * BlackScholes.norm_cdf(d2)

    @staticmethod
    def implied_vol(
        C_obs: float, S: float, K: float, T: float,
        lo: float = 1e-4, hi: float = 5.0,
        tol: float = 1e-5, maxit: int = 60,
    ) -> Optional[float]:
        if T <= 0 or S <= 0 or K <= 0:
            return None
        intrinsic = max(S - K, 0.0)
        if C_obs < intrinsic - 1e-6 or C_obs > S + 1e-6:
            return None
        if C_obs <= intrinsic + 1e-6:
            return None
        f_lo = BlackScholes.call_price(S, K, T, lo) - C_obs
        f_hi = BlackScholes.call_price(S, K, T, hi) - C_obs
        if f_lo * f_hi > 0:
            return None
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            f_mid = BlackScholes.call_price(S, K, T, mid) - C_obs
            if abs(f_mid) < tol:
                return mid
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)
