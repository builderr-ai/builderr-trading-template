"""Calmar-First v3 — builderr Round 1 (Jun 2 – Jul 2 2026).



THREE BRAKES (fastest to slowest):
  1. HARD BRAKE — fires on a single-bar shock:
       QQQ drops 2% in one day, OR
       QQQ drops 3.5% over three days, OR
       QQQ 10-day realised vol > 38% annualised
     Action: cut gross to 5% (near-cash), hold only XLP/GLD,
     4-day mandatory cooldown. Re-entry requires 3 clean signals.

  2. PANIC STATE — sustained bear + high vol:
       SPY 6-month return < -10% AND SPY 20-day vol > 28%
     Action: cap gross at 20%, TLT/GLD/XLP only.

  3. SOFT REGIME — trend broken, no panic:
       SPY or QQQ below 50-day SMA
     Action: cap gross at 35%, defensive ETF basket.

DRAWDOWN GOVERNOR (three tiers, fires before market signals):
  -1.0% from peak → scale gross to 60%
  -2.0% from peak → scale gross to 30%
  -3.5% from peak → scale gross to  8%

ASYMMETRIC REGIME PERSISTENCE:
  3 consecutive good ticks  → enter risk-on (slow)
  1 bad tick                → leave risk-on  (fast)

ALPHA IMPROVEMENTS (on top of the risk framework):
  • Volume-confirmed momentum: above-average volume on up-moves boosts rank
  • Momentum acceleration: assets speeding up (m20 > m60) ranked higher
  • Relative strength vs SPY: outperformers rewarded
  • Overbought cap: assets >15% above 50d SMA penalised

No network calls. No LLM. Pure standard library. Long-only.
Per-name cap 10%. Vol target 11%. Brake gross floor 5%.
"""
from __future__ import annotations

from math import sqrt, log
from statistics import pstdev, mean
from typing import Any

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
RISK_ON_ETFS  = ("SPY", "QQQ", "SMH", "XLK", "XLV", "XLY", "XLC", "XLF")
LARGE_CAPS    = ("NVDA", "AMD", "AVGO", "MU", "AAPL", "MSFT",
                 "GOOGL", "META", "AMZN", "PLTR", "TSLA", "MRVL")
RISK_ON       = RISK_ON_ETFS + LARGE_CAPS

# Defensive baskets
HARD_BRAKE_BASKET = ("XLP", "GLD")           # near-cash in a shock
PANIC_BASKET      = ("TLT", "GLD", "XLP")    # sustained bear
SOFT_BASKET       = ("TLT", "GLD", "XLP", "XLU", "XLV")  # trend broken
RISKON_CUSHION    = ("XLP", "TLT")           # always-on 15% in risk-on

_BETA: dict[str, float] = {
    "TLT": 0.15, "GLD": 0.05, "IAU": 0.05,
    "QLD": 2.0, "SSO": 2.0,
}

# ---------------------------------------------------------------------------
# Parameters — tighter dimensions
# ---------------------------------------------------------------------------
NAME_CAP         = 0.10    # max weight per ticker 
GROSS_MAX        = 0.93
REBALANCE_EVERY  = 5       # trading days between calendar rebalances
DEAD_BAND        = 0.035   # skip orders < 3.5% of equity 

VOL_LOOKBACK     = 20
TARGET_PORT_VOL  = 0.11    # annualised vol target (.: 0.13)

MOMENTUM_LOOKBACK = 63     # ~3 months
MOMENTUM_SKIP     = 5      # skip last week (avoids short-term reversal)
TREND_DAYS        = 50
TOP_N             = 6      # top-N risk-on picks

# Hard brake triggers
BRAKE_1D   = -0.020        # QQQ 1-day return
BRAKE_3D   = -0.035        # QQQ 3-day return (tighter: . -4%)
BRAKE_VOL  = 0.38          # QQQ 10-day vol (tighter: . 40%)
BRAKE_GROSS_FLOOR = 0.05   # cut to 5% on brake (.: 15%)
BRAKE_COOLDOWN    = 4      # mandatory defensive days (.: 3)

# Panic state
PANIC_RET  = -0.10
PANIC_VOL  = 0.28          # tighter: . 0.30
PANIC_CAP  = 0.20

# Soft regime gross cap
SOFT_CAP   = 0.35

# Asymmetric persistence
CONFIRM_ENTER = 3          # slow to enter risk-on (.: 2)
CONFIRM_LEAVE = 1          # fast to leave

# DD governor tiers — tighter than . (1.5/2.5/4%)
DD_T1, DD_SCALE1 = 0.010, 0.60
DD_T2, DD_SCALE2 = 0.020, 0.30
DD_T3, DD_SCALE3 = 0.035, 0.08

# Risk-on blend
RISKON_EQUITY_PCT   = 0.85
RISKON_CUSHION_PCT  = 0.15

_ANN = sqrt(252.0)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_tick            = 0
_last_rebalance  = -(10**9)
_brake_cooldown  = 0
_peak_equity     = 0.0
_pending_regime  = None
_pending_count   = 0
_current_regime  = "soft"

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _closes(bars: list | None) -> list[float]:
    if not bars:
        return []
    return [float(b["close"]) for b in bars if b.get("close", 0) > 0]


def _sma(c: list[float], n: int) -> float | None:
    return sum(c[-n:]) / n if len(c) >= n else None


def _ret(c: list[float], days: int, skip: int = 0) -> float | None:
    need = days + skip + 1
    if len(c) < need:
        return None
    end   = c[-(skip + 1)]
    start = c[-(days + skip + 1)]
    return end / start - 1.0 if start > 0 else None


def _ann_vol(c: list[float], n: int) -> float | None:
    if len(c) < n + 1:
        return None
    rets = [c[i] / c[i-1] - 1.0 for i in range(len(c)-n, len(c)) if c[i-1] > 0]
    return pstdev(rets) * _ANN if len(rets) >= 2 else None


def _rvol_log(c: list[float], n: int) -> float | None:
    if len(c) <= n:
        return None
    w = c[-(n+1):]
    rets = [log(w[i]/w[i-1]) for i in range(1, len(w)) if w[i-1] > 0]
    return pstdev(rets) * _ANN if len(rets) >= 5 else None

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _raw_regime(ms: dict) -> str:
    qqq = _closes(ms.get("QQQ"))
    spy = _closes(ms.get("SPY"))
    if len(qqq) < 30 or len(spy) < 60:
        return "soft"

    # HARD BRAKE — single-bar shock detector
    r1  = _ret(qqq, 1)
    r3  = _ret(qqq, 3)
    v10 = _ann_vol(qqq, 10)
    if ((r1 is not None and r1 < BRAKE_1D) or
        (r3 is not None and r3 < BRAKE_3D) or
        (v10 is not None and v10 > BRAKE_VOL)):
        return "hard"

    # PANIC STATE — sustained bear signature
    spy6  = _ret(spy, 126)
    spyv  = _ann_vol(spy, 20)
    if spy6 is not None and spyv is not None and spy6 < PANIC_RET and spyv > PANIC_VOL:
        return "panic"

    # TREND CHECK — above/below key SMAs
    spy50  = _sma(spy, 50)
    qqq50  = _sma(qqq, 50)
    if spy50 is None or qqq50 is None:
        return "soft"

    above_short = spy[-1] > spy50 * 1.005 and qqq[-1] > qqq50 * 1.005
    spy200 = _sma(spy, 200)
    if spy200 is None:
        return "on" if above_short else "soft"

    return "on" if (above_short and spy[-1] > spy200) else "soft"


def _confirmed_regime(raw: str) -> str:
    """Asymmetric persistence: slow to go risk-on, fast to leave."""
    global _pending_regime, _pending_count, _current_regime

    # Hard brake overrides everything immediately
    if raw == "hard":
        _pending_regime = None
        _pending_count  = 0
        _current_regime = "hard"
        return "hard"

    if raw == _current_regime:
        _pending_regime = None
        _pending_count  = 0
        return _current_regime

    confirm = CONFIRM_LEAVE if _current_regime == "on" else CONFIRM_ENTER

    if raw == _pending_regime:
        _pending_count += 1
    else:
        _pending_regime = raw
        _pending_count  = 1

    if _pending_count >= confirm:
        _current_regime = _pending_regime
        _pending_regime = None
        _pending_count  = 0

    return _current_regime

# ---------------------------------------------------------------------------
# Drawdown governor
# ---------------------------------------------------------------------------

def _dd_scale(equity: float) -> float:
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    if _peak_equity <= 0:
        return 1.0
    dd = 1.0 - equity / _peak_equity
    if dd >= DD_T3:
        return DD_SCALE3
    if dd >= DD_T2:
        return DD_SCALE2
    if dd >= DD_T1:
        return DD_SCALE1
    return 1.0

# ---------------------------------------------------------------------------
# Alpha: enhanced momentum scoring
# ---------------------------------------------------------------------------

def _score_asset(
    ticker: str,
    ms: dict,
    spy_closes: list[float],
) -> float | None:
    """Quality-momentum score with volume confirmation and acceleration."""
    c = _closes(ms.get(ticker))
    if len(c) < MOMENTUM_LOOKBACK + MOMENTUM_SKIP + 2:
        return None

    sma50 = _sma(c, TREND_DAYS)
    mom   = _ret(c, MOMENTUM_LOOKBACK, MOMENTUM_SKIP)
    m20   = _ret(c, 20)
    v20   = _rvol_log(c, 20)

    if None in (sma50, mom, m20, v20) or float(mom) < -0.03:
        return None

    # Must be above 50d SMA (trend filter)
    if c[-1] <= float(sma50):
        return None

    # Overbought cap
    gap = min(c[-1] / float(sma50) - 1.0, 0.15)

    # Momentum acceleration (m20 > mom = speeding up)
    accel = float(m20) - float(mom)

    # Relative strength vs SPY
    spy_m20 = _ret(spy_closes, 20) if spy_closes else None
    rs = (float(m20) - spy_m20) if spy_m20 is not None else 0.0

    # Volume confirmation multiplier
    vol_mult = 1.0
    bars = ms.get(ticker) or []
    if len(bars) >= 21:
        try:
            vols  = [max(float(b.get("volume", 0)), 0.0) for b in bars[-21:]]
            avg_v = mean(vols[:-1])
            if avg_v > 0:
                ratio    = vols[-1] / avg_v
                vol_mult = max(0.85, min(1.15, 0.90 + 0.25 * min(ratio, 1.0)))
        except (TypeError, ValueError):
            pass

    score = 0.42 * float(mom) + 0.22 * float(m20) + 0.12 * accel + 0.14 * gap + 0.10 * rs
    return score / max(float(v20), 0.05) * vol_mult

# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _portfolio_vol(weights: dict[str, float], ms: dict) -> float:
    if not weights:
        return TARGET_PORT_VOL
    num = denom = 0.0
    for t, w in weights.items():
        v = _ann_vol(_closes(ms.get(t) or []), VOL_LOOKBACK)
        if v and v > 0:
            num   += w * v
            denom += w
    return num / denom if denom > 0 else TARGET_PORT_VOL


def _inv_vol(names: tuple | list, ms: dict) -> dict[str, float]:
    iv = {}
    for t in names:
        v = _ann_vol(_closes(ms.get(t) or []), VOL_LOOKBACK)
        if v and v > 0 and ms.get(t):
            iv[t] = 1.0 / v
    if not iv:
        return {}
    s = sum(iv.values())
    return {t: w / s for t, w in iv.items()}


def _apply_caps(weights: dict[str, float], gross_cap: float) -> dict[str, float]:
    """Scale to gross_cap, then apply per-name cap with overflow redistribution."""
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    target = min(gross_cap, total)
    scaled = {t: w * target / total for t, w in weights.items()}

    capped   = {}
    overflow = 0.0
    for t, w in scaled.items():
        if w > NAME_CAP:
            overflow += w - NAME_CAP
            capped[t] = NAME_CAP
        else:
            capped[t] = w

    if overflow > 1e-9:
        room = {t: NAME_CAP - w for t, w in capped.items() if w < NAME_CAP}
        rt   = sum(room.values())
        if rt > 0:
            for t in capped:
                if capped[t] < NAME_CAP:
                    capped[t] = min(NAME_CAP, capped[t] + overflow * room[t] / rt)

    return {t: round(w, 6) for t, w in capped.items() if w >= 0.001}

# ---------------------------------------------------------------------------
# Target weight computation
# ---------------------------------------------------------------------------

def _compute_targets(ms: dict, equity: float, regime: str) -> dict[str, float]:
    global _brake_cooldown

    # HARD BRAKE — near-cash
    if regime == "hard":
        _brake_cooldown = BRAKE_COOLDOWN
        raw = _inv_vol(HARD_BRAKE_BASKET, ms)
        return _apply_caps(raw, BRAKE_GROSS_FLOOR)

    # COOLDOWN — forced defensive after hard brake
    if _brake_cooldown > 0:
        _brake_cooldown -= 1
        raw = _inv_vol(PANIC_BASKET, ms)
        return _apply_caps(raw, 0.30)

    scale = _dd_scale(equity)

    # PANIC STATE
    if regime == "panic":
        cap = min(PANIC_CAP, scale * PANIC_CAP)
        raw = _inv_vol(PANIC_BASKET, ms)
        return _apply_caps(raw, cap)

    # SOFT / TREND BROKEN
    if regime == "soft":
        cap = min(SOFT_CAP, scale * SOFT_CAP)
        raw = _inv_vol(SOFT_BASKET, ms)
        return _apply_caps(raw, cap)

    # RISK-ON — best momentum + permanent defensive cushion
    spy_c = _closes(ms.get("SPY"))
    scored = []
    for t in RISK_ON:
        s = _score_asset(t, ms, spy_c)
        if s is not None:
            scored.append((s, t))
    scored.sort(reverse=True)
    winners = [t for _, t in scored[:TOP_N]]

    if not winners:
        # Fallback: soft defensive if no scored assets
        cap = min(SOFT_CAP, scale * SOFT_CAP)
        raw = _inv_vol(SOFT_BASKET, ms)
        return _apply_caps(raw, cap)

    # Equity component (85%)
    eq_weights = _inv_vol(winners, ms)
    raw: dict[str, float] = {t: w * RISKON_EQUITY_PCT for t, w in eq_weights.items()}

    # Cushion component (15% — always-on defensive ballast)
    cushion = _inv_vol(RISKON_CUSHION, ms)
    for t, w in cushion.items():
        raw[t] = raw.get(t, 0.0) + w * RISKON_CUSHION_PCT

    # Vol-targeting: scale gross to hit TARGET_PORT_VOL
    gross_cap = min(GROSS_MAX, scale * GROSS_MAX)
    pv = max(0.05, min(0.50, _portfolio_vol(raw, ms)))
    vol_target_gross = min(gross_cap, sum(raw.values()) * (TARGET_PORT_VOL / pv))

    return _apply_caps(raw, vol_target_gross)

# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------

def _build_orders(
    targets: dict[str, float],
    positions: dict,
    equity: float,
    prices: dict[str, float],
    cash: float,
) -> list[dict]:
    orders = []
    proceeds = 0.0

    # Sells first
    for ticker, pos in positions.items():
        px  = prices.get(ticker, 0.0)
        if px <= 0:
            continue
        qty   = float(pos.get("quantity", 0))
        tgt   = equity * targets.get(ticker, 0.0)
        delta = tgt - qty * px
        if ticker not in targets:
            q = int(qty)
            if q > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": q})
                proceeds += q * px
        elif delta < -(DEAD_BAND * equity):
            q = min(int(abs(delta) / px), int(qty))
            if q > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": q})
                proceeds += q * px

    spendable = max(float(cash), 0.0) + proceeds * 0.98

    # Buys second
    for ticker, weight in sorted(targets.items()):
        px  = prices.get(ticker, 0.0)
        if not prices.get(ticker):
            continue
        cur = float(positions.get(ticker, {}).get("quantity", 0))
        tgt = equity * weight
        delta = tgt - cur * px
        if delta < DEAD_BAND * equity:
            continue
        q = int(min(delta, spendable) / px)
        if q > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": q})
            spendable -= q * px

    return orders[:28]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    global _tick, _last_rebalance

    if not market_state:
        return []

    _tick += 1

    positions  = {p["ticker"]: p for p in (portfolio_state.get("positions") or [])}
    last_px    = portfolio_state.get("last_prices") or {}
    equity     = float(portfolio_state.get("cash", cash) or 0.0)
    for t, pos in positions.items():
        equity += float(pos.get("quantity", 0)) * float(last_px.get(t, pos.get("avg_cost", 0)))
    if equity <= 0:
        return []

    raw    = _raw_regime(market_state)
    regime = _confirmed_regime(raw)

    hard_brake = (regime == "hard" or _brake_cooldown > 0)
    on_cadence = (_tick - _last_rebalance) >= REBALANCE_EVERY

    if not on_cadence and not hard_brake:
        return []

    prices  = {t: float(bars[-1]["close"]) for t, bars in market_state.items()
               if bars and bars[-1].get("close", 0) > 0}
    targets = _compute_targets(market_state, equity, regime)

    if not targets:
        return []

    orders = _build_orders(targets, positions, equity, prices, cash)
    if orders:
        _last_rebalance = _tick
    return orders
