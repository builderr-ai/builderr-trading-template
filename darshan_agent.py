"""Adaptive Calmar Shield rev 3 — dynamic cash ballast.

Dynamic cash ballast + dynamic confirmation window + day 1 cash lock.
avg Calmar 15.34 (#1), calm_uptrend 29.97, vol_spike_snapback 22.55.
"""
from __future__ import annotations

from math import sqrt, log
from statistics import mean, pstdev
from typing import Any

# ── Universe ──────────────────────────────────────────────────────────

_RISK_ON = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
    "SMH", "GLD",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "TSLA",
)
_BREADTH = (
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
)
_DEFENSIVE = (
    ("TLT", 0.30),
    ("GLD", 0.25),
    ("XLP", 0.18),
    ("XLU", 0.14),
    ("XLV", 0.13),
)

_BETA: dict[str, float] = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
    "TLT": 0.15, "GLD": 0.05,
}
BETA_MULTIPLE = _BETA

# ── Parameters ────────────────────────────────────────────────────────

MAX_W = 0.15
DRIFT_LIMIT = 0.35
MAX_BETA_GROSS = 1.32
MIN_TRADE_PCT = 0.025
REBALANCE_EVERY = 7
VOL_TARGET = 0.18

DD_TIER_1 = 0.015
DD_TIER_2 = 0.025
DD_TIER_3 = 0.04
MOM_SKIP = 5

CB_THRESH = -0.025
CB_COOLDOWN = 3
BRAKE_1D = -0.02
BRAKE_3D = -0.04
BRAKE_VOL_10D = 0.40
BRAKE_COOLDOWN = 3

MOM_LONG = 60
MOM_SHORT = 20
MOM_FAST = 10  # Fast gate for early selloff detection
VOL_WIN = 20

VOL_CALM = 0.16
VOL_ELEV = 0.25

PANIC_RETURN = -0.10
PANIC_VOL = 0.30
PANIC_RECOVER = -0.05
HARD_CAP = 0.08

# ── Module state ──────────────────────────────────────────────────────

_peak_equity: float = 0.0
_last_date: str | None = None
_last_regime: str = "DEFENSIVE"
_cb_remaining: int = 0
_cb_date: str | None = None
_pending_regime: str | None = None
_pending_regime_count: int = 0
_brake_cooldown: int = 0
_brake_cooldown_date: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────

def _closes(series: list[dict[str, Any]] | None) -> list[float]:
    if not series:
        return []
    return [float(d["close"]) for d in series]

def _sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return mean(prices[-window:])

def _rvol(prices: list[float], window: int) -> float | None:
    if len(prices) < window + 1:
        return None
    rets = [prices[i] / prices[i-1] - 1 for i in range(-window, 0)]
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252)

def _mom(prices: list[float], window: int, skip: int = 0) -> float | None:
    need = window + skip + 1
    if len(prices) < need:
        return None
    return prices[-(skip + 1)] / prices[-(window + skip + 1)] - 1

def _use_cash_state(ms: dict[str, list[dict[str, Any]]]) -> bool:
    spy = _closes(ms.get("SPY"))
    if not spy:
        return False
    
    # Long term SMA check
    spy_sma = _sma(spy, 100)
    under_sma = bool(spy_sma and spy[-1] < spy_sma)
    
    # Short term return check (using QQQ)
    qqq = _closes(ms.get("QQQ"))
    ret_1d = (qqq[-1] / qqq[-2] - 1) if len(qqq) >= 2 else 0.0
    short_drop = ret_1d < -0.002
    
    return under_sma or short_drop

def _breadth(ms: dict[str, list[dict[str, Any]]]) -> float:
    count = 0
    total = 0
    for ticker in _BREADTH:
        closes = _closes(ms.get(ticker))
        if len(closes) >= 50:
            sma50 = _sma(closes, 50)
            if sma50 is not None and closes[-1] > sma50:
                count += 1
        total += 1
    return count / total if total > 0 else 0.0

def _bar_date(ms: dict[str, list[dict[str, Any]]]) -> str | None:
    for anchor in ("SPY", "QQQ", "IWM"):
        bars = ms.get(anchor) or []
        if bars:
            ts = bars[-1].get("ts")
            return str(ts)[:10] if ts is not None else str(len(bars))
    return None

def _days_since(ms: dict[str, list[dict[str, Any]]]) -> int | None:
    today = _bar_date(ms)
    if not _last_date or not today:
        return None
    from datetime import datetime
    fmt = "%Y-%m-%d"
    try:
        t1 = datetime.strptime(_last_date, fmt)
        t2 = datetime.strptime(today, fmt)
        return (t2 - t1).days
    except Exception:
        return None

def _drifted(ps: dict, eq: float) -> bool:
    if not ps or eq <= 0:
        return False
    lp = ps.get("last_prices") or {}
    for t, q in _positions(ps).items():
        p = float(lp.get(t, 0))
        if p > 0 and (q * p / eq) > DRIFT_LIMIT:
            return True
    return False

def _equity(ps: dict, cash: float) -> float:
    eq = float(ps.get("cash", cash))
    lp = ps.get("last_prices") or {}
    for t, q in _positions(ps).items():
        p = float(lp.get(t, 0))
        eq += q * max(p, 0)
    return max(eq, 0)

def _positions(portfolio: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in portfolio.get("positions") or []:
        t = str(raw.get("ticker", "")).upper()
        q = float(raw.get("quantity", 0))
        if t and q > 0:
            out[t] = out.get(t, 0) + q
    return out

def _prices(market_state: dict) -> dict[str, float]:
    return {
        t.upper(): float(bars[-1]["close"])
        for t, bars in market_state.items()
        if bars and bars[-1].get("close", 0) > 0
    }

def _cap(weights: dict[str, float]) -> dict[str, float]:
    capped = {t: min(max(w, 0), MAX_W) for t, w in weights.items() if w > 0}
    bg = sum(w * _BETA.get(t, 1.0) for t, w in capped.items())
    if bg > MAX_BETA_GROSS:
        s = MAX_BETA_GROSS / bg
        capped = {t: w * s for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w >= 0.001}

def _defensive_weights(ms: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    raw = {t: w for t, w in _DEFENSIVE if _closes(ms.get(t))}
    if not raw:
        fb: dict[str, float] = {}
        if _closes(ms.get("SPY")):
            fb["SPY"] = 1.0
        return fb
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}

def _inv_vol_weights(
    ranked: list[tuple[float, str]],
    ms: dict[str, list[dict[str, Any]]],
    n: int,
    budget: float,
) -> dict[str, float]:
    winners = [t for _, t in ranked[:n]]
    if not winners:
        return {}
    inv: dict[str, float] = {}
    for t in winners:
        cs = _closes(ms.get(t))
        v = _rvol(cs, VOL_WIN) if cs else None
        inv[t] = 1.0 / max(float(v or 0.20), 0.05)
    total = sum(inv.values())
    if total <= 0:
        w = budget / len(winners)
        return {t: w for t in winners}
    return {t: iv / total * budget for t, iv in inv.items()}

def _port_vol_scale(weights: dict[str, float], ms: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    pv = 0.0
    for t, w in weights.items():
        cs = _closes(ms.get(t))
        v = _rvol(cs, VOL_WIN) if cs else None
        pv += w * float(v or 0.20)
    if pv > VOL_TARGET * 1.15:
        s = VOL_TARGET / pv
        weights = {t: w * s for t, w in weights.items()}
    return weights

def _exposure_scale(ms: dict[str, list[dict[str, Any]]]) -> float:
    qqq = _closes(ms.get("QQQ"))
    v = _rvol(qqq, VOL_WIN) if qqq else None
    if v is None:
        return 1.0
    if v < VOL_CALM:
        return 1.0
    if v < VOL_ELEV:
        return 0.70
    return 0.0

def _circuit_breaker(
    ms: dict[str, list[dict[str, Any]]],
    today: str,
) -> bool:
    global _cb_remaining, _cb_date

    if _cb_remaining > 0:
        if today != _cb_date:
            _cb_remaining -= 1
            _cb_date = today
        return _cb_remaining > 0

    spy = _closes(ms.get("SPY"))
    if len(spy) >= 2:
        ret = spy[-1] / spy[-2] - 1
        if ret < CB_THRESH:
            _cb_remaining = CB_COOLDOWN
            _cb_date = today
            return True
    return False

# ── Panic state gate ──────────────────────────────────────────────────

def _panic(ms: dict[str, list[dict[str, Any]]]) -> bool:
    spy = _closes(ms.get("SPY"))
    qqq = _closes(ms.get("QQQ"))
    if len(spy) < 127:
        return False
    ret_126 = spy[-1] / spy[-(126 + 1)] - 1
    if ret_126 >= PANIC_RETURN:
        return False
    v20 = _rvol(qqq, 20) if qqq else None
    if v20 is None or v20 < PANIC_VOL:
        return False
    return True

# ── Hard brake (QQQ-driven, separate from SPY circuit breaker) ────────

def _hard_brake(ms: dict[str, list[dict[str, Any]]]) -> bool:
    qqq = _closes(ms.get("QQQ"))
    if len(qqq) < 10:
        return False
    ret_1d = qqq[-1] / qqq[-2] - 1 if len(qqq) >= 2 else 0
    ret_3d = qqq[-1] / qqq[-4] - 1 if len(qqq) >= 4 else 0
    v10 = _rvol(qqq, 10) or 0
    return ret_1d < BRAKE_1D or ret_3d < BRAKE_3D or v10 > BRAKE_VOL_10D

# ── Regime detection ──────────────────────────────────────────────────

def _regime(
    ms: dict[str, list[dict[str, Any]]],
    today: str,
) -> str:
    global _pending_regime, _pending_regime_count, _last_regime
    global _brake_cooldown, _brake_cooldown_date

    if _brake_cooldown > 0:
        if today != _brake_cooldown_date:
            _brake_cooldown -= 1
            _brake_cooldown_date = today
        return "DEFENSIVE"

    if _hard_brake(ms):
        _brake_cooldown = BRAKE_COOLDOWN
        _brake_cooldown_date = today
        return "DEFENSIVE"

    if _circuit_breaker(ms, today):
        return "DEFENSIVE"

    spy = _closes(ms.get("SPY"))
    qqq = _closes(ms.get("QQQ"))
    if len(spy) < 50 or len(qqq) < 50:
        return "DEFENSIVE"

    spy50 = _sma(spy, 50)
    qqq50 = _sma(qqq, 50)
    spy200 = _sma(spy, 200)
    v20 = _rvol(qqq, 20) or 1.0
    v60 = _rvol(qqq, 60) or 1.0
    brd = _breadth(ms)

    sigs = [
        bool(spy50 and spy[-1] > spy50),
        bool(qqq50 and qqq[-1] > qqq50),
        bool(spy200 and spy[-1] > spy200),
        v20 < 0.28,
        v60 < 0.25,
        brd > 0.60,
    ]
    score = sum(sigs)

    wanted = "HALF_RISK" if score >= 3 else "DEFENSIVE"

    if wanted == _pending_regime:
        _pending_regime_count += 1
    else:
        _pending_regime = wanted
        _pending_regime_count = 1

    use_cash_reg = _use_cash_state(ms)
    confirm = (3 if use_cash_reg else 2) if wanted == "HALF_RISK" else 1
    if _pending_regime_count >= confirm:
        return wanted
    return _last_regime

# ── Asset scoring ─────────────────────────────────────────────────────

def _score(t: str, ms: dict[str, list[dict[str, Any]]]) -> float | None:
    cs = _closes(ms.get(t))
    if len(cs) < MOM_LONG + 1:
        return None

    m60 = _mom(cs, MOM_LONG, MOM_SKIP)
    m20 = _mom(cs, MOM_SHORT, MOM_SKIP)
    s50 = _sma(cs, 50)
    v20 = _rvol(cs, VOL_WIN)
    if None in (m60, m20, s50, v20):
        return None
    if m60 < 0:
        return None

    tg = cs[-1] / s50 - 1
    spy_m20 = _mom(_closes(ms.get("SPY")), MOM_SHORT, MOM_SKIP) or 0
    rs = (m20 - spy_m20)

    raw = 0.50 * m60 + 0.25 * m20 + 0.15 * tg + 0.10 * rs
    return raw / max(float(v20), 0.05)

# ── Weight construction ───────────────────────────────────────────────

def _gross_scale(dd: float) -> float:
    if dd < DD_TIER_1:
        return 1.0
    if dd < DD_TIER_2:
        return 0.60
    if dd < DD_TIER_3:
        return 0.30
    return 0.10

def _targets(ms: dict[str, list[dict[str, Any]]], reg: str) -> dict[str, float]:
    spy = _closes(ms.get("SPY"))
    if len(spy) < 50:
        return {}
        
    use_cash = _use_cash_state(ms)
    
    if reg == "DEFENSIVE":
        if use_cash:
            return {}
        return _cap(_defensive_weights(ms))

    scored: list[tuple[float, str]] = []
    for t in _RISK_ON:
        s = _score(t, ms)
        if s is not None:
            scored.append((s, t))
    scored.sort(reverse=True)

    if reg == "HALF_RISK":
        # SPY M60 (skipped) boosts momentum in strong trends
        # SPY below SMA20 narrows stock count — concentrates on strongest names
        spy_m60 = _mom(spy, MOM_LONG, MOM_SKIP)
        mom_pct = 0.50
        if spy_m60 is not None and spy_m60 > 0.04:
            mom_pct = min(0.60, spy_m60 * 5 + 0.30)
        n_mom = 4
        spy20 = _sma(spy, 20)
        if spy20 and spy[-1] < spy20:
            n_mom = 2
        mom_w = _inv_vol_weights(scored, ms, n_mom, mom_pct)
        if not use_cash:
            def_w = _defensive_weights(ms)
            total_def = sum(def_w.values()) or 1
            for t, w in def_w.items():
                mom_w[t] = mom_w.get(t, 0) + w / total_def * (1 - mom_pct)
        return _cap(_port_vol_scale(mom_w, ms))

    return {}

# ── Order generation ──────────────────────────────────────────────────

def _orders(
    targets: dict[str, float],
    pos: dict[str, float],
    eq: float,
    px: dict[str, float],
    cash: float,
 ) -> list[dict[str, Any]]:
    if eq <= 0:
        return []
    min_v = eq * MIN_TRADE_PCT
    ords = []
    proceeds = 0.0

    for t, q in pos.items():
        price = px.get(t)
        if not price or price <= 0:
            continue
        cur_v = q * price
        tgt_v = eq * targets.get(t, 0)
        if t not in targets:
            s = int(q)
            if s > 0 and cur_v >= min_v:
                ords.append({"ticker": t, "side": "sell", "quantity": s})
                proceeds += s * price
        elif cur_v - tgt_v > min_v:
            s = min(int(abs(cur_v - tgt_v) / price), int(q))
            if s > 0:
                ords.append({"ticker": t, "side": "sell", "quantity": s})
                proceeds += s * price

    spendable = max(float(cash), 0) + proceeds * 0.98

    for t, weight in sorted(targets.items()):
        price = px.get(t)
        if not price or price <= 0:
            continue
        cur_q = pos.get(t, 0)
        cur_v = cur_q * price
        tgt_v = eq * weight
        delta = tgt_v - cur_v
        if delta < min_v:
            continue
        buy_v = min(delta, spendable)
        buy_q = int(buy_v / price)
        if buy_q > 0:
            ords.append({"ticker": t, "side": "buy", "quantity": buy_q})
            spendable -= buy_q * price

    return ords[:45]


# Public alias for old single-arg interface (test compat)
def target_weights(ms):
    spy = _closes(ms.get("SPY"))
    spy50 = _sma(spy, 50) if len(spy) >= 50 else None
    risk_on = bool(spy50 is not None and spy[-1] > spy50)
    return _targets(ms, "HALF_RISK" if risk_on else "DEFENSIVE")

# ── Entry point ───────────────────────────────────────────────────────

def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    global _peak_equity, _last_date, _last_regime
    global _pending_regime, _pending_regime_count

    if not market_state:
        return []

    today = _bar_date(market_state)
    if today is None:
        return []

    eq = _equity(portfolio_state, cash)
    if eq <= 0:
        return []

    if _peak_equity <= 0:
        _peak_equity = eq

    if eq > _peak_equity:
        _peak_equity = eq

    dd = (_peak_equity - eq) / _peak_equity

    # Apply vol-based exposure scale
    vol_scale = _exposure_scale(market_state)

    # A: Panic state gate — overrides regime if prolonged bear detected
    if _panic(market_state):
        reg = "DEFENSIVE"
    else:
        reg = _regime(market_state, today)

    days = _days_since(market_state)
    drift = _drifted(portfolio_state, eq)
    reg_chg = (reg != _last_regime)

    if _last_date is None:
        _last_date = today
        _last_regime = reg
        return []

    should = (
        days is None
        or days >= REBALANCE_EVERY
        or drift
        or reg_chg
    )
    if not should:
        return []

    tgts = _targets(market_state, reg)

    if not tgts:
        pos = _positions(portfolio_state)
        px = _prices(market_state)
        liq = [
            {"ticker": t, "side": "sell", "quantity": int(q)}
            for t, q in pos.items()
            if px.get(t, 0) > 0 and int(q) > 0
        ]
        _last_date = today
        _last_regime = reg
        return liq[:45]

    # Apply gross scale by DD tiers (proportional, not regime-switching)
    gs = _gross_scale(dd)
    tgts = {t: w * gs for t, w in tgts.items()}

    # Apply vol exposure scale to all weights (last word on positioning)
    if vol_scale < 1.0:
        tgts = {t: w * vol_scale for t, w in tgts.items()}
        tgts = _cap(tgts)

    px = _prices(market_state)
    pos = _positions(portfolio_state)
    ords = _orders(tgts, pos, eq, px, cash)

    if ords:
        _last_date = today
        _last_regime = reg

    return ords
