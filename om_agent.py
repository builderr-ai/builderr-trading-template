"""Sentinel v3 — all-weather, capped AI/semis sleeve, TREND-GATED drawdown brake.

Contest objective: maximize Calmar = annualized return / worst drawdown.
That makes the *denominator* (drawdown) the whole game, not raw return.
So this bot is built to keep the equity curve smooth and shallow-dipping while
still capturing most of an uptrend. It is deliberately ALL-WEATHER, not
fair-weather: it gives up some upside in roaring bull windows to avoid the deep
holes that wreck Calmar (and that the held-out stress rerun re-tests).

No network, no LLM, no API keys, standard library only. No lookahead: every
signal is computed from the provided trailing daily bars and nothing else.

Four well-documented moves, layered (see ANATOMY.md / AGENT_BRIEF.md):
  1. Hold what's working   -> per-name 50-day trend gate; only own names in uptrends.
  2. Get out when it breaks -> graded market risk-switch on SPY/QQQ trend breadth.
  3. Size by calm, not hope -> inverse-volatility weights + a portfolio vol target.
  4. Never bet the farm     -> hard per-name cap, no leverage, and a portfolio
                               drawdown circuit-breaker that de-grosses as the
                               book falls (a soft CPPI-style floor on drawdown).

Diversifiers (GLD gold, TLT long bonds) sit alongside equities so that when the
growth sleeve fails its trend gate in a crash, the book can still hold low- or
negatively-correlated ballast instead of dumping everything to a 0% return.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# ---------------------------------------------------------------------------
# Universe sleeves (all 1x beta, all liquid, all in universe.json).
# No leveraged ETFs anywhere: leverage enlarges drawdowns, which is poison for
# Calmar. Max theoretical gross here is ~1.0x.
# ---------------------------------------------------------------------------
GROWTH = ("SPY", "QQQ", "SMH", "XLK")        # broad market + tech/semis upside
SEMIS = ("NVDA", "AMD", "MU", "MRVL", "AVGO")  # AI/chip leaders = capped upside sleeve
DEFENSIVE = ("XLP", "XLU", "XLV")            # low-vol staples / utilities / health
DIVERSIFIERS = ("GLD", "TLT")               # gold + long bonds = crisis ballast
CANDIDATES = GROWTH + SEMIS + DEFENSIVE + DIVERSIFIERS
TILT_SLEEVE = set(GROWTH) | set(SEMIS)       # momentum tilt applies to these

# ---------------------------------------------------------------------------
# Parameters. Each is a standard, documented value; none is curve-fit to a
# specific window. (robustness_test.py perturbs every one of these by +/-20%.)
# ---------------------------------------------------------------------------
TREND_SMA = 50          # per-name trend gate and market risk-switch (Faber-style)
SLOW_SMA = 100          # second market-trend check for the breadth score
VOL_DAYS = 20           # realized-vol lookback for inverse-vol sizing
MOM_FAST, MOM_SLOW = 60, 120   # blended momentum for a gentle growth tilt
TARGET_VOL = 0.10       # ~10% annualized portfolio vol target (conservative)
MAX_WEIGHT = 0.22       # hard per-name cap for ETFs, comfortably under the 30% rule
STOCK_MAX_WEIGHT = 0.12 # tighter cap for single stocks (concentration risk)
SEMIS_CAP = 0.40        # aggregate cap on the whole AI/semis sleeve
MAX_GROSS = 0.98        # never fully spent; always keep a little dry powder
REBALANCE_EVERY = 5     # structural rebalance cadence (trading days) ~ weekly
DRIFT_LIMIT = 0.27      # force a rebalance if any name drifts past this
MIN_TRADE_PCT = 0.015   # ignore dust trades below 1.5% of equity

# Drawdown circuit-breaker: as the book falls from its peak, cut exposure. This
# bounds how deep the drawdown can get -> directly defends the Calmar denominator.
DD_SOFT, DD_SOFT_MULT = 0.04, 0.50   # >=4% off peak -> halve exposure (trend-broken only)
DD_HARD, DD_HARD_MULT = 0.07, 0.25   # >=7% off peak -> quarter exposure (trend-broken only)
DD_CATASTROPHE = 0.12                # even in an uptrend, a >=12% hole still cuts risk

# Graded market risk-switch: exposure multiplier by trend-breadth score (0..3).
REGIME_EXPOSURE = {3: 1.00, 2: 0.70, 1: 0.40, 0: 0.15}

# Leverage table only to *defend* the cap; we never buy these, but if a future
# edit adds one, the gross math stays honest.
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0, "FAS": 3.0,
    "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0, "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

# Module state persists within a single live run (resets per regime in admission,
# which the brake handles gracefully by rebuilding the peak from current equity).
_peak_equity: float = 0.0
_last_rebalance_date: str | None = None


# ---------------------------------------------------------------------------
# Small, defensive helpers (never raise; degrade to "no signal").
# ---------------------------------------------------------------------------
def _closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out: list[float] = []
    for b in bars:
        try:
            c = float(b["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if c <= 0:
            return []
        out.append(c)
    return out


def _sma(values: list[float], n: int) -> float | None:
    n = int(n)
    return mean(values[-n:]) if len(values) >= n else None


def _mom(values: list[float], n: int) -> float | None:
    n = int(n)
    if len(values) <= n or values[-(n + 1)] <= 0:
        return None
    return values[-1] / values[-(n + 1)] - 1.0


def _ann_vol(values: list[float], n: int) -> float | None:
    n = int(n)
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets = [window[i] / window[i - 1] - 1.0 for i in range(1, len(window)) if window[i - 1] > 0]
    if len(rets) < 5:
        return None
    v = pstdev(rets) * sqrt(252.0)
    return max(v, 1e-6)


def _positions(portfolio_state: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in portfolio_state.get("positions", []) or []:
        try:
            t = str(raw.get("ticker", "")).upper()
            q = float(raw.get("quantity", 0.0))
        except (TypeError, ValueError):
            continue
        if t and q > 0:
            out[t] = out.get(t, 0.0) + q
    return out


def _equity(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last = portfolio_state.get("last_prices", {}) or {}
    for t, q in _positions(portfolio_state).items():
        try:
            px = float(last.get(t, 0.0))
        except (TypeError, ValueError):
            px = 0.0
        total += q * max(px, 0.0)
    return max(total, 0.0)


def _latest_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    return str(bars[-1].get("ts", len(bars)))[:10]


def _days_between(market_state: dict[str, list[dict[str, Any]]], ref: str | None) -> int | None:
    if ref is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if ref not in dates:
        return None
    return len(dates) - dates.index(ref) - 1


# ---------------------------------------------------------------------------
# Regime + sizing.
# ---------------------------------------------------------------------------
def _breadth_score(market_state: dict[str, list[dict[str, Any]]]) -> int:
    """0..3 count of healthy broad-market trend conditions."""
    spy, qqq = _closes(market_state.get("SPY")), _closes(market_state.get("QQQ"))
    score = 0
    s50 = _sma(spy, TREND_SMA)
    s100 = _sma(spy, SLOW_SMA)
    q50 = _sma(qqq, TREND_SMA)
    if spy and s50 is not None and spy[-1] > s50:
        score += 1
    if spy and s100 is not None and spy[-1] > s100:
        score += 1
    if qqq and q50 is not None and qqq[-1] > q50:
        score += 1
    return score


def _target_weights(market_state: dict[str, list[dict[str, Any]]], equity: float) -> dict[str, float]:
    # 1) Base exposure from market trend breadth (reused by the brake below).
    score = _breadth_score(market_state)
    base = REGIME_EXPOSURE[score]

    # 2) Build the eligible pool: each candidate must pass its OWN 50-day trend
    #    gate (don't hold a falling knife) and have a usable vol estimate.
    pool: dict[str, dict[str, float]] = {}
    for t in CANDIDATES:
        cs = _closes(market_state.get(t))
        sma = _sma(cs, TREND_SMA)
        vol = _ann_vol(cs, VOL_DAYS)
        if not cs or sma is None or vol is None or cs[-1] <= sma:
            continue
        m_f, m_s = _mom(cs, MOM_FAST), _mom(cs, MOM_SLOW)
        mom = (0.5 * (m_f or 0.0)) + (0.5 * (m_s or 0.0))
        pool[t] = {"vol": vol, "mom": mom}
    if not pool:
        return {}

    # 3) Inverse-vol raw weights (calmer names get more), with a gentle positive
    #    momentum tilt on the growth sleeve only. Tilt is capped so it can only
    #    nudge, never dominate -> keeps the book diversified and robust.
    raw: dict[str, float] = {}
    for t, s in pool.items():
        w = 1.0 / s["vol"]
        if t in TILT_SLEEVE:
            w *= 1.0 + max(min(s["mom"], 0.30), 0.0)  # 0..+30% nudge, never negative
        raw[t] = w
    tot = sum(raw.values())
    weights = {t: w / tot for t, w in raw.items()}

    # 4) Portfolio vol target: scale the whole book toward ~TARGET_VOL using a
    #    diversification-haircut estimate of portfolio vol. Calm -> ~1.0; jumpy -> down.
    est_port_vol = sum(weights[t] * pool[t]["vol"] for t in weights) * 0.80
    vol_scalar = TARGET_VOL / est_port_vol if est_port_vol > 1e-6 else 1.0
    vol_scalar = max(0.40, min(vol_scalar, 1.0))

    # 5) Drawdown circuit-breaker: de-gross as the book falls from its peak.
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    dd = 0.0 if _peak_equity <= 0 else 1.0 - equity / _peak_equity
    # Trend-gate the brake: hard tiers only when the market trend is broken
    # (breadth_score <= 1). In a healthy uptrend, treat a dip as a shakeout and
    # hold -- only a catastrophic hole (>=12%) cuts risk. This stops the
    # "sell the dip, miss the snapback" whipsaw that sank the fast-de-risk bots.
    dd_mult = 1.0
    if score <= 1:
        if dd >= DD_HARD:
            dd_mult = DD_HARD_MULT
        elif dd >= DD_SOFT:
            dd_mult = DD_SOFT_MULT
    elif dd >= DD_CATASTROPHE:
        dd_mult = DD_HARD_MULT

    exposure = min(base * vol_scalar * dd_mult, MAX_GROSS)

    # 6) Apply exposure, hard per-name cap, then re-normalize any capped excess.
    scaled = {t: w * exposure for t, w in weights.items()}
    scaled = {t: min(w, STOCK_MAX_WEIGHT if t in SEMIS else MAX_WEIGHT) for t, w in scaled.items()}
    # Aggregate semis cap: if the chip sleeve totals more than SEMIS_CAP, scale it down.
    semis_total = sum(w for t, w in scaled.items() if t in SEMIS)
    if semis_total > SEMIS_CAP:
        k = SEMIS_CAP / semis_total
        scaled = {t: (w * k if t in SEMIS else w) for t, w in scaled.items()}

    # Honor beta-gross cap defensively (all 1x here, so this is a no-op guard).
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in scaled.items())
    if beta_gross > 1.5:
        k = 1.45 / beta_gross
        scaled = {t: w * k for t, w in scaled.items()}

    return {t: round(w, 6) for t, w in scaled.items() if w > 0.005}


# ---------------------------------------------------------------------------
# Order construction: sell-first, then buy with realistic available cash.
# ---------------------------------------------------------------------------
def _orders(targets, positions, equity, prices, cash):
    if equity <= 0:
        return []
    min_trade = equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    proceeds = 0.0

    for t, q in positions.items():
        px = prices.get(t)
        if not px or px <= 0:
            continue
        cur_val = q * px
        tgt_val = equity * targets.get(t, 0.0)
        if t not in targets:
            sell_q = int(q)
            if sell_q > 0 and cur_val >= min_trade:
                orders.append({"ticker": t, "side": "sell", "quantity": sell_q})
                proceeds += sell_q * px
        elif tgt_val - cur_val < -min_trade:
            sell_q = min(int((cur_val - tgt_val) // px), int(q))
            if sell_q > 0:
                orders.append({"ticker": t, "side": "sell", "quantity": sell_q})
                proceeds += sell_q * px

    spendable = max(float(cash), 0.0) + proceeds * 0.98
    for t, w in sorted(targets.items(), key=lambda kv: -kv[1]):
        px = prices.get(t)
        if not px or px <= 0:
            continue
        cur_q = positions.get(t, 0.0)
        delta = equity * w - cur_q * px
        if delta < min_trade:
            continue
        buy_q = int(min(delta, spendable) // px)
        if buy_q > 0:
            orders.append({"ticker": t, "side": "buy", "quantity": buy_q})
            spendable -= buy_q * px

    return orders[:45]


def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    """Return a list of long-only buy/sell orders for today."""
    global _last_rebalance_date
    if not market_state:
        return []
    date = _latest_date(market_state)
    if date is None:
        return []

    equity = _equity(portfolio_state, cash)
    positions = _positions(portfolio_state)
    prices = {t: cs[-1] for t in market_state if (cs := _closes(market_state.get(t)))}

    # Drawdown state is cheap to keep current every day so the brake reacts fast.
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    dd = 0.0 if _peak_equity <= 0 else 1.0 - equity / _peak_equity

    days = _days_between(market_state, _last_rebalance_date)
    drifted = any(
        prices.get(t, 0.0) > 0 and (q * prices[t] / equity) > DRIFT_LIMIT
        for t, q in positions.items()
    ) if equity > 0 else False
    # Rebalance on cadence, on drift, on first call, OR immediately when the
    # drawdown brake is active (so we de-risk the same day, not next week).
    due = (
        _last_rebalance_date is None
        or days is None
        or days >= REBALANCE_EVERY
        or drifted
        or (dd >= DD_SOFT and _breadth_score(market_state) <= 1)
    )
    if not due:
        return []

    targets = _target_weights(market_state, equity)
    if not targets and not positions:
        return []  # nothing to do; stay in cash

    orders = _orders(targets, positions, equity, prices, cash)
    if orders:
        _last_rebalance_date = date
    return orders
