"""Forward-only Calmar agent for builderr trading v0.

Design target
-------------
The live board now scores from the first market open after submission, so this
agent is intentionally feed-forward: it uses only the bars supplied in
``market_state`` and keeps all research priors as static ticker lists.

The current market prior is AI infrastructure leadership: semiconductors,
memory, optical/networking, and data-center power equipment. The prior only
decides what names are worth ranking; price action, volatility, drawdown, and a
small walk-forward linear model decide whether and how much to own.

Rules guardrails
----------------
* Long-only, no network, no LLM, no dependencies.
* Per-name target cap is 24.5%, below the 30% concentration rule.
* Beta-adjusted gross is clamped below 1.35x, below the 1.5x leverage cap.
* 2x ETFs are used only as a small overlay in calm, confirmed uptrends.
"""
from __future__ import annotations

from math import sqrt
from statistics import pstdev
from typing import Any


# ---- Universe -----------------------------------------------------------------

# Current-event prior: AI compute, memory, networking/optics, and data-center
# power. All of these are in the frozen builderr universe as of round open except
# a few harmlessly skipped fallbacks.
AI_INFRA = (
    "MU", "AMD", "MRVL", "AVGO", "NVDA", "SMH", "SOXX",
    "LRCX", "AMAT", "KLAC", "ASML", "ARM", "QCOM", "MPWR", "NXPI", "ON", "TXN",
    "COHR", "GLW", "CIEN", "ANET", "VRT", "APH", "DELL",
    "ORCL", "PLTR", "APP", "MSFT", "META", "GOOGL", "AMZN",
)
AI_POWER = (
    "GEV", "VST", "CEG", "ETN", "PWR", "NRG", "SO", "NEE", "WTS", "PH", "TT", "FIX",
)
BROAD_RISK = (
    "QQQ", "SPY", "IWM", "XLK", "XLC", "XLI", "XLF", "XLY", "XLE", "XLV",
)
DEFENSIVE = (
    "XLP", "XLU", "XLV", "GLD", "TLT", "XLE",
)
RISK_CANDIDATES = tuple(dict.fromkeys(AI_INFRA + AI_POWER + BROAD_RISK))
ALL_CANDIDATES = tuple(dict.fromkeys(RISK_CANDIDATES + DEFENSIVE))

THEME_PRIOR = {
    "MU": 0.020, "AMD": 0.018, "MRVL": 0.017, "AVGO": 0.015, "NVDA": 0.014,
    "SMH": 0.014, "SOXX": 0.013, "COHR": 0.013, "GLW": 0.012, "ANET": 0.011,
    "VRT": 0.012, "GEV": 0.011, "ETN": 0.010, "PWR": 0.010, "CEG": 0.009,
    "ORCL": 0.008, "PLTR": 0.008, "APP": 0.008,
}

BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}


# ---- Risk knobs ---------------------------------------------------------------

ANN = sqrt(252.0)
MIN_BARS = 65
MAX_WEIGHT = 0.245
MAX_BETA_GROSS = 1.35
DRIFT_LIMIT = 0.285
MIN_TRADE_PCT = 0.012
REBALANCE_EVERY_BARS = 2
MAX_ORDERS = 45

TARGET_VOL = 0.22
VOL_FULL_MAX = 0.36
VOL_NEUTRAL_MAX = 0.46
HARD_BRAKE_3D = -0.060
HARD_BRAKE_5D = -0.085
HARD_BRAKE_VOL10 = 0.72

ML_FEATURES = 10
ML_HORIZON = 5
ML_TRAIN_DAYS = 126


# ---- Persistent process state -------------------------------------------------

_tick_count = 0
_last_seen_stamp: str | None = None
_last_rebalance_tick = -10**9
_last_regime: str | None = None
_last_targets: dict[str, float] = {}
_peak_equity = 0.0
_stress_cooldown = 0
_closes_cache: dict[tuple[int, int, str, float], list[float]] = {}


# ---- Market helpers -----------------------------------------------------------

def _clean_ticker(ticker: Any) -> str:
    return str(ticker or "").upper()


def _closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    try:
        last = bars[-1]
        key = (id(bars), len(bars), str(last.get("ts", "")), float(last["close"]))
        cached = _closes_cache.get(key)
        if cached is not None:
            return cached
    except (KeyError, TypeError, ValueError):
        key = None
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0.0:
            return []
        out.append(close)
    if key is not None:
        if len(_closes_cache) > 5000:
            _closes_cache.clear()
        _closes_cache[key] = out
    return out


def _sma(values: list[float], n: int, end: int | None = None) -> float | None:
    if end is None:
        end = len(values) - 1
    start = end - n + 1
    if start < 0 or end >= len(values):
        return None
    return sum(values[start:end + 1]) / n


def _ret(values: list[float], n: int, end: int | None = None) -> float | None:
    if end is None:
        end = len(values) - 1
    start = end - n
    if start < 0 or end >= len(values):
        return None
    base = values[start]
    if base <= 0.0:
        return None
    return values[end] / base - 1.0


def _ann_vol(values: list[float], n: int, end: int | None = None) -> float | None:
    if end is None:
        end = len(values) - 1
    start = end - n
    if start < 0 or end >= len(values):
        return None
    rets = []
    for i in range(start + 1, end + 1):
        prev = values[i - 1]
        if prev <= 0.0:
            return None
        rets.append(values[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * ANN


def _drawdown_from_high(values: list[float], n: int, end: int | None = None) -> float:
    if end is None:
        end = len(values) - 1
    start = max(0, end - n + 1)
    if start > end or end >= len(values):
        return 0.0
    high = max(values[start:end + 1])
    return values[end] / high - 1.0 if high > 0.0 else 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _latest_stamp(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    for ticker in ("QQQ", "SPY", "SMH"):
        bars = market_state.get(ticker)
        if bars:
            return str(bars[-1].get("ts", len(bars)))
    for bars in market_state.values():
        if bars:
            return str(bars[-1].get("ts", len(bars)))
    return None


def _price_map(
    market_state: dict[str, list[dict[str, Any]]],
    portfolio_state: dict[str, Any],
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, price in (portfolio_state.get("last_prices", {}) or {}).items():
        try:
            p = float(price)
        except (TypeError, ValueError):
            continue
        if p > 0.0:
            prices[_clean_ticker(ticker)] = p
    for ticker, bars in market_state.items():
        clean = _clean_ticker(ticker)
        if clean in prices:
            continue
        closes = _closes(bars)
        if closes:
            prices[clean] = closes[-1]
    return prices


def _positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        try:
            ticker = _clean_ticker(raw.get("ticker"))
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if not ticker or qty <= 0.0:
            continue
        if ticker not in out:
            out[ticker] = {"quantity": 0.0, "avg_cost": avg_cost}
        prev_qty = out[ticker]["quantity"]
        total_qty = prev_qty + qty
        if total_qty > 0.0:
            out[ticker]["avg_cost"] = (
                (out[ticker]["avg_cost"] * prev_qty + avg_cost * qty) / total_qty
            )
        out[ticker]["quantity"] = total_qty
    return out


def _equity(
    portfolio_state: dict[str, Any],
    cash: float,
    prices: dict[str, float],
) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        try:
            total = float(cash)
        except (TypeError, ValueError):
            total = 0.0
    for ticker, pos in _positions(portfolio_state).items():
        price = prices.get(ticker, pos["avg_cost"])
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 1e-9)


# ---- Walk-forward linear model ------------------------------------------------

def _feature_vector(
    closes: list[float],
    bench: list[float],
    idx: int,
    bench_idx: int,
) -> list[float] | None:
    r5 = _ret(closes, 5, idx)
    r10 = _ret(closes, 10, idx)
    r21 = _ret(closes, 21, idx)
    r63 = _ret(closes, 63, idx)
    b21 = _ret(bench, 21, bench_idx)
    sma20 = _sma(closes, 20, idx)
    sma50 = _sma(closes, 50, idx)
    vol20 = _ann_vol(closes, 20, idx)
    if None in (r5, r10, r21, r63, b21, sma20, sma50, vol20):
        return None
    assert r5 is not None and r10 is not None and r21 is not None and r63 is not None
    assert b21 is not None and sma20 is not None and sma50 is not None and vol20 is not None
    px = closes[idx]
    gap20 = px / sma20 - 1.0
    gap50 = px / sma50 - 1.0
    dd20 = _drawdown_from_high(closes, 20, idx)
    accel = r10 - (r21 / 2.0)
    rel21 = r21 - b21
    return [
        1.0,
        _clip(r5 / 0.060, -3.0, 3.0),
        _clip(r10 / 0.090, -3.0, 3.0),
        _clip(r21 / 0.140, -3.0, 3.0),
        _clip(r63 / 0.300, -3.0, 3.0),
        _clip(gap20 / 0.080, -3.0, 3.0),
        _clip(gap50 / 0.160, -3.0, 3.0),
        _clip(rel21 / 0.160, -3.0, 3.0),
        _clip((0.32 - vol20) / 0.220, -3.0, 3.0),
        _clip((accel - abs(dd20)) / 0.120, -3.0, 3.0),
    ]


def _train_linear_edge(
    market_state: dict[str, list[dict[str, Any]]],
    candidates: tuple[str, ...],
) -> list[float]:
    """Train a tiny SGD model on past feature -> next-5-day-return samples.

    The model only uses samples whose targets are already inside the provided
    history. It is deliberately low-capacity; the output is a small tie-breaker,
    not the primary trading signal.
    """
    bench = _closes(market_state.get("QQQ")) or _closes(market_state.get("SPY"))
    if len(bench) < MIN_BARS + ML_HORIZON:
        return [0.0] * ML_FEATURES

    weights = [0.0] * ML_FEATURES
    lr = 0.010
    l2 = 0.002
    start_offset = -min(ML_TRAIN_DAYS, len(bench) - ML_HORIZON - 64)
    if start_offset >= -ML_HORIZON - 8:
        return weights

    usable = [t for t in candidates if len(_closes(market_state.get(t))) >= MIN_BARS + ML_HORIZON]
    for offset in range(start_offset, -ML_HORIZON, 2):
        bench_idx = len(bench) + offset
        for ticker in usable:
            values = _closes(market_state.get(ticker))
            idx = len(values) + offset
            target_idx = idx + ML_HORIZON
            if idx < 64 or target_idx >= len(values) or bench_idx < 64:
                continue
            x = _feature_vector(values, bench, idx, bench_idx)
            if x is None:
                continue
            target = _clip((values[target_idx] / values[idx] - 1.0) / 0.080, -2.5, 2.5)
            pred = sum(w * v for w, v in zip(weights, x))
            err = target - pred
            for i, v in enumerate(x):
                weights[i] = weights[i] * (1.0 - lr * l2) + lr * err * v
    return [_clip(w, -0.40, 0.40) for w in weights]


def _ml_prediction(
    values: list[float],
    bench: list[float],
    weights: list[float],
) -> float:
    if len(values) < MIN_BARS or len(bench) < MIN_BARS:
        return 0.0
    x = _feature_vector(values, bench, len(values) - 1, len(bench) - 1)
    if x is None:
        return 0.0
    return _clip(sum(w * v for w, v in zip(weights, x)), -2.0, 2.0)


# ---- Regime, ranking, and weights -------------------------------------------

def _risk_regime(market_state: dict[str, list[dict[str, Any]]]) -> str:
    qqq = _closes(market_state.get("QQQ"))
    spy = _closes(market_state.get("SPY"))
    smh = _closes(market_state.get("SMH"))
    if len(qqq) < 55 or len(spy) < 55:
        return "hard"

    q_r3 = _ret(qqq, 3) or 0.0
    q_r5 = _ret(qqq, 5) or 0.0
    q_r10 = _ret(qqq, 10) or 0.0
    q_vol10 = _ann_vol(qqq, 10) or 0.20
    q_vol20 = _ann_vol(qqq, 20) or 0.20
    q_sma20 = _sma(qqq, 20)
    q_sma50 = _sma(qqq, 50)
    s_sma50 = _sma(spy, 50)
    smh_sma50 = _sma(smh, 50) if len(smh) >= 50 else None

    if q_r3 < HARD_BRAKE_3D or q_r5 < HARD_BRAKE_5D or q_vol10 > HARD_BRAKE_VOL10:
        return "hard"
    if q_sma20 is None or q_sma50 is None or s_sma50 is None:
        return "defensive"

    spy_ok = spy[-1] > s_sma50 * 0.995
    qqq_ok = qqq[-1] > q_sma50 * 0.995
    qqq_fast = qqq[-1] > q_sma20 and q_sma20 > q_sma50
    smh_ok = bool(smh and smh_sma50 and smh[-1] > smh_sma50 * 0.990)

    if spy_ok and qqq_ok and qqq_fast and smh_ok and q_vol20 < VOL_FULL_MAX:
        return "full"
    if (qqq_ok or (qqq[-1] > q_sma20 and q_r10 > 0.020)) and q_vol20 < VOL_NEUTRAL_MAX:
        return "neutral"
    if qqq[-1] > q_sma20 and q_r10 > 0.0:
        return "soft"
    return "defensive"


def _score_name(
    ticker: str,
    market_state: dict[str, list[dict[str, Any]]],
    bench: list[float],
    ml_weights: list[float],
    defensive: bool = False,
) -> tuple[float, float] | None:
    values = _closes(market_state.get(ticker))
    if len(values) < MIN_BARS:
        return None
    price = values[-1]
    if price < 5.0:
        return None

    r10 = _ret(values, 10)
    r21 = _ret(values, 21)
    r63 = _ret(values, 63)
    sma20 = _sma(values, 20)
    sma50 = _sma(values, 50)
    vol20 = _ann_vol(values, 20)
    bench21 = _ret(bench, 21) if bench else 0.0
    if None in (r10, r21, r63, sma20, sma50, vol20):
        return None
    assert r10 is not None and r21 is not None and r63 is not None
    assert sma20 is not None and sma50 is not None and vol20 is not None

    trend_gap = price / sma50 - 1.0
    fast_gap = price / sma20 - 1.0
    rel21 = r21 - (bench21 or 0.0)
    dd20 = abs(_drawdown_from_high(values, 20))
    ml_edge = _ml_prediction(values, bench, ml_weights)

    if defensive:
        score = (
            0.36 * r63 + 0.22 * r21 + 0.12 * r10
            + 0.12 * trend_gap + 0.08 * fast_gap
            - 0.10 * vol20 - 0.08 * dd20
        )
        return (score, vol20) if score > -0.015 and price > sma20 * 0.970 else None

    # Momentum is the base signal; the model and theme prior are small nudges.
    score = (
        0.38 * r63 + 0.23 * r21 + 0.17 * r10
        + 0.12 * trend_gap + 0.07 * rel21
        - 0.11 * vol20 - 0.08 * dd20
        + 0.020 * ml_edge
        + THEME_PRIOR.get(ticker, 0.0)
    )

    trend_ok = price > sma20 and (price > sma50 or r10 > 0.035)
    if not trend_ok or score <= 0.0:
        return None
    return score, vol20


def _ranked(
    market_state: dict[str, list[dict[str, Any]]],
    candidates: tuple[str, ...],
    defensive: bool = False,
) -> list[tuple[float, str, float]]:
    bench = _closes(market_state.get("QQQ")) or _closes(market_state.get("SPY"))
    ml_weights = _train_linear_edge(market_state, candidates) if not defensive else [0.0] * ML_FEATURES
    ranked: list[tuple[float, str, float]] = []
    for ticker in candidates:
        if ticker not in market_state:
            continue
        scored = _score_name(ticker, market_state, bench, ml_weights, defensive)
        if scored is None:
            continue
        score, vol = scored
        ranked.append((score, ticker, vol))
    ranked.sort(key=lambda item: (item[0], THEME_PRIOR.get(item[1], 0.0), item[1]), reverse=True)
    return ranked


def _portfolio_drawdown_scale(equity: float) -> float:
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    if _peak_equity <= 0.0:
        return 1.0
    dd = 1.0 - equity / _peak_equity
    if dd >= 0.10:
        return 0.25
    if dd >= 0.07:
        return 0.45
    if dd >= 0.04:
        return 0.72
    return 1.0


def _vol_budget(market_state: dict[str, list[dict[str, Any]]], base_budget: float) -> float:
    qqq = _closes(market_state.get("QQQ"))
    q_vol = _ann_vol(qqq, 20) if qqq else None
    if not q_vol or q_vol <= 0.0:
        return base_budget
    scale = _clip(TARGET_VOL / q_vol, 0.45, 1.05)
    return min(base_budget, base_budget * scale)


def _allocate(
    ranked: list[tuple[float, str, float]],
    budget: float,
    top_n: int,
) -> dict[str, float]:
    selected = ranked[:top_n]
    if not selected or budget <= 0.0:
        return {}

    raw: dict[str, float] = {}
    for score, ticker, vol in selected:
        risk = max(vol, 0.16)
        raw[ticker] = max(score, 0.010) / (risk ** 0.70)

    weights: dict[str, float] = {}
    remaining_budget = budget
    remaining = dict(raw)
    for _ in range(8):
        total_raw = sum(remaining.values())
        if total_raw <= 0.0 or remaining_budget <= 0.0:
            break
        changed = False
        for ticker, raw_weight in list(remaining.items()):
            proposed = remaining_budget * raw_weight / total_raw
            if proposed > MAX_WEIGHT:
                weights[ticker] = MAX_WEIGHT
                remaining_budget -= MAX_WEIGHT
                del remaining[ticker]
                changed = True
        if not changed:
            for ticker, raw_weight in remaining.items():
                weights[ticker] = remaining_budget * raw_weight / total_raw
            break
    return {t: round(w, 6) for t, w in weights.items() if w > 0.002}


def _scale_beta(weights: dict[str, float]) -> dict[str, float]:
    clipped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.001}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in clipped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        clipped = {t: w * scale for t, w in clipped.items()}
    return {t: round(w, 6) for t, w in clipped.items() if w > 0.002}


def _overlay_ok(market_state: dict[str, list[dict[str, Any]]]) -> bool:
    qqq = _closes(market_state.get("QQQ"))
    smh = _closes(market_state.get("SMH"))
    if len(qqq) < 55 or not market_state.get("QLD") or not market_state.get("SSO"):
        return False
    q20 = _sma(qqq, 20)
    q50 = _sma(qqq, 50)
    qv20 = _ann_vol(qqq, 20) or 1.0
    r10 = _ret(qqq, 10) or 0.0
    smh_ok = True
    if len(smh) >= 50:
        smh50 = _sma(smh, 50)
        smh_ok = bool(smh50 and smh[-1] > smh50 and (_ret(smh, 10) or 0.0) > 0.0)
    return bool(q20 and q50 and q20 > q50 and qqq[-1] > q20 and r10 > 0.015 and qv20 < 0.30 and smh_ok)


def target_weights(
    market_state: dict[str, list[dict[str, Any]]],
    total_equity: float = 100_000.0,
) -> dict[str, float]:
    """Compute target portfolio weights from historical bars only."""
    global _last_regime, _stress_cooldown

    regime = _risk_regime(market_state)
    if regime == "hard":
        _stress_cooldown = 2
    elif _stress_cooldown > 0:
        _stress_cooldown -= 1
        if regime == "full":
            regime = "neutral"

    dd_scale = _portfolio_drawdown_scale(total_equity)
    _last_regime = regime

    if regime == "hard":
        ranked_def = _ranked(market_state, DEFENSIVE, defensive=True)
        return _scale_beta(_allocate(ranked_def, 0.12, 2))

    if regime == "defensive":
        ranked_def = _ranked(market_state, DEFENSIVE, defensive=True)
        return _scale_beta(_allocate(ranked_def, 0.34 * dd_scale, 3))

    if regime == "soft":
        ranked = _ranked(market_state, tuple(dict.fromkeys(DEFENSIVE + BROAD_RISK + AI_POWER)), defensive=False)
        return _scale_beta(_allocate(ranked, _vol_budget(market_state, 0.46) * dd_scale, 4))

    ranked = _ranked(market_state, RISK_CANDIDATES, defensive=False)
    if not ranked:
        ranked_def = _ranked(market_state, DEFENSIVE, defensive=True)
        return _scale_beta(_allocate(ranked_def, 0.25 * dd_scale, 3))

    base_budget = 0.98 if regime == "full" else 0.78
    budget = _vol_budget(market_state, base_budget) * dd_scale
    weights: dict[str, float] = {}
    overlay = regime == "full" and dd_scale > 0.99 and _overlay_ok(market_state)
    if overlay:
        overlay_weights = {"QLD": 0.10, "SSO": 0.05}
        weights.update(overlay_weights)
        budget = max(0.0, budget - sum(overlay_weights.values()))

    top_n = 5 if regime == "full" else 4
    weights.update(_allocate(ranked, budget, top_n))
    return _scale_beta(weights)


# ---- Order generation ---------------------------------------------------------

def _needs_rebalance(
    positions: dict[str, dict[str, float]],
    prices: dict[str, float],
    targets: dict[str, float],
    equity: float,
    regime: str,
    regime_changed: bool,
) -> bool:
    if not positions and targets:
        return True
    if regime_changed or regime in {"hard", "defensive"}:
        return True
    if _tick_count - _last_rebalance_tick >= REBALANCE_EVERY_BARS:
        return True
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if not price or equity <= 0.0:
            continue
        weight = pos["quantity"] * price / equity
        if ticker not in targets and weight > 0.004:
            return True
        if weight > DRIFT_LIMIT:
            return True
        if ticker in targets and abs(weight - targets[ticker]) > 0.045:
            return True
    for ticker, target in targets.items():
        price = prices.get(ticker)
        if price and abs(target - (positions.get(ticker, {}).get("quantity", 0.0) * price / equity)) > 0.045:
            return True
    return False


def _orders_to_targets(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if equity <= 0.0:
        return []

    min_trade = max(25.0, equity * MIN_TRADE_PCT)
    orders: list[dict[str, object]] = []
    expected_cash = max(float(cash_available or 0.0), 0.0)

    # Sells first so buys have realistic cash behind them.
    for ticker in sorted(positions):
        pos = positions[ticker]
        price = prices.get(ticker)
        if not price or price <= 0.0:
            continue
        current_value = pos["quantity"] * price
        target_value = equity * targets.get(ticker, 0.0)
        delta = target_value - current_value
        if ticker not in targets:
            if pos["quantity"] > 0.0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": round(pos["quantity"], 4)})
                expected_cash += current_value * 0.998
        elif delta < -min_trade:
            sell_qty = min(pos["quantity"], abs(delta) / price)
            if sell_qty > 0.0001:
                orders.append({"ticker": ticker, "side": "sell", "quantity": round(sell_qty, 4)})
                expected_cash += sell_qty * price * 0.998

    # Buys second, largest target gaps first.
    buy_plan: list[tuple[float, str, float]] = []
    for ticker, weight in targets.items():
        price = prices.get(ticker)
        if not price or price <= 0.0:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = equity * weight
        delta = target_value - current_value
        if delta > min_trade:
            buy_plan.append((delta, ticker, price))
    buy_plan.sort(reverse=True)

    for delta, ticker, price in buy_plan:
        buy_value = min(delta, expected_cash * 0.985)
        qty = buy_value / price
        if qty > 0.0001 and buy_value >= min_trade:
            orders.append({"ticker": ticker, "side": "buy", "quantity": round(qty, 4)})
            expected_cash -= buy_value
        if len(orders) >= MAX_ORDERS:
            break
    return orders[:MAX_ORDERS]


def decide(
    market_state: dict[str, list[dict[str, Any]]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict[str, object]]:
    """Return long-only orders for the next fill."""
    global _tick_count, _last_seen_stamp, _last_rebalance_tick, _last_targets

    if not market_state:
        return []
    stamp = _latest_stamp(market_state)
    if stamp is None:
        return []
    if stamp == _last_seen_stamp:
        return []
    _last_seen_stamp = stamp
    _tick_count += 1

    prices = _price_map(market_state, portfolio_state)
    positions = _positions(portfolio_state)
    total_equity = _equity(portfolio_state, cash, prices)
    regime_before = _last_regime or "unknown"
    targets = target_weights(market_state, total_equity)
    regime_now = _last_regime or regime_before
    regime_changed = regime_before != "unknown" and regime_now != regime_before

    if not targets and not positions:
        return []

    if not _needs_rebalance(positions, prices, targets, total_equity, regime_now, regime_changed):
        return []

    orders = _orders_to_targets(targets, positions, total_equity, prices, float(portfolio_state.get("cash", cash) or 0.0))
    if orders:
        _last_rebalance_tick = _tick_count
        _last_targets = targets
    return orders
