"""NIM-Powered Calmar Rotation Hybrid.

Contest objective: maximize 60-day forward Calmar, not raw return.

Two-layer architecture:
  Layer 1 — Deterministic Calmar Rotation Hybrid (risk-off/risk-on toggle,
            sector momentum ranking, position sizing, leverage caps).
  Layer 2 — NVIDIA NIM inference (regime classification overlay).
            If NIM agrees with risk-off, de-risk faster.
            If NIM sees MEAN_REVERT_UP, add opportunistic buys.
            If NIM times out (>4s), fall back to Layer 1 silently.

NIM is optional — the agent works fine without it. NIM enhances timing,
the deterministic layer handles everything else.

Long-only. No short-selling. Beta-adjusted gross <= 1.5x.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
import urllib.error
from math import sqrt
from statistics import mean, pstdev
from typing import Any

# ---------------------------------------------------------------------------
# NIM Configuration
# ---------------------------------------------------------------------------

NIM_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
NIM_BASE_URL = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.environ.get("NVIDIA_NIM_MODEL", "meta/llama-3.1-8b-instruct")
NIM_TIMEOUT = 4.0  # hard 4s ceiling — must leave margin for 5s decide() limit

# ---------------------------------------------------------------------------
# Strategy Constants (Calmar Rotation Hybrid)
# ---------------------------------------------------------------------------

RISK_CANDIDATES = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "SMH",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
)
DEFENSIVE_WEIGHTS = (
    ("XLP", 0.24),
    ("XLU", 0.24),
    ("XLV", 0.20),
    ("XLE", 0.12),
)
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

REBALANCE_EVERY_DAYS = 5
MAX_WEIGHT = 0.24
DRIFT_LIMIT = 0.27
MAX_BETA_GROSS = 1.35
MIN_TRADE_PCT = 0.015
MIN_CONFIDENCE = 0.30

_last_rebalance_bar_date: str | None = None
_last_targets: dict[str, float] = {}

# ---------------------------------------------------------------------------
# NIM Inference
# ---------------------------------------------------------------------------

NIM_SYSTEM_PROMPT = """Classify the market regime and return ONLY a JSON object, no other text.

Output format (nothing else):
{"regime":"TREND_UP","action":"BUY","confidence":0.7}

Regimes: TREND_UP, TREND_DOWN, MEAN_REVERT_UP, MEAN_REVERT_DOWN, CHOP
Actions: BUY, SELL, HOLD (long-only: SELL closes existing longs)

Rules (match FIRST whose conditions are met):
- MEAN_REVERT_UP: z20 < -1.5 AND ret20 between -0.12 and -0.03 AND mom5 > -0.01 → BUY
- MEAN_REVERT_DOWN: z20 > 2.5 AND ret20 > 0.05 → HOLD
- TREND_DOWN: ret20 < -0.05 AND vol20 > 0.025 AND z20 < -1.5 AND dd > 0.10 AND mom5 < -0.03 → SELL or HOLD, NEVER BUY
- TREND_UP: ret20 > 0.03 AND vol20 < 0.02 AND z20 > 1.5 → BUY
- CHOP: abs(ret20) < 0.01 AND abs(z20) < 0.5 → HOLD
- If confidence < 0.4 → HOLD

CRITICAL DISAMBIGUATION — MEAN_REVERT_UP vs TREND_DOWN:
Both show negative z20 and negative ret20. Use these rules in order:

IF z20 < -1.5 AND ret20 is between -0.03 and -0.12:
  - mom5 > -0.01 → This is MEAN_REVERT_UP. Price decline has stabilized. BUY.
  - mom5 < -0.03 AND dd > 0.10 → This is TREND_DOWN. Decline accelerating. SELL/HOLD.
  - If ambiguous: prefer MEAN_REVERT_UP when mom5 > -0.02 and dd < 0.08

A stock that fell 8% over 20 days but is flat/up over the last 5 days
is bouncing from support = MEAN_REVERT_UP, NOT TREND_DOWN.
TREND_DOWN requires CONTINUED deterioration (mom5 negative, dd > 10%).
"""


def _call_nim(prompt: str) -> dict | None:
    """Call NIM with hard timeout. Returns None on any failure."""
    if not NIM_API_KEY:
        return None
    payload = json.dumps({
        "model": NIM_MODEL,
        "messages": [
            {"role": "system", "content": NIM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{NIM_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NIM_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=NIM_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content") or msg.get("reasoning")
            if not content:
                return None
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            return json.loads(content)
    except Exception:
        return None


def _deterministic_fallback(market_state: dict) -> dict:
    """Pure Python regime classification when NIM times out."""
    spy_bars = market_state.get("SPY", [])
    closes_list = []
    for b in spy_bars:
        try:
            c = float(b["close"])
            if c > 0:
                closes_list.append(c)
        except (KeyError, TypeError, ValueError):
            continue

    if len(closes_list) < 21:
        return {"regime": "CHOP", "action": "HOLD", "confidence": 0.50}

    ret20 = (closes_list[-1] / closes_list[-21] - 1.0) if len(closes_list) >= 21 else 0.0
    mom5 = (closes_list[-1] / closes_list[-6] - 1.0) if len(closes_list) >= 6 else 0.0

    if len(closes_list) >= 21:
        rets = [(closes_list[i] / closes_list[i-1] - 1.0) for i in range(-20, 0)]
        vol20 = pstdev(rets) if len(rets) > 1 else 0.0
    else:
        vol20 = 0.0

    if len(closes_list) >= 20:
        window = closes_list[-20:]
        mu = mean(window)
        sigma = pstdev(window) if len(window) > 1 else 0.0001
        z20 = (closes_list[-1] - mu) / sigma if sigma > 0 else 0.0
    else:
        z20 = 0.0

    peak = max(closes_list) if closes_list else 1.0
    dd = (peak - closes_list[-1]) / peak if peak > 0 else 0.0

    if z20 < -1.5 and -0.12 <= ret20 <= -0.03 and mom5 > -0.01:
        return {"regime": "MEAN_REVERT_UP", "action": "BUY", "confidence": 0.60}
    if z20 > 2.5 and ret20 > 0.05:
        return {"regime": "MEAN_REVERT_DOWN", "action": "HOLD", "confidence": 0.65}
    if ret20 < -0.05 and vol20 > 0.025 and z20 < -1.5 and dd > 0.10:
        return {"regime": "TREND_DOWN", "action": "SELL", "confidence": 0.70}
    if ret20 > 0.03 and vol20 < 0.02 and z20 > 1.5:
        return {"regime": "TREND_UP", "action": "BUY", "confidence": 0.65}
    return {"regime": "CHOP", "action": "HOLD", "confidence": 0.55}


_REGIME_MAP = {
    "BEARISH": "TREND_DOWN", "BEAR": "TREND_DOWN", "DOWN": "TREND_DOWN", "DECLINE": "TREND_DOWN",
    "BULLISH": "TREND_UP", "BULL": "TREND_UP", "UP": "TREND_UP", "ADVANCE": "TREND_UP",
    "OVERSOLD": "MEAN_REVERT_UP", "BOUNCE": "MEAN_REVERT_UP",
    "OVERBOUGHT": "MEAN_REVERT_DOWN", "EXHAUSTION": "MEAN_REVERT_DOWN",
    "RANGE_BOUND": "CHOP", "SIDEWAYS": "CHOP", "NEUTRAL": "CHOP",
}


def _normalize_nim_response(resp: dict) -> dict:
    """Normalize regime names from 8B model to canonical form."""
    if not resp:
        return resp
    regime = resp.get("regime", "")
    resp["regime"] = _REGIME_MAP.get(regime.upper().replace(" ", "_"), regime)
    action = resp.get("action", "HOLD").upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    resp["action"] = action
    try:
        resp["confidence"] = max(0.0, min(1.0, float(resp.get("confidence", 0.5))))
    except (TypeError, ValueError):
        resp["confidence"] = 0.5
    return resp


def compute_agreement_score(nim_regime: str, features: dict) -> float:
    """Score 0.0–1.0: how well NIM regime aligns with deterministic signals.

    Use this instead of raw NIM confidence for position sizing.
    Raw LLM confidence is meaningless (0.80 for both correct and wrong calls).
    Agreement score measures whether the features actually support the regime label.
    """
    z20 = features.get("spy_zscore", 0)
    ret20 = features.get("spy_rolling_return_20", 0)
    mom5 = features.get("spy_momentum", {}).get("short", 0)
    vol20 = features.get("spy_volatility", 0)
    dd = features.get("spy_drawdown", 0)

    if nim_regime == "TREND_UP":
        return min(1.0, max(0.0, ret20 * 10 + (z20 / 3) - vol20 * 20))
    if nim_regime == "TREND_DOWN":
        return min(1.0, max(0.0, -ret20 * 8 + dd * 3 + vol20 * 15))
    if nim_regime == "MEAN_REVERT_UP":
        return min(1.0, max(0.0, (-z20 / 2) + (mom5 + 0.01) * 30))
    if nim_regime == "MEAN_REVERT_DOWN":
        return min(1.0, max(0.0, (z20 / 3) + ret20 * 8))
    if nim_regime == "CHOP":
        return min(1.0, max(0.0, 0.8 - abs(ret20) * 10 - abs(z20) * 0.3))
    return 0.3
    regime = resp.get("regime", "")
    resp["regime"] = _REGIME_MAP.get(regime.upper().replace(" ", "_"), regime)
    action = resp.get("action", "HOLD").upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    resp["action"] = action
    try:
        resp["confidence"] = max(0.0, min(1.0, float(resp.get("confidence", 0.5))))
    except (TypeError, ValueError):
        resp["confidence"] = 0.5
    return resp


def _build_nim_prompt(market_state: dict) -> str | None:
    """Build a compact prompt from market_state for NIM."""
    spy_bars = market_state.get("SPY", [])
    qqq_bars = market_state.get("QQQ", [])
    if len(spy_bars) < 21:
        return None

    spy_closes = []
    for b in spy_bars:
        try:
            c = float(b["close"])
            if c > 0:
                spy_closes.append(c)
        except (KeyError, TypeError, ValueError):
            continue

    qqq_closes = []
    for b in qqq_bars:
        try:
            c = float(b["close"])
            if c > 0:
                qqq_closes.append(c)
        except (KeyError, TypeError, ValueError):
            continue

    if len(spy_closes) < 21:
        return None

    # Compute features
    ret20 = (spy_closes[-1] / spy_closes[-21] - 1.0) if len(spy_closes) >= 21 else 0.0
    ret60 = (spy_closes[-1] / spy_closes[-61] - 1.0) if len(spy_closes) >= 61 else 0.0
    mom5 = (spy_closes[-1] / spy_closes[-6] - 1.0) if len(spy_closes) >= 6 else 0.0
    mom20 = ret20

    # Volatility
    if len(spy_closes) >= 21:
        rets = [(spy_closes[i] / spy_closes[i-1] - 1.0) for i in range(-20, 0)]
        vol20 = pstdev(rets) if len(rets) > 1 else 0.0
    else:
        vol20 = 0.0

    # Z-score
    if len(spy_closes) >= 20:
        window = spy_closes[-20:]
        mu = mean(window)
        sigma = pstdev(window) if len(window) > 1 else 0.0001
        z20 = (spy_closes[-1] - mu) / sigma if sigma > 0 else 0.0
    else:
        z20 = 0.0

    # Drawdown
    peak = max(spy_closes) if spy_closes else 1.0
    dd = (peak - spy_closes[-1]) / peak if peak > 0 else 0.0

    parts = [
        f"SPY: last={spy_closes[-1]:.2f}, ret20={ret20:.4f}, ret60={ret60:.4f}, "
        f"vol20={vol20:.4f}, z20={z20:.2f}, dd={dd:.4f}, mom5={mom5:.4f}, mom20={mom20:.4f}",
    ]

    if len(qqq_closes) >= 21:
        qqq_vol = pstdev([(qqq_closes[i]/qqq_closes[i-1]-1.0) for i in range(-20, 0)])
        qqq_mom5 = (qqq_closes[-1] / qqq_closes[-6] - 1.0) if len(qqq_closes) >= 6 else 0.0
        parts.append(f"QQQ: last={qqq_closes[-1]:.2f}, vol20={qqq_vol:.4f}, mom5={qqq_mom5:.4f}")

    parts.append("Portfolio: equity=100000, cash%=0.50, positions=0")

    # Pre-label regime hint to guide the model
    if z20 < -1.5 and mom5 > -0.01:
        parts.append("Regime_hint: OVERSOLD_BOUNCE_POSSIBLE — mom5 suggests stabilization")
    elif z20 < -1.5 and mom5 < -0.03:
        parts.append("Regime_hint: CONTINUED_DECLINE — mom5 confirms downtrend")
    elif z20 > 2.5 and ret20 > 0.05:
        parts.append("Regime_hint: OVERBOUGHT_EXHAUSTION")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic Helpers (Calmar Rotation Hybrid)
# ---------------------------------------------------------------------------

def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return mean(values[-n:])


def momentum(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    if start <= 0:
        return None
    return values[-1] / start - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        existing = positions.setdefault(ticker, {"quantity": 0.0, "avg_cost": avg_cost})
        existing["quantity"] += qty
        existing["avg_cost"] = avg_cost or existing["avg_cost"]
    return positions


def equity(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _latest_bar_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    if ts is None:
        return str(len(bars))
    return str(ts)[:10]


def _days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def _risk_off_targets(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    return {ticker: weight for ticker, weight in DEFENSIVE_WEIGHTS if closes(market_state.get(ticker))}


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def target_weights(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < 50 or len(qqq) < 50:
        return {}

    spy_sma50 = sma(spy, 50)
    qqq_sma50 = sma(qqq, 50)
    qqq_vol20 = realized_vol(qqq, 20)
    risk_on = bool(
        spy_sma50 is not None
        and qqq_sma50 is not None
        and qqq_vol20 is not None
        and spy[-1] > spy_sma50
        and qqq[-1] > qqq_sma50
        and qqq_vol20 < 0.35
    )
    if not risk_on:
        return _scale_caps(_risk_off_targets(market_state))

    scored: list[tuple[float, str]] = []
    for ticker in RISK_CANDIDATES:
        values = closes(market_state.get(ticker))
        if len(values) < 61:
            continue
        mom60 = momentum(values, 60)
        mom20 = momentum(values, 20)
        trend50 = sma(values, 50)
        vol20 = realized_vol(values, 20)
        if mom60 is None or mom20 is None or trend50 is None or vol20 is None:
            continue
        trend_gap = values[-1] / trend50 - 1.0
        score = (0.55 * mom60) + (0.25 * mom20) + (0.20 * trend_gap) - (0.15 * vol20)
        if score > 0.0:
            scored.append((score, ticker))

    scored.sort(reverse=True)
    winners = [ticker for _, ticker in scored[:5]]
    if not winners:
        return _scale_caps(_risk_off_targets(market_state))

    qqq_sma20 = sma(qqq, 20)
    qqq_mom20 = momentum(qqq, 20)
    overlay_on = bool(
        qqq_sma20 is not None
        and qqq_sma50 is not None
        and qqq_mom20 is not None
        and qqq_sma20 > qqq_sma50
        and qqq_mom20 > 0.0
        and qqq_vol20 < 0.28
        and closes(market_state.get("QLD"))
        and closes(market_state.get("SSO"))
    )

    weights: dict[str, float] = {}
    base_budget = 0.76 if overlay_on else 0.92
    per_winner = min(MAX_WEIGHT - 0.02, base_budget / len(winners))
    for ticker in winners:
        weights[ticker] = per_winner

    if overlay_on:
        weights["QLD"] = 0.11
        weights["SSO"] = 0.07

    return _scale_caps(weights)


def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0:
        return []

    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        qty = pos["quantity"]
        current_value = qty * price
        target_value = total_equity * targets.get(ticker, 0.0)
        delta = target_value - current_value
        if ticker not in targets:
            sell_qty = int(qty)
            if sell_qty > 0 and current_value >= min_trade:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
        elif delta < -min_trade:
            sell_qty = min(int(abs(delta) // price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price

    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.98)

    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = total_equity * weight
        delta = target_value - current_value
        if delta < min_trade:
            continue
        buy_value = min(delta, spendable)
        buy_qty = int(buy_value // price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price

    return orders[:45]


def _has_position_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
    if total_equity <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False


# ---------------------------------------------------------------------------
# Main Decision Function
# ---------------------------------------------------------------------------

def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """Return a list of long-only buy/sell orders.

    Two-layer architecture:
      Layer 1: Deterministic Calmar Rotation Hybrid
      Layer 2: NIM regime overlay (optional, fails safe to Layer 1)
    """
    global _last_rebalance_bar_date, _last_targets

    if not market_state:
        return []

    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []

    total_equity = equity(portfolio_state, cash)

    # --- HARD CRASH GUARD: sell immediately, bypass NIM entirely ---
    # For clear crash regimes, don't wait for NIM latency. Sell now.
    _spy_bars = market_state.get("SPY", [])
    _spy_cs = []
    for _b in _spy_bars:
        try:
            _c = float(_b["close"])
            if _c > 0:
                _spy_cs.append(_c)
        except (KeyError, TypeError, ValueError):
            continue
    if len(_spy_cs) >= 21:
        _r20 = _spy_cs[-1] / _spy_cs[-21] - 1.0
        _z = 0.0
        _w = _spy_cs[-20:]
        _mu = mean(_w)
        _sig = pstdev(_w) if len(_w) > 1 else 0.0001
        _z = (_spy_cs[-1] - _mu) / _sig if _sig > 0 else 0.0
        _ret_s = [(_spy_cs[i] / _spy_cs[i-1] - 1.0) for i in range(-20, 0)]
        _vol = pstdev(_ret_s) if len(_ret_s) > 1 else 0.0
        _pk = max(_spy_cs)
        _dd = (_pk - _spy_cs[-1]) / _pk if _pk > 0 else 0.0

        if _r20 < -0.15 and _z < -1.5 and _vol > 0.015:
            # Clear crash — sell all positions immediately
            _positions = portfolio_state.get("positions", [])
            _orders = []
            for _p in _positions:
                _qty = int(float(_p.get("quantity", 0)))
                if _qty > 0:
                    _orders.append({"ticker": _p["ticker"], "side": "sell", "quantity": _qty})
            if _orders:
                return _orders

    # --- Layer 2: NIM regime overlay (optional, hard timeout) ---
    nim_regime = None
    nim_action = None
    nim_confidence = 0.5
    effective_confidence = 0.0
    nim_start = time.time()
    nim_prompt = _build_nim_prompt(market_state)
    if nim_prompt:
        nim_result = _call_nim(nim_prompt)
        if nim_result and isinstance(nim_result, dict):
            nim_result = _normalize_nim_response(nim_result)
            nim_regime = nim_result.get("regime")
            nim_action = nim_result.get("action")
            nim_confidence = nim_result.get("confidence", 0.5)
        else:
            # NIM failed — use deterministic fallback instead of blind CHOP/HOLD
            fb = _deterministic_fallback(market_state)
            nim_regime = fb.get("regime")
            nim_action = fb.get("action")
            nim_confidence = fb.get("confidence", 0.5)
    nim_elapsed = time.time() - nim_start

    # If NIM took too long or we're close to deadline, skip NIM enhancement
    if nim_elapsed > 3.5:
        nim_regime = None
        nim_action = None

    # --- DETERMINISTIC REGIME OVERRIDE: correct model misclassifications ---
    spy_bars = market_state.get("SPY", [])
    spy_closes_list = []
    for b in spy_bars:
        try:
            c = float(b["close"])
            if c > 0:
                spy_closes_list.append(c)
        except (KeyError, TypeError, ValueError):
            continue

    if len(spy_closes_list) >= 21:
        _ret20 = spy_closes_list[-1] / spy_closes_list[-21] - 1.0
        _mom5 = (spy_closes_list[-1] / spy_closes_list[-6] - 1.0) if len(spy_closes_list) >= 6 else 0.0
        _window = spy_closes_list[-20:]
        _mu = mean(_window)
        _sigma = pstdev(_window) if len(_window) > 1 else 0.0001
        _z20 = (spy_closes_list[-1] - _mu) / _sigma if _sigma > 0 else 0.0
        _peak = max(spy_closes_list)
        _dd = (_peak - spy_closes_list[-1]) / _peak if _peak > 0 else 0.0
        _rets = [(spy_closes_list[i] / spy_closes_list[i-1] - 1.0) for i in range(-20, 0)] if len(spy_closes_list) >= 21 else []
        _vol20 = pstdev(_rets) if len(_rets) > 1 else 0.0

        # Build features dict for agreement scoring
        _features = {
            "spy_zscore": _z20,
            "spy_rolling_return_20": _ret20,
            "spy_momentum": {"short": _mom5},
            "spy_volatility": _vol20,
            "spy_drawdown": _dd,
        }

        # Compute agreement score — how well NIM regime matches features
        if nim_regime:
            agreement = compute_agreement_score(nim_regime, _features)
            effective_confidence = 0.3 * nim_confidence + 0.7 * agreement

            # Gate: skip NIM overlay if agreement is too low
            if effective_confidence < MIN_CONFIDENCE:
                nim_regime = None
                nim_action = None

        # MEAN_REVERT_UP: override model if it says TREND_DOWN but mom5 shows bounce
        if _z20 < -1.5 and -0.12 <= _ret20 <= -0.03 and _mom5 > -0.01:
            nim_regime = "MEAN_REVERT_UP"
            nim_action = "BUY"
        # TREND_DOWN: override model if it says MEAN_REVERT_UP but decline is accelerating
        if _ret20 < -0.05 and _z20 < -1.5 and _dd > 0.10 and _mom5 < -0.03:
            nim_regime = "TREND_DOWN"
            nim_action = "SELL" if portfolio_state.get("positions") else "HOLD"

    # --- Layer 1: Deterministic Calmar Rotation Hybrid ---
    days_since = _days_since_rebalance(market_state)
    drifted = _has_position_drifted(portfolio_state, total_equity)
    should_rebalance = (
        _last_rebalance_bar_date is None
        or days_since is None
        or days_since >= REBALANCE_EVERY_DAYS
        or drifted
    )
    if not should_rebalance:
        return []

    targets = target_weights(market_state)

    # --- NIM overlay: adjust targets based on regime ---
    if nim_regime == "TREND_DOWN" and nim_action in ("SELL", "HOLD"):
        # NIM says downtrend — go to defensive even if deterministic says risk-on
        targets = _scale_caps(_risk_off_targets(market_state))
    elif nim_regime == "MEAN_REVERT_UP" and nim_action == "BUY":
        # NIM sees oversold bounce — keep risk-on, don't flip to defensive
        pass  # targets remain as-is from deterministic layer
    elif nim_regime == "CHOP":
        # Choppy market — reduce position sizes proportional to confidence
        # Higher effective_confidence = more reduction (trust the CHOP call)
        if targets:
            chop_factor = 1.0 - (0.3 * effective_confidence)
            targets = {t: w * chop_factor for t, w in targets.items()}
            targets = _scale_caps(targets)

    if not targets:
        return []

    prices = _market_prices(market_state)
    positions = current_positions(portfolio_state)
    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
    if orders:
        _last_rebalance_bar_date = latest_date
        _last_targets = targets
    return orders
