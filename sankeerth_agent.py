from __future__ import annotations

from math import floor, sqrt
from statistics import pstdev
from typing import Any


THEME = (
    "MU", "MRVL", "AMD", "SOXX", "SMH", "NVDA", "AVGO", "QCOM",
    "XLK", "QQQ", "LRCX", "AMAT", "KLAC", "TSM", "PLTR", "ORCL",
    "META", "AMZN", "GOOGL", "MSFT", "LLY",
)
DEFENSIVE = ("XLP", "XLU", "XLV")
OVERLAY = ("QLD", "SSO")

BETA = {
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
}

TOP_N = 5
NAME_CAP = 0.265
OVERLAY_CAP = 0.090
MAX_BETA_GROSS = 1.42
CORE_FULL = 0.98
CORE_NEUTRAL = 0.82
DEF_GROSS = 0.26
MIN_TRADE_PCT = 0.020
REBALANCE_DAYS = 2
TRAIL_STOP = 0.095

_peak_equity = 0.0
_last_rebalance_date: str | None = None
_highs: dict[str, float] = {}


def _date(row: dict[str, Any]) -> str:
    return str(row.get("ts", ""))[:10]


def _closes(ms: dict[str, list[dict[str, Any]]], ticker: str) -> list[float]:
    out: list[float] = []
    for row in ms.get(ticker, []) or []:
        try:
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def _sma(values: list[float], n: int) -> float | None:
    return sum(values[-n:]) / n if len(values) >= n else None


def _ret(values: list[float], n: int) -> float | None:
    if len(values) < n + 1:
        return None
    base = values[-(n + 1)]
    return values[-1] / base - 1.0 if base > 0 else None


def _vol(values: list[float], n: int) -> float | None:
    if len(values) < n + 1:
        return None
    rets = []
    for i in range(len(values) - n, len(values)):
        prev = values[i - 1]
        if prev <= 0:
            return None
        rets.append(values[i] / prev - 1.0)
    return pstdev(rets) * sqrt(252.0) if len(rets) >= 5 else None


def _positions(portfolio: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio.get("positions", []) or []:
        try:
            ticker = str(raw.get("ticker", "")).upper()
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if ticker and qty > 0:
            old = out.get(ticker, {"quantity": 0.0, "avg_cost": avg_cost})
            total = old["quantity"] + qty
            out[ticker] = {
                "quantity": total,
                "avg_cost": ((old["avg_cost"] * old["quantity"]) + (avg_cost * qty)) / total if total > 0 else avg_cost,
            }
    return out


def _price(ms: dict[str, list[dict[str, Any]]], ticker: str, last: dict[str, Any]) -> float | None:
    values = _closes(ms, ticker)
    if values:
        return values[-1]
    try:
        p = float(last.get(ticker, 0.0))
        return p if p > 0 else None
    except (TypeError, ValueError):
        return None


def _equity(ms: dict[str, list[dict[str, Any]]], portfolio: dict[str, Any], cash: float) -> float:
    try:
        total = float(portfolio.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last = portfolio.get("last_prices", {}) or {}
    for ticker, pos in _positions(portfolio).items():
        price = _price(ms, ticker, last) or pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _latest_date(ms: dict[str, list[dict[str, Any]]]) -> str | None:
    for ticker in ("SPY", "QQQ"):
        rows = ms.get(ticker) or []
        if rows:
            return _date(rows[-1])
    for rows in ms.values():
        if rows:
            return _date(rows[-1])
    return None


def _days_since(ms: dict[str, list[dict[str, Any]]], start: str | None) -> int | None:
    if start is None:
        return None
    rows = ms.get("SPY") or ms.get("QQQ") or []
    dates = [_date(row) for row in rows]
    if start not in dates:
        return None
    return len(dates) - dates.index(start) - 1


def _regime(ms: dict[str, list[dict[str, Any]]]) -> tuple[str, bool]:
    spy = _closes(ms, "SPY")
    qqq = _closes(ms, "QQQ")
    if len(spy) < 55 or len(qqq) < 55:
        return "CASH", False

    spy20, spy50 = _sma(spy, 20), _sma(spy, 50)
    qqq20, qqq50 = _sma(qqq, 20), _sma(qqq, 50)
    qv20, qv10 = _vol(qqq, 20), _vol(qqq, 10)
    r3, r5 = _ret(qqq, 3), _ret(qqq, 5)
    if (
        (r3 is not None and r3 < -0.050)
        or (r5 is not None and r5 < -0.070)
        or (qv10 is not None and qv10 > 0.58)
        or (spy50 is not None and spy[-1] < spy50 * 0.990)
        or (qqq50 is not None and qqq[-1] < qqq50 * 0.990)
    ):
        return "CASH", False
    if None in (spy20, spy50, qqq20, qqq50, qv20):
        return "NEUTRAL", False

    full = (
        spy[-1] > spy20
        and qqq[-1] > qqq20
        and spy[-1] > spy50 * 1.002
        and qqq[-1] > qqq50 * 1.002
        and qv20 < 0.38
    )
    calm = full and qv20 < 0.27 and (_ret(qqq, 10) or 0.0) > 0.0
    return ("FULL" if full else "NEUTRAL"), calm


def _score(ms: dict[str, list[dict[str, Any]]], ticker: str) -> tuple[float, float] | None:
    values = _closes(ms, ticker)
    if len(values) < 55:
        return None
    r3 = _ret(values, 3)
    r5 = _ret(values, 5)
    r10 = _ret(values, 10)
    r21 = _ret(values, 21)
    r42 = _ret(values, 42)
    s10 = _sma(values, 10)
    s20 = _sma(values, 20)
    s50 = _sma(values, 50)
    vol20 = _vol(values, 20)
    if None in (r3, r5, r10, r21, r42, s10, s20, s50, vol20):
        return None
    if values[-1] <= s10 or values[-1] <= s20 or values[-1] <= s50:
        return None
    if r5 <= 0 or r10 <= 0:
        return None
    gap = values[-1] / s50 - 1.0
    if gap > 0.45 or vol20 > 1.15:
        return None
    raw = 0.24 * r3 + 0.32 * r5 + 0.24 * r10 + 0.14 * r21 + 0.06 * r42 + 0.04 * gap
    score = raw / max(vol20, 0.08)
    return (score, vol20) if score > 0 else None


def _leader_targets(ms: dict[str, list[dict[str, Any]]], gross: float) -> dict[str, float]:
    scored = []
    for ticker in THEME:
        out = _score(ms, ticker)
        if out:
            score, vol = out
            scored.append((score, vol, ticker))
    scored.sort(reverse=True)
    selected = scored[:TOP_N]
    if not selected:
        return {}

    # Rank-linear base, then slight inverse-vol adjustment. This keeps the book
    # concentrated in true leaders without letting one ultra-low-vol name dominate.
    n = len(selected)
    denom = n * (n + 1) / 2.0
    weights = {}
    for rank, (_score_v, vol, ticker) in enumerate(selected, start=1):
        rank_weight = (n - rank + 1) / denom
        vol_adj = min(1.25, max(0.75, 0.34 / max(vol, 0.08)))
        weights[ticker] = min(NAME_CAP, gross * rank_weight * vol_adj)

    total = sum(weights.values())
    if total > gross and total > 0:
        scale = gross / total
        weights = {ticker: weight * scale for ticker, weight in weights.items()}
    return {ticker: round(weight, 6) for ticker, weight in weights.items() if weight > 0.004}


def _def_targets(ms: dict[str, list[dict[str, Any]]], gross: float) -> dict[str, float]:
    names = []
    for ticker in DEFENSIVE:
        values = _closes(ms, ticker)
        s50 = _sma(values, 50)
        if values and s50 and values[-1] > s50 * 0.985:
            names.append(ticker)
    if not names:
        return {}
    per = min(0.11, gross / len(names))
    return {ticker: per for ticker in names}


def _targets(ms: dict[str, list[dict[str, Any]]], state: str, calm: bool, dd_scale: float) -> dict[str, float]:
    if state == "CASH":
        return _def_targets(ms, DEF_GROSS * 0.55 * dd_scale)

    gross = (CORE_FULL if state == "FULL" else CORE_NEUTRAL) * dd_scale
    weights = _leader_targets(ms, gross)
    if not weights:
        return _def_targets(ms, DEF_GROSS * dd_scale)

    if state == "FULL" and calm and all(_closes(ms, ticker) for ticker in OVERLAY):
        weights["QLD"] = min(OVERLAY_CAP, 0.060 * dd_scale)
        weights["SSO"] = min(OVERLAY_CAP, 0.045 * dd_scale)

    beta_gross = sum(weight * BETA.get(ticker, 1.0) for ticker, weight in weights.items())
    if beta_gross > MAX_BETA_GROSS and beta_gross > 0:
        scale = MAX_BETA_GROSS / beta_gross
        weights = {ticker: weight * scale for ticker, weight in weights.items()}
    return {ticker: round(weight, 6) for ticker, weight in weights.items() if weight > 0.004}


def _stops(ms: dict[str, list[dict[str, Any]]], positions: dict[str, dict[str, float]], last: dict[str, Any]) -> list[dict[str, object]]:
    global _highs

    orders = []
    for ticker in list(_highs):
        if ticker not in positions:
            del _highs[ticker]
    for ticker, pos in positions.items():
        price = _price(ms, ticker, last)
        if price is None:
            continue
        high = max(_highs.get(ticker, price), price)
        _highs[ticker] = high
        if price < high * (1.0 - TRAIL_STOP):
            orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})
            del _highs[ticker]
    return orders


def _orders(ms, targets, positions, equity, cash, stop_orders):
    last: dict[str, Any] = {}
    orders = list(stop_orders)
    stopped = {order["ticker"] for order in stop_orders}
    proceeds = 0.0
    min_trade = equity * MIN_TRADE_PCT

    for ticker in sorted(positions):
        if ticker in stopped:
            continue
        price = _price(ms, ticker, last)
        if not price:
            continue
        qty = positions[ticker]["quantity"]
        current = qty * price
        target = equity * targets.get(ticker, 0.0)
        if ticker not in targets:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
            proceeds += current
        elif current - target > min_trade:
            sell_qty = min(floor((current - target) / price), floor(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": float(sell_qty)})
                proceeds += sell_qty * price

    spendable = max(float(cash or 0.0), 0.0) + proceeds * 0.98
    for ticker in sorted(targets, key=lambda key: (-targets[key], key)):
        price = _price(ms, ticker, last)
        if not price:
            continue
        held = positions.get(ticker, {}).get("quantity", 0.0)
        target_qty = floor(equity * targets[ticker] / price)
        delta = target_qty - held
        if delta > 0 and delta * price > min_trade:
            buy_qty = floor(min(delta * price, spendable) / price)
            if buy_qty > 0:
                orders.append({"ticker": ticker, "side": "buy", "quantity": float(buy_qty)})
                spendable -= buy_qty * price

    sells = [order for order in orders if order["side"] == "sell"]
    buys = [order for order in orders if order["side"] == "buy"]
    return (sells + buys)[:45]


def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    global _last_rebalance_date, _peak_equity

    try:
        ms = market_state or {}
        today = _latest_date(ms)
        if today is None:
            return []

        portfolio = portfolio_state or {}
        equity = _equity(ms, portfolio, cash)
        if equity <= 0:
            return []
        _peak_equity = max(_peak_equity, equity)
        drawdown = equity / _peak_equity - 1.0 if _peak_equity > 0 else 0.0
        dd_scale = 0.25 if drawdown <= -0.090 else 0.50 if drawdown <= -0.050 else 1.0

        positions = _positions(portfolio)
        state, calm = _regime(ms)
        stops = _stops(ms, positions, portfolio.get("last_prices", {}) or {})
        days = _days_since(ms, _last_rebalance_date)
        scheduled = _last_rebalance_date is None or days is None or days >= REBALANCE_DAYS
        derisk = state == "CASH" and bool(positions)
        if not scheduled and not derisk and not stops:
            return []

        targets = _targets(ms, state, calm, dd_scale) if scheduled or derisk else {}
        orders = _orders(ms, targets, positions, equity, cash, stops)
        if orders and (scheduled or derisk):
            _last_rebalance_date = today
        return orders
    except Exception:
        return []
