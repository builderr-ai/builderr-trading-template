"""Robert AI Infra: drawdown-aware AI momentum for builderr trading v0.

This agent is intentionally boring where it matters:

* no network, no LLM, no API keys, no third-party imports
* long-only orders over the public builderr universe
* position caps below the 30% rule and beta-adjusted gross below 1.35x
* a fast stress gate so a 30-day Calmar run is not ruined by one drawdown

The strategy is deliberately long AI/AI-infra for the 30-day live window, but
only when the tape agrees. It combines broad-market trend following,
cross-sectional momentum, a small gated leveraged overlay, volatility scaling,
and an equity-curve guard.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any


RISK_CANDIDATES = (
    "SMH", "XLK", "QQQ", "NVDA", "MSFT", "META", "AMZN", "GOOGL", "AAPL",
    "XLC", "XLY", "SPY", "XLE", "XLI", "XLV",
)
DEFENSIVE_CANDIDATES = ("XLP", "XLU", "XLV", "XLI", "XLE", "SPY")
AI_INFRA_BIAS = {
    "SMH": 0.035,
    "XLK": 0.028,
    "QQQ": 0.024,
    "NVDA": 0.022,
    "MSFT": 0.018,
    "META": 0.012,
    "AMZN": 0.010,
    "GOOGL": 0.008,
    "AAPL": 0.006,
}
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

REBALANCE_EVERY_DAYS = 5
MAX_WEIGHT = 0.235
DRIFT_LIMIT = 0.265
MAX_BETA_GROSS = 1.35
MIN_TRADE_PCT = 0.012

_last_rebalance_bar_date: str | None = None
_last_targets: dict[str, float] = {}
_peak_equity: float | None = None
_drawdown_guard: bool = False


def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0.0:
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
    base = values[-(n + 1)]
    if base <= 0.0:
        return None
    return values[-1] / base - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0.0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def max_drawdown(values: list[float], n: int) -> float | None:
    if len(values) < 2:
        return None
    window = values[-n:] if len(values) >= n else values
    peak = window[0]
    drawdown = 0.0
    for value in window:
        peak = max(peak, value)
        if peak > 0.0:
            drawdown = max(drawdown, peak / value - 1.0)
    return drawdown


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
        if qty <= 0.0:
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
    return str(ts)[:10] if ts is not None else str(len(bars))


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
        values = closes(bars)
        if values:
            prices[ticker.upper()] = values[-1]
    return prices


def _regime(market_state: dict[str, list[dict[str, Any]]]) -> str:
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < 60 or len(qqq) < 60:
        return "unknown"

    spy_sma20 = sma(spy, 20)
    spy_sma50 = sma(spy, 50)
    qqq_sma20 = sma(qqq, 20)
    qqq_sma50 = sma(qqq, 50)
    spy_mom5 = momentum(spy, 5)
    spy_mom10 = momentum(spy, 10)
    spy_mom20 = momentum(spy, 20)
    qqq_mom10 = momentum(qqq, 10)
    qqq_mom20 = momentum(qqq, 20)
    qqq_vol20 = realized_vol(qqq, 20)
    spy_dd20 = max_drawdown(spy, 20)
    qqq_dd20 = max_drawdown(qqq, 20)

    if None in (
        spy_sma20, spy_sma50, qqq_sma20, qqq_sma50, spy_mom5, spy_mom10,
        spy_mom20, qqq_mom10, qqq_mom20, qqq_vol20, spy_dd20, qqq_dd20,
    ):
        return "unknown"

    strong_rebound = (
        spy[-1] > spy_sma20
        and qqq[-1] > qqq_sma20
        and spy_mom10 > 0.055
        and qqq_mom10 > 0.075
        and spy_mom5 > -0.010
    )
    stress = (
        qqq_vol20 > 0.34
        or spy_mom5 < -0.035
        or spy_mom10 < -0.055
        or qqq_mom10 < -0.070
        or spy_dd20 > 0.070
        or qqq_dd20 > 0.090
    )
    if stress and not strong_rebound:
        return "stress"

    risk_on = (
        spy[-1] > spy_sma50
        and qqq[-1] > qqq_sma50
        and spy_mom10 > -0.018
        and qqq_mom10 > -0.025
        and qqq_vol20 < 0.30
    )
    if risk_on:
        return "risk_on"

    recovery = (
        spy[-1] > spy_sma20
        and qqq[-1] > qqq_sma20
        and spy_mom10 > 0.020
        and qqq_mom10 > 0.030
        and qqq_mom20 > -0.020
        and qqq_vol20 < 0.50
    )
    return "recovery" if recovery else "defensive"


def _score_asset(values: list[float]) -> float | None:
    if len(values) < 61:
        return None
    mom120 = momentum(values, 120)
    mom60 = momentum(values, 60)
    mom20 = momentum(values, 20)
    trend50 = sma(values, 50)
    vol20 = realized_vol(values, 20)
    dd60 = max_drawdown(values, 60)
    if mom120 is None:
        mom120 = mom60
    if None in (mom120, mom60, mom20, trend50, vol20, dd60):
        return None
    if mom60 < -0.015 or mom20 < -0.040:
        return None
    trend_gap = values[-1] / trend50 - 1.0
    return (
        0.05 * mom120
        + 0.50 * mom60
        + 0.25 * mom20
        + 0.20 * trend_gap
        - 0.14 * vol20
    )


def _rank_assets(
    market_state: dict[str, list[dict[str, Any]]],
    universe: tuple[str, ...],
) -> list[tuple[float, float, str]]:
    ranked: list[tuple[float, float, str]] = []
    for ticker in universe:
        values = closes(market_state.get(ticker))
        score = _score_asset(values)
        vol = realized_vol(values, 20) if values else None
        if score is not None and vol is not None and score > 0.0:
            score += AI_INFRA_BIAS.get(ticker, 0.0)
            ranked.append((score, max(vol, 0.05), ticker))
    ranked.sort(reverse=True)
    return ranked


def _allocate_from_ranked(
    ranked: list[tuple[float, float, str]],
    budget: float,
    count: int,
    max_weight: float,
) -> dict[str, float]:
    selected = ranked[:count]
    if not selected:
        return {}
    raw: dict[str, float] = {ticker: 1.0 for _, _, ticker in selected}
    total = sum(raw.values())
    if total <= 0.0:
        return {}

    weights = {ticker: min((value / total) * budget, max_weight) for ticker, value in raw.items()}
    leftover = max(0.0, budget - sum(weights.values()))
    uncapped = [ticker for ticker, value in weights.items() if value < max_weight - 1e-9]
    while leftover > 0.002 and uncapped:
        add = leftover / len(uncapped)
        next_uncapped: list[str] = []
        spent = 0.0
        for ticker in uncapped:
            room = max_weight - weights[ticker]
            bump = min(room, add)
            weights[ticker] += bump
            spent += bump
            if weights[ticker] < max_weight - 1e-9:
                next_uncapped.append(ticker)
        if spent <= 1e-9:
            break
        leftover -= spent
        uncapped = next_uncapped
    return weights


def _defensive_targets(
    market_state: dict[str, list[dict[str, Any]]],
    stress: bool,
) -> dict[str, float]:
    budget = 0.30 if stress else 0.58
    count = 3 if stress else 4
    max_weight = 0.12 if stress else 0.17
    ranked = _rank_assets(market_state, DEFENSIVE_CANDIDATES)
    if not ranked:
        fallback = [t for t in ("XLP", "XLU", "XLV") if closes(market_state.get(t))]
        if not fallback:
            return {}
        weight = min(max_weight, budget / len(fallback))
        return {ticker: weight for ticker in fallback}
    return _allocate_from_ranked(ranked, budget, count, max_weight)


def _overlay_allowed(market_state: dict[str, list[dict[str, Any]]]) -> bool:
    qqq = closes(market_state.get("QQQ"))
    smh = closes(market_state.get("SMH"))
    spy = closes(market_state.get("SPY"))
    if not closes(market_state.get("TQQQ")) or not closes(market_state.get("SOXL")):
        return False
    if len(qqq) < 80 or len(spy) < 80 or len(smh) < 80:
        return False
    qqq_sma20 = sma(qqq, 20)
    qqq_sma50 = sma(qqq, 50)
    qqq_mom20 = momentum(qqq, 20)
    qqq_vol20 = realized_vol(qqq, 20)
    smh_sma20 = sma(smh, 20)
    smh_sma50 = sma(smh, 50)
    smh_mom20 = momentum(smh, 20)
    smh_vol20 = realized_vol(smh, 20)
    spy_mom20 = momentum(spy, 20)
    if None in (
        qqq_sma20, qqq_sma50, qqq_mom20, qqq_vol20,
        smh_sma20, smh_sma50, smh_mom20, smh_vol20, spy_mom20,
    ):
        return False
    return bool(
        qqq_sma20 > qqq_sma50
        and smh_sma20 > smh_sma50
        and qqq_mom20 > 0.025
        and smh_mom20 > 0.035
        and spy_mom20 > 0.010
        and qqq_vol20 < 0.24
        and smh_vol20 < 0.38
    )


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = {
        ticker: min(max(weight, 0.0), MAX_WEIGHT)
        for ticker, weight in weights.items()
        if weight > 0.0
    }
    beta_gross = sum(weight * BETA_MULTIPLE.get(ticker, 1.0) for ticker, weight in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {ticker: weight * scale for ticker, weight in capped.items()}
    return {ticker: round(weight, 6) for ticker, weight in capped.items() if weight > 0.001}


def target_weights(
    market_state: dict[str, list[dict[str, Any]]],
    force_stress: bool = False,
) -> dict[str, float]:
    regime = "stress" if force_stress else _regime(market_state)
    if regime == "unknown":
        return {}
    if regime == "stress":
        return _scale_caps(_defensive_targets(market_state, stress=True))
    if regime == "defensive":
        return _scale_caps(_defensive_targets(market_state, stress=False))

    ranked = _rank_assets(market_state, RISK_CANDIDATES)
    if not ranked:
        return _scale_caps(_defensive_targets(market_state, stress=(regime != "risk_on")))

    if regime == "recovery":
        return _scale_caps(_allocate_from_ranked(ranked, budget=0.64, count=4, max_weight=0.18))

    overlay_on = _overlay_allowed(market_state)
    qqq = closes(market_state.get("QQQ"))
    qqq_vol20 = realized_vol(qqq, 20) or 0.30
    base_budget = 0.92
    if qqq_vol20 > 0.24:
        base_budget = 0.82
    if overlay_on:
        base_budget = 0.76

    weights = _allocate_from_ranked(ranked, budget=base_budget, count=5, max_weight=0.22)
    if not weights:
        return _scale_caps(_defensive_targets(market_state, stress=False))
    if overlay_on:
        weights["SOXL"] = 0.07
        weights["TQQQ"] = 0.05
    return _scale_caps(weights)


def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0.0:
        return []

    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0.0:
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

    spendable = max(float(cash_available), 0.0) + sell_proceeds * 0.98
    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if price is None or price <= 0.0:
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
    if total_equity <= 0.0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0.0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False


def _update_drawdown_guard(total_equity: float) -> bool:
    global _peak_equity, _drawdown_guard
    if total_equity <= 0.0:
        return True
    if _peak_equity is None:
        _peak_equity = total_equity
    _peak_equity = max(_peak_equity, total_equity)
    drawdown = 1.0 - (total_equity / _peak_equity)
    if drawdown > 0.055:
        _drawdown_guard = True
    elif _drawdown_guard and drawdown < 0.020:
        _drawdown_guard = False
    return _drawdown_guard


def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    """Return long-only buy/sell orders for the current daily decision."""
    global _last_rebalance_bar_date, _last_targets

    if not market_state:
        return []
    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []

    total_equity = equity(portfolio_state, cash)
    guard_on = _update_drawdown_guard(total_equity)
    regime = _regime(market_state)
    force_stress = guard_on and regime != "risk_on"
    days_since = _days_since_rebalance(market_state)
    drifted = _has_position_drifted(portfolio_state, total_equity)
    should_rebalance = (
        _last_rebalance_bar_date is None
        or days_since is None
        or days_since >= REBALANCE_EVERY_DAYS
        or drifted
        or force_stress
        or regime == "stress"
    )
    if not should_rebalance:
        return []

    targets = target_weights(market_state, force_stress=force_stress)
    if not targets:
        return []

    prices = _market_prices(market_state)
    positions = current_positions(portfolio_state)
    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
    if orders:
        _last_rebalance_bar_date = latest_date
        _last_targets = targets
    return orders
