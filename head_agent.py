"""Builderr Trading v0 agent built for forward Calmar robustness.

Design goals:
  * survive hidden crash / chop / rotation regimes
  * favor lower drawdowns over raw upside
  * use adaptive exposure instead of binary all-in / all-out logic
  * size positions by signal quality, volatility, and correlation crowding

The strategy stays long-only, uses only daily bars already provided, and keeps
cash as a first-class asset by often running below 1.0x gross.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

OFFENSIVE_ETFS = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE", "XLB", "SMH",
)
DEFENSIVE = ("XLP", "XLU", "XLV", "XLE", "GLD", "TLT")
BREADTH_SET = OFFENSIVE_ETFS

BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

MAX_WEIGHT = 0.23
MAX_BETA_GROSS = 1.04
DRIFT_LIMIT = 0.28
MIN_TRADE_PCT = 0.012
REBALANCE_DAYS = {"on": 6, "soft": 4, "hard": 2}
COOLDOWN_DAYS = 3
TARGET_MARKET_VOL = {"on": 0.16, "soft": 0.11, "hard": 0.08}

_last_rebalance_bar_date: str | None = None
_last_regime: str | None = None
_cooldown_days: int = 0
_last_targets: dict[str, float] = {}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    series: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        series.append(close)
    return series


def daily_returns(values: list[float], n: int | None = None) -> list[float]:
    if len(values) < 2:
        return []
    window = values if n is None else values[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0:
            return []
        rets.append(window[i] / prev - 1.0)
    return rets


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return mean(values[-n:])


def momentum(values: list[float], n: int, skip: int = 0) -> float | None:
    need = n + skip + 1
    if len(values) < need:
        return None
    end = values[-(skip + 1)]
    start = values[-(n + skip + 1)]
    if start <= 0 or end <= 0:
        return None
    return end / start - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    rets = daily_returns(values, n)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def sharpe_like(values: list[float], n: int) -> float:
    rets = daily_returns(values, n)
    if len(rets) < 5:
        return 0.0
    vol = pstdev(rets)
    if vol <= 1e-9:
        return 0.0
    return clamp((mean(rets) / vol) * sqrt(252.0) / 4.0, -1.0, 1.0)


def max_drawdown(values: list[float], n: int) -> float:
    window = values[-n:]
    if len(window) < 2:
        return 0.0
    peak = window[0]
    worst = 0.0
    for price in window:
        if price > peak:
            peak = price
        if peak > 0:
            worst = max(worst, (peak - price) / peak)
    return worst


def positive_day_ratio(values: list[float], n: int) -> float:
    rets = daily_returns(values, n)
    if not rets:
        return 0.0
    up_days = sum(1 for r in rets if r > 0.0)
    return ((up_days / len(rets)) - 0.5) * 2.0


def off_high(values: list[float], n: int) -> float:
    if len(values) < n:
        return 0.0
    high = max(values[-n:])
    if high <= 0:
        return 0.0
    return 1.0 - (values[-1] / high)


def correlation(values_a: list[float], values_b: list[float], n: int) -> float:
    rets_a = daily_returns(values_a, n)
    rets_b = daily_returns(values_b, n)
    if len(rets_a) < 5 or len(rets_b) < 5:
        return 0.0
    size = min(len(rets_a), len(rets_b))
    a = rets_a[-size:]
    b = rets_b[-size:]
    mean_a = mean(a)
    mean_b = mean(b)
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    if var_a <= 1e-12 or var_b <= 1e-12:
        return 0.0
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(size))
    return cov / sqrt(var_a * var_b)


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
    return str(ts)[:10] if ts is not None else str(len(bars))


def _days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(bar.get("ts", i))[:10] for i, bar in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        series = closes(bars)
        if series:
            prices[ticker.upper()] = series[-1]
    return prices


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


def _target_distance(left: dict[str, float], right: dict[str, float]) -> float:
    names = set(left) | set(right)
    return sum(abs(left.get(name, 0.0) - right.get(name, 0.0)) for name in names)


def _market_breadth(market_state: dict[str, list[dict[str, Any]]]) -> float:
    good = 0
    total = 0
    for ticker in BREADTH_SET:
        values = closes(market_state.get(ticker))
        sma50 = sma(values, 50)
        mom63 = momentum(values, 63, 5)
        if mom63 is None:
            mom63 = momentum(values, 20)
        if not values or sma50 is None or mom63 is None:
            continue
        total += 1
        if values[-1] > sma50 and mom63 > 0.0:
            good += 1
    return (good / total) if total else 0.0


def _classify_regime(market_state: dict[str, list[dict[str, Any]]]) -> tuple[str, float, float, float]:
    # Regime classification favors avoiding deep losses over catching every rally.
    # A fast crash brake handles violent breaks, then a smoother composite score
    # separates calm uptrends from uncertain "soft" markets.
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < 50 or len(qqq) < 50:
        return "hard", 0.0, 0.40, -1.0

    breadth = _market_breadth(market_state)
    qqq_vol20 = realized_vol(qqq, 20) or 0.40
    qqq_vol10 = realized_vol(qqq, 10) or qqq_vol20
    qqq_drop3 = momentum(qqq, 3) or 0.0
    qqq_drop5 = momentum(qqq, 5) or 0.0
    qqq_off10 = off_high(qqq, 10)

    if (
        qqq_drop3 <= -0.045
        or qqq_drop5 <= -0.065
        or qqq_off10 >= 0.08
        or qqq_vol10 >= 0.60
    ):
        return "hard", breadth, qqq_vol20, -1.0

    anchor = 100 if len(spy) >= 100 and len(qqq) >= 100 else 50
    spy_sma = sma(spy, anchor)
    qqq_sma = sma(qqq, anchor)
    qqq_sma50 = sma(qqq, 50)
    spy_gap = (spy[-1] / spy_sma - 1.0) if spy_sma else -0.05
    qqq_gap = (qqq[-1] / qqq_sma - 1.0) if qqq_sma else -0.05
    spy_mom = momentum(spy, 63, 5)
    qqq_mom = momentum(qqq, 63, 5)
    if spy_mom is None:
        spy_mom = momentum(spy, 20) or 0.0
    if qqq_mom is None:
        qqq_mom = momentum(qqq, 20) or 0.0

    trend_term = 0.5 * clamp(spy_gap / 0.08, -1.0, 1.0) + 0.5 * clamp(qqq_gap / 0.10, -1.0, 1.0)
    breadth_term = clamp((breadth - 0.40) / 0.25, -1.0, 1.0)
    momentum_term = 0.5 * clamp(spy_mom / 0.10, -1.0, 1.0) + 0.5 * clamp(qqq_mom / 0.12, -1.0, 1.0)
    vol_term = clamp((0.28 - qqq_vol20) / 0.16, -1.0, 1.0)
    regime_score = (
        0.35 * trend_term
        + 0.25 * breadth_term
        + 0.20 * momentum_term
        + 0.20 * vol_term
    )

    hard_condition = bool(
        qqq_gap < -0.02
        and breadth < 0.25
        and qqq_vol20 > 0.35
    )
    if hard_condition or regime_score < -0.30:
        return "hard", breadth, qqq_vol20, regime_score

    strong_on = bool(
        qqq_gap > 0.009
        and spy_gap > 0.004
        and breadth >= 0.54
        and qqq_mom > 0.0
        and qqq_vol20 < 0.34
    )
    clear_off = bool(
        regime_score < 0.05
        or breadth < 0.45
        or qqq_mom < 0.0
        or (qqq_sma50 is not None and qqq[-1] < qqq_sma50)
    )

    if _last_regime == "on":
        regime = "soft" if clear_off else "on"
    else:
        regime = "on" if strong_on or regime_score > 0.27 else "soft"
    return regime, breadth, qqq_vol20, regime_score


def _asset_score(
    values: list[float],
    spy_mom: float,
    qqq_mom: float,
    relaxed: bool = False,
) -> tuple[float, float] | None:
    # Score trend quality, not just speed. The best Calmar names tend to have
    # persistent momentum with modest volatility and shallow pullbacks.
    if len(values) < 70:
        return None

    price = values[-1]
    sma20 = sma(values, 20)
    sma50 = sma(values, 50)
    sma100 = sma(values, 100) or sma50
    mom21 = momentum(values, 21)
    mom63 = momentum(values, 63, 5)
    mom126 = momentum(values, 126, 10)
    if mom126 is None:
        mom126 = mom63
    vol20 = realized_vol(values, 20)
    if None in (sma20, sma50, sma100, mom21, mom63, mom126, vol20):
        return None

    trend50 = price / sma50 - 1.0
    trend100 = price / sma100 - 1.0
    accel = sma20 / sma50 - 1.0
    rel_spy = mom63 - spy_mom
    rel_qqq = mom63 - qqq_mom
    sharpe20 = sharpe_like(values, 20)
    persistence = positive_day_ratio(values, 20)
    dd20 = max_drawdown(values, 20)
    dd63 = max_drawdown(values, 63)
    off_high20 = off_high(values, 20)

    if relaxed:
        if trend50 <= -0.03 or mom21 <= -0.08:
            return None
    else:
        if price <= sma50 or mom63 <= 0.0 or trend100 <= -0.02:
            return None

    score = (
        0.18 * mom126
        + 0.18 * mom63
        + 0.08 * mom21
        + 0.10 * rel_spy
        + 0.07 * rel_qqq
        + 0.10 * trend50
        + 0.07 * trend100
        + 0.05 * accel
        + 0.07 * sharpe20
        + 0.05 * persistence
        - 0.12 * dd20
        - 0.08 * dd63
        - 0.08 * vol20
        - 0.05 * off_high20
    )
    if score <= 0.0:
        return None
    return score, max(vol20, 0.10)


def _rank_bucket(
    tickers: tuple[str, ...],
    market_state: dict[str, list[dict[str, Any]]],
    spy_mom: float,
    qqq_mom: float,
    relaxed: bool,
) -> tuple[list[tuple[float, str]], dict[str, tuple[float, float]]]:
    ranked: list[tuple[float, str]] = []
    score_map: dict[str, tuple[float, float]] = {}
    for ticker in tickers:
        scored = _asset_score(closes(market_state.get(ticker)), spy_mom, qqq_mom, relaxed=relaxed)
        if scored is None:
            continue
        score_map[ticker] = scored
        ranked.append((scored[0], ticker))
    ranked.sort(reverse=True)
    return ranked, score_map


def _select_greedy(
    ranked: list[tuple[float, str]],
    market_state: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[str]:
    selected: list[str] = []
    cache = {ticker: closes(market_state.get(ticker)) for _, ticker in ranked}
    for _, ticker in ranked:
        crowded = False
        for held in selected:
            if correlation(cache[ticker], cache[held], 20) > 0.96:
                crowded = True
                break
        if crowded:
            continue
        selected.append(ticker)
        if len(selected) >= limit:
            break
    return selected


def _correlation_penalties(
    winners: list[str],
    market_state: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    if len(winners) <= 1:
        return {ticker: 1.0 for ticker in winners}

    penalties: dict[str, float] = {}
    cache = {ticker: closes(market_state.get(ticker)) for ticker in winners}
    for ticker in winners:
        corr_values = [
            max(0.0, correlation(cache[ticker], cache[other], 20))
            for other in winners if other != ticker
        ]
        avg_corr = mean(corr_values) if corr_values else 0.0
        penalties[ticker] = clamp(1.10 - 0.45 * avg_corr, 0.60, 1.05)
    return penalties


def _gross_target(regime: str, breadth: float, market_vol: float, regime_score: float) -> float:
    vol_scalar = clamp(TARGET_MARKET_VOL[regime] / max(market_vol, 0.10), 0.50, 1.20)
    if regime == "hard":
        return clamp(0.04 + 0.10 * max(breadth, 0.0), 0.04, 0.14)
    if regime == "soft":
        return clamp(0.18 + 0.17 * vol_scalar * max(breadth, 0.25) + 0.05 * max(regime_score, 0.0), 0.18, 0.42)
    return clamp(0.76 + 0.19 * vol_scalar * (0.58 + breadth) + 0.05 * max(regime_score, 0.0), 0.76, 1.04)


def _allocate_weights(
    winners: list[str],
    score_map: dict[str, tuple[float, float]],
    corr_penalties: dict[str, float],
    gross: float,
) -> dict[str, float]:
    # Size by score and inverse volatility, then haircut crowded exposures so
    # the book is less vulnerable to one-factor unwind risk.
    if not winners or gross <= 0.0:
        return {}

    raw = {}
    for ticker in winners:
        if ticker not in score_map:
            continue
        score, vol = score_map[ticker]
        raw_signal = max(score, 0.0) ** 1.15
        raw[ticker] = (raw_signal / vol) * corr_penalties.get(ticker, 1.0)
    if not raw:
        return {}

    weights = {ticker: 0.0 for ticker in raw}
    target_gross = min(gross, MAX_BETA_GROSS)
    remaining = target_gross
    active = set(raw)

    while active and remaining > 0.0001:
        total_raw = sum(raw[ticker] for ticker in active)
        if total_raw <= 0.0:
            break
        newly_capped: set[str] = set()
        for ticker in list(active):
            proposal = remaining * (raw[ticker] / total_raw)
            room = MAX_WEIGHT - weights[ticker]
            add = min(max(proposal, 0.0), room)
            weights[ticker] += add
            if room - add <= 1e-9:
                newly_capped.add(ticker)
        remaining = target_gross - sum(weights.values())
        if not newly_capped:
            break
        active -= newly_capped

    return {ticker: round(weight, 6) for ticker, weight in weights.items() if weight > 0.001}


def target_weights(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    spy_mom = momentum(spy, 63, 5)
    qqq_mom = momentum(qqq, 63, 5)
    if spy_mom is None:
        spy_mom = momentum(spy, 20)
    if qqq_mom is None:
        qqq_mom = momentum(qqq, 20)
    if len(spy) < 70 or len(qqq) < 70 or spy_mom is None or qqq_mom is None:
        return {}

    regime, breadth, market_vol, regime_score = _classify_regime(market_state)
    relaxed = regime != "on"

    etf_ranked, etf_scores = _rank_bucket(OFFENSIVE_ETFS, market_state, spy_mom, qqq_mom, relaxed=relaxed)
    defensive_ranked, defensive_scores = _rank_bucket(DEFENSIVE, market_state, spy_mom, qqq_mom, relaxed=True)

    score_map = {**etf_scores, **defensive_scores}
    ranked: list[tuple[float, str]]
    limit: int

    if regime == "hard":
        ranked = defensive_ranked[:3]
        limit = 3
    elif regime == "soft":
        ranked = defensive_ranked[:3]
        if breadth >= 0.53 and etf_ranked:
            ranked += etf_ranked[:1]
        ranked.sort(reverse=True)
        limit = 4
    else:
        ranked = list(etf_ranked[:5])
        ranked.sort(reverse=True)
        limit = 5

    winners = _select_greedy(ranked, market_state, limit)
    if not winners and regime != "hard":
        winners = _select_greedy(defensive_ranked[:2], market_state, 2)
    corr_penalties = _correlation_penalties(winners, market_state)
    gross = _gross_target(regime, breadth, market_vol, regime_score)
    return _allocate_weights(winners, score_map, corr_penalties, gross)


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


def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    # Re-risk slowly, de-risk quickly, and ignore small target changes that
    # would mostly add turnover rather than improve the drawdown profile.
    global _last_rebalance_bar_date, _last_regime, _cooldown_days, _last_targets

    if not market_state:
        return []

    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []

    total_equity = equity(portfolio_state, cash)
    days_since = _days_since_rebalance(market_state)
    drifted = _has_position_drifted(portfolio_state, total_equity)
    raw_regime, _, _, _ = _classify_regime(market_state)

    regime = raw_regime
    if raw_regime == "hard":
        _cooldown_days = COOLDOWN_DAYS
    elif _cooldown_days > 0:
        _cooldown_days -= 1
        if raw_regime == "on":
            regime = "soft"

    cadence = REBALANCE_DAYS.get(regime, 5)
    regime_changed = _last_regime is not None and regime != _last_regime
    forced_derisk = _last_regime == "on" and regime != "on"
    should_rebalance = (
        _last_rebalance_bar_date is None
        or days_since is None
        or days_since >= cadence
        or drifted
        or regime_changed
        or forced_derisk
    )

    targets = target_weights(market_state) if should_rebalance else {}
    if not should_rebalance and targets:
        should_rebalance = _target_distance(_last_targets, targets) >= 0.24
    if not should_rebalance:
        _last_regime = regime
        return []
    if not targets:
        targets = target_weights(market_state)
    if not targets and regime == "hard":
        _last_regime = regime
        _last_rebalance_bar_date = latest_date
        _last_targets = {}
        return []
    if not targets:
        _last_regime = regime
        return []

    prices = _market_prices(market_state)
    positions = current_positions(portfolio_state)
    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
    _last_regime = regime
    if orders or regime_changed:
        _last_rebalance_bar_date = latest_date
        _last_targets = targets
    return orders
