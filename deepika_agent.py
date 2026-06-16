"""Improved Calmar Rotation Hybrid - Enhanced for all market conditions."""
from __future__ import annotations
from math import sqrt
from statistics import mean, pstdev
from typing import Any

RISK_CANDIDATES = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "SMH",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
)
DEFENSIVE_WEIGHTS = (
    ("XLP", 0.22),
    ("XLU", 0.22),
    ("XLV", 0.22),
    ("XLE", 0.10),
    ("IWM", 0.08),
)
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

REBALANCE_EVERY_DAYS = 4
MAX_WEIGHT = 0.22
DRIFT_LIMIT = 0.25
MAX_BETA_GROSS = 1.30
MIN_TRADE_PCT = 0.012
STOP_LOSS_THRESHOLD = 0.06
VOL_HIGH = 0.30
VOL_EXTREME = 0.45

_last_rebalance_bar_date = None
_last_targets = {}
_entry_equity = None


def closes(bars):
    if not bars:
        return []
    out = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def sma(values, n):
    if len(values) < n:
        return None
    return mean(values[-n:])


def momentum(values, n):
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    if start <= 0:
        return None
    return values[-1] / start - 1.0


def realized_vol(values, n):
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


def current_positions(portfolio_state):
    positions = {}
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


def equity(portfolio_state, cash):
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


def _latest_bar_date(market_state):
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    if ts is None:
        return str(len(bars))
    return str(ts)[:10]


def _days_since_rebalance(market_state):
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def _market_prices(market_state):
    prices = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def _risk_off_targets(market_state):
    return {ticker: weight for ticker, weight in DEFENSIVE_WEIGHTS if closes(market_state.get(ticker))}


def _scale_caps(weights):
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def _market_regime(spy, qqq):
    if len(spy) < 50 or len(qqq) < 50:
        return "unknown"
    spy_sma20 = sma(spy, 20)
    spy_sma50 = sma(spy, 50)
    qqq_sma20 = sma(qqq, 20)
    qqq_sma50 = sma(qqq, 50)
    qqq_vol20 = realized_vol(qqq, 20)
    spy_mom10 = momentum(spy, 10)
    if qqq_vol20 and qqq_vol20 > VOL_EXTREME:
        return "extreme_vol"
    if (spy_sma50 and spy[-1] < spy_sma50 * 0.97 and
            qqq_sma50 and qqq[-1] < qqq_sma50 * 0.97):
        return "bear"
    if (spy_sma20 and spy[-1] < spy_sma20 and
            spy_mom10 and spy_mom10 < -0.03):
        return "selloff"
    if (qqq_vol20 and qqq_vol20 > VOL_HIGH):
        return "high_vol"
    if (spy_sma50 and spy[-1] > spy_sma50 and
            qqq_sma50 and qqq[-1] > qqq_sma50 and
            qqq_sma20 and qqq_sma50 and qqq_sma20 > qqq_sma50):
        return "bull"
    return "neutral"


def target_weights(market_state):
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < 50 or len(qqq) < 50:
        return {}

    regime = _market_regime(spy, qqq)

    if regime in ("bear", "extreme_vol"):
        return _scale_caps(_risk_off_targets(market_state))

    if regime == "selloff":
        defensive = dict(DEFENSIVE_WEIGHTS[:3])
        scored = []
        for ticker in ("SPY", "XLV", "XLP", "XLU"):
            values = closes(market_state.get(ticker))
            if len(values) < 21:
                continue
            mom20 = momentum(values, 20)
            vol20 = realized_vol(values, 20)
            if mom20 and vol20 and mom20 > -0.05:
                scored.append((mom20 / (vol20 + 0.01), ticker))
        scored.sort(reverse=True)
        result = {t: 0.15 for t in defensive}
        for _, t in scored[:2]:
            if t not in result:
                result[t] = 0.12
        return _scale_caps(result)

    scored = []
    for ticker in RISK_CANDIDATES:
        values = closes(market_state.get(ticker))
        if len(values) < 61:
            continue
        mom60 = momentum(values, 60)
        mom20 = momentum(values, 20)
        mom10 = momentum(values, 10)
        trend50 = sma(values, 50)
        vol20 = realized_vol(values, 20)
        if None in (mom60, mom20, mom10, trend50, vol20):
            continue
        if vol20 > 0.55:
            continue
        trend_gap = values[-1] / trend50 - 1.0
        vol_penalty = max(0, vol20 - 0.20) * 0.5
        score = (0.45 * mom60 + 0.30 * mom20 + 0.15 * mom10 +
                 0.15 * trend_gap - 0.20 * vol20 - vol_penalty)
        if score > 0.0:
            scored.append((score, ticker))

    scored.sort(reverse=True)
    top_n = 4 if regime == "high_vol" else 5
    winners = [ticker for _, ticker in scored[:top_n]]

    if not winners:
        return _scale_caps(_risk_off_targets(market_state))

    qqq_vol20 = realized_vol(qqq, 20)
    qqq_sma20 = sma(qqq, 20)
    qqq_sma50 = sma(qqq, 50)
    qqq_mom20 = momentum(qqq, 20)
    overlay_on = bool(
        regime == "bull" and
        qqq_sma20 and qqq_sma50 and qqq_sma20 > qqq_sma50 and
        qqq_mom20 and qqq_mom20 > 0.02 and
        qqq_vol20 and qqq_vol20 < 0.22 and
        closes(market_state.get("QLD")) and closes(market_state.get("SSO"))
    )

    weights = {}
    base_budget = 0.72 if overlay_on else 0.90
    if regime == "high_vol":
        base_budget *= 0.85
    per_winner = min(MAX_WEIGHT - 0.02, base_budget / len(winners))
    for ticker in winners:
        weights[ticker] = per_winner

    if overlay_on:
        weights["QLD"] = 0.10
        weights["SSO"] = 0.06

    return _scale_caps(weights)


def orders_to_rebalance(targets, positions, total_equity, prices, cash_available):
    if total_equity <= 0:
        return []
    min_trade = total_equity * MIN_TRADE_PCT
    orders = []
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


def _has_position_drifted(portfolio_state, total_equity):
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


def decide(market_state, portfolio_state, cash):
    """Return a list of long-only buy/sell orders."""
    global _last_rebalance_bar_date, _last_targets, _entry_equity

    if not market_state:
        return []

    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []

    total_equity = equity(portfolio_state, cash)
    if _entry_equity is None:
        _entry_equity = total_equity

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
    if not targets:
        return []

    prices = _market_prices(market_state)
    positions = current_positions(portfolio_state)
    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
    if orders:
        _last_rebalance_bar_date = latest_date
        _last_targets = targets
    return orders
