"""Calmar Rotation Hybrid - Optimized for Max Risk-Adjusted Returns.

Upgrades:
1. Asymmetric Crash Switch: Uses a fast 20-day SMA for QQQ to exit tech drops early.
2. Volatility-Targeted Sizing: Allocates heavier weights to smooth, low-volatility winners.
"""
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

_last_rebalance_bar_date: str | None = None
_last_targets: dict[str, float] = {}

def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars: return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0: return []
        out.append(close)
    return out

def sma(values: list[float], n: int) -> float | None:
    if len(values) < n: return None
    return mean(values[-n:])

def momentum(values: list[float], n: int) -> float | None:
    if len(values) <= n: return None
    start = values[-(n + 1)]
    if start <= 0: return None
    return values[-1] / start - 1.0

def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) <= n: return None
    window = values[-(n + 1):]
    rets = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0: return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 5: return None
    return pstdev(rets) * sqrt(252.0)

def current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker: continue
        try:
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError): continue
        if qty <= 0: continue
        existing = positions.setdefault(ticker, {"quantity": 0.0, "avg_cost": avg_cost})
        existing["quantity"] += qty
        existing["avg_cost"] = avg_cost or existing["avg_cost"]
    return positions

def equity(portfolio_state: dict[str, Any], cash: float) -> float:
    try: total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError): total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try: price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError): price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)

def _latest_bar_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars: return None
    ts = bars[-1].get("ts")
    if ts is None: return str(len(bars))
    return str(ts)[:10]

def _days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_rebalance_bar_date is None: return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates: return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1

def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs: prices[ticker.upper()] = cs[-1]
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
    if len(spy) < 50 or len(qqq) < 50: return {}

    spy_sma50 = sma(spy, 50)
    qqq_sma20 = sma(qqq, 20) # UPGRADE: Fast Crash Detector for Tech
    qqq_sma50 = sma(qqq, 50)
    qqq_vol20 = realized_vol(qqq, 20)
    
    risk_on = bool(
        spy_sma50 is not None
        and qqq_sma20 is not None
        and qqq_vol20 is not None
        and spy[-1] > spy_sma50
        and qqq[-1] > qqq_sma20 # UPGRADE: Bail out of tech instantly if it drops
        and qqq_vol20 < 0.35
    )
    if not risk_on:
        return _scale_caps(_risk_off_targets(market_state))

    scored: list[tuple[float, str]] = []
    for ticker in RISK_CANDIDATES:
        values = closes(market_state.get(ticker))
        if len(values) < 61: continue
        mom60 = momentum(values, 60)
        mom20 = momentum(values, 20)
        trend50 = sma(values, 50)
        vol20 = realized_vol(values, 20)
        if mom60 is None or mom20 is None or trend50 is None or vol20 is None: continue
        trend_gap = values[-1] / trend50 - 1.0
        score = (0.55 * mom60) + (0.25 * mom20) + (0.20 * trend_gap) - (0.15 * vol20)
        if score > 0.0: scored.append((score, ticker))

    scored.sort(reverse=True)
    winners = [ticker for _, ticker in scored[:5]]
    if not winners: return _scale_caps(_risk_off_targets(market_state))

    qqq_mom20 = momentum(qqq, 20)
    overlay_on = bool(
        qqq_sma20 is not None and qqq_sma50 is not None and qqq_mom20 is not None
        and qqq_sma20 > qqq_sma50 and qqq_mom20 > 0.0 and qqq_vol20 < 0.28
        and closes(market_state.get("QLD")) and closes(market_state.get("SSO"))
    )

    # UPGRADE: Volatility-Targeted Position Sizing
    inv_vols = {}
    for ticker in winners:
        v = realized_vol(closes(market_state.get(ticker)), 20)
        inv_vols[ticker] = 1.0 / v if v and v > 0 else 1.0
        
    total_inv_vol = sum(inv_vols.values())
    base_budget = 0.76 if overlay_on else 0.92
    weights: dict[str, float] = {}
    
    for ticker in winners:
        raw_weight = (inv_vols[ticker] / total_inv_vol) * base_budget
        weights[ticker] = min(MAX_WEIGHT - 0.02, raw_weight)

    if overlay_on:
        weights["QLD"] = 0.11
        weights["SSO"] = 0.07

    return _scale_caps(weights)

def orders_to_rebalance(
    targets: dict[str, float], positions: dict[str, dict[str, float]],
    total_equity: float, prices: dict[str, float], cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0: return []
    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0: continue
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
        if price is None or price <= 0: continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = total_equity * weight
        delta = target_value - current_value
        if delta < min_trade: continue
        buy_value = min(delta, spendable)
        buy_qty = int(buy_value // price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price

    return orders[:45]

def _has_position_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
    if total_equity <= 0: return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try: price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError): price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False

def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    global _last_rebalance_bar_date, _last_targets

    if not market_state: return []
    latest_date = _latest_bar_date(market_state)
    if latest_date is None: return []

    total_equity = equity(portfolio_state, cash)
    days_since = _days_since_rebalance(market_state)
    drifted = _has_position_drifted(portfolio_state, total_equity)
    
    should_rebalance = (
        _last_rebalance_bar_date is None
        or days_since is None
        or days_since >= REBALANCE_EVERY_DAYS
        or drifted
    )
    if not should_rebalance: return []

    targets = target_weights(market_state)
    if not targets: return []

    prices = _market_prices(market_state)
    positions = current_positions(portfolio_state)
    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
    
    if orders:
        _last_rebalance_bar_date = latest_date
        _last_targets = targets
    return orders