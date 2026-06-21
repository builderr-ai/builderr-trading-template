"""Relative Momentum (Stat Arb Adaptation).
Contest objective: maximize 60-day forward Calmar, not raw return.
Core idea:
Exploit pricing inefficiencies between correlated pairs by buying the outperformer.
Strict risk-off switch: if the broader market (SPY) loses its 50-day trend, 
move entirely to defensive assets to protect the Calmar score.
"""
from __future__ import annotations
from math import sqrt
from statistics import mean, pstdev
from typing import Any

# --- CONFIGURATION ---
# We will buy the winner of each pair.
PAIRS = [
    ("NVDA", "AMD"),    # Semiconductors
    ("MSFT", "GOOGL"),  # Big Tech
    ("META", "AMZN"),   # Big Tech
    ("AVGO", "MRVL"),   # AI / Chips
    ("V", "MA"),        # Payments
    ("NFLX", "DIS"),    # Media / Streaming
]

REBALANCE_EVERY_DAYS = 5  # Check pairs weekly to avoid over-trading
MAX_WEIGHT = 0.24         # Hard cap per ticker (rules say < 30%)
DRIFT_LIMIT = 0.27
MAX_BETA_GROSS = 1.35
MIN_TRADE_PCT = 0.015

# Beta multiples for the leverage cap (from AGENT_BRIEF.md)
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

_last_rebalance_bar_date: str | None = None
_last_targets: dict[str, float] = {}

# --- HELPER FUNCTIONS ---
def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars: return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
            if close <= 0: return []
            out.append(close)
        except (KeyError, TypeError, ValueError): return []
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
    rets = [window[i] / window[i-1] - 1.0 for i in range(1, len(window)) if window[i-1] > 0]
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
    return str(ts)[:10] if ts else str(len(bars))

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

def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}

# --- THE STRATEGY LOGIC ---
def target_weights(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    spy = closes(market_state.get("SPY"))
    if len(spy) < 50: return {}

    # 1. MARKET REGIME FILTER (The Safety Switch)
    # If SPY is below its 50-day average, the market is in a downtrend.
    # We hide in defensive Consumer Staples (XLP) and Utilities (XLU).
    spy_sma50 = sma(spy, 50)
    risk_on = spy[-1] > spy_sma50

    if not risk_on:
        return _scale_caps({"XLP": 0.50, "XLU": 0.50})

    # 2. RELATIVE MOMENTUM (Stat Arb Adaptation)
    winners = []
    for t1, t2 in PAIRS:
        c1 = closes(market_state.get(t1))
        c2 = closes(market_state.get(t2))

        # Calculate 20-day momentum for both stocks in the pair
        mom1 = momentum(c1, 20)
        mom2 = momentum(c2, 20)

        if mom1 is not None and mom2 is not None:
            # We only buy the winner if its absolute momentum is > 0.
            # This prevents us from buying a "falling knife" if both stocks are crashing.
            if mom1 > mom2 and mom1 > 0:
                winners.append((mom1, t1))
            elif mom2 > mom1 and mom2 > 0:
                winners.append((mom2, t2))

    # Sort the winning stocks by their momentum, pick the top 4
    winners.sort(reverse=True)
    top_winners = [ticker for _, ticker in winners[:4]]

    if not top_winners:
        return _scale_caps({"XLP": 0.50, "XLU": 0.50})

    # 3. ALLOCATE WEIGHTS
    # We pick max 4 stocks. We allocate 20% to each (80% invested, 20% cash buffer).
    # This keeps us well under the 30% concentration limit and 1.5x leverage cap.
    weight_per_stock = 0.20
    targets = {ticker: weight_per_stock for ticker in top_winners}

    return _scale_caps(targets)

# --- EXECUTION & ORDER GENERATION ---
def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0: return []
    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    # Sells first
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

    # Buys second
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

def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """Return a list of long-only buy/sell orders."""
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