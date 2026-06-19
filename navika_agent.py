from __future__ import annotations

from typing import Any

# --- Strategy: Simple Moving Average Trend Follower ---
# Logic: hold SPY when its short-term trend is above its long-term trend
# (an uptrend), otherwise move fully to cash. This avoids overtrading by
# only acting on the underlying trend signal, not daily price noise.

TICKER = "SPY"
SHORT_WINDOW = 10
LONG_WINDOW = 50
MAX_POSITION_WEIGHT = 0.25  # stay well under their 30% concentration cap

def _closes(bars: list[dict]) -> list[float]:
    """Extract closing prices from a list of bar dicts."""
    return [float(b["close"]) for b in bars]


def _sma(prices: list[float], window: int) -> float | None:
    """Simple moving average over the last `window` prices."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _current_holding(portfolio_state: dict, ticker: str) -> int:
    """Return how many shares of `ticker` we currently hold."""
    for position in portfolio_state.get("positions", []):
        if position["ticker"] == ticker:
            return int(position["quantity"])
    return 0


def _equity(portfolio_state: dict, cash: float) -> float:
    """Total portfolio value: cash + market value of all positions."""
    last_prices = portfolio_state.get("last_prices", {})
    total = cash
    for position in portfolio_state.get("positions", []):
        price = last_prices.get(position["ticker"], position.get("avg_cost", 0))
        total += position["quantity"] * price
    return total


def decide(
    market_state: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict]:
    """
    Trend-following strategy on a single ticker (SPY).

    Buy and hold SPY whenever its short-term moving average is above its
    long-term moving average (an uptrend). Exit to cash whenever the
    short-term average drops below the long-term average (a downtrend).
    """
    bars = market_state.get(TICKER, [])

    if len(bars) < LONG_WINDOW:
        return []

    closes = _closes(bars)
    short_avg = _sma(closes, SHORT_WINDOW)
    long_avg = _sma(closes, LONG_WINDOW)

    if short_avg is None or long_avg is None:
        return []

    uptrend = short_avg > long_avg

    current_price = closes[-1]
    if current_price <= 0:
        return []

    held_shares = _current_holding(portfolio_state, TICKER)
    equity = _equity(portfolio_state, cash)

    orders: list[dict] = []

    if uptrend:
        target_shares = int((equity * MAX_POSITION_WEIGHT) / current_price)
        diff = target_shares - held_shares
        if diff > 0:
            orders.append({"ticker": TICKER, "side": "buy", "quantity": diff})
        elif diff < 0:
            orders.append({"ticker": TICKER, "side": "sell", "quantity": abs(diff)})
    else:
        if held_shares > 0:
            orders.append({"ticker": TICKER, "side": "sell", "quantity": held_shares})

    return orders
