"""
Golden Cross Momentum Strategy (Fixed for Concentration Limits)
Contest objective: maximize 60-day forward Calmar.
Core idea:
Trade a custom universe of 50 mega-cap S&P 500 / DJIA stocks.
Buy only when a stock passes a strict multi-factor bullish filter:
1. Golden Cross state (50-day MA > 200-day MA)
2. Volume confirmation (Today's volume > 20-day avg volume)
3. Trend Strength (Price > 200-day MA, and within 2% of 52-week high)
4. Multi-Timeframe (Weekly trend is bullish: Close > 20-week MA)
Macro Safety Switch: If SPY is below its 200-day MA, move to defensive ETFs.
FIX: Defensive allocations are now strictly capped at 24% each to avoid the 30% concentration breach.
"""
from __future__ import annotations
from math import sqrt
from statistics import mean, pstdev
from typing import Any

# Top 50 highly liquid S&P 500 / DJIA stocks
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
    "XOM", "JNJ", "MA", "PG", "HD", "COST", "AVGO", "MRK", "ABBV", "CVX",
    "LLY", "PEP", "KO", "ADBE", "AMD", "TMO", "WMT", "CSCO", "MCD", "LIN",
    "ACN", "ABT", "CRM", "DHR", "TXN", "NEE", "PM", "UNP", "RTX", "LOW",
    "ORCL", "HON", "AMGN", "IBM", "QCOM", "INTU", "CAT", "GS", "BA", "SPGI"
]

REBALANCE_EVERY_DAYS = 5
MAX_WEIGHT = 0.24         # Hard cap per ticker (rules say < 30%)
DRIFT_LIMIT = 0.27
MAX_BETA_GROSS = 1.35
MIN_TRADE_PCT = 0.015

BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

_last_rebalance_bar_date: str | None = None

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

def volumes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars: return []
    out: list[float] = []
    for bar in bars:
        try:
            vol = float(bar.get("volume", 0))
            out.append(vol)
        except (TypeError, ValueError): return []
    return out

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

# --- THE STRATEGY LOGIC ---
def check_golden_cross_signal(bars: list[dict[str, Any]]) -> bool:
    """Checks if a single stock passes all the strict bullish filters."""
    if len(bars) < 260:
        return False
        
    c = closes(bars)
    v = volumes(bars)
    
    if not c or not v: return False
    
    close = c[-1]
    vol = v[-1]
    
    sma50 = mean(c[-50:])
    sma200 = mean(c[-200:])
    
    # 1. Golden Cross State (50-day MA > 200-day MA)
    if sma50 <= sma200:
        return False
        
    # 2. Volume Confirmation (Today's volume > 20-day average)
    avg_vol_20 = mean(v[-20:])
    if avg_vol_20 <= 0 or vol < avg_vol_20 * 1.10: # Must be at least 10% above average
        return False
        
    # 3. Trend Strength
    if close <= sma200:
        return False
        
    high_52w = max(c[-252:])
    if close < high_52w * 0.98: # Must be within 2% of 52-week high
        return False
        
    # 4. Multi-Timeframe Confirmation (Weekly chart bullish)
    # Synthesize weekly closes by taking every 5th trading day
    weekly_closes = [c[i] for i in range(len(c)) if i % 5 == 4]
    if len(weekly_closes) < 20:
        return False
        
    weekly_sma20 = mean(weekly_closes[-20:]) # 20-week SMA (~100-day SMA)
    if close <= weekly_sma20:
        return False
        
    return True

def target_weights(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    spy = closes(market_state.get("SPY"))
    if len(spy) < 200:
        return {}
        
    # MACRO SAFETY SWITCH
    spy_sma200 = mean(spy[-200:])
    if spy[-1] <= spy_sma200:
        # Market is in a long-term downtrend. Hide in defensive ETFs.
        # FIXED: Split into 4 assets at 24% each to strictly avoid the 30% concentration limit.
        return {"XLP": 0.24, "XLU": 0.24, "XLV": 0.24, "GLD": 0.24}
        
    # Scan our custom universe of 50 mega-caps
    winners = []
    for ticker in UNIVERSE:
        bars = market_state.get(ticker)
        if not bars:
            continue
            
        if check_golden_cross_signal(bars):
            # Rank by how close it is to the 52-week high (strongest momentum first)
            c = closes(bars)
            high_52w = max(c[-252:])
            strength = c[-1] / high_52w
            winners.append((strength, ticker))
            
    if not winners:
        # No winners found, go defensive (again, capped at 24% each)
        return {"XLP": 0.24, "XLU": 0.24, "XLV": 0.24, "GLD": 0.24}
        
    # Sort by strength and take top 5
    winners.sort(reverse=True)
    top_5 = [ticker for _, ticker in winners[:5]]
    
    # Equal weight the top 5 (20% each, safely under the 24% max)
    weight = 0.20
    targets = {ticker: weight for ticker in top_5}
    
    # Apply contest caps (beta scaling)
    capped = {t: min(w, MAX_WEIGHT) for t, w in targets.items()}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
        
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}

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
    global _last_rebalance_bar_date
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
    return orders
