"""Skip-Week Momentum + Trend Filter + Volatility Brake Strategy.

Philosophy: Hold what's working. Get out when it breaks. Size by calm, never bet the farm.

Core rules:
  1. Buckets: TECH, AI_SEMIS, INTERNET, FINANCIAL, ENERGY, DEFENSIVE, BROAD.
  2. Market filter: SPY 200-day SMA determines risk-on/off.
  3. Momentum: Skip-week (63d return skipping 3d) avoids short-term reversals.
  4. Trend filter: Current price > 50-day SMA required.
  5. Selection: Best 3 bucket winners that pass momentum & trend.
  6. Volatility brake: If SPY 20d vol > 30%, reduce exposure to 60%.
  7. Sizing: 1/volatility weighting, cap 27% per position, min 20% cash.
"""
from __future__ import annotations

from statistics import mean, stdev

# Buckets: each bucket contains related names. Pick best momentum from each.
BUCKETS = {
    "TECH": (
        "QQQ", "XLK",
        "MSFT", "AAPL", "AVGO",
        "ADBE", "CRM", "ORCL",
    ),
    "AI_SEMIS": (
        "NVDA", "AMD", "AVGO", "MU",
        "QCOM", "MRVL", "AMAT",
        "LRCX", "KLAC", "TSM",
    ),
    "INTERNET": (
        "META", "GOOGL", "AMZN",
        "NFLX", "UBER",
    ),
    "FINANCIAL": (
        "XLF", "JPM", "V", "MA",
    ),
    "ENERGY": (
        "XLE", "XOM", "CVX",
    ),
    "DEFENSIVE": (
        "XLV", "XLP", "XLU",
        "WMT", "COST", "LLY",
    ),
    "BROAD": (
        "SPY",
    ),
}

# Configuration
REBALANCE_EVERY_TICKS = 7
DRIFT_LIMIT = 0.27
POSITION_CAP = 0.28  # max 28% per position
MIN_CASH = 0.20  # keep at least 20% cash
TARGET_TOTAL_EXPOSURE = 0.80  # target ~80% invested
TRADE_MIN_PCT = 0.02

# Momentum configuration
SKIP_WEEK_LOOKBACK = 63  # 63 trading days
SKIP_RECENT_DAYS = 3  # skip most recent 3 trading days
TREND_SMA_DAYS = 50  # trend filter: price > 50d SMA
VOL_BRAKE_THRESHOLD = 0.30  # if SPY 20d vol > 30%, reduce exposure to 60%

TOP_BUCKETS = 3  # hold best 3 bucket winners

# State (persists across ticks)
_tick = 0
_last_rebalance = -10**9


def _closes(bars):
    """Extract closing prices from bars."""
    return [float(b["close"]) for b in bars] if bars else []


def _sma(closes, n):
    """Simple moving average."""
    if len(closes) < n:
        return None
    return mean(closes[-n:])


def _return(bars, days):
    """Total return over N days."""
    closes = _closes(bars)
    if len(closes) < days + 1:
        return None
    return (closes[-1] / closes[-(days + 1)] - 1.0) if closes[-(days + 1)] > 0 else None


def _volatility(bars, days=20):
    """20-day realized daily volatility."""
    closes = _closes(bars)
    if len(closes) < days + 1:
        return 0.001
    recent = closes[-(days + 1):]
    daily_rets = [recent[i] / recent[i - 1] - 1.0 for i in range(1, len(recent)) if recent[i - 1] > 0]
    if len(daily_rets) < 2:
        return 0.001
    vol = stdev(daily_rets)
    return max(vol, 0.001)


def _skip_week_momentum(bars):
    """Skip-week momentum: compare price 3 days ago vs 66 days ago."""
    closes = _closes(bars)
    total_lookback = SKIP_RECENT_DAYS + SKIP_WEEK_LOOKBACK
    if len(closes) < total_lookback + 1:
        return None
    
    price_recent = closes[-(SKIP_RECENT_DAYS + 1)]  # price 3 days ago
    price_old = closes[-(total_lookback + 1)]  # price 66 days ago
    
    if price_old <= 0:
        return None
    
    return price_recent / price_old - 1.0


def _ann_volatility(bars, days=20):
    """Annualized volatility for vol brake check."""
    daily_vol = _volatility(bars, days)
    return daily_vol * (252 ** 0.5)  # annualize


def _is_risk_on(market_state):
    """Check if SPY > 200 SMA. Risk ON = True, Risk OFF = False."""
    spy_bars = market_state.get("SPY") or []
    if len(spy_bars) < 200:
        return False
    
    closes = _closes(spy_bars)
    sma200 = _sma(closes, 200)
    
    if sma200 is None:
        return False
    
    return closes[-1] > sma200


def _get_exposure_multiplier(market_state):
    """Volatility brake: reduce exposure if SPY annualized 20d vol > 30%."""
    spy_bars = market_state.get("SPY") or []
    if not spy_bars:
        return 1.0
    
    ann_vol = _ann_volatility(spy_bars)
    if ann_vol > VOL_BRAKE_THRESHOLD:
        return 0.75  # exposure becomes 60% of equity when high vol
    return 1.0

def _target_weights(market_state):
    """Compute target weights using bucket strategy with skip-week momentum and trend filter."""
    # If risk is OFF, return all cash (empty targets dict).
    if not _is_risk_on(market_state):
        return {}
    
    # For each bucket, find the member with highest positive skip-week momentum AND price > 50d SMA.
    required_history = max(200, TREND_SMA_DAYS, SKIP_WEEK_LOOKBACK + SKIP_RECENT_DAYS) + 1
    bucket_winners = {}
    for bucket_name, members in BUCKETS.items():
        best_member = None
        best_momentum = None
        
        for ticker in members:
            bars = market_state.get(ticker) or []
            closes = _closes(bars)
            if len(closes) < required_history:
                continue
            
            # Check trend filter: price > 50d SMA
            sma50 = _sma(closes, TREND_SMA_DAYS)
            if sma50 is None or closes[-1] <= sma50:
                continue  # fail trend filter
            
            # Check skip-week momentum
            momentum = _skip_week_momentum(bars)
            
            # Only consider positive momentum.
            if momentum is not None and momentum > 0:
                if best_momentum is None or momentum > best_momentum:
                    best_momentum = momentum
                    best_member = ticker
        
        # If this bucket has a winner passing both filters, track it.
        if best_member is not None and best_momentum is not None:
            bucket_winners[best_member] = best_momentum
    
    if not bucket_winners:
        # No bucket winners with positive momentum: return cash.
        return {}
    
    # Rank bucket winners by skip-week momentum, take top TOP_BUCKETS (3).
    ranked = sorted(bucket_winners.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranked[:TOP_BUCKETS]]
    
    if not selected:
        return {}
    
    # Size by 1/volatility.
    inv_vol_weights = {}
    for ticker in selected:
        bars = market_state.get(ticker) or []
        vol = _volatility(bars)
        if vol < 0.001:
            vol = 0.001
        inv_vol_weights[ticker] = 1.0 / vol
    
    total_inv_vol = sum(inv_vol_weights.values())
    if total_inv_vol <= 0:
        return {}
    
    # Apply volatility brake: reduce exposure if market vol is high.
    exposure_mult = _get_exposure_multiplier(market_state)
    available_for_investment = TARGET_TOTAL_EXPOSURE * exposure_mult
    
    weights = {}
    for ticker in selected:
        w = (inv_vol_weights[ticker] / total_inv_vol) * available_for_investment
        w = min(w, POSITION_CAP)  # cap at 27%
        weights[ticker] = w
    
    return weights


def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance
    _tick += 1
    
    # Parse current portfolio.
    positions = {p["ticker"]: p for p in portfolio_state.get("positions", [])}
    last_prices = portfolio_state.get("last_prices", {})
    equity = portfolio_state.get("cash", cash)
    for tk, pos in positions.items():
        px = last_prices.get(tk) or pos.get("avg_cost", 0)
        equity += pos["quantity"] * px
    
    if equity <= 0:
        equity = 1.0
    
    # Check if rebalance is needed: time-based or drift-based.
    should_rebalance = _tick - _last_rebalance >= REBALANCE_EVERY_TICKS
    if not should_rebalance:
        for tk, pos in positions.items():
            px = last_prices.get(tk) or pos.get("avg_cost", 0)
            if px > 0:
                frac = (pos["quantity"] * px) / equity
                if frac > DRIFT_LIMIT:
                    should_rebalance = True
                    break
    
    if not should_rebalance:
        return []
    
    # Compute target weights.
    targets = _target_weights(market_state)
    
    orders = []
    
    # Sell positions not in targets (or if risk is OFF, sell everything).
    for tk, pos in positions.items():
        if tk not in targets and pos["quantity"] > 0:
            orders.append({"ticker": tk, "side": "sell", "quantity": pos["quantity"]})
    
    # Rebalance existing and add new positions.
    for tk, target_weight in targets.items():
        bars = market_state.get(tk) or []
        if not bars:
            continue
        
        px = float(bars[-1]["close"])
        if px <= 0:
            continue
        
        # Current holding.
        cur_qty = positions.get(tk, {}).get("quantity", 0)
        cur_val = cur_qty * px
        
        # Target value.
        target_val = equity * target_weight
        delta_val = target_val - cur_val
        
        # Skip tiny trades.
        if abs(delta_val) < TRADE_MIN_PCT * equity:
            continue
        
        delta_qty = int(delta_val // px)
        
        if delta_qty > 0:
            orders.append({"ticker": tk, "side": "buy", "quantity": delta_qty})
        elif delta_qty < 0 and cur_qty > 0:
            orders.append({"ticker": tk, "side": "sell", "quantity": min(abs(delta_qty), cur_qty)})
    
    if orders:
        _last_rebalance = _tick
    
    return orders
