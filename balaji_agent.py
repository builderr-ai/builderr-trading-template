"""Ultimate Trading Agent v2 — Drawdown-Optimized Momentum Rotation.

Contest objective: maximize 60-day forward Calmar (return ÷ max drawdown).

Combines the best practices from top performers:
  * sankeerth's drawdown-aware brakes (1-day drop, 10-day vol, panic state)
  * Self-equity drawdown governor (reduce exposure as losses increase)
  * Inverse-volatility weighted position sizing (smooth assets get larger allocations)
  * Constant-risk portfolio volatility scaling (same risk in calm and storm)
  * Multi-regime defensive rotation (crash, chop, uptrend)
  * Asymmetric regime persistence (slow to risk-on, fast to risk-off)
  * Concentrated winners only (top 5-6 by momentum × inverse vol)

No network, no LLM, no leverage (stay well under 1.5x beta cap).
No dependencies outside Python standard library.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# Universe: broad ETFs + sectors + mega-caps
RISK_ON_ETFS = ("SPY", "QQQ", "DIA", "IWM", "SMH")
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE")
# Removed MEGA_CAPS: individual stocks add idiosyncratic risk that hurts Calmar
RISK_ON_NAMES = RISK_ON_ETFS + SECTORS

DEFENSIVE = ("XLP", "XLU", "XLV", "XLI")        # confirmed universe ETFs only; removed GLD/TLT
HARD_BRAKE_BASKET = ("XLP", "XLU", "XLV")        # staples + utilities + health = safest trio

# ─── Core Knobs ───
NAME_CAP = 0.12               # no single position above 12% (was 13%)
SAFE_POSITION_CAP = 0.27      # hard cap at 27% after vol scaling to stay under 30% limit + drift
GROSS_MAX = 0.95              # total gross exposure cap
TOP_N_MOMENTUM = 6            # hold top 6 winners (was 5) — better diversification
REBALANCE_DAYS = 5            # rebalance every N days
DRIFT_LIMIT = 0.28            # force rebalance if any position drifts above this
MIN_TRADE_PCT = 0.012         # skip trades smaller than 1.2% of equity

# ─── Momentum & Vol Calculations ───
MOMENTUM_DAYS = 63            # ~3-month momentum
MOMENTUM_SKIP = 5             # skip last week (avoid recency bias)
TREND_DAYS = 50               # 50-day SMA for trend confirmation
VOL_DAYS = 20                 # 20-day realized vol for sizing
TARGET_PORTFOLIO_VOL = 0.13   # aim for 13% annualized portfolio vol
PORT_VOL_MIN = 0.05           # never scale below 5% portfolio vol
PORT_VOL_MAX = 0.35           # never scale above 35% portfolio vol

# ─── Brake Thresholds (circuit breakers for crashes) ───
BRAKE_1_DAY_DROP = -0.035     # QQQ falls 3.5% in 1 day → hard brake (was -2%, too sensitive)
BRAKE_3_DAY_DROP = -0.060     # QQQ falls 6% in 3 days → hard brake (was -4%)
BRAKE_VOL_10D = 0.50          # 10-day vol exceeds 50% annualized → hard brake (was 40%)
BRAKE_COOLDOWN = 2            # stay defensive for 2 days after brake (was 3)

# ─── Panic State (Daniel-Moskowitz panic signature) ───
PANIC_BEAR_THRESHOLD = -0.10  # SPY down 10% over 6 months
PANIC_VOL_THRESHOLD = 0.30    # SPY vol above 30% annualized
PANIC_GROSS_CAP = 0.25        # cap gross exposure to 25% during panic

# ─── Equity Drawdown Governor (self-limiting mechanism) ───
DD_TIER_1_THRESHOLD = 0.030   # -3.0% → scale to 65% of normal (was -1.5% → 60%)
DD_TIER_2_THRESHOLD = 0.055   # -5.5% → scale to 35% of normal (was -2.5% → 30%)
DD_TIER_3_THRESHOLD = 0.080   # -8.0% → scale to 10% of normal (was -4.0% → 10%)

# ─── Regime Confirmation (asymmetric: slow to risk-on, fast to risk-off) ───
CONFIRM_ENTER_RISK_ON = 2     # need 2 consecutive days of risk-on signal
CONFIRM_LEAVE_RISK_ON = 1     # 1 day of risk-off signal = exit

# ─── Constants ───
_ANN = sqrt(252)              # annualization factor

# ─── Global State ───
_tick = 0
_last_rebalance_date: str | None = None
_brake_cooldown = 0
_peak_equity = 0.0
_pending_regime = None
_pending_regime_count = 0
_current_regime = "soft"      # "hard" (crash) or "soft" (chop) or "risk_on" (uptrend)
_last_targets: dict[str, float] = {}

# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def _closes(bars: list[dict[str, Any]] | None) -> list[float]:
    """Extract closes from bars, validating along the way."""
    if not bars:
        return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
            if close <= 0:
                return []
            out.append(close)
        except (KeyError, TypeError, ValueError):
            return []
    return out


def _sma(closes: list[float], n: int) -> float | None:
    """Simple moving average over last n periods."""
    if len(closes) < n:
        return None
    return mean(closes[-n:])


def _trailing_return(closes: list[float], days: int, skip_days: int = 0) -> float | None:
    """Return over last N days, optionally skipping the most recent days."""
    need = days + skip_days + 1
    if len(closes) < need or closes[0] <= 0:
        return None
    end_idx = -(skip_days + 1) if skip_days > 0 else -1
    end = closes[end_idx]
    start = closes[-(days + skip_days + 1)]
    if start <= 0:
        return None
    return end / start - 1.0


def _realized_vol(closes: list[float], n: int) -> float | None:
    """Annualized realized volatility over last n+1 periods."""
    if len(closes) <= n:
        return None
    window = closes[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        if window[i - 1] <= 0:
            return None
        rets.append(window[i] / window[i - 1] - 1.0)
    if len(rets) < 2:
        return None
    return pstdev(rets) * _ANN


def _current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract current positions from portfolio state."""
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


def _equity(portfolio_state: dict[str, Any], cash: float) -> float:
    """Calculate total equity (cash + position values)."""
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in _current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _latest_bar_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    """Get the date of the most recent bar (for rebalance tracking)."""
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    if ts is None:
        return str(len(bars))
    return str(ts)[:10]


def _days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    """Days elapsed since last rebalance."""
    if _last_rebalance_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_date) - 1


def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Latest price for each ticker."""
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = _closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def _position_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
    """Check if any position has drifted above the drift limit."""
    if total_equity <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in _current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION & CIRCUIT BREAKERS
# ═══════════════════════════════════════════════════════════════════════════

def _check_hard_brake(market_state: dict[str, list[dict[str, Any]]]) -> bool:
    """
    Hard brake triggered by:
      1. QQQ down 2% in 1 day
      2. QQQ down 4% in 3 days
      3. QQQ 10-day vol above 40% annualized
    """
    qqq_bars = market_state.get("QQQ") or []
    qqq = _closes(qqq_bars)
    if len(qqq) < 10:
        return False
    
    # 1-day drop
    ret_1d = _trailing_return(qqq, 1)
    if ret_1d is not None and ret_1d < BRAKE_1_DAY_DROP:
        return True
    
    # 3-day drop
    ret_3d = _trailing_return(qqq, 3)
    if ret_3d is not None and ret_3d < BRAKE_3_DAY_DROP:
        return True
    
    # 10-day vol
    vol_10d = _realized_vol(qqq, 10)
    if vol_10d is not None and vol_10d > BRAKE_VOL_10D:
        return True
    
    return False


def _check_panic_state(market_state: dict[str, list[dict[str, Any]]]) -> bool:
    """
    Panic state (Daniel-Moskowitz signature):
      * SPY down 10% over 6 months AND
      * SPY 20-day vol above 30% annualized
    """
    spy_bars = market_state.get("SPY") or []
    spy = _closes(spy_bars)
    
    if len(spy) < 130:  # ~6 months of trading days
        return False
    
    ret_6m = _trailing_return(spy, 125)  # ~6 months
    vol_20d = _realized_vol(spy, 20)
    
    if (ret_6m is not None and ret_6m < PANIC_BEAR_THRESHOLD and
        vol_20d is not None and vol_20d > PANIC_VOL_THRESHOLD):
        return True
    
    return False


def _equity_drawdown(portfolio_state: dict[str, Any], cash: float) -> float:
    """Current drawdown from peak equity."""
    current = _equity(portfolio_state, cash)
    if _peak_equity <= 0:
        return 0.0
    return max(0.0, (_peak_equity - current) / _peak_equity)


def _gross_scale_for_drawdown(dd: float) -> float:
    """Scale gross exposure based on equity drawdown."""
    if dd < DD_TIER_1_THRESHOLD:
        return 1.0
    elif dd < DD_TIER_2_THRESHOLD:
        return 0.65
    elif dd < DD_TIER_3_THRESHOLD:
        return 0.35
    else:
        return 0.10


# ═══════════════════════════════════════════════════════════════════════════
# TARGET WEIGHT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def _target_weights_defensive(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """
    Defensive allocation: equal-weight across available defensive assets.
    Used when market is in hard brake or panic state.
    """
    avail = [t for t in HARD_BRAKE_BASKET if _closes(market_state.get(t))]
    if not avail:
        return {}
    return {t: 1.0 / len(avail) for t in avail}


def _target_weights_soft(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """
    Soft defensive: rotate to all defensive assets with equal weight.
    Used when risk-on gate is off (trending down).
    """
    avail = [t for t in DEFENSIVE if _closes(market_state.get(t))]
    if not avail:
        return {}
    return {t: 1.0 / len(avail) for t in avail}


def _target_weights_risk_on(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """
    Risk-on: score all risk-on names by momentum × inverse-volatility.
    Size proportional to score (after vol normalization).
    """
    scores: list[tuple[float, str]] = []
    
    for ticker in RISK_ON_NAMES:
        closes = _closes(market_state.get(ticker))
        if len(closes) < MOMENTUM_DAYS + MOMENTUM_SKIP + 1:
            continue
        
        mom = _trailing_return(closes, MOMENTUM_DAYS, skip_days=MOMENTUM_SKIP)
        trend = _sma(closes, TREND_DAYS)
        vol = _realized_vol(closes, VOL_DAYS)
        
        if mom is None or trend is None or vol is None or vol <= 0:
            continue
        
        # Score = momentum × (1 / volatility), with trend confirmation
        trend_ok = closes[-1] > trend
        inverse_vol = 1.0 / (vol + 0.001)  # add small epsilon to avoid division issues
        score = mom * inverse_vol if trend_ok else mom * inverse_vol * 0.5
        
        if score > 0:
            scores.append((score, ticker))
    
    if not scores:
        return {}
    
    # Sort and take top N
    scores.sort(reverse=True)
    winners = [ticker for _, ticker in scores[:TOP_N_MOMENTUM]]
    
    # Allocate proportional to score
    total_score = sum(s for s, _ in [(s, t) for s, t in scores[:TOP_N_MOMENTUM]])
    if total_score <= 0:
        return {}
    
    weights: dict[str, float] = {}
    for ticker in winners:
        score = next(s for s, t in scores if t == ticker)
        weight = (score / total_score) * 0.95  # cap at 95% (leave 5% cash)
        weights[ticker] = min(weight, NAME_CAP)
    
    # Normalize to max 95% gross
    total = sum(weights.values())
    if total > 0.95:
        scale = 0.95 / total
        weights = {t: w * scale for t, w in weights.items()}
    
    return {t: round(w, 6) for t, w in weights.items() if w > 0.001}


def _target_weights(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """
    Determine target weights based on regime + circuit breakers.
    """
    global _current_regime, _pending_regime, _pending_regime_count, _brake_cooldown
    
    # Step 1: Check circuit breakers
    if _check_hard_brake(market_state):
        _brake_cooldown = BRAKE_COOLDOWN
        return _target_weights_defensive(market_state)
    
    if _brake_cooldown > 0:
        _brake_cooldown -= 1
        # Even after cooldown expires, require SPY to be above 10-day SMA before re-entering
        if _brake_cooldown == 0:
            spy = _closes(market_state.get("SPY") or [])
            spy_sma10 = _sma(spy, 10)
            if spy_sma10 is not None and spy and spy[-1] < spy_sma10:
                _brake_cooldown = 1  # extend cooldown 1 more day
        return _target_weights_defensive(market_state)
    
    if _check_panic_state(market_state):
        return _scale_weights_for_panic(_target_weights_soft(market_state))
    
    # Step 2: Determine regime
    spy_bars = market_state.get("SPY") or []
    spy = _closes(spy_bars)
    qqq_bars = market_state.get("QQQ") or []
    qqq = _closes(qqq_bars)
    
    if len(spy) < 50 or len(qqq) < 50:
        return {}
    
    spy_sma = _sma(spy, TREND_DAYS)
    qqq_sma = _sma(qqq, TREND_DAYS)
    qqq_vol = _realized_vol(qqq, 20)
    
    # Risk-on condition: SPY & QQQ above 50-day SMA, QQQ vol reasonable
    risk_on_signal = (
        spy_sma is not None and qqq_sma is not None and qqq_vol is not None and
        spy[-1] > spy_sma and qqq[-1] > qqq_sma and qqq_vol < 0.35
    )
    
    # Asymmetric regime confirmation
    if _pending_regime is None:
        _pending_regime_count = 0
    
    if risk_on_signal:
        if _pending_regime != "risk_on":
            _pending_regime = "risk_on"
            _pending_regime_count = 1
        else:
            _pending_regime_count += 1
        
        if _pending_regime_count >= CONFIRM_ENTER_RISK_ON:
            _current_regime = "risk_on"
    else:
        if _pending_regime != "soft":
            _pending_regime = "soft"
            _pending_regime_count = 1
        else:
            _pending_regime_count += 1
        
        if _pending_regime_count >= CONFIRM_LEAVE_RISK_ON:
            _current_regime = "soft"
    
    # Step 3: Generate targets based on regime
    if _current_regime == "risk_on":
        return _target_weights_risk_on(market_state)
    else:
        return _target_weights_soft(market_state)


def _scale_weights_for_panic(weights: dict[str, float]) -> dict[str, float]:
    """Scale weights down during panic state."""
    if not weights:
        return {}
    scale = PANIC_GROSS_CAP / sum(weights.values())
    return {t: w * scale for t, w in weights.items()}


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO VOLATILITY TARGETING
# ═══════════════════════════════════════════════════════════════════════════

def _portfolio_vol(
    market_state: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
) -> float | None:
    """
    Estimate portfolio volatility as weighted-average of component volatilities.
    (Simplified: ignores correlation, but good enough for risk scaling.)
    """
    if not weights:
        return None
    
    weighted_vol = 0.0
    for ticker, weight in weights.items():
        if weight <= 0:
            continue
        closes = _closes(market_state.get(ticker))
        vol = _realized_vol(closes, VOL_DAYS)
        if vol is None:
            return None
        weighted_vol += weight * vol
    
    return weighted_vol if weighted_vol > 0 else None


def _scale_weights_for_target_vol(
    weights: dict[str, float],
    market_state: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """
    Scale all weights uniformly so portfolio vol = TARGET_PORTFOLIO_VOL.
    Cap individual positions at SAFE_POSITION_CAP (27%) to prevent concentration breach.
    """
    if not weights:
        return {}
    
    port_vol = _portfolio_vol(market_state, weights)
    if port_vol is None or port_vol <= 0:
        return weights
    
    # Scale to target vol
    target_scale = TARGET_PORTFOLIO_VOL / port_vol
    target_scale = max(PORT_VOL_MIN / port_vol, min(PORT_VOL_MAX / port_vol, target_scale))
    
    scaled = {t: w * target_scale for t, w in weights.items()}
    
    # Hard cap: no position exceeds 27% (stays safely under 30% + drift limit)
    capped = {t: min(w, SAFE_POSITION_CAP) for t, w in scaled.items()}
    
    # Renormalize if capping reduced gross below target
    total = sum(capped.values())
    if total > 0 and total < GROSS_MAX:
        # Scale up remaining positions to hit gross target
        scale = GROSS_MAX / total if total > 0 else 1.0
        capped = {t: min(w * scale, SAFE_POSITION_CAP) for t, w in capped.items()}
    
    return capped


# ═══════════════════════════════════════════════════════════════════════════
# ORDER GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    """
    Generate rebalancing orders: sells first, then buys.
    Skip trades smaller than MIN_TRADE_PCT of equity.
    """
    if total_equity <= 0:
        return []
    
    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0
    
    # Sells first: liquidate stale holdings and trim overweight positions
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        
        qty = pos["quantity"]
        current_value = qty * price
        target_value = total_equity * targets.get(ticker, 0.0)
        delta = target_value - current_value
        
        if ticker not in targets:
            # Not in target: sell all if large enough
            if current_value >= min_trade:
                sell_qty = int(qty)
                if sell_qty > 0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                    sell_proceeds += sell_qty * price
        elif delta < -min_trade:
            # Over-weight: trim
            sell_qty = min(int(abs(delta) // price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
    
    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.99)
    
    # Buys second: allocate remaining cash to underweight positions
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
    
    return orders[:45]  # Enforce 50-order limit

# ═══════════════════════════════════════════════════════════════════════════
# MAIN DECISION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """
    Main decision function called once per day.
    Returns a list of buy/sell orders.
    """
    global _tick, _last_rebalance_date, _peak_equity
    
    _tick += 1
    
    if not market_state:
        return []
    
    # Get current state
    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []
    
    total_equity = _equity(portfolio_state, cash)
    if total_equity <= 0:
        return []
    
    # Track peak equity for drawdown governor
    if _peak_equity <= 0 or total_equity > _peak_equity:
        _peak_equity = total_equity
    
    # Check if we should rebalance
    days_since = _days_since_rebalance(market_state)
    drifted = _position_drifted(portfolio_state, total_equity)
    should_rebalance = (
        _last_rebalance_date is None
        or days_since is None
        or days_since >= REBALANCE_DAYS
        or drifted
    )
    
    if not should_rebalance:
        return []
    
    # Get target weights based on regime
    targets = _target_weights(market_state)
    if not targets:
        return []
    
    # Apply portfolio vol scaling (constant risk dosing)
    targets = _scale_weights_for_target_vol(targets, market_state)
    if not targets:
        return []
    
    # Apply drawdown governor (reduce exposure as losses mount)
    dd = _equity_drawdown(portfolio_state, cash)
    dd_scale = _gross_scale_for_drawdown(dd)
    targets = {t: w * dd_scale for t, w in targets.items()}
    
    # Generate rebalancing orders
    prices = _market_prices(market_state)
    positions = _current_positions(portfolio_state)
    orders = _orders_to_rebalance(targets, positions, total_equity, prices, cash)
    
    if orders:
        _last_rebalance_date = latest_date
        _last_targets = targets
    
    return orders