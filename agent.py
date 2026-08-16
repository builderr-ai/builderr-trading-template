"""
Momentum + Risk-Off Rotation bot — builderr trading challenge Round 2.

Built on top of the original vishwas_agent.py codebase. Four targeted
additions from Arnav's Round 1 winning algorithm to fix the downside loss
problem while keeping the upside intact:

  1. DRAWDOWN TAPER: tracks peak equity from start of scoring. At -6%
     drawdown, cuts target exposure to 50%. At -10%, cuts to 25%. This
     is the single biggest fix for large downside losses — it scales back
     automatically as the portfolio bleeds, not after it's already too late.

  2. FASTER CRASH BRAKE: adds a 3-day QQQ return check alongside the
     existing vol-spike switch. If QQQ drops more than 4% in 3 days, the
     bot flattens to cash immediately — this catches sharp moves before
     realized volatility even has time to spike.

  3. RE-ENTRY HYSTERESIS: after going to cash, requires QQQ to clearly
     reclaim its SMA (price > SMA * 1.01, a 1% buffer) before rebuying.
     Previously the bot could re-enter the moment QQQ touched the SMA from
     below, which meant buying right at resistance in a choppy market.

  4. CHURN REDUCTION: MIN_TRADE_PCT raised from 1% to 3% of equity. The
     previous version was generating ~19 trades in 15 days on 8 names —
     that's whipsaw from names flickering in/out of the top-4 on tiny
     score differences. 3% threshold absorbs that noise without missing
     real rebalances.

Everything else from the original codebase is unchanged:
  - Same basket (NVDA, AMD, MU, MRVL, AVGO, SMH, AAPL, MSFT)
  - Same slow SMA switch (QQQ below 100-day → cash)
  - Same vol-spike switch (20-day vol > 1.8x 100-day baseline → cash)
  - Same vol-ratio throttle on total exposure
  - Same weekly rebalance cadence
  - Same position cap (24%, under the 30% rule)
  - Same gross cap (~96% deployed, under the 1.5x leverage cap)
"""

from __future__ import annotations
from statistics import mean, pstdev
from typing import Any

# ---- Parameters -----------------------------------------------------------
BASKET = ("NVDA", "AMD", "MU", "MRVL", "AVGO", "SMH", "AAPL", "MSFT")
MARKET_TICKER = "QQQ"
TOP_K = 4
MAX_WEIGHT = 0.24
MOM_LOOKBACK = 63
TREND_SMA = 50
MARKET_SMA = 100
VOL_LOOKBACK = 20
VOL_SPIKE_MULT = 1.8
MIN_WEIGHT_MULT = 0.5
REBALANCE_EVERY_DAYS = 5
MIN_TRADE_PCT = 0.03          # raised from 0.01 → reduces whipsaw churn

# ---- New: drawdown taper parameters (from Arnav's breakdown) --------------
DD_HALF = -0.06               # at -6% from peak → cut exposure to 50%
DD_LOCK = -0.10               # at -10% from peak → cut exposure to 25%
TAPER_HALF = 0.50
TAPER_LOCK = 0.25

# ---- New: faster crash brake ----------------------------------------------
FAST_CRASH_LOOKBACK = 3       # days
FAST_CRASH_THRESHOLD = -0.04  # -4% over 3 days → go to cash immediately

# ---- New: re-entry hysteresis --------------------------------------------
REENTRY_BUFFER = 0.01         # QQQ must be 1% above SMA to re-enter from cash

# ---- Persistent state -----------------------------------------------------
_last_rebalance_date: str | None = None
_peak_equity: float = 0.0            # tracks the high-water mark for drawdown taper
_in_cash_state: bool = False         # tracks whether we're in a risk-off cash state


# ---- Small helpers (unchanged from original) ------------------------------
def _closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out = []
    for b in bars:
        try:
            c = float(b["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if c <= 0:
            return []
        out.append(c)
    return out


def _sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return mean(values[-n:])


def _momentum(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    if start <= 0:
        return None
    return values[-1] / start - 1.0


def _daily_returns(values: list[float]) -> list[float]:
    return [values[i] / values[i - 1] - 1.0
            for i in range(1, len(values)) if values[i - 1] > 0]


def _realized_vol(values: list[float], n: int) -> float | None:
    rets = _daily_returns(values)
    if len(rets) < n:
        return None
    window = rets[-n:]
    if len(window) < 2:
        return None
    return pstdev(window) * (252 ** 0.5)


def _vol_is_spiking(values: list[float]) -> bool:
    current_vol = _realized_vol(values, VOL_LOOKBACK)
    baseline_vol = _realized_vol(values, MARKET_SMA)
    if current_vol is None or baseline_vol is None or baseline_vol <= 0:
        return False
    return current_vol > baseline_vol * VOL_SPIKE_MULT


def _fast_crash(values: list[float]) -> bool:
    """NEW: 3-day return crash check — catches sharp drops before vol spikes."""
    r = _momentum(values, FAST_CRASH_LOOKBACK)
    return r is not None and r < FAST_CRASH_THRESHOLD


def _bar_date(market_state: dict, ticker: str) -> str | None:
    bars = market_state.get(ticker) or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    return str(ts)[:10] if ts is not None else str(len(bars))


def _days_since(market_state: dict, ticker: str, last_date: str | None) -> int | None:
    if last_date is None:
        return None
    bars = market_state.get(ticker) or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if last_date not in dates:
        return None
    return len(dates) - dates.index(last_date) - 1


def _positions(portfolio_state: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty = float(raw.get("quantity", 0.0))
        except (TypeError, ValueError):
            continue
        if qty > 0:
            out[ticker] = {"quantity": qty}
    return out


def _equity(portfolio_state: dict, cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in _positions(portfolio_state).items():
        price = last_prices.get(ticker)
        if price:
            total += pos["quantity"] * float(price)
    return max(total, 0.0)


# ---- NEW: drawdown taper multiplier ---------------------------------------
def _taper_mult(current_equity: float) -> float:
    """
    Returns a multiplier (0.25-1.0) based on drawdown from peak equity.
    1.0 = no taper (normal), 0.5 = half exposure, 0.25 = quarter exposure.
    Peak equity is updated every day we're above the previous high.
    """
    global _peak_equity
    if current_equity > _peak_equity:
        _peak_equity = current_equity
    if _peak_equity <= 0:
        return 1.0
    dd = (current_equity / _peak_equity) - 1.0
    if dd <= DD_LOCK:
        return TAPER_LOCK    # -10% or worse → 25% exposure
    elif dd <= DD_HALF:
        return TAPER_HALF    # -6% to -10% → 50% exposure
    else:
        return 1.0           # less than -6% → full exposure


# ---- Core signal: what to hold and how much (with taper applied) ----------
def target_weights(
    market_state: dict,
    taper: float = 1.0,
) -> dict[str, float]:
    global _in_cash_state

    qqq = _closes(market_state.get(MARKET_TICKER))
    if len(qqq) < MARKET_SMA:
        return {}

    market_sma = _sma(qqq, MARKET_SMA)
    if market_sma is None:
        return {}

    # ---- NEW: re-entry hysteresis -----------------------------------------
    # If we're in cash, require QQQ to be 1% above SMA to re-enter.
    # If we're invested, only exit when QQQ drops below SMA (no buffer).
    if _in_cash_state:
        slow_risk_on = qqq[-1] > market_sma * (1.0 + REENTRY_BUFFER)
    else:
        slow_risk_on = qqq[-1] > market_sma

    # ---- Fast switches: vol spike OR 3-day crash (either forces cash) ------
    fast_risk_off = _vol_is_spiking(qqq) or _fast_crash(qqq)

    if not slow_risk_on or fast_risk_off:
        _in_cash_state = True
        return {}

    _in_cash_state = False

    # ---- Momentum ranking (unchanged) -------------------------------------
    scored: list[tuple[float, str]] = []
    for ticker in BASKET:
        values = _closes(market_state.get(ticker))
        if len(values) <= MOM_LOOKBACK or len(values) < TREND_SMA:
            continue
        mom = _momentum(values, MOM_LOOKBACK)
        trend = _sma(values, TREND_SMA)
        if mom is None or trend is None:
            continue
        if values[-1] <= trend:
            continue
        scored.append((mom, ticker))

    scored.sort(reverse=True)
    winners = [t for _, t in scored[:TOP_K]]
    if not winners:
        return {}

    # ---- Vol-ratio throttle (unchanged) -----------------------------------
    ratios: list[float] = []
    for t in winners:
        values = _closes(market_state.get(t))
        current_vol = _realized_vol(values, VOL_LOOKBACK)
        own_baseline = _realized_vol(values, MARKET_SMA)
        if current_vol is not None and own_baseline is not None and own_baseline > 0:
            ratios.append(current_vol / own_baseline)

    if ratios:
        avg_ratio = mean(ratios)
        vol_throttle = max(MIN_WEIGHT_MULT, min(1.0, 1.0 / avg_ratio)) if avg_ratio > 0 else 1.0
    else:
        vol_throttle = 1.0

    # ---- NEW: apply drawdown taper on top of vol throttle -----------------
    combined_throttle = vol_throttle * taper
    base_slice = (0.96 / len(winners)) * combined_throttle
    return {t: min(MAX_WEIGHT, base_slice) for t in winners}


# ---- Turn target weights into orders (unchanged except MIN_TRADE_PCT) -----
def _orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict]:
    if total_equity <= 0:
        return []
    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict] = []
    sell_proceeds = 0.0

    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if not price or price <= 0:
            continue
        current_value = pos["quantity"] * price
        target_value = total_equity * targets.get(ticker, 0.0)
        delta = target_value - current_value
        if ticker not in targets:
            qty = int(pos["quantity"])
            if qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
                sell_proceeds += qty * price
        elif delta < -min_trade:
            qty = min(int(abs(delta) // price), int(pos["quantity"]))
            if qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
                sell_proceeds += qty * price

    spendable = max(cash_available, 0.0) + sell_proceeds * 0.98

    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if not price or price <= 0:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = total_equity * weight
        delta = target_value - current_value
        if delta < min_trade:
            continue
        buy_value = min(delta, spendable)
        qty = int(buy_value // price)
        if qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": qty})
            spendable -= qty * price

    return orders[:40]


# ---- Entry point ----------------------------------------------------------
def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    """Called once per day. Returns a list of long-only orders."""
    global _last_rebalance_date, _peak_equity, _in_cash_state

    if not market_state:
        return []

    latest_date = _bar_date(market_state, MARKET_TICKER)
    if latest_date is None:
        return []

    qqq = _closes(market_state.get(MARKET_TICKER))

    # Fast risk-off check — runs every day regardless of rebalance schedule
    fast_risk_off_today = (
        (_vol_is_spiking(qqq) or _fast_crash(qqq)) if qqq else False
    )

    days_since = _days_since(market_state, MARKET_TICKER, _last_rebalance_date)
    scheduled_rebalance = (
        _last_rebalance_date is None
        or days_since is None
        or days_since >= REBALANCE_EVERY_DAYS
    )

    positions = _positions(portfolio_state)
    holding_anything = len(positions) > 0

    should_act = scheduled_rebalance or (fast_risk_off_today and holding_anything)
    if not should_act:
        return []

    # ---- NEW: compute taper from current drawdown -------------------------
    current_equity = _equity(portfolio_state, cash)
    taper = _taper_mult(current_equity)

    prices = {t: _closes(b)[-1] for t, b in market_state.items() if _closes(b)}

    # Pass taper into target_weights so it scales exposure down during drawdowns
    targets = target_weights(market_state, taper=taper)

    orders = _orders_to_rebalance(targets, positions, current_equity, prices, cash)

    if scheduled_rebalance:
        _last_rebalance_date = latest_date
    return orders
