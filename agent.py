<<<<<<< Updated upstream
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
=======
import numpy as np
import pandas as pd
import warnings

# Force-silence any environment warnings for the online server
warnings.filterwarnings("ignore")

class InstitutionalAlphaEngine:
    @staticmethod
    def calculate_hurst_exponent(close_prices: np.ndarray, max_lags: int =
     10) -> float:
>>>>>>> Stashed changes
        try:
            if len(close_prices) < max_lags * 2: 
                return 0.50
            lags = np.arange(2, max_lags)
            variances = []
            for lag in lags:
                diffs = close_prices[lag:] - close_prices[:-lag]
                std_dev = np.std(diffs)
                variances.append(std_dev if std_dev > 0 else 1e-6)
            poly = np.polyfit(np.log(lags), np.log(variances), 1)
            return float(np.clip(poly[0] * 2.0, 0.0, 1.0))
        except Exception: 
            return 0.50

<<<<<<< Updated upstream

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
=======
    @classmethod
    def evaluate_asset(cls, df: pd.DataFrame) -> dict:
        metrics = {"signal": "HOLD", "alpha_score": 0.0, "atr_pct": 0.01, "price": 0.0}
>>>>>>> Stashed changes
        try:
            if df is None or len(df) < 30:
                return metrics
            
            col_map = {str(c).lower().strip(): c for c in df.columns}
            close_key = col_map.get('close', col_map.get('price', df.columns[-1]))
            high_key = col_map.get('high', close_key)
            low_key = col_map.get('low', close_key)
            
            closes = df[close_key].to_numpy(dtype=float)
            highs = df[high_key].to_numpy(dtype=float)
            lows = df[low_key].to_numpy(dtype=float)
            
            current_price = closes[-1]
            metrics["price"] = current_price
            prices_series = pd.Series(closes)

            ema_9 = prices_series.ewm(span=9, adjust=False).mean().to_numpy()[-1]
            ema_50 = prices_series.ewm(span=50, adjust=False).mean().to_numpy()[-1]
            ema_100 = prices_series.ewm(span=100, adjust=False).mean().to_numpy()[-1] if len(closes) >= 100 else ema_50

            hl = highs - lows
            hc = np.abs(highs - np.roll(closes, 1))
            lc = np.abs(lows - np.roll(closes, 1))
            hc[0], lc[0] = 0, 0
            atr = pd.Series(np.maximum(hl, np.maximum(hc, lc))).rolling(window=14).mean().to_numpy()[-1]
            if np.isnan(atr) or atr <= 0: atr = current_price * 0.01
            metrics["atr_pct"] = float(atr / current_price)

            momentum = prices_series.pct_change().tail(5).mean()
            if np.isnan(momentum): momentum = 0.0

            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-6))))
            rsi_val = rsi.to_numpy()[-1]
            if np.isnan(rsi_val): rsi_val = 50.0

            hurst_val = cls.calculate_hurst_exponent(closes[-30:])

            if hurst_val > 0.55:  # Trend
                if current_price > ema_100 and momentum > 0 and rsi_val < 75:
                    if current_price > ema_9:
                        metrics["signal"] = "BUY"
                        metrics["alpha_score"] = float(momentum * 100.0)
                elif current_price < ema_50 or rsi_val > 80:
                    metrics["signal"] = "SELL"

            elif hurst_val < 0.45:  # Mean-Reversion
                rolling_mean = prices_series.rolling(window=20).mean().to_numpy()[-1]
                lower_floor = rolling_mean - (1.5 * atr)
                upper_ceiling = rolling_mean + (1.5 * atr)

                if current_price <= lower_floor or rsi_val < 32:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(100.0 - rsi_val)
                elif current_price >= upper_ceiling or rsi_val > 68:
                    metrics["signal"] = "SELL"
            
            else:  # Pivot
                if rsi_val < 26:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(50.0 - rsi_val)
                elif rsi_val > 74:
                    metrics["signal"] = "SELL"

            return metrics
        except Exception:
            return metrics

def decide(market_state: dict, portfolio_state: dict, cash: float) -> list:
    try:
<<<<<<< Updated upstream
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in _positions(portfolio_state).items():
        price = last_prices.get(ticker)
        if price:
            total += pos["quantity"] * float(price)
    return max(total, 0.0)
=======
        orders = []
        if not market_state:
            return orders
>>>>>>> Stashed changes

        # 1. Total Portfolio Accounting & Real-Time Exposure Tracking
        total_portfolio_value = float(cash)
        current_gross_exposure = 0.0
        current_positions = {}
        active_prices = {}

<<<<<<< Updated upstream
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
=======
        for ticker, p_val in portfolio_state.items():
            qty_held = p_val.get('quantity', p_val.get('qty', 0)) if isinstance(p_val, dict) else p_val
            if qty_held and float(qty_held) > 0:
                current_positions[ticker] = float(qty_held)

        for ticker, bars in market_state.items():
            if bars:
                try:
                    last_bar = bars[-1]
                    price = last_bar.get("close", last_bar.get("price", last_bar.get("open", 0)))
                    active_prices[ticker] = float(price)
                except Exception:
                    continue

        # Calculate exact Net Asset Value (NAV) and starting Gross Exposure
        for ticker, qty in current_positions.items():
            if ticker in active_prices:
                pos_value = qty * active_prices[ticker]
                total_portfolio_value += pos_value
                current_gross_exposure += pos_value

        available_cash = float(cash)
        buy_candidates = []

        # 2. Extract Immediate Closures to Free Up Leverage Space First
        for ticker, bars in market_state.items():
            if not bars or len(bars) < 30:
                continue
            
            try:
                df = pd.DataFrame(bars)
                analysis = InstitutionalAlphaEngine.evaluate_asset(df)
                current_price = active_prices.get(ticker, analysis["price"])
                
                if current_price <= 0:
                    continue

                qty_held = current_positions.get(ticker, 0.0)

                if analysis["signal"] == "SELL" and qty_held > 0:
                    orders.append({"ticker": str(ticker), "side": "sell", "quantity": int(qty_held)})
                    # Credit exposure and cash pools back immediately for the current calculation turn
                    current_gross_exposure -= (qty_held * current_price)
                    available_cash += (qty_held * current_price)
                
                elif analysis["signal"] == "BUY":
                    buy_candidates.append({
                        "ticker": str(ticker),
                        "price": float(current_price),
                        "score": float(analysis["alpha_score"]),
                        "atr_pct": float(analysis["atr_pct"]),
                        "qty_held": float(qty_held)
                    })
            except Exception:
                continue

        # 3. Dynamic Leverage Allocation Layer
        # Hard ceiling: Maximum total exposure allowed across the whole portfolio is 1.42x NAV
        max_absolute_exposure = total_portfolio_value * 1.42
        
        # Sort opportunities by highest calculated Alpha score
        buy_candidates = sorted(buy_candidates, key=lambda x: x["score"], reverse=True)[:5]

        for candidate in buy_candidates:
            try:
                # Continuous real-time check of remaining room below the leverage ceiling
                remaining_leverage_room = max_absolute_exposure - current_gross_exposure
                if remaining_leverage_room <= 0:
                    break

                ticker = candidate["ticker"]
                price = candidate["price"]
                atr_pct = candidate["atr_pct"]
                qty_held = candidate["qty_held"]

                # Inverse-Volatility sizing base metric
                base_allocation = total_portfolio_value * (0.02 / (atr_pct + 1e-5))
                
                # Interlocking Guardrails: Protect cash, individual concentration limits, and overall leverage room
                target_spend = min(base_allocation, total_portfolio_value * 0.15, available_cash * 0.35, remaining_leverage_room)
                max_allowed_spend = (total_portfolio_value * 0.24) - (qty_held * price)
                
                final_spend = min(target_spend, max_allowed_spend)
                if final_spend > 0:
                    quantity = int(final_spend / price)
                    if quantity > 0:
                        orders.append({"ticker": ticker, "side": "buy", "quantity": quantity})
                        available_cash -= (quantity * price)
                        current_gross_exposure += (quantity * price)
            except Exception:
                continue

        return orders

    except Exception:
        return []
>>>>>>> Stashed changes
