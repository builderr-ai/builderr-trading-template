"""Ultimate Trading Agent v2 — Drawdown-Optimized Momentum Rotation."""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

RISK_ON_ETFS = ("SPY", "QQQ", "DIA", "IWM", "SMH")
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE")
RISK_ON_NAMES = RISK_ON_ETFS + SECTORS

DEFENSIVE = ("XLP", "XLU", "XLV", "XLI")
HARD_BRAKE_BASKET = ("XLP", "XLU", "XLV")

NAME_CAP = 0.12
SAFE_POSITION_CAP = 0.27
GROSS_MAX = 0.95
TOP_N_MOMENTUM = 6
REBALANCE_DAYS = 5
DRIFT_LIMIT = 0.28
MIN_TRADE_PCT = 0.012

MOMENTUM_DAYS = 63
MOMENTUM_SKIP = 5
TREND_DAYS = 50
VOL_DAYS = 20
TARGET_PORTFOLIO_VOL = 0.13
PORT_VOL_MIN = 0.05
PORT_VOL_MAX = 0.35

BRAKE_1_DAY_DROP = -0.035
BRAKE_3_DAY_DROP = -0.060
BRAKE_VOL_10D = 0.50
BRAKE_COOLDOWN = 2

PANIC_BEAR_THRESHOLD = -0.10
PANIC_VOL_THRESHOLD = 0.30
PANIC_GROSS_CAP = 0.25

DD_TIER_1_THRESHOLD = 0.020
DD_TIER_2_THRESHOLD = 0.040
DD_TIER_3_THRESHOLD = 0.070

CONFIRM_ENTER_RISK_ON = 2
CONFIRM_LEAVE_RISK_ON = 1
CONFIRM_ENTER_AFTER_BRAKE = 3  # need more confirmations to re-enter risk-on after a hard brake

_ANN = sqrt(252)

_tick = 0
_last_rebalance_date = None
_brake_cooldown = 0
_peak_equity = 0.0
_pending_regime = None
_pending_regime_count = 0
_current_regime = "soft"
_last_targets = {}
_recently_braked = False   # set True when a hard brake fires; cleared once we go risk-on

def _closes(bars):
    if not bars:
        return []
    out = []
    for bar in bars:
        try:
            close = float(bar["close"])
            if close <= 0:
                return []
            out.append(close)
        except (KeyError, TypeError, ValueError):
            return []
    return out

def _sma(closes, n):
    if len(closes) < n:
        return None
    return mean(closes[-n:])

def _trailing_return(closes, days, skip_days=0):
    need = days + skip_days + 1
    if len(closes) < need or closes[0] <= 0:
        return None
    end_idx = -(skip_days + 1) if skip_days > 0 else -1
    end = closes[end_idx]
    start = closes[-(days + skip_days + 1)]
    if start <= 0:
        return None
    return end / start - 1.0

def _realized_vol(closes, n):
    if len(closes) <= n:
        return None
    window = closes[-(n + 1):]
    rets = []
    for i in range(1, len(window)):
        if window[i - 1] <= 0:
            return None
        rets.append(window[i] / window[i - 1] - 1.0)
    if len(rets) < 2:
        return None
    return pstdev(rets) * _ANN

def _current_positions(portfolio_state):
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

def _equity(portfolio_state, cash):
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

def _latest_bar_date(market_state):
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    if ts is None:
        return str(len(bars))
    return str(ts)[:10]

def _days_since_rebalance(market_state):
    if _last_rebalance_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_date) - 1

def _market_prices(market_state):
    prices = {}
    for ticker, bars in market_state.items():
        cs = _closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices

def _position_drifted(portfolio_state, total_equity):
    if total_equity <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in _current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price <= 0:
            continue
        current_weight = pos["quantity"] * price / total_equity
        target_weight = _last_targets.get(ticker, 0.0)
        # Trigger only if actual drift FROM TARGET exceeds limit (not absolute weight)
        if abs(current_weight - target_weight) > DRIFT_LIMIT:
            return True
        # Also catch absolute breach as a hard safety (e.g. orphaned large position)
        if current_weight > 0.30:
            return True
    return False

def _check_hard_brake(market_state):
    qqq_bars = market_state.get("QQQ") or []
    qqq = _closes(qqq_bars)
    if len(qqq) < 10:
        return False
    ret_1d = _trailing_return(qqq, 1)
    if ret_1d is not None and ret_1d < BRAKE_1_DAY_DROP:
        return True
    ret_3d = _trailing_return(qqq, 3)
    if ret_3d is not None and ret_3d < BRAKE_3_DAY_DROP:
        return True
    vol_10d = _realized_vol(qqq, 10)
    if vol_10d is not None and vol_10d > BRAKE_VOL_10D:
        return True
    return False

def _check_panic_state(market_state):
    spy_bars = market_state.get("SPY") or []
    spy = _closes(spy_bars)
    if len(spy) < 130:
        return False
    ret_6m = _trailing_return(spy, 125)
    vol_20d = _realized_vol(spy, 20)
    if (ret_6m is not None and ret_6m < PANIC_BEAR_THRESHOLD and
        vol_20d is not None and vol_20d > PANIC_VOL_THRESHOLD):
        return True
    return False

def _check_active_decline(market_state):
    """Detect 3+ consecutive red days in SPY — catch declines as they happen."""
    spy_bars = market_state.get("SPY") or []
    spy = _closes(spy_bars)
    if len(spy) < 4:
        return False
    if spy[-1] < spy[-2] and spy[-2] < spy[-3] and spy[-3] < spy[-4]:
        return True
    return False

def _equity_drawdown(portfolio_state, cash):
    current = _equity(portfolio_state, cash)
    if _peak_equity <= 0:
        return 0.0
    return max(0.0, (_peak_equity - current) / _peak_equity)

def _gross_scale_for_drawdown(dd):
    if dd < DD_TIER_1_THRESHOLD:
        return 1.0
    elif dd < DD_TIER_2_THRESHOLD:
        return 0.65
    elif dd < DD_TIER_3_THRESHOLD:
        return 0.35
    else:
        return 0.10

def _target_weights_defensive(market_state):
    avail = [t for t in HARD_BRAKE_BASKET if _closes(market_state.get(t))]
    if not avail:
        return {}
    return {t: 1.0 / len(avail) for t in avail}

def _target_weights_soft(market_state):
    avail = [t for t in DEFENSIVE if _closes(market_state.get(t))]
    if not avail:
        return {}
    return {t: 1.0 / len(avail) for t in avail}

def _target_weights_risk_on(market_state):
    scores = []
    for ticker in RISK_ON_NAMES:
        closes = _closes(market_state.get(ticker))
        if len(closes) < MOMENTUM_DAYS + MOMENTUM_SKIP + 1:
            continue
        mom = _trailing_return(closes, MOMENTUM_DAYS, skip_days=MOMENTUM_SKIP)
        trend = _sma(closes, TREND_DAYS)
        vol = _realized_vol(closes, VOL_DAYS)
        if mom is None or trend is None or vol is None or vol <= 0:
            continue
        trend_ok = closes[-1] > trend
        inverse_vol = 1.0 / (vol + 0.001)
        score = mom * inverse_vol if trend_ok else mom * inverse_vol * 0.5
        if score > 0:
            scores.append((score, ticker))
    if not scores:
        return {}
    scores.sort(reverse=True)
    winners = [ticker for _, ticker in scores[:TOP_N_MOMENTUM]]
    total_score = sum(s for s, _ in [(s, t) for s, t in scores[:TOP_N_MOMENTUM]])
    if total_score <= 0:
        return {}
    weights = {}
    for ticker in winners:
        score = next(s for s, t in scores if t == ticker)
        weight = (score / total_score) * 0.95
        weights[ticker] = min(weight, NAME_CAP)
    total = sum(weights.values())
    if total > 0.95:
        scale = 0.95 / total
        weights = {t: w * scale for t, w in weights.items()}
    return {t: round(w, 6) for t, w in weights.items() if w > 0.001}

def _target_weights(market_state):
    global _current_regime, _pending_regime, _pending_regime_count, _brake_cooldown, _recently_braked
    if _check_hard_brake(market_state):
        _brake_cooldown = BRAKE_COOLDOWN
        _recently_braked = True
        _current_regime = "soft"
        _pending_regime = None
        _pending_regime_count = 0
        return _target_weights_defensive(market_state)
    if _brake_cooldown > 0:
        _brake_cooldown -= 1
        if _brake_cooldown == 0:
            spy = _closes(market_state.get("SPY") or [])
            spy_sma10 = _sma(spy, 10)
            if spy_sma10 is not None and spy and spy[-1] < spy_sma10:
                _brake_cooldown = 1
        return _target_weights_defensive(market_state)
    if _check_panic_state(market_state):
        return _scale_weights_for_panic(_target_weights_soft(market_state))
    if _check_active_decline(market_state):
        return _target_weights_soft(market_state)
    spy_bars = market_state.get("SPY") or []
    spy = _closes(spy_bars)
    qqq_bars = market_state.get("QQQ") or []
    qqq = _closes(qqq_bars)
    if len(spy) < 50 or len(qqq) < 50:
        return {}
    spy_sma = _sma(spy, TREND_DAYS)
    qqq_sma = _sma(qqq, TREND_DAYS)
    qqq_vol = _realized_vol(qqq, 20)
    risk_on_signal = (
        spy_sma is not None and qqq_sma is not None and qqq_vol is not None and
        spy[-1] > spy_sma and qqq[-1] > qqq_sma and qqq_vol < 0.35
    )
    if _pending_regime is None:
        _pending_regime_count = 0
    if risk_on_signal:
        if _pending_regime != "risk_on":
            _pending_regime = "risk_on"
            _pending_regime_count = 1
        else:
            _pending_regime_count += 1
        required = CONFIRM_ENTER_AFTER_BRAKE if _recently_braked else CONFIRM_ENTER_RISK_ON
        if _pending_regime_count >= required:
            _current_regime = "risk_on"
            _recently_braked = False
    else:
        if _pending_regime != "soft":
            _pending_regime = "soft"
            _pending_regime_count = 1
        else:
            _pending_regime_count += 1
        if _pending_regime_count >= CONFIRM_LEAVE_RISK_ON:
            _current_regime = "soft"
    if _current_regime == "risk_on":
        return _target_weights_risk_on(market_state)
    else:
        return _target_weights_soft(market_state)

def _scale_weights_for_panic(weights):
    if not weights:
        return {}
    scale = PANIC_GROSS_CAP / sum(weights.values())
    return {t: w * scale for t, w in weights.items()}

def _portfolio_vol(market_state, weights):
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

def _scale_weights_for_target_vol(weights, market_state):
    if not weights:
        return {}
    port_vol = _portfolio_vol(market_state, weights)
    if port_vol is None or port_vol <= 0:
        return weights
    target_scale = TARGET_PORTFOLIO_VOL / port_vol
    target_scale = max(PORT_VOL_MIN / port_vol, min(PORT_VOL_MAX / port_vol, target_scale))
    scaled = {t: w * target_scale for t, w in weights.items()}
    capped = {t: min(w, SAFE_POSITION_CAP) for t, w in scaled.items()}
    return capped

def _orders_to_rebalance(targets, positions, total_equity, prices, cash_available):
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
            if current_value >= min_trade:
                sell_qty = int(qty)
                if sell_qty > 0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                    sell_proceeds += sell_qty * price
        elif delta < -min_trade:
            sell_qty = min(int(abs(delta) // price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.99)
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

def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance_date, _peak_equity, _last_targets
    _tick += 1
    if not market_state:
        return []
    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []
    total_equity = _equity(portfolio_state, cash)
    if total_equity <= 0:
        return []
    if _peak_equity <= 0 or total_equity > _peak_equity:
        _peak_equity = total_equity
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
    targets = _target_weights(market_state)
    if not targets:
        return []
    targets = _scale_weights_for_target_vol(targets, market_state)
    if not targets:
        return []
    dd = _equity_drawdown(portfolio_state, cash)
    dd_scale = _gross_scale_for_drawdown(dd)
    targets = {t: w * dd_scale for t, w in targets.items()}
    prices = _market_prices(market_state)
    positions = _current_positions(portfolio_state)
    orders = _orders_to_rebalance(targets, positions, total_equity, prices, cash)
    if orders:
        _last_rebalance_date = latest_date
        _last_targets = targets
    return orders