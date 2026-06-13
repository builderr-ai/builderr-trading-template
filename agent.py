"""Builderr Trading v0 agent built for forward Calmar robustness.

Adapted and optimized from Sankeerth's "Drawdown-Aware Defensive Trend" strategy:
  * sizes positions by inverse realized volatility and scales to hit a vol target.
  * de-risks quickly via 1-day/3-day drop and 10-day volatility brakes on QQQ.
  * caps drawdowns using a self-equity drawdown governor.
"""
from __future__ import annotations

from statistics import pstdev

# -----------------------------------------------------------------------------
# universe
# -----------------------------------------------------------------------------
RISK_ON_ETFS = ("SPY", "QQQ", "SMH", "XLK", "XLV", "XLY", "XLC", "XLF")
LARGE_CAP = (
    "NVDA", "AMD", "AVGO", "MU", "MRVL",
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "PLTR", "TSLA",
)
RISK_ON = RISK_ON_ETFS + LARGE_CAP
DEFENSIVE = ("XLP", "XLU", "XLE", "GLD", "TLT")
HARD_BRAKE_BASKET = ("XLP", "XLU", "GLD")
SOFT_DEFENSIVE = ("XLP", "XLU")

# -----------------------------------------------------------------------------
# knobs
# -----------------------------------------------------------------------------
NAME_CAP = 0.12
GROSS_MAX = 0.95
REBALANCE_EVERY = 5
DEAD_BAND = 0.03
ADAPTIVE_THRESHOLD = 0.005

VOL_LOOKBACK = 20
LONG_VOL_LOOKBACK = 60
TARGET_PORT_VOL = 0.15
PORT_VOL_FLOOR = 0.05
PORT_VOL_CEILING = 0.50

MOMENTUM_LOOKBACK = 63
MOMENTUM_SKIP = 5
NAME_TREND_DAYS = 50
TOP_N_RISKON = 7


# brake
BRAKE_R1 = -0.020
BRAKE_R3 = -0.040
BRAKE_VOL_10D = 0.40
BRAKE_COOLDOWN = 3

# panic state (Daniel-Moskowitz)
PANIC_BEAR_RET = -0.10
PANIC_VOL = 0.30
PANIC_GROSS_CAP = 0.25

# asymmetric regime persistence
CONFIRM_ENTER_RISKON = 2
CONFIRM_LEAVE_RISKON = 1

# self-DD governor
DD_TIER_1 = 0.015
DD_TIER_2 = 0.025
DD_TIER_3 = 0.040

# blend in risk-on
RISKON_RISK_PCT = 0.85
RISKON_DEF_PCT = 0.15

_ANN = 252 ** 0.5

# -----------------------------------------------------------------------------
# state
# -----------------------------------------------------------------------------
_tick = 0
_last_rebalance = -10**9
_brake_cooldown = 0
_peak_equity = 0.0
_pending_regime = None
_pending_count = 0
_current_regime = "soft"


# -----------------------------------------------------------------------------
# primitives
# -----------------------------------------------------------------------------
def _closes(bars):
    return [float(b["close"]) for b in bars] if bars else []


def _sma(closes, n):
    return sum(closes[-n:]) / n if len(closes) >= n else None


def _ret(closes, days, skip=0):
    need = days + skip + 1
    if len(closes) < need:
        return None
    end = closes[-(skip + 1)]
    start = closes[-(days + skip + 1)]
    return end / start - 1.0 if start > 0 else None


def _ann_vol(closes, n):
    if len(closes) < n + 1:
        return None
    rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] > 0:
            rets.append(closes[i] / closes[i - 1] - 1.0)
    if len(rets) < 2:
        return None
    return pstdev(rets) * _ANN


# -----------------------------------------------------------------------------
# regime
# -----------------------------------------------------------------------------
def _raw_regime(market_state):
    qqq = _closes(market_state.get("QQQ") or [])
    spy = _closes(market_state.get("SPY") or [])
    if len(qqq) < 30 or len(spy) < 60:
        return "soft"

    r1, r3 = _ret(qqq, 1), _ret(qqq, 3)
    v10 = _ann_vol(qqq, 10)
    if (r1 is not None and r1 < BRAKE_R1) \
       or (r3 is not None and r3 < BRAKE_R3) \
       or (v10 is not None and v10 > BRAKE_VOL_10D):
        return "hard"

    spy_6mo = _ret(spy, 126)
    spy_v20 = _ann_vol(spy, 20)
    if spy_6mo is not None and spy_v20 is not None \
       and spy_6mo < PANIC_BEAR_RET and spy_v20 > PANIC_VOL:
        return "panic"

    spy_50 = _sma(spy, 50)
    qqq_50 = _sma(qqq, 50)
    if spy_50 is None or qqq_50 is None:
        return "soft"

    above_short = spy[-1] > spy_50 * 1.005 and qqq[-1] > qqq_50 * 1.005

    # Long trend gate only if we have enough history; otherwise short gate alone.
    spy_200 = _sma(spy, 200)
    if spy_200 is None:
        return "on" if above_short else "soft"

    above_long = spy[-1] > spy_200
    if above_short and above_long:
        return "on"
    return "soft"


def _confirm_regime(raw):
    """Asymmetric persistence: slow to enter risk-on, fast to leave."""
    global _pending_regime, _pending_count, _current_regime

    if raw == "hard":
        _pending_regime, _pending_count = None, 0
        _current_regime = "hard"
        return "hard"

    if raw == _current_regime:
        _pending_regime, _pending_count = None, 0
        return _current_regime

    if _current_regime == "on":
        confirm = CONFIRM_LEAVE_RISKON
    else:
        confirm = CONFIRM_ENTER_RISKON

    if raw == _pending_regime:
        _pending_count += 1
    else:
        _pending_regime, _pending_count = raw, 1

    if _pending_count >= confirm:
        _current_regime = _pending_regime
        _pending_regime, _pending_count = None, 0

    return _current_regime


# -----------------------------------------------------------------------------
# DD governor
# -----------------------------------------------------------------------------
def _dd_factor(equity):
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    if _peak_equity <= 0:
        return 1.0
    dd = 1.0 - equity / _peak_equity
    if dd >= DD_TIER_3:
        return 0.10
    if dd >= DD_TIER_2:
        return 0.30
    if dd >= DD_TIER_1:
        return 0.60
    return 1.0


# -----------------------------------------------------------------------------
# weighting
# -----------------------------------------------------------------------------
def _portfolio_vol_estimate(weights, market_state):
    if not weights:
        return TARGET_PORT_VOL
    num = 0.0
    denom = 0.0
    for t, w in weights.items():
        v = _ann_vol(_closes(market_state.get(t) or []), VOL_LOOKBACK)
        if v and v > 0:
            num += w * v
            denom += w
    if denom <= 0:
        return TARGET_PORT_VOL
    return num / denom


def _inv_vol_weights(names, market_state):
    inv = {}
    for t in names:
        v = _ann_vol(_closes(market_state.get(t) or []), VOL_LOOKBACK)
        if v and v > 0 and market_state.get(t):
            inv[t] = 1.0 / v
    if not inv:
        return {}
    s = sum(inv.values())
    return {t: w / s for t, w in inv.items()}


def _xs_momentum_filter(market_state, basket):
    qualifiers = []
    for t in basket:
        closes = _closes(market_state.get(t) or [])
        if not closes:
            continue
        sma = _sma(closes, NAME_TREND_DAYS)
        mom = _ret(closes, MOMENTUM_LOOKBACK, MOMENTUM_SKIP)
        if sma is None or mom is None:
            continue
        if closes[-1] > sma and mom > 0:
            qualifiers.append((mom, t))
    qualifiers.sort(reverse=True)
    return [t for _, t in qualifiers[:TOP_N_RISKON]]


def _apply_caps(weights, gross_cap):
    if not weights:
        return {}
    total_raw = sum(weights.values())
    if total_raw <= 0:
        return {}
    target_total = min(gross_cap, total_raw)
    scaled = {t: w * target_total / total_raw for t, w in weights.items()}
    capped = {}
    overflow = 0.0
    for t, w in scaled.items():
        if w > NAME_CAP:
            overflow += w - NAME_CAP
            capped[t] = NAME_CAP
        else:
            capped[t] = w
    if overflow > 1e-9:
        room = {t: NAME_CAP - w for t, w in capped.items() if w < NAME_CAP}
        room_total = sum(room.values())
        if room_total > 0:
            for t in capped:
                if capped[t] < NAME_CAP:
                    extra = overflow * room[t] / room_total
                    capped[t] = min(NAME_CAP, capped[t] + extra)
    return capped


def _targets(market_state, equity, regime):
    global _brake_cooldown

    if regime == "hard":
        _brake_cooldown = BRAKE_COOLDOWN
        raw = _inv_vol_weights(HARD_BRAKE_BASKET, market_state)
        return _apply_caps(raw, 0.15)

    if _brake_cooldown > 0:
        _brake_cooldown -= 1
        raw = _inv_vol_weights(DEFENSIVE, market_state)
        return _apply_caps(raw, 0.30)

    dd_cap = _dd_factor(equity)

    if regime == "panic":
        cap = min(PANIC_GROSS_CAP, dd_cap * PANIC_GROSS_CAP)
        raw = _inv_vol_weights(DEFENSIVE, market_state)
        return _apply_caps(raw, cap)

    if regime == "soft":
        cap = min(0.40, dd_cap * 0.40)
        raw = _inv_vol_weights(DEFENSIVE, market_state)
        return _apply_caps(raw, cap)

    winners = _xs_momentum_filter(market_state, RISK_ON)
    if not winners:
        cap = min(0.30, dd_cap * 0.30)
        raw = _inv_vol_weights(DEFENSIVE, market_state)
        return _apply_caps(raw, cap)

    risk_w = _inv_vol_weights(winners, market_state)
    def_w = _inv_vol_weights(SOFT_DEFENSIVE, market_state)
    raw = {t: w * RISKON_RISK_PCT for t, w in risk_w.items()}
    for t, w in def_w.items():
        raw[t] = raw.get(t, 0.0) + w * RISKON_DEF_PCT

    gross_cap = min(GROSS_MAX, dd_cap * GROSS_MAX)
    sum_raw = sum(raw.values())
    port_vol = _portfolio_vol_estimate(raw, market_state)
    port_vol = max(PORT_VOL_FLOOR, min(PORT_VOL_CEILING, port_vol))
    vol_target_gross = min(gross_cap, sum_raw * (TARGET_PORT_VOL / port_vol))

    return _apply_caps(raw, vol_target_gross)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance
    _tick += 1

    positions = {p["ticker"]: p for p in portfolio_state.get("positions", []) or []}
    last = portfolio_state.get("last_prices", {}) or {}
    equity = portfolio_state.get("cash", cash)
    for tk, pos in positions.items():
        equity += pos["quantity"] * last.get(tk, pos.get("avg_cost", 0))
    if equity <= 0:
        return []

    raw_regime = _raw_regime(market_state)
    regime = _confirm_regime(raw_regime)

    derisk = regime == "hard" or _brake_cooldown > 0
    on_cadence = _tick - _last_rebalance >= REBALANCE_EVERY
    is_scheduled = on_cadence and not derisk and regime == "on"

    if not on_cadence and not derisk:
        return []

    targets = _targets(market_state, equity, regime)
    if not targets:
        return []

    if is_scheduled:
        score_map = {}
        for t in RISK_ON:
            closes = _closes(market_state.get(t) or [])
            if not closes:
                continue
            sma = _sma(closes, NAME_TREND_DAYS)
            mom = _ret(closes, MOMENTUM_LOOKBACK, MOMENTUM_SKIP)
            if sma is not None and mom is not None and closes[-1] > sma and mom > 0:
                score_map[t] = mom
        
        current_score = 0.0
        for ticker, pos in positions.items():
            p = last.get(ticker, 0.0)
            if p > 0 and equity > 0:
                w = (pos["quantity"] * p) / equity
                current_score += w * score_map.get(ticker, 0.0)
        
        target_score = 0.0
        for ticker, w in targets.items():
            target_score += w * score_map.get(ticker, 0.0)
            
        improvement = target_score - current_score
        if improvement <= ADAPTIVE_THRESHOLD:
            return []

    orders = []
    for ticker, pos in positions.items():
        if ticker not in targets and pos["quantity"] > 0:
            orders.append({
                "ticker": ticker, "side": "sell", "quantity": pos["quantity"],
            })

    for ticker, weight in targets.items():
        bars = market_state.get(ticker)
        if not bars:
            continue
        px = float(bars[-1]["close"])
        if px <= 0:
            continue
        cur_qty = positions.get(ticker, {}).get("quantity", 0)
        delta = int((equity * weight - cur_qty * px) // px)
        if abs(delta * px) < DEAD_BAND * equity:
            continue
        if delta > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": delta})
        elif delta < 0 and cur_qty > 0:
            orders.append({
                "ticker": ticker, "side": "sell",
                "quantity": min(abs(delta), cur_qty),
            })

    if orders:
        _last_rebalance = _tick
    return orders
