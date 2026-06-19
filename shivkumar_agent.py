"""HMM + Hawkes Combined Trading Agent — v12 Full Combo.

HMM regime detection + Hawkes intensity engine combined trading agent.
Unique in competition as only HMM+Hawkes agent.

Key elements:
  - _detect_hmm_regime(spy_bars) → returns (regime, confidence)
  - _detect_events(closes) → returns (events_dict, score)
  - _update_hawkes_intensity(ticker, events, score) → adaptive decay
  - _combo_signal(...) → ENTER/EXIT logic
  - decide(...) → main entry point
"""
from __future__ import annotations

from math import sqrt, log
from statistics import mean, pstdev

# ---------------------------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------------------------
RISK_ON = (
    "SPY", "QQQ", "SMH",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
)
DEFENSIVE = ("XLP", "XLU", "TLT", "GLD")
SELECT = RISK_ON + DEFENSIVE

# ---------------------------------------------------------------------------
# HMM REGIME DETECTION WITH CONFIDENCE
# ---------------------------------------------------------------------------
HMM_VOL_WINDOW = 20
HMM_MIN_BARS = 30
VOL_HIGH_THRESHOLD = 0.22
VOL_LOW_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# HAWKES ENGINE
# ---------------------------------------------------------------------------
K_RETURN = 1.5
HAWKES_DECAY = 0.8
HAWKES_DECAY_STRESS = 0.3

WEIGHT_LARGE_RETURN = 1.0
WEIGHT_BREAKOUT = 1.5
WEIGHT_BREAKDOWN = 1.5
WEIGHT_MOM_BURST = 1.3

ROLLING_WINDOW = 20

# ---------------------------------------------------------------------------
# COMBO THRESHOLDS
# ---------------------------------------------------------------------------
ENTER_HMM_CONFIDENCE = 0.50
ENTER_HAWKES_SCORE = 1.0
ENTER_INTENSITY_MIN = 2.0

EXIT_INTENSITY_OVERHEAT = 8.0
EXIT_HMM_CONFIDENCE = 0.20
EXIT_HAWKES_SCORE = 0.0

# ---------------------------------------------------------------------------
# POSITION SIZING
# ---------------------------------------------------------------------------
NAME_CAP = 0.18
GROSS_MAX = 1.00
MAX_POSITIONS = 6
TREND_SMA = 50
DEAD_BAND = 0.03

# ---------------------------------------------------------------------------
# CRASH PROTECTION
# ---------------------------------------------------------------------------
BRAKE_3D = -0.03
BRAKE_5D = -0.05
MARKET_STRESS_THRESHOLD = 0.05

VOL_CALM = 0.16
VOL_ELEV = 0.26
FAST_BRAKE_1D = -0.015

DD_TIER_1 = 0.015
DD_TIER_2 = 0.025
DD_TIER_3 = 0.04


def _drawdown_scale(dd):
    if dd < DD_TIER_1:
        return 1.0
    if dd < DD_TIER_2:
        return 0.60
    if dd < DD_TIER_3:
        return 0.30
    return 0.10

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
_tick = 0
_ticker_intensity = {}
_ticker_event_history = {}
_last_signal = {}
_currently_held = set()
_entry_tick = {}

_hmm_history = []
_regime_history = []

_peak_equity = 0.0

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _closes(bars):
    return [float(b["close"]) for b in bars] if bars else []


def _sma(prices, window):
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _annualized_vol(closes, window=20):
    if len(closes) < window + 1:
        return None
    rets = []
    for i in range(len(closes) - window, len(closes)):
        if closes[i-1] > 0:
            rets.append(closes[i] / closes[i-1] - 1)
    if len(rets) < 2:
        return None
    return pstdev(rets) * sqrt(252)


def _annualized_return(closes, window=20):
    if len(closes) < window + 1:
        return None
    total_ret = closes[-1] / closes[-(window+1)] - 1
    return (1 + total_ret) ** (252 / window) - 1


# =========================================================================
# HMM REGIME DETECTION WITH CONFIDENCE (0-1)
# =========================================================================

def _detect_hmm_regime(spy_bars):
    """Detect market regime AND confidence level.
    
    Returns: (regime, confidence)
    regime: TREND, SIDEWAYS, or HIGH_VOLATILITY
    confidence: 0.0 to 1.0 (how confident is the model)
    """
    global _hmm_history, _regime_history
    
    closes = _closes(spy_bars)
    if len(closes) < HMM_MIN_BARS:
        return "SIDEWAYS", 0.5
    
    latest_ret = log(closes[-1] / closes[-2]) if closes[-2] > 0 else 0.0
    latest_vol = _annualized_vol(closes, HMM_VOL_WINDOW)
    
    if latest_vol is None:
        return "SIDEWAYS", 0.5
    
    _hmm_history.append((latest_ret, latest_vol))
    if len(_hmm_history) > 60:
        _hmm_history.pop(0)
    
    vol_score = 0
    if latest_vol > VOL_HIGH_THRESHOLD:
        vol_score = 2
    elif latest_vol > VOL_LOW_THRESHOLD:
        vol_score = 1
    
    ret_20d = _annualized_return(closes, 20) or 0
    ret_60d = _annualized_return(closes, 60) or 0
    
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200) if len(closes) >= 200 else None
    above_sma50 = closes[-1] > sma50 if sma50 else False
    above_sma200 = closes[-1] > sma200 if sma200 else True
    
    vol_trend = 0
    if len(_hmm_history) >= 10:
        recent_vol = mean([v for _, v in _hmm_history[-5:]])
        older_vol = mean([v for _, v in _hmm_history[-10:-5]])
        if recent_vol > older_vol * 1.2:
            vol_trend = 1
    
    # Determine regime
    if vol_score >= 2 and latest_vol > 0.25:
        regime = "HIGH_VOLATILITY"
    elif vol_score >= 1 and vol_trend > 0 and latest_ret < -0.02:
        regime = "HIGH_VOLATILITY"
    elif latest_vol > 0.22 and ret_20d < -0.08:
        regime = "HIGH_VOLATILITY"
    elif vol_score <= 1 and above_sma50 and above_sma200:
        if ret_20d > 0.05 or ret_60d > 0.08:
            regime = "TREND"
        else:
            regime = "SIDEWAYS"
    elif vol_score <= 1 and ret_60d > 0.10 and latest_ret > 0:
        regime = "TREND"
    else:
        regime = "SIDEWAYS"
    
    # Calculate confidence (0-1)
    confidence = 0.5  # Base
    
    # Vol clarity
    if vol_score == 2 and latest_vol > 0.30:
        confidence += 0.2
    elif vol_score == 0 and latest_vol < 0.08:
        confidence += 0.2
    elif vol_score == 1:
        confidence -= 0.1  # Ambiguous
    
    # Trend clarity
    if regime == "TREND":
        if above_sma50 and above_sma200:
            confidence += 0.15
        if ret_20d > 0.10:
            confidence += 0.1
    elif regime == "HIGH_VOLATILITY":
        if latest_vol > 0.30:
            confidence += 0.2
        if ret_20d < -0.05:
            confidence += 0.1
    elif regime == "SIDEWAYS":
        if abs(ret_20d) < 0.03:
            confidence += 0.1
    
    # Regime stability
    _regime_history.append(regime)
    if len(_regime_history) > 5:
        _regime_history.pop(0)
    
    if len(_regime_history) >= 3:
        recent_same = sum(1 for r in _regime_history[-3:] if r == regime)
        if recent_same >= 3:
            confidence += 0.1
        elif recent_same == 2:
            confidence += 0.05
        else:
            confidence -= 0.1
    
    confidence = max(0.0, min(1.0, confidence))
    
    # Regime persistence
    if len(_regime_history) >= 2 and _regime_history[-1] != _regime_history[-2]:
        if len(_regime_history) >= 3 and _regime_history[-2] == _regime_history[-3]:
            return _regime_history[-2], confidence * 0.8
    
    return regime, confidence


# =========================================================================
# HAWKES EVENT DETECTION
# =========================================================================

def _detect_events(closes):
    """Detect events and return score (0-5+)."""
    events = {
        'large_return': 0,
        'breakout': 0,
        'breakdown': 0,
        'momentum_burst': 0,
    }

    if len(closes) < ROLLING_WINDOW + 2:
        return events, 0

    window = closes[-(ROLLING_WINDOW + 1):-1]
    if len(window) < 2:
        return events, 0

    rets = [window[i] / window[i - 1] - 1.0
            for i in range(1, len(window))
            if window[i - 1] > 0]
    rolling_vol = pstdev(rets) if len(rets) >= 2 else 0.001

    rolling_max = max(window)
    rolling_min = min(window)

    today_ret = closes[-1] / closes[-2] - 1.0 if closes[-2] > 0 else 0.0

    if abs(today_ret) > K_RETURN * rolling_vol and rolling_vol > 0:
        events['large_return'] = 1

    if closes[-1] > rolling_max:
        events['breakout'] = 1
    elif closes[-1] < rolling_min:
        events['breakdown'] = 1

    if len(closes) >= 4:
        r0 = closes[-1] / closes[-2] - 1.0 if closes[-2] > 0 else 0.0
        r1 = closes[-2] / closes[-3] - 1.0 if closes[-3] > 0 else 0.0
        r2 = closes[-3] / closes[-4] - 1.0 if closes[-4] > 0 else 0.0
        if (r0 > 0 and r1 > 0 and r2 > 0) or (r0 < 0 and r1 < 0 and r2 < 0):
            events['momentum_burst'] = 1

    score = (
        WEIGHT_LARGE_RETURN * events['large_return'] +
        WEIGHT_BREAKOUT * events['breakout'] +
        WEIGHT_BREAKDOWN * events['breakdown'] +
        WEIGHT_MOM_BURST * events['momentum_burst']
    )

    return events, score


def _update_hawkes_intensity(ticker, events, score):
    """Update Hawkes intensity with adaptive decay."""
    global _ticker_intensity
    prev_intensity = _ticker_intensity.get(ticker, 0.0)
    
    is_stress = events.get('breakdown', 0)
    decay = HAWKES_DECAY_STRESS if is_stress else HAWKES_DECAY
    
    new_intensity = decay * prev_intensity + score
    _ticker_intensity[ticker] = new_intensity
    
    return new_intensity, is_stress


# =========================================================================
# COMBO SIGNAL: v12 Full Combo
# =========================================================================

def _combo_signal(hmm_regime, hmm_confidence, score, intensity, events, qqq_trend_ok, closes):
    """v12 COMBO:
    
    ENTER when ALL true:
      - HMM Regime is TREND or SIDEWAYS (NOT HIGH_VOL)
      - HMM Confidence > 0.50
      - Hawkes Score > 1
      - Hawkes Intensity > 2
    
    EXIT when ANY true:
      - HMM Regime is HIGH_VOL
      - Breakdown event
      - Market stress > 5%
      - Intensity > 8 (overheat)
      - HMM confidence < 0.20
      - Hawkes score <= 0
    """
    # EXIT CONDITIONS (priority order)
    if events.get('breakdown', 0): return "exit"
    if hmm_regime == "HIGH_VOLATILITY": return "exit"
    spy_stress = _market_stress_rate("SPY")
    qqq_stress = _market_stress_rate("QQQ")
    if spy_stress > MARKET_STRESS_THRESHOLD or qqq_stress > MARKET_STRESS_THRESHOLD: return "exit"
    if intensity > EXIT_INTENSITY_OVERHEAT: return "exit"
    if hmm_confidence < EXIT_HMM_CONFIDENCE: return "exit"
    if score <= EXIT_HAWKES_SCORE: return "exit"
    
    # ENTER CONDITIONS (regime-dependent)
    if hmm_regime == "TREND":
        if hmm_confidence <= 0.50: return "exit"
        if score <= 1.0: return "exit"
        if intensity <= 2.0: return "exit"
    elif hmm_regime == "SIDEWAYS":
        if hmm_confidence <= 0.50: return "exit"
        if score <= 2.0: return "exit"
        if intensity <= 2.5: return "exit"
    else:
        return "exit"
    
    # Trend filters
    if len(closes) >= TREND_SMA:
        sma50 = sum(closes[-TREND_SMA:]) / TREND_SMA
        if closes[-1] <= sma50: return "exit"
    if not qqq_trend_ok: return "exit"
    
    return "enter"


# =========================================================================
# MARKET STRESS
# =========================================================================

def _update_event_history(ticker, events):
    if ticker not in _ticker_event_history:
        _ticker_event_history[ticker] = []
    _ticker_event_history[ticker].append(events)
    if len(_ticker_event_history[ticker]) > 10:
        _ticker_event_history[ticker].pop(0)


def _market_stress_rate(ticker):
    history = _ticker_event_history.get(ticker, [])
    if not history:
        return 0.0
    return sum(1 for e in history if e.get('breakdown', 0)) / len(history)


def _vol_scale(qqq_closes):
    v = _annualized_vol(qqq_closes, 20) if qqq_closes else None
    if v is None:
        return 1.0
    if v < VOL_CALM:
        return 1.0
    if v < VOL_ELEV:
        return 0.50
    return 0.0


# =========================================================================
# MAIN DECIDE FUNCTION
# =========================================================================

def decide(market_state, portfolio_state, cash):
    global _tick, _ticker_intensity, _last_signal, _currently_held, _entry_tick, _peak_equity

    _tick += 1

    positions = {p["ticker"]: p for p in portfolio_state.get("positions", [])}
    last_prices = portfolio_state.get("last_prices", {})
    equity = portfolio_state.get("cash", cash)
    for tk, pos in positions.items():
        equity += pos["quantity"] * last_prices.get(tk, pos.get("avg_cost", 0))
    if equity <= 0:
        return []

    if _peak_equity <= 0:
        _peak_equity = equity
    if equity > _peak_equity:
        _peak_equity = equity
    dd = (_peak_equity - equity) / _peak_equity
    dd_scale = _drawdown_scale(dd)

    # --- HMM Regime Detection with Confidence on SPY ---
    spy_bars = market_state.get("SPY")
    regime = "SIDEWAYS"
    hmm_confidence = 0.5
    if spy_bars and len(spy_bars) >= HMM_MIN_BARS:
        regime, hmm_confidence = _detect_hmm_regime(spy_bars)

    # --- QQQ trend check ---
    qqq_bars = market_state.get("QQQ")
    qqq_closes = _closes(qqq_bars or [])
    qqq_trend_ok = True
    if len(qqq_closes) >= 5:
        qqq_sma5 = sum(qqq_closes[-5:]) / 5
        if qqq_closes[-1] < qqq_sma5:
            qqq_trend_ok = False

    # --- Market-wide trend exit ---
    if qqq_bars and len(qqq_closes) >= 5:
        qqq_sma5 = sum(qqq_closes[-5:]) / 5
        if qqq_closes[-1] < qqq_sma5:
            orders = []
            for ticker, pos in positions.items():
                if pos["quantity"] > 0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})
            if orders:
                _currently_held = set()
                _last_signal = {}
                _entry_tick = {}
                return orders

    # --- Emergency crash brakes ---
    if len(qqq_closes) >= 4:
        qqq_3d = qqq_closes[-1] / qqq_closes[-4] - 1.0
        if qqq_3d < BRAKE_3D:
            orders = []
            for ticker, pos in positions.items():
                if pos["quantity"] > 0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})
            _currently_held = set()
            _last_signal = {}
            _entry_tick = {}
            return orders
    if len(qqq_closes) >= 6:
        qqq_5d = qqq_closes[-1] / qqq_closes[-6] - 1.0
        if qqq_5d < BRAKE_5D:
            orders = []
            for ticker, pos in positions.items():
                if pos["quantity"] > 0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})
            _currently_held = set()
            _last_signal = {}
            _entry_tick = {}
            return orders

    # --- Fast 1-day crash brake ---
    if len(qqq_closes) >= 2:
        qqq_1d = qqq_closes[-1] / qqq_closes[-2] - 1.0
        if qqq_1d < FAST_BRAKE_1D:
            orders = []
            for ticker, pos in positions.items():
                if pos["quantity"] > 0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})
            _currently_held = set()
            _last_signal = {}
            _entry_tick = {}
            return orders

    # --- Market-wide crash exit ---
    for market_ticker in ("SPY", "QQQ"):
        market_bars = market_state.get(market_ticker)
        if market_bars and len(market_bars) >= 4:
            market_closes = _closes(market_bars)
            market_ret_3d = market_closes[-1] / market_closes[-4] - 1.0 if market_closes[-4] > 0 else 0.0
            if market_ret_3d < -0.02:
                orders = []
                for ticker, pos in positions.items():
                    if pos["quantity"] > 0:
                        orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})
                if orders:
                    _currently_held = set()
                    _last_signal = {}
                    _entry_tick = {}
                    return orders

    # --- Individual ticker crash exit ---
    for ticker, pos in positions.items():
        if pos["quantity"] > 0:
            bars = market_state.get(ticker)
            if bars and len(bars) >= 4:
                closes = _closes(bars)
                ret_3d = closes[-1] / closes[-4] - 1.0 if closes[-4] > 0 else 0.0
                if ret_3d < -0.02:
                    orders = [{"ticker": ticker, "side": "sell", "quantity": pos["quantity"]}]
                    if ticker in _currently_held:
                        _currently_held.discard(ticker)
                    _last_signal[ticker] = "exit"
                    return orders

    # --- Vol-based exposure scaling ---
    vol_scale = _vol_scale(qqq_closes)

    # --- Hawkes + v12 Full Combo ---
    new_enter = []
    new_exit = []
    current_in_position = []

    for ticker in SELECT:
        bars = market_state.get(ticker)
        if not bars:
            continue
        closes = _closes(bars)
        if len(closes) < ROLLING_WINDOW + 2:
            continue

        events, score = _detect_events(closes)
        intensity, is_stress = _update_hawkes_intensity(ticker, events, score)
        _update_event_history(ticker, events)

        signal = _combo_signal(regime, hmm_confidence, score, intensity, events, qqq_trend_ok, closes)
        prev_signal = _last_signal.get(ticker, "hold")

        if signal == "enter":
            current_in_position.append(ticker)

        if signal != prev_signal:
            _last_signal[ticker] = signal
            if signal == "enter" and prev_signal != "enter":
                new_enter.append(ticker)
                _entry_tick[ticker] = _tick
            elif signal == "exit" and prev_signal == "enter":
                new_exit.append(ticker)

    # --- Build target portfolio ---
    orders = []
    
    in_position_sorted = sorted(current_in_position,
                          key=lambda t: _ticker_intensity.get(t, 0),
                          reverse=True)

    held_not_exited = [t for t in _currently_held
                       if t not in new_exit and t not in in_position_sorted]

    target_names = in_position_sorted[:MAX_POSITIONS]
    remaining_slots = MAX_POSITIONS - len(target_names)
    if remaining_slots > 0:
        target_names.extend(held_not_exited[:remaining_slots])

    # Regime-based gross cap + drawdown scaling
    gross_cap = 1.0 if regime != "HIGH_VOLATILITY" else 0.30
    effective_gross_max = min(GROSS_MAX, gross_cap) * dd_scale
    effective_gross_max *= vol_scale

    n_targets = len(target_names)
    if n_targets > 0:
        per_name = min(effective_gross_max / n_targets, NAME_CAP)
        targets = {t: per_name for t in target_names}
    else:
        targets = {}

    _currently_held = set(target_names)

    # --- Generate orders ---
    for ticker, pos in positions.items():
        if ticker not in targets and pos["quantity"] > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": pos["quantity"]})

    for ticker, weight in targets.items():
        bars = market_state.get(ticker)
        if not bars:
            continue
        px = float(bars[-1]["close"])
        if px <= 0:
            continue
        cur_qty = positions.get(ticker, {}).get("quantity", 0)
        target_value = equity * weight
        cur_value = cur_qty * px
        delta_value = target_value - cur_value

        if abs(delta_value) < DEAD_BAND * equity:
            continue

        if delta_value > 0:
            buy_qty = int(delta_value // px)
            if buy_qty > 0:
                orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
        elif delta_value < 0 and cur_qty > 0:
            sell_qty = min(int(abs(delta_value) // px), int(cur_qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})

    return orders
