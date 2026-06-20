"""Calmar Defender Hybrid — v7 submission for Builderr Round 1.

Mixes the proven v6 momentum engine with defensive ideas from the HMM+Hawkes
agent to lower drawdown without destroying Calmar or creating HMM+Hawkes-level
turnover.

Changes from v6:
  * Volatility-target gross scaling: reduce exposure smoothly when QQQ vol rises.
  * Tighter crash brakes and vol-exit thresholds.
  * Tighter drawdown governor tiers (2% / 4% / 7%).
  * Conditional fast EMA20 trend exit (active only when vol is elevated).
  * Per-ticker breakdown stop-loss and one-shot crisis exit.

Kept from v6:
  * Full universe scan, score^2 / inverse-vol sizing, EMA50/SMA100 trend filter.
  * No leverage/inverse/VIX products.
  * Long-only, standard library only.

Only Python standard library. No network, no LLM, no API keys.
"""
from __future__ import annotations

from math import sqrt, log
from statistics import mean, pstdev
from typing import Any

# ---------------------------------------------------------------------------
# Tunable parameters (user wanted only these four as primary knobs)
# ---------------------------------------------------------------------------
SMA_FAST = 50          # short-term trend filter
SMA_SLOW = 100         # long-term trend confirmation
REBALANCE_DAYS = 7     # rebalance cadence in risk-on

# Crash brake thresholds. Tighter = safer but more false positives.
BRAKE_1D = -0.028      # QQQ one-day drop
BRAKE_2D = -0.045      # QQQ two-day drop (hidden-regime accelerator)
BRAKE_3D = -0.052      # QQQ three-day drop
BRAKE_VOL_10D = 0.48   # QQQ 10-day annualized vol

# ---------------------------------------------------------------------------
# Fixed policy parameters (round numbers, robust to +/- 20% change)
# ---------------------------------------------------------------------------
MOMENTUM_DAYS = 63
MOMENTUM_SKIP = 2
VOL_DAYS = 20

TOP_N = 5
NAME_CAP = 0.22
GROSS_TARGET_ON = 0.95
MAX_BETA_GROSS = 1.35

DRIFT_LIMIT = 0.27
MIN_TRADE_PCT = 0.015

# Vol gate for risk-on. Must be below this to go full risk.
VOL_ENTRY_MAX = 0.24
VOL_EXIT_MAX = 0.32

# Hysteresis bands around SMA (fractional).
ENTRY_BAND = 0.009
EXIT_BAND = 0.00

# Cooldown after a hard-stress tick before full risk-on is allowed again.
COOLDOWN_DAYS = 3

# Drawdown governor tiers.
DD_TIER_1 = 0.020
DD_TIER_2 = 0.040
DD_TIER_3 = 0.070

# Volatility-target sizing (smooth risk management borrowed from HMM+Hawkes idea).
VOL_TARGET = 0.09        # target annualized portfolio vol
MIN_VOL_FOR_SCALE = 0.08

# Conditional fast EMA20 exit: only active when vol is already elevated.
FAST_EXIT_VOL = 0.13
FAST_EXIT_BAND = 0.010

# Crisis one-shot exit: emergency sell when vol spikes with momentum loss.
CRISIS_VOL = 0.20
CRISIS_RETURN_5D = -0.030

# Soft regime can hold a small defensive sleeve instead of cash.
# Set to empty tuple for 100% cash in soft; uncomment weights for defensive test.
SOFT_DEFENSIVE_WEIGHTS: tuple[tuple[str, float], ...] = (
    # ("XLP", 0.10),
    # ("XLU", 0.10),
    # ("XLV", 0.10),
)

_ANN = sqrt(252.0)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
OFFENSIVE_UNIVERSE = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "XLC", "XLRE", "XLB", "SMH",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "MU", "MRVL", "QCOM", "PLTR", "CRM",
    "JPM", "V", "MA", "UNH", "LLY", "XOM", "CVX",
)

DEFENSIVE_UNIVERSE = ("XLP", "XLU", "XLV", "XLE", "GLD", "TLT")

BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_last_rebalance_bar_date: str | None = None
_last_regime: str | None = None
_last_targets: dict[str, float] = {}
_cooldown_days: int = 0
_peak_equity: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    if not bars:
        return []
    out: list[float] = []
    for bar in bars:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return mean(values[-n:])


def ema(values: list[float], n: int) -> float | None:
    """Exponential moving average - faster response than SMA."""
    if len(values) < n:
        return None
    k = 2.0 / (n + 1)
    out = values[0]
    for v in values[1:]:
        out = v * k + out * (1 - k)
    return out


def momentum(values: list[float], n: int, skip: int = 0) -> float | None:
    need = n + skip + 1
    if len(values) < need:
        return None
    end = values[-(skip + 1)]
    start = values[-need]
    if start <= 0 or end <= 0:
        return None
    return end / start - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) < n + 1:
        return None
    rets: list[float] = []
    for i in range(len(values) - n, len(values)):
        prev = values[i - 1]
        if prev <= 0:
            return None
        rets.append(values[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * _ANN


def current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
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


def equity(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _latest_bar_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    return str(ts)[:10] if ts is not None else str(len(bars))


def _days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(bar.get("ts", i))[:10] for i, bar in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        series = closes(bars)
        if series:
            prices[ticker.upper()] = series[-1]
    return prices


def _vol_target_scale(market_state: dict[str, list[dict[str, Any]]]) -> float:
    """Scale gross exposure inversely with broad market vol."""
    qqq = closes(market_state.get("QQQ"))
    if len(qqq) < VOL_DAYS + 1:
        return 1.0
    vol20 = realized_vol(qqq, VOL_DAYS) or 0.30
    return min(1.0, VOL_TARGET / max(vol20, MIN_VOL_FOR_SCALE))


def _breakdown_exits(market_state: dict[str, list[dict[str, Any]]], positions: dict[str, dict[str, float]]) -> set[str]:
    """Return tickers that have broken below a recent floor (per-ticker stop)."""
    exits: set[str] = set()
    for ticker in positions:
        bars = market_state.get(ticker)
        if not bars:
            continue
        series = closes(bars)
        if len(series) < 22:
            continue
        price = series[-1]
        low_20 = min(series[-21:-1])
        if price < low_20 * 0.985:
            exits.add(ticker.upper())
    return exits


def _is_crisis(market_state: dict[str, list[dict[str, Any]]]) -> bool:
    """One-shot crisis detector: vol elevated and market dropping fast."""
    qqq = closes(market_state.get("QQQ"))
    if len(qqq) < VOL_DAYS + 1:
        return False
    vol20 = realized_vol(qqq, VOL_DAYS) or 0.0
    r5 = momentum(qqq, 5) or 0.0
    return vol20 >= CRISIS_VOL and r5 <= CRISIS_RETURN_5D


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def _classify_regime(market_state: dict[str, list[dict[str, Any]]]) -> str:
    """Return 'hard', 'soft', or 'on'."""
    global _last_regime, _cooldown_days

    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < SMA_SLOW or len(qqq) < SMA_SLOW:
        return "hard"

    # Fast crash brake.
    r1 = momentum(qqq, 1) or 0.0
    r2 = momentum(qqq, 2) or 0.0
    r3 = momentum(qqq, 3) or 0.0
    v10 = realized_vol(qqq, 10) or 0.0
    if r1 <= BRAKE_1D or r2 <= BRAKE_2D or r3 <= BRAKE_3D or v10 >= BRAKE_VOL_10D:
        _cooldown_days = COOLDOWN_DAYS
        return "hard"

    spy_sma_fast = ema(spy, SMA_FAST)
    qqq_sma_fast = ema(qqq, SMA_FAST)
    spy_sma_slow = sma(spy, SMA_SLOW)
    qqq_sma_slow = sma(qqq, SMA_SLOW)
    vol20 = realized_vol(qqq, VOL_DAYS)
    if vol20 is None:
        vol20 = 0.30

    if None in (spy_sma_fast, qqq_sma_fast, spy_sma_slow, qqq_sma_slow):
        return "hard"

    # Hysteresis: harder to enter, easier to exit.
    clearly_on = (
        spy[-1] > spy_sma_fast * (1 + ENTRY_BAND)
        and qqq[-1] > qqq_sma_fast * (1 + ENTRY_BAND)
        and spy[-1] > spy_sma_slow
        and qqq[-1] > qqq_sma_slow
        and vol20 < VOL_ENTRY_MAX
    )
    # Conditional fast EMA20 exit: helps in selloffs but avoids whipsaws in calm low-vol trends.
    qqq_ema20 = ema(qqq, 20)
    spy_ema20 = ema(spy, 20)
    fast_off = (
        vol20 > FAST_EXIT_VOL
        and (
            (qqq_ema20 is not None and qqq[-1] < qqq_ema20 * (1 - FAST_EXIT_BAND))
            or (spy_ema20 is not None and spy[-1] < spy_ema20 * (1 - FAST_EXIT_BAND))
        )
    )

    clearly_off = (
        spy[-1] < spy_sma_fast * (1 - EXIT_BAND)
        or qqq[-1] < qqq_sma_fast * (1 - EXIT_BAND)
        or fast_off
        or vol20 > VOL_EXIT_MAX
    )

    if _last_regime == "on":
        regime = "soft" if clearly_off else "on"
    else:
        regime = "on" if clearly_on else "soft"

    # Cooldown after hard stress: step through soft before full on.
    if _cooldown_days > 0:
        _cooldown_days -= 1
        if regime == "on":
            regime = "soft"

    return regime


# ---------------------------------------------------------------------------
# Target weights
# ---------------------------------------------------------------------------

def _score(values: list[float]) -> tuple[float, float] | None:
    """Return (score, vol20) for a candidate, or None if not eligible."""
    if len(values) < SMA_SLOW:
        return None
    price = values[-1]
    sma_fast = ema(values, SMA_FAST)
    # Average daily log return over the momentum window, annualized.
    # Smoother and less noisy than point-to-point momentum.
    mom: float | None = None
    start_idx = len(values) - MOMENTUM_DAYS - MOMENTUM_SKIP - 1
    end_idx = len(values) - MOMENTUM_SKIP - 1
    if start_idx >= 0 and end_idx > start_idx:
        log_rets: list[float] = []
        for i in range(start_idx + 1, end_idx + 1):
            if values[i - 1] > 0:
                log_rets.append(log(values[i] / values[i - 1]))
        if log_rets:
            mom = (sum(log_rets) / len(log_rets)) * 252.0
    vol20 = realized_vol(values, VOL_DAYS)
    if sma_fast is None or mom is None or vol20 is None:
        return None
    if price <= sma_fast or mom <= 0.0:
        return None
    trend_gap = price / sma_fast - 1.0
    score = 0.6 * mom + 0.4 * trend_gap
    if score <= 0.0:
        return None
    return score, max(vol20, 0.10)


def _defensive_targets(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    if not SOFT_DEFENSIVE_WEIGHTS:
        return {}
    available = [t for t, _ in SOFT_DEFENSIVE_WEIGHTS if closes(market_state.get(t))]
    if not available:
        return {}
    weights = {t: w for t, w in SOFT_DEFENSIVE_WEIGHTS if t in available}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {t: w / total * GROSS_TARGET_ON for t, w in weights.items()}


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    """Cap per-name dollar weight and beta-adjusted gross."""
    capped = {t: min(max(w, 0.0), NAME_CAP) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def _target_weights_for_regime(
    regime: str, market_state: dict[str, list[dict[str, Any]]], forced_exits: set[str] | None = None
) -> dict[str, float]:
    """Compute target dollar weights for a given regime."""
    forced_exits = forced_exits or set()
    if regime in ("soft", "hard"):
        weights = _scale_caps(_defensive_targets(market_state))
        weights = {t: w for t, w in weights.items() if t.upper() not in forced_exits}
        return weights

    # Risk-on: full universe scan. Score every ticker the exchange provides.
    # Exclude leveraged/inverse products and volatility derivatives.
    _EXCLUDED_TICKERS = set(BETA_MULTIPLE) | {
        "VIX", "VIXY", "UVXY", "SVXY", "TVIX", "VXX",
        "SPXU", "SH", "SDS", "SPXS", "SQQQ", "QID",
    }
    scored: list[tuple[float, str]] = []
    vol_map: dict[str, float] = {}
    for ticker in market_state:
        t = ticker.upper()
        if t in _EXCLUDED_TICKERS or t in forced_exits:
            continue
        values = closes(market_state.get(ticker))
        result = _score(values)
        if result is None:
            continue
        score, vol = result
        scored.append((score, t))
        vol_map[t] = vol

    if not scored:
        return _scale_caps(_defensive_targets(market_state))

    scored.sort(reverse=True)
    winners = [ticker for _, ticker in scored[:TOP_N]]

    raw: dict[str, float] = {}
    for ticker in winners:
        score = next(s for s, t in scored if t == ticker)
        vol = vol_map[ticker]
        # Squared score concentrates allocation on the strongest names while
        # still inverse-vol sizing. This lifts Calmar by cutting losers faster.
        raw[ticker] = (score * score) / vol

    total_raw = sum(raw.values())
    if total_raw <= 0:
        return _scale_caps(_defensive_targets(market_state))

    vol_scale = _vol_target_scale(market_state)
    weights = {t: (w / total_raw) * GROSS_TARGET_ON * vol_scale for t, w in raw.items()}
    return _scale_caps(weights)


def target_weights(market_state: dict[str, list[dict[str, Any]]], forced_exits: set[str] | None = None) -> dict[str, float]:
    """Compute target dollar weights. Residual is cash."""
    regime = _classify_regime(market_state)
    return _target_weights_for_regime(regime, market_state, forced_exits)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _has_position_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
    if total_equity <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False


def _target_distance(left: dict[str, float], right: dict[str, float]) -> float:
    names = set(left) | set(right)
    return sum(abs(left.get(name, 0.0) - right.get(name, 0.0)) for name in names)


def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0:
        return []

    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    # Sells first.
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

    # Buys second.
    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.98)
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


def _drawdown_scale(current_equity: float) -> float:
    global _peak_equity
    if current_equity > _peak_equity:
        _peak_equity = current_equity
    if _peak_equity <= 0:
        return 1.0
    dd = max(0.0, (_peak_equity - current_equity) / _peak_equity)
    if dd < DD_TIER_1:
        return 1.0
    if dd < DD_TIER_2:
        return 0.65
    if dd < DD_TIER_3:
        return 0.35
    return 0.10


def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """Return long-only buy/sell orders."""
    global _last_rebalance_bar_date, _last_regime, _last_targets, _peak_equity

    if not market_state:
        return []

    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []

    total_equity = equity(portfolio_state, cash)
    if total_equity <= 0:
        return []

    if _peak_equity <= 0 or total_equity > _peak_equity:
        _peak_equity = total_equity

    days_since = _days_since_rebalance(market_state)
    drifted = _has_position_drifted(portfolio_state, total_equity)

    raw_regime = _classify_regime(market_state)
    regime_changed = _last_regime is not None and raw_regime != _last_regime
    forced_derisk = _last_regime == "on" and raw_regime != "on"
    crisis = _is_crisis(market_state)

    positions = current_positions(portfolio_state)
    forced_exits = _breakdown_exits(market_state, positions)

    # Per-ticker breakdown stop: sell immediately on any bar.
    if forced_exits and not regime_changed:
        prices = _market_prices(market_state)
        orders: list[dict[str, object]] = []
        for ticker in forced_exits:
            qty = int(positions.get(ticker, {}).get("quantity", 0.0))
            if qty > 0 and prices.get(ticker, 0.0) > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
        if orders:
            return orders

    # Crisis mode: emergency derisk to cash even between scheduled rebalances.
    if crisis and positions and not regime_changed:
        prices = _market_prices(market_state)
        orders = []
        for ticker, pos in positions.items():
            qty = int(pos["quantity"])
            if qty > 0 and prices.get(ticker, 0.0) > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
        if orders:
            return orders

    # --- HMM-style per-ticker momentum guard: sell any held ticker whose momentum has turned negative ---
    # Only active when the overall market regime is still "on" (we are in risk-on mode).
    # This prevents us from holding a name that has lost its momentum while the broad trend is still up.
    if raw_regime == "on" and not regime_changed:
        prices = _market_prices(market_state)
        momentum_exits: list[dict[str, object]] = []
        for ticker, pos in positions.items():
            if pos["quantity"] <= 0:
                continue
            values = closes(market_state.get(ticker))
            if not values or len(values) < SMA_SLOW:
                continue
            sma_fast = ema(values, SMA_FAST)
            mom: float | None = None
            start_idx = len(values) - MOMENTUM_DAYS - MOMENTUM_SKIP - 1
            end_idx = len(values) - MOMENTUM_SKIP - 1
            if start_idx >= 0 and end_idx > start_idx:
                log_rets: list[float] = []
                for i in range(start_idx + 1, end_idx + 1):
                    if values[i - 1] > 0:
                        log_rets.append(log(values[i] / values[i - 1]))
                if log_rets:
                    mom = (sum(log_rets) / len(log_rets)) * 252.0
            if sma_fast is None or mom is None or values[-1] <= sma_fast or mom <= 0.0:
                qty = int(pos["quantity"])
                if qty > 0 and prices.get(ticker, 0.0) > 0:
                    momentum_exits.append({"ticker": ticker, "side": "sell", "quantity": qty})
        if momentum_exits:
            return momentum_exits

    should_rebalance = (
        _last_rebalance_bar_date is None
        or days_since is None
        or days_since >= REBALANCE_DAYS
        or drifted
        or regime_changed
        or forced_derisk
    )
    if not should_rebalance:
        return []

    targets = _target_weights_for_regime(raw_regime, market_state, forced_exits)

    # Apply drawdown governor.
    dd_scale = _drawdown_scale(total_equity)
    if dd_scale < 1.0:
        targets = {t: w * dd_scale for t, w in targets.items()}
        targets = {t: w for t, w in targets.items() if w > 0.001}

    # Skip tiny target changes to reduce turnover.
    if not regime_changed and _target_distance(_last_targets, targets) < 0.20:
        _last_regime = raw_regime
        return []

    prices = _market_prices(market_state)
    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)

    _last_regime = raw_regime
    if orders or regime_changed:
        _last_rebalance_bar_date = latest_date
        _last_targets = targets
    return orders
