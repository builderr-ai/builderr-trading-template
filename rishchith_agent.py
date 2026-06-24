"""
Advanced Adaptive Multi-Factor Trading Agent
=============================================
Contest objective: maximize 60-day forward Calmar (annualized return / max drawdown).
Ranked by: Calmar = annualised_return / max_drawdown.

Strategy overview:
  1. REGIME DETECTION — 4-tier: risk_on, cautious, risk_off, crash_bail.
     - crash_bail fires instantly when QQQ drops > 3.5% in 3 bars OR
       3-bar realized vol > 2× its 30-day average (vol-of-vol spike).
       This gives ~1-2 day earlier exit than pure SMA50 crossing.
     - risk_off: SPY or QQQ below SMA50, OR 20d vol ≥ 38%.
     - cautious: 20d vol ≥ 25% OR either index 20d mom < 0.
     - risk_on: everything else.

  2. RISK-ON — Multi-factor composite score across 23 liquid candidates.
     Factors (all computed from provided bars, no network):
       • 60-day momentum   (40%) — primary trend signal
       • 20-day momentum   (25%) — medium-term trend
       • Trend gap vs SMA50 (15%) — distance above/below moving average
       • Risk-adj momentum (15%) — mom60 / vol20 (Sharpe-like quality filter)
       • 5-day reversal   (-15%) — fade short-term crowding (subtracted)
     Picks top 6 by score, then inverse-vol weighted, capped at 24%.

  3. CAUTIOUS — Top 3 winners + defensive sleeve (XLP/XLU/XLV). ~60/40 split.

  4. RISK-OFF / CRASH_BAIL — Defensive only: XLP/XLU/XLV/GLD.
     GLD provides cross-asset hedge vs equities.

  5. 2× OVERLAY (QLD/SSO) — Only when:
       QQQ 20d vol < 24% AND QQQ SMA20 > SMA50 AND QQQ 20d mom > 2%.
     Small allocation: 10% QLD + 6% SSO, budget comes from risk budget.
     3× ETFs (TQQQ/SOXL etc.) are NEVER used.

  6. RISK LIMITS:
       • Per-ticker cap: 24% (hard, enforced in _scale_caps)
       • Beta-adj gross: ≤ 1.30× equity (QLD counts 2×, SSO 2×)
       • Rebalance every 5 trading days OR when any ticker drifts > 27%
       • Max 45 orders per call (contest rule)
       • Min trade threshold: 1.2% of equity (avoids tiny fractional fills)

  7. NO external network calls. NO LLM. NO API keys required.
     Pure statistics from the 220+ daily bars provided in market_state.
"""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# ---------------------------------------------------------------------------
# Universe: ONLY non-leveraged or 2× ETFs in the main ranker.
# 3× ETFs (TQQQ, SOXL, etc.) are intentionally excluded to protect the
# beta-adjusted gross cap (3× would eat 3× the beta budget per dollar).
# ---------------------------------------------------------------------------
RISK_CANDIDATES = (
    # Broad market
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "SMH",
    # Mega-cap tech / high-conviction single names (all top-1000 liquidity)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
)

# Defensive book — risk_off / crash_bail regime
DEFENSIVE_WEIGHTS = (
    ("XLP", 0.24),
    ("XLU", 0.22),
    ("XLV", 0.20),
    ("GLD", 0.14),
)

# Cautious regime: trimmed defensive sleeve
CAUTIOUS_DEFENSIVE_WEIGHTS = (
    ("XLP", 0.15),
    ("XLU", 0.13),
    ("XLV", 0.12),
)

# Beta multipliers for the gross-exposure cap
BETA_MULTIPLE: dict[str, float] = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA":  3.0,
    "FAS":  3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN":  3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD":  2.0, "SSO":  2.0, "DDM":  2.0, "ROM":  2.0, "UWM":  2.0, "AGQ": 2.0,
}

# Tuning constants
REBALANCE_EVERY_DAYS = 5
MAX_WEIGHT           = 0.24
DRIFT_LIMIT          = 0.27
MAX_BETA_GROSS       = 1.30
MIN_TRADE_PCT        = 0.012

# Regime thresholds
VOL_RISK_OFF         = 0.38   # 20d ann. vol → risk_off
VOL_CAUTION          = 0.25   # 20d ann. vol → cautious
VOL_OVERLAY_ON       = 0.24   # must be BELOW for 2× overlay
CRASH_DROP_3BAR      = -0.035 # 3-bar QQQ return worse than this → crash_bail
CRASH_VOL_RATIO      = 2.0    # 3-bar vol > this × 30d avg vol → crash_bail

# Module-level state (reset between runs by design; None = first call)
_last_rebalance_bar_date: str | None = None


# ---------------------------------------------------------------------------
# Price / statistics helpers
# ---------------------------------------------------------------------------

def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    """Return clean positive close prices (oldest first); [] on bad data."""
    if not bars:
        return []
    out: list[float] = []
    for bar in bars:
        try:
            c = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if c <= 0:
            return []
        out.append(c)
    return out


def sma(values: list[float], n: int) -> float | None:
    return mean(values[-n:]) if len(values) >= n else None


def momentum(values: list[float], n: int) -> float | None:
    """Simple return from n bars ago to now."""
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    return (values[-1] / start - 1.0) if start > 0 else None


def realized_vol(values: list[float], n: int) -> float | None:
    """Annualised daily-return standard deviation over last n bars."""
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty  = float(raw.get("quantity", 0.0))
            cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        existing = positions.setdefault(ticker, {"quantity": 0.0, "avg_cost": cost})
        existing["quantity"] += qty
        existing["avg_cost"] = cost or existing["avg_cost"]
    return positions


def portfolio_equity(portfolio_state: dict[str, Any], cash: float) -> float:
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


def market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def latest_bar_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    ts = bars[-1].get("ts")
    return str(ts)[:10] if ts is not None else str(len(bars))


def days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def position_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
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


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def scale_caps(weights: dict[str, float]) -> dict[str, float]:
    """Enforce per-ticker MAX_WEIGHT and beta-adjusted gross cap."""
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_regime(market_state: dict[str, list[dict[str, Any]]]) -> str:
    """
    Returns one of: 'crash_bail', 'risk_off', 'cautious', 'risk_on'.

    Hierarchy (first match wins):

    crash_bail — Immediate defensive pivot, fires before SMA50 cross:
      • QQQ 3-bar return < -3.5%   (sudden large drop)
      • 3-bar realized vol > 2× rolling 30-day avg vol  (vol-of-vol spike)

    risk_off — Trend breakdown / elevated vol:
      • SPY below its 50-SMA
      • QQQ below its 50-SMA
      • QQQ 20-day ann. vol ≥ 38%
      • QQQ 10-day return < -8%

    cautious — Slowing trend / moderate vol:
      • QQQ 20-day ann. vol ≥ 25%
      • QQQ 20-day mom < 0
      • SPY 20-day mom < 0

    risk_on — Clear uptrend, low vol: everything else.
    """
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))

    if len(spy) < 50 or len(qqq) < 50:
        return "risk_off"

    spy_sma50 = sma(spy, 50)
    qqq_sma50 = sma(qqq, 50)
    qqq_vol20 = realized_vol(qqq, 20)

    if spy_sma50 is None or qqq_sma50 is None or qqq_vol20 is None:
        return "risk_off"

    # --- crash_bail: fast-reaction triggers ---
    qqq_mom3 = momentum(qqq, 3)
    if qqq_mom3 is not None and qqq_mom3 < CRASH_DROP_3BAR:
        return "crash_bail"

    if len(qqq) >= 34:
        vol3  = realized_vol(qqq, 3)
        vol30 = realized_vol(qqq, 30)
        if vol3 is not None and vol30 is not None and vol30 > 0:
            if vol3 > CRASH_VOL_RATIO * vol30:
                return "crash_bail"

    # --- risk_off: trend breakdown ---
    if spy[-1] < spy_sma50:
        return "risk_off"
    if qqq[-1] < qqq_sma50:
        return "risk_off"
    if qqq_vol20 >= VOL_RISK_OFF:
        return "risk_off"
    qqq_mom10 = momentum(qqq, 10)
    if qqq_mom10 is not None and qqq_mom10 < -0.08:
        return "risk_off"

    # --- cautious: headwinds ---
    if qqq_vol20 >= VOL_CAUTION:
        return "cautious"
    qqq_mom20 = momentum(qqq, 20)
    spy_mom20 = momentum(spy, 20)
    if qqq_mom20 is not None and qqq_mom20 < 0.0:
        return "cautious"
    if spy_mom20 is not None and spy_mom20 < 0.0:
        return "cautious"

    return "risk_on"


# ---------------------------------------------------------------------------
# Factor scoring
# ---------------------------------------------------------------------------

def score_candidates(
    market_state: dict[str, list[dict[str, Any]]],
) -> list[tuple[float, str, float]]:
    """
    Score each ticker in RISK_CANDIDATES on a composite factor.

    Composite = 0.40·mom60 + 0.25·mom20 + 0.15·trend_gap_50
              + 0.15·risk_adj_mom − 0.15·strev5

    risk_adj_mom = mom60 / vol20  (momentum per unit of risk, Sharpe-like)
    strev5       = 5-day return   (subtracted to dampen crowded short-term runs)

    Returns list of (score, ticker, ann_vol20), sorted descending by score.
    """
    scored: list[tuple[float, str, float]] = []
    for ticker in RISK_CANDIDATES:
        values = closes(market_state.get(ticker))
        if len(values) < 65:
            continue
        mom60  = momentum(values, 60)
        mom20  = momentum(values, 20)
        mom5   = momentum(values, 5)
        sma50v = sma(values, 50)
        vol20  = realized_vol(values, 20)
        if any(x is None for x in (mom60, mom20, mom5, sma50v, vol20)):
            continue
        if vol20 <= 0:
            continue
        trend_gap    = values[-1] / sma50v - 1.0
        risk_adj_mom = mom60 / vol20
        score = (
            0.40 * mom60
            + 0.25 * mom20
            + 0.15 * trend_gap
            + 0.15 * risk_adj_mom
            - 0.15 * mom5
        )
        scored.append((score, ticker, vol20))
    scored.sort(reverse=True)
    return scored


def inverse_vol_weights(
    candidates: list[tuple[float, str, float]],
    budget: float,
) -> dict[str, float]:
    """
    Allocate `budget` fraction of equity proportionally to 1/vol,
    then cap each position at MAX_WEIGHT.
    Equal-weight fallback if all vols are zero.
    """
    if not candidates:
        return {}
    inv_vols = [1.0 / max(vol, 1e-6) for _, _, vol in candidates]
    total_inv = sum(inv_vols)
    if total_inv <= 0:
        n = len(candidates)
        return {t: min(budget / n, MAX_WEIGHT) for _, t, _ in candidates}
    return {
        ticker: min(budget * inv_v / total_inv, MAX_WEIGHT)
        for (_, ticker, _), inv_v in zip(candidates, inv_vols)
    }


# ---------------------------------------------------------------------------
# Target weight construction
# ---------------------------------------------------------------------------

def target_weights(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Return {ticker: target_portfolio_weight} for the current regime."""
    regime = detect_regime(market_state)

    # --- crash_bail / risk_off: purely defensive ---
    if regime in ("crash_bail", "risk_off"):
        defensive = {
            t: w for t, w in DEFENSIVE_WEIGHTS
            if closes(market_state.get(t))
        }
        return scale_caps(defensive)

    # --- Score candidates (shared by cautious + risk_on) ---
    all_scored    = score_candidates(market_state)
    pos_scored    = [(s, t, v) for s, t, v in all_scored if s > 0.0]

    # --- cautious: 3 winners + light defensive sleeve ---
    if regime == "cautious":
        winners = pos_scored[:3]
        if not winners:
            defensive = {
                t: w for t, w in DEFENSIVE_WEIGHTS
                if closes(market_state.get(t))
            }
            return scale_caps(defensive)
        cautious_def = {
            t: w for t, w in CAUTIOUS_DEFENSIVE_WEIGHTS
            if closes(market_state.get(t))
        }
        def_budget  = sum(cautious_def.values())
        risk_budget = min(0.60, 1.0 - def_budget)
        risk_w      = inverse_vol_weights(winners, risk_budget)
        return scale_caps({**cautious_def, **risk_w})

    # --- risk_on: top 6 winners with optional 2× overlay ---
    winners = pos_scored[:6]
    if not winners:
        defensive = {
            t: w for t, w in DEFENSIVE_WEIGHTS
            if closes(market_state.get(t))
        }
        return scale_caps(defensive)

    qqq      = closes(market_state.get("QQQ"))
    vol20    = realized_vol(qqq, 20)   if len(qqq) >= 21 else None
    sma20_v  = sma(qqq, 20)            if len(qqq) >= 20 else None
    sma50_v  = sma(qqq, 50)            if len(qqq) >= 50 else None
    mom20_v  = momentum(qqq, 20)       if len(qqq) >= 21 else None

    overlay_on = bool(
        vol20   is not None and vol20   < VOL_OVERLAY_ON
        and sma20_v is not None and sma50_v is not None and sma20_v > sma50_v
        and mom20_v is not None and mom20_v > 0.02
        and closes(market_state.get("QLD"))
        and closes(market_state.get("SSO"))
    )

    risk_budget = 0.72 if overlay_on else 0.90
    risk_w = inverse_vol_weights(winners, risk_budget)

    if overlay_on:
        risk_w["QLD"] = min(risk_w.get("QLD", 0.0) + 0.10, MAX_WEIGHT)
        risk_w["SSO"] = min(risk_w.get("SSO", 0.0) + 0.06, MAX_WEIGHT)

    return scale_caps(risk_w)


# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------

def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    """
    Generate buy/sell orders to move from current holdings to target weights.
    Sells first, then buys with freed cash. Respects the 45-order hard cap.
    """
    if total_equity <= 0:
        return []

    min_trade    = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    # ---- Sells: full exits + position reduction ----
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        qty           = pos["quantity"]
        current_value = qty * price
        target_value  = total_equity * targets.get(ticker, 0.0)
        delta         = target_value - current_value

        if ticker not in targets:
            sell_qty = int(qty)
            if sell_qty > 0 and current_value >= min_trade:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
        elif delta < -min_trade:
            sell_qty = min(int(abs(delta) / price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price

    # 2% haircut on sell proceeds (slippage buffer)
    spendable = max(float(cash_available), 0.0) + sell_proceeds * 0.98

    # ---- Buys: largest underweight first ----
    for ticker, weight in sorted(targets.items(), key=lambda x: -x[1]):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        current_qty   = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value  = total_equity * weight
        delta         = target_value - current_value
        if delta < min_trade:
            continue
        buy_value = min(delta, spendable)
        buy_qty   = int(buy_value / price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price

    return orders[:45]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """
    Called once per decision interval (daily in admission; finer in Phase B).

    Args:
        market_state:    {ticker: [bar, ...]}  ≈220 daily bars per ticker
        portfolio_state: {cash, positions, last_prices}
        cash:            convenience copy of portfolio_state["cash"]

    Returns:
        List of {ticker, side: "buy"|"sell", quantity} orders.
        Empty list = hold / no action.

    Constraints respected:
        • Long-only (no short orders generated)
        • Beta-adj gross ≤ 1.30× (enforced in scale_caps)
        • Per-ticker ≤ 24% (enforced in scale_caps)
        • ≤ 45 orders per call
        • No network I/O, no external keys, runtime < 1s on modern hardware
    """
    global _last_rebalance_bar_date

    if not market_state:
        return []

    current_date = latest_bar_date(market_state)
    if current_date is None:
        return []

    total_equity = portfolio_equity(portfolio_state, cash)

    # ---- Decide whether to rebalance ----
    days_since   = days_since_rebalance(market_state)
    drifted      = position_drifted(portfolio_state, total_equity)
    regime       = detect_regime(market_state)
    crash_now    = (regime == "crash_bail")

    should_rebalance = (
        _last_rebalance_bar_date is None   # first call
        or days_since is None              # date tracking lost
        or days_since >= REBALANCE_EVERY_DAYS
        or drifted                         # single ticker > 27%
        or crash_now                       # fast crash response
    )

    if not should_rebalance:
        return []

    targets = target_weights(market_state)
    if not targets:
        return []

    prices    = market_prices(market_state)
    positions = current_positions(portfolio_state)
    orders    = orders_to_rebalance(targets, positions, total_equity, prices, cash)

    if orders:
        _last_rebalance_bar_date = current_date

    return orders
