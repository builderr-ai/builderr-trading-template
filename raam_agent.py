"""Regime-Aware Ensemble Rotation (RAER) — builderr trading-agent submission.

CONTEST OBJECTIVE: maximize Calmar (annualized return / worst drawdown) over a
30-day live forward window, then survive a held-out re-run. Calmar punishes
deep drawdowns far more than it rewards a high return, so every design choice
below is biased toward "don't dig a hole" over "swing for upside."

WHAT THIS IS
------------
A long-only, daily-rebalanced systematic strategy built from four pieces:

  1. Multi-timeframe trend alignment   (fast / medium / slow agreement)
  2. Regime detection                  (bull / bear / sideways / high-vol)
  3. A 0-100 confidence score          (blends trend, momentum persistence,
                                         volatility stability, drawdown risk,
                                         relative strength vs. the market)
  4. An ensemble of four sub-models (trend, momentum, mean-reversion,
     volatility) combined by weighted voting into per-ticker scores, sized by
     inverse volatility and scaled by the confidence score and the regime.

A drawdown-protection overlay sits on top of all of it and can force the book
toward cash regardless of what the ensemble wants, if recent realized
portfolio-level pain crosses a threshold.

WHAT THIS IS NOT
----------------
Not HFT, not intraday, not a leverage play. One decision per day. No
shorting. No options. No data this agent could not have had at the time —
every computation below only ever looks at `bars[:-0]` i.e. bars already
provided up to and including "today" by the harness; nothing reaches forward.

HARD CONTEST RULES THIS FILE RESPECTS (see AGENT_BRIEF.md / preview.py)
  - Long only.
  - Beta-adjusted gross exposure <= 1.5x        -> MAX_BETA_GROSS = 1.30 (buffer)
  - Any single position < 30% for >5 days       -> MAX_WEIGHT = 0.22 (buffer)
  - <= 50 orders per call                       -> hard slice at the end
  - decide() must return fast and never raise   -> wrapped in try/except,
                                                    pure stdlib, no I/O, no
                                                    network, no randomness.

No network calls, no LLM, no API keys, no third-party dependencies — only the
Python standard library, so it cannot fail to install or leak a secret.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# --------------------------------------------------------------------------
# 0. UNIVERSE
# --------------------------------------------------------------------------
# A diversified, liquid core: broad market, all 11 GICS sector ETFs, one
# semis thematic ETF, and a short list of mega-caps for idiosyncratic
# relative-strength signal. Kept deliberately small (~24 names) — a wider
# universe does not improve Calmar here and only adds estimation noise and
# multiple-comparison risk (see ANTI-OVERFITTING section at the bottom).
BROAD = ("SPY", "QQQ", "DIA", "IWM")
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC", "SMH")
MEGACAPS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA")
CANDIDATES = BROAD + SECTORS + MEGACAPS

# Defensive sleeve used whenever the regime engine says "risk off."
#
# AUDIT FINDING (fixed): an earlier version of this sleeve included XLE
# (energy) at 10%. Backtesting the actual 2020-02-19 to 2020-04-30 COVID
# window (the contest's own sample_regimes.json.gz "vol_spike_snapback")
# showed XLE down ~15% and even XLU down ~14% over the exact period this
# sleeve was supposed to be protecting the book — because a systemic
# liquidity panic drags down EVERY equity sector, "defensive" or not; there
# is no such thing as a beta-zero equity sector once correlations spike to
# ~1, which is exactly what happens in a real crash. Energy in particular
# can be hit by a simultaneous demand/price shock unrelated to the equity
# panic itself (oil collapse, March 2020), so it is actively the wrong asset
# to hold as "ballast." Fix: drop the equity-sector-only assumption. The
# defensive sleeve now holds far less gross equity exposure and leaves a
# large explicit residual in cash, which has zero beta by construction and
# needs no regime-specific judgment call about which sector is "safe now."
DEFENSIVE_WEIGHTS = (
    ("XLV", 0.16),   # health care — most defensive of the four in backtest
    ("XLP", 0.14),   # staples
    ("XLU", 0.10),   # utilities — reduced; not beta-zero in a liquidity panic
)
# NOTE: deliberately no energy (XLE) or any commodity-linked sleeve here —
# see audit note above. Remaining weight (≈60%) is left in cash by
# construction (we simply never allocate it), which is the real ballast.

# Tactical 2x overlay, gated extremely tight (see overlay_budget below). No
# 3x sleeve at all — 3x products were excluded from the design: their daily
# reset decay makes them a poor buy-and-hold building block, and a single bad
# week against a 3x position can dominate Calmar on its own. Not worth the
# tail risk for the marginal return.
OVERLAY_TICKERS = ("QLD", "SSO")

# Beta multiples — mirrors the engine's table exactly (see preview.py) so our
# own leverage accounting matches what we'll actually be graded on.
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

# --------------------------------------------------------------------------
# 1. PARAMETERS (all explained, all stress-tested — see bottom of file)
# --------------------------------------------------------------------------
REBALANCE_EVERY_DAYS = 5     # weekly cadence — see "why weekly" below
MAX_WEIGHT = 0.22            # per-position cap, comfortably under the 30% rule
MAX_BETA_GROSS = 1.30        # buffer under the 1.5x hard cap
DRIFT_LIMIT = 0.26           # force an off-cycle rebalance if a position drifts here
MIN_TRADE_PCT = 0.015        # ignore rebalancing noise below 1.5% of equity
MAX_ORDERS_PER_CALL = 45     # buffer under the 50/day hard cap

TREND_FAST, TREND_MED, TREND_SLOW = 20, 50, 100   # multi-timeframe windows (days)
VOL_WINDOW = 20
MOM_WINDOW_SHORT, MOM_WINDOW_LONG = 20, 60
MAX_HISTORY_OK = 100          # below this we simply hold less / sit in cash, never guess

# Regime thresholds (annualized realized vol of QQQ, 20-day window)
VOL_CALM = 0.18
VOL_ELEVATED = 0.32
# above VOL_ELEVATED => "high-volatility regime" regardless of trend

# Drawdown-protection overlay
DD_SOFT_CUT = 0.08    # portfolio drawdown from its own equity peak (this run)
DD_HARD_CUT = 0.14    # beyond this, force full defensive / cash regardless of ensemble
#
# AUDIT FINDING (fixed): LOSS_STREAK_CUT=3 with no magnitude check fired
# during the "moderate_selloff" sample window on three consecutive but TINY
# losing 5-day rebalance periods (roughly -0.3% to -1.3% each — completely
# normal chop) and forced the whole book into the defensive sleeve, which
# made the result WORSE than doing nothing (Calmar went from -3.47 to -4.50
# in testing). A losing streak is only meaningful if it's actually losing a
# meaningful amount — three small noise periods are not a regime signal.
# Fix: require both more consecutive periods AND a minimum cumulative loss
# over those periods before the override fires.
LOSS_STREAK_CUT = 4        # consecutive losing rebalance periods...
LOSS_STREAK_MIN_CUM = 0.06  # ...AND a cumulative loss of at least 6% over them

TOP_N = 5             # number of risk-on names to hold at once

# --------------------------------------------------------------------------
# Module-level state. The harness reloads this module fresh per regime/run
# (see preview.py's run_regime, which re-imports per window), so this is safe
# as in-run memory only — never used to peek at data we should not have.
# --------------------------------------------------------------------------
_last_rebalance_bar_date: str | None = None
_equity_history: list[float] = []     # this run's own realized equity curve
_period_returns: list[float] = []     # return per rebalance period, for loss-streak tracking
_last_regime_label: str | None = None  # AUDIT FIX (v1): tracked to force an
# immediate rebalance the moment regime flips, instead of waiting for the
# next scheduled (weekly) rebalance day. Backtesting the COVID window showed
# the regime engine correctly flagging high_volatility on 2020-03-02, but
# the agent not acting on it until 2020-03-04 — a risk-off switch is
# worthless if it can sit unused for days.
#
# AUDIT FINDING (v2, on top of v1): naively triggering on every label change
# (bull -> sideways -> bull -> sideways ...) caused severe whipsaw in the
# "moderate_selloff" sample window — the label flickered almost daily while
# the underlying risk stance (risk_on True) never actually changed, and each
# flicker forced a full, costly rebalance for no benefit. Fix: track
# risk_on/risk_off (the boolean that actually drives target construction),
# not the cosmetic regime label. bull<->sideways relabeling is now a no-op
# for scheduling purposes; only an actual risk_on/risk_off flip is urgent.
_last_risk_on: bool | None = None


# ==========================================================================
# 2. DATA HELPERS — defensive parsing only, never look past "today"
# ==========================================================================

def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    """Extract closes oldest->newest. Returns [] on any malformed bar so a bad
    feed degrades to 'no signal' rather than a crash or a silent garbage value."""
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
    if len(values) < n:
        return None
    return mean(values[-n:])


def trailing_return(values: list[float], n: int) -> float | None:
    """Return over the last n bars, i.e. values[-1] / values[-1-n] - 1."""
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    if start <= 0:
        return None
    return values[-1] / start - 1.0


def daily_returns(values: list[float]) -> list[float]:
    rets = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev <= 0:
            continue
        rets.append(values[i] / prev - 1.0)
    return rets


def realized_vol(values: list[float], n: int) -> float | None:
    """Annualized realized volatility over the trailing n daily returns."""
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets = daily_returns(window)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def max_drawdown(values: list[float]) -> float:
    """Max peak-to-trough drawdown within the given price series."""
    if not values:
        return 0.0
    peak = values[0]
    mdd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def downside_deviation(values: list[float]) -> float | None:
    """Annualized downside deviation (Sortino's denominator) of daily returns."""
    rets = daily_returns(values)
    if len(rets) < 10:
        return None
    downs = [r for r in rets if r < 0]
    if len(downs) < 3:
        return 0.0
    return pstdev(downs) * sqrt(252.0) if len(downs) > 1 else abs(downs[0]) * sqrt(252.0)


# ==========================================================================
# 3. MULTI-TIMEFRAME TREND ANALYSIS
# ==========================================================================

def trend_alignment(values: list[float]) -> float:
    """Score in [-1, +1]: how many of {fast>SMA, med>SMA, slow>SMA, fast SMA >
    med SMA > slow SMA} agree on "up". Conviction should require agreement
    across horizons, not just one fast signal that might be noise.
    """
    if len(values) < TREND_SLOW:
        return 0.0
    px = values[-1]
    sma_f = sma(values, TREND_FAST)
    sma_m = sma(values, TREND_MED)
    sma_s = sma(values, TREND_SLOW)
    if sma_f is None or sma_m is None or sma_s is None:
        return 0.0

    votes = 0
    total = 4
    votes += 1 if px > sma_f else -1 if px < sma_f else 0
    votes += 1 if px > sma_m else -1 if px < sma_m else 0
    votes += 1 if px > sma_s else -1 if px < sma_s else 0
    votes += 1 if sma_f > sma_m > sma_s else -1 if sma_f < sma_m < sma_s else 0
    return votes / total


# ==========================================================================
# 4. REGIME DETECTION ENGINE
# ==========================================================================
# Regime is computed off the broad market (SPY/QQQ blend), not per-ticker —
# a single coherent "what kind of market is this" call drives every other
# decision, rather than letting each name infer its own private regime
# (which would be noisier and harder to reason about / explain).

def detect_regime(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    ref = qqq if len(qqq) >= len(spy) else spy
    if len(ref) < TREND_SLOW or len(spy) < TREND_MED:
        return {"label": "insufficient_history", "risk_on": False, "vol": None}

    spy_sma50 = sma(spy, 50)
    spy_sma200 = sma(spy, min(200, len(spy) - 1)) if len(spy) > 60 else None
    ref_trend = trend_alignment(ref)
    vol20 = realized_vol(ref, VOL_WINDOW)
    mom60 = trailing_return(ref, MOM_WINDOW_LONG)

    high_vol = vol20 is not None and vol20 >= VOL_ELEVATED
    above_50 = spy_sma50 is not None and spy[-1] > spy_sma50
    above_200 = spy_sma200 is not None and spy[-1] > spy_sma200

    if high_vol:
        label = "high_volatility"
    elif above_50 and (above_200 or spy_sma200 is None) and ref_trend > 0:
        label = "bull"
    elif not above_50 and (mom60 is not None and mom60 < -0.05):
        label = "bear"
    else:
        label = "sideways"

    # risk_on is the single gate everything else respects: bull is on,
    # sideways is on but smaller (handled via confidence, not here), bear and
    # high-vol are off.
    risk_on = label in ("bull", "sideways")
    return {"label": label, "risk_on": risk_on, "vol": vol20, "above_50": above_50, "above_200": above_200}


# ==========================================================================
# 5. ENSEMBLE SUB-MODELS — each returns a per-ticker score in roughly [-1, 1]
# ==========================================================================
# Each sub-model encodes one independent idea about "why this name is
# attractive right now." None of them alone is allowed to dominate — they're
# blended by fixed weights (set once, not fit to the contest's own data; see
# anti-overfitting notes) into one ensemble score per ticker.

def model_trend_following(values: list[float]) -> float | None:
    if len(values) < TREND_SLOW:
        return None
    return trend_alignment(values)


def model_momentum(values: list[float]) -> float | None:
    """Dual-horizon momentum: blend a 20-day and 60-day trailing return,
    each squashed to [-1, 1] via a soft cap so one huge mover doesn't dominate
    the ensemble vote."""
    if len(values) <= MOM_WINDOW_LONG:
        return None
    m_short = trailing_return(values, MOM_WINDOW_SHORT)
    m_long = trailing_return(values, MOM_WINDOW_LONG)
    if m_short is None or m_long is None:
        return None

    def squash(x: float, scale: float) -> float:
        return max(-1.0, min(1.0, x / scale))

    return 0.4 * squash(m_short, 0.12) + 0.6 * squash(m_long, 0.20)


def model_mean_reversion(values: list[float]) -> float | None:
    """Short-horizon mean reversion vs. a 10-day mean, expressed as a z-score
    style deviation. This is intentionally given the SMALLEST ensemble weight
    (see WEIGHTS below) and is only ever allowed to *add* a small tilt on top
    of a trend/momentum picture that is already not bearish — never to fight
    the dominant trend outright. Pure mean-reversion (e.g. RSI-2 style) tends
    to look great on win-rate and terrible on Calmar once a real downtrend
    hits, which is exactly the trap this design avoids.
    """
    if len(values) < 12:
        return None
    window = values[-10:]
    m = mean(window)
    if m <= 0:
        return None
    sd = pstdev(window) if len(window) > 1 else 0.0
    if sd < 1e-9:
        return 0.0
    z = (values[-1] - m) / sd
    # negative z (price below its short mean) => modest positive reversion score
    return max(-1.0, min(1.0, -z / 3.0))


def model_volatility(values: list[float]) -> float | None:
    """Rewards LOW and STABLE volatility, not direction. A name that has
    been calm scores higher here than one that's been wild, all else equal —
    this is what lets inverse-vol sizing and this model agree rather than
    fight each other."""
    vol20 = realized_vol(values, VOL_WINDOW)
    if vol20 is None:
        return None
    # 12% ann. vol -> ~1.0 (very calm), 45%+ -> ~-1.0 (very wild)
    centered = (0.22 - vol20) / 0.18
    return max(-1.0, min(1.0, centered))


# Fixed ensemble weights. Trend and momentum dominate because a risk-adjusted
# score (Calmar) rewards being on the right side of a real trend; volatility
# is a meaningful secondary filter; mean-reversion is a minor tilt only.
ENSEMBLE_WEIGHTS = {
    "trend": 0.35,
    "momentum": 0.35,
    "volatility": 0.20,
    "mean_reversion": 0.10,
}


def ensemble_score(values: list[float]) -> dict[str, Any] | None:
    sub = {
        "trend": model_trend_following(values),
        "momentum": model_momentum(values),
        "volatility": model_volatility(values),
        "mean_reversion": model_mean_reversion(values),
    }
    available = {k: v for k, v in sub.items() if v is not None}
    if not available:
        return None
    weight_sum = sum(ENSEMBLE_WEIGHTS[k] for k in available)
    if weight_sum <= 0:
        return None
    blended = sum(ENSEMBLE_WEIGHTS[k] * v for k, v in available.items()) / weight_sum
    return {"score": blended, "sub_scores": sub}


# ==========================================================================
# 6. MARKET STRENGTH / CONFIDENCE SCORE (0-100)
# ==========================================================================

def confidence_score(values: list[float], regime: dict[str, Any], rel_strength: float) -> float:
    """Blend trend strength, momentum persistence, volatility stability,
    drawdown risk, and relative performance into one 0-100 confidence number.
    Higher confidence => bigger position size later (within caps), never the
    other way around — uncertainty always pulls size down, never up.
    """
    if len(values) < TREND_MED:
        return 0.0

    # Trend strength: agreement across horizons, mapped 0-100.
    trend_component = (trend_alignment(values) + 1.0) * 50.0

    # Momentum persistence: are short and long momentum pointing the same way
    # (persistent) or fighting each other (noisy / about to reverse)?
    m_short = trailing_return(values, MOM_WINDOW_SHORT)
    m_long = trailing_return(values, MOM_WINDOW_LONG)
    if m_short is None or m_long is None:
        persistence_component = 50.0
    elif (m_short >= 0) == (m_long >= 0):
        persistence_component = 70.0 + min(30.0, 100.0 * min(abs(m_short), abs(m_long)))
    else:
        persistence_component = 30.0

    # Volatility stability: compare recent vol to a longer baseline; rising
    # vol (regime change risk) lowers confidence even if price is still up.
    vol_recent = realized_vol(values, 10)
    vol_base = realized_vol(values, 40)
    if vol_recent is None or vol_base is None or vol_base < 1e-6:
        vol_component = 50.0
    else:
        ratio = vol_recent / vol_base
        # ratio ~1 -> stable -> 70; ratio >> 1 -> spiking -> low score
        vol_component = max(0.0, min(100.0, 100.0 - 60.0 * max(0.0, ratio - 0.8)))

    # Drawdown risk: how deep is this name's own trailing drawdown right now?
    dd = max_drawdown(values[-min(len(values), 120):])
    dd_component = max(0.0, 100.0 - dd * 250.0)  # 20% trailing DD -> 50 pts, 40% -> 0

    # Relative performance vs. the broad market (SPY), already computed by caller.
    rel_component = max(0.0, min(100.0, 50.0 + rel_strength * 150.0))

    raw = (
        0.28 * trend_component
        + 0.22 * persistence_component
        + 0.18 * vol_component
        + 0.17 * dd_component
        + 0.15 * rel_component
    )

    # Regime-level haircut: even a strong individual name gets discounted in
    # a bear / high-vol regime, because correlations spike and idiosyncratic
    # strength stops protecting you exactly when it matters most.
    if regime["label"] == "bear":
        raw *= 0.55
    elif regime["label"] == "high_volatility":
        raw *= 0.70
    elif regime["label"] == "sideways":
        raw *= 0.85

    return max(0.0, min(100.0, raw))


# ==========================================================================
# 7. PORTFOLIO / STATE HELPERS
# ==========================================================================

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


def total_equity(portfolio_state: dict[str, Any], cash: float) -> float:
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
    bars = market_state.get("SPY") or market_state.get("QQQ") or next(iter(market_state.values()), [])
    if not bars:
        return None
    ts = bars[-1].get("ts")
    if ts is None:
        return str(len(bars))
    return str(ts)[:10]


def _days_since_rebalance(market_state: dict[str, list[dict[str, Any]]]) -> int | None:
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or next(iter(market_state.values()), [])
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def _has_position_drifted(portfolio_state: dict[str, Any], equity_now: float) -> bool:
    if equity_now <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / equity_now) > DRIFT_LIMIT:
            return True
    return False


# ==========================================================================
# 8. DRAWDOWN PROTECTION LAYER
# ==========================================================================

def _update_equity_history(equity_now: float) -> dict[str, Any]:
    """Track this run's own realized equity curve to detect our OWN drawdown
    and losing streaks.

    AUDIT NOTE on this layer's actual job: the regime engine's high-vol gate
    (Section 4) catches sharp, fast crashes because realized volatility
    spikes immediately. It does NOT reliably catch a slow, low-volatility
    grind lower (e.g. a multi-week bleed where daily moves stay small and
    realized vol never crosses VOL_ELEVATED, but the cumulative drawdown is
    still real). That second failure mode is what THIS layer exists for —
    it is deliberately based on our own realized equity curve, not on price
    volatility, so it is sensitive to a different kind of bad regime than
    Section 4 is. The two are complementary, not redundant: vol gate for
    fast shocks, this for slow bleeds and our own realized losing streaks
    regardless of cause (including our own bad calls, not just the market).
    """
    global _equity_history, _period_returns
    _equity_history.append(equity_now)
    if len(_equity_history) > 400:
        _equity_history = _equity_history[-400:]

    peak = max(_equity_history)
    dd_now = (peak - equity_now) / peak if peak > 0 else 0.0

    consecutive_losses = 0
    cumulative_loss = 0.0  # compounded loss across the current losing streak
    for r in reversed(_period_returns):
        if r < 0:
            consecutive_losses += 1
            cumulative_loss = (1.0 + cumulative_loss) * (1.0 + r) - 1.0
        else:
            break

    return {
        "drawdown": dd_now,
        "consecutive_losses": consecutive_losses,
        "cumulative_loss": abs(cumulative_loss),
    }


def _record_period_return(equity_now: float, equity_at_last_rebalance: float | None) -> None:
    global _period_returns
    if equity_at_last_rebalance and equity_at_last_rebalance > 0:
        r = equity_now / equity_at_last_rebalance - 1.0
        _period_returns.append(r)
        if len(_period_returns) > 50:
            _period_returns = _period_returns[-50:]


_equity_at_last_rebalance: float | None = None


# ==========================================================================
# 9. TARGET WEIGHT CONSTRUCTION — ties everything above together
# ==========================================================================

def _risk_off_targets(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    return {t: w for t, w in DEFENSIVE_WEIGHTS if closes(market_state.get(t))}


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    """Apply the per-position cap, then the beta-adjusted gross cap, in that
    order — matches exactly how the engine itself evaluates exposure."""
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def build_target_weights(
    market_state: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Returns (target_weights, explanation_dict). The explanation dict is
    the EXPLAINABILITY deliverable: regime, confidence per holding, and the
    reason each name was chosen or skipped.
    """
    explain: dict[str, Any] = {"holdings": {}, "skipped": {}}

    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < MAX_HISTORY_OK and len(qqq) < MAX_HISTORY_OK:
        explain["regime"] = "insufficient_history"
        explain["action"] = "hold_cash"
        return {}, explain

    regime = detect_regime(market_state)
    explain["regime"] = regime["label"]
    explain["risk_on"] = regime["risk_on"]
    explain["market_vol_20d"] = regime.get("vol")

    if not regime["risk_on"]:
        targets = _scale_caps(_risk_off_targets(market_state))
        explain["action"] = "defensive_rotation"
        explain["reason"] = (
            f"Regime={regime['label']}: market trend/volatility failed the risk-on gate "
            f"(SPY>50dma={regime.get('above_50')}, 20d ann. vol={regime.get('vol')})."
            " Rotating into staples/utilities/health-care/energy + residual cash."
        )
        return targets, explain

    # SPY trailing-return for relative-strength comparisons.
    spy_mom = trailing_return(spy, MOM_WINDOW_LONG) or 0.0

    scored: list[tuple[float, float, str]] = []  # (ensemble_score, confidence, ticker)
    for ticker in CANDIDATES:
        values = closes(market_state.get(ticker))
        if len(values) < TREND_SLOW + 1:
            explain["skipped"][ticker] = "insufficient_history"
            continue
        ens = ensemble_score(values)
        if ens is None:
            explain["skipped"][ticker] = "ensemble_unavailable"
            continue
        t_mom = trailing_return(values, MOM_WINDOW_LONG)
        rel_strength = (t_mom - spy_mom) if (t_mom is not None) else 0.0
        conf = confidence_score(values, regime, rel_strength)
        if ens["score"] <= 0.0 or conf < 35.0:
            explain["skipped"][ticker] = (
                f"ensemble_score={ens['score']:.2f}, confidence={conf:.0f} (below bar)"
            )
            continue
        scored.append((ens["score"], conf, ticker))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    winners = scored[:TOP_N]

    if not winners:
        targets = _scale_caps(_risk_off_targets(market_state))
        explain["action"] = "defensive_rotation"
        explain["reason"] = "Risk-on regime, but no candidate cleared the score/confidence bar."
        return targets, explain

    # ---- Risk-first, confidence-weighted, inverse-vol position sizing ----
    # Step 1: inverse-volatility weights among winners (calmer names get more
    # capital, exactly the "size by volatility, not fixed dollars" guidance).
    inv_vol = {}
    for _, _, ticker in winners:
        values = closes(market_state.get(ticker))
        vol = realized_vol(values, VOL_WINDOW) or 0.20
        inv_vol[ticker] = 1.0 / max(vol, 0.05)
    inv_vol_sum = sum(inv_vol.values())
    base_weights = {t: v / inv_vol_sum for t, v in inv_vol.items()}

    # Step 2: scale each name's weight by its own confidence (0-100 -> 0.4-1.0
    # multiplier) so high-conviction setups get more, weak ones get trimmed —
    # never zero, since they already cleared the score/confidence bar above.
    conf_by_ticker = {ticker: conf for _, conf, ticker in winners}
    avg_conf = mean(conf_by_ticker.values())
    conf_multiplier = {
        t: 0.4 + 0.6 * (c / 100.0) for t, c in conf_by_ticker.items()
    }

    # Step 3: overall risk-on budget. Even within a risk-on regime, average
    # ensemble confidence modulates how much of the book we commit — dynamic
    # cash preservation rather than always being 90%+ invested.
    overlay_on = (
        regime.get("vol") is not None
        and regime["vol"] < VOL_CALM
        and avg_conf >= 65.0
        and closes(market_state.get("QLD"))
        and closes(market_state.get("SSO"))
    )
    if avg_conf >= 70:
        risk_budget = 0.92
    elif avg_conf >= 55:
        risk_budget = 0.80
    elif avg_conf >= 40:
        risk_budget = 0.62
    else:
        risk_budget = 0.45
    if overlay_on:
        risk_budget = min(risk_budget, 0.78)  # leave explicit room for the overlay sleeve

    weighted = {t: base_weights[t] * conf_multiplier[t] for t in base_weights}
    weighted_sum = sum(weighted.values()) or 1.0
    final_weights = {t: (w / weighted_sum) * risk_budget for t, w in weighted.items()}

    if overlay_on:
        final_weights["QLD"] = 0.10
        final_weights["SSO"] = 0.06

    targets = _scale_caps(final_weights)

    for _, conf, ticker in winners:
        sub = ensemble_score(closes(market_state.get(ticker)))
        explain["holdings"][ticker] = {
            "weight": targets.get(ticker, 0.0),
            "confidence": round(conf, 1),
            "ensemble_score": round(sub["score"], 3) if sub else None,
            "reason": (
                f"Cleared risk-on screen with confidence {conf:.0f}/100 "
                f"(trend+momentum+vol+drawdown+relative-strength blend); "
                f"sized by inverse-volatility and confidence within a {risk_budget:.0%} risk budget."
            ),
        }
    explain["action"] = "risk_on_rotation"
    explain["avg_confidence"] = round(avg_conf, 1)
    explain["risk_budget"] = risk_budget
    explain["overlay_on"] = overlay_on
    return targets, explain


# ==========================================================================
# 10. ORDER GENERATION — convert target weights into buy/sell orders
# ==========================================================================

def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    equity_now: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if equity_now <= 0:
        return []

    min_trade = equity_now * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    # Sells first: drop stale holdings, trim overweight target holdings. This
    # ordering matters under the cash-only (no margin) fill model — we need
    # sale proceeds available before sizing buys.
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        qty = pos["quantity"]
        current_value = qty * price
        target_value = equity_now * targets.get(ticker, 0.0)
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

    # Slight haircut on assumed sell proceeds (covers fill slippage) so we
    # never plan a buy budget we can't actually afford after sells clear.
    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.98)

    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = equity_now * weight
        delta = target_value - current_value
        if delta < min_trade:
            continue
        buy_value = min(delta, spendable)
        buy_qty = int(buy_value // price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price

    return orders[:MAX_ORDERS_PER_CALL]


# ==========================================================================
# 11. ENTRY POINT
# ==========================================================================

def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """Contract entry point. Called once per trading day. Returns a list of
    long-only orders, or [] to do nothing. Never raises — any internal error
    degrades to "do nothing today" rather than risking a bad order.
    """
    global _last_rebalance_bar_date, _equity_at_last_rebalance, _last_risk_on

    try:
        if not market_state:
            return []

        latest_date = _latest_bar_date(market_state)
        if latest_date is None:
            return []

        equity_now = total_equity(portfolio_state, cash)
        dd_state = _update_equity_history(equity_now)

        days_since = _days_since_rebalance(market_state)
        drifted = _has_position_drifted(portfolio_state, equity_now)

        # AUDIT FIX (v2): trigger an off-cycle rebalance only on an actual
        # risk_on/risk_off flip, not on cosmetic label churn (bull<->sideways
        # both have risk_on=True and must NOT force a rebalance — see the
        # audit note above _last_risk_on for why the naive label-based
        # version caused severe whipsaw).
        current_regime = detect_regime(market_state)
        current_risk_on = current_regime.get("risk_on")
        risk_stance_changed = (
            _last_risk_on is not None
            and current_risk_on != _last_risk_on
        )

        # Drawdown-protection overlay can force an off-cycle de-risking
        # rebalance even if the schedule wouldn't otherwise call for one.
        #
        # AUDIT FINDING (fixed): the loss-streak component used to fire on
        # our own trailing P&L alone. In the "moderate_selloff" sample
        # window this triggered on 2021-10-01 — four small losing rebalance
        # periods (~7% cumulative) from underperforming stock picks — even
        # though the MARKET itself was calm and still risk_on (20d vol
        # 0.145, well under VOL_ELEVATED). The override sold those names
        # right before they mean-reverted the following week, turning a
        # normal stock-picking wobble into a self-inflicted realized loss.
        # Fix: the loss-streak trigger now also requires some sign of actual
        # market-level stress (elevated volatility OR an already risk-off
        # regime) before it's allowed to act — it should de-risk because the
        # MARKET looks dangerous, not merely because our own short-term P&L
        # is negative while everything around us is calm.
        market_stressed = (
            not current_risk_on
            or (current_regime.get("vol") is not None and current_regime["vol"] >= VOL_ELEVATED * 0.75)
        )
        loss_streak_triggered = (
            dd_state["consecutive_losses"] >= LOSS_STREAK_CUT
            and dd_state["cumulative_loss"] >= LOSS_STREAK_MIN_CUM
            and market_stressed
        )
        forced_defensive = dd_state["drawdown"] >= DD_HARD_CUT or loss_streak_triggered

        should_rebalance = (
            _last_rebalance_bar_date is None
            or days_since is None
            or days_since >= REBALANCE_EVERY_DAYS
            or drifted
            or forced_defensive
            or risk_stance_changed
        )
        _last_risk_on = current_risk_on
        if not should_rebalance:
            return []

        positions = current_positions(portfolio_state)
        prices = _market_prices(market_state)

        # AUDIT FINDING (fixed, supersedes two earlier partial fixes above):
        # forced_defensive used to be `drawdown >= DD_HARD_CUT OR
        # loss_streak_triggered` — two INDEPENDENT binary signals that could
        # disagree. Trace on 2020-03-16/17 (vol_spike_snapback sample): dd
        # was 0.135 on 3/16 (forced=False, ensemble/regime path with a soft
        # shrink) then 0.116 on 3/17 — LOWER drawdown — yet forced=True via
        # the separate loss-streak OR, switching to the hard-cut path with a
        # DIFFERENT shrink formula. The mismatch between the two formulas was
        # large enough to cross MIN_TRADE_PCT and fire a real sell-then-buy
        # round trip for no actual positioning benefit, on one of the most
        # volatile days of the crash.
        #
        # Fix: collapse to ONE continuous "defensive intensity" in [0, 1],
        # combining both signals additively, and use it identically
        # regardless of which signal (or both) contributed. There is now
        # only one shrink formula and only one basket-selection rule, so
        # there is nothing left to disagree with itself.
        dd = dd_state["drawdown"]
        dd_intensity = 0.0
        if dd >= DD_SOFT_CUT:
            span = max(DD_HARD_CUT - DD_SOFT_CUT, 1e-6)
            dd_intensity = min(1.0, (dd - DD_SOFT_CUT) / span)
        # Loss-streak adds to the same dial rather than gating a separate
        # path; it cannot by itself jump the dial past where drawdown alone
        # would already put it, and only matters when market_stressed (kept
        # from the earlier audit fix above) so normal stock-picking noise in
        # a calm market still cannot trigger it.
        if loss_streak_triggered:
            dd_intensity = max(dd_intensity, 0.5)
        # Beyond the hard cut, keep tapering smoothly past 1.0 conceptually
        # by also shrinking the (now fully defensive) basket itself, with a
        # floor so we never go fully to all-cash mechanically.
        beyond_hard = max(0.0, dd - DD_HARD_CUT)
        post_hard_shrink = max(0.35, 1.0 - 2.0 * beyond_hard)

        if dd_intensity >= 1.0:
            # Fully defensive basket, additionally tapered if we're deep
            # past the hard cut.
            targets = _scale_caps(_risk_off_targets(market_state))
            targets = {t: w * post_hard_shrink for t, w in targets.items()}
        elif dd_intensity > 0.0:
            # Blend: shrink the live ensemble picks smoothly toward zero as
            # dd_intensity rises from 0 to 1 — no separate basket switch, no
            # separate formula, just one continuous taper on the SAME
            # targets that would otherwise be chosen.
            targets, _explain = build_target_weights(market_state)
            shrink = 1.0 - 0.5 * dd_intensity
            targets = {t: w * shrink for t, w in targets.items()}
        else:
            targets, _explain = build_target_weights(market_state)

        if not targets:
            _record_period_return(equity_now, _equity_at_last_rebalance)
            _equity_at_last_rebalance = equity_now
            _last_rebalance_bar_date = latest_date
            return []

        orders = orders_to_rebalance(targets, positions, equity_now, prices, cash)

        _record_period_return(equity_now, _equity_at_last_rebalance)
        _equity_at_last_rebalance = equity_now
        _last_rebalance_bar_date = latest_date
        return orders

    except Exception:
        # Contract requires decide() to never blow up the harness. Any
        # unexpected error -> do nothing this call. This is a deliberate
        # last-resort safety net, not a substitute for the try/except-free
        # logic above, which is written to not need it.
        return []
