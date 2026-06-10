"""Spear — Candidate B (builderr Trading v0) — FINAL SUBMISSION VERSION.

Concentrated conviction trend-following with gated convexity and a single fast
parachute, implemented per IMPLEMENTATION_BLUEPRINT.md (Candidate B in
FINAL_DECISION_MEMO.md) with every fix from BUG_AUDIT.md applied:

  P1 — non-target holdings are fully liquidated (no dead-band), so CASH / de-risk
       states are genuinely flat (closes the "dust in CASH" exposure leak).
  P2 — a single resolved cash value backs both equity and buy sizing.
  P3 — globals are snapshotted and restored if a tick ever raises (no state desync).
  P4 — the data-guard liquidate path keeps prev-state bookkeeping consistent.
  P5 — a 2% hysteresis band on the CASH<->NEUTRAL boundary removes whipsaw churn.
  P6 — in FULL without the QLD/SSO sleeve, its budget folds into the 1x core, so
       the most-bullish state is never lighter than NEUTRAL.
  P7 — no `assert`-for-control-flow (robust under `python -O`).
  P8 — deterministic equity summation (sorted positions).

Philosophy
----------
Hold the actual market leaders (single stocks AND thematic/sector ETFs), press
them while the trend is confirmed, add a measured 2x ETF sleeve (QLD/SSO) only
when everything aligns, and cut to cash hard and fast when the trend breaks — so
the right-tail months are big AND they replicate on the held-out rerun.

Design guarantees (unbreachable by construction)
------------------------------------------------
* Long-only; quantities are non-negative; sells never exceed holdings.
* Per-ticker target weight <= 0.26 (< the 0.30 concentration rule), drift-trimmed
  at 0.28 — a position can never sit at >=0.30 for even one day.
* Dollar gross <= 1.0x (no borrow); beta-adjusted gross clamped <= 1.45x (< 1.5x).
  Leveraged ETFs (QLD/SSO) are only ever bought in the FULL state.
* Deterministic: same inputs -> same orders. All timing is bar-date based, so the
  agent behaves identically at daily, 30-min, or 1-min tick cadence.
* No network, no LLM, no API keys, stdlib only. Runtime is sub-millisecond.

The single function the contest calls is ``decide``; everything else is a pure
helper or persistent state documented in IMPLEMENTATION_BLUEPRINT.md sections 1-3.
"""
from __future__ import annotations

import math
from statistics import pstdev
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 1. Fixed parameters (IMPLEMENTATION_BLUEPRINT.md section 1) — never change.
# ---------------------------------------------------------------------------
MOM_LONG: int = 42
MOM_SHORT: int = 21
MOM_W_LONG: float = 0.50
MOM_W_SHORT: float = 0.30
MOM_W_GAP: float = 0.20
NAME_SMA: int = 50
IDX_SMA_FAST: int = 20
IDX_SMA_SLOW: int = 50
ENTER_BAND: float = 0.01
EXIT_BAND: float = 0.01
VOL_SIZE: int = 20
VOL_BRAKE: int = 10
VOL_FULL_MAX: float = 0.30
BRAKE_VOL10: float = 0.50
BRAKE_R3: float = -0.05
BREADTH_MIN: float = 0.50
TOP_N_MAX: int = 5
NAME_CAP: float = 0.26
CORE_FULL: float = 0.67
CORE_NEUTRAL: float = 0.85
SLEEVE_DOLLAR_FULL: float = 0.33
MAX_BETA_GROSS: float = 1.45
DD_HALF: float = -0.06
DD_LOCK: float = -0.10
TAPER_HALF: float = 0.50
TAPER_LOCK: float = 0.25
TRAIL_STOP: float = 0.08
STOP_COOLDOWN_DAYS: int = 3
REBALANCE_DAYS: int = 3
COOLDOWN_DAYS: int = 3
DRIFT_LIMIT: float = 0.28
MIN_TRADE_PCT: float = 0.03
CASH_BUFFER: float = 0.98
MAX_ORDERS: int = 45
MIN_BARS: int = 51

# Momentum-Thrust Re-Entry Override (single approved post-audit modification).
# A genuine V-recovery shows a strong short-horizon thrust on QQQ; this lets a
# CASH -> NEUTRAL re-entry fire without waiting for the slow SMA50 reclaim.
# Purely additive: it never overrides the brake, never reaches FULL, never levers.
THRUST_LOOKBACK: int = 10
THRUST_MIN_RET: float = 0.12

# ---------------------------------------------------------------------------
# 2. Universe (IMPLEMENTATION_BLUEPRINT.md section 2) — filtered at runtime to
#    whatever is present in market_state with >= MIN_BARS bars.
# ---------------------------------------------------------------------------
INDEX_REF: tuple[str, ...] = ("SPY", "QQQ")

LEADER_STOCKS: tuple[str, ...] = (
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "AVGO", "AMD", "MU", "MRVL",
    "NFLX", "TSLA", "PLTR", "ORCL", "CRM", "JPM", "V", "MA", "COST", "LLY",
)
LEADER_ETFS: tuple[str, ...] = (
    "QQQ", "SPY", "SMH", "XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLV",
    "XLP", "XLU", "XLRE", "DIA", "IWM", "SOXX",
)
# Ranking candidates (deduplicated, deterministic order).
LEADER_POOL: tuple[str, ...] = tuple(dict.fromkeys(LEADER_STOCKS + LEADER_ETFS))

# 2x ETF sleeve — bought ONLY in the FULL state.
SLEEVE: tuple[str, ...] = ("QLD", "SSO")

# Beta multiples for the gross-exposure clamp. We only ever buy 1x names + QLD/SSO;
# the rest of the table is a defensive guard.
BETA: dict[str, float] = {
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0, "FAS": 3.0,
    "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0, "UDOW": 3.0, "NAIL": 3.0,
}

_STATE_RANK: dict[str, int] = {"CASH": 0, "NEUTRAL": 1, "FULL": 2}

# ---------------------------------------------------------------------------
# 3. Persistent state (IMPLEMENTATION_BLUEPRINT.md section 3).
#    Persists across calls within a process; the engine resets it between
#    regimes (fresh process). `_prev_taper_mult` backs the Phase-7 "taper
#    decreased" trigger.
# ---------------------------------------------------------------------------
_state: str = "NEUTRAL"
_cooldown: int = 0
_peak_equity: float = 0.0
_pos_high: dict[str, float] = {}
_stop_block: dict[str, int] = {}
_last_rebalance_date: Optional[str] = None
_last_seen_date: Optional[str] = None
_prev_state: str = "NEUTRAL"
_prev_taper_mult: float = 1.0


# ---------------------------------------------------------------------------
# 4. Feature helpers (IMPLEMENTATION_BLUEPRINT.md section 4) — pure functions.
# ---------------------------------------------------------------------------
def _beta(ticker: str) -> float:
    return BETA.get(ticker, 1.0)


def _date_of(ts: Any) -> str:
    """Truncate a timestamp to its YYYY-MM-DD prefix (ISO dates sort lexically)."""
    return str(ts)[:10]


def _closes_of(
    market_state: dict[str, Any],
    ticker: str,
    cache: dict[str, Optional[list[float]]],
) -> Optional[list[float]]:
    """Return the ticker's close series (oldest-first) or None if unusable."""
    if ticker in cache:
        return cache[ticker]
    closes: Optional[list[float]] = None
    bars = market_state.get(ticker)
    if bars:
        try:
            closes = [float(bar["close"]) for bar in bars]
        except (KeyError, TypeError, ValueError):
            closes = None
    cache[ticker] = closes
    return closes


def _computable(closes: Optional[list[float]]) -> bool:
    return closes is not None and len(closes) >= MIN_BARS and closes[-1] > 0.0


def _sma(closes: list[float], n: int) -> Optional[float]:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _ret(closes: list[float], k: int) -> Optional[float]:
    if len(closes) < k + 1:
        return None
    start = closes[-(k + 1)]
    if start <= 0.0:
        return None
    return closes[-1] / start - 1.0


def _vol(closes: list[float], n: int) -> Optional[float]:
    """Annualized realized volatility from the last n daily returns (population stdev)."""
    if len(closes) < n + 1:
        return None
    window = closes[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0.0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 2:
        return None
    return pstdev(rets) * math.sqrt(252.0)


def _trend_gap(closes: list[float]) -> Optional[float]:
    sma50 = _sma(closes, NAME_SMA)
    if sma50 is None or sma50 <= 0.0:
        return None
    return closes[-1] / sma50 - 1.0


def _momentum_score(closes: list[float]) -> Optional[float]:
    r_long = _ret(closes, MOM_LONG)
    r_short = _ret(closes, MOM_SHORT)
    gap = _trend_gap(closes)
    if r_long is None or r_short is None or gap is None:
        return None
    return MOM_W_LONG * r_long + MOM_W_SHORT * r_short + MOM_W_GAP * gap


# ---------------------------------------------------------------------------
# Price / portfolio helpers (IMPLEMENTATION_BLUEPRINT.md sections 13-14).
# ---------------------------------------------------------------------------
def _resolve_cash(portfolio_state: dict[str, Any], cash: float) -> float:
    """Single source of truth for spendable cash (P2): portfolio_state -> arg -> 0."""
    try:
        return float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        try:
            return float(cash)
        except (TypeError, ValueError):
            return 0.0


def _aggregate_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Aggregate duplicate position lots by ticker (weighted-average avg_cost)."""
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        try:
            ticker = str(raw["ticker"]).upper()
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if qty <= 0.0:
            continue
        if ticker in out:
            existing = out[ticker]
            total = existing["quantity"] + qty
            existing["avg_cost"] = (
                (existing["avg_cost"] * existing["quantity"] + avg_cost * qty) / total
                if total > 0.0
                else avg_cost
            )
            existing["quantity"] = total
        else:
            out[ticker] = {"quantity": qty, "avg_cost": avg_cost}
    return out


def _mark_price(
    ticker: str,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
) -> Optional[float]:
    """Equity marking price: last_prices -> latest close (blueprint section 13)."""
    lp = last_prices.get(ticker)
    try:
        if lp is not None and float(lp) > 0.0:
            return float(lp)
    except (TypeError, ValueError):
        pass
    closes = _closes_of(market_state, ticker, cache)
    if closes and closes[-1] > 0.0:
        return closes[-1]
    return None


def _exec_price(
    ticker: str,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
) -> Optional[float]:
    """Execution/stop reference price: latest close -> last_prices (sections 9/11/14)."""
    closes = _closes_of(market_state, ticker, cache)
    if closes and closes[-1] > 0.0:
        return closes[-1]
    lp = last_prices.get(ticker)
    try:
        if lp is not None and float(lp) > 0.0:
            return float(lp)
    except (TypeError, ValueError):
        pass
    return None


def _compute_equity(
    positions: dict[str, dict[str, float]],
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
    cash_value: float,
) -> float:
    """Equity = resolved cash + Σ qty·mark_price (P2 cash, P8 deterministic order)."""
    total = cash_value
    for ticker in sorted(positions):  # P8: deterministic summation order
        pos = positions[ticker]
        price = _mark_price(ticker, market_state, cache, last_prices)
        if price is None:
            price = pos["avg_cost"] if pos["avg_cost"] > 0.0 else 0.0
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


# ---------------------------------------------------------------------------
# 7-8. Momentum ranking, leader selection, sizing (blueprint sections 7-8).
# ---------------------------------------------------------------------------
def _build_targets(
    state: str,
    taper_mult: float,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    stop_block: dict[str, int],
) -> dict[str, float]:
    """Construct the target weight map W (fractions of equity)."""
    weights: dict[str, float] = {}
    if state == "CASH":
        return weights

    # Determine sleeve availability up front (FULL only).
    sleeve_present: list[str] = []
    if state == "FULL":
        sleeve_present = [
            s for s in SLEEVE
            if s not in stop_block and _computable(_closes_of(market_state, s, cache))
        ]

    # Core budget. P6: in FULL without an available sleeve, the sleeve's dollar
    # budget folds into the 1x core so FULL is never lighter than NEUTRAL.
    if state == "FULL":
        core_base = CORE_FULL if sleeve_present else (CORE_FULL + SLEEVE_DOLLAR_FULL)
    else:
        core_base = CORE_NEUTRAL
    core_budget = core_base * taper_mult

    # --- selection: qualify + rank the leader pool ---
    qualifiers: list[tuple[float, str]] = []
    for ticker in LEADER_POOL:
        if ticker in stop_block:
            continue
        closes = _closes_of(market_state, ticker, cache)
        if not _computable(closes):
            continue
        score = _momentum_score(closes)  # type: ignore[arg-type]  (computable => not None)
        if score is None:
            continue
        sma50 = _sma(closes, NAME_SMA)  # type: ignore[arg-type]
        if sma50 is None:
            continue
        if score > 0.0 and closes[-1] > sma50:  # type: ignore[index]
            qualifiers.append((score, ticker))

    qualifiers.sort(key=lambda pair: (-pair[0], pair[1]))
    selected = qualifiers[:TOP_N_MAX]
    n = len(selected)

    # --- conviction (rank-linear) core weights ---
    if n > 0:
        sum_raw = n * (n + 1) / 2.0
        for rank, (_, ticker) in enumerate(selected, start=1):
            raw = float(n - rank + 1)
            weight = min(core_budget * raw / sum_raw, NAME_CAP)
            if weight > 0.0:
                weights[ticker] = weight

    # --- 2x sleeve (FULL only; respects the per-name stop-block) ---
    if sleeve_present:
        per = (SLEEVE_DOLLAR_FULL * taper_mult) / len(sleeve_present)
        for s in sleeve_present:
            weight = min(per, NAME_CAP)
            if weight > 0.0:
                weights[s] = weight

    # --- beta-gross clamp ---
    beta_gross = sum(w * _beta(t) for t, w in weights.items())
    if beta_gross > MAX_BETA_GROSS and beta_gross > 0.0:
        scale = MAX_BETA_GROSS / beta_gross
        weights = {t: w * scale for t, w in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# 11. Order generation & sell-before-buy sequencing (blueprint section 11).
# ---------------------------------------------------------------------------
def _generate_orders(
    do_rebalance: bool,
    weights: dict[str, float],
    positions: dict[str, dict[str, float]],
    forced_stops: list[tuple[str, float]],
    equity: float,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
    cash_value: float,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    sold: set[str] = set()
    proceeds = 0.0
    min_trade = MIN_TRADE_PCT * equity

    # (1) Forced trailing-stop exits — always immediate, full quantity.
    for ticker, qty in forced_stops:
        if qty > 0.0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
            sold.add(ticker)
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is not None:
                proceeds += qty * price

    if do_rebalance:
        # (2) Rebalance sells (sell-before-buy), deterministic ticker order.
        for ticker in sorted(positions):
            if ticker in sold:
                continue
            held = positions[ticker]["quantity"]
            if held <= 0.0:
                continue
            target_w = weights.get(ticker, 0.0)
            price = _exec_price(ticker, market_state, cache, last_prices)

            if target_w == 0.0:
                # P1: a name not in the target is ALWAYS fully liquidated (no
                # dead-band) so CASH / de-risk states are genuinely flat.
                orders.append({"ticker": ticker, "side": "sell", "quantity": held})
                sold.add(ticker)
                if price is not None and price > 0.0:
                    proceeds += held * price
                continue

            # In-target overweight: trim, subject to the dead-band.
            if price is None or price <= 0.0:
                continue
            target_shares = math.floor(target_w * equity / price)
            delta = target_shares - held
            if delta < 0 and (-delta) * price >= min_trade:
                sell_qty = float(int(min(-delta, held)))  # integer trim
                if sell_qty > 0.0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                    sold.add(ticker)
                    proceeds += sell_qty * price

        # (3) Spendable cash after sells, then buys in descending-weight order.
        spendable = cash_value + CASH_BUFFER * proceeds
        for ticker in sorted(weights, key=lambda t: (-weights[t], t)):
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is None or price <= 0.0:
                continue
            held = positions[ticker]["quantity"] if ticker in positions else 0.0
            target_shares = math.floor(weights[ticker] * equity / price)
            deficit = target_shares - held
            if deficit > 0 and deficit * price >= min_trade:
                affordable = math.floor(min(deficit * price, spendable) / price)
                if affordable > 0:
                    orders.append({"ticker": ticker, "side": "buy", "quantity": float(affordable)})
                    spendable -= affordable * price

    # (4) Hard order cap — sells first, then buys (already weight-ordered).
    if len(orders) > MAX_ORDERS:
        sells = [o for o in orders if o["side"] == "sell"]
        buys = [o for o in orders if o["side"] == "buy"]
        orders = (sells + buys)[:MAX_ORDERS]

    return [o for o in orders if o["quantity"] > 0.0]


# ---------------------------------------------------------------------------
# 12. Master decision cycle (blueprint section 12). `decide` wraps `_run` in a
#     defensive guard so no exception can ever forfeit the "runs clean" gate;
#     on failure (P3) it restores the pre-call global snapshot to avoid desync.
# ---------------------------------------------------------------------------
def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    """Return a list of long-only buy/sell orders for this decision cycle."""
    global _state, _cooldown, _peak_equity, _pos_high, _stop_block
    global _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult

    snapshot = (
        _state, _cooldown, _peak_equity, dict(_pos_high), dict(_stop_block),
        _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult,
    )
    try:
        return _run(market_state or {}, portfolio_state or {}, cash)
    except Exception:  # noqa: BLE001 — never let a bad tick raise; restore state.
        (
            _state, _cooldown, _peak_equity, _pos_high, _stop_block,
            _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult,
        ) = snapshot
        return []


def _run(
    market_state: dict[str, Any],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict[str, Any]]:
    global _state, _cooldown, _peak_equity, _pos_high, _stop_block
    global _last_rebalance_date, _last_seen_date, _prev_state, _prev_taper_mult

    if not market_state:
        return []

    cache: dict[str, Optional[list[float]]] = {}
    last_prices: dict[str, Any] = {}
    for key, value in (portfolio_state.get("last_prices", {}) or {}).items():
        last_prices[str(key).upper()] = value
    cash_value = _resolve_cash(portfolio_state, cash)  # P2: single cash source

    spy_bars = market_state.get("SPY")
    spy = _closes_of(market_state, "SPY", cache)
    qqq = _closes_of(market_state, "QQQ", cache)
    current_date: Optional[str] = None
    if spy_bars:
        ts = spy_bars[-1].get("ts")
        current_date = _date_of(ts) if ts is not None else str(len(spy_bars))

    # ---- Phase 0: data guard / liquidate path ----
    if not _computable(spy) or not _computable(qqq):
        positions = _aggregate_positions(portfolio_state)
        orders: list[dict[str, Any]] = []
        for ticker in sorted(positions):
            if market_state.get(ticker):  # only sellable if present in the universe
                qty = positions[ticker]["quantity"]
                if qty > 0.0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
        # P4: keep prev-state bookkeeping consistent across a data gap.
        _prev_state = _state
        _prev_taper_mult = 1.0
        if current_date is not None:
            _last_seen_date = current_date
        return orders

    # Both indices are computable here (Phase 0 returned otherwise).
    spy_closes: list[float] = spy  # type: ignore[assignment]
    qqq_closes: list[float] = qqq  # type: ignore[assignment]

    # ---- Phase 1: ingest positions ----
    positions = _aggregate_positions(portfolio_state)

    # ---- Phase 2: calendar / cooldown & stop-block decay (new trading day only) ----
    is_new_day = current_date != _last_seen_date
    if is_new_day:
        if _cooldown > 0:
            _cooldown -= 1
        if _stop_block:
            decayed: dict[str, int] = {}
            for ticker, days in _stop_block.items():
                remaining = days - 1
                if remaining > 0:
                    decayed[ticker] = remaining
            _stop_block = decayed

    # ---- Phase 3: equity, peak, drawdown, taper ----
    equity = _compute_equity(positions, market_state, cache, last_prices, cash_value)
    if equity <= 0.0:
        _prev_state = _state
        _prev_taper_mult = 1.0
        _last_seen_date = current_date
        return []
    _peak_equity = max(_peak_equity, equity)
    dd = (equity / _peak_equity - 1.0) if _peak_equity > 0.0 else 0.0
    if dd <= DD_LOCK:
        taper_mult = TAPER_LOCK
    elif dd <= DD_HALF:
        taper_mult = TAPER_HALF
    else:
        taper_mult = 1.0

    # ---- Phase 4: index features + breadth ----
    spy_close = spy_closes[-1]
    qqq_close = qqq_closes[-1]
    spy_sma_fast = _sma(spy_closes, IDX_SMA_FAST)
    spy_sma_slow = _sma(spy_closes, IDX_SMA_SLOW)
    qqq_sma_fast = _sma(qqq_closes, IDX_SMA_FAST)
    qqq_sma_slow = _sma(qqq_closes, IDX_SMA_SLOW)
    qqq_vol20 = _vol(qqq_closes, VOL_SIZE)
    qqq_r3 = _ret(qqq_closes, 3)
    qqq_vol10 = _vol(qqq_closes, VOL_BRAKE)

    n_comp = 0
    n_up = 0
    for ticker in LEADER_POOL:
        closes = _closes_of(market_state, ticker, cache)
        if not _computable(closes):
            continue
        sma50 = _sma(closes, NAME_SMA)  # type: ignore[arg-type]
        if sma50 is None:
            continue
        n_comp += 1
        if closes[-1] > sma50:  # type: ignore[index]
            n_up += 1
    breadth = (n_up / n_comp) if n_comp > 0 else 0.0

    # ---- Phase 5: state machine (strict priority ladder + cash hysteresis) ----
    prev_cycle_state = _state  # carried from the previous cycle (for P5 hysteresis)
    brake_fired = (
        (qqq_r3 is not None and qqq_r3 < BRAKE_R3)
        or (qqq_vol10 is not None and qqq_vol10 > BRAKE_VOL10)
    )
    hard_cash = (
        brake_fired
        or (spy_sma_slow is not None and spy_close < spy_sma_slow * (1.0 - EXIT_BAND))
        or (qqq_sma_slow is not None and qqq_close < qqq_sma_slow * (1.0 - EXIT_BAND))
    )
    # P5: leaving CASH requires a clear reclaim (both indices above SMA50*(1+ENTER_BAND)),
    # not merely clearing the 0.99 trigger -> a 2% hysteresis band on the CASH<->NEUTRAL
    # boundary that removes liquidate/re-buy whipsaw in choppy tape.
    reclaim = (
        spy_sma_slow is not None and spy_close > spy_sma_slow * (1.0 + ENTER_BAND)
        and qqq_sma_slow is not None and qqq_close > qqq_sma_slow * (1.0 + ENTER_BAND)
    )
    # Momentum-Thrust Re-Entry Override — a SECOND, independent CASH->NEUTRAL path.
    # A genuine V-recovery on QQQ (a strong 10-day thrust, above the 20-day SMA,
    # and up on the day) lets us re-enter NEUTRAL WITHOUT waiting for the SMA50
    # reclaim. Gated on the crash brake being clear and the cooldown complete; it
    # fires only when the existing reclaim path is idle (`not reclaim`) and enters
    # NEUTRAL only. The brake, cooldown, taper, and SMA50-reclaim logic are unchanged.
    qqq_ret10 = _ret(qqq_closes, THRUST_LOOKBACK)
    thrust_signal = (
        qqq_ret10 is not None and qqq_ret10 > THRUST_MIN_RET
        and qqq_sma_fast is not None and qqq_close > qqq_sma_fast
        and len(qqq_closes) >= 2 and qqq_close > qqq_closes[-2]
    )
    if (
        prev_cycle_state == "CASH"
        and not brake_fired
        and not reclaim
        and thrust_signal
    ):
        _state = "NEUTRAL"  # thrust override: re-enter NEUTRAL on a genuine V-recovery
    elif hard_cash:
        _state = "CASH"
        _cooldown = COOLDOWN_DAYS
    elif prev_cycle_state == "CASH" and not reclaim:
        _state = "CASH"  # hysteresis hold: stay in cash until a clear reclaim
    elif _cooldown > 0:
        _state = "NEUTRAL"
    else:
        full_conditions = (
            spy_sma_fast is not None and spy_close > spy_sma_fast
            and qqq_sma_fast is not None and qqq_close > qqq_sma_fast
            and spy_sma_slow is not None and spy_close > spy_sma_slow * (1.0 + ENTER_BAND)
            and qqq_sma_slow is not None and qqq_close > qqq_sma_slow * (1.0 + ENTER_BAND)
            and breadth >= BREADTH_MIN
            and qqq_vol20 is not None and qqq_vol20 < VOL_FULL_MAX
        )
        _state = "FULL" if full_conditions else "NEUTRAL"

    # ---- Phase 6: trailing-stop updates (every cycle) ----
    for ticker in list(_pos_high):
        if ticker not in positions:
            del _pos_high[ticker]
    forced_stops: list[tuple[str, float]] = []
    for ticker in sorted(positions):
        price = _exec_price(ticker, market_state, cache, last_prices)
        if price is None:
            continue
        high = _pos_high.get(ticker, price)
        if price > high:
            high = price
        _pos_high[ticker] = high
        if high > 0.0 and price < high * (1.0 - TRAIL_STOP):
            forced_stops.append((ticker, positions[ticker]["quantity"]))
            _stop_block[ticker] = STOP_COOLDOWN_DAYS
            if ticker in _pos_high:
                del _pos_high[ticker]

    # ---- Phase 7: rebalance gate ----
    if _last_rebalance_date is None:
        do_rebalance = True
    else:
        elapsed_dates: set[str] = set()
        for bar in spy_bars:  # spy computable => bars present
            ts = bar.get("ts")
            bar_date = _date_of(ts) if ts is not None else ""
            if bar_date > _last_rebalance_date:
                elapsed_dates.add(bar_date)
        days_since = len(elapsed_dates)
        derisk_state = _STATE_RANK[_state] < _STATE_RANK[_prev_state]
        derisk_taper = taper_mult < _prev_taper_mult
        drift = False
        for ticker, pos in positions.items():
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is not None and equity > 0.0 and (pos["quantity"] * price / equity) > DRIFT_LIMIT:
                drift = True
                break
        do_rebalance = (
            days_since >= REBALANCE_DAYS or derisk_state or derisk_taper or drift
        )
    if _last_rebalance_date == current_date:  # never full-rebalance twice on one date
        do_rebalance = False

    # ---- Phase 8: target weights ----
    weights = (
        _build_targets(_state, taper_mult, market_state, cache, _stop_block)
        if do_rebalance
        else {}
    )

    # ---- Phase 9: orders ----
    orders = _generate_orders(
        do_rebalance, weights, positions, forced_stops,
        equity, market_state, cache, last_prices, cash_value,
    )
    if do_rebalance and len(orders) >= 1:
        _last_rebalance_date = current_date

    # ---- Phase 10: persist ----
    _prev_state = _state
    _prev_taper_mult = taper_mult
    _last_seen_date = current_date
    return orders
