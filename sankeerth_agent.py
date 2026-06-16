"""CDaR (Conditional Drawdown-at-Risk) allocator.

Long-only ETF / mega-cap portfolio that explicitly minimizes recent portfolio
drawdown -- the denominator of the Calmar ratio -- by solving the CDaR linear
program (Chekhlov-Uryasev epigraph form) with a projected subgradient, then
stepping to a defensive book when SPY/QQQ lose their 50-day trend. Every weight
is capped at 24% and beta-adjusted gross at 1.35x. Pure standard library.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

RISK_CANDIDATES = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
    "SMH", "KRE",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO",
)
DEFENSIVE_WEIGHTS = (("XLP", 0.30), ("XLU", 0.30), ("XLV", 0.24), ("XLE", 0.16))

BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0, "FAS": 3.0,
    "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0, "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

LOOKBACK = 120
ALPHA = 0.10
RETURN_TILT = 0.15
OPT_ITERS = 150
MAX_WEIGHT = 0.24
MAX_BETA_GROSS = 1.35
MIN_TRADE_PCT = 0.015
DRIFT_LIMIT = 0.27
REBALANCE_EVERY_DAYS = 5
RISK_ON_EXPOSURE = 0.95
VOL_CEILING = 0.40

_last_rebalance_bar_date: str | None = None


def closes(bars: list[dict[str, Any]] | None) -> list[float]:
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
    if len(values) <= n or values[-(n + 1)] <= 0:
        return None
    return values[-1] / values[-(n + 1)] - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    w = values[-(n + 1):]
    rets = [w[i] / w[i - 1] - 1.0 for i in range(1, len(w)) if w[i - 1] > 0]
    if len(rets) < 5:
        return None
    return pstdev(rets) * math.sqrt(252.0)


def _aligned_returns(market_state: dict, tickers: list[str], lookback: int):
    series: dict[str, list[float]] = {}
    for t in tickers:
        cs = closes(market_state.get(t))
        if len(cs) < lookback + 2:
            continue
        w = cs[-(lookback + 1):]
        rets = [w[i] / w[i - 1] - 1.0 for i in range(1, len(w)) if w[i - 1] > 0]
        if len(rets) >= lookback - 2:
            series[t] = rets
    if not series:
        return 0, {}
    T = min(len(r) for r in series.values())
    return T, {t: r[-T:] for t, r in series.items()}


def _project_capped_simplex(w: dict[str, float], cap: float, budget: float) -> dict[str, float]:
    keys = list(w)
    lo = min(w.values()) - budget
    hi = max(w.values())
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = sum(min(cap, max(0.0, w[k] - mid)) for k in keys)
        if s > budget:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2.0
    return {k: min(cap, max(0.0, w[k] - tau)) for k in keys}


def _cdar_weights(series: dict[str, list[float]], tickers: list[str],
                  cap: float, budget: float, alpha: float,
                  lam: float, iters: int) -> dict[str, float]:
    T = len(series[tickers[0]])
    n = len(tickers)
    if n == 1:
        return {tickers[0]: min(cap, budget)}

    X = {t: [0.0] * (T + 1) for t in tickers}
    for t in tickers:
        r = series[t]
        acc = 0.0
        for k in range(T):
            acc += r[k]
            X[t][k + 1] = acc
    totals = {t: X[t][T] for t in tickers}

    w = _project_capped_simplex({t: budget / n for t in tickers}, cap, budget)
    k_tail = max(1, int(math.ceil(alpha * T)))
    base_eta = 2.0
    best_w, best_obj = dict(w), None

    for it in range(iters):
        cum = [0.0] * (T + 1)
        for k in range(1, T + 1):
            cum[k] = sum(w[t] * X[t][k] for t in tickers)
        peak_idx = [0] * (T + 1)
        best_k = 0
        for k in range(1, T + 1):
            if cum[k] >= cum[best_k]:
                best_k = k
            peak_idx[k] = best_k
        dd = [cum[peak_idx[k]] - cum[k] for k in range(T + 1)]

        tail = sorted(range(1, T + 1), key=lambda k: dd[k], reverse=True)[:k_tail]
        cdar = sum(dd[k] for k in tail) / k_tail
        obj = cdar - lam * sum(w[t] * totals[t] for t in tickers)
        if best_obj is None or obj < best_obj:
            best_obj, best_w = obj, dict(w)

        grad = {t: 0.0 for t in tickers}
        for k in tail:
            p = peak_idx[k]
            for t in tickers:
                grad[t] += (X[t][p] - X[t][k]) / k_tail
        for t in tickers:
            grad[t] -= lam * totals[t]

        eta = base_eta / math.sqrt(it + 1)
        nxt = {t: w[t] - eta * grad[t] for t in tickers}
        w = _project_capped_simplex(nxt, cap, budget)

    return best_w


def _risk_off_targets(market_state: dict) -> dict[str, float]:
    avail = [(t, wt) for t, wt in DEFENSIVE_WEIGHTS if closes(market_state.get(t))]
    if not avail:
        return {}
    s = sum(wt for _, wt in avail)
    return {t: 0.50 * (wt / s) for t, wt in avail}


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    bg = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if bg > MAX_BETA_GROSS:
        s = MAX_BETA_GROSS / bg
        capped = {t: w * s for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def target_weights(market_state: dict) -> dict[str, float]:
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < 50 or len(qqq) < 50:
        return {}

    qqq_vol = realized_vol(qqq, 20)
    risk_on = bool(
        spy[-1] > (sma(spy, 50) or 1e18)
        and qqq[-1] > (sma(qqq, 50) or 1e18)
        and qqq_vol is not None and qqq_vol < VOL_CEILING
    )
    if not risk_on:
        return _scale_caps(_risk_off_targets(market_state))

    candidates = [t for t in RISK_CANDIDATES if closes(market_state.get(t))]
    T, series = _aligned_returns(market_state, candidates, LOOKBACK)
    if T < 30 or len(series) < 2:
        return _scale_caps(_risk_off_targets(market_state))

    pool = sorted(t for t in series if (momentum(closes(market_state[t]), 60) or -1) > 0)
    if len(pool) < 2:
        return _scale_caps(_risk_off_targets(market_state))
    series = {t: series[t] for t in pool}

    raw = _cdar_weights(series, pool, MAX_WEIGHT, RISK_ON_EXPOSURE,
                        ALPHA, RETURN_TILT, OPT_ITERS)
    return _scale_caps(raw)


def current_positions(portfolio_state: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        t = str(raw.get("ticker", "")).upper()
        if not t:
            continue
        try:
            qty, ac = float(raw.get("quantity", 0.0)), float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        e = out.setdefault(t, {"quantity": 0.0, "avg_cost": ac})
        e["quantity"] += qty
        e["avg_cost"] = ac or e["avg_cost"]
    return out


def equity(portfolio_state: dict, cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last = portfolio_state.get("last_prices", {}) or {}
    for t, pos in current_positions(portfolio_state).items():
        try:
            px = float(last.get(t, pos["avg_cost"]))
        except (TypeError, ValueError):
            px = pos["avg_cost"]
        total += pos["quantity"] * max(px, 0.0)
    return max(total, 0.0)


def _latest_bar_date(market_state: dict) -> str | None:
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    if not bars:
        return None
    return str(bars[-1].get("ts", len(bars)))[:10]


def _days_since_rebalance(market_state: dict) -> int | None:
    if _last_rebalance_bar_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_bar_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_bar_date) - 1


def _market_prices(market_state: dict) -> dict[str, float]:
    out = {}
    for t, bars in market_state.items():
        cs = closes(bars)
        if cs:
            out[t.upper()] = cs[-1]
    return out


def _has_drifted(portfolio_state: dict, total_equity: float) -> bool:
    if total_equity <= 0:
        return False
    last = portfolio_state.get("last_prices", {}) or {}
    for t, pos in current_positions(portfolio_state).items():
        try:
            px = float(last.get(t, pos["avg_cost"]))
        except (TypeError, ValueError):
            px = pos["avg_cost"]
        if px > 0 and pos["quantity"] * px / total_equity > DRIFT_LIMIT:
            return True
    return False


def orders_to_rebalance(targets, positions, total_equity, prices, cash_available):
    if total_equity <= 0:
        return []
    min_trade = total_equity * MIN_TRADE_PCT
    orders, proceeds = [], 0.0
    for t, pos in positions.items():
        px = prices.get(t)
        if not px or px <= 0:
            continue
        cur = pos["quantity"] * px
        tgt = total_equity * targets.get(t, 0.0)
        if t not in targets:
            q = int(pos["quantity"])
            if q > 0 and cur >= min_trade:
                orders.append({"ticker": t, "side": "sell", "quantity": q})
                proceeds += q * px
        elif tgt - cur < -min_trade:
            q = min(int((cur - tgt) // px), int(pos["quantity"]))
            if q > 0:
                orders.append({"ticker": t, "side": "sell", "quantity": q})
                proceeds += q * px
    spendable = max(float(cash_available), 0.0) + proceeds * 0.98
    for t, w in sorted(targets.items()):
        px = prices.get(t)
        if not px or px <= 0:
            continue
        cur = positions.get(t, {}).get("quantity", 0.0) * px
        delta = total_equity * w - cur
        if delta < min_trade:
            continue
        q = int(min(delta, spendable) // px)
        if q > 0:
            orders.append({"ticker": t, "side": "buy", "quantity": q})
            spendable -= q * px
    return orders[:45]


def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    global _last_rebalance_bar_date
    if not market_state:
        return []
    latest = _latest_bar_date(market_state)
    if latest is None:
        return []
    total_equity = equity(portfolio_state, cash)
    days = _days_since_rebalance(market_state)
    should = (_last_rebalance_bar_date is None or days is None
              or days >= REBALANCE_EVERY_DAYS or _has_drifted(portfolio_state, total_equity))
    if not should:
        return []
    targets = target_weights(market_state)
    if not targets:
        return []
    orders = orders_to_rebalance(targets, current_positions(portfolio_state),
                                 total_equity, _market_prices(market_state), cash)
    if orders:
        _last_rebalance_bar_date = latest
    return orders
