"""Calmar Warming candidate agent.

Extends calmar_defense_plus with a "warming" regime between soft defensive
and full risk-on. Warming blends defensive sectors with broad beta during
recovery phases to capture rebound P&L without sacrificing Calmar.

No network, no LLM, no third-party dependencies.
"""
from __future__ import annotations

from statistics import pstdev


RISK_ON = (
    "SPY", "QQQ", "SMH", "XLK", "XLV", "XLY", "XLC", "XLF",
    "NVDA", "AMD", "AVGO", "MU", "MRVL", "AAPL", "MSFT", "GOOGL",
    "META", "AMZN", "PLTR", "TSLA",
)
DEFENSIVE_CORE = ("XLP", "XLV", "XLRE", "XLF", "XLE", "XLU")
HARD_CORE = ("XLP", "XLV", "XLRE", "XLU")
STABILITY_SLEEVE = ("LMT", "RTX", "NOC")
RATE_HEDGES = ("TLT", "GLD")
WARMING_BASKET = ("SPY", "QQQ", "XLK", "SMH", "XLV", "XLP", "XLRE",
                   "NVDA", "AAPL", "MSFT")

NAME_CAP = 0.25
DEF_CAP = 0.14
STABILITY_CAP = 0.035
GROSS_MAX = 1.00
SOFT_GROSS = 0.60
HARD_GROSS = 0.32
PANIC_GROSS = 0.30
WARMING_GROSS = 0.80
REBALANCE_EVERY = 4
DEAD_BAND = 0.018

VOL_LOOKBACK = 20
TARGET_VOL = 0.18
PORT_VOL_FLOOR = 0.06
PORT_VOL_CEILING = 0.45

MOM_LONG = 63
MOM_SHORT = 20
MOM_SKIP = 5
TREND = 50
TOP_N = 5

BRAKE_R1 = -0.020
BRAKE_R3 = -0.042
BRAKE_VOL_10D = 0.42
BRAKE_COOLDOWN = 2
PANIC_RET = -0.10
PANIC_VOL = 0.30

DD1 = 0.018
DD2 = 0.032
DD3 = 0.055

_ANN = 252 ** 0.5
_tick = 0
_last_rebalance = -10 ** 9
_last_regime = None
_brake_cooldown = 0
_peak_equity = 0.0
_pending_regime = None
_pending_count = 0
_current_regime = "soft"


def _closes(bars):
    out = []
    for bar in bars or []:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def _sma(closes, n):
    return sum(closes[-n:]) / n if len(closes) >= n else None


def _ret(closes, days, skip=0):
    if len(closes) < days + skip + 1:
        return None
    start = closes[-(days + skip + 1)]
    end = closes[-(skip + 1)] if skip else closes[-1]
    return end / start - 1.0 if start > 0 else None


def _vol(closes, n):
    if len(closes) < n + 1:
        return None
    rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] <= 0:
            return None
        rets.append(closes[i] / closes[i - 1] - 1.0)
    return pstdev(rets) * _ANN if len(rets) > 1 else None


def _positions(portfolio_state):
    out = {}
    for pos in portfolio_state.get("positions", []) or []:
        ticker = str(pos.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty = float(pos.get("quantity", 0.0))
            avg_cost = float(pos.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty > 0:
            out[ticker] = {"quantity": qty, "avg_cost": avg_cost}
    return out


def _equity(portfolio_state, cash):
    last = portfolio_state.get("last_prices", {}) or {}
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    for ticker, pos in _positions(portfolio_state).items():
        try:
            price = float(last.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0:
            total += pos["quantity"] * price
    return max(total, 0.0)


def _raw_regime(market_state):
    spy = _closes(market_state.get("SPY") or [])
    qqq = _closes(market_state.get("QQQ") or [])
    if len(spy) < 60 or len(qqq) < 60:
        return "soft"

    r1 = _ret(qqq, 1)
    r3 = _ret(qqq, 3)
    v10 = _vol(qqq, 10)
    if (r1 is not None and r1 < BRAKE_R1) or (r3 is not None and r3 < BRAKE_R3) or (v10 is not None and v10 > BRAKE_VOL_10D):
        return "hard"

    spy_6m = _ret(spy, 126)
    spy_v20 = _vol(spy, 20)
    if spy_6m is not None and spy_v20 is not None and spy_6m < PANIC_RET and spy_v20 > PANIC_VOL:
        return "panic"

    spy_50 = _sma(spy, TREND)
    qqq_50 = _sma(qqq, TREND)
    spy_200 = _sma(spy, 200)
    qqq_v20 = _vol(qqq, 20)

    if (spy_50 is not None and qqq_50 is not None and qqq_v20 is not None
            and spy[-1] > spy_50 * 1.004 and qqq[-1] > qqq_50 * 1.004
            and qqq_v20 < 0.35 and (spy_200 is None or spy[-1] > spy_200)):
        return "on"

    spy_10 = _sma(spy, 10)
    qqq_10 = _sma(qqq, 10)
    qqq_3d = _ret(qqq, 3)
    if (spy_10 is not None and qqq_10 is not None
            and spy[-1] > spy_10 and qqq[-1] > qqq_10
            and qqq_3d is not None and qqq_3d >= -0.01
            and qqq_v20 is not None and qqq_v20 < 0.45):
        return "warming"

    return "soft"


def _confirm_regime(raw):
    global _current_regime, _pending_regime, _pending_count
    if raw in ("hard", "panic"):
        _current_regime = raw
        _pending_regime = None
        _pending_count = 0
        return raw
    if raw == _current_regime:
        _pending_regime = None
        _pending_count = 0
        return _current_regime
    confirm = 1 if _current_regime in ("on", "warming") else 2
    if raw == _pending_regime:
        _pending_count += 1
    else:
        _pending_regime = raw
        _pending_count = 1
    if _pending_count >= confirm:
        _current_regime = _pending_regime
        _pending_regime = None
        _pending_count = 0
    return _current_regime


def _dd_scale(equity):
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    if _peak_equity <= 0:
        return 1.0
    dd = 1.0 - equity / _peak_equity
    if dd >= DD3:
        return 0.10
    if dd >= DD2:
        return 0.30
    if dd >= DD1:
        return 0.60
    return 1.0


def _score(market_state, tickers, defensive=False):
    scored = []
    for ticker in tickers:
        closes = _closes(market_state.get(ticker) or [])
        if len(closes) < MOM_LONG + MOM_SKIP + 1:
            continue
        r63 = _ret(closes, MOM_LONG, MOM_SKIP)
        r20 = _ret(closes, MOM_SHORT)
        sma = _sma(closes, TREND)
        vol = _vol(closes, VOL_LOOKBACK)
        if r63 is None or r20 is None or sma is None or vol is None or vol <= 0:
            continue
        trend_gap = closes[-1] / sma - 1.0
        raw = 0.52 * r63 + 0.35 * r20 + 0.13 * trend_gap
        if defensive:
            raw += 0.015
        elif closes[-1] <= sma or raw <= 0:
            continue
        if raw > -0.025:
            scored.append((raw / max(vol, 0.08), ticker))
    scored.sort(reverse=True)
    return scored


def _make_weights(scores, gross, cap, limit=None):
    selected = scores[:limit] if limit else scores
    if not selected or gross <= 0:
        return {}
    min_score = min(score for score, _ in selected)
    shifted = [(score - min_score + 0.01, ticker) for score, ticker in selected]
    total = sum(score for score, _ in shifted)
    weights = {ticker: min(cap, gross * score / total) for score, ticker in shifted if total > 0}
    for _ in range(4):
        unused = gross - sum(weights.values())
        if unused <= 1e-9:
            break
        room = {ticker: cap - weight for ticker, weight in weights.items() if weight < cap - 1e-9}
        room_total = sum(room.values())
        if room_total <= 0:
            break
        for ticker, room_amt in room.items():
            weights[ticker] = min(cap, weights[ticker] + unused * room_amt / room_total)
    return {ticker: weight for ticker, weight in weights.items() if weight > 0.002}


def _add_sleeve(targets, sleeve, gross):
    if not sleeve or gross <= 0:
        return
    total = sum(sleeve.values())
    if total <= 0:
        return
    for ticker, weight in sleeve.items():
        targets[ticker] = targets.get(ticker, 0.0) + weight * gross / total


def _estimate_vol(weights, market_state):
    if not weights:
        return TARGET_VOL
    total = 0.0
    weighted = 0.0
    for ticker, weight in weights.items():
        v = _vol(_closes(market_state.get(ticker) or []), VOL_LOOKBACK)
        if v and v > 0:
            weighted += weight * v
            total += weight
    return weighted / total if total > 0 else TARGET_VOL


def _finalize(weights, gross_cap, market_state):
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    port_vol = max(PORT_VOL_FLOOR, min(PORT_VOL_CEILING, _estimate_vol(weights, market_state)))
    target_gross = min(gross_cap, total * TARGET_VOL / port_vol)
    scale = target_gross / total
    out = {}
    for ticker, weight in weights.items():
        cap = STABILITY_CAP if ticker in STABILITY_SLEEVE else DEF_CAP if ticker in DEFENSIVE_CORE or ticker in RATE_HEDGES else NAME_CAP
        out[ticker] = min(cap, weight * scale)
    return {ticker: weight for ticker, weight in out.items() if weight > 0.002}


def _targets(market_state, equity, regime):
    global _brake_cooldown
    dd = _dd_scale(equity)

    if regime == "hard":
        _brake_cooldown = BRAKE_COOLDOWN
        core = _make_weights(_score(market_state, HARD_CORE, defensive=True), HARD_GROSS, DEF_CAP, 4)
        return _finalize(core, HARD_GROSS * dd, market_state)

    if _brake_cooldown > 0:
        _brake_cooldown -= 1
        core = _make_weights(_score(market_state, HARD_CORE, defensive=True), HARD_GROSS, DEF_CAP, 4)
        return _finalize(core, HARD_GROSS * dd, market_state)

    if regime == "panic":
        cap = PANIC_GROSS
        targets = _make_weights(_score(market_state, DEFENSIVE_CORE, defensive=True), cap * 0.82, DEF_CAP, 5)
        stabilizers = _make_weights(_score(market_state, STABILITY_SLEEVE, defensive=True), 1.0, STABILITY_CAP, 2)
        hedges = _make_weights(_score(market_state, RATE_HEDGES, defensive=True), 1.0, DEF_CAP, 1)
        _add_sleeve(targets, stabilizers, cap * 0.10)
        _add_sleeve(targets, hedges, cap * 0.08)
        return _finalize(targets, cap * dd, market_state)

    if regime == "soft":
        cap = SOFT_GROSS
        targets = _make_weights(_score(market_state, DEFENSIVE_CORE, defensive=True), cap * 0.85, DEF_CAP, 5)
        stabilizers = _make_weights(_score(market_state, STABILITY_SLEEVE, defensive=True), 1.0, STABILITY_CAP, 2)
        hedges = _make_weights(_score(market_state, RATE_HEDGES, defensive=True), 1.0, DEF_CAP, 1)
        _add_sleeve(targets, stabilizers, cap * 0.09)
        _add_sleeve(targets, hedges, cap * 0.06)
        return _finalize(targets, cap * dd, market_state)

    if regime == "warming":
        risk = _make_weights(_score(market_state, WARMING_BASKET), 0.90, NAME_CAP, 6)
        if not risk:
            core = _make_weights(_score(market_state, DEFENSIVE_CORE, defensive=True), 0.40, DEF_CAP, 4)
            return _finalize(core, 0.40 * dd, market_state)
        core = _make_weights(_score(market_state, DEFENSIVE_CORE, defensive=True), 1.0, DEF_CAP, 2)
        stabilizers = _make_weights(_score(market_state, STABILITY_SLEEVE, defensive=True), 1.0, STABILITY_CAP, 1)
        targets = dict(risk)
        _add_sleeve(targets, core, 0.06)
        _add_sleeve(targets, stabilizers, 0.02)
        return _finalize(targets, WARMING_GROSS * dd, market_state)

    risk = _make_weights(_score(market_state, RISK_ON), 0.98, NAME_CAP, TOP_N)
    if not risk:
        core = _make_weights(_score(market_state, DEFENSIVE_CORE, defensive=True), 0.40, DEF_CAP, 4)
        return _finalize(core, 0.40 * dd, market_state)
    core = _make_weights(_score(market_state, DEFENSIVE_CORE, defensive=True), 1.0, DEF_CAP, 2)
    stabilizers = _make_weights(_score(market_state, STABILITY_SLEEVE, defensive=True), 1.0, STABILITY_CAP, 1)
    targets = dict(risk)
    _add_sleeve(targets, core, 0.05)
    _add_sleeve(targets, stabilizers, 0.015)
    return _finalize(targets, GROSS_MAX * dd, market_state)


def _orders(targets, positions, market_state, equity, cash):
    prices = {}
    for ticker, bars in market_state.items():
        closes = _closes(bars)
        if closes:
            prices[str(ticker).upper()] = closes[-1]

    orders = []
    min_trade = DEAD_BAND * equity
    sell_proceeds = 0.0
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if not price:
            continue
        qty = pos["quantity"]
        delta_value = equity * targets.get(ticker, 0.0) - qty * price
        if ticker not in targets:
            sell_qty = int(qty)
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
        elif delta_value < -min_trade:
            sell_qty = min(int(abs(delta_value) // price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price

    spendable = max(float(cash or 0.0), 0.0) + sell_proceeds * 0.99
    for ticker, weight in sorted(targets.items(), key=lambda item: item[1], reverse=True):
        price = prices.get(ticker)
        if not price:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        delta_value = equity * weight - current_qty * price
        if delta_value < min_trade:
            continue
        buy_qty = int(min(delta_value, spendable) // price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price
    return orders[:45]


def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance, _last_regime
    _tick += 1
    equity = _equity(portfolio_state, cash)
    if equity <= 0:
        return []
    regime = _confirm_regime(_raw_regime(market_state))
    urgent = regime in ("hard", "panic") or _brake_cooldown > 0
    risk_off = _last_regime in ("on", "warming") and regime in ("soft", "hard", "panic")
    _last_regime = regime
    if not urgent and not risk_off and _tick - _last_rebalance < REBALANCE_EVERY:
        return []
    targets = _targets(market_state, equity, regime)
    orders = _orders(targets, _positions(portfolio_state), market_state, equity, cash)
    if orders:
        _last_rebalance = _tick
    return orders
