"""
agent.py — Optimal Conservative (builderr)
==========================================
Proven core: 3× momentum + TLT risk-off on SPY < 20d or VIX ≥ 25.

Targeted upgrades:
  • VIX 16–25: stay invested but scale 3× book to 70% (gradual de-risk)
  • Sector ETF fallback when no 3× name has positive momentum
  • Risk-off: 90% TLT (proven crash hedge)

No lookahead. Long-only, gross ≤ 1.45×, ≤ 25% per name.
"""
from __future__ import annotations

import math
from typing import Any

LEVERAGED_3X = ["TQQQ", "SOXL", "UPRO"]
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "SMH")
BENCHMARK = "SPY"
VIX_TICKER = "VIX"

TREND_WINDOW = 20
VIX_ELEVATED = 16
VIX_PANIC = 25
MOM_LONG = 63
MOM_SHORT = 21
MOM_LONG_WT = 0.40
MOM_SHORT_WT = 0.60
SECTOR_MOM = 60
VOL_WINDOW = 20
TARGET_VOL = 0.12
MAX_POSITION = 0.25
TOP_N = 2
TOP_SECTORS = 4
MIN_BARS = 70
GROSS_CAP = 1.45
LVRG_MULT = {t: 3.0 for t in LEVERAGED_3X}


def _closes(bars: list[dict]) -> list[float]:
    return [float(b["close"]) for b in bars]


def _sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _momentum(prices: list[float], lookback: int) -> float | None:
    if len(prices) < lookback + 1:
        return None
    base = prices[-(lookback + 1)]
    if base <= 0:
        return None
    return (prices[-1] / base) - 1.0


def _realised_vol(prices: list[float], window: int) -> float | None:
    if len(prices) < window + 1:
        return None
    log_rets = []
    for i in range(-window, 0):
        prev, curr = prices[i - 1], prices[i]
        if prev > 0 and curr > 0:
            log_rets.append(math.log(curr / prev))
    if len(log_rets) < 5:
        return None
    mean = sum(log_rets) / len(log_rets)
    variance = sum((r - mean) ** 2 for r in log_rets) / len(log_rets)
    return math.sqrt(variance) * math.sqrt(252)


def _equity(portfolio_state: dict, cash: float) -> float:
    last_prices = portfolio_state.get("last_prices", {})
    return cash + sum(
        p["quantity"] * last_prices.get(p["ticker"], p.get("avg_cost", 0))
        for p in portfolio_state.get("positions", [])
    )


def _holdings(portfolio_state: dict) -> dict[str, int]:
    return {p["ticker"]: int(p["quantity"]) for p in portfolio_state.get("positions", [])}


def _gross_leverage(weights: dict[str, float]) -> float:
    return sum(abs(w) * LVRG_MULT.get(t, 1.0) for t, w in weights.items())


def _get_price(ticker: str, portfolio_state: dict, market_state: dict) -> float | None:
    price = portfolio_state.get("last_prices", {}).get(ticker)
    if not price or price <= 0:
        bars = market_state.get(ticker, [])
        price = bars[-1]["close"] if bars else None
    return float(price) if price and price > 0 else None


def _exposure_scale(vix: float) -> float:
    if vix <= 0 or vix < VIX_ELEVATED:
        return 1.0
    if vix < VIX_PANIC:
        return 0.70
    return 0.0


def _rotate_to_tlt(
    holdings: dict[str, int],
    portfolio_state: dict,
    market_state: dict,
    equity: float,
) -> list[dict]:
    orders: list[dict] = []
    for ticker, qty in holdings.items():
        if ticker != "TLT" and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
    price = _get_price("TLT", portfolio_state, market_state)
    if price:
        target_qty = int((equity * 0.90) / price)
        diff = target_qty - holdings.get("TLT", 0)
        if diff > 0:
            orders.append({"ticker": "TLT", "side": "buy", "quantity": diff})
        elif diff < 0:
            orders.append({"ticker": "TLT", "side": "sell", "quantity": abs(diff)})
    return orders


def _sector_targets(market_state: dict, scale: float) -> dict[str, float]:
    ranked = []
    for t in SECTORS:
        ret = _momentum(_closes(market_state.get(t) or []), SECTOR_MOM)
        if ret is not None:
            ranked.append((ret, t))
    ranked.sort(reverse=True)
    winners = [t for r, t in ranked[:TOP_SECTORS] if r > 0]
    if not winners:
        return {}
    w = min(scale * 0.92 / len(winners), MAX_POSITION)
    return {t: w for t in winners}


def _leveraged_targets(market_state: dict, scale: float) -> dict[str, float]:
    scores: dict[str, float] = {}
    vols: dict[str, float] = {}
    for ticker in LEVERAGED_3X:
        bars = market_state.get(ticker, [])
        if not bars or len(bars) < MOM_LONG + 2:
            continue
        closes = _closes(bars)
        ml = _momentum(closes, MOM_LONG)
        ms = _momentum(closes, MOM_SHORT)
        rv = _realised_vol(closes, VOL_WINDOW)
        if any(x is None for x in (ml, ms, rv)) or rv <= 0:
            continue
        blend = MOM_LONG_WT * ml + MOM_SHORT_WT * ms
        if blend > 0:
            scores[ticker] = blend
            vols[ticker] = rv

    winners = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
    if not winners:
        return {}

    n = len(winners)
    vol_per_slot = (TARGET_VOL * scale) / n
    raw = {t: min(vol_per_slot / vols[t], MAX_POSITION) for t in winners}
    gross = _gross_leverage(raw)
    cap = GROSS_CAP * scale
    if gross > cap:
        raw = {t: w * (cap / gross) for t, w in raw.items()}
    return raw


def _risk_on_targets(market_state: dict, scale: float) -> dict[str, float]:
    targets = _leveraged_targets(market_state, scale)
    if targets:
        return targets
    targets = _sector_targets(market_state, scale)
    if targets:
        return targets
    gld_bars = market_state.get("GLD", [])
    if gld_bars and len(gld_bars) >= VOL_WINDOW + 1:
        rv = _realised_vol(_closes(gld_bars), VOL_WINDOW)
        if rv and rv > 0:
            return {"GLD": min(TARGET_VOL * scale / rv, MAX_POSITION)}
    return {}


def _apply_targets(
    targets: dict[str, float],
    holdings: dict[str, int],
    portfolio_state: dict,
    market_state: dict,
    equity: float,
) -> list[dict]:
    orders: list[dict] = []
    for ticker, qty in holdings.items():
        if ticker not in targets and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
    for ticker, weight in targets.items():
        price = _get_price(ticker, portfolio_state, market_state)
        if not price:
            continue
        target_qty = int((equity * weight) / price)
        diff = target_qty - holdings.get(ticker, 0)
        if diff > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": diff})
        elif diff < 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": abs(diff)})
    return orders


def decide(
    market_state: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict]:
    spy_bars = market_state.get(BENCHMARK, [])
    if len(spy_bars) < MIN_BARS:
        return []

    equity = _equity(portfolio_state, cash)
    holdings = _holdings(portfolio_state)

    spy_closes = _closes(spy_bars)
    vix_bars = market_state.get(VIX_TICKER, [])
    vix = float(vix_bars[-1]["close"]) if vix_bars else 0.0

    spy_sma = _sma(spy_closes, TREND_WINDOW)
    above_sma = spy_sma is not None and spy_closes[-1] > spy_sma
    scale = _exposure_scale(vix)

    if not above_sma or scale == 0.0:
        return _rotate_to_tlt(holdings, portfolio_state, market_state, equity)

    targets = _risk_on_targets(market_state, scale)
    if not targets:
        return _rotate_to_tlt(holdings, portfolio_state, market_state, equity)

    return _apply_targets(targets, holdings, portfolio_state, market_state, equity)
