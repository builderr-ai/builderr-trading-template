"""Calmar AI Guard — built with Cursor for builderr Round 1.

Goal: maximize Calmar (return / max drawdown) over the live 30-day window.

Stack:
  • Three regimes + crash brake (from drawdown-momentum, the house bar to beat)
  • Vol targeting + inverse-vol sizing
  • AI/chip momentum tilt in calm uptrends (NVDA, AMD, MU, MRVL, AVGO, SMH)
  • Immediate de-risk on stress; weekly rebalance otherwise
  • Drift guard at 27% per name (stays under 30% cap)

Long-only, no leveraged ETFs, per-name cap 18%, gross <= 1.0x.
"""
from __future__ import annotations

from statistics import pstdev

AI_CORE = frozenset({"NVDA", "AMD", "MU", "MRVL", "AVGO", "SMH"})
AI_TILT = 0.03

RISK_ON = (
    "NVDA", "AMD", "MU", "MRVL", "AVGO", "SMH",
    "QQQ", "SPY", "XLK",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLRE",
)
DEFENSIVE = ("XLP", "XLU", "XLV")

TREND_SMA = 100
NAME_SMA = 50
MOM_FAST, MOM_FAST_SKIP = 63, 5
MOM_SLOW = 126
INDEX_MOM_MIN = -0.02
TREND_BAND = 0.01
VOL_LOOKBACK = 20
TARGET_VOL = 0.14
TOP_N = 6
NAME_CAP = 0.18
GROSS_MAX = 1.00
DEF_GROSS_SOFT = 0.28
DEF_GROSS_HARD = 0.08
REBALANCE_EVERY = 5
DEAD_BAND = 0.03
DRIFT_LIMIT = 0.27
COOLDOWN_TICKS = 1

BRAKE_3D, BRAKE_5D = -0.06, -0.08
BRAKE_VOL_10D = 0.70

_ANN = 252 ** 0.5
_tick = 0
_last_rebalance = -10**9
_last_regime = None
_cooldown = 0


def _closes(bars: list[dict]) -> list[float]:
    return [float(b["close"]) for b in bars] if bars else []


def _sma(closes: list[float], n: int) -> float | None:
    return sum(closes[-n:]) / n if len(closes) >= n else None


def _ret(closes: list[float], days: int, skip: int = 0) -> float | None:
    need = days + skip + 1
    if len(closes) < need:
        return None
    end = closes[-(skip + 1)]
    start = closes[-(days + skip + 1)]
    return end / start - 1.0 if start > 0 else None


def _ann_vol(closes: list[float], n: int) -> float | None:
    if len(closes) < n + 1:
        return None
    rets = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(len(closes) - n, len(closes))
        if closes[i - 1] > 0
    ]
    if len(rets) < 2:
        return None
    return pstdev(rets) * _ANN


def _market_vol(market_state: dict) -> float:
    v = _ann_vol(_closes(market_state.get("QQQ") or []), VOL_LOOKBACK)
    return v if v and v > 0 else 0.20


def _regime(market_state: dict) -> str:
    qqq = _closes(market_state.get("QQQ") or [])
    spy = _closes(market_state.get("SPY") or [])
    if not qqq or not spy:
        return "hard"

    r3, r5, v10 = _ret(qqq, 3), _ret(qqq, 5), _ann_vol(qqq, 10)
    if (r3 is not None and r3 < BRAKE_3D) or (r5 is not None and r5 < BRAKE_5D) or (v10 and v10 > BRAKE_VOL_10D):
        return "hard"

    spy_sma, qqq_sma = _sma(spy, TREND_SMA), _sma(qqq, TREND_SMA)
    idx_mom = _ret(qqq, MOM_SLOW)
    if spy_sma is None or qqq_sma is None or idx_mom is None:
        return "soft"

    strong_on = (
        spy[-1] > spy_sma * (1 + TREND_BAND)
        and qqq[-1] > qqq_sma * (1 + TREND_BAND)
        and idx_mom >= INDEX_MOM_MIN
    )
    clearly_off = qqq[-1] < qqq_sma * (1 - TREND_BAND) or idx_mom < INDEX_MOM_MIN
    if _last_regime == "on":
        return "soft" if clearly_off else "on"
    return "on" if strong_on else "soft"


def _inv_vol_weights(names: list[str], market_state: dict, gross: float) -> dict[str, float]:
    inv: dict[str, float] = {}
    for t in names:
        v = _ann_vol(_closes(market_state.get(t) or []), VOL_LOOKBACK)
        if v and v > 0:
            inv[t] = 1.0 / v
    if not inv:
        return {}
    total = sum(inv.values())
    return {t: min(NAME_CAP, gross * w / total) for t, w in inv.items()}


def _rank(market_state: dict, universe: tuple[str, ...], ai_tilt: bool) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for t in universe:
        closes = _closes(market_state.get(t) or [])
        sma = _sma(closes, NAME_SMA)
        mf, ms = _ret(closes, MOM_FAST, MOM_FAST_SKIP), _ret(closes, MOM_SLOW)
        if sma is None or mf is None or ms is None or not closes:
            continue
        score = 0.5 * mf + 0.3 * ms + 0.2 * (closes[-1] / sma - 1.0)
        if ai_tilt and t in AI_CORE:
            score += AI_TILT
        if score > 0 and closes[-1] > sma:
            ranked.append((score, t))
    ranked.sort(reverse=True)
    return [t for _, t in ranked[:TOP_N]]


def _target_weights(market_state: dict, regime: str) -> dict[str, float]:
    if regime == "hard":
        avail = [t for t in DEFENSIVE if market_state.get(t)]
        return _inv_vol_weights(avail, market_state, DEF_GROSS_HARD) if avail else {}

    if regime == "on":
        gross = min(GROSS_MAX, TARGET_VOL / _market_vol(market_state))
        winners = _rank(market_state, RISK_ON, ai_tilt=True)
        return _inv_vol_weights(winners, market_state, gross) if winners else {}

    avail = [t for t in DEFENSIVE if market_state.get(t)]
    if not avail:
        return {}
    winners = _rank(market_state, tuple(avail), ai_tilt=False)
    return _inv_vol_weights(winners, market_state, DEF_GROSS_SOFT) if winners else {}


def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance, _last_regime, _cooldown
    _tick += 1

    positions = {p["ticker"]: p for p in portfolio_state.get("positions", [])}
    last_prices = portfolio_state.get("last_prices", {})
    equity = portfolio_state.get("cash", cash) + sum(
        p["quantity"] * last_prices.get(t, p.get("avg_cost", 0)) for t, p in positions.items()
    )
    if equity <= 0:
        return []

    regime = _regime(market_state)
    if regime == "hard":
        _cooldown = COOLDOWN_TICKS
    elif _cooldown > 0:
        _cooldown -= 1
        if regime == "on":
            regime = "soft"

    derisk = _last_regime is not None and regime != _last_regime and (
        regime == "hard" or (regime == "soft" and _last_regime == "on")
    )
    drifted = equity > 0 and any(
        p["quantity"] * last_prices.get(t, p.get("avg_cost", 0)) / equity > DRIFT_LIMIT
        for t, p in positions.items()
    )
    on_cadence = _tick - _last_rebalance >= REBALANCE_EVERY
    _last_regime = regime
    if not on_cadence and not derisk and not drifted:
        return []

    targets = _target_weights(market_state, regime)

    orders = []
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
        delta = int((equity * weight - cur_qty * px) // px)
        if abs(delta * px) < DEAD_BAND * equity:
            continue
        if delta > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": delta})
        elif delta < 0 and cur_qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": min(abs(delta), cur_qty)})

    if orders:
        _last_rebalance = _tick
    return orders
