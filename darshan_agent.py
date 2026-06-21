"""Concentrated momentum + regime gate + inverse-vol.

Full exposure in the trend via inverse-vol weighting, one clean exit
on SMA100/CUSUM breach, CPPI floor as backup parachute.
"""

from __future__ import annotations
from math import sqrt
from statistics import mean, pstdev

_UNIVERSE = (
    "NVDA", "AMD", "MU", "MRVL", "AVGO", "SMH",
    "AAPL", "MSFT", "META", "GOOGL", "AMZN",
    "QQQ", "SPY",
)

# ── Parameters ────────────────────────────────────────────────────────

MOM_WINDOW = 60
MOM_SKIP = 5
SMA_STOCK = 50
SMA_MARKET = 100
TOP_N = 8
NAME_CAP = 0.18
MIN_TRADE_PCT = 0.02
REBALANCE_EVERY = 2
VOL_WIN = 20
VOL_TARGET = 0.25          # loose — only bites in genuine vol spikes
CPPI_FLOOR = 0.93          # soft backup parachute

# ── CUSUM regime detector ─────────────────────────────────────────────
_CUSUM_H = 1.8
_CUSUM_K = 0.25
_CUSUM_REF = 40

# ── State ─────────────────────────────────────────────────────────────
_last_date: str | None = None
_peak_equity: float = 0.0
_tick: int = 0

# ── Helpers ───────────────────────────────────────────────────────────

def _closes(series):
    if not series:
        return []
    return [float(d["close"]) for d in series]

def _sma(prices, n):
    if len(prices) < n:
        return None
    return mean(prices[-n:])

def _mom(prices, n, skip=0):
    if len(prices) < n + skip + 1:
        return None
    return prices[-(skip + 1)] / prices[-(n + skip + 1)] - 1

def _rvol(prices, n):
    if len(prices) < n + 1:
        return None
    rets = [prices[i] / prices[i-1] - 1 for i in range(-n, 0)]
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252)

def _bar_date(ms):
    for anchor in ("QQQ", "SPY"):
        bars = ms.get(anchor) or []
        if bars:
            ts = bars[-1].get("ts")
            return str(ts)[:10] if ts is not None else str(len(bars))
    return None

def _equity(ps, cash):
    eq = float(ps.get("cash", cash))
    lp = ps.get("last_prices") or {}
    for raw in ps.get("positions") or []:
        t = str(raw.get("ticker", "")).upper()
        q = float(raw.get("quantity", 0))
        p = float(lp.get(t, 0))
        if t and q > 0 and p > 0:
            eq += q * p
    return max(eq, 0)

def _positions(ps):
    out = {}
    for raw in ps.get("positions") or []:
        t = str(raw.get("ticker", "")).upper()
        q = float(raw.get("quantity", 0))
        if t and q > 0:
            out[t] = out.get(t, 0) + q
    return out

def _prices(ms):
    return {
        t.upper(): float(bars[-1]["close"])
        for t, bars in ms.items()
        if bars and bars[-1].get("close", 0) > 0
    }

# ── CUSUM early-warning overlay ───────────────────────────────────────

def _cusum_bear(prices):
    if len(prices) < 30:
        return False
    rets = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        if prev <= 0:
            return False
        rets.append(prices[i] / prev - 1)
    if len(rets) < 20:
        return False
    span = min(_CUSUM_REF, len(rets))
    window = rets[-span:]
    std = (sum(r * r for r in window) / len(window)) ** 0.5
    if std < 1e-8:
        return False
    k = _CUSUM_K * std
    H = _CUSUM_H * std
    lookback = min(30, len(rets))
    S = 0.0
    for r in rets[-lookback:]:
        S = max(0.0, S + (-r) - k)
    return S > H

def _regime(ms):
    qqq = _closes(ms.get("QQQ"))
    spy = _closes(ms.get("SPY"))
    if len(qqq) < 100 or len(spy) < 20:
        return "risk_off"
    qqq_sma100 = _sma(qqq, 100)
    spy_sma200 = _sma(spy, 200)
    sma_off = qqq_sma100 is None or qqq[-1] < qqq_sma100
    cusum_bear = _cusum_bear(qqq)
    near_sma = (qqq_sma100 is not None
                and qqq[-1] < qqq_sma100 * 1.03
                and qqq[-1] >= qqq_sma100)
    spy_off = spy_sma200 is not None and spy[-1] < spy_sma200
    if sma_off or spy_off or (cusum_bear and near_sma):
        return "risk_off"
    return "risk_on"

# ── Weighting ─────────────────────────────────────────────────────────

def _inverse_vol_weights(names, ms):
    weights = {}
    for t in names:
        cs = _closes(ms.get(t))
        v = _rvol(cs, VOL_WIN) if cs else None
        if v and v > 0:
            weights[t] = 1.0 / max(v, 0.05)
    total = sum(weights.values()) or 1.0
    return {t: w / total for t, w in weights.items()}

# ── Core logic ────────────────────────────────────────────────────────

def decide(market_state, portfolio_state, cash):
    global _last_date, _peak_equity, _tick
    _tick += 1
    if not market_state:
        return []
    today = _bar_date(market_state)
    if today is None:
        return []
    if _last_date is None:
        _last_date = today
        return []
    eq = _equity(portfolio_state, cash)
    if eq <= 0:
        return []
    px = _prices(market_state)
    pos = _positions(portfolio_state)
    _peak_equity = max(_peak_equity, eq)

    # ── 1. Regime gate — single hard exit ─────────────────────────────
    regime = _regime(market_state)
    if regime == "risk_off":
        _last_date = today
        orders = [
            {"ticker": t, "side": "sell", "quantity": int(q)}
            for t, q in pos.items()
            if px.get(t, 0) > 0 and int(q) > 0
        ]
        return orders[:45]

    # ── 2. CPPI floor — soft backup parachute ─────────────────────────
    floor = _peak_equity * CPPI_FLOOR
    cppi_buffer = (eq - floor) / (eq * (1.0 - CPPI_FLOOR) + 0.01)
    cppi_multiple = min(1.0, max(0.0, cppi_buffer))

    # ── 3. Momentum ranking ───────────────────────────────────────────
    scored = []
    for t in _UNIVERSE:
        cs = _closes(market_state.get(t))
        if len(cs) < max(MOM_WINDOW + MOM_SKIP, SMA_STOCK) + 1:
            continue
        sma50 = _sma(cs, SMA_STOCK)
        mom = _mom(cs, MOM_WINDOW, MOM_SKIP)
        if sma50 is None or mom is None:
            continue
        if cs[-1] <= sma50 or mom <= 0:
            continue
        scored.append((mom, t))
    scored.sort(reverse=True)
    winners = [t for _, t in scored[:TOP_N]]

    # ── 4. Inverse-vol weights + name cap ─────────────────────────────
    if not winners:
        return []
    raw_weights = _inverse_vol_weights(winners, market_state)
    total_raw = sum(raw_weights.values())
    if total_raw <= 0:
        return []

    capped = {}
    overflow = 0.0
    for t, w in raw_weights.items():
        capped_w = min(w, NAME_CAP)
        capped[t] = capped_w
        overflow += w - capped_w

    if overflow > 0.001:
        room = {t: NAME_CAP - capped[t] for t in capped if capped[t] < NAME_CAP}
        room_total = sum(room.values())
        if room_total > 0:
            for t in sorted(capped, key=lambda x: -room.get(x, 0)):
                if capped[t] < NAME_CAP:
                    capped[t] = min(NAME_CAP, capped[t] + overflow * room[t] / room_total)

    # ── 5. Vol-target + CPPI ─────────────────────────────────────────
    port_vol = 0.0
    wsum = sum(capped.values())
    if wsum > 0:
        num = 0.0
        for t, w in capped.items():
            v = _rvol(_closes(market_state.get(t) or []), VOL_WIN)
            if v and v > 0:
                num += (w / wsum) * v
        port_vol = max(num, 0.01)
    vol_scale = min(1.0, VOL_TARGET / port_vol)
    gross_scale = cppi_multiple * vol_scale
    targets = {t: w * gross_scale for t, w in capped.items() if w * gross_scale > 0.005}

    if not targets:
        return []

    # ── 6. Rebalance check ───────────────────────────────────────────
    from datetime import datetime
    days = None
    if _last_date and today:
        try:
            d1 = datetime.strptime(_last_date, "%Y-%m-%d")
            d2 = datetime.strptime(today, "%Y-%m-%d")
            days = (d2 - d1).days
        except Exception:
            pass
    if days is None:
        days = REBALANCE_EVERY if (_tick % REBALANCE_EVERY == 0) else 0
    if days < REBALANCE_EVERY:
        return []

    # ── 7. Orders ─────────────────────────────────────────────────────
    min_v = eq * MIN_TRADE_PCT
    orders = []
    proceeds = 0.0

    for t, q in pos.items():
        price = px.get(t)
        if not price or price <= 0:
            continue
        cur_v = q * price
        tgt_v = eq * targets.get(t, 0)
        if t not in targets:
            s = int(q)
            if s > 0 and cur_v >= min_v:
                orders.append({"ticker": t, "side": "sell", "quantity": s})
                proceeds += s * price
        elif cur_v - tgt_v > min_v:
            s = min(int(abs(cur_v - tgt_v) / price), int(q))
            if s > 0:
                orders.append({"ticker": t, "side": "sell", "quantity": s})
                proceeds += s * price

    spendable = max(float(cash), 0) + proceeds * 0.98

    for t, w in sorted(targets.items(), key=lambda x: -x[1]):
        price = px.get(t)
        if not price or price <= 0:
            continue
        cur = pos.get(t, 0)
        cur_v = cur * price
        tgt_v = eq * w
        delta = tgt_v - cur_v
        if delta < min_v:
            continue
        buy_v = min(delta, spendable)
        qty = int(buy_v / price)
        if qty > 0:
            orders.append({"ticker": t, "side": "buy", "quantity": qty})
            spendable -= qty * price

    if orders:
        _last_date = today
    return orders[:45]
