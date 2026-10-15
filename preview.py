"""Local admission PREVIEW — see your bot clear the safety bar in ~10 seconds.

    python preview.py                          # previews agent.py
    python preview.py example_sector_rotation.py

No engine, no network, no keys, no install — pure standard library. It runs your
bot across three real, PUBLIC market windows (a calm uptrend, a mild selloff, and
the 2020 COVID vol-spike + snapback) and prints the same shape of report the real
admission email gives you, plus a verdict on the safety bar admission actually gates on.

What this DOES tell you, definitively:
  • Your bot runs clean (no crash) on real data.
  • It respects the live limits: gross leverage <= 1.5x, no single position >= 30%.
  • It doesn't blow up (>50% drawdown).
Those three are exactly what admission checks — and they depend on YOUR logic, not on
which window we use. So if you clear them here, you're very likely to be admitted.

What it does NOT tell you: your official numbers. Real admission runs centrally on
hidden historical regimes so it's identical for everyone. The Sharpe/Calmar/return
below are on sample windows — illustrative, not your score. The 60-day live forward
test is what actually ranks you.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "sample_regimes.json.gz"

# Beta multiples for the gross-exposure (leverage) check — same table as the engine.
BETA_3X = {"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU", "CURE", "DRN", "UDOW", "NAIL"}
BETA_2X = {"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"}
LEVERAGE_CAP = 1.5          # gross beta-adjusted exposure / equity
CONCENTRATION_CAP = 0.30    # any single position / equity
CATASTROPHE_DD = 0.50       # >50% drawdown = blow-up
SLIP_EQUITY = 0.0005        # 5 bps
SLIP_LEVERAGED = 0.0010     # 10 bps
START_CASH = 100_000.0


def beta(ticker: str) -> float:
    if ticker in BETA_3X:
        return 3.0
    if ticker in BETA_2X:
        return 2.0
    return 1.0


def load_decide(path: Path):
    spec = importlib.util.spec_from_file_location("agent", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "decide"):
        raise SystemExit("your file must define decide(market_state, portfolio_state, cash)")
    return mod.decide


def _expand(rows):
    """Compact [ts,o,h,l,c,v] -> the bar dicts the contract specifies."""
    return [
        {"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
        for r in rows
    ]


def run_regime(agent_path: Path, name: str, regime: dict) -> dict:
    # Re-load the agent fresh for every regime so module-level globals reset —
    # the real engine runs each regime in its own process, so we mirror that.
    decide = load_decide(agent_path)
    
    # Reset rebalance log if available
    if hasattr(decide.__module__, "reset_rebalance_log"):
        decide.__module__.reset_rebalance_log()
    
    bars = {t: _expand(rows) for t, rows in regime["bars"].items()}
    # Trading-day timeline = union of dates, sorted.
    all_dates = sorted({b["ts"] for rows in bars.values() for b in rows})
    eval_dates = [d for d in all_dates if regime["eval_start"] <= d <= regime["eval_end"]]

    cash = START_CASH
    positions: dict[str, float] = {}  # ticker -> qty
    avg_cost: dict[str, float] = {}
    equity_curve: list[float] = []
    peak_gross = 0.0
    peak_conc = 0.0
    conc_streak: dict[str, int] = {}   # ticker -> consecutive days >= 30%
    max_conc_streak = 0                # worst run of >=30% on any single ticker
    trades = 0
    pending: list[dict] = []  # orders to fill at next day's open
    errors = 0

    def close_on(ticker: str, date: str):
        for b in bars.get(ticker, []):
            if b["ts"] == date:
                return b["close"]
        return None

    def open_on(ticker: str, date: str):
        for b in bars.get(ticker, []):
            if b["ts"] == date:
                return b["open"]
        return None

    for i, date in enumerate(eval_dates):
        # 1. Fill yesterday's orders at today's open (deterministic, with slippage).
        for o in pending:
            px = open_on(o["ticker"], date)
            if px is None:
                continue
            slip = SLIP_LEVERAGED if beta(o["ticker"]) > 1 else SLIP_EQUITY
            if o["side"] == "buy":
                fill = px * (1 + slip)
                qty = o["quantity"]
                cost = fill * qty
                if cost > cash:  # can't buy more than cash allows (long-only, no margin debt)
                    qty = cash / fill if fill > 0 else 0
                    cost = fill * qty
                if qty <= 0:
                    continue
                held = positions.get(o["ticker"], 0.0)
                prev_cost = avg_cost.get(o["ticker"], 0.0) * held
                positions[o["ticker"]] = held + qty
                avg_cost[o["ticker"]] = (prev_cost + cost) / (held + qty)
                cash -= cost
                trades += 1
            else:  # sell
                held = positions.get(o["ticker"], 0.0)
                qty = min(o["quantity"], held)
                if qty <= 0:
                    continue
                fill = px * (1 - slip)
                cash += fill * qty
                positions[o["ticker"]] = held - qty
                trades += 1
        pending = []

        # 2. Mark to market on today's close.
        prices = {t: close_on(t, date) for t in bars}
        prices = {t: p for t, p in prices.items() if p is not None}
        pos_value = {t: positions.get(t, 0.0) * prices.get(t, 0.0) for t in positions}
        equity = cash + sum(pos_value.values())
        equity = max(equity, 1e-9)
        equity_curve.append(equity)

        # 3. Risk telemetry (what admission gates on).
        gross = sum(abs(v) * beta(t) for t, v in pos_value.items()) / equity
        peak_gross = max(peak_gross, gross)
        # Concentration breach = a single position held >= 30% for >5 CONSECUTIVE days
        # (matches the engine — a brief excursion is fine, a sustained one isn't).
        for t in bars:
            frac = abs(pos_value.get(t, 0.0)) / equity
            peak_conc = max(peak_conc, frac)
            if frac >= CONCENTRATION_CAP:
                conc_streak[t] = conc_streak.get(t, 0) + 1
                max_conc_streak = max(max_conc_streak, conc_streak[t])
            else:
                conc_streak[t] = 0

        # 4. Ask the bot for orders (history = everything up to & including today).
        market_state = {t: [b for b in bars[t] if b["ts"] <= date] for t in bars}
        portfolio_state = {
            "cash": cash,
            "positions": [
                {"ticker": t, "quantity": q, "avg_cost": avg_cost.get(t, 0.0)}
                for t, q in positions.items() if q > 0
            ],
            "last_prices": prices,
        }
        try:
            orders = decide(market_state, portfolio_state, cash)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"    ! decide() raised on {date}: {e!r}")
            orders = []
        for o in orders or []:
            try:
                if o["side"] in ("buy", "sell") and float(o["quantity"]) > 0 and o["ticker"] in bars:
                    pending.append({"ticker": o["ticker"], "side": o["side"], "quantity": float(o["quantity"])})
            except (KeyError, TypeError, ValueError):
                errors += 1

    ret = equity_curve[-1] / START_CASH - 1 if equity_curve else 0.0
    mdd = _max_drawdown(equity_curve)
    sharpe = _sharpe(equity_curve)
    calmar = (_annualize(ret, len(equity_curve)) / mdd) if mdd > 1e-9 else 0.0
    
    # Capture rebalance summary if available
    rebalance_summary = ""
    if hasattr(decide.__module__, "get_rebalance_summary"):
        rebalance_summary = decide.__module__.get_rebalance_summary()
    
    return {
        "name": name, "ret": ret, "mdd": mdd, "sharpe": sharpe, "calmar": calmar,
        "trades": trades, "peak_gross": peak_gross, "peak_conc": peak_conc,
        "max_conc_streak": max_conc_streak, "errors": errors, "days": len(equity_curve),
        "rebalance_summary": rebalance_summary,
    }


def _max_drawdown(curve: list[float]) -> float:
    peak, mdd = -1e18, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _sharpe(curve: list[float]) -> float:
    if len(curve) < 3:
        return 0.0
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0
    return (mean / sd) * math.sqrt(252)


def _annualize(total_ret: float, days: int) -> float:
    if days <= 0:
        return 0.0
    return (1 + total_ret) ** (252 / days) - 1


def main() -> int:
    agent_file = sys.argv[1] if len(sys.argv) > 1 else "agent.py"
    if not DATA.exists():
        raise SystemExit(f"missing {DATA.name} — re-fork the template (it ships with the sample data).")
    agent_path = HERE / agent_file
    load_decide(agent_path)  # fail fast with a clear message if decide() is missing
    regimes = json.loads(gzip.open(DATA, "rb").read())

    print(f"=== builderr local admission preview: {agent_file} ===")
    print("Running your bot across 3 real, public sample windows...\n")
    results = [run_regime(agent_path, name, reg) for name, reg in regimes.items()]

    print(f"  {'window':20s} {'Ret':>7s} {'MaxDD':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'Trades':>7s}")
    for r in results:
        print(f"  {r['name']:20s} {r['ret']*100:6.2f}% {r['mdd']*100:6.2f}% "
              f"{r['sharpe']:7.2f} {r['calmar']:7.2f} {r['trades']:7d}")

    clean = all(r["errors"] == 0 for r in results)
    peak_gross = max(r["peak_gross"] for r in results)
    peak_conc = max(r["peak_conc"] for r in results)
    worst_streak = max(r["max_conc_streak"] for r in results)
    worst_dd = max(r["mdd"] for r in results)
    total_trades = sum(r["trades"] for r in results)

    lev_ok = peak_gross <= LEVERAGE_CAP + 1e-6
    conc_ok = worst_streak <= 5  # breach only if held >= 30% for MORE than 5 consecutive days
    dd_ok = worst_dd < CATASTROPHE_DD

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print("\nSafety bar (exactly what admission gates on):")
    print(f"  [{mark(clean)}] runs clean, no errors")
    print(f"  [{mark(lev_ok)}] no leverage breach   (peak gross {peak_gross:.2f}x  <= {LEVERAGE_CAP}x)")
    print(f"  [{mark(conc_ok)}] no concentration breach (peak {peak_conc*100:.0f}%; "
          f"longest run >= {int(CONCENTRATION_CAP*100)}% was {worst_streak}d, limit is 5d)")
    print(f"  [{mark(dd_ok)}] no blow-up           (worst drawdown {worst_dd*100:.1f}%  < {int(CATASTROPHE_DD*100)}%)")

    admitted = clean and lev_ok and conc_ok and dd_ok
    print()
    if admitted and total_trades == 0:
        print("VERDICT: clears the safety bar — but your bot never traded. It would be ADMITTED,")
        print("         yet it's sitting in cash. Check your signals before you submit.")
    elif admitted:
        print("VERDICT: ✓ You clear admission's safety bar — you're very likely to be ADMITTED.")
        print("         Push your repo + email submit@builderr.ai to lock in your official run.")
    else:
        print("VERDICT: ✗ Not yet. Fix the FAIL line(s) above — those are the same limits the real")
        print("         admission enforces, so this would be rejected. The numbers are easy to chase;")
        print("         the safety bar is the part that actually gates you.")
    print("\n(Numbers above are on SAMPLE public windows — illustrative, NOT your official score.")
    print(" Real admission runs centrally on hidden regimes; the 60-day live test is what ranks you.)")
    return 0 if admitted else 1


if __name__ == "__main__":
    sys.exit(main())
