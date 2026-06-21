"""Live leaderboard runner — produces real, daily-refreshed standings for the
reference ("house") bots on LIVE market data. A GitHub Action runs this each
market day and commits leaderboard.json; the site reads it.

This is honest content, not fakery:
  • The bots are the real reference strategies + admitted entrants in this repo.
  • Numbers are COMPUTED from running them on real daily bars (yfinance), never hardcoded.
  • Each runs a $100,000 paper account from SCORE_START (the fair common window) to the latest bar, and
    we report the simple, human numbers: account value, P&L, and trades.

It reuses the same fill model and metrics as preview.py, so a bot scores here the
same way it would in the real eval.

    python live_runner.py            # writes leaderboard.json

Needs: yfinance (installed in the Action). Not part of the no-dep builder workflow.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
OUT = HERE / "leaderboard.json"

def _load_universe() -> list[str]:
    """The tradeable universe is the FROZEN snapshot in universe.json (top ~1000
    US names by liquidity, built by build_universe.py at round open). Same list
    for the board and the admission engine; stable for the whole round."""
    f = HERE / "universe.json"
    if f.exists():
        try:
            tickers = (json.loads(f.read_text()) or {}).get("tickers") or []
            if tickers:
                return list(dict.fromkeys(tickers))
        except Exception:  # noqa: BLE001
            pass
    # fallback if the snapshot is somehow missing
    return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
            "AVGO", "AMD", "MU", "MRVL", "TQQQ", "SOXL", "QLD", "SSO"]


UNIVERSE = _load_universe()

# The live field — file -> (display name, label). House/reference bots set the
# bar; real Round 1 entrants are labeled as such, never disguised as house bots.
FIELD = [
    ("drawdown_momentum.py",         "drawdown-momentum",      "house · the bar to beat"),
    ("seed_dual_momentum.py",        "dual-momentum-rotation", "house · all-weather"),
    ("ai_momentum.py",               "ai-momentum-basket",     "house · aggressive"),
    ("example_sector_rotation.py",   "sector-rotation",        "reference"),
    ("example_vol_target.py",        "vol-target",             "reference"),
    # Real Round 1 entrants (NOT house bots) — scored on the same live window.
    ("opu_agent.py",                 "opu",                    "round 1 · entrant"),
    ("robert_agent.py",              "robert",                 "round 1 · entrant"),
    ("mohit_agent.py",               "mohit",                  "round 1 · entrant"),
    ("zaid_agent.py",                "zaid",                   "round 1 · entrant"),
    ("sumegh_agent.py",              "sumegh",                 "round 1 · entrant"),
    ("shyam_agent.py",               "shyam",                  "round 1 · entrant"),
    ("harsimran_agent.py",           "harsimran",              "round 1 · entrant"),
    ("sankeerth_agent.py",           "sankeerth",              "round 1 · entrant"),
    ("siddu_agent.py",               "siddu",                  "round 1 · entrant"),
    ("rohit_agent.py",               "rohit",                  "round 1 · entrant"),
    ("arnav_agent.py",               "arnav",                  "round 1 · entrant"),
    ("nagarjuna_agent.py",           "nagarjuna",              "round 1 · entrant"),
    ("balaji_agent.py",              "balaji",                 "round 1 · entrant"),
    ("ajai_agent.py",                "ajai",                   "round 1 · entrant"),
    ("aksham_agent.py",              "aksham",                 "round 1 · entrant"),
    ("darshan_agent.py",             "darshan",                "round 1 · entrant"),
    ("tanishq_agent.py",             "tanishq",                "round 1 · entrant"),
    ("aarya_agent.py",               "aarya",                  "round 1 · entrant"),
    # yog + krunal submitted the stock template strategy unmodified (allowed;
    # told them) — identical code, so identical numbers until they iterate.
    ("yog_agent.py",                 "yog",                    "round 1 · entrant"),
    ("krunal_agent.py",              "krunal",                 "round 1 · entrant"),
    # Rohan: regime bot with an optional external-LLM call that is INERT in the
    # eval (no network/key) — it cleanly falls back to pure-Python logic, which
    # is what we score. Told him.
    ("rohan_agent.py",               "rohan",                  "round 1 · entrant"),
    ("dev_agent.py",                 "dev",                    "round 1 · entrant"),
    ("deepika_agent.py",             "deepika",                "round 1 · entrant"),
    ("om_agent.py",                  "om",                     "round 1 · entrant"),
    ("raam_agent.py",                "raam",                   "round 1 · entrant"),
    ("navika_agent.py",              "navika",                 "round 1 · entrant"),
    ("yuva_agent.py",                "yuva",                   "round 1 · entrant"),
    ("shivkumar_agent.py",           "shivkumar",              "round 1 · entrant"),
]

# Private entrants (read-only deploy-key path). Their CODE never enters this
# PUBLIC repo. We score them locally from PRIVATE_DIR (gitignored) and publish
# only their numbers, persisted in private_results.json — so the cron (which
# can't see their code) keeps them on the board with their last-scored result.
PRIVATE_DIR = HERE / "private_agents"
PRIVATE_RESULTS = HERE / "private_results.json"
PRIVATE_FIELD = [
    ("eshwar_agent.py",              "eshwar",                 "round 1 · entrant"),
]

EVAL_DAYS = 60       # (history sizing only) trailing window used when fetching bars
WARMUP_DAYS = 220    # extra history so 200-day signals work
START_CASH = 100_000.0
ROUND_START = "2026-06-02"   # Round 1 admission opened
# Fair scoring: entries for Round 1 closed after the last submission (2026-06-17), and
# every agent is re-based to a fresh $100k here. The scored days are all AFTER the last
# entry, so no one can optimise a bot against market history they had already seen.
SCORE_START = "2026-06-18"   # legacy fallback only; per-bot ENTRY dates below are authoritative

# Per-agent submission date (the email/commit date). A bot is scored ONLY from the
# first market session AFTER this date — forward-only — so no one can optimise against
# market history they had already seen, and submitting later gives zero edge.
# "2026-06-01" = live since the Round 1 open (house/reference benchmarks).
# Going forward: set this to the date the entry email arrived.
ENTRY = {
    "drawdown-momentum": "2026-06-01", "dual-momentum-rotation": "2026-06-01",
    "ai-momentum-basket": "2026-06-01", "sector-rotation": "2026-06-01", "vol-target": "2026-06-01",
    "opu": "2026-06-02", "robert": "2026-06-02", "mohit": "2026-06-03",
    "zaid": "2026-06-04", "sumegh": "2026-06-04", "shyam": "2026-06-06",
    "harsimran": "2026-06-19", "sankeerth": "2026-06-18", "siddu": "2026-06-07",
    "rohit": "2026-06-08", "eshwar": "2026-06-08", "arnav": "2026-06-09",
    "nagarjuna": "2026-06-09", "balaji": "2026-06-10", "ajai": "2026-06-11",
    "aksham": "2026-06-11", "darshan": "2026-06-19", "tanishq": "2026-06-11",
    "aarya": "2026-06-12", "yog": "2026-06-13", "krunal": "2026-06-13",
    "rohan": "2026-06-15", "dev": "2026-06-19", "deepika": "2026-06-15",
    "om": "2026-06-17", "raam": "2026-06-17",
    "navika": "2026-06-18", "yuva": "2026-06-18", "shivkumar": "2026-06-17",
}
CHART_START = "2026-06-01"   # common x-axis for the illustrative race chart (Round 1 open)
SLIP_EQUITY = 0.0005
SLIP_LEVERAGED = 0.0010
BETA_3X = {"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU", "CURE", "DRN", "UDOW", "NAIL"}
BETA_2X = {"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"}


def beta(t: str) -> float:
    return 3.0 if t in BETA_3X else 2.0 if t in BETA_2X else 1.0


def load_decide(filename: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide


def load_decide_from(path: Path):
    """Load decide() from an arbitrary path (used for local-only private agents)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide


def _rows_from_df(df, need):
    cols = {str(c).lower(): c for c in df.columns}
    if not {"open", "high", "low", "close"} <= set(cols):
        return None
    rows = []
    for ts, r in df.iterrows():
        try:
            o, h, l, c = float(r[cols["open"]]), float(r[cols["high"]]), float(r[cols["low"]]), float(r[cols["close"]])
            vv = r[cols["volume"]] if "volume" in cols else 0
            v = int(vv) if vv == vv else 0
        except (KeyError, ValueError, TypeError):
            continue
        if any(x != x for x in (o, h, l, c)):
            continue
        rows.append({"ts": ts.strftime("%Y-%m-%d"), "open": o, "high": h, "low": l, "close": c, "volume": v})
    return rows[-need:] if len(rows) >= need - 60 else None  # tolerate short histories


CHUNK = 200  # tickers per batched yfinance call (1000-name universe → ~5 calls)


def fetch_bars() -> dict[str, list[dict]]:
    """Fetch daily bars for the whole (~1000-name) universe in batched chunks —
    one yfinance call per chunk, tolerant of any ticker/chunk that fails."""
    need = EVAL_DAYS + WARMUP_DAYS + 30
    bars: dict[str, list[dict]] = {}
    for i in range(0, len(UNIVERSE), CHUNK):
        chunk = UNIVERSE[i:i + CHUNK]
        try:
            raw = yf.download(chunk, period="2y", interval="1d", auto_adjust=True,
                              progress=False, threads=True, group_by="ticker")
        except Exception:  # noqa: BLE001
            continue
        if raw is None or getattr(raw, "empty", True):
            continue
        multi = hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1
        for t in chunk:
            try:
                df = raw[t] if multi else raw
            except KeyError:
                continue
            if df is None or df.empty:
                continue
            r = _rows_from_df(df, need)
            if r:
                bars[t] = r
    return bars


def run_bot(decide, bars: dict[str, list[dict]], entry_date: str = SCORE_START) -> dict:
    """Run a $100k paper account from the first session AFTER entry_date (the bot's
    submission) to the latest bar — forward-only, so it is never scored on days its
    author could already have seen.

    The agent decides on info known through the prior close, and orders fill at each
    session's OPEN (+/- slippage) — so the book actually holds through, and captures,
    the day's move.
    """
    all_dates = sorted({b["ts"] for rows in bars.values() for b in rows})
    eval_dates = [d for d in all_dates if d > entry_date] or all_dates[-1:]
    cash = START_CASH
    positions: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    curve: list[float] = []
    trades = 0

    def price(t, date, field):
        for b in bars.get(t, []):
            if b["ts"] == date:
                return b[field]
        return None

    for date in eval_dates:
        open_px = {t: p for t in bars if (p := price(t, date, "open")) is not None}
        close_px = {t: p for t in bars if (p := price(t, date, "close")) is not None}

        # The contest launched the night BEFORE June 2's open, so the book is set
        # on info known through the prior close, and orders execute at THIS day's
        # OPEN — i.e. the agent actually holds through (and captures) the session.
        market_state = {t: [b for b in bars[t] if b["ts"] < date] for t in bars}
        prior_close = {t: ms[-1]["close"] for t, ms in market_state.items() if ms}
        portfolio_state = {
            "cash": cash,
            "positions": [{"ticker": t, "quantity": q, "avg_cost": avg_cost.get(t, 0.0)}
                          for t, q in positions.items() if q > 0],
            "last_prices": prior_close,
        }
        try:
            orders = decide(market_state, portfolio_state, cash) or []
        except Exception:
            orders = []

        for o in orders:
            try:
                tk, side, qty = o["ticker"], o["side"], float(o["quantity"])
            except (KeyError, TypeError, ValueError):
                continue
            if side not in ("buy", "sell") or qty <= 0 or tk not in open_px:
                continue
            px = open_px[tk]  # fill at the day's OPEN
            slip = SLIP_LEVERAGED if beta(tk) > 1 else SLIP_EQUITY
            if side == "buy":
                fill = px * (1 + slip)
                if fill * qty > cash:
                    qty = cash / fill if fill > 0 else 0
                if qty <= 0:
                    continue
                held = positions.get(tk, 0.0)
                avg_cost[tk] = (avg_cost.get(tk, 0.0) * held + fill * qty) / (held + qty) if held + qty > 0 else fill
                positions[tk] = held + qty
                cash -= fill * qty
                trades += 1
            else:
                held = positions.get(tk, 0.0)
                qty = min(qty, held)
                if qty <= 0:
                    continue
                cash += px * (1 - slip) * qty
                positions[tk] = held - qty
                trades += 1

        # mark to the day's CLOSE
        equity = max(cash + sum(positions.get(t, 0.0) * close_px.get(t, 0.0) for t in positions), 1e-9)
        curve.append(equity)

    equity = curve[-1] if curve else START_CASH
    return {
        "equity": round(equity, 2),
        "pnl": round(equity - START_CASH, 2),
        "ret": equity / START_CASH - 1,
        "trades": trades,
        "days": len(curve),   # live market days scored (forward-only window since entry)
        "curve": [round(x, 2) for x in curve],  # daily mark-to-close equity, aligned to eval_dates
        "cash": round(cash, 2),
        # current holdings (ticker -> shares) so the site can mark them live at
        # intraday prices between daily runs — real mark-to-market, not faked motion.
        "holdings": [{"t": t, "q": round(q, 4)} for t, q in positions.items() if q > 0],
    }


def _mdd(curve):
    peak, mdd = -1e18, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _sharpe(curve):
    if len(curve) < 3:
        return 0.0
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    sd = math.sqrt(var)
    return (mean / sd) * math.sqrt(252) if sd > 1e-12 else 0.0


def main() -> int:
    bars = fetch_bars()
    if len(bars) < 12:
        print(f"fetched only {len(bars)} tickers — refusing to overwrite leaderboard.json")
        return 1
    asof = sorted({b["ts"] for rows in bars.values() for b in rows})[-1]
    rows = []
    curves: dict[str, list] = {}   # name -> daily equity curve, for history.json (the race chart)
    chart_dates = [d for d in sorted({b["ts"] for rs in bars.values() for b in rs}) if d > CHART_START]
    for filename, name, label in FIELD:
        entry = ENTRY.get(name, SCORE_START)
        try:
            m = run_bot(load_decide(filename), bars, entry)
        except Exception as e:  # noqa: BLE001
            print(f"skip {filename}: {e!r}")
            continue
        rows.append({"name": name, "label": label,
                     "equity": m["equity"], "pnl": m["pnl"],
                     "ret": round(m["ret"], 4), "trades": m["trades"],
                     "days": m["days"], "since": entry,
                     # for live intraday marking on the site (public bots only)
                     "cash": m["cash"], "holdings": m["holdings"]})
        # right-align the per-bot curve on the common chart axis (flat $100k before entry)
        curves[name] = [START_CASH] * max(0, len(chart_dates) - len(m["curve"])) + m["curve"]
        print(f"  {name:24s} ${m['equity']:,.0f}  P&L {m['pnl']:+,.0f} ({m['ret']*100:+.2f}%)  {m['days']}d  Trades={m['trades']}")

    # Private entrants: score locally if their (gitignored) code is present;
    # otherwise fall back to last-scored numbers in private_results.json. Either
    # way, only numbers are published — their code never enters this public repo.
    saved = {}
    if PRIVATE_RESULTS.exists():
        try:
            saved = json.loads(PRIVATE_RESULTS.read_text()) or {}
        except Exception:  # noqa: BLE001
            saved = {}
    for filename, name, label in PRIVATE_FIELD:
        entry = ENTRY.get(name, SCORE_START)
        p = PRIVATE_DIR / filename
        if p.exists():
            try:
                m = run_bot(load_decide_from(p), bars, entry)
                saved[name] = {"label": label, "equity": m["equity"], "pnl": m["pnl"],
                               "ret": round(m["ret"], 4), "trades": m["trades"],
                               "days": m["days"], "since": entry, "as_of": asof}
                print(f"  {name:24s} (private) ${m['equity']:,.0f}  P&L {m['pnl']:+,.0f} ({m['ret']*100:+.2f}%)  {m['days']}d  Trades={m['trades']}")
            except Exception as e:  # noqa: BLE001
                print(f"skip private {filename}: {e!r}")
        rec = saved.get(name)
        if rec:
            rows.append({"name": name, "label": rec["label"], "equity": rec["equity"],
                         "pnl": rec["pnl"], "ret": rec["ret"], "trades": rec["trades"],
                         "days": rec.get("days"), "since": rec.get("since")})
    if saved:
        PRIVATE_RESULTS.write_text(json.dumps(saved, indent=2))

    rows.sort(key=lambda r: r["ret"], reverse=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_market_date": asof,
        "round_start": ROUND_START,
        "scoring": "forward-only",
        "start_cash": START_CASH,
        "note": "Live Round 1 — forward-only scoring. Each agent starts a $100,000 paper account at the first market open AFTER it was submitted, and is scored only from there — so no one can optimise against market history they had already seen, and submitting later gives no edge. 'days' is each bot's live window so far. Same data and fills for everyone, refreshed each market day. The winner is the best risk-adjusted result (Calmar) over its live window — there is no historical re-run, because in trading all past data is public and fittable, so the forward window is the only real out-of-sample test. Submissions close ~3 days before the round ends, so every bot gets a minimum live window.",
        "bots": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} ({len(rows)} bots, as of {asof})")

    # ---- history.json — per-day equity curves for the "strategy race" chart ----
    # The board is a single snapshot; this is the time series behind it. We publish
    # the market line + a few archetype-representative strategy curves (anonymized,
    # grouped by style) + the rest of the field as faint context. Honest framing:
    # each curve is the strategy's CURRENT version replayed over the round on real
    # daily bars (no lookahead) — not a frozen daily standing. Private entrants are
    # excluded here (we publish only their numbers on the board, never a full curve).
    eval_dates = [d for d in sorted({b["ts"] for rs in bars.values() for b in rs}) if d > CHART_START]
    qbars = {b["ts"]: b for b in bars.get("QQQ", [])}
    market: list[float] = []
    if eval_dates and qbars:
        base = qbars.get(eval_dates[0], {}).get("open") or next(iter(qbars.values()))["open"]
        last = START_CASH
        for d in eval_dates:
            if d in qbars and base:
                last = round(START_CASH * qbars[d]["close"] / base, 2)
            market.append(last)
    FEATURED = [  # one clean representative per archetype — names hidden, plain-English style
        ("sumegh",                 "All-in on AI & chips"),
        ("mohit",                  "Aggressive, amplified bet"),
        ("dual-momentum-rotation", "Spread across the market"),
        ("sankeerth",              "Defensive, sells fast on drops"),
    ]
    feat_names = {n for n, _ in FEATURED}
    featured = [{"label": lbl, "curve": curves[n]} for n, lbl in FEATURED if curves.get(n)]
    field = [c for n, c in curves.items() if n not in feat_names and c]
    hist = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_market_date": asof,
        "round_start": ROUND_START,
        "start_cash": START_CASH,
        "dates": eval_dates,
        "market": {"label": "Nasdaq-100 (the market)", "curve": market},
        "featured": featured,
        "field": field,
        "note": ("Each line is a strategy's current version replayed over the round on real "
                 "daily bars, no peeking at the future. Names are hidden on purpose; lines are "
                 "grouped by trading style. The board is the snapshot; this is the time series behind it."),
    }
    (HERE / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"wrote history.json ({len(featured)} featured + {len(field)} field, {len(eval_dates)} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
