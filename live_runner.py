"""Live leaderboard runner — produces real, daily-refreshed standings for the
reference ("house") bots on LIVE market data. A GitHub Action runs this each
market day and commits leaderboard.json; the site reads it.

This is honest content, not fakery:
  • The bots are the real reference strategies + admitted entrants in this repo.
  • Numbers are COMPUTED from running them on real daily bars (yfinance), never hardcoded.
  • Each runs a $100,000 paper account from its first scored market session to the latest bar, and
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


ROUND2_FETCH_UNIVERSE = [
    "SPY", "QQQ",
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "AVGO", "AMD", "MU", "MRVL",
    "NFLX", "TSLA", "PLTR", "ORCL", "CRM", "JPM", "V", "MA", "COST", "LLY",
    "SMH", "XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLV", "XLP", "XLU",
    "XLRE", "DIA", "IWM", "SOXX", "QLD", "SSO", "TQQQ", "SOXL", "UPRO", "SPXL",
]
FROZEN_UNIVERSE = set(_load_universe())
UNIVERSE = [t for t in ROUND2_FETCH_UNIVERSE if t in FROZEN_UNIVERSE]

# The live field — file -> (display name, label). Round 2's benchmark is the
# previous winner (Arnav), scored from the Round 2 open. QQQ remains as market
# context only; the line to beat is Arnav.
FIELD = [
    ("qqq_benchmark.py",             "QQQ",                    "market context · QQQ buy & hold"),
]

# Private entrants (read-only deploy-key path). Their CODE never enters this
# PUBLIC repo. We score them locally from PRIVATE_DIR (gitignored) and publish
# only their numbers, persisted in private_results.json — so the cron (which
# can't see their code) keeps them on the board with their last-scored result.
PRIVATE_DIR = HERE / "private_agents"
PRIVATE_RESULTS = HERE / "private_results.json"
PRIVATE_FIELD = [
    ("arnav_agent.py",               "arnav",                  "benchmark · Round 1 winner"),
    # Round 2 entrants. Entrant code stays gitignored in private_agents/; only
    # numbers are published.
    ("eshwar_agent.py",              "eshwar",                 "round 2 · entrant"),
    ("opu_agent.py",                 "opu",                    "round 2 · entrant"),
    ("robert_agent.py",              "robert",                 "round 2 · entrant"),
    ("mohit_agent.py",               "mohit",                  "round 2 · entrant"),
    ("zaid_agent.py",                "zaid",                   "round 2 · entrant"),
    ("sumegh_agent.py",              "sumegh",                 "round 2 · entrant"),
    ("softpeanut_agent.py",          "softpeanut",             "round 2 · entrant"),
    ("nikil_agent.py",               "nikil",                  "round 2 · entrant"),
    ("shyam_agent.py",               "shyam",                  "round 2 · entrant"),
    ("harsimran_agent.py",           "harsimran",              "round 2 · entrant"),
    ("sankeerth_agent.py",           "sankeerth",              "round 2 · entrant"),
    ("siddu_agent.py",               "siddu",                  "round 2 · entrant"),
    ("rohit_agent.py",               "rohit",                  "round 2 · entrant"),
    ("nagarjuna_agent.py",           "nagarjuna",              "round 2 · entrant"),
    ("balaji_agent.py",              "balaji",                 "round 2 · entrant"),
    ("ajai_agent.py",                "ajai",                   "round 2 · entrant"),
    ("aksham_agent.py",              "aksham",                 "round 2 · entrant"),
    ("darshan_agent.py",             "darshan",                "round 2 · entrant"),
    ("tanishq_agent.py",             "tanishq",                "round 2 · entrant"),
    ("aarya_agent.py",               "aarya",                  "round 2 · entrant"),
    ("yog_agent.py",                 "yog",                    "round 2 · entrant"),
    ("krunal_agent.py",              "krunal",                 "round 2 · entrant"),
    ("rohan_agent.py",               "rohan",                  "round 2 · entrant"),
    ("dev_agent.py",                 "dev",                    "round 2 · entrant"),
    ("deepika_agent.py",             "deepika",                "round 2 · entrant"),
    ("om_agent.py",                  "om",                     "round 2 · entrant"),
    ("raam_agent.py",                "raam",                   "round 2 · entrant"),
    ("navika_agent.py",              "navika",                 "round 2 · entrant"),
    ("yuva_agent.py",                "yuva",                   "round 2 · entrant"),
    ("shivkumar_agent.py",           "shivkumar",              "round 2 · entrant"),
    ("sham_agent.py",                "sham",                   "round 2 · entrant"),
    ("rishchith_agent.py",           "rishchith",              "round 2 · entrant"),
    ("meet_agent.py",                "meet",                   "round 2 · entrant"),
    ("vishwas_agent.py",             "vishwas",                "round 2 · entrant"),
    ("mahesh_agent.py",              "mahesh",                 "round 2 · entrant"),
    ("harsh_agent.py",                "harsh",                  "round 2 · entrant"),
    ("arya_agent.py",                 "arya",                   "round 2 · entrant"),
    ("aaryan_agent.py",              "aaryan",                 "round 2 · entrant"),
    ("elamaran_agent.py",            "elamaran",               "round 2 · entrant"),
    ("dhruv_agent.py",               "dhruv",                  "round 2 · entrant"),
    ("vishal_agent.py",              "vishal",                  "round 2 · entrant"),
    ("ddrives_agent.py",             "ddrives",                 "round 2 · entrant · endpoint"),
]

EVAL_DAYS = 60       # (history sizing only) trailing window used when fetching bars
WARMUP_DAYS = 220    # extra history so 200-day signals work
START_CASH = 100_000.0
ROUND_ID = "trading-v0-round-2"
ROUND_NAME = "Round 2"
ROUND_START = "2026-07-07"   # Round 2 starts at the Jul 7, 2026 US market open.
ROUND_STATUS = "live"        # Current board: refresh against the latest fetched market bars.
PRIZE_POOL_USD = 1_000
SCORE_START = ROUND_START    # legacy fallback only; per-bot ENTRY dates below are authoritative

# Per-agent first scored market session. A bot is scored only from sessions on or
# after this date — forward-only — so no one can optimise against market history
# they had already seen, and submitting later gives zero edge.
# "2026-07-07" = live since the Round 2 open.
ENTRY = {
    "QQQ": ROUND_START,
    "arnav": ROUND_START,
    "eshwar": ROUND_START,
    "opu": ROUND_START,
    "robert": ROUND_START,
    "mohit": ROUND_START,
    "zaid": ROUND_START,
    "sumegh": ROUND_START,
    "shyam": ROUND_START,
    "harsimran": ROUND_START,
    # Latest revision received Jul 22; score only from the first session after receipt.
    "sankeerth": "2026-07-23",
    "siddu": ROUND_START,
    "rohit": ROUND_START,
    "nagarjuna": ROUND_START,
    "balaji": "2026-07-31",
    # Revision 2 arrived before the Jul 21 open; score it forward from Jul 21.
    # Final revision received before the Aug 5 market open; score it forward only.
    "ajai": "2026-08-05",
    "aksham": ROUND_START,
    "darshan": ROUND_START,
    "tanishq": ROUND_START,
    "aarya": ROUND_START,
    "yog": ROUND_START,
    "krunal": ROUND_START,
    "rohan": ROUND_START,
    "dev": ROUND_START,
    # Revision 3 received before the Aug 6 market open; do not back-score it.
    "deepika": "2026-08-06",
    "om": ROUND_START,
    "raam": ROUND_START,
    "navika": ROUND_START,
    "yuva": ROUND_START,
    "shivkumar": ROUND_START,
    # Revision received before the Aug 10 market open; score it forward only.
    "sham": "2026-08-10",
    "rishchith": "2026-07-08",
    "meet": "2026-07-30",
    "vishwas": ROUND_START,
    "mahesh": "2026-08-17",
    "harsh": "2026-08-17",
    # These submissions arrived before the Aug 19 US open.
    "softpeanut": "2026-08-19",
    "nikil": "2026-08-19",
    # Arya's revision arrived during the Aug 19 session; score forward from the next session.
    "arya": "2026-08-20",
    # Revision 3 arrived on Aug 8; score it forward from the next US market session.
    "aaryan": "2026-08-10",
    # Corrected submission arrived before the Jul 13 US market open.
    "elamaran": "2026-07-13",
    # Latest revision received Jul 27; score only from the first session after receipt.
    "dhruv": "2026-07-28",
    # Submission arrived after the Jul 20 close; score it forward from Jul 21.
    "vishal": "2026-07-21",
    # Endpoint submission arrived after the Aug 10 close; score from Aug 11.
    "ddrives": "2026-08-17",
}
CHART_START = ROUND_START    # common x-axis for the illustrative race chart (Round 2 open)
SLIP_EQUITY = 0.0005
SLIP_LEVERAGED = 0.0010
MAX_NAME_WEIGHT = 0.30
MAX_BETA_GROSS = 1.50
MAX_ORDERS_PER_DECISION = 100
BETA_3X = {"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU", "CURE", "DRN", "UDOW", "NAIL"}
BETA_2X = {"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"}
PRIZE_SPLIT = ["$600", "$250", "$150"]
POINTS_TABLE = [30, 20, 15, 10, 8, 6, 4, 3, 2, 2]  # max 100 builder points per challenge
BENCHMARK_NAME = "arnav"


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
    return getattr(mod, "AGENT", mod.decide)


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


def _intraday_bar_from_df(df):
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
    if not rows:
        return None
    day = rows[-1]["ts"]
    day_rows = [r for r in rows if r["ts"] == day]
    return {
        "ts": day,
        "open": day_rows[0]["open"],
        "high": max(r["high"] for r in day_rows),
        "low": min(r["low"] for r in day_rows),
        "close": day_rows[-1]["close"],
        "volume": sum(r["volume"] for r in day_rows),
    }


CHUNK = 80  # bounded yfinance batches; large multi-ticker calls can stall


def fetch_bars() -> dict[str, list[dict]]:
    """Fetch daily bars for the whole (~1000-name) universe in batched chunks —
    one yfinance call per chunk, tolerant of any ticker/chunk that fails."""
    need = EVAL_DAYS + WARMUP_DAYS + 30
    bars: dict[str, list[dict]] = {}
    for i in range(0, len(UNIVERSE), CHUNK):
        chunk = UNIVERSE[i:i + CHUNK]
        try:
            raw = yf.download(chunk, period="2y", interval="1d", auto_adjust=True,
                              progress=False, threads=16, group_by="ticker", timeout=8)
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
    # During market hours Yahoo's 1d feed may lag at the prior close. Roll the
    # current 1-minute session into a synthetic daily bar so Round 2 marks from
    # today's open to the latest traded price.
    try:
        raw = yf.download(UNIVERSE, period="1d", interval="1m", auto_adjust=True,
                          progress=False, threads=16, group_by="ticker", timeout=8)
    except Exception:  # noqa: BLE001
        raw = None
    if raw is not None and not getattr(raw, "empty", True):
        multi = hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1
        for t in UNIVERSE:
            try:
                df = raw[t] if multi else raw
            except KeyError:
                continue
            if df is None or df.empty:
                continue
            b = _intraday_bar_from_df(df)
            if not b:
                continue
            current = bars.get(t, [])
            if current and current[-1]["ts"] == b["ts"]:
                current[-1] = b
            elif not current or current[-1]["ts"] < b["ts"]:
                current.append(b)
            bars[t] = current
    return bars


def run_bot(decide, bars: dict[str, list[dict]], entry_date: str = SCORE_START, *, enforce_limits: bool = True) -> dict:
    """Run a $100k paper account from the first scored session on/after entry_date
    to the latest bar — forward-only, so it is never scored before its published
    start.

    The agent decides on info known through the prior close, and orders fill at each
    session's OPEN (+/- slippage) — so the book actually holds through, and captures,
    the day's move.
    """
    all_dates = sorted({b["ts"] for rows in bars.values() for b in rows})
    eval_dates = [d for d in all_dates if d >= entry_date]
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

        # The book is set on info known through the prior close, and orders execute
        # at THIS day's OPEN — i.e. the agent actually holds through, and captures,
        # the session.
        market_state = {t: [b for b in bars[t] if b["ts"] < date] for t in bars}
        prior_close = {t: ms[-1]["close"] for t, ms in market_state.items() if ms}
        portfolio_state = {
            "cash": cash,
            "positions": [{"ticker": t, "quantity": q, "avg_cost": avg_cost.get(t, 0.0)}
                          for t, q in positions.items() if q > 0],
            "last_prices": prior_close,
        }
        try:
            session_decide = getattr(decide, "decide_for_session", None)
            if callable(session_decide):
                orders = session_decide(date, market_state, portfolio_state, cash) or []
            else:
                orders = decide(market_state, portfolio_state, cash) or []
        except Exception:
            orders = []

        # Entrant output is untrusted. The engine owns the final safety boundary:
        # capped input, finite quantities, sell-before-buy execution, and the
        # published 30% single-name / 1.5x beta-gross limits at opening fills.
        normalized = []
        for o in orders[:MAX_ORDERS_PER_DECISION] if isinstance(orders, list) else []:
            try:
                tk = str(o["ticker"]).strip().upper()
                side, qty = o["side"], float(o["quantity"])
            except (KeyError, TypeError, ValueError):
                continue
            if side not in ("buy", "sell") or not math.isfinite(qty) or qty <= 0 or tk not in open_px:
                continue
            normalized.append((tk, side, qty))

        for tk, side, qty in sorted(normalized, key=lambda x: 0 if x[1] == "sell" else 1):
            px = open_px[tk]  # fill at the day's OPEN
            slip = SLIP_LEVERAGED if beta(tk) > 1 else SLIP_EQUITY
            if side == "buy":
                fill = px * (1 + slip)
                equity_at_open = max(
                    cash + sum(positions.get(t, 0.0) * open_px.get(t, 0.0) for t in positions),
                    1e-9,
                )
                held = positions.get(tk, 0.0)
                if enforce_limits:
                    concentration_room = max(0.0, MAX_NAME_WEIGHT * equity_at_open - held * fill)
                    beta_used = sum(
                        positions.get(t, 0.0) * open_px.get(t, 0.0) * beta(t)
                        for t in positions
                    )
                    beta_room = max(0.0, MAX_BETA_GROSS * equity_at_open - beta_used)
                    max_notional = min(cash, concentration_room, beta_room / beta(tk))
                else:
                    max_notional = cash
                qty = min(qty, max_notional / fill if fill > 0 else 0.0)
                if qty <= 0:
                    continue
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
            m = run_bot(load_decide(filename), bars, entry, enforce_limits="market context" not in label.lower())
        except Exception as e:  # noqa: BLE001
            print(f"skip {filename}: {e!r}")
            continue
        rows.append({"name": name, "label": label,
                     "equity": m["equity"], "pnl": m["pnl"],
                     "ret": round(m["ret"], 4), "trades": m["trades"],
                     "days": m["days"], "since": entry, "as_of": asof,
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
    active_private_names = {name for _, name, _ in PRIVATE_FIELD}
    saved = {name: rec for name, rec in saved.items() if name in active_private_names}
    for filename, name, label in PRIVATE_FIELD:
        entry = ENTRY.get(name, SCORE_START)
        p = PRIVATE_DIR / filename
        # Endpoint rows are valid only under the current locked-forward start.
        # Drop any legacy historical-replay record instead of carrying it over.
        if "endpoint" in label.lower() and saved.get(name, {}).get("since") != entry:
            saved.pop(name, None)
        if p.exists():
            try:
                m = run_bot(load_decide_from(p), bars, entry)
                aligned = [START_CASH] * max(0, len(chart_dates) - len(m["curve"])) + m["curve"]
                if m["days"] > 0:
                    saved[name] = {"label": label, "equity": m["equity"], "pnl": m["pnl"],
                                   "ret": round(m["ret"], 4), "trades": m["trades"],
                                   "days": m["days"], "since": entry, "as_of": asof,
                                   "curve": aligned}
                    print(f"  {name:24s} (private) ${m['equity']:,.0f}  P&L {m['pnl']:+,.0f} ({m['ret']*100:+.2f}%)  {m['days']}d  Trades={m['trades']}")
                else:
                    print(f"  {name:24s} (private) pending first scored session from {entry}")
            except Exception as e:  # noqa: BLE001
                print(f"skip private {filename}: {e!r}")
        rec = saved.get(name)
        if rec:
            rows.append({"name": name, "label": rec["label"], "equity": rec["equity"],
                         "pnl": rec["pnl"], "ret": rec["ret"], "trades": rec["trades"],
                         "days": rec.get("days"), "since": rec.get("since"),
                         "as_of": rec.get("as_of")})
            if rec.get("curve"):
                curves[name] = rec["curve"]   # numbers only — keeps the race chart intact
    if saved:
        PRIVATE_RESULTS.write_text(json.dumps(saved, indent=2))

    rows.sort(key=lambda r: (r["ret"], r.get("equity", 0.0)), reverse=True)
    rows_by_name = {r["name"]: r for r in rows}
    benchmark = rows_by_name.get(BENCHMARK_NAME)
    benchmark_ret = benchmark.get("ret", 0.0) if benchmark else 0.0
    entrants = [
        r for r in rows
        if "entrant" in (r.get("label") or "").lower() and r.get("as_of") == asof
    ]
    qualified = [r for r in entrants if r.get("ret", -1e9) > benchmark_ret]
    prize_positions = []
    for rank, (row, prize) in enumerate(zip(qualified, PRIZE_SPLIT), 1):
        prize_positions.append({
            "rank": rank,
            "name": row["name"],
            "ret": row.get("ret"),
            "prize": prize,
        })
    points_positions = []
    for rank, (row, points) in enumerate(zip(qualified, POINTS_TABLE), 1):
        points_positions.append({
            "rank": rank,
            "name": row["name"],
            "ret": row.get("ret"),
            "points": points,
        })
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_market_date": asof,
        "round_id": ROUND_ID,
        "round_name": ROUND_NAME,
        "round_start": ROUND_START,
        "scoring": "forward-only",
        "start_cash": START_CASH,
        "round_status": ROUND_STATUS,
        "prize_pool_usd": PRIZE_POOL_USD,
        "prize_split": {
            "first": PRIZE_SPLIT[0],
            "second": PRIZE_SPLIT[1],
            "third": PRIZE_SPLIT[2],
        },
        "benchmark": {
            "name": BENCHMARK_NAME,
            "label": "Round 1 winner",
            "rule": "Round 2 prize and builder points require beating Arnav's Round 2 forward return.",
            "ret": benchmark_ret,
        },
        "prize_positions": prize_positions,
        "points_positions": points_positions,
        "builder_points_model": {
            "status": "active",
            "max_points_per_challenge": 100,
            "qualifier_rule": "Only entrants who beat the published benchmark qualify for points.",
            "top_10_points": POINTS_TABLE,
            "unused_points": "Unused points are not awarded if fewer than 10 entrants beat the benchmark.",
        },
        "profile_points_model": {
            "status": "active",
            "description": "Each challenge can award at most 100 builder points. For benchmark rounds, only entries that beat the benchmark qualify; top 10 qualifiers receive 30/20/15/10/8/6/4/3/2/2 points.",
            "top_10_points": POINTS_TABLE,
            "benchmark_points": f"Beat {BENCHMARK_NAME}, the prior round winner, over the Round 2 forward window.",
        },
        "note": "Live Round 2 — standings refresh from the latest fetched market bars. Arnav, the Round 1 winner, is the published benchmark. Prize positions and builder points are entrant-only and only unlock for current entries that beat Arnav over the Round 2 forward window. A row whose as_of date trails the board is shown for continuity but cannot hold a prize or points position until it is rescored. Each agent starts a $100,000 paper account at its first scored market session and is scored only from there — so no one can optimise against market history they had already seen, and submitting later gives no edge. 'days' is each bot's live window so far. Same data and fills for everyone.",
        "bots": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} ({len(rows)} bots, as of {asof})")

    # ---- history.json — per-day equity curves for the "strategy race" chart ----
    # The board is a single snapshot; this is the time series behind it. We publish
    # the market line + a few archetype-representative strategy curves (anonymized,
    # grouped by style) + the rest of the field as faint context. Honest framing:
    # each curve is the strategy's CURRENT version replayed over the round on real
    # daily bars (no lookahead) — not a frozen daily standing. Private entrant
    # curves are anonymized here; names stay on the board snapshot only.
    eval_dates = [d for d in sorted({b["ts"] for rs in bars.values() for b in rs}) if d >= CHART_START]
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
        ("arnav", "Round 1 winner benchmark"),
        ("QQQ",   "Nasdaq-100 market context"),
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
