"""Build the trading universe — top ~1000 liquid US names, FROZEN at round open.

"Dynamic" the right way: we rank a broad candidate pool (S&P 500 + Nasdaq-100 +
S&P MidCap 400 + popular ETFs + retail favorites) by trailing dollar-volume,
take the top N, and write a single committed snapshot (universe.json). The board
and the admission engine both read that snapshot, so the tradeable set is the
same for everyone and STABLE for the whole round (a name can't drop out mid-round).

Run once at round open:  python build_universe.py
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

TOP_N = 1000
UA = {"User-Agent": "Mozilla/5.0 (builderr universe builder)"}

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "ndx": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

# Always-include tradeable ETFs + leveraged sleeves (kept regardless of rank).
ETFS = [
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
    "SMH", "SOXX", "IGV", "ARKK", "ARKQ", "XBI", "IBB", "KRE", "ITB", "XHB",
    "GDX", "GLD", "SLV", "TLT", "HYG", "USO", "VNQ", "EEM", "EFA",
]
LEVERAGED = ["TQQQ", "SOXL", "UPRO", "SPXL", "QLD", "SSO"]
RETAIL = ["PLTR", "COIN", "SOFI", "HOOD", "RBLX", "DKNG", "RIVN", "LCID", "SNAP", "PINS", "ROKU", "U", "DASH"]


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def candidate_stocks() -> set[str]:
    syms: set[str] = set()
    for name, url in WIKI.items():
        try:
            r = requests.get(url, headers=UA, timeout=30)
            tables = pd.read_html(io.StringIO(r.text))
            for t in tables:
                col = next((c for c in t.columns if str(c).lower() in ("symbol", "ticker", "ticker symbol")), None)
                if col is None:
                    continue
                for s in t[col].dropna():
                    s = _norm(s)
                    if 1 <= len(s) <= 6 and all(ch.isalpha() or ch == "-" for ch in s):
                        syms.add(s)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {name} failed: {e!r}")
    print(f"  candidate stocks from index lists: {len(syms)}")
    return syms


def rank_by_dollar_volume(tickers: list[str]) -> list[str]:
    """Median daily dollar-volume over ~3 months, descending. Chunked + tolerant."""
    dv: dict[str, float] = {}
    for i in range(0, len(tickers), 150):
        chunk = tickers[i:i + 150]
        try:
            raw = yf.download(chunk, period="3mo", interval="1d", auto_adjust=True,
                              progress=False, threads=True, group_by="ticker")
        except Exception as e:  # noqa: BLE001
            print(f"  ! chunk {i} failed: {e!r}")
            continue
        multi = hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1
        for t in chunk:
            try:
                df = raw[t] if multi else raw
                cols = {str(c).lower(): c for c in df.columns}
                px = df[cols["close"]]
                vol = df[cols["volume"]]
                series = (px * vol).dropna()
                if len(series) >= 20:
                    dv[t] = float(series.median())
            except Exception:  # noqa: BLE001
                continue
        time.sleep(1)
    return sorted(dv, key=lambda t: dv[t], reverse=True)


def main() -> int:
    print("Building universe snapshot…")
    stocks = candidate_stocks() | set(RETAIL)
    ranked = rank_by_dollar_volume(sorted(stocks))
    print(f"  ranked by dollar-volume: {len(ranked)}")

    keep_stocks = ranked[: max(0, TOP_N - len(ETFS) - len(LEVERAGED))]
    universe: list[str] = []
    seen: set[str] = set()
    for t in keep_stocks + ETFS + LEVERAGED:
        if t not in seen:
            seen.add(t)
            universe.append(t)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "top ~1000 US names by trailing dollar-volume (S&P 500 + Nasdaq-100 + S&P 400/600 + ETFs), frozen at round open",
        "count": len(universe),
        "tickers": universe,
    }
    with open("universe.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote universe.json — {len(universe)} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
