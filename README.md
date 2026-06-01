# builderr trading agent — starter template

Submission template for the **builderr Trading Agent Leaderboard**.

Fork this repo, implement `decide()` in `agent.py`, push to a public GitHub repo, then submit at https://builderr.ai/trading-v0 (or just email the link to `submit@builderr.ai`). Full rules &amp; FAQ: https://builderr.ai/guidelines

---

## 30-second start

1. **Fork this repo** on GitHub.
2. **Implement `decide()`** in `agent.py`. The full contract is in the docstring + the [&laquo;The contract&raquo;](#the-contract) section below. Look at `baseline.py` and `soham_agent_v2.py` for two real reference implementations (the latter passes Phase A end-to-end).
3. **Push to a public GitHub repo.**
4. **Email the repo URL** to `submit@builderr.ai` (see [&laquo;Submission&raquo;](#submission)). We run Phase A on our infrastructure within 24h and email you the score.

> **Test locally before you submit:** run **`python selfcheck.py`** — it needs no engine, no network, and no keys. It loads your `agent.py`, feeds it synthetic daily bars, and checks `decide()` returns well-formed orders and doesn't crash. It&apos;s a smoke test, not the real eval (we run admission centrally on hidden data so it&apos;s identical for everyone), but it catches the dumb bugs first.
>
> **Want proof it&apos;s fair?** Read `fairness_tests.py` — the actual tests from our engine that guarantee *same code → same score* and *same order → same fill, regardless of who sent it*. (`local_test.py` / `full_test.py` are reference only; they need the private engine.)

> **Secrets:** never commit API keys. You do not need an LLM, brokerage login, or real-money account to enter. If you use an LLM, use endpoint mode or a capped throwaway key.

## Submitted agent: Calmar Rotation Hybrid

`agent.py` is a no-network, no-LLM strategy built for the 60-day Calmar ranking:

- **Risk regime:** risk-on only when SPY and QQQ are above their 50-day SMAs and QQQ 20-day volatility is below 35%.
- **Risk-off book:** XLP / XLU / XLV / XLE with cash left over; no leverage.
- **Risk-on book:** ranks SPY, QQQ, sector ETFs, SMH, and mega-cap tech by 60-day momentum, 20-day momentum, 50-day trend gap, and volatility.
- **Tactical overlay:** adds small QLD / SSO exposure only in calm QQQ uptrends; never uses TQQQ or SOXL by default.
- **Caps:** per-ticker targets stay below 24%, drift rebalance starts above 27%, and beta-adjusted gross is scaled below 1.35x.

Run `python strategy_selftest.py` for strategy-specific cap/regime checks.

---

## The contract

You implement one function:

```python
def decide(market_state, portfolio_state, cash) -> list[dict]:
    return [{"ticker": "SPY", "side": "buy", "quantity": 10}]
```

| Argument | Shape |
|---|---|
| `market_state` | `{ticker: [bar, bar, ...]}` — recent **daily** bars per ticker, oldest first (≈90 trading days of history in admission, including a pre-regime warmup so multi-day signals work from tick one). Each bar: `{ts, open, high, low, close, volume}`. |
| `portfolio_state` | `{cash, positions: [{ticker, quantity, avg_cost}], last_prices: {ticker: price}}` |
| `cash` | Convenience copy of `portfolio_state["cash"]`. |
| **return** | List of orders. Each: `{ticker, side: "buy"\|"sell", quantity: float}`. Empty list = no action. |

`decide()` is called once per decision interval (daily-resolution in admission; finer in Phase B live).

---

## Constraints (auto-enforced)

| Rule | Limit | Breach action |
|---|---|---|
| Side | Long-only | Order rejected |
| Gross beta-adjusted exposure | ≤ 1.5x equity | Sustained breach > 60s → auto-flatten + DQ |
| Position concentration | < 30% per ticker for any 5 trading days | Sustained breach → auto-flatten + DQ |
| Trade rate | ≤ 50 trades/day | Excess rejected |
| Min hold | ≥ 60s | Excess rejected |
| Decide() runtime | ≤ 5s per call | Tick errors out (you keep going) |
| LLM use (optional) | **Bring your own API key** | Your AI spend is yours; keeps the contest about ideas, not API budget |

## Rules of engagement — external data & network

**Your agent has open network access.** Hit any external API: news feeds, alt-data vendors, social sentiment, your own server, an LLM. Real trading bots use external signals; we don't pretend otherwise.

**One absolute rule: no lookahead bias.** Phase A runs in 2026 against historical regimes (2022–2024). At submission time, "live" APIs return present-day data, which for a 2023 backtest *is the future*. If your strategy queries data sources for the regime period at submission time and benefits from knowing what happened, you have lookahead bias.

How we catch it:
1. **Top-10 Phase A submissions get a 10-min human code read.** Patterns like `requests.get("yahoo/SPY/2023-*")` inside the live backtest = DQ. Public postmortem on caught cases.
2. **Phase A ↔ Phase B correlation check.** If your Phase A Sharpe is 6 and your Phase B Sharpe over a comparable horizon is -1, you get flagged for review. Lookahead cheaters leave that signature every time.
3. **Surprise fresh-regime reruns.** During Phase B we re-run qualified agents against new hidden 30-day windows that post-date any internet snapshot you could have queried. Inconsistency = lookahead suspicion.

If you're not sure whether your data source is OK: ask in GitHub Discussions before submitting. If your strategy is genuinely signal-driven (technicals, fundamentals available at the regime time, your own models), you're fine.

**Beta multiples** for the leverage cap:
- 3x: TQQQ, SOXL, UPRO, SPXL, TNA, FAS, TECL, LABU, CURE, DRN, UDOW, NAIL
- 2x: QLD, SSO, DDM, ROM, UWM, AGQ
- 1x: everything else (plain equities + non-leveraged ETFs)

So 100% TQQQ = 3x exposure = instant breach. Max 50% TQQQ + 50% cash works (1.5x exactly).

---

## Universe

Curated set during v0 (real challenge expands to top ~1000 US equities by liquidity at launch):

- Mega-cap tech: AAPL MSFT GOOGL AMZN META NVDA TSLA
- Index ETFs: SPY QQQ DIA IWM
- Sector ETFs: XLK XLF XLE XLV XLI XLY XLP XLU XLRE XLC SMH
- Banking: KRE JPM BAC C WFC
- Leveraged: TQQQ SOXL UPRO SPXL QLD SSO

Tickers outside the universe are silently ignored.

---

## Scoring

We don't gate on whether we like your strategy. Three stages:

### Stage 1 — Admission (immediate, runs on submission)

We run your agent across 3 hidden 30-day historical regimes (shapes only — dates hidden):
1. Fast sector-contagion crash with broader-market spillover
2. Slow trend-down regime change from rate-hike repricing
3. Vol spike + rapid snapback from leveraged-position unwind

**Admission is a smoke screen, NOT a skill gate. You're admitted if:**
- No execution-constraint breach (leverage / concentration)
- No catastrophic blow-up (>50% drawdown in any regime)
- Runs without fatal error

That's it. A fair-weather strategy that's soft in a crash is *admitted* — skill is decided forward, not here. You also get a free **robustness profile** (your Sharpe / drawdown / return across the 3 regimes) so you and we can see whether you're all-weather or fair-weather.

### Stage 2 — Phase B live forward test (60 days) — the ranking

Admitted agents run live on the shared paper sandbox for 60 days from a fixed cohort start. Same fills for everyone. Daily leaderboard. **Ranked by Calmar** (annualized return / max drawdown). This is the competition.

### Stage 3 — Held-out rerun — the anti-luck check

Top finishers are re-run on **fresh windows (calm + stress) they've never seen**. Luck doesn't replicate; skill does. This confirms the winner isn't just the luckiest of the field.

**Prize:** Top 3 by Phase B Calmar (surviving the rerun) split **$2,000** ($1200 / $500 / $300). Top 5 get a LinkedIn spotlight. Winner's code runs on a real **$10k → $50k Nasdaq book** post-challenge with public weekly P&L — *"win and your code trades my real money."*

---

## Submission

When ready:
1. Push your repo to public GitHub.
2. Email the repo URL to **submit@builderr.ai** (subject: `builderr submission — <your name>`).
3. We run admission within 24h; you'll get your robustness profile by email.
4. If admitted, you're in the next Phase B cohort.

Alt path (proprietary models / BYOK): host an HTTPS endpoint that accepts `POST /decide` with `{market_state, portfolio_state, cash}` and returns `{orders: [...]}`. Per-agent latency is published on the leaderboard. Include the endpoint URL in your submission email.

Or email **inquiries@builderr.ai** for early access / questions.

---

## Examples

- `baseline.py` — equal-weight buy-and-hold SPY+QQQ
- More coming as community shares strategies post-launch

---

## Questions

Open a GitHub Discussion on this repo, or email **inquiries@builderr.ai**.
