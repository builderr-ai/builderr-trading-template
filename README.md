# builderr trading agent — starter template

Submission template for the **builderr Trading Agent Leaderboard**.

Fork this repo, implement `decide()` in `agent.py`, then send us the repo — **public or private, your call** (private repos use a read-only deploy key; see [«Submission»](#submission)). Submit at https://builderr.ai/trading-v0 (or just email the link to `submit@builderr.ai`). Full rules &amp; FAQ: https://builderr.ai/guidelines

---

> **Building with an AI assistant?** Paste **[`AGENT_BRIEF.md`](AGENT_BRIEF.md)** into Claude/ChatGPT/Cursor and describe your idea — it contains the full contract, rules, scoring, and traps, so your AI can write a compliant bot with you. Fastest cold start.

## 30-second start

1. **Fork this repo** on GitHub.
2. **Implement `decide()`** in `agent.py` — or just rename `baseline.py` to `agent.py` for a 5-minute first submission that gets admitted. The full contract is in the docstring + the [&laquo;The contract&raquo;](#the-contract) section below. `baseline.py`, `example_sector_rotation.py`, and `ai_momentum.py` are real reference bots you can read, run, and beat.
3. **See it clear admission — locally, in ~10 seconds:** run **`python preview.py`**. No engine, no network, no keys, no install. It runs your bot across three real public market windows and prints the same shape of report the real admission email gives you, plus a PASS/FAIL on the safety bar admission actually gates on (clean run, leverage cap, concentration cap, no blow-up). If it says you clear the bar, you're very likely to be admitted.
4. **Push to a GitHub repo** — public, or private with a read-only deploy key (your call; [«Submission»](#submission) explains the trade-offs).
5. **Email the repo URL** to `submit@builderr.ai` (see [&laquo;Submission&raquo;](#submission)). We run admission and email you the score the same day (usually within a few hours). You can resubmit and iterate anytime before your cohort locks — your first try is not your last.

> **`preview.py` vs `selfcheck.py`:** `preview.py` is the one to run — it shows you clearing admission with real numbers. `selfcheck.py` is an even-quicker, data-free smoke test (synthetic bars, just checks `decide()` returns well-formed orders and doesn't crash). Neither is the official eval — we run admission centrally on hidden regimes so it's identical for everyone — but a clean `preview.py` is a strong predictor of admission.
>
> **Want proof it&apos;s fair?** Read `fairness_tests.py` — the actual tests from our engine that guarantee *same code → same score* and *same order → same fill, regardless of who sent it*. (`local_test.py` / `full_test.py` are reference only; they need the private engine.)

> **Secrets:** never commit API keys. You do not need an LLM, brokerage login, or real-money account to enter. If you use an LLM, use endpoint mode or a capped throwaway key.

## Submitted agent: Calmar Rotation Hybrid

`agent.py` is a no-network, no-LLM strategy built for the live Calmar ranking (Round 1: June 2 – July 2, 2026):

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
| `market_state` | `{ticker: [bar, bar, ...]}` — recent **daily** bars per ticker, oldest first (≈220 trading days, ~10 months, including a pre-regime warmup so even 200-day signals work from tick one). Each bar: `{ts, open, high, low, close, volume}`. |
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

The **top ~1000 US names by liquidity** — basically every stock and ETF people actually trade. It's ranked by trailing dollar-volume (from the S&P 500 + Nasdaq-100 + S&P 400/600 + popular ETFs) and **frozen at round open** so the tradeable set is identical for everyone and stable for the whole round. Anything outside it is silently ignored.

The exact frozen list lives in [`universe.json`](universe.json) (the board and the admission engine both read it). It includes all the obvious names — AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO MU MRVL QCOM PLTR COIN JPM V MA UNH LLY XOM … — plus broad/sector/thematic ETFs (SPY QQQ IWM, XLK…XLB, SMH GLD TLT …) and the long leveraged sleeves (3x: TQQQ SOXL UPRO SPXL · 2x: QLD SSO, which count 3x/2x toward the 1.5x gross cap). Regenerate with `python build_universe.py`.

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

### Stage 2 — Live forward test — the ranking

**Round 1 runs June 2 – July 2, 2026 (30 days).** Admitted agents run live on the shared paper sandbox over the window. Same fills for everyone. Daily leaderboard. **Ranked by Calmar** (annualized return / max drawdown). This is the competition. Submissions are open now — the earlier you're admitted, the more of the window your bot trades.

### Stage 3 — Held-out rerun — the anti-luck check

Top finishers are re-run on **fresh windows (calm + stress) they've never seen**. Luck doesn't replicate; skill does. This confirms the winner isn't just the luckiest of the field.

**Prize:** Top 3 by Phase B Calmar (surviving the rerun) split **$2,000** ($1200 / $500 / $300). Top 5 get a LinkedIn spotlight. Winner's code runs on a real **$100k Nasdaq book** post-challenge, with weekly P&L posted publicly on a live ticker from week one — *"win and your code trades my real money."*

---

## Submission

You don't have to make your code public. Pick the path you're comfortable with — same competition, same scoring, regardless. All three: email the link to **submit@builderr.ai** (subject: `builderr submission — <your name>`); we run admission and email your robustness profile the same day (usually within a few hours); if admitted you're in the live round (Round 1: June 2 – July 2).

**1. Public repo** *(simplest)*
Push to a public GitHub repo, email the URL. Zero access setup and you get a public proof-of-work piece — but the field can read your strategy while the contest runs, and a public repo is the easiest place to leak a key. Good if you don't mind being open (or you'll open it after the contest anyway).

**2. Private repo, read-only access** *(protects your edge)*
Keep the repo private. Email us first; we reply with a **read-only deploy key** (one line). You paste it into `Settings → Deploy keys` with **"Allow write access" left OFF**, then reply. We clone, you delete the key after.
- We get **read access to that one repo and nothing else** — we *cannot push to it*, can't see your other repos, and access dies when you remove the key.
- Why not "add us as a collaborator"? On a personal GitHub repo a collaborator gets **write** access. We don't want that and you shouldn't grant it. A deploy key is read-only and scoped to the single repo.

**3. Endpoint mode** *(airtight — you never share code)*
Host an HTTPS endpoint that accepts `POST /decide` with `{market_state, portfolio_state, cash}` and returns `{orders: [...]}`. We send data; you return orders. Your code, prompts, and any API keys never leave your server. Per-agent latency is published on the leaderboard so it stays fair. Include the endpoint URL in your email.

> Whichever you pick: we only ever read and run your code to score it. We don't reuse your strategy, and you keep the IP (this template is MIT; your repo stays yours).

Or email **inquiries@builderr.ai** for early access / questions.

---

## Examples

- `baseline.py` — equal-weight buy-and-hold SPY+QQQ
- More coming as community shares strategies post-launch

---

## Questions

Open a GitHub Discussion on this repo, or email **inquiries@builderr.ai**.
