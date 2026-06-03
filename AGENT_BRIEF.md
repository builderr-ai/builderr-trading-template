# builderr Trading Agent — Build Brief (paste this into your AI)

**New here? Copy this whole file, paste it into your AI assistant (Claude, ChatGPT, Cursor…),
and say: "help me build a trading agent for this challenge."** It has everything the AI needs —
the contract, the rules, what to optimize for, and the traps to avoid. You don't need to be a
quant or a deep coder; you need a clear idea and this brief.

---

## In one line
You write **one small Python function, `decide()`**, that each day looks at recent prices and
returns buy/sell orders. We run it on a shared market sandbox. The best **risk-adjusted**
performer over 30 days wins.

## The contract (the only code you must write)

```python
def decide(market_state, portfolio_state, cash) -> list[dict]:
    # Return orders, e.g. [{"ticker": "SPY", "side": "buy", "quantity": 10}]
    # Return [] to do nothing this day.
    return []
```

- **`market_state`** — `{ticker: [bar, bar, ...]}`. Recent **daily** bars per ticker, oldest
  first (~220 trading days, so even a 200-day average works from day one).
  Each bar: `{ts, open, high, low, close, volume}`.
- **`portfolio_state`** — `{cash, positions: [{ticker, quantity, avg_cost}], last_prices: {ticker: price}}`.
- **`cash`** — your spendable cash (same as `portfolio_state["cash"]`).
- **returns** — a list of orders, each `{ticker, side: "buy" | "sell", quantity}`.

Your function is called **once per day**. That's it — no event loop, no broker, no infra.

## The rules (auto-enforced — keep your bot inside these)

| Rule | Limit |
|---|---|
| Long only | no short-selling (v0) |
| Leverage | gross exposure ≤ **1.5×** your equity (beta-adjusted — see below) |
| Concentration | no single position ≥ **30%** for more than 5 trading days |
| Trades per day | ≤ **50** |
| Minimum hold | ≥ **60 seconds** |
| `decide()` runtime | ≤ **5 seconds** per call |

**Beta multiples** (for the leverage cap): 3× = TQQQ, SOXL, UPRO, SPXL, TNA, FAS, TECL, LABU,
CURE, DRN, UDOW, NAIL · 2× = QLD, SSO, DDM, ROM, UWM, AGQ · 1× = everything else.
So 100% TQQQ = 3× = breach. 50% TQQQ + 50% cash = 1.5× = OK.

## Universe (what you can trade)
The **top ~1000 US names by liquidity** — basically everything that actually trades: AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO MU MRVL QCOM PLTR COIN JPM V MA UNH LLY XOM CVX … plus broad/sector/thematic ETFs (SPY QQQ IWM, XLK…XLB, SMH GLD TLT …) and long leveraged sleeves (3x: TQQQ SOXL UPRO SPXL · 2x: QLD SSO).
- Ranked by trailing dollar-volume and **frozen at round open** (same list for everyone, stable all round). Exact list: **`universe.json`** in the repo.
- Leveraged ETFs count 3x/2x toward the 1.5x gross cap; everything else is 1x. Anything off the list is silently ignored.

Tickers outside the universe are ignored.

## How you're scored
1. **Admission (instant, on submit).** A safety check, *not* a skill test. You're in if your bot
   runs cleanly, respects the caps, and doesn't blow up (>50% drawdown) across 3 hidden past
   market periods. You get a "robustness profile" (how it behaved) back by email.
2. **Round 1 — the ranking (June 2 – July 2, 2026, 30 days).** Your bot trades live on the shared
   sandbox. Ranked by **Calmar = annualized return ÷ worst drawdown.** Plain version:
   **+10% with a −2% dip beats +30% with a −25% dip.** Make money *without a deep hole.*
3. **Re-run.** Top finishers are re-run on market windows they never saw — to confirm skill, not luck.

## What to optimize for — and what this is NOT
- **NOT** high-frequency trading, **NOT** "trade the most," **NOT** max-leverage gambling,
  **NOT** a latency or spending race. Daily decisions + a 60s minimum hold mean your speed,
  servers, and budget don't matter.
- **IS** a calm, risk-adjusted strategy that survives: a risk-off switch, sane position sizing,
  robust enough to hold up live and on unseen windows. **The best idea wins, not the fastest machine.**

## Best practices
- **A risk-off switch beats a clever entry.** Going to cash when SPY is below its 50/200-day
  average does more for your Calmar than any fancy signal.
- **Size by volatility, not fixed dollars.** Cut size as volatility rises.
- **Change every parameter ±20%.** If performance collapses, you overfit — it'll die live.
- **Keep it simple.** Fewer parameters = fewer ways to fool yourself.
- **Leveraged ETFs (TQQQ/SOXL) decay in chop** — tactical only, never buy-and-hold.

## Common traps (these lose, fast)
- **Lookahead** — using data your bot couldn't have had in real time → disqualified.
- **Curve-fitting** — tuning until history looks perfect → dies live.
- **Over-leverage** — the 1.5× cap auto-flattens you.
- **No drawdown management** — one bad stretch ends your run.

## Starting points (real, in this repo)
- **Strategies people use:** dual / absolute momentum · sector rotation + SMA risk-off · volatility targeting.
- **Read the reference bots:** `seed_dual_momentum.py`, `example_sector_rotation.py`,
  `example_vol_target.py`, and `baseline.py` (the simplest bot that still gets admitted).

## Test locally first (no install, no network, no keys)
- **`python preview.py`** — runs your bot on real sample windows and tells you if you'd clear
  the safety bar (~10 seconds).
- **`python selfcheck.py`** — quick smoke test; also warns if you've committed a secret.

## Submit
Implement `decide()` in `agent.py` → push to a GitHub repo (public, or private with a read-only
deploy key) → email the link to **submit@builderr.ai**. Resubmit anytime during Round 1.

## If you win — know this going in
Finish **top 3** and your agent is shared with everyone who backed the bounty; **#1 trades a real
$100k** on Nasdaq. Below top 3, nothing of yours is shared. You keep ownership either way.

---

### Now tell your AI what you want
> "Using the contract and rules above, write me a `decide()` function that **[your idea — e.g.
> holds the strongest sector ETF by 3-month momentum, but moves to cash when SPY is below its
> 200-day average]**. Keep every position under 30% and gross leverage under 1.5×, and don't use
> any data from the future."

Full rules & FAQ: https://builderr.ai/guidelines · Build guide: https://builderr.ai/start
