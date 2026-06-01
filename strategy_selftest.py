"""Strategy-level checks for agent.py.

No network, no private engine, no third-party packages. These are not the
official builderr evals; they catch contract, cap, and regime bugs before
submission.

Run:
    python strategy_selftest.py
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import agent


UNIVERSE = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "SMH",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "QLD", "SSO",
)


def bars(start: float, returns: list[float]) -> list[dict]:
    out = []
    px = start
    d = date(2024, 1, 1)
    for r in returns:
        px *= 1.0 + r
        out.append({
            "ts": d.isoformat(),
            "open": px,
            "high": px * 1.01,
            "low": px * 0.99,
            "close": px,
            "volume": 1_000_000,
        })
        d += timedelta(days=1)
    return out


def market(kind: str) -> dict[str, list[dict]]:
    if kind == "risk_off":
        base = [-0.003] * 90
        defensive = [0.0005] * 90
        return {t: bars(100.0, defensive if t in {"XLP", "XLU", "XLV", "XLE"} else base) for t in UNIVERSE}

    if kind == "high_vol":
        calm_up = [0.002] * 90
        qqq_chop = ([0.035, -0.03] * 45)
        data = {t: bars(100.0, calm_up) for t in UNIVERSE}
        data["QQQ"] = bars(100.0, qqq_chop)
        return data

    # Low-vol risk-on, with differentiated momentum.
    data = {t: bars(100.0, [0.001] * 90) for t in UNIVERSE}
    for t in ("SMH", "NVDA", "XLK"):
        data[t] = bars(100.0, [0.004] * 90)
    for t in ("QQQ", "AAPL", "META"):
        data[t] = bars(100.0, [0.0025] * 90)
    data["SPY"] = bars(100.0, [0.0018] * 90)
    data["QLD"] = bars(100.0, [0.0048] * 90)
    data["SSO"] = bars(100.0, [0.0034] * 90)
    return data


def reset_agent_state() -> None:
    agent._last_rebalance_bar_date = None
    agent._last_targets = {}


def beta_gross(weights: dict[str, float]) -> float:
    return sum(w * agent.BETA_MULTIPLE.get(t, 1.0) for t, w in weights.items())


def test_empty_data_returns_no_orders() -> None:
    reset_agent_state()
    assert agent.decide({}, {"cash": 100_000, "positions": [], "last_prices": {}}, 100_000) == []


def test_insufficient_history_returns_no_targets() -> None:
    short_market = {t: bars(100.0, [0.001] * 40) for t in UNIVERSE}
    assert agent.target_weights(short_market) == {}


def test_risk_off_uses_defensive_book() -> None:
    weights = agent.target_weights(market("risk_off"))
    assert set(weights).issubset({"XLP", "XLU", "XLV", "XLE"})
    assert weights


def test_risk_on_selects_positive_momentum() -> None:
    weights = agent.target_weights(market("risk_on"))
    assert {"SMH", "NVDA", "XLK"} & set(weights)
    assert len(weights) >= 4


def test_high_vol_disables_leverage() -> None:
    weights = agent.target_weights(market("high_vol"))
    assert "QLD" not in weights
    assert "SSO" not in weights


def test_caps_hold() -> None:
    for kind in ("risk_off", "high_vol", "risk_on"):
        weights = agent.target_weights(market(kind))
        assert all(w < 0.240001 for w in weights.values()), (kind, weights)
        assert beta_gross(weights) <= 1.350001, (kind, weights, beta_gross(weights))


def test_orders_are_bounded_and_fast() -> None:
    reset_agent_state()
    m = market("risk_on")
    latest = {t: b[-1]["close"] for t, b in m.items()}
    portfolio = {"cash": 100_000.0, "positions": [], "last_prices": latest}
    start = time.perf_counter()
    orders = agent.decide(m, portfolio, 100_000.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05, elapsed
    assert 0 < len(orders) < 50, orders
    assert all(o["side"] in {"buy", "sell"} and o["quantity"] > 0 for o in orders)
    assert agent.decide(m, portfolio, 100_000.0) == []


def test_tiny_stale_position_is_not_sold() -> None:
    orders = agent.orders_to_rebalance(
        targets={"SPY": 0.20},
        positions={"XYZ": {"quantity": 0.5, "avg_cost": 100.0}},
        total_equity=100_000.0,
        prices={"XYZ": 100.0, "SPY": 500.0},
        cash_available=0.0,
    )
    assert orders == []


def run() -> None:
    tests = [
        test_empty_data_returns_no_orders,
        test_insufficient_history_returns_no_targets,
        test_risk_off_uses_defensive_book,
        test_risk_on_selects_positive_momentum,
        test_high_vol_disables_leverage,
        test_caps_hold,
        test_orders_are_bounded_and_fast,
        test_tiny_stale_position_is_not_sold,
    ]
    for test in tests:
        test()
    print(f"✓ {len(tests)} strategy checks passed.")


if __name__ == "__main__":
    run()
