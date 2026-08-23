"""Focused checks for live_runner.py.

Run:
    python live_runner_selftest.py
"""
from __future__ import annotations

import live_runner


def _bar(ts: str, open_: float, close: float) -> dict:
    return {
        "ts": ts,
        "open": open_,
        "high": max(open_, close),
        "low": min(open_, close),
        "close": close,
        "volume": 1_000_000,
    }


def _buy_one_qqq(_market_state: dict, portfolio_state: dict, _cash: float) -> list[dict]:
    if portfolio_state.get("positions"):
        return []
    return [{"ticker": "QQQ", "side": "buy", "quantity": 1}]


def test_same_day_entry_scores_opening_session() -> None:
    bars = {
        "QQQ": [
            _bar("2026-07-06", 100.0, 100.0),
            _bar("2026-07-07", 100.0, 110.0),
        ],
    }

    result = live_runner.run_bot(_buy_one_qqq, bars, "2026-07-07")

    assert result["days"] == 1, result
    assert result["trades"] == 1, result
    assert result["equity"] > live_runner.START_CASH, result


def test_future_entry_does_not_backfill() -> None:
    bars = {
        "QQQ": [
            _bar("2026-07-06", 100.0, 100.0),
            _bar("2026-07-07", 100.0, 110.0),
        ],
    }

    result = live_runner.run_bot(_buy_one_qqq, bars, "2026-07-08")

    assert result["days"] == 0, result
    assert result["trades"] == 0, result
    assert result["equity"] == live_runner.START_CASH, result


def test_every_board_row_has_an_explicit_start() -> None:
    names = {name for _, name, _ in [*live_runner.FIELD, *live_runner.PRIVATE_FIELD]}
    assert names <= set(live_runner.ENTRY), names - set(live_runner.ENTRY)
    assert live_runner.ENTRY["aaryan"] == "2026-08-10"
    assert live_runner.ENTRY["meet"] == "2026-08-24"
    assert live_runner.ENTRY["elamaran"] == "2026-07-13"
    assert live_runner.ENTRY["ddrives"] == "2026-08-17"


def test_session_aware_agent_receives_exact_fill_session() -> None:
    class SessionAgent:
        def __init__(self):
            self.seen = []

        def decide_for_session(self, session_date, _market, portfolio, _cash):
            self.seen.append(session_date)
            return [] if portfolio.get("positions") else [{"ticker": "QQQ", "side": "buy", "quantity": 1}]

    agent = SessionAgent()
    bars = {
        "QQQ": [
            _bar("2026-07-06", 100.0, 100.0),
            _bar("2026-07-07", 100.0, 101.0),
            _bar("2026-07-08", 101.0, 102.0),
        ],
    }
    live_runner.run_bot(agent, bars, "2026-07-07")
    assert agent.seen == ["2026-07-07", "2026-07-08"], agent.seen


def test_engine_enforces_concentration_and_beta_caps_across_split_orders() -> None:
    def malicious(_market, _portfolio, _cash):
        return [
            {"ticker": "TQQQ", "side": "buy", "quantity": 1_000_000},
            {"ticker": "TQQQ", "side": "buy", "quantity": 1_000_000},
            {"ticker": "QQQ", "side": "buy", "quantity": float("nan")},
        ]

    bars = {
        "TQQQ": [_bar("2026-07-06", 100.0, 100.0), _bar("2026-07-07", 100.0, 100.0)],
        "QQQ": [_bar("2026-07-06", 100.0, 100.0), _bar("2026-07-07", 100.0, 100.0)],
    }
    result = live_runner.run_bot(malicious, bars, "2026-07-07")
    holdings = {h["t"]: h["q"] for h in result["holdings"]}
    tqqq_notional = holdings.get("TQQQ", 0.0) * 100.0
    assert tqqq_notional <= live_runner.START_CASH * live_runner.MAX_NAME_WEIGHT + 0.01, result
    assert tqqq_notional * 3.0 <= live_runner.START_CASH * live_runner.MAX_BETA_GROSS + 0.01, result
    assert "QQQ" not in holdings, result


def test_market_context_can_remain_full_buy_and_hold() -> None:
    bars = {"QQQ": [_bar("2026-07-06", 100.0, 100.0), _bar("2026-07-07", 100.0, 110.0)]}
    def buy_full(_market, _portfolio, _cash):
        return [{"ticker": "QQQ", "side": "buy", "quantity": 1_000_000}]
    result = live_runner.run_bot(buy_full, bars, "2026-07-07", enforce_limits=False)
    holdings = {h["t"]: h["q"] for h in result["holdings"]}
    assert holdings["QQQ"] * 110.0 > live_runner.START_CASH * live_runner.MAX_NAME_WEIGHT, result


def run() -> None:
    test_same_day_entry_scores_opening_session()
    test_future_entry_does_not_backfill()
    test_every_board_row_has_an_explicit_start()
    test_session_aware_agent_receives_exact_fill_session()
    test_engine_enforces_concentration_and_beta_caps_across_split_orders()
    test_market_context_can_remain_full_buy_and_hold()
    print("live_runner_selftest: PASS")


if __name__ == "__main__":
    run()
