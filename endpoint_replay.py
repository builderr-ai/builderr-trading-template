"""Replay immutable, pre-session endpoint decisions without network access."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


class EndpointReplayAgent:
    def __init__(self, decision_log: Path):
        self.decision_log = decision_log

    def _records(self) -> list[dict]:
        if not self.decision_log.exists():
            return []
        try:
            data = json.loads(self.decision_log.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        records = data.get("records", []) if isinstance(data, dict) else []
        if not isinstance(records, list):
            return []
        previous = None
        for record in records:
            if not isinstance(record, dict) or record.get("previous_record_sha256") != previous:
                return []
            unsigned = {k: v for k, v in record.items() if k != "record_sha256"}
            actual = hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if record.get("record_sha256") != actual:
                return []
            previous = actual
        return records

    def has_locked_decision_for(self, session_date: str) -> bool:
        """Return whether this endpoint has an untampered decision for a session.

        A missing pre-session capture is evaluator state, not an instruction for
        the entrant to make a no-op decision.  Callers use this to preserve the
        last comparable score until a later valid capture is available.
        """
        return sum(r.get("target_session") == session_date for r in self._records()) == 1

    def decide_for_session(self, session_date, market_state, _portfolio_state, _cash):  # noqa: ANN001
        market_dates = [bars[-1].get("ts") for bars in market_state.values() if bars]
        market_as_of = max((str(d) for d in market_dates if d), default=None)
        matches = [r for r in self._records() if r.get("target_session") == session_date]
        if len(matches) != 1 or matches[0].get("market_as_of") != market_as_of:
            return []
        orders = matches[0].get("orders")
        return orders if isinstance(orders, list) else []

    def __call__(self, _market_state, _portfolio_state, _cash):
        return []
