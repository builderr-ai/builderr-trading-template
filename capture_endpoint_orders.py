"""Capture remote endpoint orders before a market session and lock them for replay."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

import live_runner
from endpoint_security import EndpointSecurityError, call_endpoint

HERE = Path(__file__).parent
ENDPOINT_AGENTS = ("ddrives_agent.py",)


def _load_private_agent(filename: str):
    path = live_runner.PRIVATE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"capture_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for field in ("ENDPOINT_URL", "ENTRY_DATE", "DECISION_LOG", "AGENT"):
        if not hasattr(module, field):
            raise RuntimeError(f"{filename} missing {field}")
    return module


def _next_session_open() -> datetime:
    status = yf.Market("us_market").status
    market_open = status.get("open") if status else None
    if not isinstance(market_open, datetime):
        raise RuntimeError("market calendar did not return the next session open")
    if market_open.tzinfo is None:
        market_open = market_open.replace(tzinfo=timezone.utc)
    return market_open.astimezone(timezone.utc)


def _append_locked(path: Path, record: dict) -> None:
    data = {"version": 1, "records": []}
    if path.exists():
        data = json.loads(path.read_text())
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise RuntimeError(f"invalid decision log: {path}")
    if any(r.get("target_session") == record["target_session"] for r in records):
        print(f"{path.stem}: {record['target_session']} already locked")
        return
    previous = records[-1].get("record_sha256") if records else None
    record["previous_record_sha256"] = previous
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def capture_one(module, bars: dict, target_open: datetime) -> None:  # noqa: ANN001
    target_session = target_open.date().isoformat()
    if target_session < module.ENTRY_DATE:
        print(f"{module.DECISION_LOG.stem}: entry begins {module.ENTRY_DATE}; next session is {target_session}")
        return
    if datetime.now(timezone.utc) >= target_open:
        raise RuntimeError(f"refusing late capture for {target_session}; market session has begun")

    market_as_of = max(b["ts"] for rows in bars.values() for b in rows)
    if market_as_of >= target_session:
        raise RuntimeError(f"market payload is not point-in-time: {market_as_of} >= {target_session}")
    if module.DECISION_LOG.exists():
        existing = json.loads(module.DECISION_LOG.read_text()).get("records", [])
        if any(r.get("target_session") == target_session for r in existing):
            print(f"{module.DECISION_LOG.stem}: {target_session} already locked")
            return

    state = live_runner.run_bot(module.AGENT, bars, module.ENTRY_DATE)
    last_prices = {ticker: rows[-1]["close"] for ticker, rows in bars.items() if rows}
    portfolio_state = {
        "cash": state["cash"],
        "positions": [
            {"ticker": h["t"], "quantity": h["q"], "avg_cost": 0.0}
            for h in state["holdings"]
        ],
        "last_prices": last_prices,
    }
    request_payload = {
        "market_state": bars,
        "portfolio_state": portfolio_state,
        "cash": state["cash"],
    }
    result = call_endpoint(module.ENDPOINT_URL, request_payload)
    record = {
        "target_session": target_session,
        "market_as_of": market_as_of,
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latency_ms": result["latency_ms"],
        "request_sha256": result["request_sha256"],
        "response_sha256": result["response_sha256"],
        "orders": result["orders"],
    }
    _append_locked(module.DECISION_LOG, record)
    print(f"{module.DECISION_LOG.stem}: locked {len(result['orders'])} orders for {target_session}")


def main() -> int:
    try:
        target_open = _next_session_open()
        modules = [_load_private_agent(name) for name in ENDPOINT_AGENTS]
        if all(
            module.DECISION_LOG.exists()
            and any(
                r.get("target_session") == target_open.date().isoformat()
                for r in json.loads(module.DECISION_LOG.read_text()).get("records", [])
            )
            for module in modules
        ):
            print(f"endpoint decisions already locked for {target_open.date().isoformat()}")
            return 0
        bars = live_runner.fetch_bars()
        if len(bars) < 12:
            raise RuntimeError(f"only {len(bars)} tickers fetched")
        for module in modules:
            capture_one(module, bars, target_open)
        return 0
    except (EndpointSecurityError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"endpoint capture failed closed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
