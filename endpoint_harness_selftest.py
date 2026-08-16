"""Focused adversarial checks for the remote endpoint boundary."""
from __future__ import annotations

import hashlib
import json
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

from endpoint_replay import EndpointReplayAgent
from endpoint_security import EndpointSecurityError, sanitize_orders, validate_endpoint_url


def _dns(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


def test_url_gate_rejects_ssrf_and_unsafe_schemes() -> None:
    with patch("socket.getaddrinfo", return_value=_dns("35.252.153.74")):
        assert validate_endpoint_url("https://35.252.153.74.sslip.io/decide/token")
    for url, ip in [
        ("http://example.com/decide", "93.184.216.34"),
        ("https://127.0.0.1/decide", "127.0.0.1"),
        ("https://169.254.169.254/metadata", "169.254.169.254"),
        ("https://user:pass@example.com/decide", "93.184.216.34"),
        ("https://example.com:8443/decide", "93.184.216.34"),
    ]:
        with patch("socket.getaddrinfo", return_value=_dns(ip)):
            try:
                validate_endpoint_url(url)
            except EndpointSecurityError:
                pass
            else:
                raise AssertionError(f"unsafe endpoint accepted: {url}")


def test_order_schema_drops_executable_and_nonfinite_payloads() -> None:
    payload = {"orders": [
        {"ticker": " qqq ", "side": "buy", "quantity": 2},
        {"ticker": "QQQ", "side": "buy", "quantity": float("inf")},
        {"ticker": "QQQ", "side": "hold", "quantity": 2},
        {"ticker": "__import__('os').system('id')", "side": "buy", "quantity": 1},
        "not-an-order",
    ]}
    assert sanitize_orders(payload) == [{"ticker": "QQQ", "side": "buy", "quantity": 2.0}]


def _signed_record(record: dict) -> dict:
    record = {**record, "previous_record_sha256": None}
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def test_replay_requires_exact_session_market_date_and_valid_hash_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decisions.json"
        record = _signed_record({
            "target_session": "2026-08-17",
            "market_as_of": "2026-08-14",
            "orders": [{"ticker": "QQQ", "side": "buy", "quantity": 1.0}],
        })
        path.write_text(json.dumps({"version": 1, "records": [record]}))
        agent = EndpointReplayAgent(path)
        market = {"QQQ": [{"ts": "2026-08-14", "close": 100.0}]}
        assert len(agent.decide_for_session("2026-08-17", market, {}, 100_000)) == 1
        assert agent.decide_for_session("2026-08-18", market, {}, 100_000) == []
        tampered = json.loads(path.read_text())
        tampered["records"][0]["orders"][0]["quantity"] = 999_999
        path.write_text(json.dumps(tampered))
        assert agent.decide_for_session("2026-08-17", market, {}, 100_000) == []


def run() -> None:
    test_url_gate_rejects_ssrf_and_unsafe_schemes()
    test_order_schema_drops_executable_and_nonfinite_payloads()
    test_replay_requires_exact_session_market_date_and_valid_hash_chain()
    print("endpoint_harness_selftest: PASS")


if __name__ == "__main__":
    run()
