"""Security boundary for remote trading-agent endpoints."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import socket
import ssl
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

MAX_RESPONSE_BYTES = 1_000_000
MAX_ORDERS = 100
TIMEOUT_SECONDS = 20


class EndpointSecurityError(RuntimeError):
    pass


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def validate_endpoint_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise EndpointSecurityError("endpoint must use HTTPS")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise EndpointSecurityError("endpoint credentials and non-443 ports are not allowed")
    if parsed.fragment:
        raise EndpointSecurityError("endpoint fragments are not allowed")

    try:
        infos = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise EndpointSecurityError(f"endpoint DNS failed: {exc}") from exc
    addresses = {item[4][0] for item in infos}
    if not addresses:
        raise EndpointSecurityError("endpoint DNS returned no addresses")
    for raw in addresses:
        if not ipaddress.ip_address(raw).is_global:
            raise EndpointSecurityError(f"endpoint resolves to non-public address: {raw}")

    if parsed.hostname.endswith(".sslip.io"):
        prefix = parsed.hostname.removesuffix(".sslip.io")
        try:
            embedded = ipaddress.ip_address(prefix)
        except ValueError as exc:
            raise EndpointSecurityError("invalid sslip.io address") from exc
        if not embedded.is_global or str(embedded) not in addresses:
            raise EndpointSecurityError("sslip.io DNS does not match its embedded public address")
    return url


def sanitize_orders(payload) -> list[dict]:  # noqa: ANN001
    orders = payload.get("orders") if isinstance(payload, dict) else payload
    if not isinstance(orders, list):
        raise EndpointSecurityError("response must contain an orders list")
    clean = []
    for order in orders[:MAX_ORDERS]:
        if not isinstance(order, dict):
            continue
        ticker, side, quantity = order.get("ticker"), order.get("side"), order.get("quantity")
        if not isinstance(ticker, str) or not ticker.isascii():
            continue
        ticker = ticker.strip().upper()
        if not ticker or len(ticker) > 10 or side not in {"buy", "sell"}:
            continue
        try:
            quantity = float(quantity)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(quantity) or quantity <= 0:
            continue
        clean.append({"ticker": ticker, "side": side, "quantity": quantity})
    return clean


def call_endpoint(url: str, request_payload: dict) -> dict:
    validate_endpoint_url(url)
    body = json.dumps(request_payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}), _NoRedirects(), HTTPSHandler(context=ssl.create_default_context()))
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise EndpointSecurityError(f"endpoint HTTP error: {exc.code}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise EndpointSecurityError("endpoint response exceeded 1 MB")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EndpointSecurityError("endpoint returned invalid JSON") from exc
    return {
        "orders": sanitize_orders(decoded),
        "latency_ms": latency_ms,
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }
