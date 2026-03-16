#!/usr/bin/env python3
"""Minimal OpenRouter credit tracking utilities for run scripts."""

from __future__ import annotations

import atexit
import json
import urllib.request
from typing import Callable

OPENROUTER_AUTH_KEY_API = "https://openrouter.ai/api/v1/auth/key"


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def fetch_credit_snapshot(api_key: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        OPENROUTER_AUTH_KEY_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ProxyPrior Credit Tracker",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload.get("data", payload)
    usage = _to_float(data.get("usage"))
    limit = _to_float(data.get("limit"))
    remaining = None
    if usage is not None and limit is not None:
        remaining = limit - usage

    return {
        "usage": usage,
        "limit": limit,
        "remaining": remaining,
    }


def _format_line(prefix: str, snapshot: dict) -> str:
    usage = snapshot.get("usage")
    limit = snapshot.get("limit")
    remaining = snapshot.get("remaining")

    usage_text = f"{usage:.6f}" if isinstance(usage, float) else "unknown"
    limit_text = f"{limit:.6f}" if isinstance(limit, float) else "unknown"
    remaining_text = f"{remaining:.6f}" if isinstance(remaining, float) else "unknown"
    return f"{prefix} usage={usage_text} limit={limit_text} remaining={remaining_text}"


def start_credit_tracking(
    *,
    api_key: str | None,
    log_fn: Callable[[str], None],
    print_fn: Callable[[str], None] = print,
) -> None:
    if not api_key:
        return

    start_snapshot: dict | None = None

    try:
        start_snapshot = fetch_credit_snapshot(api_key)
        line = _format_line("openrouter_credits_start", start_snapshot)
        log_fn(line)
        print_fn(line)
    except Exception as exc:
        msg = f"openrouter_credits_start unavailable error={exc}"
        log_fn(msg)
        print_fn(msg)

    def _on_exit() -> None:
        try:
            end_snapshot = fetch_credit_snapshot(api_key)
            line = _format_line("openrouter_credits_end", end_snapshot)
            log_fn(line)
            print_fn(line)
            if start_snapshot and isinstance(start_snapshot.get("usage"), float) and isinstance(end_snapshot.get("usage"), float):
                used = float(end_snapshot["usage"]) - float(start_snapshot["usage"])
                delta_line = f"openrouter_credits_delta used_this_run={used:.6f}"
                log_fn(delta_line)
                print_fn(delta_line)
        except Exception as exc:
            msg = f"openrouter_credits_end unavailable error={exc}"
            log_fn(msg)
            print_fn(msg)

    atexit.register(_on_exit)
