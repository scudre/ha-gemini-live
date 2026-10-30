"""Vertex AI support for the Gemini Live integration."""

from __future__ import annotations

import json
from typing import Any

REQUIRED_KEY_FIELDS = ("client_email", "private_key", "token_uri", "project_id")


class VertexKeyError(ValueError):
    """A service-account key is unusable; the message never contains key data."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def parse_service_account_key(raw: bytes | str) -> dict[str, Any]:
    """Parse and validate an uploaded service-account JSON key."""
    try:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise VertexKeyError("invalid_key_file") from None
    if not isinstance(data, dict):
        raise VertexKeyError("invalid_key_file")
    if data.get("type") != "service_account":
        raise VertexKeyError("not_service_account")
    if not all(data.get(field) for field in REQUIRED_KEY_FIELDS):
        raise VertexKeyError("missing_key_fields")
    return data
