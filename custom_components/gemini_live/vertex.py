"""Vertex AI support for the Gemini Live integration."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from .gemini import GeminiLiveClient

VERTEX_CONNECT_TIMEOUT = 20
VERTEX_RUNTIME_ERROR_PREFIX = "Gemini Live (Vertex AI) request failed: "
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
REQUIRED_KEY_FIELDS = ("client_email", "private_key", "token_uri", "project_id")
_VERTEX_RUNTIME_ADVICE = (
    "Check the Google Cloud project, or use Reconfigure to update "
    "the service account key, location or model."
)

_CREDENTIALS_CACHE: dict[tuple[str, str], Any] = {}
_CREDENTIALS_LOCK = threading.Lock()


class VertexRuntimeMessage(str):
    """User-facing Vertex AI error text that also keeps Google's raw text."""

    error_detail: str


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


def _refreshed_credentials(credentials_info: dict[str, Any]) -> Any:
    """Return cached service-account credentials holding a valid token.

    Blocking. google-genai refreshes credentials without a token inside its
    async connect(), which would block the event loop, so refresh them here.
    """
    from google.auth.transport.requests import Request  # noqa: PLC0415
    from google.oauth2 import service_account  # noqa: PLC0415

    cache_key = (
        credentials_info["client_email"],
        credentials_info.get("private_key_id", ""),
    )
    with _CREDENTIALS_LOCK:
        credentials = _CREDENTIALS_CACHE.get(cache_key)
        if credentials is None:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info, scopes=[_CLOUD_PLATFORM_SCOPE]
            )
            _CREDENTIALS_CACHE[cache_key] = credentials
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials


def create_vertex_genai_client(
    credentials_info: dict[str, Any], location: str, http_options: Any = None
) -> Any:
    """Build a Vertex AI Google Gen AI client; blocking, run in an executor."""
    from google import genai  # noqa: PLC0415

    credentials = _refreshed_credentials(credentials_info)
    return genai.Client(
        vertexai=True,
        project=credentials_info["project_id"],
        location=location,
        credentials=credentials,
        http_options=http_options,
    )


async def async_create_vertex_gemini_client(
    hass: Any,
    credentials_info: dict[str, Any],
    location: str,
    affective_dialog: bool = False,
) -> GeminiLiveClient:
    """Create the Vertex AI client and wrap it in the neutral adapter."""
    http_options = None
    if affective_dialog:
        from google.genai import types  # noqa: PLC0415

        # Affective dialog needs the v1beta1 endpoint on Vertex AI.
        http_options = types.HttpOptions(api_version="v1beta1")
    client = await hass.async_add_executor_job(
        create_vertex_genai_client, credentials_info, location, http_options
    )
    return GeminiLiveClient(client)


def vertex_error_text(exc: BaseException) -> str:
    """Return Google's message verbatim, or the class name when it is empty."""
    return str(exc).strip() or type(exc).__name__


async def async_validate_vertex_connection(
    hass: Any, credentials_info: dict[str, Any], location: str, model: str
) -> str | None:
    """Open and close a Live session; return an error text or None on success."""
    try:
        async with asyncio.timeout(VERTEX_CONNECT_TIMEOUT):
            client = await hass.async_add_executor_job(
                create_vertex_genai_client, credentials_info, location, None
            )
            async with client.aio.live.connect(
                model=model, config={"response_modalities": ["AUDIO"]}
            ):
                pass
    except TimeoutError:
        return f"Timed out after {VERTEX_CONNECT_TIMEOUT} seconds connecting to Vertex AI"
    except Exception as exc:  # noqa: BLE001
        return vertex_error_text(exc)
    return None


def vertex_runtime_error_message(exc: BaseException) -> str | None:
    """Return user-facing text for a Google API or auth error in the chain."""
    from google.auth import exceptions as auth_errors  # noqa: PLC0415
    from google.genai import errors  # noqa: PLC0415

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, errors.APIError) and current.code != 1000
        ) or isinstance(current, auth_errors.GoogleAuthError):
            detail = vertex_error_text(current)
            separator = " " if detail.endswith(".") else ". "
            message = VertexRuntimeMessage(
                f"{VERTEX_RUNTIME_ERROR_PREFIX}{detail}{separator}"
                f"{_VERTEX_RUNTIME_ADVICE}"
            )
            message.error_detail = detail
            return message
        current = current.__cause__ or current.__context__
    return None
