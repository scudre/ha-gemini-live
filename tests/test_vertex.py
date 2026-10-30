"""Tests for Vertex AI service-account key parsing."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from gemini_live import vertex
from gemini_live.const import is_google_provider
from gemini_live.gemini import GeminiLiveClient
from gemini_live.vertex import (
    VERTEX_RUNTIME_ERROR_PREFIX,
    VertexKeyError,
    async_create_vertex_gemini_client,
    async_validate_vertex_connection,
    parse_service_account_key,
    vertex_error_text,
    vertex_runtime_error_message,
)

_KEY = {
    "type": "service_account",
    "project_id": "proj",
    "private_key": "-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----\n",
    "client_email": "sa@proj.iam.gserviceaccount.com",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "1",
}


def _without(key: str) -> dict:
    return {k: v for k, v in _KEY.items() if k != key}


_FAILURES = [
    (b"\xff\xfe garbage", "invalid_key_file"),
    (b"not json", "invalid_key_file"),
    (b"[1, 2]", "invalid_key_file"),
    (json.dumps({**_KEY, "type": "authorized_user"}).encode(), "not_service_account"),
    (json.dumps(_without("private_key")).encode(), "missing_key_fields"),
]


def test_parse_accepts_service_account_bytes():
    assert parse_service_account_key(json.dumps(_KEY).encode()) == _KEY


def test_parse_accepts_str():
    assert parse_service_account_key(json.dumps(_KEY)) == _KEY


def test_parse_rejects_non_json_bytes():
    with pytest.raises(VertexKeyError) as err:
        parse_service_account_key(b"\xff\xfe garbage")
    assert err.value.reason == "invalid_key_file"


def test_parse_rejects_json_array():
    with pytest.raises(VertexKeyError) as err:
        parse_service_account_key(b"[1, 2]")
    assert err.value.reason == "invalid_key_file"


def test_parse_rejects_other_credential_type():
    raw = json.dumps({**_KEY, "type": "authorized_user"}).encode()
    with pytest.raises(VertexKeyError) as err:
        parse_service_account_key(raw)
    assert err.value.reason == "not_service_account"


def test_parse_rejects_missing_fields():
    with pytest.raises(VertexKeyError) as err:
        parse_service_account_key(json.dumps(_without("private_key")).encode())
    assert err.value.reason == "missing_key_fields"


@pytest.mark.parametrize(("raw", "reason"), _FAILURES)
def test_key_error_message_never_contains_private_key(raw, reason):
    with pytest.raises(VertexKeyError) as err:
        parse_service_account_key(raw)
    assert err.value.reason == reason
    assert _KEY["private_key"] not in str(err.value)


def test_key_error_hides_key_inside_invalid_json():
    raw = ('{"private_key": "' + _KEY["private_key"] + '" oops').encode()
    with pytest.raises(VertexKeyError) as err:
        parse_service_account_key(raw)
    assert "BEGIN PRIVATE KEY" not in str(err.value)


def test_is_google_provider():
    assert is_google_provider("gemini")
    assert is_google_provider("vertex")
    for other in ("openai", "qwen", "personaplex", None):
        assert not is_google_provider(other)


INFO = {"project_id": "proj", "client_email": "a@b", "private_key": "k"}


class _Hass:
    async def async_add_executor_job(self, func, *args):
        return func(*args)


def _install_fakes(monkeypatch, connect=None):
    captured = {}

    class _FakeHttpOptions:
        def __init__(self, api_version=None):
            self.api_version = api_version
            captured["api_version"] = api_version

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.kwargs = kwargs
            self.aio = SimpleNamespace(live=SimpleNamespace(connect=connect))

    genai = SimpleNamespace(
        types=SimpleNamespace(HttpOptions=_FakeHttpOptions), Client=_FakeClient
    )

    class _Creds:
        @staticmethod
        def from_service_account_info(info, scopes=None):
            captured["creds_info"] = info
            captured["scopes"] = scopes
            return "fake-creds"

    service_account = SimpleNamespace(Credentials=_Creds)
    oauth2 = SimpleNamespace(service_account=service_account)
    monkeypatch.setitem(
        sys.modules, "google", SimpleNamespace(genai=genai, oauth2=oauth2)
    )
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", service_account)
    return captured


async def test_vertex_client_uses_project_from_key_and_location(monkeypatch):
    captured = _install_fakes(monkeypatch)

    client = await async_create_vertex_gemini_client(_Hass(), INFO, "us")

    assert isinstance(client, GeminiLiveClient)
    assert captured["client_kwargs"] == {
        "vertexai": True,
        "project": "proj",
        "location": "us",
        "credentials": "fake-creds",
        "http_options": None,
    }
    assert captured["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]


async def test_vertex_client_affective_dialog_uses_v1beta1(monkeypatch):
    captured = _install_fakes(monkeypatch)

    await async_create_vertex_gemini_client(
        _Hass(), INFO, "us", affective_dialog=True
    )

    assert captured["api_version"] == "v1beta1"
    assert captured["client_kwargs"]["http_options"] is not None


async def test_validate_connection_success_returns_none(monkeypatch):
    events = []

    class _CM:
        async def __aenter__(self):
            events.append("enter")

        async def __aexit__(self, *exc):
            events.append("exit")

    def connect(**kwargs):
        events.append(kwargs)
        return _CM()

    _install_fakes(monkeypatch, connect)

    result = await async_validate_vertex_connection(_Hass(), INFO, "us", "m")

    assert result is None
    assert events[1:] == ["enter", "exit"]
    assert events[0] == {"model": "m", "config": {"response_modalities": ["AUDIO"]}}


async def test_validate_connection_passes_google_message_through(monkeypatch):
    def connect(**kwargs):
        raise Exception("403 PERMISSION_DENIED: nope")

    _install_fakes(monkeypatch, connect)

    result = await async_validate_vertex_connection(_Hass(), INFO, "us", "m")

    assert result == "403 PERMISSION_DENIED: nope"


async def test_validate_connection_times_out(monkeypatch):
    class _CM:
        async def __aenter__(self):
            await asyncio.sleep(5)

        async def __aexit__(self, *exc):
            return False

    _install_fakes(monkeypatch, lambda **kwargs: _CM())
    monkeypatch.setattr(vertex, "VERTEX_CONNECT_TIMEOUT", 0.01)

    result = await async_validate_vertex_connection(_Hass(), INFO, "us", "m")

    assert result == "Timed out after 0.01 seconds connecting to Vertex AI"


def test_error_text_falls_back_to_class_name():
    assert vertex_error_text(RuntimeError()) == "RuntimeError"
    assert vertex_error_text(RuntimeError(" boom ")) == "boom"


def test_runtime_message_wraps_api_error():
    from google.genai import errors  # noqa: PLC0415

    api_error = errors.APIError(403, {"error": {"message": "denied"}})
    try:
        raise RuntimeError("outer") from api_error
    except RuntimeError as exc:
        message = vertex_runtime_error_message(exc)

    assert message is not None
    assert message.startswith(VERTEX_RUNTIME_ERROR_PREFIX)
    assert "Reconfigure" in message
    assert "ai.dev" not in message
    assert "ai.studio" not in message


def test_runtime_message_ignores_normal_close_and_other_errors():
    from google.genai import errors  # noqa: PLC0415

    assert (
        vertex_runtime_error_message(errors.APIError(1000, {"error": {"message": "x"}}))
        is None
    )
    assert vertex_runtime_error_message(ValueError("x")) is None
