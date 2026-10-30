"""Tests for Vertex AI service-account key parsing."""

import json

import pytest

from gemini_live.const import is_google_provider
from gemini_live.vertex import VertexKeyError, parse_service_account_key

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
