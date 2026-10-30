"""Tests for the config flow schema's barge-in and affective dialog settings."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import voluptuous as vol
from homeassistant.const import CONF_LLM_HASS_API
from gemini_live import config_flow
from gemini_live.config_flow import (
    PROVIDER_SELECTOR,
    GeminiLiveConfigFlow,
    GeminiLiveOptionsFlowHandler,
    _needs_model_specific_refresh,
    _provider_schema,
    _strip_unsupported_settings,
)
from gemini_live.const import (
    CONF_AFFECTIVE_DIALOG,
    CONF_API_KEY,
    CONF_MODEL,
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARCH_GROUNDING,
    CONF_ENABLE_SOUND_RESPONSES,
    CONF_SUPPORT_BARGE_IN,
    CONF_THINKING_LEVEL,
    CONF_PROVIDER,
    CONF_VERTEX_CREDENTIALS,
    CONF_VERTEX_KEY_FILE,
    CONF_VERTEX_LOCATION,
    CONF_VOICE,
    DEFAULT_AFFECTIVE_DIALOG,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SEARCH_GROUNDING,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_SUPPORT_BARGE_IN,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_PERSONAPLEX,
    PROVIDER_VERTEX,
    VERTEX_AVAILABLE_MODELS,
    VERTEX_DEFAULT_LOCATION,
)

_VALID_VOICE = {PROVIDER_GEMINI: "Puck", PROVIDER_OPENAI: "marin"}


def _validator(schema: vol.Schema, key: str) -> Any:
    return next(
        value for marker, value in schema.schema.items() if marker.schema == key
    )


def _available_models(schema: vol.Schema) -> list[str]:
    model_validator = _validator(schema, CONF_MODEL)
    if hasattr(model_validator, "container"):
        return list(model_validator.container)
    return list(model_validator.config["options"])


def test_gemini_models_use_dropdown_and_include_extended_thinking() -> None:
    schema = _provider_schema(PROVIDER_GEMINI)
    model_selector = _validator(schema, CONF_MODEL)

    assert model_selector.config["mode"] == "dropdown"
    assert "gemini-3.8-live-extended-thinking" in _available_models(schema)


def test_search_grounding_is_a_gemini_preference() -> None:
    gemini_keys = [marker.schema for marker in _provider_schema(PROVIDER_GEMINI).schema]
    openai_keys = [marker.schema for marker in _provider_schema(PROVIDER_OPENAI).schema]

    assert CONF_SEARCH_GROUNDING in gemini_keys
    assert CONF_SEARCH_GROUNDING not in openai_keys

    result = _provider_schema(PROVIDER_GEMINI)(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )
    assert result[CONF_SEARCH_GROUNDING] is DEFAULT_SEARCH_GROUNDING
    assert DEFAULT_SEARCH_GROUNDING is False


def test_response_timeout_defaults_and_exposes_range() -> None:
    schema = _provider_schema(PROVIDER_GEMINI)
    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_RESPONSE_TIMEOUT] == DEFAULT_RESPONSE_TIMEOUT
    timeout_selector = _validator(schema, CONF_RESPONSE_TIMEOUT)
    assert timeout_selector.config["min"] == 15
    assert timeout_selector.config["max"] == 120


def test_sound_responses_are_opt_in_and_preserved_in_options() -> None:
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        input_data = {
            CONF_API_KEY: "key",
            CONF_MODEL: _available_models(_provider_schema(provider))[0],
            CONF_VOICE: _VALID_VOICE[provider],
        }
        assert (
            _provider_schema(provider)(input_data)[CONF_ENABLE_SOUND_RESPONSES] is False
        )
        assert (
            _provider_schema(provider, {CONF_ENABLE_SOUND_RESPONSES: True})(input_data)[
                CONF_ENABLE_SOUND_RESPONSES
            ]
            is True
        )
    assert CONF_ENABLE_SOUND_RESPONSES not in [
        marker.schema for marker in _provider_schema(PROVIDER_PERSONAPLEX).schema
    ]


def test_llm_api_selector_lists_registered_apis(monkeypatch) -> None:
    """Offer every registered API, including third-party memory APIs."""
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [
            SimpleNamespace(id="assist", name="Assist"),
            SimpleNamespace(id="memory", name="Memory Management"),
        ],
    )

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_LLM_HASS_API: ["assist", "memory"]},
        SimpleNamespace(),
    )
    api_selector = _validator(schema, CONF_LLM_HASS_API)

    assert api_selector.config["multiple"] is True
    assert api_selector.config["options"] == [
        {"value": "assist", "label": "Assist"},
        {"value": "memory", "label": "Memory Management"},
    ]
    marker = next(
        marker for marker in schema.schema if marker.schema == CONF_LLM_HASS_API
    )
    assert marker.description["suggested_value"] == ["assist", "memory"]


def test_llm_api_selector_defaults_to_assist_and_ignores_missing_apis(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    schema = _provider_schema(PROVIDER_OPENAI, hass=SimpleNamespace())
    marker = next(
        marker for marker in schema.schema if marker.schema == CONF_LLM_HASS_API
    )
    assert marker.description["suggested_value"] == ["assist"]
    assert marker.default() == ["assist"]

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gpt-realtime-2.1",
            CONF_VOICE: _VALID_VOICE[PROVIDER_OPENAI],
        }
    )
    assert result[CONF_LLM_HASS_API] == ["assist"]

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_LLM_HASS_API: ["removed-api"]},
        SimpleNamespace(),
    )
    marker = next(
        marker for marker in schema.schema if marker.schema == CONF_LLM_HASS_API
    )
    assert marker.description["suggested_value"] == []


def test_llm_api_selector_preserves_explicit_empty_selection(monkeypatch) -> None:
    """Do not re-enable Assist after a user explicitly disables all APIs."""
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_LLM_HASS_API: []},
        SimpleNamespace(),
    )
    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_LLM_HASS_API] == []


def test_llm_api_selector_hidden_for_provider_without_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    keys = [
        marker.schema
        for marker in _provider_schema(
            PROVIDER_PERSONAPLEX, hass=SimpleNamespace()
        ).schema
    ]
    assert CONF_LLM_HASS_API not in keys


def test_show_text_setting_is_removed() -> None:
    """Do not expose the former integration-owned display callback setting."""
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI, PROVIDER_PERSONAPLEX):
        keys = [marker.schema for marker in _provider_schema(provider).schema]
        assert "show_text" not in keys


def test_thinking_level_only_shown_for_extended_thinking() -> None:
    for model, expected in (
        ("gemini-3.8-live-extended-thinking", True),
        ("gemini-3.8-live", False),
        ("gemini-3.1-flash-live-preview", False),
    ):
        schema = _provider_schema(PROVIDER_GEMINI, {CONF_MODEL: model})
        keys = [marker.schema for marker in schema.schema]
        assert (CONF_THINKING_LEVEL in keys) is expected

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_MODEL: "gemini-3.8-live-extended-thinking"},
    )
    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live-extended-thinking",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )
    assert result[CONF_THINKING_LEVEL] == DEFAULT_THINKING_LEVEL


def test_model_change_refreshes_before_model_specific_settings() -> None:
    extended = {CONF_MODEL: "gemini-3.8-live-extended-thinking"}
    assert _needs_model_specific_refresh(extended)

    extended[CONF_THINKING_LEVEL] = "medium"
    assert not _needs_model_specific_refresh(extended)

    native_audio = {CONF_MODEL: "gemini-2.5-flash-native-audio-preview-12-2025"}
    assert _needs_model_specific_refresh(native_audio)

    native_audio[CONF_AFFECTIVE_DIALOG] = True
    assert not _needs_model_specific_refresh(native_audio)


def test_unsupported_model_settings_are_removed() -> None:
    result = _strip_unsupported_settings(
        {
            CONF_MODEL: "gemini-3.1-flash-live-preview",
            CONF_AFFECTIVE_DIALOG: True,
            CONF_THINKING_LEVEL: "high",
        }
    )

    assert CONF_AFFECTIVE_DIALOG not in result
    assert CONF_THINKING_LEVEL not in result


def test_schema_shows_barge_in_for_both_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: True
    )
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)
        keys = [marker.schema for marker in schema.schema]
        assert CONF_SUPPORT_BARGE_IN in keys


def test_barge_in_hidden_when_core_lacks_interruption_support(
    monkeypatch,
) -> None:
    """Hide the barge-in option entirely when Core cannot interrupt TTS."""
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: False
    )

    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)
        keys = [marker.schema for marker in schema.schema]
        assert CONF_SUPPORT_BARGE_IN not in keys

        result = schema(
            {
                CONF_API_KEY: "key",
                CONF_MODEL: _available_models(schema)[0],
                CONF_VOICE: _VALID_VOICE[provider],
            }
        )
        assert CONF_SUPPORT_BARGE_IN not in result


def test_barge_in_setting_stripped_without_core_interruption_support(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: False
    )

    result = _strip_unsupported_settings(
        {
            CONF_MODEL: "gemini-3.8-live",
            CONF_SUPPORT_BARGE_IN: True,
        }
    )

    assert CONF_SUPPORT_BARGE_IN not in result


def test_barge_in_defaults_to_false(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: True
    )
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)

        result = schema(
            {
                CONF_API_KEY: "key",
                CONF_MODEL: _available_models(schema)[0],
                CONF_VOICE: _VALID_VOICE[provider],
            }
        )

        assert result[CONF_SUPPORT_BARGE_IN] is DEFAULT_SUPPORT_BARGE_IN
        assert DEFAULT_SUPPORT_BARGE_IN is False


def test_barge_in_setting_persists_through_reconfigure(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: True
    )
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider, {CONF_SUPPORT_BARGE_IN: True})

        result = schema(
            {
                CONF_API_KEY: "key",
                CONF_MODEL: _available_models(schema)[0],
                CONF_VOICE: _VALID_VOICE[provider],
            }
        )

        assert result[CONF_SUPPORT_BARGE_IN] is True


def test_affective_dialog_only_shown_for_supported_models() -> None:
    for config, expected in (
        ({CONF_MODEL: "gemini-2.5-flash-native-audio-preview-12-2025"}, True),
        ({CONF_MODEL: "gemini-3.8-live"}, False),
        ({CONF_MODEL: "gemini-3.8-live-extended-thinking"}, False),
        ({CONF_MODEL: "gemini-3.1-flash-live-preview"}, False),
        ({}, False),
    ):
        schema = _provider_schema(PROVIDER_GEMINI, config)
        keys = [marker.schema for marker in schema.schema]
        assert (CONF_AFFECTIVE_DIALOG in keys) is expected

    schema = _provider_schema(PROVIDER_OPENAI, {CONF_MODEL: "gpt-realtime-2.1"})
    keys = [marker.schema for marker in schema.schema]
    assert CONF_AFFECTIVE_DIALOG not in keys


def test_affective_dialog_defaults_to_false() -> None:
    model = "gemini-2.5-flash-native-audio-preview-12-2025"
    schema = _provider_schema(PROVIDER_GEMINI, {CONF_MODEL: model})

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: model,
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_AFFECTIVE_DIALOG] is DEFAULT_AFFECTIVE_DIALOG
    assert DEFAULT_AFFECTIVE_DIALOG is False


def test_affective_dialog_setting_persists_through_reconfigure() -> None:
    model = "gemini-2.5-flash-native-audio-preview-12-2025"
    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_MODEL: model, CONF_AFFECTIVE_DIALOG: True},
    )

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: model,
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_AFFECTIVE_DIALOG] is True


# --- Google Vertex AI -------------------------------------------------------

_COMPONENT_DIR = Path(config_flow.__file__).resolve().parent
_KEY = {
    "type": "service_account",
    "project_id": "my-project",
    "client_email": "ha@my-project.iam.gserviceaccount.com",
    "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
    "token_uri": "https://oauth2.googleapis.com/token",
}
_NEW_KEY = {**_KEY, "client_email": "new@my-project.iam.gserviceaccount.com"}


# Home Assistant swaps voluptuous for probatio once it is imported, so marker
# classes are taken from the module that built the schema.
def _marker(schema: vol.Schema, key: str) -> Any:
    return next(marker for marker in schema.schema if marker.schema == key)


def _upload_required(schema: vol.Schema) -> bool:
    marker = _marker(schema, CONF_VERTEX_KEY_FILE)
    assert isinstance(marker, (config_flow.vol.Required, config_flow.vol.Optional))
    return isinstance(marker, config_flow.vol.Required)


class _FakeEntries:
    def __init__(self, entry: Any) -> None:
        self._entry = entry

    def async_get_known_entry(self, entry_id: str) -> Any:
        assert entry_id == self._entry.entry_id
        return self._entry


class _FakeHass:
    def __init__(self, entry: Any = None) -> None:
        self.config_entries = _FakeEntries(entry)

    async def async_add_executor_job(self, func, *args):
        return func(*args)


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    """Serve fake uploads by file id and record each read."""
    files: dict[str, bytes] = {}
    opened: list[str] = []

    @contextmanager
    def _process_uploaded_file(_hass, file_id):
        opened.append(file_id)
        path = tmp_path / file_id
        path.write_bytes(files[file_id])
        yield path

    monkeypatch.setattr(config_flow, "process_uploaded_file", _process_uploaded_file)
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )
    return SimpleNamespace(files=files, opened=opened)


@pytest.fixture
def validate(monkeypatch):
    """Record connection checks and return a configurable result."""
    state = SimpleNamespace(calls=[], result=None)

    async def _validate(_hass, credentials, location, model):
        state.calls.append((credentials, location, model))
        return state.result

    monkeypatch.setattr(config_flow, "async_validate_vertex_connection", _validate)
    return state


def _vertex_input(**extra: Any) -> dict[str, Any]:
    return {
        CONF_VERTEX_LOCATION: "us",
        CONF_MODEL: "gemini-3.8-live",
        CONF_VOICE: "Puck",
        **extra,
    }


def _setup_flow() -> GeminiLiveConfigFlow:
    flow = GeminiLiveConfigFlow()
    flow.hass = _FakeHass()
    flow.context = {"source": "user"}
    return flow


def _vertex_entry() -> SimpleNamespace:
    return SimpleNamespace(
        entry_id="entry-1",
        data={
            CONF_PROVIDER: PROVIDER_VERTEX,
            CONF_VERTEX_CREDENTIALS: dict(_KEY),
            CONF_VERTEX_LOCATION: "us",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: "Puck",
        },
        options={},
    )


def _reconfigure_flow(entry: Any) -> tuple[GeminiLiveConfigFlow, dict[str, Any]]:
    flow = GeminiLiveConfigFlow()
    flow.hass = _FakeHass(entry)
    flow.context = {"source": "reconfigure", "entry_id": entry.entry_id}
    saved: dict[str, Any] = {}

    def _update(entry_arg, **kwargs):
        saved.update(kwargs, entry=entry_arg)
        return {"type": "abort", "reason": "reconfigure_successful"}

    flow.async_update_reload_and_abort = _update
    return flow, saved


def test_vertex_is_offered_as_a_provider() -> None:
    options = PROVIDER_SELECTOR.config["options"]
    values = [option["value"] for option in options]

    assert {"value": PROVIDER_VERTEX, "label": "Google Vertex AI"} in options
    assert values.index(PROVIDER_VERTEX) == values.index(PROVIDER_GEMINI) + 1


def test_vertex_schema_has_upload_location_and_no_api_key() -> None:
    schema = _provider_schema(PROVIDER_VERTEX)
    keys = [marker.schema for marker in schema.schema]

    assert CONF_API_KEY not in keys
    assert _upload_required(schema)
    assert _validator(schema, CONF_VERTEX_KEY_FILE).config["accept"] == (
        ".json,application/json"
    )
    location = _marker(schema, CONF_VERTEX_LOCATION)
    assert isinstance(location, config_flow.vol.Required)
    assert location.default() == VERTEX_DEFAULT_LOCATION == "us"
    assert _validator(schema, CONF_VERTEX_LOCATION).config["custom_value"] is True
    assert _available_models(schema) == VERTEX_AVAILABLE_MODELS
    assert _validator(schema, CONF_MODEL).config["custom_value"] is True
    assert _marker(schema, CONF_MODEL).default() == "gemini-3.8-live"
    assert CONF_SEARCH_GROUNDING in keys


def test_vertex_upload_optional_when_key_stored() -> None:
    schema = _provider_schema(
        PROVIDER_VERTEX,
        {CONF_VERTEX_CREDENTIALS: dict(_KEY), CONF_VERTEX_LOCATION: "eu"},
    )

    assert not _upload_required(schema)
    assert _marker(schema, CONF_VERTEX_LOCATION).default() == "eu"


async def test_new_vertex_entry_stores_parsed_key(uploads, validate) -> None:
    uploads.files["upload-1"] = json.dumps(_KEY).encode()
    flow = _setup_flow()

    form = await flow.async_step_user({CONF_PROVIDER: PROVIDER_VERTEX})
    assert form["type"] == "form"
    assert form["description_placeholders"]["provider"] == "Google Vertex AI"

    result = await flow.async_step_provider(
        _vertex_input(
            **{CONF_VERTEX_KEY_FILE: "upload-1", CONF_VERTEX_LOCATION: " us "}
        )
    )

    assert result["type"] == "create_entry"
    assert result["title"] == "Gemini Live (Vertex AI)"
    data = result["data"]
    assert data[CONF_VERTEX_CREDENTIALS] == _KEY
    assert data[CONF_VERTEX_LOCATION] == "us"
    assert data[CONF_MODEL] == "gemini-3.8-live"
    assert data[CONF_PROVIDER] == PROVIDER_VERTEX
    assert CONF_VERTEX_KEY_FILE not in data
    assert uploads.opened == ["upload-1"]
    assert validate.calls == [(_KEY, "us", "gemini-3.8-live")]


async def test_bad_key_file_shows_form_error(uploads, validate) -> None:
    uploads.files["upload-1"] = b"nope"
    flow = _setup_flow()
    await flow.async_step_user({CONF_PROVIDER: PROVIDER_VERTEX})

    result = await flow.async_step_provider(
        _vertex_input(**{CONF_VERTEX_KEY_FILE: "upload-1"})
    )

    assert result["type"] == "form"
    assert result["errors"] == {"base": "invalid_key_file"}
    assert _upload_required(result["data_schema"])
    assert validate.calls == []


async def test_connection_failure_passes_google_error_through(
    uploads, validate
) -> None:
    uploads.files["upload-1"] = json.dumps(_KEY).encode()
    validate.result = "403 PERMISSION_DENIED"
    flow = _setup_flow()
    await flow.async_step_user({CONF_PROVIDER: PROVIDER_VERTEX})

    result = await flow.async_step_provider(
        _vertex_input(**{CONF_VERTEX_KEY_FILE: "upload-1"})
    )

    assert result["type"] == "form"
    assert result["errors"] == {"base": "cannot_connect"}
    assert result["description_placeholders"]["error"] == "403 PERMISSION_DENIED"
    assert result["description_placeholders"]["provider"] == "Google Vertex AI"
    assert not _upload_required(result["data_schema"])


async def test_retry_after_connection_error_does_not_require_reupload(
    uploads, validate
) -> None:
    uploads.files["upload-1"] = json.dumps(_KEY).encode()
    validate.result = "403 PERMISSION_DENIED"
    flow = _setup_flow()
    await flow.async_step_user({CONF_PROVIDER: PROVIDER_VERTEX})
    await flow.async_step_provider(_vertex_input(**{CONF_VERTEX_KEY_FILE: "upload-1"}))

    validate.result = None
    result = await flow.async_step_provider(_vertex_input())

    assert result["type"] == "create_entry"
    assert result["data"][CONF_VERTEX_CREDENTIALS] == _KEY
    assert uploads.opened == ["upload-1"]
    assert validate.calls[-1] == (_KEY, "us", "gemini-3.8-live")


async def test_reconfigure_without_upload_keeps_credentials(uploads, validate) -> None:
    flow, saved = _reconfigure_flow(_vertex_entry())

    form = await flow.async_step_reconfigure()
    assert not _upload_required(form["data_schema"])

    result = await flow.async_step_reconfigure(_vertex_input())

    assert result["reason"] == "reconfigure_successful"
    assert saved["options"] == {}
    assert saved["data_updates"][CONF_VERTEX_CREDENTIALS] == _KEY
    assert saved["data_updates"][CONF_VERTEX_LOCATION] == "us"
    assert CONF_VERTEX_KEY_FILE not in saved["data_updates"]
    assert validate.calls == []
    assert uploads.opened == []


async def test_reconfigure_location_change_revalidates(uploads, validate) -> None:
    flow, saved = _reconfigure_flow(_vertex_entry())

    result = await flow.async_step_reconfigure(
        _vertex_input(**{CONF_VERTEX_LOCATION: "eu"})
    )

    assert result["reason"] == "reconfigure_successful"
    assert validate.calls == [(_KEY, "eu", "gemini-3.8-live")]
    assert saved["data_updates"][CONF_VERTEX_LOCATION] == "eu"


async def test_reconfigure_connection_error_shows_form(uploads, validate) -> None:
    validate.result = "404 NOT_FOUND"
    flow, saved = _reconfigure_flow(_vertex_entry())

    result = await flow.async_step_reconfigure(
        _vertex_input(**{CONF_VERTEX_LOCATION: "eu"})
    )

    assert result["type"] == "form"
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": "cannot_connect"}
    assert result["description_placeholders"]["error"] == "404 NOT_FOUND"
    assert not _upload_required(result["data_schema"])
    assert saved == {}


async def test_reconfigure_retry_keeps_newly_uploaded_key(uploads, validate) -> None:
    uploads.files["upload-2"] = json.dumps(_NEW_KEY).encode()
    validate.result = "403 PERMISSION_DENIED"
    flow, saved = _reconfigure_flow(_vertex_entry())
    await flow.async_step_reconfigure(
        _vertex_input(**{CONF_VERTEX_KEY_FILE: "upload-2"})
    )

    validate.result = None
    result = await flow.async_step_reconfigure(_vertex_input())

    assert result["reason"] == "reconfigure_successful"
    assert saved["data_updates"][CONF_VERTEX_CREDENTIALS] == _NEW_KEY
    assert validate.calls[-1] == (_NEW_KEY, "us", "gemini-3.8-live")


async def test_options_flow_new_upload_replaces_key(uploads, validate) -> None:
    uploads.files["upload-2"] = json.dumps(_NEW_KEY).encode()
    entry = _vertex_entry()
    flow = GeminiLiveOptionsFlowHandler()
    flow.hass = _FakeHass(entry)
    flow.handler = entry.entry_id

    result = await flow.async_step_init(
        _vertex_input(**{CONF_VERTEX_KEY_FILE: "upload-2"})
    )

    assert result["type"] == "create_entry"
    assert result["data"][CONF_VERTEX_CREDENTIALS] == _NEW_KEY
    assert CONF_VERTEX_KEY_FILE not in result["data"]
    assert validate.calls == [(_NEW_KEY, "us", "gemini-3.8-live")]


async def test_options_flow_without_upload_keeps_key(uploads, validate) -> None:
    entry = _vertex_entry()
    flow = GeminiLiveOptionsFlowHandler()
    flow.hass = _FakeHass(entry)
    flow.handler = entry.entry_id

    result = await flow.async_step_init(_vertex_input())

    assert result["type"] == "create_entry"
    assert result["data"][CONF_VERTEX_CREDENTIALS] == _KEY
    assert validate.calls == []


async def test_options_flow_bad_key_shows_error(uploads, validate) -> None:
    uploads.files["upload-3"] = json.dumps({"type": "authorized_user"}).encode()
    entry = _vertex_entry()
    flow = GeminiLiveOptionsFlowHandler()
    flow.hass = _FakeHass(entry)
    flow.handler = entry.entry_id

    result = await flow.async_step_init(
        _vertex_input(**{CONF_VERTEX_KEY_FILE: "upload-3"})
    )

    assert result["type"] == "form"
    assert result["errors"] == {"base": "not_service_account"}
    assert validate.calls == []


def test_gemini_schema_unchanged() -> None:
    keys = [marker.schema for marker in _provider_schema(PROVIDER_GEMINI).schema]

    assert CONF_API_KEY in keys
    assert CONF_VERTEX_KEY_FILE not in keys
    assert CONF_VERTEX_LOCATION not in keys
    assert not _validator(_provider_schema(PROVIDER_GEMINI), CONF_MODEL).config.get(
        "custom_value"
    )


def test_vertex_strings_present_and_files_identical() -> None:
    strings_raw = (_COMPONENT_DIR / "strings.json").read_bytes()
    assert strings_raw == (_COMPONENT_DIR / "translations" / "en.json").read_bytes()
    strings = json.loads(strings_raw)

    assert "Google Vertex AI" in strings["config"]["step"]["user"]["description"]
    for step in (
        strings["config"]["step"]["provider"],
        strings["config"]["step"]["reconfigure"],
        strings["options"]["step"]["init"],
    ):
        assert step["data"][CONF_VERTEX_KEY_FILE] == "Service account key (JSON)"
        assert step["data"][CONF_VERTEX_LOCATION] == "Location"
        assert "leave empty to keep the current key" in (
            step["data_description"][CONF_VERTEX_KEY_FILE]
        )
        assert CONF_VERTEX_LOCATION in step["data_description"]
    for errors in (strings["config"]["error"], strings["options"]["error"]):
        assert set(errors) >= {
            "invalid_key_file",
            "not_service_account",
            "missing_key_fields",
            "cannot_connect",
        }
        assert errors["cannot_connect"] == "Could not connect to Vertex AI: {error}"


def test_manifest_depends_on_file_upload() -> None:
    manifest = json.loads((_COMPONENT_DIR / "manifest.json").read_text())
    assert "file_upload" in manifest["dependencies"]
