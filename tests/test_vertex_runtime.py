"""Tests for Vertex AI runtime entities and Repairs issues."""

import json
from pathlib import Path
from types import SimpleNamespace

from google.genai.errors import APIError

from gemini_live import conversation, stt
from gemini_live.const import (
    CONF_API_KEY,
    CONF_PROVIDER,
    CONF_VERTEX_CREDENTIALS,
    CONF_VERTEX_LOCATION,
    PROVIDER_VERTEX,
)
from gemini_live.vertex import VERTEX_RUNTIME_ERROR_PREFIX

COMPONENT = Path(__file__).parent.parent / "custom_components" / "gemini_live"
KEY = {"type": "service_account", "project_id": "p"}


def _entry(**data):
    return SimpleNamespace(entry_id="e1", title="Vertex", data=data, options={})


def _vertex_entry(**extra):
    return _entry(
        **{CONF_PROVIDER: PROVIDER_VERTEX, CONF_VERTEX_CREDENTIALS: KEY, **extra}
    )


async def _setup(module, entry):
    added = []
    await module.async_setup_entry(
        SimpleNamespace(), entry, lambda entities: added.extend(entities)
    )
    return added


async def test_vertex_provider_dispatches_to_vertex_entities():
    entry = _vertex_entry()
    (stt_entity,) = await _setup(stt, entry)
    (agent,) = await _setup(conversation, entry)
    assert type(stt_entity) is stt.VertexGeminiSTT
    assert type(agent) is conversation.VertexGeminiConversationAgent
    assert agent.error_response == (
        "Sorry, I could not get a response from Gemini Live (Vertex AI)."
    )


def test_vertex_credentials_come_from_stored_key():
    config = {CONF_VERTEX_CREDENTIALS: KEY, CONF_API_KEY: "k"}
    assert stt.VertexGeminiSTT._credentials(None, config) is KEY
    assert stt.GeminiLiveSTT._credentials(None, config) == "k"
    assert conversation.VertexGeminiConversationAgent._credentials(None, config) is KEY
    assert conversation.GeminiLiveConversationAgent._credentials(None, config) == "k"


async def _capture(monkeypatch, module, entity_class, entry):
    captured = {}

    async def fake(hass, credentials, location, affective_dialog=False):
        captured.update(
            credentials=credentials,
            location=location,
            affective_dialog=affective_dialog,
        )
        return object()

    monkeypatch.setattr(module, "async_create_vertex_gemini_client", fake)
    hass = SimpleNamespace()
    entity = (
        entity_class(entry)
        if module is stt
        else entity_class(hass, entry)
    )
    entity.hass = hass
    await entity._async_create_client(KEY)
    return captured


async def test_vertex_stt_client_uses_entry_location(monkeypatch):
    entry = _vertex_entry(**{CONF_VERTEX_LOCATION: "eu"})
    captured = await _capture(monkeypatch, stt, stt.VertexGeminiSTT, entry)
    assert captured["location"] == "eu"
    assert captured["credentials"] is KEY


async def test_vertex_client_defaults_location_to_us(monkeypatch):
    entry = _vertex_entry()
    captured = await _capture(monkeypatch, stt, stt.VertexGeminiSTT, entry)
    assert captured["location"] == "us"
    captured = await _capture(
        monkeypatch,
        conversation,
        conversation.VertexGeminiConversationAgent,
        entry,
    )
    assert captured["location"] == "us"


def test_vertex_runtime_errors_never_use_ai_studio_text():
    exc = APIError(429, {"error": {"message": "you exceeded your current quota"}})
    for cls in (stt.VertexGeminiSTT, conversation.VertexGeminiConversationAgent):
        message = cls._api_error_message(exc)
        assert message.startswith(VERTEX_RUNTIME_ERROR_PREFIX)
        for banned in ("ai.dev", "ai.studio", "prepayment", "Google AI project"):
            assert banned not in message


def test_vertex_error_creates_vertex_repairs_issue(monkeypatch):
    created = []
    monkeypatch.setattr(
        "gemini_live.stt.async_create_issue",
        lambda *args, **kwargs: created.append((args, kwargs)),
    )
    message = f"{VERTEX_RUNTIME_ERROR_PREFIX}Permission denied"
    stt._report_api_error(SimpleNamespace(), _vertex_entry(), "gemini_live", message)
    args, kwargs = created[0]
    assert args[2] == "vertex_api_error_e1"
    assert kwargs["translation_key"] == "vertex_api_error"
    assert kwargs["learn_more_url"] == "https://console.cloud.google.com/vertex-ai"
    assert kwargs["translation_placeholders"] == {
        "entry_title": "Vertex",
        "error": "Permission denied",
    }


def test_clear_api_issues_includes_vertex_issue(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "gemini_live.stt.async_delete_issue", lambda *args: cleared.append(args[2])
    )
    stt._clear_api_issues(SimpleNamespace(), _vertex_entry(), "gemini_live")
    assert "vertex_api_error_e1" in cleared


def test_gemini_conversation_error_behaviour_unchanged():
    exc = APIError(1011, "You exceeded your current quota")
    agent_class = conversation.GeminiLiveConversationAgent
    assert agent_class._api_error_message(exc) == stt._QUOTA_USER_MESSAGE


def test_vertex_issue_strings_present_and_files_identical():
    strings = (COMPONENT / "strings.json").read_bytes()
    assert strings == (COMPONENT / "translations" / "en.json").read_bytes()
    issue = json.loads(strings)["issues"]["vertex_api_error"]
    assert issue["title"] == "Gemini Live (Vertex AI) request failed"
    assert "{entry_title}" in issue["description"]
    assert "{error}" in issue["description"]


def test_vertex_conversation_event_matches_gemini():
    assert (
        conversation.VertexGeminiConversationAgent.conversation_event
        == conversation.GeminiLiveConversationAgent.conversation_event
    )


def _refresh_error():
    from google.auth.exceptions import RefreshError  # noqa: PLC0415

    return RefreshError(
        "invalid_grant: Invalid JWT Signature.",
        {"error": "invalid_grant", "error_description": "Invalid JWT Signature."},
    )


async def test_vertex_voice_turn_reports_revoked_key(monkeypatch, caplog):
    from homeassistant.components.stt import SpeechResultState  # noqa: PLC0415
    from test_stt_barge_in import (  # noqa: PLC0415
        MIC_CHUNK,
        FakeHass,
        MicStream,
        _make_entity,
        _metadata,
    )

    hass = FakeHass()
    entity, manager, turns = _make_entity(
        hass,
        {CONF_PROVIDER: PROVIDER_VERTEX, CONF_VERTEX_CREDENTIALS: KEY},
        stt.VertexGeminiSTT,
    )
    issues = []
    monkeypatch.setattr(
        "gemini_live.stt.async_create_issue",
        lambda *args, **kwargs: issues.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "gemini_live.stt.active_pipeline_context",
        lambda *_: ("conversation-1", None, None),
    )

    async def failing_client(*_args, **_kwargs):
        raise _refresh_error()

    monkeypatch.setattr(stt, "async_create_vertex_gemini_client", failing_client)
    mic = MicStream()
    mic.put(MIC_CHUNK)
    mic.close()
    result = await entity._async_process_audio_stream_sdk(
        _metadata(), mic.chunks(), KEY, "m", "v", "", True, False, False
    )

    assert result.result == SpeechResultState.SUCCESS
    assert result.text.startswith(VERTEX_RUNTIME_ERROR_PREFIX)
    assert "invalid_grant" in result.text
    args, kwargs = issues[0]
    assert args[2] == "vertex_api_error_test_entry"
    assert kwargs["translation_key"] == "vertex_api_error"
    assert turns.take_voice_turn("conversation-1", result.text).complete_conversation
    assert not manager.should_continue_conversation("conversation-1")
    assert not any(record.exc_info for record in caplog.records)


async def test_vertex_typed_turn_reports_revoked_key(monkeypatch):
    from test_stt_barge_in import FakeHass, _make_entity  # noqa: PLC0415

    hass = FakeHass()
    stt_entity, _manager, _ = _make_entity(
        hass,
        {
            CONF_PROVIDER: PROVIDER_VERTEX,
            CONF_VERTEX_CREDENTIALS: KEY,
            "model": "m",
            "voice": "v",
        },
        stt.VertexGeminiSTT,
    )
    agent = conversation.VertexGeminiConversationAgent(hass, stt_entity.entry)
    issues = []
    monkeypatch.setattr(
        "gemini_live.stt.async_create_issue",
        lambda *args, **kwargs: issues.append(kwargs),
    )

    async def failing_client(*_args, **_kwargs):
        raise _refresh_error()

    async def llm_api(_context):
        return None, [], "instruction"

    monkeypatch.setattr(
        conversation, "async_create_vertex_gemini_client", failing_client
    )
    agent._async_get_llm_api = llm_api
    result = await agent._async_process_text_live(
        "hello", SimpleNamespace(as_llm_context=lambda _: None), "conversation-1"
    )

    assert result.startswith(VERTEX_RUNTIME_ERROR_PREFIX)
    assert "invalid_grant" in result
    assert issues[0]["translation_key"] == "vertex_api_error"
    assert issues[0]["translation_placeholders"]["error"].startswith("('invalid_grant")


def test_vertex_repairs_issue_shows_only_google_text(monkeypatch):
    created = []
    monkeypatch.setattr(
        "gemini_live.stt.async_create_issue",
        lambda *args, **kwargs: created.append(kwargs),
    )
    from google.auth.exceptions import RefreshError  # noqa: PLC0415

    for exc in (
        APIError(403, {"error": {"message": "Permission denied."}}),
        RefreshError("Token request failed."),
    ):
        created.clear()
        message = stt.VertexGeminiSTT._api_error_message(exc)
        stt._report_api_error(
            SimpleNamespace(), _vertex_entry(), "gemini_live", message
        )
        error = created[0]["translation_placeholders"]["error"]
        assert error == str(exc)
        assert "Reconfigure" not in error
        assert ".." not in error


async def test_vertex_tts_entity_uses_vertex_name():
    from gemini_live import tts  # noqa: PLC0415

    (entity,) = await _setup(tts, _vertex_entry())
    assert type(entity) is tts.VertexGeminiTTS
    assert entity.name == "Gemini Live (Vertex AI)"
    (gemini,) = await _setup(tts, _entry())
    assert type(gemini) is tts.GeminiLiveTTS
