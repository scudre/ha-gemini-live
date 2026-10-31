"""Config flow for live voice-model providers."""

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm, selector

from .compat import supports_tts_interruption
from .const import (
    AVAILABLE_MODELS,
    AVAILABLE_VOICES_INFO,
    CONF_AFFECTIVE_DIALOG,
    CONF_API_KEY,
    CONF_DETAILED_LOGGING,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_ENABLE_SOUND_RESPONSES,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARCH_GROUNDING,
    CONF_SUPPORT_BARGE_IN,
    CONF_SYSTEM_INSTRUCTION,
    CONF_THINKING_LEVEL,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    CONF_VOICE,
    DEFAULT_AFFECTIVE_DIALOG,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_ENABLE_SOUND_RESPONSES,
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SEARCH_GROUNDING,
    DEFAULT_SUPPORT_BARGE_IN,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_TRANSCRIBE_GPT,
    DEFAULT_VOICE,
    DOMAIN,
    OPENAI_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_VOICES_INFO,
    OPENAI_DEFAULT_MODEL,
    OPENAI_DEFAULT_VOICE,
    PERSONAPLEX_AVAILABLE_MODELS,
    PERSONAPLEX_AVAILABLE_VOICES_INFO,
    PERSONAPLEX_DEFAULT_MODEL,
    PERSONAPLEX_DEFAULT_VOICE,
    MAX_RESPONSE_TIMEOUT,
    MIN_RESPONSE_TIMEOUT,
    PROVIDER_QWEN,
    QWEN_AVAILABLE_MODELS,
    QWEN_DEFAULT_MODEL,
    CONF_TRANSCRIBE_QWEN,
    QWEN_AVAILABLE_VOICES,
    QWEN_DEFAULT_VOICE,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_PERSONAPLEX,
    PROVIDER_VERTEX,
    THINKING_LEVELS,
    CONF_VERTEX_CREDENTIALS,
    CONF_VERTEX_KEY_FILE,
    CONF_VERTEX_LOCATION,
    VERTEX_AVAILABLE_MODELS,
    VERTEX_DEFAULT_LOCATION,
    VERTEX_DEFAULT_MODEL,
    VERTEX_LOCATIONS,
    is_google_provider,
    supports_affective_dialog,
    supports_thinking_level,
)
from .vertex import (
    VertexKeyError,
    async_validate_vertex_connection,
    parse_service_account_key,
)

VERTEX_PROVIDER_NAME = "Google Vertex AI"

PROVIDER_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=PROVIDER_GEMINI, label="Google Gemini"),
            selector.SelectOptionDict(
                value=PROVIDER_VERTEX, label=VERTEX_PROVIDER_NAME
            ),
            selector.SelectOptionDict(value=PROVIDER_OPENAI, label="OpenAI"),
            selector.SelectOptionDict(value=PROVIDER_QWEN, label="QwenCloud"),
            selector.SelectOptionDict(
                value=PROVIDER_PERSONAPLEX, label="fal.ai PersonaPlex"
            ),
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _model_selector(
    models: list[str], custom_value: bool = False
) -> selector.SelectSelector:
    """Return a compact dropdown for a provider's model choices."""
    config = selector.SelectSelectorConfig(
        options=models,
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
    if custom_value:
        config["custom_value"] = True
    return selector.SelectSelector(config)


GEMINI_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(
                value=name,
                label=f"{name} - {gender}, {description}",
            )
            for name, gender, description in AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

OPENAI_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=name, label=f"{name} - {description}")
            for name, description in OPENAI_AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

PERSONAPLEX_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=name, label=f"{name} - {description}")
            for name, description in PERSONAPLEX_AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _provider(config: dict[str, Any]) -> str:
    """Return the configured provider, defaulting legacy entries to Gemini."""
    return config.get(CONF_PROVIDER, PROVIDER_GEMINI)


def _provider_schema(
    provider: str,
    config: dict[str, Any] | None = None,
    hass: HomeAssistant | None = None,
) -> vol.Schema:
    """Build a provider-specific setup/options schema."""
    current = config or {}
    is_qwen = provider == PROVIDER_QWEN
    is_openai = provider == PROVIDER_OPENAI
    is_personaplex = provider == PROVIDER_PERSONAPLEX
    is_vertex = provider == PROVIDER_VERTEX
    models = (
        QWEN_AVAILABLE_MODELS
        if is_qwen
        else PERSONAPLEX_AVAILABLE_MODELS
        if is_personaplex
        else OPENAI_AVAILABLE_MODELS
        if is_openai
        else AVAILABLE_MODELS
    )
    default_model = (
        QWEN_DEFAULT_MODEL
        if is_qwen
        else PERSONAPLEX_DEFAULT_MODEL
        if is_personaplex
        else OPENAI_DEFAULT_MODEL
        if is_openai
        else DEFAULT_MODEL
    )
    default_voice = (
        PERSONAPLEX_DEFAULT_VOICE
        if is_personaplex
        else OPENAI_DEFAULT_VOICE
        if is_openai
        else DEFAULT_VOICE
    )
    voice_selector = (
        PERSONAPLEX_VOICE_SELECTOR
        if is_personaplex
        else OPENAI_VOICE_SELECTOR
        if is_openai
        else GEMINI_VOICE_SELECTOR
    )
    if is_qwen:
        default_voice = QWEN_DEFAULT_VOICE
        voice_selector = _model_selector(QWEN_AVAILABLE_VOICES)
    transcribe_key = (
        CONF_TRANSCRIBE_QWEN
        if is_qwen
        else CONF_TRANSCRIBE_GPT
        if is_openai
        else CONF_TRANSCRIBE_GEMINI
    )
    default_transcribe = (
        DEFAULT_TRANSCRIBE_GPT if is_openai else DEFAULT_TRANSCRIBE_GEMINI
    )

    fields: dict[vol.Marker, Any] = {}
    if is_vertex:
        # The project comes from the key; a stored key makes the upload optional.
        key_file_marker = (
            vol.Optional if current.get(CONF_VERTEX_CREDENTIALS) else vol.Required
        )
        fields[key_file_marker(CONF_VERTEX_KEY_FILE)] = selector.FileSelector(
            selector.FileSelectorConfig(accept=".json,application/json")
        )
        fields[
            vol.Required(
                CONF_VERTEX_LOCATION,
                default=current.get(CONF_VERTEX_LOCATION, VERTEX_DEFAULT_LOCATION),
            )
        ] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=VERTEX_LOCATIONS,
                mode=selector.SelectSelectorMode.DROPDOWN,
                custom_value=True,
            )
        )
        fields[
            vol.Required(
                CONF_MODEL,
                default=current.get(CONF_MODEL, VERTEX_DEFAULT_MODEL),
            )
        ] = _model_selector(VERTEX_AVAILABLE_MODELS, custom_value=True)
    else:
        api_key_field = (
            vol.Required(CONF_API_KEY, default=current[CONF_API_KEY])
            if CONF_API_KEY in current
            else vol.Required(CONF_API_KEY)
        )
        fields[api_key_field] = str
        fields[
            vol.Required(
                CONF_MODEL,
                default=current.get(CONF_MODEL, default_model),
            )
        ] = _model_selector(models)
    fields |= {
        vol.Required(
            CONF_VOICE,
            default=current.get(CONF_VOICE, default_voice),
        ): voice_selector,
        vol.Optional(
            CONF_SYSTEM_INSTRUCTION,
            description={"suggested_value": current.get(CONF_SYSTEM_INSTRUCTION, "")},
        ): str,
        vol.Optional(
            CONF_DETAILED_LOGGING,
            default=current.get(CONF_DETAILED_LOGGING, False),
        ): selector.BooleanSelector(),
        vol.Optional(
            transcribe_key,
            default=current.get(transcribe_key, default_transcribe),
        ): selector.BooleanSelector(),
        vol.Optional(
            CONF_RESPONSE_TIMEOUT,
            default=current.get(CONF_RESPONSE_TIMEOUT, DEFAULT_RESPONSE_TIMEOUT),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=MIN_RESPONSE_TIMEOUT,
                max=MAX_RESPONSE_TIMEOUT,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="seconds",
            )
        ),
    }
    if not is_personaplex:
        fields[
            vol.Optional(
                CONF_ENABLE_SOUND_RESPONSES,
                default=current.get(
                    CONF_ENABLE_SOUND_RESPONSES, DEFAULT_ENABLE_SOUND_RESPONSES
                ),
            )
        ] = selector.BooleanSelector()
    if hass is not None and not is_personaplex:
        apis = [
            selector.SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(hass)
        ]
        known_api_ids = {api["value"] for api in apis}
        selected_apis = current.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST])
        if isinstance(selected_apis, str):
            selected_apis = [selected_apis]
        selected_apis = [api_id for api_id in selected_apis if api_id in known_api_ids]
        fields[
            vol.Optional(
                CONF_LLM_HASS_API,
                default=selected_apis,
                description={"suggested_value": selected_apis},
            )
        ] = selector.SelectSelector(
            selector.SelectSelectorConfig(options=apis, multiple=True)
        )
    if is_google_provider(provider):
        # All Gemini Live models offered by this integration support Google's
        # server-side Search grounding tool. It is a separate, opt-in
        # preference from the former Assist search-tool prompt hint.
        fields[
            vol.Optional(
                CONF_SEARCH_GROUNDING,
                default=current.get(
                    CONF_SEARCH_GROUNDING,
                    DEFAULT_SEARCH_GROUNDING,
                ),
            )
        ] = selector.BooleanSelector()
    elif is_openai:
        fields[
            vol.Optional(
                CONF_ENCOURAGE_WEB_SEARCH,
                default=current.get(
                    CONF_ENCOURAGE_WEB_SEARCH,
                    DEFAULT_ENCOURAGE_WEB_SEARCH,
                ),
            )
        ] = selector.BooleanSelector()
    if not is_personaplex and supports_tts_interruption():
        # Barge-in needs Core's complete TTS interruption path. When the
        # installed Core cannot propagate interruptions, the setting cannot do
        # anything, so it is hidden entirely and never blocks setup.
        fields[
            vol.Optional(
                CONF_SUPPORT_BARGE_IN,
                default=current.get(
                    CONF_SUPPORT_BARGE_IN,
                    DEFAULT_SUPPORT_BARGE_IN,
                ),
            )
        ] = selector.BooleanSelector()
    # The affective-dialog switch is Gemini-specific and only rendered for
    # models that support it; Home Assistant config forms cannot render a
    # disabled field, so it is hidden instead.
    if (
        is_google_provider(provider)
        and supports_affective_dialog(current.get(CONF_MODEL))
    ):
        fields[
            vol.Optional(
                CONF_AFFECTIVE_DIALOG,
                default=current.get(
                    CONF_AFFECTIVE_DIALOG,
                    DEFAULT_AFFECTIVE_DIALOG,
                ),
            )
        ] = selector.BooleanSelector()
    if (
        is_google_provider(provider)
        and supports_thinking_level(current.get(CONF_MODEL))
    ):
        fields[
            vol.Optional(
                CONF_THINKING_LEVEL,
                default=current.get(
                    CONF_THINKING_LEVEL,
                    DEFAULT_THINKING_LEVEL,
                ),
            )
        ] = _model_selector(THINKING_LEVELS)
    return vol.Schema(fields)


def _strip_unsupported_settings(user_input: dict[str, Any]) -> dict[str, Any]:
    """Drop settings the selected model or Core does not support."""
    if not supports_affective_dialog(user_input.get(CONF_MODEL)):
        user_input.pop(CONF_AFFECTIVE_DIALOG, None)
    if not supports_thinking_level(user_input.get(CONF_MODEL)):
        user_input.pop(CONF_THINKING_LEVEL, None)
    if not supports_tts_interruption():
        user_input.pop(CONF_SUPPORT_BARGE_IN, None)
    if user_input.get(CONF_MODEL) in QWEN_AVAILABLE_MODELS:
        user_input.pop(CONF_SEARCH_GROUNDING, None)
        user_input.pop(CONF_ENCOURAGE_WEB_SEARCH, None)
    return user_input


def _needs_model_specific_refresh(user_input: dict[str, Any]) -> bool:
    """Return whether a newly selected model needs its settings form shown."""
    model = user_input.get(CONF_MODEL)
    return (
        supports_thinking_level(model) and CONF_THINKING_LEVEL not in user_input
    ) or (supports_affective_dialog(model) and CONF_AFFECTIVE_DIALOG not in user_input)


def _read_uploaded_file(hass: HomeAssistant, file_id: str) -> bytes:
    """Read an uploaded file once; Home Assistant deletes it afterwards."""
    with process_uploaded_file(hass, file_id) as path:
        return path.read_bytes()


async def _async_read_vertex_key(
    hass: HomeAssistant,
    user_input: dict[str, Any],
    stored_credentials: dict[str, Any] | None,
) -> str | None:
    """Put the uploaded or stored key into user_input; return an error key."""
    if file_id := user_input.pop(CONF_VERTEX_KEY_FILE, None):
        try:
            raw = await hass.async_add_executor_job(
                _read_uploaded_file, hass, file_id
            )
            user_input[CONF_VERTEX_CREDENTIALS] = parse_service_account_key(raw)
        except VertexKeyError as err:
            return err.reason
        except (OSError, ValueError):
            return "invalid_key_file"
    else:
        user_input[CONF_VERTEX_CREDENTIALS] = stored_credentials
    if CONF_VERTEX_LOCATION in user_input:
        user_input[CONF_VERTEX_LOCATION] = str(user_input[CONF_VERTEX_LOCATION]).strip()
    return None


async def _async_check_vertex_connection(
    hass: HomeAssistant,
    user_input: dict[str, Any],
    current: dict[str, Any] | None,
) -> str | None:
    """Check the Vertex AI connection unless nothing it depends on changed."""
    keys = (CONF_VERTEX_CREDENTIALS, CONF_VERTEX_LOCATION, CONF_MODEL)
    if current is not None and all(
        user_input.get(key) == current.get(key) for key in keys
    ):
        return None
    return await async_validate_vertex_connection(
        hass,
        user_input[CONF_VERTEX_CREDENTIALS],
        user_input[CONF_VERTEX_LOCATION],
        user_input[CONF_MODEL],
    )


def _vertex_form_values(
    user_input: dict[str, Any], credentials: dict[str, Any] | None
) -> dict[str, Any]:
    """Return form values that keep an already provided key."""
    values = {**user_input}
    values.pop(CONF_VERTEX_KEY_FILE, None)
    if credentials:
        values[CONF_VERTEX_CREDENTIALS] = credentials
    return values


class GeminiLiveConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configure a realtime voice provider."""

    VERSION = 1
    _vertex_credentials: dict[str, Any] | None = None

    async def async_step_user(self, user_input=None):
        """Select the live model provider."""
        if user_input is not None:
            return await self.async_step_provider(provider=user_input[CONF_PROVIDER])
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROVIDER,
                        default=PROVIDER_GEMINI,
                    ): PROVIDER_SELECTOR
                }
            ),
        )

    async def async_step_provider(
        self,
        user_input=None,
        *,
        provider: str | None = None,
    ):
        """Collect settings for the selected provider."""
        selected_provider = provider or self.context[CONF_PROVIDER]
        self.context[CONF_PROVIDER] = selected_provider
        is_vertex = selected_provider == PROVIDER_VERTEX
        if user_input is not None:
            if is_vertex:
                if error := await _async_read_vertex_key(
                    self.hass, user_input, self._vertex_credentials
                ):
                    return self.async_show_form(
                        step_id="provider",
                        data_schema=_provider_schema(
                            selected_provider,
                            _vertex_form_values(user_input, self._vertex_credentials),
                            self.hass,
                        ),
                        errors={"base": error},
                        description_placeholders={"provider": VERTEX_PROVIDER_NAME},
                    )
                self._vertex_credentials = user_input[CONF_VERTEX_CREDENTIALS]
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="provider",
                    data_schema=_provider_schema(
                        selected_provider, user_input, self.hass
                    ),
                    description_placeholders={
                        "provider": (
                            VERTEX_PROVIDER_NAME if is_vertex else "Google Gemini"
                        ),
                    },
                )
            user_input = _strip_unsupported_settings(user_input)
            if is_vertex and (
                error := await _async_check_vertex_connection(
                    self.hass, user_input, None
                )
            ):
                return self.async_show_form(
                    step_id="provider",
                    data_schema=_provider_schema(
                        selected_provider, user_input, self.hass
                    ),
                    errors={"base": "cannot_connect"},
                    description_placeholders={
                        "provider": VERTEX_PROVIDER_NAME,
                        "error": error,
                    },
                )
            user_input[CONF_PROVIDER] = selected_provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            title = {
                PROVIDER_OPENAI: "GPT Realtime",
                PROVIDER_QWEN: "Qwen Realtime",
                PROVIDER_PERSONAPLEX: "PersonaPlex",
                PROVIDER_VERTEX: "Gemini Live (Vertex AI)",
            }.get(selected_provider, "Gemini Live")
            return self.async_create_entry(title=title, data=user_input)
        return self.async_show_form(
            step_id="provider",
            data_schema=_provider_schema(selected_provider, hass=self.hass),
            description_placeholders={
                "provider": {
                    PROVIDER_OPENAI: "OpenAI",
                    PROVIDER_QWEN: "QwenCloud",
                    PROVIDER_PERSONAPLEX: "fal.ai PersonaPlex",
                    PROVIDER_VERTEX: VERTEX_PROVIDER_NAME,
                }.get(selected_provider, "Google Gemini")
            },
        )

    async def async_step_reconfigure(self, user_input=None):
        """Reconfigure an existing provider entry."""
        entry = self._get_reconfigure_entry()
        config = {**entry.data, **entry.options}
        provider = _provider(config)
        is_vertex = provider == PROVIDER_VERTEX
        if user_input is not None:
            if is_vertex:
                credentials = self._vertex_credentials or config.get(
                    CONF_VERTEX_CREDENTIALS
                )
                if error := await _async_read_vertex_key(
                    self.hass, user_input, credentials
                ):
                    return self.async_show_form(
                        step_id="reconfigure",
                        data_schema=_provider_schema(
                            provider,
                            _vertex_form_values(user_input, credentials),
                            self.hass,
                        ),
                        errors={"base": error},
                    )
                self._vertex_credentials = user_input[CONF_VERTEX_CREDENTIALS]
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="reconfigure",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                )
            user_input = _strip_unsupported_settings(user_input)
            if is_vertex and (
                error := await _async_check_vertex_connection(
                    self.hass, user_input, config
                )
            ):
                return self.async_show_form(
                    step_id="reconfigure",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                    errors={"base": "cannot_connect"},
                    description_placeholders={"error": error},
                )
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_update_reload_and_abort(
                entry,
                data_updates=user_input,
                options={},
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_provider_schema(provider, config, self.hass),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the options flow."""
        return GeminiLiveOptionsFlowHandler()


class GeminiLiveOptionsFlowHandler(config_entries.OptionsFlow):
    """Manage live provider options."""

    _vertex_credentials: dict[str, Any] | None = None

    async def async_step_init(self, user_input=None):
        """Update the provider's connection and response settings."""
        config = {**self.config_entry.data, **self.config_entry.options}
        provider = _provider(config)
        is_vertex = provider == PROVIDER_VERTEX
        if user_input is not None:
            if is_vertex:
                credentials = self._vertex_credentials or config.get(
                    CONF_VERTEX_CREDENTIALS
                )
                if error := await _async_read_vertex_key(
                    self.hass, user_input, credentials
                ):
                    return self.async_show_form(
                        step_id="init",
                        data_schema=_provider_schema(
                            provider,
                            _vertex_form_values(user_input, credentials),
                            self.hass,
                        ),
                        errors={"base": error},
                    )
                self._vertex_credentials = user_input[CONF_VERTEX_CREDENTIALS]
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                )
            user_input = _strip_unsupported_settings(user_input)
            if is_vertex and (
                error := await _async_check_vertex_connection(
                    self.hass, user_input, config
                )
            ):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                    errors={"base": "cannot_connect"},
                    description_placeholders={"error": error},
                )
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_create_entry(title="", data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=_provider_schema(provider, config, self.hass),
        )
