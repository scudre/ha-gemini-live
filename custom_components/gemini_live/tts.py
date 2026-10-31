"""Text-to-Speech platform for Gemini Live."""

import asyncio
import io
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, suppress
from typing import Any
import wave

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_PREFERRED_SAMPLE_BYTES,
    ATTR_PREFERRED_SAMPLE_CHANNELS,
    ATTR_PREFERRED_SAMPLE_RATE,
    TextToSpeechEntity,
    TtsAudioType,
)
from homeassistant.components.tts.entity import TTSAudioRequest, TTSAudioResponse
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .compat import supports_tts_interruption
from .const import (
    CONF_API_KEY,
    CONF_PROVIDER,
    CONF_SUPPORT_BARGE_IN,
    CONF_TTS_MODEL,
    CONF_VOICE,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    DOMAIN,
    GEMINI_LIVE_TTS_PLACEHOLDER,
    GEMINI_TURN_STORE_KEY,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_QWEN,
    CONF_MODEL,
    QWEN_SUPPORTED_LANGUAGES,
    SUPPORTED_LANGUAGES,
)
from .gemini import async_stream_gemini_tts
from .runtime import AudioStream
from .utils import (
    PCM24kTo16kStreamResampler,
    PCMPassthrough,
    pcm_to_wav,
    streaming_wav_header,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Gemini Live TTS platform."""
    config = {**config_entry.data, **config_entry.options}
    provider = config.get(CONF_PROVIDER, PROVIDER_GEMINI)
    entity_class = {PROVIDER_OPENAI: GPTRealtimeTTS, PROVIDER_QWEN: QwenRealtimeTTS}.get(provider, GeminiLiveTTS)
    async_add_entities([entity_class(config_entry)])


class GeminiLiveTTS(TextToSpeechEntity):
    """Gemini Live TTS Entity.

    This entity does NOT synthesise audio itself. Instead, it retrieves or
    streams the native audio from the Gemini Live turn handled by the STT
    stage of the same pipeline run.

    The conversation text is always delivered via the
    `gemini_live_conversation_entry` event fired by conversation.py. Text
    without Live audio, such as a reply to a command Home Assistant handled
    locally, is read aloud by a Gemini TTS model. Otherwise a short silence
    lets the pipeline complete cleanly.
    """

    _attr_should_poll = False
    integration_domain = DOMAIN
    integration_name = "Gemini Live"
    turn_store_key = GEMINI_TURN_STORE_KEY
    supported_language_codes = SUPPORTED_LANGUAGES
    # Whether text without Live audio can be read aloud with Gemini TTS.
    speaks_text = True

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the TTS entity."""
        self.entry = entry
        config = {**entry.data, **entry.options}
        self._support_barge_in = bool(
            config.get(CONF_SUPPORT_BARGE_IN, False)
        ) and supports_tts_interruption()
        # Barge-in playback needs 16 kHz; Gemini TTS streams 24 kHz.
        self._sample_rate = 16000 if self._support_barge_in else 24000
        self._tts_client: Any = None
        self._attr_name = self.integration_name
        self._attr_unique_id = f"{entry.entry_id}_tts"

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return self.supported_language_codes

    @property
    def supported_options(self) -> list[str]:
        """Return supported options."""
        return [
            ATTR_PREFERRED_FORMAT,
            ATTR_PREFERRED_SAMPLE_RATE,
            ATTR_PREFERRED_SAMPLE_CHANNELS,
            ATTR_PREFERRED_SAMPLE_BYTES,
        ]

    @property
    def supports_audio_interrupt(self) -> bool:
        """Use Core's uncached, single-consumer stream when barge-in is enabled."""
        return self._support_barge_in

    @property
    def default_options(self) -> dict[str, Any]:
        """Use the native WAV format for interruptible playback."""
        return {ATTR_PREFERRED_FORMAT: "wav"} if self._support_barge_in else {}

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType:
        """Retrieve the cached audio response from hass.data."""
        entry_data = self.hass.data[self.integration_domain][self.entry.entry_id]
        audio = entry_data[self.turn_store_key].take_audio(message)
        _LOGGER.warning(
            "TTS: async_get_tts_audio called | message=%r | matched_audio=%s",
            message[:80] if message else "(none)",
            len(audio) if isinstance(audio, bytes) else type(audio).__name__,
        )
        if isinstance(audio, bytes):
            return "wav", audio
        if self._should_speak(message):
            pcm = b"".join([chunk async for chunk in self._async_speak(message)])
            if pcm:
                return "wav", pcm_to_wav(pcm, self._sample_rate)

        _LOGGER.debug(
            "TTS: No cached audio found; returning silence so the pipeline can finish."
        )

        return "wav", self._get_dummy_wav()

    async def async_stream_tts_audio(
        self,
        request: TTSAudioRequest,
    ) -> TTSAudioResponse:
        """Stream the live model's native audio response as it arrives."""
        if self._support_barge_in:
            native_options = {
                ATTR_PREFERRED_FORMAT: "wav",
                ATTR_PREFERRED_SAMPLE_RATE: 16000,
                ATTR_PREFERRED_SAMPLE_CHANNELS: 1,
                ATTR_PREFERRED_SAMPLE_BYTES: 2,
            }
            for option, native_value in native_options.items():
                value = request.options.get(option, native_value)
                if str(value) != str(native_value):
                    raise HomeAssistantError(
                        "Barge-in requires 16 kHz, mono, 16-bit PCM WAV playback"
                    )
        entry_data = self.hass.data[self.integration_domain][self.entry.entry_id]
        turn_store = entry_data[self.turn_store_key]
        try:
            initial_text = await anext(request.message_gen)
        except StopAsyncIteration:
            initial_text = ""

        audio = turn_store.take_streaming_audio(initial_text)
        message = initial_text

        if audio is None:
            message += "".join([chunk async for chunk in request.message_gen])
            audio = turn_store.take_audio(message)
        if audio is None and self._should_speak(message):
            return TTSAudioResponse("wav", self._async_stream_speech_wav(message))

        async def data_gen() -> AsyncGenerator[bytes]:
            if isinstance(audio, AudioStream):

                async def drain_message_stream() -> None:
                    async for _chunk in request.message_gen:
                        pass

                drain_task = asyncio.create_task(drain_message_stream())
                unsubscribe_interrupt = None
                if (
                    self._support_barge_in
                    and (
                        on_audio_interrupt := getattr(
                            request, "on_audio_interrupt", None
                        )
                    )
                    is not None
                ):
                    unsubscribe_interrupt = audio.subscribe_interrupt(
                        on_audio_interrupt
                    )
                try:
                    yield streaming_wav_header(audio.sample_rate)
                    async with aclosing(audio.async_chunks()) as chunks:
                        async for chunk in chunks:
                            yield chunk
                finally:
                    if unsubscribe_interrupt is not None:
                        unsubscribe_interrupt()
                    drain_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await drain_task
                return
            buffered_audio = audio or self._get_dummy_wav()
            if self._support_barge_in:
                with wave.open(io.BytesIO(buffered_audio), "rb") as wav:
                    header = streaming_wav_header(wav.getframerate())
                    pcm = wav.readframes(wav.getnframes())
                yield header
                yield pcm
                return
            yield buffered_audio

        _LOGGER.warning(
            "TTS: async_stream_tts_audio called | message=%r | streaming=%s",
            message[:80] if message else "(none)",
            isinstance(audio, AudioStream),
        )
        return TTSAudioResponse("wav", data_gen())

    def _should_speak(self, message: str) -> bool:
        """Return whether text has no Live audio and should be read aloud.

        A Live placeholder without audio means that turn failed; reading the
        placeholder out would be worse than silence.
        """
        return (
            self.speaks_text
            and bool(message.strip())
            and not message.startswith(GEMINI_LIVE_TTS_PLACEHOLDER)
        )

    async def _async_tts_client(self) -> Any:
        """Return the Google client used for Gemini TTS, created once."""
        if self._tts_client is None:
            from google import genai  # noqa: PLC0415

            api_key = {**self.entry.data, **self.entry.options}[CONF_API_KEY]
            self._tts_client = await self.hass.async_add_executor_job(
                lambda: genai.Client(api_key=api_key)
            )
        return self._tts_client

    async def _async_speak(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM for text read aloud by Gemini TTS; nothing on failure."""
        config = {**self.entry.data, **self.entry.options}
        converter = (
            PCM24kTo16kStreamResampler()
            if self._support_barge_in
            else PCMPassthrough()
        )
        try:
            async for pcm in async_stream_gemini_tts(
                await self._async_tts_client(),
                config.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL),
                text,
                config.get(CONF_VOICE, DEFAULT_VOICE),
            ):
                if chunk := converter.process(pcm):
                    yield chunk
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("%s text-to-speech failed: %s", self.integration_name, exc)
            return
        if chunk := converter.flush():
            yield chunk

    async def _async_stream_speech_wav(self, text: str) -> AsyncGenerator[bytes]:
        """Stream text read aloud by Gemini TTS as WAV, or silence on failure."""
        yield streaming_wav_header(self._sample_rate)
        spoke = False
        async for chunk in self._async_speak(text):
            spoke = True
            yield chunk
        if not spoke:
            # One second of silence so the pipeline still completes cleanly.
            yield b"\x00" * (self._sample_rate * 2)

    def _get_dummy_wav(self) -> bytes:
        """Return 1 second of silence as 16kHz mono 16-bit PCM WAV."""
        # 16000 samples/sec * 2 bytes/sample * 1 sec = 32000 bytes of zero
        pcm_data = b"\x00" * 32000
        return pcm_to_wav(pcm_data, 16000)


class GPTRealtimeTTS(GeminiLiveTTS):
    """Play native audio produced by the GPT Realtime turn."""

    integration_name = "GPT Realtime"
    speaks_text = False


class QwenRealtimeTTS(GeminiLiveTTS):
    """Play the native 24 kHz Qwen audio using the shared TTS handoff."""

    integration_name = "Qwen Realtime"
    speaks_text = False

    supported_language_codes = QWEN_SUPPORTED_LANGUAGES
