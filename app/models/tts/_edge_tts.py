"""Edge-tts runner."""

import logging
import os
import tempfile
from typing import Any, Dict, List

import edge_tts
import numpy as np
from asgiref.sync import async_to_sync
from edge_tts import VoicesManager
from numpy.typing import NDArray

from app.config import TTS_MODEL_PITCH, TTS_MODEL_RATE, TTS_MODEL_SAMPLE_RATE, TTS_MODEL_VOLUME
from app.models._common import to_wav

LOG = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods, broad-except
class EdgeTTSRunner:
    """Edge-tts model runner."""

    _voices: List[Dict[str, Any]] = []
    _voices_manager: VoicesManager

    def __init__(self) -> None:
        """Initialize the Edge-tts model runner."""
        self._voices_manager = async_to_sync(VoicesManager.create)()
        self._voices = self._voices_manager.voices
        self._voice_names = [voice["Name"] for voice in self._voices]
        LOG.debug("Voices: %s", self._voices)
        if not self._voices:
            raise RuntimeError("No voices loaded.")
        self._short_names = [voice["ShortName"] for voice in self._voices]

    def _get_voice(self, locale: str, language: str, gender: str, speaker_name: str) -> Dict[str, Any]:
        """Get the voice based on the Locale and/or the gender."""
        default_args = {
            "Language": "en",
            "Locale": "en-US",
            "Gender": "Female",
        }
        filter_args = {}
        if locale:
            filter_args["Locale"] = locale
            if not language:
                language = locale.split("-")[0]
        if gender and gender.lower() in ("male", "female"):
            filter_args["Gender"] = gender.capitalize()
            default_args["Gender"] = gender.capitalize()
        if language:
            filter_args["Language"] = language
        if speaker_name and locale:
            short_name = f"{locale}-{speaker_name.capitalize()}Neural"
            if short_name in self._short_names:
                return self._voices[self._short_names.index(short_name)]
        if not filter_args:
            filter_args = default_args
        result = self._voices_manager.find(**filter_args)
        if result:
            return result[0]
        result = self._voices_manager.find(**default_args)
        if result:
            return result[0]
        return self._voices[0]

    @staticmethod
    def _get_speech(
        text: str, voice: Dict[str, Any], rate: int, pitch: int, volume: int
    ) -> bytes | NDArray[np.float_] | None:
        """Get the speech using the Edge-tts model."""
        communicate: edge_tts.Communicate = edge_tts.Communicate(
            text=text, voice=voice["Name"], rate=rate, pitch=pitch, volume=volume
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            mp3_path = temp_file.name
        try:
            communicate.save_sync(mp3_path)
        except BaseException as error:
            LOG.error("Error synthesizing speech: %s", error)
            return None
        wav_data = to_wav(mp3_path, sample_rate=TTS_MODEL_SAMPLE_RATE)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return wav_data

    def __call__(self, text: str, **kwargs: Any) -> bytes | NDArray[np.float_] | None:
        """Run the Edge-tts model runner."""
        rate = kwargs.get("rate", TTS_MODEL_RATE)
        pitch = kwargs.get("pitch", TTS_MODEL_PITCH)
        volume = kwargs.get("volume", TTS_MODEL_VOLUME)
        voice = kwargs.get("voice", "")
        locale = kwargs.get("locale", "")
        language = kwargs.get("language", "")
        speaker_name = kwargs.get("speaker_name", "")
        if not language and locale:
            language = str(locale).split("-", maxsplit=1)[0]
        gender = kwargs.get("speaker_gender", "")
        if voice and voice in self._voice_names:
            return self._get_speech(text, voice, rate, pitch, volume)
        voice = self._get_voice(locale, language, gender, speaker_name)
        LOG.debug("Voice: %s", voice)
        return self._get_speech(text, voice["Name"], rate, pitch, volume)
