"""Azure speech-to-text model runner."""

import logging
import os
import tempfile
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List

import azure.cognitiveservices.speech as speechsdk
import httpx
import numpy as np
from numpy.typing import NDArray

from app.config import TTS_MODEL_AZURE_KEY, TTS_MODEL_AZURE_REGION, TTS_MODEL_SAMPLE_RATE

LOG = logging.getLogger(__name__)

FORMAT_MAPPING = {
    16000: speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
    24000: speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm,
    44100: speechsdk.SpeechSynthesisOutputFormat.Riff44100Hz16BitMonoPcm,
}
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"
SELECTED_FORMAT = FORMAT_MAPPING.get(TTS_MODEL_SAMPLE_RATE, speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)


# pylint: disable=too-few-public-methods,invalid-name,too-many-instance-attributes
@dataclass
class Voice:
    """Voice dataclass."""

    Name: str
    DisplayName: str
    LocalName: str
    ShortName: str
    Gender: str
    Locale: str
    LocaleName: str
    SampleRateHertz: str
    VoiceType: str
    Status: str
    WordsPerMinute: str | None = None
    SecondaryLocaleList: List[str] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Voice":
        """Create a voice config from a dictionary."""
        return cls(
            Name=data.get("Name", ""),
            DisplayName=data.get("DisplayName", ""),
            LocalName=data.get("LocalName", ""),
            ShortName=data.get("ShortName", ""),
            Gender=data.get("Gender", ""),
            Locale=data.get("Locale", ""),
            LocaleName=data.get("LocaleName", ""),
            SampleRateHertz=data.get("SampleRateHertz", ""),
            VoiceType=data.get("VoiceType", ""),
            Status=data.get("Status", ""),
            WordsPerMinute=data.get("WordsPerMinute"),
            SecondaryLocaleList=data.get("SecondaryLocaleList"),
        )


class VoiceManager:
    """Speaker voices."""

    def __init__(self, voices: List[Voice]) -> None:
        """Initialize the speaker voices instance."""
        self.voices = voices

    @cached_property
    def names(self) -> List[str]:
        """Get the ShortName of the voices."""
        return [voice.ShortName for voice in self.voices]

    def find(self, **kwargs: Any) -> List[Voice]:
        """Find the voice based on the keyword arguments."""
        return [
            voice for voice in self.voices if all(getattr(voice, key, "") == value for key, value in kwargs.items())
        ]


def _get_voice_manager() -> VoiceManager:
    """Get the speaker voices.

    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming#get-a-list-of-voices
    """
    # pylint: disable=broad-except,too-many-try-statements
    try:
        headers = {"Ocp-Apim-Subscription-Key": TTS_MODEL_AZURE_KEY}
        url = f"https://{TTS_MODEL_AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/voices/list"
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        voices = [Voice.from_dict(item) for item in data]
        return VoiceManager(voices)
    except BaseException:
        return VoiceManager([])


# pylint: disable=too-few-public-methods,invalid-name
class AzureTTSRunner:
    """Azure speech-to-text model runner."""

    def __init__(self) -> None:
        """Initialize the Azure speech-to-text model runner."""
        if not TTS_MODEL_AZURE_KEY:
            raise ValueError("TTS_MODEL_AZURE_KEY is not set")
        if not TTS_MODEL_AZURE_REGION:
            raise ValueError("TTS_MODEL_AZURE_REGION is not set")
        self.speech_config = speechsdk.SpeechConfig(subscription=TTS_MODEL_AZURE_KEY, region=TTS_MODEL_AZURE_REGION)
        self.speech_config.set_speech_synthesis_output_format(SELECTED_FORMAT)
        self._voice_manager = _get_voice_manager()
        self._voice_names = self._voice_manager.names
        if not self._voice_manager.voices:
            raise RuntimeError("No voices found.")
        LOG.info("Loaded %d voices", len(self._voice_manager.voices))

    def _get_voice_name(self, **kwargs: Any) -> str:
        """Get the voice based on the keyword arguments."""
        voice = kwargs.get("voice", "")
        if not voice:
            locale = kwargs.get("locale", "")
            speaker_name = kwargs.get("speaker_name", "")
            speaker_gender = kwargs.get("speaker_gender", "")
            fallback_args = {
                "Locale": "en-US",
                "Gender": "Female",
            }
            filter_args = {}
            if locale:
                filter_args["Locale"] = locale
            if speaker_name and locale:
                short_name = f"{locale}-{speaker_name.capitalize()}Neural"
                if short_name in self._voice_names:
                    return short_name
            if speaker_gender and isinstance(speaker_gender, str):
                gender = speaker_gender.capitalize()
                if gender in ("Male", "Female"):
                    filter_args["Gender"] = gender
                    fallback_args["Gender"] = gender
            if not filter_args:
                filter_args = fallback_args
            voice_filter = self._voice_manager.find(**filter_args)
            voice = voice_filter[0].ShortName if voice_filter else DEFAULT_VOICE
        if voice not in self._voice_names:
            voice = DEFAULT_VOICE
        return voice

    def __call__(self, text: str, **kwargs: Any) -> bytes | NDArray[np.float_] | None:
        """Run the Azure speech-to-text model runner."""
        voice_name = self._get_voice_name(**kwargs)
        self.speech_config.speech_synthesis_voice_name = voice_name
        temp_dir = tempfile.gettempdir()
        # pylint: disable=consider-using-with
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_file.name)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)
        result = synthesizer.speak_text(text)
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            return None
        with open(temp_file.name, "rb") as file:
            wav_data = file.read()
        if temp_file.name and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        return wav_data


# Future: support SSML
# from xml.sax.saxutils import escape
# from edge_tts.communicate import split_text_by_byte_length, calc_max_mesg_size, remove_incompatible_characters
# from app.config import (
#   TTS_MODEL_AZURE_KEY,
#   TTS_MODEL_AZURE_REGION,
#   TTS_MODEL_PITCH,
#   TTS_MODEL_RATE,
#   TTS_MODEL_VOLUME,
#   TTS_MODEL_SAMPLE_RATE
# )
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming#sample-request-1
# POST /cognitiveservices/v1 HTTP/1.1
# X-Microsoft-OutputFormat: riff-24khz-16bit-mono-pcm
# Content-Type: application/ssml+xml
# Host: westeurope.tts.speech.microsoft.com
# Content-Length: <Length>
# Authorization: Bearer [Base64 access_token]
# User-Agent: <Your application name>
# <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" \
#               xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="string">
#     <mstts:backgroundaudio src="string" volume="string" fadein="string" fadeout="string"/>
#     <voice name="string" effect="string">
#         <audio src="string"></audio>
#         <bookmark mark="string"/>
#         <break strength="string" time="string" />
#         <emphasis level="value"></emphasis>
#         <lang xml:lang="string"></lang>
#         <lexicon uri="string"/>
#         <math xmlns="http://www.w3.org/1998/Math/MathML"></math>
#         <mstts:audioduration value="string"/>
#         <mstts:ttsembedding speakerProfileId="string"></mstts:ttsembedding>
#         <mstts:express-as style="string" styledegree="value" role="string"></mstts:express-as>
#         <mstts:silence type="string" value="string"/>
#         <mstts:viseme type="string"/>
#         <p></p>
#         <phoneme alphabet="string" ph="string"></phoneme>
#         <prosody pitch="value" contour="value" range="value" rate="value" volume="value"></prosody>
#         <s></s>
#         <say-as interpret-as="string" format="string" detail="string"></say-as>
#         <sub alias="string"></sub>
#     </voice>
# </speak>
