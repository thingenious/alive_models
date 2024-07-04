"""Whisper model for text-to-speech."""

import logging
from io import BytesIO
from typing import Any

import torchaudio
import torchaudio.transforms as T
from whisperspeech.pipeline import Pipeline

from app.config import DEVICE, TTS_MODEL_REPO, TTS_MODEL_SAMPLE_RATE

WHISPER_RATE = 24000  # site-packages/whisperspeech/a2wav.py

LOG = logging.getLogger(__name__)


# pylint: disable=too-many-try-statements, broad-except, unused-argument, too-few-public-methods
class WhisperRunner:
    """Whisper model runner."""

    def __init__(self) -> None:
        """Initialize the Whisper model runner."""
        self.model = Pipeline(s2a_ref=TTS_MODEL_REPO, device=DEVICE)

    def __call__(self, text: str, **kwargs: Any) -> bytes | None:
        """Run the Whisper model."""
        try:
            audio = self.model.generate(text, lang="en", cps=10, speaker=None)
            if WHISPER_RATE != TTS_MODEL_SAMPLE_RATE:
                sampler = T.Resample(orig_freq=WHISPER_RATE, new_freq=TTS_MODEL_SAMPLE_RATE).to(DEVICE)
                audio = sampler(audio.unsqueeze(0)).squeeze(0)
            with BytesIO() as buffer:
                torchaudio.save(buffer, audio.cpu(), TTS_MODEL_SAMPLE_RATE, format="WAV")
                buffer.seek(0)
                speech = buffer.read()
            return speech
        except BaseException as error:
            LOG.error("Error: %s", error)
            return None
