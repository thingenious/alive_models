"""SpeechT5 TTS model."""

import logging
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from numpy.typing import NDArray
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from app.config import DEVICE

_DEFAULT_SPEAKER_INDEX = 50

LOG = logging.getLogger(__name__)


# pylint: disable=broad-except,too-many-try-statements,too-few-public-methods
class SpeechT5Runner:
    """SpeechT5 model runner."""

    def __init__(self) -> None:
        """Initialize the SpeechT5 model runner."""
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

    def __call__(self, text: str, **kwargs: Any) -> NDArray[np.float_] | None:
        """Run the SpeechT5 model."""
        speaker_index = kwargs.get("speaker_index")
        if speaker_index is None or not isinstance(speaker_index, int):
            speaker_index = _DEFAULT_SPEAKER_INDEX
        try:
            speaker_embeddings = torch.tensor(self.embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0).to(DEVICE)
            input_ids = self.processor(text=text, return_tensors="pt")["input_ids"].to(DEVICE)
            speech = self.model.generate_speech(input_ids, speaker_embeddings, vocoder=self.vocoder)
        except BaseException as error:
            LOG.error("Error: %s", error)
            return None
        return speech.detach().cpu().numpy()
