"""Parler TTS model."""

import logging
from typing import Any

import numpy as np
import torch
import torchaudio.transforms as T
from numpy.typing import NDArray
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

from app.config import DEVICE, TORCH_DTYPE, TTS_MODEL_REPO, TTS_MODEL_SAMPLE_RATE

LOG = logging.getLogger(__name__)


# pylint: disable=broad-except,too-many-try-statements,too-few-public-methods
class ParlerRunner:
    """Parler model runner."""

    def __init__(self) -> None:
        """Initialize the Parler model runner."""
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_REPO).to(DEVICE, dtype=TORCH_DTYPE)
        self.tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_REPO)

    def __call__(self, text: str, **kwargs: Any) -> NDArray[np.float_] | None:
        """Run the Parler model."""
        speaker_description = kwargs.get("speaker_description", "")
        if not speaker_description or not isinstance(speaker_description, str):
            speaker_description = (
                "A female speaker with a slightly low-pitched voice delivers her words quite expressively, "
                "in a very confined sounding environment with very clear audio quality. She speaks fast."
            )
        try:
            input_ids = self.tokenizer(speaker_description, return_tensors="pt").input_ids.to(DEVICE)
            prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
            generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
            audio_arr = generation.cpu().numpy().squeeze()
            sample_rate = self.model.config.sampling_rate
            if sample_rate != TTS_MODEL_SAMPLE_RATE:
                sampler = T.Resample(orig_freq=sample_rate, new_freq=TTS_MODEL_SAMPLE_RATE)
                audio_arr = sampler(torch.tensor(audio_arr).unsqueeze(0)).squeeze(0).numpy()
            return audio_arr
        except BaseException as error:
            LOG.error("Error: %s", error)
            return None
