"""Bark model."""

import logging
from typing import Any

import numpy as np
import torchaudio.transforms as T
from numpy.typing import NDArray
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, BarkModel

from app.config import DEVICE, TORCH_DTYPE, TTS_MODEL_REPO, TTS_MODEL_SAMPLE_RATE, USE_FLASH_ATTENTION

LOG = logging.getLogger(__name__)


# pylint: disable=broad-except,too-many-try-statements,unused-argument,too-few-public-methods
class BarkRunner:
    """Bark model runner."""

    def __init__(self) -> None:
        """Initialize the Bark model runner."""
        self.processor = AutoProcessor.from_pretrained(TTS_MODEL_REPO)
        model = BarkModel.from_pretrained(
            TTS_MODEL_REPO,
            torch_dtype=TORCH_DTYPE,
            attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else None,
        )
        model = model.to(DEVICE)
        model.enable_cpu_offload()
        self.model = BetterTransformer.transform(model, keep_original_model=False).to(DEVICE)

    def __call__(self, text: str, **kwargs: Any) -> NDArray[np.float_] | None:
        """Run the Bark model."""
        try:
            inputs = self.processor(text, return_tensors="pt").to(DEVICE)
            speech = self.model.generate(**inputs, do_sample=True)
            sampling_rate = self.model.generation_config.sample_rate
            if sampling_rate != TTS_MODEL_SAMPLE_RATE:
                sampler = T.Resample(orig_freq=sampling_rate, new_freq=TTS_MODEL_SAMPLE_RATE).to(DEVICE)
                speech = sampler(speech.unsqueeze(0).float()).squeeze(0)
            return speech.detach().cpu().numpy().squeeze().astype(np.float32)
        except BaseException as error:
            LOG.error("Error: %s", error)
            return None
