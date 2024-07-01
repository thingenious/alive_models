"""Seamless TTS model."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4Tv2Model

from app.config import DEVICE, TTS_MODEL_REPO

LOG = logging.getLogger(__name__)


# pylint: disable=broad-except,too-many-try-statements,unused-argument,too-few-public-methods
class SeamlessRunner:
    """Seamless model runner."""

    model: SeamlessM4TModel | SeamlessM4Tv2Model
    processor: AutoProcessor
    _is_v2: bool

    def __init__(self) -> None:
        """Initialize the Seamless model runner."""
        self.processor = AutoProcessor.from_pretrained(TTS_MODEL_REPO)
        self._is_v2 = "-v2" in TTS_MODEL_REPO
        if self._is_v2:
            self.model = SeamlessM4Tv2Model.from_pretrained(TTS_MODEL_REPO).to(DEVICE)
        else:
            self.model = SeamlessM4TModel.from_pretrained(TTS_MODEL_REPO).to(DEVICE)

    def __call__(self, text: str, **kwargs: Any) -> NDArray[np.float_] | None:
        """Run the Seamless model."""
        speaker_index = kwargs.get("speaker_index", None)
        if speaker_index is not None:
            # vocoder_speakers = self.model.config.vocoder_num_spkrs
            vocoder_speakers = getattr(self.model.config, "vocoder_num_spkrs", 0)
            if speaker_index >= vocoder_speakers:
                LOG.warning("Speaker index out of range: %s", speaker_index)
                speaker_index = None
        try:
            inputs = self.processor(text, return_tensors="pt").to(DEVICE)
            if self._is_v2:
                speech = self.model.generate(**inputs, tgt_lang="eng", speaker_id=speaker_index, speech_do_sample=True)
            else:
                speech = self.model.generate(**inputs, tgt_lang="eng", spkr_id=speaker_index, speech_do_sample=True)
            return speech[0].detach().cpu().numpy().squeeze().astype(np.float32)
        except BaseException as error:
            LOG.error("Error: %s", error)
            return None
