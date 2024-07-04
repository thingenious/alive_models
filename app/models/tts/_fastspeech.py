"""FastSpeech2 TTS model."""

import logging
from typing import Any

import numpy as np
import torch
import torchaudio.transforms as T
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from numpy.typing import NDArray

from app.config import CACHE_DIR, DEVICE, TTS_MODEL_REPO, TTS_MODEL_SAMPLE_RATE

FAIRSEQ_CACHE_DIR = CACHE_DIR / "fairseq"
FAIRSEQ_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger(__name__)


# pylint: disable=too-many-try-statements, broad-except, unused-argument, too-few-public-methods
class FastSpeechRunner:
    """FastSpeech model runner."""

    def __init__(self) -> None:
        """Initialize the FastSpeech model runner."""
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            TTS_MODEL_REPO,
            cache_dir=FAIRSEQ_CACHE_DIR,
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        model = models[0]
        self.model = model.to(DEVICE)
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.generator = task.build_generator([self.model], cfg)
        self.task = task

    def __call__(self, text: str, **kwargs: Any) -> NDArray[np.float_] | None:
        """Run the FastSpeech model."""
        speaker_index = kwargs.get("speaker_index")
        try:
            sample = TTSHubInterface.get_model_input(self.task, text, speaker=speaker_index, verbose=False)
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(DEVICE)
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(DEVICE)
            speaker = sample["speaker"] if sample["speaker"] is not None else torch.tensor([[speaker_index]])
            sample["speaker"] = speaker.to(DEVICE)
            wav, rate = TTSHubInterface.get_prediction(self.task, self.model, self.generator, sample)
        except BaseException as error:
            LOG.error("Error getting prediction: %s", error)
            return None
        try:
            if rate != TTS_MODEL_SAMPLE_RATE:
                sampler = T.Resample(orig_freq=rate, new_freq=TTS_MODEL_SAMPLE_RATE).to(DEVICE)
                # pylint: disable=not-callable
                wav = sampler(wav.unsqueeze(0)).squeeze(0)
            return wav.detach().cpu().numpy().squeeze().astype(np.float32)
        except BaseException as error:
            LOG.error("Error: %s", error)
            return None
