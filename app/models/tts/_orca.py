"""TTS model runner using pvorca."""

import logging
import os
import tempfile
from typing import Any

import httpx
import numpy as np
import pvorca
from numpy.typing import NDArray
from pvorca import Orca

from app.config import TTS_MODEL_ORCA_KEY, TTS_MODEL_SAMPLE_RATE
from app.models._common import to_wav

LOG = logging.getLogger(__name__)


def _download_male_orca_model_params(model_path: str) -> None:
    """Download male model from github."""
    url = "https://raw.githubusercontent.com/Picovoice/orca/main/lib/common/orca_params_male.pv"
    # pylint: disable=broad-except,too-many-try-statements
    LOG.debug("Downloading from %s to %s", url, model_path)
    try:
        with open(model_path, "wb") as file:
            with httpx.stream("GET", url, follow_redirects=True) as response:
                for chunk in response.iter_bytes():
                    file.write(chunk)
    except BaseException as exc:  # pylint: disable=broad-except
        LOG.error("Failed to download male model: %s", exc)
        if model_path and os.path.exists(model_path):
            os.remove(model_path)


def _download_female_orca_model_params(model_path: str) -> None:
    """Download female model from github."""
    url = "https://raw.githubusercontent.com/Picovoice/orca/main/lib/common/orca_params_female.pv"
    # pylint: disable=broad-except,too-many-try-statements
    LOG.debug("Downloading from %s to %s", url, model_path)
    try:
        with open(model_path, "wb") as file:
            with httpx.stream("GET", url, follow_redirects=True) as response:
                for chunk in response.iter_bytes():
                    file.write(chunk)
    except BaseException as exc:  # pylint: disable=broad-except
        LOG.error("Failed to download female model: %s", exc)
        if model_path and os.path.exists(model_path):
            os.remove(model_path)


# pylint: disable=too-few-public-methods
class OrcaTTSRunner:
    """Orca TTS model runner."""

    def __init__(self) -> None:
        """Initialize the Orca TTS model runner."""
        if not TTS_MODEL_ORCA_KEY:
            raise ValueError("TTS_MODEL_ORCA_KEY is not set")
        model_path = pvorca.default_model_path()
        if model_path.endswith("female.pv"):
            self._orca_female = pvorca.create(access_key=TTS_MODEL_ORCA_KEY, model_path=model_path)
            male_model_path = model_path.replace("female.pv", "male.pv")
            if not os.path.exists(male_model_path):
                _download_male_orca_model_params(male_model_path)
            if not os.path.exists(male_model_path):
                raise ValueError("Failed to download the male model params.")
            self._orca_male = pvorca.create(access_key=TTS_MODEL_ORCA_KEY, model_path=male_model_path)
        elif model_path.endswith("male.pv"):
            self._orca_male = pvorca.create(access_key=TTS_MODEL_ORCA_KEY, model_path=model_path)
            female_model_path = model_path.replace("male.pv")
            if not os.path.exists(female_model_path):
                _download_female_orca_model_params(female_model_path)
            if not os.path.exists(female_model_path):
                raise ValueError("Failed to download the female model params.")

    def __call__(self, text: str, **kwargs: Any) -> bytes | NDArray[np.float_] | None:
        """Run the Orca TTS model runner."""
        gender = str(kwargs.get("gender", "")).lower()
        if gender not in ("male", "female"):
            gender = "female"
        _orca: Orca = self._orca_male if gender == "male" else self._orca_female
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # pylint: disable=broad-except,too-many-try-statements
            try:
                _orca.synthesize(text, temp_file.name)
                with open(temp_file.name, "rb") as file:
                    wav_data = file.read()
                if TTS_MODEL_SAMPLE_RATE == 16000:
                    wav_data = to_wav(temp_file.name, sample_rate=TTS_MODEL_SAMPLE_RATE)
            except BaseException as error:
                LOG.error("Error synthesizing speech: %s", error)
                return None
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        return wav_data
