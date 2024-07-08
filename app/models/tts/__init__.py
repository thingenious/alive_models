"""TTS model inference callable."""

# pylint: disable=import-error,broad-except,too-many-try-statements,unused-argument
import base64
import logging
import tempfile
import wave
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import TTS_MODEL_REPO, TTS_MODEL_SAMPLE_RATE
from app.models._common import to_string

from .__base__ import RunnerProtocol

__all__ = [
    "TTS_INPUTS",
    "TTS_OUTPUTS",
    "tts_infer_fn",
]
TTS_INPUTS = [
    Tensor(name="data", dtype=bytes, shape=(1,)),
]
TTS_OUTPUTS = [
    Tensor(name="results", dtype=bytes, shape=(1,)),
]
LOG = logging.getLogger(__name__)
LOG.info("TTS_MODEL_REPO: %s", TTS_MODEL_REPO)

TTS_PARAMS = {
    "speaker_index": int,  # several backends
    "speaker_description": str,  # parler
    "speaker_name": str,  # azure, other?
    "speaker_gender": str,  # azure, edge_tts, orca
    "language": str,  # azure, edge_tts
    "locale": str,  # azure, edge_tts
    "rate": str,  # edge_tts
    "pitch": str,  # edge_tts
    "volume": str,  # edge_tts
}


# pylint: disable=import-outside-toplevel,too-many-return-statements
def get_runner() -> RunnerProtocol:
    """Get the TTS model runner based on the model repository."""
    if "speecht5" in TTS_MODEL_REPO:
        from ._speecht5 import SpeechT5Runner

        return SpeechT5Runner()
    if "parler" in TTS_MODEL_REPO:
        from ._parler import ParlerRunner

        return ParlerRunner()
    if "fastspeech" in TTS_MODEL_REPO:
        from ._fastspeech import FastSpeechRunner

        return FastSpeechRunner()
    if "seamless" in TTS_MODEL_REPO:
        from ._seamless import SeamlessRunner

        return SeamlessRunner()
    if "suno/bark" in TTS_MODEL_REPO:
        from ._bark import BarkRunner

        return BarkRunner()
    if "whisper" in TTS_MODEL_REPO:
        from ._whisper import WhisperRunner

        return WhisperRunner()

    if "edge" in TTS_MODEL_REPO:
        from ._edge_tts import EdgeTTSRunner

        return EdgeTTSRunner()

    if "azure" in TTS_MODEL_REPO:
        from ._azure import AzureTTSRunner

        return AzureTTSRunner()

    if "orca" in TTS_MODEL_REPO:
        from ._orca import OrcaTTSRunner

        return OrcaTTSRunner()

    raise ValueError(f"Unsupported TTS model repository: {TTS_MODEL_REPO}")


runner: RunnerProtocol = get_runner()


def get_parameters(request: Request) -> Dict[str, Any]:
    """Get the speaker index and description from the request.

    Parameters
    ----------
    request : Request
        The request.

    Returns
    -------
    Tuple[int | None, str | None]
        The speaker index and description.
    """
    if not hasattr(request, "parameters"):
        return {}
    params = {}
    for key, value_type in TTS_PARAMS.items():
        if key in request.parameters:
            try:
                value = request.parameters[key]
                if value is not None:
                    params[key] = value_type(value)
            except (ValueError, TypeError) as error:
                LOG.debug("Invalid %s parameter: %s", key, error)
    return params


def get_speech(text: str, **kwargs: Any) -> Dict[str, NDArray[np.str_]]:
    """Get the speech from the model.

    Parameters
    ----------
    text : str
        The text to synthesize.
    **kwargs : Any
        Additional keyword arguments.
    Returns
    -------
    Dict[str, NDArray[np.str_]]
        The speech as a base64 encoded string.
    """
    speech = runner(text, **kwargs)
    if speech is None:
        return {"results": np.array([""], dtype=bytes)}
    if isinstance(speech, bytes):
        with BytesIO(speech) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                speech = wav_file.readframes(wav_file.getnframes())
        speech_dump = base64.b64encode(speech)
        result = {
            "results": np.array([speech_dump], dtype=bytes),
        }
        return result
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        sf.write(file=temp_file.name, data=speech, samplerate=TTS_MODEL_SAMPLE_RATE)
        with wave.open(temp_file.name, "rb") as wav_file:
            speech = wav_file.readframes(wav_file.getnframes())
    speech_dump = base64.b64encode(speech)
    result = {
        "results": np.array([speech_dump], dtype=bytes),
    }
    return result


def tts_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.str_]]]:
    """Inference function for TTS model.

    Parameters
    ----------
    requests : List[Request]
        The requests.

    Returns
    -------
    List[Dict[str, NDArray[np.str_]]]
        The inference results.
    """
    text_data = [request.data["data"] for request in requests]
    results = []
    for index, entry in enumerate(text_data):
        text = to_string(entry)
        request = requests[index]
        request_params = get_parameters(request)
        try:
            result = get_speech(text=text, **request_params)
        except BaseException as error:
            LOG.error("Error synthesizing speech: %s", error)
            result = {"results": np.array([""], dtype=bytes)}
        results.append(result)
    return results
