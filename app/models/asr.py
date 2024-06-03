"""ASR model inference callable."""

import json
import logging
import tempfile
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

# pylint: disable=import-outside-toplevel,wrong-import-order,wrong-import-position
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from numpy.typing import NDArray

# pylint: disable=import-error,broad-except,too-many-try-statements
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import ASR_MODEL_SIZE, COMPUTE_TYPE, DEVICE, USE_FLASH_ATTENTION
from app.models._common import concat_wav_data

warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger(__name__)
ASR_INPUTS = [
    Tensor(name="data", dtype=bytes, shape=(1,)),
    Tensor(name="previous_data", dtype=bytes, shape=(1,)),
]
ASR_OUTPUTS = [
    Tensor(name="text", dtype=bytes, shape=(1,)),
    Tensor(name="segments", dtype=bytes, shape=(1,)),
]

model = WhisperModel(
    ASR_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    flash_attention=USE_FLASH_ATTENTION,
)


def _segments_to_dicts(segments: List[Segment]) -> List[Dict[str, Any]]:
    """Convert segments to dictionaries.

    Parameters
    ----------
    segments : List[Segment]
        The segment instances.

    Returns
    -------
    List[Dict[str, Any]]
        The segment dictionaries.

    """
    dicts = []
    for segment in segments:
        segment_dict = segment._asdict()
        segment_dict["words"] = [word._asdict() for word in segment.words]
        dicts.append(segment_dict)
    return dicts


def get_transcription(
    audio_data: NDArray[Any],
    previous_chunk: NDArray[Any],
) -> Tuple[str, str]:
    """Transcribe audio data.

    Parameters
    ----------
    audio_data : NDArray[Any]
        The audio data.
    previous_chunk : NDArray[Any]
        The previous audio chunk.
        The previous transcript.

    Returns
    -------
    Tuple[str, str]
        The text and segments.
    """
    base64_data = np.char.decode(audio_data.astype("bytes"), "utf-8")
    previous_base64_data = np.char.decode(previous_chunk.astype("bytes"), "utf-8")
    wav_data = concat_wav_data(previous_base64_data, base64_data)  # careful with the order
    if not wav_data:
        return "", "[]"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        temp_file.write(wav_data)
        temp_file.flush()
        segments_iter, _ = model.transcribe(
            audio=temp_file.name,
            language="en",
            task="transcribe",
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
            vad_filter=True,
            hallucination_silence_threshold=0.3,
        )
        segments = list(segments_iter)
        text = "".join([segment.text for segment in segments])
        segment_dicts = _segments_to_dicts(segments)
        segments_dump = json.dumps(segment_dicts, ensure_ascii=False).encode("utf-8").decode("utf-8")
    return text, segments_dump


def asr_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]:
    """Inference function for ASR model.

    Parameters
    ----------
    requests : List[Request]
        The requests.

    Returns
    -------
    List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]
        The inference results.
    """
    speech_data = [request.data["data"] for request in requests]
    previous_data = [request.data["previous_data"] for request in requests]
    total = len(speech_data)
    results = []
    for index in range(total):
        transcription = ""
        segments = "[]"
        audio_data = speech_data[index]
        previous_chunk = previous_data[index]
        try:
            transcription, segments = get_transcription(
                audio_data,
                previous_chunk=previous_chunk,
            )
        except BaseException as exc:
            LOG.error("Error transcribing audio: %s", exc)
        result = {
            "text": np.array([transcription], dtype=bytes),
            "segments": np.array([segments], dtype=bytes),
        }
        results.append(result)
    return results
