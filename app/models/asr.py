"""ASR model inference callable."""

import base64
import json
import logging
import tempfile
import warnings
from typing import Any, Dict, List

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from numpy.typing import NDArray

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import ASR_MODEL_SIZE, COMPUTE_TYPE, DEVICE, USE_FLASH_ATTENTION

warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger(__name__)
ASR_INPUTS = [
    Tensor(name="data", dtype=bytes, shape=(1,)),
]
ASR_OUTPUTS = [
    Tensor(name="results", dtype=bytes, shape=(1,)),
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
) -> List[Dict[str, Any]]:
    """Transcribe audio data.

    Parameters
    ----------
    audio_data : NDArray[Any]
        The audio data.

    Returns
    -------
    Tuple[str, List[Dict[str, Any]]]
        The text and segments.
    """
    base64_data = np.char.decode(audio_data.astype("bytes"), "utf-8")
    try:
        wav_data = base64.b64decode(base64_data)
    except BaseException:  # pylint: disable=broad-except
        return []
    if not wav_data:
        return []
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
        segment_dicts = _segments_to_dicts(segments)
    return segment_dicts


def asr_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.str_]]]:
    """Inference function for ASR model.

    Parameters
    ----------
    requests : List[Request]
        The requests.

    Returns
    -------
    List[Dict[str, NDArray[np.str_]]]
        The inference results.
        The final text can be obtained by joining each segment's ["text"] field.
    """
    speech_data = [request.data["data"] for request in requests]
    total = len(speech_data)
    results = []
    for index in range(total):
        segments: List[Dict[str, Any]] = []
        audio_data = speech_data[index]
        try:
            segments = get_transcription(audio_data)
        except BaseException as exc:  # pylint: disable=broad-except
            LOG.error("Error transcribing audio: %s", exc)
            results.append({"results": np.array([], dtype=bytes)})
            continue
        segments_dump = json.dumps(segments, ensure_ascii=False).encode("utf-8").decode("utf-8")
        result = {
            "results": np.array([segments_dump], dtype=bytes),
        }
        results.append(result)
    return results
