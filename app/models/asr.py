"""ASR model inference callable."""

import base64
import json
import logging
import tempfile
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from numpy.typing import NDArray

# pylint: disable=import-error,broad-except,too-many-try-statements
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import ASR_MODEL_SIZE, COMPUTE_TYPE, DEVICE, USE_FLASH_ATTENTION
from app.models._common import to_string
from app.models.asr_checker import correct_transcript

warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger(__name__)
ASR_INPUTS = [
    Tensor(name="data", dtype=bytes, shape=(1,)),
    Tensor(name="previous_transcript", dtype=bytes, shape=(1,)),
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


# text or tokens to feed as the prompt or the prefix; for more info:
# https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
# prompt: Optional[Union[str, List[int]]] = None  # for the previous context
# prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context


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


def get_transcription(audio_data: NDArray[Any], previous_transcript: str) -> Tuple[str, str]:
    """Transcribe audio data.

    Parameters
    ----------
    audio_data : NDArray[Any]
        The audio data.
    previous_transcript : str
        The previous transcript.

    Returns
    -------
    Tuple[str, str]
        The text and segments.
    """
    base64_data = np.char.decode(audio_data.astype("bytes"), "utf-8")
    wav_data = base64.b64decode(base64_data.data)
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
        transcript = "".join([segment.text for segment in segments])
        text = correct_transcript(transcript, previous_transcript)
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
    previous_transcripts = [request.data["previous_transcript"] for request in requests]
    total = len(speech_data)
    results = []
    for index in range(total):
        transcription = ""
        segments = "[]"
        audio_data = speech_data[index]
        previous_transcript = previous_transcripts[index]
        try:
            transcription, segments = get_transcription(audio_data, previous_transcript=to_string(previous_transcript))
        except BaseException as exc:
            LOG.error("Error transcribing audio: %s", exc)
        result = {
            "text": np.array([transcription], dtype=bytes),
            "segments": np.array([segments], dtype=bytes),
        }
        results.append(result)
    return results
