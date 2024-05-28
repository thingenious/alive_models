"""ASR model inference callable."""

import base64
import json
import logging
import tempfile
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from faster_whisper import WhisperModel
from numpy.typing import NDArray

# pylint: disable=import-error,broad-except,too-many-try-statements
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import ASR_MODEL_SIZE

warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger(__name__)
ASR_INPUTS = [
    Tensor(name="data", dtype=bytes, shape=(1,)),
    Tensor(name="initial_prompt", dtype=bytes, shape=(1,)),
]
ASR_OUTPUTS = [
    Tensor(name="text", dtype=bytes, shape=(1,)),
    Tensor(name="segments", dtype=bytes, shape=(1,)),
]


_have_cuda = torch.cuda.is_available()
DEVICE = "cuda" if _have_cuda else "cpu"
COMPUTE_TYPE = "float16" if _have_cuda else "float32"
# COMPUTE_TYPE = torch.float16 if _have_cuda else torch.float32
USE_FLASH_ATTENTION = _have_cuda
model = WhisperModel(
    ASR_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    flash_attention=USE_FLASH_ATTENTION,
)


def _to_string(thing: Any) -> str:
    """Convert a thing to a string."""
    if isinstance(thing, str):
        return thing
    if isinstance(thing, bytes):
        return thing.decode("utf-8")
    if isinstance(thing, np.ndarray):
        thing_ = thing[0]
        if isinstance(thing_, bytes):
            return thing_.decode("utf-8")
    return str(thing)


# text or tokens to feed as the prompt or the prefix; for more info:
# https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
# prompt: Optional[Union[str, List[int]]] = None  # for the previous context
# prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context


def get_transcription(audio_data: NDArray[Any], initial_prompt: str) -> Tuple[str, str]:
    """Transcribe audio data."""
    _initial_prompt: str | None = None
    if initial_prompt:  # if empty string, use None
        _initial_prompt = initial_prompt
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
            initial_prompt=_initial_prompt,
            word_timestamps=True,
            vad_filter=True,
            hallucination_silence_threshold=0.3,
        )
        segments = list(segments_iter)
        text = "".join([segment.text for segment in segments])
        segment_dicts = [segment._asdict() for segment in segments]
        segments_dump = json.dumps(segment_dicts, ensure_ascii=False).encode("utf-8").decode("utf-8")
    return text, segments_dump


def asr_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]:
    """Inference function for ASR model."""
    speech_data = [request.data["data"] for request in requests]
    initial_prompts = [request.data["initial_prompt"] for request in requests]
    total = len(speech_data)
    results = []
    for index in range(total):
        transcription = ""
        segments = "[]"
        audio_data = speech_data[index]
        initial_prompt = initial_prompts[index]
        try:
            transcription, segments = get_transcription(audio_data, initial_prompt=_to_string(initial_prompt))
        except BaseException as exc:
            LOG.error("Error transcribing audio: %s", exc)
        result = {
            "text": np.array([transcription], dtype=bytes),
            "segments": np.array([segments], dtype=bytes),
        }
        results.append(result)
    return results
