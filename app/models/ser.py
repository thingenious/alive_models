"""SER model inference callable."""

import base64
import logging
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import SER_MODEL_REPO

LOG = logging.getLogger(__name__)
SER_INPUTS = [Tensor(name="data", dtype=bytes, shape=(1,))]
SER_OUTPUTS = [
    Tensor(name="label", dtype=bytes, shape=(1,)),
    Tensor(name="score", dtype=np.float32, shape=(1,)),
]

# pylint: disable=wrong-import-position,wrong-import-order
from transformers import pipeline  # noqa

classifier = pipeline("audio-classification", model=SER_MODEL_REPO)


def get_audio_analysis(audio_data: bytes) -> Tuple[str, float]:
    """Get the prediction from the model.

    Parameters
    ----------
    data : bytes
        The wav audio data.

    Returns
    -------
    Tuple[str, float]
        The label and score.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        temp_file.write(audio_data)
        temp_file.flush()
        outputs = classifier(temp_file.name)  # type: ignore
    highest_score = max(outputs, key=lambda x: x["score"])
    label = highest_score["label"]
    score = highest_score["score"]
    return label, score


# pylint: disable=broad-except,too-many-try-statements
def ser_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]:
    """Inference function for SER model.

    Parameters
    ----------
    requests : List[Request]
        The requests.

    Returns
    -------
    List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]
        The inference results.
    """
    infer_inputs = [request.data["data"] for request in requests]
    total = len(infer_inputs)
    results = []
    for index in range(total):
        label = "unknown"
        score = 0.0
        data_string = infer_inputs[index]
        try:
            base64_data = np.char.decode(data_string.astype("bytes"), "utf-8")
            wav_data = base64.b64decode(base64_data)
            label, score = get_audio_analysis(wav_data)
        except BaseException as exc:
            LOG.error("Error analyzing file: %s", exc)
        result = {
            "label": np.char.encode(label, "utf-8"),
            "score": np.array([score], dtype=np.float32),
        }
        results.append(result)
    return results
