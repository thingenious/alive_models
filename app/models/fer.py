"""FER model inference callable."""

import base64
import logging
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

# pylint: disable=wrong-import-position,wrong-import-order
from deepface import DeepFace  # noqa
from numpy.typing import NDArray

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore # noqa
from pytriton.proxy.types import Request  # type: ignore # noqa

from app.config import FER_MODEL_DETECTOR_BACKEND, FER_MODEL_FACE_MIN_CONFIDENCE

LOG = logging.getLogger(__name__)
FER_INPUTS: List[Tensor] = [Tensor(name="data", dtype=bytes, shape=(1,))]
FER_OUTPUTS: List[Tensor] = [
    Tensor(name="label", dtype=bytes, shape=(1,)),
    Tensor(name="score", dtype=np.float32, shape=(1,)),
]


def _ensure_model() -> None:
    """Ensure the FER model is loaded."""
    random_img_array = np.random.rand(48, 48, 3) * 255
    random_img = np.uint8(random_img_array)
    DeepFace.analyze(
        random_img,
        actions=["emotion"],
        detector_backend=FER_MODEL_DETECTOR_BACKEND,
        enforce_detection=False,
    )


_ensure_model()


# pylint: disable=broad-except,too-many-try-statements
def get_image_analysis(data_string: NDArray[Any]) -> Tuple[str, float]:
    """Analyze an image.

    Parameters
    ----------
    data_string : NDArray[Any]
        The image data as a string.

    Returns
    -------
    Tuple[str, float]
        The label and score.
    """
    label = "unknown"
    score = 0.0
    try:
        base64_data = np.char.decode(data_string.astype("bytes"), "utf-8")
        img_data = base64.b64decode(base64_data)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
            temp_file.write(img_data)
            temp_file.flush()
            analysis_results = DeepFace.analyze(
                temp_file.name,
                actions=["emotion"],
                detector_backend=FER_MODEL_DETECTOR_BACKEND,
                enforce_detection=False,
            )
    except BaseException as exc:
        LOG.error("Error analyzing file: %s", exc)
        return label, score
    by_confidence = sorted(analysis_results, key=lambda x: x["face_confidence"], reverse=True)
    first_result = by_confidence[0]
    face_confidence = float(first_result["face_confidence"])
    if face_confidence > FER_MODEL_FACE_MIN_CONFIDENCE:
        label = first_result["dominant_emotion"]
        score = first_result["emotion"][label]
    return label, score


def fer_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]:
    """Inference function for FER model.

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
        data_string = infer_inputs[index]
        label, score = get_image_analysis(data_string)
        result = {
            "label": np.char.encode(label, "utf-8"),
            "score": np.array([score], dtype=np.float32),
        }
        results.append(result)
    return results
