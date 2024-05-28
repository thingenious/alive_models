"""NLP model inference callable."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore
from transformers import AutoTokenizer, pipeline  # noqa

from app.config import NLP_MODEL_FILE, NLP_MODEL_REPO

LOG = logging.getLogger(__name__)
NLP_INPUTS = [Tensor(name="text", dtype=bytes, shape=(1,))]
NLP_OUTPUTS = [
    Tensor(name="label", dtype=bytes, shape=(1,)),
    Tensor(name="score", dtype=np.float32, shape=(1,)),
]

model = ORTModelForSequenceClassification.from_pretrained(
    NLP_MODEL_REPO,
    file_name=NLP_MODEL_FILE,
)
tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_REPO)
classifier = pipeline(
    "text-classification",
    model=model,  # type: ignore
    tokenizer=tokenizer,
    # device=DEVICE,
    top_k=None,
    function_to_apply="sigmoid",
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


def get_text_sentiment(text: Any) -> Tuple[str, float]:
    """Get the sentiment of a text."""
    label = "unknown"
    score = 0.0
    # pylint: disable=broad-except,too-many-try-statements
    try:
        outputs = classifier(_to_string(text))
        sorted_outputs = sorted(outputs, key=lambda x: x[0]["score"], reverse=True)
        prediction: Dict[str, str | float] = sorted_outputs[0][0]
        label = str(prediction["label"])
        score = float(prediction["score"])
    except BaseException as exc:
        LOG.error("Error getting prediction: %s", exc)
    return label, score


def nlp_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]:
    """Inference function for NLP model."""
    infer_inputs = [request.data["text"] for request in requests]
    total = len(infer_inputs)
    results = []
    for index in range(total):
        input_text = infer_inputs[index]
        label, score = get_text_sentiment(input_text)
        result = {
            "label": np.char.encode(np.array(label), "utf-8"),
            "score": np.array([score], dtype=np.float32),
        }
        results.append(result)
    return results
