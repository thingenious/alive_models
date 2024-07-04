"""NLP model inference callable."""

import json
import logging
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore
from transformers import AutoTokenizer, pipeline  # noqa

from app.config import NLP_MODEL_FILE, NLP_MODEL_REPO
from app.models._common import to_string

LOG = logging.getLogger(__name__)
NLP_INPUTS = [Tensor(name="data", dtype=bytes, shape=(1,))]
NLP_OUTPUTS = [
    Tensor(name="results", dtype=bytes, shape=(1,)),
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
    top_k=None,
    function_to_apply="sigmoid",
)


def get_text_sentiment(text: str) -> List[Dict[str, str | float]]:
    """Get the sentiment of a text.

    Parameters
    ----------
    text : str
        The text to analyze.

    Returns
    -------
    List[Tuple[str, float]]
        The sentiment labels and scores
    """
    try:
        outputs = classifier(text)
    except BaseException as exc:  # pylint: disable=broad-except
        LOG.error("Error getting prediction: %s", exc)
        outputs = []
    LOG.debug("Outputs: %s", outputs)
    return outputs[0]


def nlp_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.int_] | NDArray[np.float_]]]:
    """Inference function for NLP model.

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
        input_text = infer_inputs[index]
        analysis_results = get_text_sentiment(to_string(input_text))
        result = {
            "results": np.array([json.dumps(analysis_results)], dtype=bytes),
        }
        results.append(result)
    return results
