"""LID model inference callable."""

import json
import logging
from typing import Dict, List

import fasttext
import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

# pylint: disable=import-error
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

from app.config import LID_MODEL_FILE, LID_MODEL_REPO
from app.models._common import to_string

LOG = logging.getLogger(__name__)
LID_INPUTS: List[Tensor] = [Tensor(name="data", dtype=bytes, shape=(1,))]
LID_OUTPUTS: List[Tensor] = [
    Tensor(name="results", dtype=bytes, shape=(1,)),
]

model_path = hf_hub_download(repo_id=LID_MODEL_REPO, filename=LID_MODEL_FILE)
model = fasttext.load_model(model_path)


def lid_infer_fn(requests: List[Request]) -> List[Dict[str, NDArray[np.str_]]]:
    """Inference function for LID model.

    Parameters
    ----------
    requests : List[Request]
        The requests.

    Returns
    -------
    List[Dict[str, NDArray[np.str_]]]
        The identified language with its score.
    """
    speech_data = [request.data["data"] for request in requests]
    total = len(speech_data)
    results = []
    for index in range(total):
        input_text = speech_data[index]
        # pylint: disable=broad-except,too-many-try-statements
        try:
            label, score = model.predict(to_string(input_text))
            result = {
                "label": label[0].replace("__label__", ""),
                "score": min(score[0], 1.0),
            }
        except BaseException as error:
            LOG.error("Error processing request: %s", error)
            results.append({"results": np.array([], dtype=bytes)})
            continue
        result = {
            "results": np.array([json.dumps(result)], dtype=bytes),
        }
        results.append(result)
    return results
