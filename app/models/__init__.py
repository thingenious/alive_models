"""Models to serve."""

import sys
from pathlib import Path
from typing import Callable, Dict, List

# pylint: disable=import-error,wrong-import-position,unused-import
# pyright: reportMissingImports=false
from pytriton.model_config import Tensor  # type: ignore
from pytriton.proxy.types import Request  # type: ignore

# pyright: reportUnusedImport=false
try:
    import app.config  # noqa
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import app.config  # noqa

from app.config import (
    ASR_MODEL_NAME,
    ASR_MODEL_VERSION,
    FER_MODEL_NAME,
    FER_MODEL_VERSION,
    LID_MODEL_NAME,
    LID_MODEL_VERSION,
    MODELS_TO_LOAD,
    NLP_MODEL_NAME,
    NLP_MODEL_VERSION,
    SER_MODEL_NAME,
    SER_MODEL_VERSION,
    TTS_MODEL_NAME,
    TTS_MODEL_VERSION,
)


# pylint: disable=too-few-public-methods
class ModelToLoad:
    """Model to load configuration."""

    def __init__(
        self,
        name: str,
        version: int,
        infer_fn: Callable[[List[Request]], List[Dict[str, Tensor]]],
        inputs: List[Tensor],
        outputs: List[Tensor],
    ) -> None:
        """Initialize the model configuration."""
        self.name = name
        self.version = version
        self.infer_fn = infer_fn
        self.inputs = inputs
        self.outputs = outputs


MODELS: List[ModelToLoad] = []

if "asr" in MODELS_TO_LOAD:
    from .asr import ASR_INPUTS, ASR_OUTPUTS, asr_infer_fn

    MODELS.append(
        ModelToLoad(
            name=ASR_MODEL_NAME,
            version=ASR_MODEL_VERSION,
            infer_fn=asr_infer_fn,
            inputs=ASR_INPUTS,
            outputs=ASR_OUTPUTS,
        )
    )

if "fer" in MODELS_TO_LOAD:
    from .fer import FER_INPUTS, FER_OUTPUTS, fer_infer_fn

    MODELS.append(
        ModelToLoad(
            name=FER_MODEL_NAME,
            version=FER_MODEL_VERSION,
            infer_fn=fer_infer_fn,
            inputs=FER_INPUTS,
            outputs=FER_OUTPUTS,
        )
    )

if "ser" in MODELS_TO_LOAD:
    from .ser import SER_INPUTS, SER_OUTPUTS, ser_infer_fn

    MODELS.append(
        ModelToLoad(
            name=SER_MODEL_NAME,
            version=SER_MODEL_VERSION,
            infer_fn=ser_infer_fn,
            inputs=SER_INPUTS,
            outputs=SER_OUTPUTS,
        )
    )


if "nlp" in MODELS_TO_LOAD:
    from .nlp import NLP_INPUTS, NLP_OUTPUTS, nlp_infer_fn

    MODELS.append(
        ModelToLoad(
            name=NLP_MODEL_NAME,
            version=NLP_MODEL_VERSION,
            infer_fn=nlp_infer_fn,
            inputs=NLP_INPUTS,
            outputs=NLP_OUTPUTS,
        )
    )

if "tts" in MODELS_TO_LOAD:
    from .tts import TTS_INPUTS, TTS_OUTPUTS, tts_infer_fn

    MODELS.append(
        ModelToLoad(
            name=TTS_MODEL_NAME,
            version=TTS_MODEL_VERSION,
            infer_fn=tts_infer_fn,
            inputs=TTS_INPUTS,
            outputs=TTS_OUTPUTS,
        )
    )

if "lid" in MODELS_TO_LOAD:
    from .lid import LID_INPUTS, LID_OUTPUTS, lid_infer_fn

    MODELS.append(
        ModelToLoad(
            name=LID_MODEL_NAME,
            version=LID_MODEL_VERSION,
            infer_fn=lid_infer_fn,
            inputs=LID_INPUTS,
            outputs=LID_OUTPUTS,
        )
    )


if not MODELS:
    raise ValueError("No models to load")

__all__ = ["MODELS"]
