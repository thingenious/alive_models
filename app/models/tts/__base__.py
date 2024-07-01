"""TTS runner protocol."""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


# pylint: disable=too-few-public-methods
class RunnerProtocol(Protocol):
    """Runner protocol."""

    def __call__(self, text: str, **kwargs: Any) -> bytes | NDArray[np.float_] | None:
        """Run the model."""
