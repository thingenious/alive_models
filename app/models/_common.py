"""Common functions for models."""

import base64
import logging
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


# pylint: disable=broad-except,too-many-return-statements
def to_string(thing: Any, sep: str = "") -> str:
    """Convert a thing to a string.

    Parameters
    ----------
    thing : Any
        The thing to convert.
    sep : str
        The separator to use if the thing is a list or numpy array.

    Returns
    -------
    str
        The string representation of the thing.
    """
    if isinstance(thing, str):
        return thing
    if isinstance(thing, bytes):
        try:
            return thing.decode("utf-8")
        except BaseException:
            return ""
    if isinstance(thing, list):
        return sep.join([to_string(item, sep=sep) for item in thing])
    if isinstance(thing, np.ndarray):
        try:
            return sep.join([to_string(item, sep=sep) for item in thing.tolist()])
        except BaseException:
            return ""
    return str(thing)


def _get_wav_data(wav_data: NDArray[np.str_]) -> bytes:
    """Get the wav data.

    Parameters
    ----------
    wav_data : bytes
        The wav data.

    Returns
    -------
    bytes
        The wav data.
    """
    try:
        return base64.b64decode(wav_data)
    except BaseException:
        return b""


def concat_wav_data(wav_data1: NDArray[np.str_], wav_data2: NDArray[np.str_]) -> bytes:
    """Concatenate two wav data.

    Parameters
    ----------
    wav_data1 : bytes
        The first wav data.
    wav_data2 : bytes
        The second wav data.

    Returns
    -------
    bytes
        The concatenated wav data.
    """
    base64_data1 = _get_wav_data(wav_data1)
    base64_data2 = _get_wav_data(wav_data2)
    if not base64_data2:
        return base64_data1
    if not base64_data1:
        return base64_data2
    # pylint: disable=broad-except,too-many-try-statements
    data = []
    for wav in [base64_data1, base64_data2]:
        try:
            with BytesIO(wav) as wav_io:
                with wave.open(wav_io, "rb") as wav_in:
                    data.append((wav_in.getparams(), wav_in.readframes(wav_in.getnframes())))
        except BaseException as error:
            LOG.error("Error reading wav data: %s", error)
            return b""
    try:
        with BytesIO() as output_io:
            output = wave.open(output_io, "wb")
            output.setparams(data[0][0])
            output.writeframes(data[0][1])
            output.writeframes(data[1][1])
            output.close()
            return output_io.getvalue()
    except BaseException as error:
        LOG.error("Error reading wav data 1: %s", error)
        return b""


def save_wav_data(wav_data: bytes, data_dir: Path) -> None:
    """Save wav data to a file.

    Parameters
    ----------
    wav_data : bytes
        The wav data.
    data_dir : Path
        The data directory.
    """
    file_name = datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / file_name
    with open(file_path, "wb") as file:
        file.write(wav_data)
