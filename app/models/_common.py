"""Common functions for models."""

import logging
import os
import subprocess  # nosemgrep # nosec
import tempfile
from typing import Any

import numpy as np

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


def to_wav(file_path: str, sample_rate: int = 16000) -> bytes:
    """Convert an audio file to wav format using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        wav_path = temp_file.name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-hide_banner",
            "-i",
            file_path,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            wav_path,
        ],
        check=True,
    )  # nosemgrep # nosec
    with open(wav_path, "rb") as file:
        wav_data = file.read()
    if os.path.exists(wav_path):
        os.remove(wav_path)
    return wav_data
