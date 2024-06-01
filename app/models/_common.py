"""Common functions for models."""

from typing import Any

import numpy as np


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
            sep.join([to_string(item, sep=sep) for item in thing.tolist()])
        except BaseException:
            return ""
    return str(thing)
