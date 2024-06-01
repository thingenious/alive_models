"""ASR checker module."""

import logging

LOG = logging.getLogger(__name__)


def correct_transcript(transcript: str, previous_transcript: str) -> str:
    """Try correcting the transcript.

    Parameters
    ----------
    transcript : str
        The transcript
    previous_transcript : str
        The previous transcript
    """
    LOG.debug("Transcript: #%s#", transcript)
    LOG.debug("Previous transcript: #%s#", previous_transcript)
    return transcript
