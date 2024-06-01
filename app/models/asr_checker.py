"""ASR checker module."""

import logging
import re

from transformers import pipeline
from transformers.pipelines.base import Pipeline

from app.config import ASR_CORRECTION_MODEL

# from transformers import AutoTokenizer, T5ForConditionalGeneration


corrector: Pipeline | None = None

# maybe one from:
# https://huggingface.co/models?dataset=dataset:jfleg
# https://huggingface.co/models?other=punctuation
if ASR_CORRECTION_MODEL and len(ASR_CORRECTION_MODEL.split("/")) == 2:
    corrector = pipeline("text2text-generation", model=ASR_CORRECTION_MODEL)


LOG = logging.getLogger(__name__)


def correct_transcript(transcript: str, previous_transcript: str) -> str:
    """Try correcting the transcript.

    Parameters
    ----------
    transcript : str
        The new ASR output transcript
    previous_transcript : str
        The previous ASR output transcript
    """
    if not corrector:
        return transcript
    if not transcript or not previous_transcript:
        return transcript
    LOG.debug("Transcript: #%s#", transcript)
    LOG.debug("Previous transcript: #%s#", previous_transcript)
    input_text = previous_transcript + transcript
    instruction = "Fix grammatical errors in this text: "
    prompt = f"{instruction}{input_text}"
    updated_text = corrector(prompt)[0]["generated_text"]
    LOG.debug("Updated text: #%s#", updated_text)
    if updated_text.startswith(instruction):
        updated_text = updated_text[len(instruction) :]
    text = updated_text[len(previous_transcript) :]
    # we might loose the space between punctuation (.,?and !) and the next word
    # so we add it back if needed
    with_spaces = re.sub(r"([.,?!])", r"\1 ", text)
    if previous_transcript and previous_transcript[-1] in [".", ",", "?", "!"] and with_spaces[0] != " ":
        # add a space after the punctuation coming from the previous transcript
        with_spaces = " " + with_spaces
    return with_spaces
