"""ASR checker module."""

import logging
import re

from transformers import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from app.config import ASR_CORRECTION_MODEL, DEVICE, TORCH_DTYPE

# from transformers import AutoTokenizer, T5ForConditionalGeneration


corrector: TextGenerationPipeline | None = None

# maybe one from:
# https://huggingface.co/models?dataset=dataset:jfleg
# https://huggingface.co/models?other=punctuation
if ASR_CORRECTION_MODEL and len(ASR_CORRECTION_MODEL.split("/")) == 2:
    corrector = pipeline("text2text-generation", model=ASR_CORRECTION_MODEL, device=DEVICE, torch_dtype=TORCH_DTYPE)


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
    if not corrector or not transcript or not previous_transcript:
        return transcript
    LOG.info("Transcript: #%s#", transcript)
    LOG.info("Previous transcript: #%s#", previous_transcript)
    input_text = previous_transcript + transcript
    instruction = "Fix grammatical errors in this text: "
    prompt = f"{instruction}{input_text}"
    try:
        updated_text: str = corrector(prompt)[0]["generated_text"]
    except BaseException as error:  # pylint: disable=broad-except
        LOG.error("Error while correcting the transcript: %s", error)
        return transcript
    LOG.info("Updated text: #%s#", updated_text)
    if updated_text.startswith(instruction):
        updated_text = updated_text[len(instruction) :]
    text = updated_text[len(previous_transcript) :]
    # we might loose the space between punctuation (.,?and !) and the next word
    # so we add it back if needed
    with_spaces = re.sub(r"([.,?!])", r"\1 ", text)
    if previous_transcript and previous_transcript[-1] in [".", ",", "?", "!"] and with_spaces[0] != " ":
        # add a space after the punctuation coming from the previous transcript
        with_spaces = " " + with_spaces
    return with_spaces.replace("  ", " ")