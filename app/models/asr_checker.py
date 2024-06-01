"""ASR checker module."""

import logging
import os
import shutil
from typing import Dict, Type

import faster_whisper
from huggingface_hub import snapshot_download

from app.config import ASR_CORRECTION_MODEL, DEVICE, ROOT_DIR

LOG = logging.getLogger(__name__)


def _patch_neuspell() -> None:
    """Patch neuspell to avoid `unexpected key error thrown by pytorch`."""
    # in ...../site-packages/neuspell/seq_modeling/subwordbert.py
    # remove "bert_model.embeddings.position_ids" from checkpoint_data and model_state_dict (if they exist)
    site_packages_faster_whisper = faster_whisper.__path__[0]
    site_packages = os.path.dirname(site_packages_faster_whisper)
    dir_to_check = os.path.join(site_packages, "neuspell", "seq_modeling")
    file_to_patch = os.path.join(dir_to_check, "subwordbert.py")
    if not os.path.exists(file_to_patch):
        print(f"File not found: {file_to_patch}")
        return
    bak_file = file_to_patch + ".bak"
    if os.path.exists(bak_file):
        # already patched
        return

    with open(file_to_patch, "r", encoding="utf-8") as file:
        lines = file.readlines()
    with open(file_to_patch, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line)
            if "checkpoint_data = torch.load" in line:
                file.write('    if "bert_model.embeddings.position_ids" in checkpoint_data:\n')
                file.write('        checkpoint_data.pop("bert_model.embeddings.position_ids")\n')
                file.write('    if "bert_model.embeddings.position_ids" in  checkpoint_data["model_state_dict"]:\n')
                file.write('        checkpoint_data["model_state_dict"].pop("bert_model.embeddings.position_ids")\n')

    shutil.copyfile(file_to_patch, bak_file)


_patch_neuspell()
# pylint: disable=import-outside-toplevel, wrong-import-position,wrong-import-order
from neuspell import BertChecker, SclstmChecker  # noqa
from neuspell.commons import Corrector  # noqa

# https://github.com/neuspell/neuspell
# without elmo,
# from them, only the ones in hf:
# https://github.com/neuspell/neuspell/issues/77#issuecomment-1280713577
#
# https://huggingface.co/models?other=spell-correction
# pszemraj/neuspell-subwordbert-probwordnoise
# pszemraj/neuspell-scrnn-probwordnoise
checker_cls: Dict[str, Type[Corrector]] = {
    "scrnn-probwordnoise": SclstmChecker,
    "subwordbert-probwordnoise": BertChecker,
}


checker: Corrector | None = None
if ASR_CORRECTION_MODEL in checker_cls:
    # checker = checker_cls[ASR_CORRECTION_MODEL](pretrained=True, device=DEVICE)
    # checker.from_pretrained()
    checker = checker_cls[ASR_CORRECTION_MODEL](device=DEVICE)
    # pszemraj/neuspell-scrnn-probwordnoise
    repo_id = f"pszemraj/neuspell-{ASR_CORRECTION_MODEL}"
    local_dir = ROOT_DIR / "data" / ".cache" / "neuspell" / ASR_CORRECTION_MODEL
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
    checker.from_pretrained(local_dir)


def correct_transcript(transcript: str, previous_transcript: str) -> str:
    """Try correcting the transcript.

    Parameters
    ----------
    transcript : str
        The transcript
    previous_transcript : str
        The previous transcript
    """
    LOG.debug("transcript: %s", transcript)
    LOG.debug("previous_transcript: %s", previous_transcript)
    if not checker:
        return transcript
    if not previous_transcript:
        return transcript
    combined = previous_transcript + transcript
    corrected = checker.correct_string(combined)
    LOG.debug("corrected: %s", corrected)
    return corrected[len(previous_transcript) :]
