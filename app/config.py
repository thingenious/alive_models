"""Models, names, versions, inputs and outputs configuration."""

import os
import sys
from pathlib import Path

import torch

DEBUG = os.getenv("DEBUG", "false").lower() in (
    "true",
    "1",
    "on",
    "yes",
    "y",
    "t",
)
if not DEBUG and "--debug" in sys.argv or "--log-verbose" in sys.argv:
    DEBUG = True

ROOT_DIR = Path(__file__).parent.parent.resolve()
_have_cuda = torch.cuda.is_available()
DEVICE = "cuda" if _have_cuda else "cpu"
COMPUTE_TYPE = "float16" if _have_cuda else "float32"
TORCH_DTYPE = torch.float16 if _have_cuda else torch.float32
USE_FLASH_ATTENTION = _have_cuda

__all__ = [
    "DEBUG",
    "DEVICE",
    "ROOT_DIR",
    "COMPUTE_TYPE",
    "TORCH_DTYPE",
    "FER_MODEL_NAME",
    "FER_MODEL_VERSION",
    "ASR_MODEL_NAME",
    "ASR_MODEL_VERSION",
    "ASR_MODEL_SIZE",
    "SER_MODEL_NAME",
    "SER_MODEL_VERSION",
    "SER_MODEL_REPO",
    "NLP_MODEL_NAME",
    "NLP_MODEL_VERSION",
    "NLP_MODEL_REPO",
    "NLP_MODEL_FILE",
    "USE_FLASH_ATTENTION",
]


def _set_cache_dir() -> None:
    """Set the cache directory."""
    cache_dir = ROOT_DIR / "data" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def _set_hf_home() -> None:
    """Set the Hugging Face home directory."""
    data_dir = ROOT_DIR / "data"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_home = data_dir / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)


def _set_torch_home() -> None:
    """Set the Torch home directory."""
    torch_home = ROOT_DIR / "data" / ".cache" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)


def _set_deepface_home() -> None:
    """Set the DeepFace home directory."""
    deepface_home = ROOT_DIR / "data" / ".cache"
    deepface_home.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPFACE_HOME"] = str(deepface_home)


_set_cache_dir()
_set_hf_home()
_set_torch_home()
_set_deepface_home()

ENV_PREFIX = "ALIVE_MODELS"

MODELS_STR = os.getenv(ENV_PREFIX, "")
if not MODELS_STR:
    MODELS_STR = "asr,fer,ser,nlp"

MODELS_TO_LOAD = MODELS_STR.split(",")
if not MODELS_TO_LOAD:
    MODELS_TO_LOAD = ["asr", "fer", "ser", "nlp"]

for model in MODELS_TO_LOAD:
    if model not in ["asr", "fer", "ser", "nlp"]:
        raise ValueError(f"Invalid model: {model}")

# ASR
ASR_MODEL_NAME = os.getenv(f"{ENV_PREFIX}_ASR_MODEL_NAME", "asr")
ASR_MODEL_VERSION = int(os.getenv(f"{ENV_PREFIX}_ASR_MODEL_VERSION", "1"))
_ASR_MODEL_SIZE = "large-v3"
ASR_MODEL_SIZE = os.getenv(f"{ENV_PREFIX}_ASR_MODEL_SIZE", "")
if not ASR_MODEL_SIZE:
    ASR_MODEL_SIZE = _ASR_MODEL_SIZE

# FER
FER_MODEL_DETECTOR_BACKEND = os.getenv(f"{ENV_PREFIX}_FER_MODEL_DETECTOR_BACKEND", "yolov8")
FER_MODEL_FACE_MIN_CONFIDENCE = float(os.getenv(f"{ENV_PREFIX}_FER_MODEL_FACE_MIN_CONFIDENCE", "0.7"))
FER_MODEL_NAME = os.getenv(f"{ENV_PREFIX}_FER_MODEL_NAME", "fer")
FER_MODEL_VERSION = int(os.getenv(f"{ENV_PREFIX}_FER_MODEL_VERSION", "1"))

# SER
SER_MODEL_NAME = os.getenv(f"{ENV_PREFIX}_SER_MODEL_NAME", "ser")
SER_MODEL_VERSION = int(os.getenv(f"{ENV_PREFIX}_SER_MODEL_VERSION", "1"))
_SER_MODEL_REPO = "hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0"
SER_MODEL_REPO = os.environ.get(f"{ENV_PREFIX}_SER_MODEL_REPO", "")
if not SER_MODEL_REPO:
    SER_MODEL_REPO = _SER_MODEL_REPO

# NLP
NLP_MODEL_NAME = os.getenv(f"{ENV_PREFIX}_NLP_MODEL_NAME", "nlp")
NLP_MODEL_VERSION = int(os.getenv(f"{ENV_PREFIX}_NLP_MODEL_VERSION", "1"))
_NLP_MODEL_REPO = "SamLowe/roberta-base-go_emotions-onnx"
NLP_MODEL_REPO = os.environ.get(f"{ENV_PREFIX}_NLP_MODEL_REPO", "")
if not NLP_MODEL_REPO:
    NLP_MODEL_REPO = _NLP_MODEL_REPO
_NLP_MODEL_FILE = "onnx/model_quantized.onnx"
NLP_MODEL_FILE = os.environ.get(f"{ENV_PREFIX}_NLP_MODEL_FILE", "")
if not NLP_MODEL_FILE:
    NLP_MODEL_FILE = _NLP_MODEL_FILE
