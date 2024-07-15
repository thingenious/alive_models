"""Start a docker/podman container."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

try:
    from dev._common import DEFAULT_CONTAINER_NAME, KEY_PREFIX, add_common_args, get_container_cmd, get_tag, run_command
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dev._common import DEFAULT_CONTAINER_NAME, KEY_PREFIX, add_common_args, get_container_cmd, get_tag, run_command


_DEFAULT_HTTP_PORT = int(os.environ.get(f"{KEY_PREFIX}_HTTP_PORT", "8000"))
_DEFAULT_GRPC_PORT = int(os.environ.get(f"{KEY_PREFIX}_GRPC_PORT", "8001"))
_DEFAULT_METRICS_PORT = int(os.environ.get(f"{KEY_PREFIX}_METRICS_PORT", "8002"))
_DEFAULT_SAGEMAKER_PORT = int(os.environ.get(f"{KEY_PREFIX}_SAGEMAKER_PORT", "8080"))

_DEFAULT_ASR_MODEL_NAME = os.environ.get(f"{KEY_PREFIX}_ASR_MODEL_NAME", "asr")
_DEFAULT_ASR_MODEL_VERSION = int(os.environ.get(f"{KEY_PREFIX}_ASR_MODEL_VERSION", "1"))
_DEFAULT_ASR_MODEL_SIZE = os.environ.get(f"{KEY_PREFIX}_ASR_MODEL_SIZE", "large-v3")

_DEFAULT_FER_MODEL_NAME = os.environ.get(f"{KEY_PREFIX}_FER_MODEL_NAME", "fer")
_DEFAULT_FER_MODEL_VERSION = int(os.environ.get(f"{KEY_PREFIX}_FER_MODEL_VERSION", "1"))
_DEFAULT_FER_MODEL_DETECTOR_BACKEND = os.environ.get(f"{KEY_PREFIX}_FER_MODEL_DETECTOR_BACKEND", "yolov8")
_DEFAULT_FER_MODEL_FACE_MIN_CONFIDENCE = float(os.environ.get(f"{KEY_PREFIX}_FER_MODEL_FACE_MIN_CONFIDENCE", "0.7"))

_DEFAULT_SER_MODEL_NAME = os.environ.get(f"{KEY_PREFIX}_SER_MODEL_NAME", "ser")
_DEFAULT_SER_MODEL_VERSION = int(os.environ.get(f"{KEY_PREFIX}_SER_MODEL_VERSION", "1"))
_DEFAULT_SER_MODEL_REPO = os.environ.get(
    f"{KEY_PREFIX}_SER_MODEL_REPO",
    "hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0",
)

_DEFAULT_NLP_MODEL_NAME = os.environ.get(f"{KEY_PREFIX}_NLP_MODEL_NAME", "nlp")
_DEFAULT_NLP_MODEL_VERSION = int(os.environ.get(f"{KEY_PREFIX}_NLP_MODEL_VERSION", "1"))
_DEFAULT_NLP_MODEL_REPO = os.environ.get(f"{KEY_PREFIX}_NLP_MODEL_REPO", "SamLowe/roberta-base-go_emotions-onnx")
_DEFAULT_NLP_MODEL_FILE = os.environ.get(f"{KEY_PREFIX}_NLP_MODEL_FILE", "onnx/model_quantized.onnx")

_DEFAULT_TTS_MODEL_NAME = os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_NAME", "tts")
_DEFAULT_TTS_MODEL_VERSION = int(os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_VERSION", "1"))
_DEFAULT_TTS_MODEL_REPO = os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_REPO", "microsoft/speecht5_tts")
_DEFAULT_TTS_MODEL_VOCODER = os.environ.get(f"{KEY_PREFIX}_TTS_VOCODER", "microsoft/speecht5_hifigan")
_XVECTORS = "Matthijs/cmu-arctic-xvectors"
_DEFAULT_TTS_MODEL_EMBEDDINGS_DATASET = os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_EMBEDDINGS_DATASET", _XVECTORS)
_DEFAULT_TTS_MODEL_RATE = os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_RATE", "+0%")
_DEFAULT_TTS_MODEL_PITCH = os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_PITCH", "+0Hz")
_DEFAULT_TTS_MODEL_VOLUME = os.environ.get(f"{KEY_PREFIX}_TTS_MODEL_VOLUME", "+0%")
_DEFAULT_TTS_AZURE_REGION = os.environ.get(f"{KEY_PREFIX}_TTS_AZURE_REGION", "")
_DEFAULT_TTS_AZURE_KEY = os.environ.get(f"{KEY_PREFIX}_TTS_AZURE_KEY", "")
_DEFAULT_TTS_ORCA_KEY = os.environ.get(f"{KEY_PREFIX}_TTS_ORCA_KEY", "")


def add_asr_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add ASR model arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.

    Notes
    -----
    The following arguments are added:
    - `--asr-model-name`: ASR model name.
    - `--asr-model-version`: ASR model version.
    - `--asr-model-size`: ASR model size.
    - `--asr-correction-model`: ASR correction model.
    """
    asr_group = parser.add_argument_group("ASR model configuration")
    asr_group.add_argument(
        "--asr-model-name",
        type=str,
        default=_DEFAULT_ASR_MODEL_NAME,
        help=f"ASR model name, (default: {_DEFAULT_ASR_MODEL_NAME}).",
    )
    asr_group.add_argument(
        "--asr-model-version",
        type=int,
        default=_DEFAULT_ASR_MODEL_VERSION,
        help=f"ASR model version, (default: {_DEFAULT_ASR_MODEL_VERSION}).",
    )
    asr_group.add_argument(
        "--asr-model-size",
        type=str,
        default=_DEFAULT_ASR_MODEL_SIZE,
        help=f"ASR model size, (default: {_DEFAULT_ASR_MODEL_SIZE}).",
    )


def add_fer_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add FER model arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.

    Notes
    -----
    The following arguments are added:
    - `--fer-model-name`: FER model name.
    - `--fer-model-version`: FER model version.
    - `--fer-model-detector-backend`: FER model detector backend.
    - `--fer-model-face-min-confidence`: FER model face min confidence.
    """
    fer_group = parser.add_argument_group("FER model configuration")
    fer_group.add_argument(
        "--fer-model-name",
        type=str,
        default=_DEFAULT_FER_MODEL_NAME,
        help=f"FER model name, (default: {_DEFAULT_FER_MODEL_NAME}).",
    )
    fer_group.add_argument(
        "--fer-model-version",
        type=int,
        default=_DEFAULT_FER_MODEL_VERSION,
        help=f"FER model version, (default: {_DEFAULT_FER_MODEL_VERSION}).",
    )
    fer_group.add_argument(
        "--fer-model-detector-backend",
        type=str,
        default=_DEFAULT_FER_MODEL_DETECTOR_BACKEND,
        help=f"FER model detector backend, (default: {_DEFAULT_FER_MODEL_DETECTOR_BACKEND}).",
    )
    fer_group.add_argument(
        "--fer-model-face-min-confidence",
        type=float,
        default=_DEFAULT_FER_MODEL_FACE_MIN_CONFIDENCE,
        help=f"FER model face min confidence, (default: {_DEFAULT_FER_MODEL_FACE_MIN_CONFIDENCE}).",
    )


def add_ser_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add SER model arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.

    Notes
    -----
    The following arguments are added:
    - `--ser-model-name`: SER model name.
    - `--ser-model-version`: SER model version.
    - `--ser-model-repo`: SER model repo.
    """
    ser_group = parser.add_argument_group("SER model configuration")
    ser_group.add_argument(
        "--ser-model-name",
        type=str,
        default=_DEFAULT_SER_MODEL_NAME,
        help=f"SER model name, (default: {_DEFAULT_SER_MODEL_NAME}).",
    )
    ser_group.add_argument(
        "--ser-model-version",
        type=int,
        default=_DEFAULT_SER_MODEL_VERSION,
        help=f"SER model version, (default: {_DEFAULT_SER_MODEL_VERSION}).",
    )
    ser_group.add_argument(
        "--ser-model-repo",
        type=str,
        default=_DEFAULT_SER_MODEL_REPO,
        help=f"SER model repo, (default: {_DEFAULT_SER_MODEL_REPO}).",
    )


def add_nlp_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add NLP model arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.

    Notes
    -----
    The following arguments are added:
    - `--nlp-model-name`: NLP model name.
    - `--nlp-model-version`: NLP model version.
    - `--nlp-model-repo`: NLP model repo.
    - `--nlp-model-file`: NLP model file.
    """
    nlp_group = parser.add_argument_group("NLP model configuration")
    nlp_group.add_argument(
        "--nlp-model-name",
        type=str,
        default=_DEFAULT_NLP_MODEL_NAME,
        help=f"NLP model name, (default: {_DEFAULT_NLP_MODEL_NAME}).",
    )
    nlp_group.add_argument(
        "--nlp-model-version",
        type=int,
        default=_DEFAULT_NLP_MODEL_VERSION,
        help=f"NLP model version, (default: {_DEFAULT_NLP_MODEL_VERSION}).",
    )
    nlp_group.add_argument(
        "--nlp-model-repo",
        type=str,
        default=_DEFAULT_NLP_MODEL_REPO,
        help=f"NLP model repo, (default: {_DEFAULT_NLP_MODEL_REPO}).",
    )
    nlp_group.add_argument(
        "--nlp-model-file",
        type=str,
        default=_DEFAULT_NLP_MODEL_FILE,
        help=f"NLP model file, (default: {_DEFAULT_NLP_MODEL_FILE}).",
    )


def add_tts_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add TTS model arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.

    Notes
    -----
    The following arguments are added:
    - `--tts-model-name`: TTS model name.
    - `--tts-model-version`: TTS model version.
    - `--tts-model-repo`: TTS model repo.
    - `--tts-model-vocoder`: TTS vocoder.
    - `--tts-model-embeddings-dataset`: TTS embeddings dataset.
    - `--tts-model-rate`: TTS rate.
    - `--tts-model-pitch`: TTS pitch.
    - `--tts-model-volume`: TTS volume.
    - `--tts-azure-region`: TTS Azure region.
    - `--tts-azure-key`: TTS Azure key.
    - `--tts-orca-key`: TTS Orca key.
    """
    tts_group = parser.add_argument_group("TTS model configuration")
    tts_group.add_argument(
        "--tts-model-name",
        type=str,
        default=_DEFAULT_TTS_MODEL_NAME,
        help=f"TTS model name, (default: {_DEFAULT_TTS_MODEL_NAME}).",
    )
    tts_group.add_argument(
        "--tts-model-version",
        type=int,
        default=_DEFAULT_TTS_MODEL_VERSION,
        help=f"TTS model version, (default: {_DEFAULT_TTS_MODEL_VERSION}).",
    )
    tts_group.add_argument(
        "--tts-model-repo",
        type=str,
        default=_DEFAULT_TTS_MODEL_REPO,
        help=f"TTS model repo, (default: {_DEFAULT_TTS_MODEL_REPO}).",
    )
    tts_group.add_argument(
        "--tts-model-vocoder",
        type=str,
        default=_DEFAULT_TTS_MODEL_VOCODER,
        help=f"TTS vocoder, (default: {_DEFAULT_TTS_MODEL_VOCODER}).",
    )
    tts_group.add_argument(
        "--tts-model-embeddings-dataset",
        type=str,
        default=_XVECTORS,
        help=f"TTS embeddings dataset, (default: {_XVECTORS}).",
    )
    tts_group.add_argument(
        "--tts-model-rate",
        type=str,
        default=_DEFAULT_TTS_MODEL_RATE,
        help=f"TTS rate, (default: {_DEFAULT_TTS_MODEL_RATE}).",
    )
    tts_group.add_argument(
        "--tts-model-pitch",
        type=str,
        default=_DEFAULT_TTS_MODEL_PITCH,
        help=f"TTS pitch, (default: {_DEFAULT_TTS_MODEL_PITCH}).",
    )
    tts_group.add_argument(
        "--tts-model-volume",
        type=str,
        default=_DEFAULT_TTS_MODEL_VOLUME,
        help=f"TTS volume, (default: {_DEFAULT_TTS_MODEL_VOLUME}).",
    )
    tts_group.add_argument(
        "--tts-azure-region",
        type=str,
        default=_DEFAULT_TTS_AZURE_REGION,
        help=f"TTS Azure region, (default: {_DEFAULT_TTS_AZURE_REGION}).",
    )
    tts_group.add_argument(
        "--tts-azure-key",
        type=str,
        default=_DEFAULT_TTS_AZURE_KEY,
        help="TTS Azure key.",
    )
    tts_group.add_argument(
        "--tts-orca-key",
        type=str,
        default=_DEFAULT_TTS_ORCA_KEY,
        help="TTS Orca key.",
    )


def cli() -> argparse.ArgumentParser:
    """Get the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        The CLI parser.

    Notes
    -----
    The following arguments are used:
    - `--container-name`: Name of the container to use.
    - `--http-port`: HTTP port to use.
    - `--grpc-port`: gRPC port to use.
    - `--metrics-port`: Metrics port to use.
    - `--sagemaker-port`: Sagemaker port to use.
    - `--debug`: Enable debug mode (verbose logging).
    - Additional arguments added by `add_common_args`:
        - `--base-image`: Base image to use.
        - `--image-tag`: Tag of the image to use.
        - `--platform`: Set platform if the image is multi-platform.
        - `--container-command`: The container command to use.
    - Additional arguments added by `add_asr_cli_args`.
    - Additional arguments added by `add_fer_cli_args`.
    - Additional arguments added by `add_ser_cli_args`.
    - Additional arguments added by `add_nlp_cli_args`.
    - Additional arguments added by `add_tts_cli_args`.
    """
    parser = argparse.ArgumentParser(description="Start a docker/podman container.")
    add_common_args(parser)
    parser.add_argument(
        "--container-name",
        type=str,
        default=DEFAULT_CONTAINER_NAME,
        help=f"Name of the container to use, (default: {DEFAULT_CONTAINER_NAME}).",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=_DEFAULT_HTTP_PORT,
        help=f"HTTP port to use, (default: {_DEFAULT_HTTP_PORT}).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=_DEFAULT_GRPC_PORT,
        help=f"gRPC port to use, (default: {_DEFAULT_GRPC_PORT}).",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=_DEFAULT_METRICS_PORT,
        help=f"Metrics port to use, (default: {_DEFAULT_METRICS_PORT}).",
    )
    parser.add_argument(
        "--sagemaker-port",
        type=int,
        default=_DEFAULT_SAGEMAKER_PORT,
        help=f"Sagemaker port to use, (default: {_DEFAULT_SAGEMAKER_PORT}).",
    )
    add_asr_cli_args(parser)
    add_fer_cli_args(parser)
    add_ser_cli_args(parser)
    add_nlp_cli_args(parser)
    add_tts_cli_args(parser)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (verbose logging).",
    )
    return parser


def get_port_args(args: argparse.Namespace) -> List[str]:
    """Get port arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of port arguments.
    """
    http_port = args.http_port
    grpc_port = args.grpc_port
    metrics_port = args.metrics_port
    sagemaker_port = args.sagemaker_port
    port_args = [
        "-p",
        f"{http_port}:{http_port}",
        "-p",
        f"{grpc_port}:{grpc_port}",
        "-p",
        f"{metrics_port}:{metrics_port}",
        "-p",
        f"{sagemaker_port}:{sagemaker_port}",
    ]
    return port_args


def get_gpu_args(container_cmd: str) -> List[str]:
    """Get gpu related arguments to use in the command.

    Parameters
    ----------
    container_cmd : str
        The container command to use (docker or podman).

    Returns
    -------
    List[str]
        The list of gpu arguments.
    """
    args = ["-e", "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all"]
    if "docker" in container_cmd:
        args.extend(["--gpus", "all"])
    elif "podman" in container_cmd:
        args.extend(["--device", "nvidia.com/gpu=all"])
    return args


def _ensure_volume(container_cmd: str, volume_name: str) -> None:
    """Ensure that the volume with the given name exists.

    Parameters
    ----------
    container_cmd : str
        The container command to use (docker or podman).
    volume_name : str
        The name of the volume to ensure.
    """
    command = [container_cmd, "volume", "create", volume_name]
    run_command(command)


def get_volume_args(container_cmd: str) -> List[str]:
    """Get volume arguments to include."""
    volumes_mapping = {
        "data_cache": "/opt/ml/data",
    }
    volume_args = []
    for volume_name, volume_path in volumes_mapping.items():
        _ensure_volume(container_cmd=container_cmd, volume_name=volume_name)
        volume_args.extend(["-v", f"{volume_name}:{volume_path}"])
    return volume_args


def get_asr_env_args(args: argparse.Namespace) -> List[str]:
    """Get ASR environment arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of environment arguments.
    """
    asr_model_name = args.asr_model_name
    asr_model_version = args.asr_model_version
    asr_model_size = args.asr_model_size
    env_args = [
        "-e",
        f"{KEY_PREFIX}_ASR_MODEL_NAME={asr_model_name}",
        "-e",
        f"{KEY_PREFIX}_ASR_MODEL_VERSION={asr_model_version}",
        "-e",
        f"{KEY_PREFIX}_ASR_MODEL_SIZE={asr_model_size}",
    ]
    return env_args


def get_fer_env_args(args: argparse.Namespace) -> List[str]:
    """Get FER environment arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of environment arguments.
    """
    fer_model_name = args.fer_model_name
    fer_model_version = args.fer_model_version
    env_args = [
        "-e",
        f"{KEY_PREFIX}_FER_MODEL_NAME={fer_model_name}",
        "-e",
        f"{KEY_PREFIX}_FER_MODEL_VERSION={fer_model_version}",
        "-e",
        f"{KEY_PREFIX}_FER_MODEL_DETECTOR_BACKEND={args.fer_model_detector_backend}",
        "-e",
        f"{KEY_PREFIX}_FER_MODEL_FACE_MIN_CONFIDENCE={args.fer_model_face_min_confidence}",
    ]
    return env_args


def get_ser_env_args(args: argparse.Namespace) -> List[str]:
    """Get SER environment arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of environment arguments.
    """
    ser_model_name = args.ser_model_name
    ser_model_version = args.ser_model_version
    ser_model_repo = args.ser_model_repo
    env_args = [
        "-e",
        f"{KEY_PREFIX}_SER_MODEL_NAME={ser_model_name}",
        "-e",
        f"{KEY_PREFIX}_SER_MODEL_VERSION={ser_model_version}",
        "-e",
        f"{KEY_PREFIX}_SER_MODEL_REPO={ser_model_repo}",
    ]
    return env_args


def get_nlp_env_args(args: argparse.Namespace) -> List[str]:
    """Get NLP environment arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of environment arguments.
    """
    nlp_model_name = args.nlp_model_name
    nlp_model_version = args.nlp_model_version
    nlp_model_repo = args.nlp_model_repo
    nlp_model_file = args.nlp_model_file
    env_args = [
        "-e",
        f"{KEY_PREFIX}_NLP_MODEL_NAME={nlp_model_name}",
        "-e",
        f"{KEY_PREFIX}_NLP_MODEL_VERSION={nlp_model_version}",
        "-e",
        f"{KEY_PREFIX}_NLP_MODEL_REPO={nlp_model_repo}",
        "-e",
        f"{KEY_PREFIX}_NLP_MODEL_FILE={nlp_model_file}",
    ]
    return env_args


def get_tts_env_args(args: argparse.Namespace) -> List[str]:
    """Get TTS environment arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of environment arguments.
    """
    env_args = [
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_NAME={args.tts_model_name}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_VERSION={args.tts_model_version}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_REPO={args.tts_model_repo}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_VOCODER={args.tts_model_vocoder}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_EMBEDDINGS_DATASET={args.tts_model_embeddings_dataset}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_RATE={args.tts_model_rate}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_PITCH={args.tts_model_pitch}",
        "-e",
        f"{KEY_PREFIX}_TTS_MODEL_VOLUME={args.tts_model_volume}",
        "-e",
        f"{KEY_PREFIX}_TTS_AZURE_REGION={args.tts_azure_region}",
    ]
    if args.tts_azure_key:
        env_args.extend(["-e", f"{KEY_PREFIX}_TTS_AZURE_KEY={args.tts_azure_key}"])
    if args.tts_orca_key:
        env_args.extend(["-e", f"{KEY_PREFIX}_TTS_ORCA_KEY={args.tts_orca_key}"])
    return env_args


def get_environment_args(args: argparse.Namespace) -> List[str]:
    """Get environment arguments to include in the command.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    List[str]
        The list of environment arguments.
    """
    http_port = args.http_port
    grpc_port = args.grpc_port
    metrics_port = args.metrics_port
    sagemaker_port = args.sagemaker_port
    prefix_models = os.environ.get(KEY_PREFIX, "")
    use_flash_attention = os.environ.get("USE_FLASH_ATTENTION", "true").lower() != "false"
    env_args = [
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        f"USE_FLASH_ATTENTION={str(use_flash_attention).lower()}",
        "-e",
        f"{KEY_PREFIX}_HTTP_PORT={http_port}",
        "-e",
        f"{KEY_PREFIX}_GRPC_PORT={grpc_port}",
        "-e",
        f"{KEY_PREFIX}_METRICS_PORT={metrics_port}",
        "-e",
        f"{KEY_PREFIX}_SAGEMAKER_PORT={sagemaker_port}",
        "-e",
        "HF_HOME=/opt/ml/data/.cache/huggingface",
        "-e",
        "TORCH_HOME=/opt/ml/data/.cache/torch",
        "-e",
        "DEEPFACE_HOME=/opt/ml/data/.cache",
    ]
    if prefix_models:
        env_args.append("-e")
        env_args.append(f"{KEY_PREFIX}={prefix_models}")
    if args.debug is True:
        env_args.append("-e")
        env_args.append("DEBUG=1")
    env_args.extend(get_asr_env_args(args))
    env_args.extend(get_fer_env_args(args))
    env_args.extend(get_ser_env_args(args))
    env_args.extend(get_nlp_env_args(args))
    env_args.extend(get_tts_env_args(args))
    return env_args


def stop_container(container_cmd: str, container_name: str) -> None:
    """Stop a container.

    Parameters
    ----------
    container_cmd : str
        The container command to use.
    container_name : str
        The name of the container to stop.
    """
    command = [container_cmd, "stop", container_name]
    run_command(command, allow_error=True)


def get_command_args(container_cmd: str, args: argparse.Namespace) -> List[str]:
    """Get command arguments to include in the command.

    Parameters
    ----------
    container_cmd : str
        The container command to use.

    Returns
    -------
    List[str]
        The list of command arguments.
    """
    cmd_args = [container_cmd, "run"]
    platform = args.platform
    if platform:
        cmd_args.extend(["--platform", platform])
    cmd_args.extend(["--rm", "-d", "--init", "--name", args.container_name])
    port_args = get_port_args(args)
    cmd_args.extend(port_args)
    gpu_args = get_gpu_args(container_cmd)
    cmd_args.extend(gpu_args)
    volume_args = get_volume_args(container_cmd)
    cmd_args.extend(volume_args)
    environment_args = get_environment_args(args)
    cmd_args.extend(environment_args)
    if args.container_command == "podman":
        # in case a container was already used (in compose?)
        cmd_args.append("--replace")
    return cmd_args


def main() -> None:
    """Parse command line args and start a container."""
    args = cli().parse_args()
    container_cmd = args.container_command or get_container_cmd()
    # stop if already started
    stop_container(container_cmd, args.container_name)
    command = get_command_args(container_cmd, args)
    image_tag = get_tag(args.base_image, args.image_tag)
    image = f"{args.image_name}:{image_tag}"
    command.append(image)
    run_command(command)
    print(f"Use (with -f to follow): \n{container_cmd} logs {args.container_name}")
    print("To view the container's logs.")


if __name__ == "__main__":
    if "--dev" in sys.argv:
        run_command([sys.executable, "-m", "app", "--debug"])
    else:
        main()
