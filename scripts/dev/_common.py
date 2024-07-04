"""Common functions for build,start,stop,deploy."""

import argparse
import os
import shutil
import subprocess  # nosemgrep # nosec
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DOT_ENV_PATH = ROOT_DIR / ".env"

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    # make sure an .env file exists
    if DOT_ENV_PATH.exists():
        load_dotenv(dotenv_path=DOT_ENV_PATH, override=True)
    else:
        DOT_ENV_EXAMPLE_PATH = ROOT_DIR / ".env.example"
        if DOT_ENV_EXAMPLE_PATH.exists():
            shutil.copyfile(DOT_ENV_EXAMPLE_PATH, DOT_ENV_PATH)
            load_dotenv(dotenv_path=DOT_ENV_PATH, override=True)
        else:
            DOT_ENV_PATH.touch()
        load_dotenv(override=True)


KEY_PREFIX = "ALIVE_MODELS"
DEFAULT_CONTAINER_NAME = os.environ.get("CONTAINER_NAME", KEY_PREFIX.lower())
#
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
_IMAGE_CHOICES = (
    "nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04",
    "nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04",
    "nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04",
    "nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04",
)
_DEFAULT_IMAGE = os.environ.get("CONTAINER_IMAGE", "localhost/alive_models")
_DEFAULT_TAG = os.environ.get("CONTAINER_TAG", None)
_DEFAULT_BASE_IMAGE = os.environ.get(f"{KEY_PREFIX}_BASE_IMAGE", _IMAGE_CHOICES[0])
_DEFAULT_PLATFORM = os.environ.get(f"{KEY_PREFIX}_PLATFORM", "linux/amd64")


def set_container_base_image(base_image: str) -> None:
    """Set the base image in the containerfile.

    On podman this:
        ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
        FROM --platform=linux/amd64 $BASE_IMAGE
    seems to not respect the --build-arg (when 'ARG' is before 'FROM')
    """
    if base_image not in _IMAGE_CHOICES:
        raise ValueError(f"Invalid base image: {base_image}")
    containerfile = ROOT_DIR / "Containerfile"
    with containerfile.open("r") as f:
        lines = f.readlines()
    with containerfile.open("w") as f:
        for line in lines:
            if line.startswith("ARG BASE_IMAGE="):
                f.write(f"ARG BASE_IMAGE={base_image}\n")
            else:
                f.write(line)


def reset_container_base_image() -> None:
    """Reset the base image in the containerfile."""
    containerfile = ROOT_DIR / "Containerfile"
    with containerfile.open("r") as f:
        lines = f.readlines()
    with containerfile.open("w") as f:
        for line in lines:
            if line.startswith("ARG BASE_IMAGE="):
                f.write(f"ARG BASE_IMAGE={_IMAGE_CHOICES[0]}\n")
            else:
                f.write(line)


def get_container_cmd() -> str:
    """Get the container command to use.

    Returns
    -------
    str
        The container command to use. Either "docker" or "podman".
    """
    from_env = os.environ.get("CONTAINER_COMMAND", "")
    if from_env and from_env in ["docker", "podman"]:
        return from_env
    # prefer podman over docker if found
    if shutil.which("podman"):
        return "podman"
    if not shutil.which("docker"):
        raise RuntimeError("Could not find docker or podman.")
    return "docker"


def run_command(cmd: List[str], cwd: Path | None = None, allow_error: bool = True, silent: bool = False) -> None:
    """Run a command.

    Parameters
    ----------
    cmd : List[str]
        The command to run.
    cwd : Path, optional
        The current working directory, by default None.
    allow_error : bool, optional
        Whether to allow errors, by default True.
    silent : bool, optional
        Whether to print what command is being run, by default False.
    """
    if not cwd:
        cwd = ROOT_DIR
    if silent is False:
        print(f"Running command: \n{' '.join(cmd)}\n")
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            env=os.environ,
            stdout=sys.stdout if silent is False else subprocess.DEVNULL,
            stderr=sys.stderr if allow_error is False else subprocess.DEVNULL,
        )  # nosemgrep # nosec
    except subprocess.CalledProcessError as error:
        if allow_error:
            return
        raise RuntimeError(f"Error running command: {error}") from error


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.

    Notes
    -----
    The following arguments are added:
    - `--image-name`: Name of the image to build.
    - `--image-tag`: Tag of the image to build.
    - `--base-image`: Base image to use.
    - `--platform`: Set platform if the image is multi-platform.
    - `--container-command`: The container command to use.
    """
    parser.add_argument(
        "--image-name",
        type=str,
        default=_DEFAULT_IMAGE,
        help="Name of the image to build.",
    )
    parser.add_argument(
        "--image-tag",
        type=str,
        default=_DEFAULT_TAG,
        help="Tag of the image to build.",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=_DEFAULT_BASE_IMAGE,
        choices=_IMAGE_CHOICES,
        help="Base image to use.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=_DEFAULT_PLATFORM,
        help="Set platform if the image is multi-platform.",
    )
    parser.add_argument(
        "--container-command",
        default=get_container_cmd(),
        choices=["docker", "podman"],
        help="The container command to use.",
    )


def get_version() -> str:
    """Read the version from ../app/version.py."""
    fallback = "latest"
    version_file = ROOT_DIR / "app" / "version.py"
    if not version_file.exists():
        return fallback
    with version_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split('"')[1]
    return fallback


def get_tag(base_image: str, tag: str | None) -> str:
    """Get the tag to use for the image (to build or run).

    If the tag is not provided, it is inferred from the base image.

    Parameters
    ----------
    base_image : str
        The base image to use.
    tag : str, optional
        The tag to use, by default None.
    """
    if not tag:
        # e.g. nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
        nv_tag = base_image.split(":")[-1]
        cuda_version = nv_tag.split("-")[0]
        return f"{get_version()}-cuda-{cuda_version}"
    return tag


def set_tag(tag: str) -> None:
    """Set the tag in the environment and in the .env file.

    Parameters
    ----------
    tag : str
        The tag to set.
    """
    # setting the env is not enough for {docker,podman} compose
    # if the key exists in .env and it is not the one we want or it is empty,
    # it will be preferred in compose.yaml
    os.environ["CONTAINER_TAG"] = tag
    # so, we also update the .env file
    with DOT_ENV_PATH.open("r") as f:
        lines = f.readlines()
    key_found = False
    with DOT_ENV_PATH.open("w") as f:
        for line in lines:
            if line.startswith("CONTAINER_TAG="):
                key_found = True
                f.write(f"CONTAINER_TAG={tag}\n")
            else:
                f.write(line)
        if not key_found:
            f.write(f"CONTAINER_TAG={tag}\n")


def _ensure_a_link_to_dot_env_in_deploy_compose_folder_exists() -> None:
    """Ensure that a link to .env in the deploy/compose folder exists."""
    compose_folder = ROOT_DIR / "deploy" / "compose"
    if not compose_folder.exists():
        return
    existing = compose_folder / ".env"
    if existing.exists():
        return
    cwd = os.getcwd()
    os.chdir(compose_folder)
    link_src = "../../.env"  # on windows?
    if os.name == "nt":
        link_src = "..\\..\\.env"
    link_dst = ".env"
    try:
        os.symlink(link_src, link_dst)
    except BaseException:  # pylint: disable=broad-except
        print(f"Could not create a link from {link_src} to {link_dst}.")
    finally:
        os.chdir(cwd)


_ensure_a_link_to_dot_env_in_deploy_compose_folder_exists()
