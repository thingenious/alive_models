"""Build docker/podman image."""

import argparse
import os
import sys
from pathlib import Path

try:
    from dev._common import ROOT_DIR, add_common_args, get_container_cmd, get_tag, run_command
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dev._common import ROOT_DIR, add_common_args, get_container_cmd, get_tag, run_command


_DEFAULT_SQUASH = os.environ.get("ALIVE_MODELS_BUILD_SQUASH", "false").lower() == "true"
_DEFAULT_PUSH = os.environ.get("ALIVE_MODELS_PUSH", "false").lower() == "true"
_DEFAULT_NO_CACHE = os.environ.get("ALIVE_MODELS_NO_CACHE", "false").lower() == "true"


def cli() -> argparse.ArgumentParser:
    """Get the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        The CLI parser.

    Notes
    -----
    The following arguments are used:
    - `--build-arg`: Build arguments.
    - `--squash`: Squash the image layers.
    - `--push`: Push the image to the registry after building.
    - `--no-cache`: Do not use cache when building the image.
    - Additional arguments added by `add_common_args`:
        - `--image-name`: Name of the image to build.
        - `--image-tag`: Tag of the image to build.
        - `--base-image`: Base image to use.
        - `--platform`: Set platform if the image is multi-platform.
        - `--container-command`: The container command to use.
    """
    parser = argparse.ArgumentParser(description="Build a podman/docker image.")
    add_common_args(parser)
    parser.add_argument(
        "--build-arg",
        type=str,
        action="append",
        help="Build arguments.",
    )
    parser.add_argument(
        "--squash",
        action="store_true",
        default=_DEFAULT_SQUASH,
        help="Squash the image layers.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=_DEFAULT_PUSH,
        help="Push the image to the registry after building.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=_DEFAULT_NO_CACHE,
        help="Do not use cache when building the image.",
    )
    return parser


def build_image(
    cwd: Path,
    containerfile: str,
    args: argparse.Namespace,
) -> str:
    """Build the image.

    Parameters
    ----------
    cwd : Path
        The current working directory.
    containerfile : str
        The containerfile to use.
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    str
        The built `image:tag`.
    """
    tag = get_tag(args.base_image, args.image_tag)
    image = args.image_name
    build_args = args.build_arg
    base_image = args.base_image
    squash = args.squash
    push = args.push
    no_cache = args.no_cache
    platform = args.platform
    build_args = build_args or []
    container_command = args.container_command or get_container_cmd()
    cmd = [
        container_command,
        "build",
        "-f",
        containerfile,
        "--tag",
        f"{image}:{tag}",
    ]
    # squash: bool = kwargs.get("squash", False)
    # push: bool = kwargs.get("push", False)
    # no_cache: bool = kwargs.get("no_cache", False)
    if squash is True:
        cmd.append("--squash")
    if no_cache is True:
        cmd.append("--no-cache")
    if platform:
        cmd.extend(["--platform", platform])
    for arg in build_args:
        cmd.extend(["--build-arg", arg])
    if "base_image" not in " ".join(build_args):
        cmd.extend(["--build-arg", f"BASE_IMAGE={base_image}"])
    cmd.append(".")
    try:
        run_command(cmd, cwd=cwd, allow_error=False, silent=False)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    if push is True:
        run_command([container_command, "push", f"{image}:{tag}"])
    return f"{image}:{tag}"


def main() -> None:
    """Get the CLI arguments and build the image."""
    args = cli().parse_args()
    containerfile = "Containerfile" if (ROOT_DIR / "Containerfile").is_file() else "Dockerfile"
    if not (ROOT_DIR / containerfile).is_file():
        raise FileNotFoundError(f"Could not find {containerfile}.")
    built_image = build_image(args=args, containerfile=containerfile, cwd=ROOT_DIR)
    print(f"Built image: {built_image}")
    print("Example usage:")
    container_cmd = get_container_cmd()
    runtime_arg = "--gpus all" if container_cmd == "docker" else "--device nvidia.com/gpu=all"
    container_name = args.image_name.replace("_", "-").split("/")[-1]
    cmd = (
        f"{container_cmd} run -it"
        f" --rm {runtime_arg}"
        f" --name {container_name}"
        f" --platform {args.platform}"
        f" -e NVIDIA_VISIBLE_DEVICES=all"
        f" -p 8000:8000 {built_image}"
    )
    print(f"  {cmd}")


if __name__ == "__main__":
    main()
