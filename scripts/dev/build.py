"""Build docker/podman image."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import toml

try:
    from dev._common import (
        ROOT_DIR,
        add_common_args,
        get_container_cmd,
        get_tag,
        reset_container_base_image,
        run_command,
        set_container_base_image,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dev._common import (
        ROOT_DIR,
        add_common_args,
        get_container_cmd,
        get_tag,
        reset_container_base_image,
        run_command,
        set_container_base_image,
    )


_DEFAULT_SQUASH = os.environ.get("ALIVE_MODELS_BUILD_SQUASH", "false").lower() == "true"
_DEFAULT_PUSH = os.environ.get("ALIVE_MODELS_PUSH", "false").lower() == "true"
_DEFAULT_NO_CACHE = os.environ.get("ALIVE_MODELS_NO_CACHE", "false").lower() == "true"
_IMAGE_TITLE = "ALIVE Models"


def get_image_repo(toml_dict: Dict[str, Any]) -> str:
    """Get the image repository."""
    urls = toml_dict.get("project", {}).get("urls", {})
    repo_url = urls.get("repository", "http://localhost:5000/alive_models.git")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    if repo_url.startswith("ssh://"):
        repo_url = f"https://{repo_url[6:]}"
    return repo_url


def get_image_description(toml_dict: Dict[str, Any]) -> str:
    """Get the image description."""
    project = toml_dict.get("project", {})
    description = project.get("description", "ALIVE Models")
    return description


def get_image_authors(toml_dict: Dict[str, Any]) -> str:
    """Get the image authors."""
    authors = toml_dict.get("project", {}).get("authors", [])
    authors = [f"{author['name']} <{author['email']}>" for author in authors]
    return ", ".join(authors)


def get_image_licenses(toml_dict: Dict[str, Any]) -> str:
    """Get the image licenses."""
    licenses = toml_dict.get("project", {}).get("license", {}).get("text", "MIT")
    return licenses


def get_container_labels() -> Dict[str, str]:
    """Get the container labels."""
    toml_dict = toml.load(ROOT_DIR / "pyproject.toml")
    repo = get_image_repo(toml_dict)
    description = get_image_description(toml_dict)
    authors = get_image_authors(toml_dict)
    licenses = get_image_licenses(toml_dict)
    return {
        "org.opencontainers.image.licenses": licenses,
        "org.opencontainers.image.source": repo,
        "org.opencontainers.image.title": _IMAGE_TITLE,
        "org.opencontainers.image.description": description,
        "org.opencontainers.image.authors": authors,
    }


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
    set_container_base_image(base_image)
    for key, value in get_container_labels().items():
        cmd.extend(["--label", f"{key}={value}"])
    try:
        run_command(cmd, cwd=cwd, allow_error=False, silent=False)
    except RuntimeError as e:
        print(e)
        reset_container_base_image()
        sys.exit(1)
    reset_container_base_image()
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
