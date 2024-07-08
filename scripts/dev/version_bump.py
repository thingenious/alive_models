"""Set version.

Usage:

    python scripts/dev/on_version.py               => get the version (app/version.py) and update the files that use it
    python scripts/dev/on_version.py 0.1.2         => set version to 0.1.2
    python scripts/dev/on_version.py --patch       => bump the version (if it's 0.2.12, it will be 0.2.13)
    python scripts/dev/on_version.py --minor        => bump the version (if it's 0.1.12, it will be 0.2.0)
    python scripts/dev/on_version.py --major  => bump the version (if it's 1.5.63, it will be 2.0.0)
"""

import os
import sys
from pathlib import Path
from typing import Tuple

try:
    from dev._common import ROOT_DIR, get_version, run_command
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dev._common import ROOT_DIR, get_version, run_command


FILES_TO_UPDATE = [
    # in repo:
    ".env.example",
    "deploy/k8s/alive-models/Chart.yaml",
    "deploy/k8s/alive-models/values.yaml",
    # optional/.gitignored:
    ".env",
]
# we might also need to re-generate the manifest.example.yaml in ./deploy/k8s
# e.g.: image: localhost:5000/alive_models:1.0.1-cuda-12.5.0

PREFIXES_TO_LOOKUP = [
    # e.g.: K8S_HELM_deployment_image_tag=1.0.1-cuda-12.5.0
    "K8S_HELM_deployment_image_tag=",
    # e.g.: CONTAINER_TAG=1.0.1-cuda-12.5.0
    "CONTAINER_TAG=",
    # in chart.yaml:
    # version: 2.0.1
    # appVersion: "2.0.1"
    "version: ",
    "appVersion: ",
    # in deploy/k8s/alive-models/values.yaml:"
    #     tag: 1.0.1-cuda-12.5.0"
    "    tag: ",
]


def update_version_py(to: str) -> None:
    """Update version in version.py."""
    version_file = ROOT_DIR / "app" / "version.py"
    with open(version_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(version_file, "w", encoding="utf-8") as file:
        for line in lines:
            if line.startswith("__version__"):
                file.write(f'__version__ = "{to}"\n')
            else:
                file.write(line)


def update_tag(file_path: Path, tag: str) -> None:
    """Update tag in file."""
    # e.g. old tag: 0.1.1-cuda-12.5.0 -> new tag: 0.1.2-cuda-12.5.0
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            for prefix in PREFIXES_TO_LOOKUP:
                if line.startswith(prefix) and line.strip() != prefix:  # no empty values
                    # keep the part after the tag e.g. (cuda-12.5.0):
                    # CONTAINER_TAG=1.0.1-cuda-12.5.0 => CONTAINER_TAG=1.0.2-cuda-12.5.0
                    without_prefix = line.split(prefix)[1]
                    old_tag = without_prefix.split("-")[0]
                    if old_tag.endswith("\n"):
                        old_tag = old_tag[: -len("\n")]
                    # if the old tag is surrounded by quotes, keep them:
                    # appVersion: "1.0.1" => appVersion: "1.0.2"
                    if old_tag.startswith('"') and old_tag.endswith('"'):
                        new_tag = f'"{tag}"'
                    else:
                        new_tag = tag
                    # print(f"file: {file_path},\nprefix: {prefix}, old tag: {old_tag} => new tag: {new_tag}")
                    line = line.replace(f"{prefix}{old_tag}", f"{prefix}{new_tag}")
                    break
            file.write(line)


def get_new_version(existing_version: str) -> str:
    """Bump version."""
    # examples:
    #    0.0.4 => 0.0.5
    # or 0.1.3 => 0.2.0
    # or 1.2.3 => 2.0.0
    digit_to_inc = 2
    if "--minor" in sys.argv:
        digit_to_inc = 1
    elif "--major" in sys.argv:
        digit_to_inc = 0
    version_digits = []
    version_parts = existing_version.split(".")
    for index, digit in enumerate(version_parts):
        if not digit.isdigit():
            raise ValueError("Invalid version format.")
        if not index == digit_to_inc:
            version_digits.append(digit)
        else:
            version_digits.append(str(int(digit) + 1))
    return ".".join(version_digits)


def get_cuda_version() -> str:
    """Get CUDA version from the BASE_IMAGE."""
    base_image = os.environ.get("ALIVE_MODELS_BASE_IMAGE", "nvcr.io/nvidia/cuda:12.5.0-devel-ubuntu22.04")
    cuda_version = base_image.split(":")[1].split("-")[0]
    return cuda_version


def check_version() -> Tuple[str, bool]:
    """Check if version is provided as an argument."""
    existing_version = get_version()
    if len(sys.argv) == 2:
        version_arg = sys.argv[1]
        if version_arg in ("--patch", "--minor", "--major"):
            new_version = get_new_version(existing_version)
            return new_version, True
        # check if the version is valid (e.g. 1.0.1)
        if not version_arg.count(".") == 2:
            raise ValueError("Invalid version format.")
        version_digits = version_arg.split(".")
        for digit in version_digits:
            if not digit.isdigit():
                raise ValueError("Invalid version format.")
        return version_arg, existing_version != version_arg
    return get_version(), True


def update_values_yaml(tag: str) -> None:
    """Update tag in values.yaml."""
    values_yaml = ROOT_DIR / "deploy" / "k8s" / "alive-models" / "values.yaml"
    with open(values_yaml, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(values_yaml, "w", encoding="utf-8") as file:
        for line in lines:
            if line.startswith("    tag:"):
                line = f"     tag: {tag}\n"
            file.write(line)


def make_k8s_template(file_path: Path) -> None:
    """Generate a new manifest.example.yaml."""
    cmd = [
        sys.executable,
        "scripts/dev/deploy.py",
        "template",
        "--output-file",
        str(file_path),
    ]
    run_command(cmd, cwd=ROOT_DIR)


def main() -> None:
    """Set version."""
    tag, update_py = check_version()
    if update_py is True:
        update_version_py(tag)
    for file in FILES_TO_UPDATE:
        file_path = ROOT_DIR / file
        if not file_path.is_file():
            continue
        update_tag(file_path, tag)
    make_k8s_template(file_path=ROOT_DIR / "deploy" / "k8s" / "manifest.example.yaml")


if __name__ == "__main__":
    main()
