"""Set version.

Usage:

    python scripts/dev/on_version.py               => get the version (app/version.py) and update the files that use it
    python scripts/dev/on_version.py 0.0.2         => set version to 0.0.2
    python scripts/dev/on_version.py bump          => bump the version (if it's 0.0.1, it will be 0.0.2)
    python scripts/dev/on_version.py bump --minor  => bump the version (if it's 0.0.12, it will be 0.1.0)
    python scripts/dev/on_version.py bump --major  => bump the version (if it's 1.5.63, it will be 2.0.0)
"""

import sys
from pathlib import Path
from typing import Tuple

try:
    from scripts.dev._common import ROOT_DIR, get_version, run_command
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.dev._common import ROOT_DIR, get_version, run_command


FILES_TO_UPDATE = [
    # in repo:
    ".env.example",
    "deploy/k8s/alive-models/Chart.yaml",
    # optional/.gitignored:
    ".env",
]
# we might also need to re-generate the manifest.example.yaml in ./deploy/k8s
# e.g.: image: localhost:5000/alive_models:0.0.1-cuda-12.4.1

PREFIXES_TO_LOOKUP = [
    # e.g.: K8S_HELM_deployment.image.tag=0.0.1-cuda-12.4.1
    "K8S_HELM_deployment.image.tag=",
    # e.g.: CONTAINER_TAG=0.0.1-cuda-12.4.1
    "CONTAINER_TAG=",
    # in chart.yaml:
    # version: 0.0.1
    # appVersion: "0.0.1"
    "version: ",
    "appVersion: ",
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
    # e.g. old tag: 0.0.1 => new tag: 0.0.2
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            for prefix in PREFIXES_TO_LOOKUP:
                if line.startswith(prefix):
                    # keep the part after the tag e.g. (cuda-12.4.1):
                    # CONTAINER_TAG=0.0.1-cuda-12.4.1 => CONTAINER_TAG=0.0.2-cuda-12.4.1
                    without_prefix = line.split(prefix)[1]
                    old_tag = without_prefix.split("-")[0]
                    if old_tag.endswith("\n"):
                        old_tag = old_tag[: -len("\n")]
                    # if the old tag is surrounded by quotes, keep them:
                    # appVersion: "0.0.1" => appVersion: "0.0.2"
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
    version_digits = existing_version.split(".")
    for i in range(digit_to_inc, 3):
        version_digits[i] = "0"
    version_digits[digit_to_inc] = str(int(version_digits[digit_to_inc]) + 1)
    return ".".join(version_digits)


def check_version() -> Tuple[str, bool]:
    """Check if version is provided as an argument."""
    existing_version = get_version()
    if len(sys.argv) == 2:
        version_arg = sys.argv[1]
        if version_arg == "bump":
            return get_new_version(existing_version), True
        # check if the version is valid (e.g. 0.0.1)
        if not version_arg.count(".") == 2:
            raise ValueError("Invalid version format.")
        version_digits = version_arg.split(".")
        for digit in version_digits:
            if not digit.isdigit():
                raise ValueError("Invalid version format.")
        return version_arg, existing_version != version_arg
    return get_version(), True


def make_k8s_template() -> None:
    """Generate a new manifest.example.yaml."""
    cmd = [
        sys.executable,
        "scripts/dev/deploy.py",
        "template",
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
    make_k8s_template()


if __name__ == "__main__":
    main()
