"""Deploy the application using docker/podman compose or kubernetes.

compose: Run docker/podman compose up/down in ../deploy/compose directory.
k8s: run `helm template ...` and `kubectl apply ...` in ../deploy/k8s directory.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List

try:
    from dev._common import DOT_ENV_PATH, ROOT_DIR, add_common_args, get_container_cmd, get_tag, run_command, set_tag
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dev._common import DOT_ENV_PATH, ROOT_DIR, add_common_args, get_container_cmd, get_tag, run_command, set_tag


def get_container_compose_command() -> List[str]:
    """Get the container compose command to use.

    Check if "podman compose" or "docker compose" is available and return it.
    If not, check if "podman-compose" or "docker-compose" is available and return it.

    Returns
    -------
    List[str]
        The container compose command to use.

    Raises
    ------
    RuntimeError
        If neither "podman[-]compose" nor "docker[-]compose" is available.
    """
    container_command = get_container_cmd()
    other_command = "podman" if container_command == "docker" else "docker"
    for command in [container_command, other_command]:
        # pylint: disable=too-many-try-statements
        try:
            run_command([command, "compose", "--version"], allow_error=False, silent=True)
            return [command, "compose"]
        except RuntimeError:
            try:
                run_command([f"{command}-compose", "--version"], allow_error=False, silent=True)
                return [f"{command}-compose"]
            except RuntimeError:
                continue
    raise RuntimeError("Could not find docker-compose or podman-compose.")


def have_kubectl() -> bool:
    """Check if kubectl is available.

    Returns
    -------
    bool
        True if kubectl is available, False otherwise.
    """
    try:
        run_command(["kubectl", "version", "--client"], allow_error=False, silent=True)
    except RuntimeError:
        return False
    return True


def have_helm() -> bool:
    """Check if helm is available.

    Returns
    -------
    bool
        True if helm is available, False otherwise.
    """
    try:
        run_command(["helm", "version"], allow_error=False, silent=True)
    except RuntimeError:
        return False
    return True


def compose_up(args: argparse.Namespace) -> None:
    """Run docker or podman compose up.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.
    """
    container_command = get_container_compose_command()
    cwd = ROOT_DIR / "deploy" / "compose"
    compose_yaml = cwd / "compose.yaml"
    if not compose_yaml.exists():
        print(f"Could not find {compose_yaml}. Skipping compose up.")
        return
    tag = get_tag(args.base_image, args.image_tag)
    set_tag(tag)
    run_command(container_command + ["up", "-d"], cwd=cwd)


def compose_down(args: argparse.Namespace) -> None:
    """Run docker or podman compose down.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.
    """
    cwd = ROOT_DIR / "deploy" / "compose"
    compose_yaml = cwd / "compose.yaml"
    if not compose_yaml.exists():
        print(f"Could not find {compose_yaml}. Skipping compose down.")
        return
    container_command = get_container_compose_command()
    tag = get_tag(args.base_image, args.image_tag)
    set_tag(tag)
    run_command(container_command + ["down"], cwd=cwd)


def k8s_apply() -> None:
    """Run kubectl apply."""
    manifest_yaml = ROOT_DIR / "deploy" / "k8s" / "manifest.yaml"
    if not manifest_yaml.exists():
        print(f"Could not find {manifest_yaml}. Skipping k8s apply.")
        return
    if not have_kubectl():
        print("Could not find kubectl. Skipping k8s apply.")
        return
    namespace = os.getenv("K8S_HELM_namespace", "default")
    if namespace and namespace != "default":
        run_command(["kubectl", "create", "namespace", namespace])
    run_command(["kubectl", "apply", "-f", str(manifest_yaml)])


def k8s_delete() -> None:
    """Run kubectl delete."""
    manifest_yaml = ROOT_DIR / "deploy" / "k8s" / "manifest.yaml"
    if not manifest_yaml.exists():
        print(f"Could not find {manifest_yaml}. Skipping k8s delete.")
        return
    if not have_kubectl():
        print("Could not find kubectl. Skipping k8s delete.")
        return
    run_command(["kubectl", "delete", "-f", str(manifest_yaml)])


def get_helm_values() -> List[str]:
    """Get helm values from environment variables.

    Returns
    -------
    List[str]
        The list of helm values.
    """
    if not DOT_ENV_PATH.is_file():
        return []
    # get all the values in .env that start with K8S_HELM_
    values = []
    with open(DOT_ENV_PATH, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.startswith("K8S_HELM_"):
                key, value = line.strip().split("=", 1)
                if str(value):
                    values.extend(["--set", f"{key.replace('K8S_HELM_', '')}={value}"])
    return values


def k8s_template() -> None:
    """Run helm template."""
    chart_dir = ROOT_DIR / "deploy" / "k8s" / "alive-models"
    if not chart_dir.is_dir():
        print(f"Could not find {chart_dir}. Skipping k8s template.")
        return
    if not have_helm():
        print("Could not find helm. Skipping k8s template.")
        return
    template_name = os.environ.get("HELM_TEMPLATE_NAME", "alive")
    cmd = [
        "helm",
        "template",
        str(chart_dir),
    ]
    if template_name:
        cmd.append(f"--name-template={template_name}")
    cmd.extend(get_helm_values())
    with tempfile.TemporaryDirectory(delete=True) as tmp_folder:
        # cmd.append(tmp_folder)
        cmd.extend(["--output-dir", tmp_folder])
        run_command(cmd)
        manifest_yaml = ROOT_DIR / "deploy" / "k8s" / "manifest.yaml"
        with open(manifest_yaml, "w", encoding="utf-8") as f_out:
            for file_path in Path(tmp_folder).rglob("*.yaml"):
                with open(file_path, "r", encoding="utf-8") as f_in:
                    f_out.write(f_in.read())
    print(f"Generated {manifest_yaml}.")


def cli() -> argparse.ArgumentParser:
    """Get the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        The CLI parser.

    Notes
    -----
    The CLI parser has the following options:
    - action: The action to perform. Either "up", "down", "template", "apply", or "delete".
    The first two actions run docker/podman compose up/down in ../deploy/compose directory.
    The last ones run `helm template ...` and `kubectl apply/delete -f ...` in ../deploy/k8s directory.
    """
    parser = argparse.ArgumentParser(description="Deploy the application using docker/podman compose or kubernetes.")
    parser.add_argument(
        "action",
        choices=["up", "down", "template", "apply", "delete"],
        help=(
            "Action to perform. "
            "up: Run podman/docker[-]compose up. "
            "down: Run podman/docker[-]compose down. "
            "template: Run helm template. "
            "apply: Run kubectl apply. "
            "delete: Run kubectl delete."
        ),
    )
    add_common_args(parser)
    return parser


def main() -> None:
    """Parse arguments and run the action."""
    args, _ = cli().parse_known_args()
    if args.action == "up":
        compose_up(args=args)
    elif args.action == "down":
        compose_down(args=args)
    elif args.action == "template":
        k8s_template()
    elif args.action == "apply":
        k8s_apply()
    elif args.action == "delete":
        k8s_delete()


if __name__ == "__main__":
    main()
