"""Stop the docker container if it exists."""

import argparse
import sys
from pathlib import Path

try:
    from dev._common import DEFAULT_CONTAINER_NAME, get_container_cmd, run_command
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dev._common import DEFAULT_CONTAINER_NAME, get_container_cmd, run_command


def main() -> None:
    """Run the main function."""
    default_container_command = get_container_cmd()
    parser = argparse.ArgumentParser(description="Stop the model docker container if it exists.")
    parser.add_argument(
        "--container-command",
        default=default_container_command,
        choices=["docker", "podman"],
        help="The container command to use.",
    )
    parser.add_argument(
        "--container-name",
        default=DEFAULT_CONTAINER_NAME,
        help=f"The container name to stop (default: {DEFAULT_CONTAINER_NAME}).",
    )
    args = parser.parse_args()
    container_command = args.container_command or get_container_cmd()
    run_command([container_command, "stop", args.container_name], allow_error=True)


if __name__ == "__main__":
    main()
