"""Generate requirements txt files from pyproject.toml."""

import os
import shutil
from pathlib import Path
from typing import List

import toml

PYPROJECT_TOML = Path(__file__).parent.parent.parent / "pyproject.toml"
REQUIREMENTS_DIR = Path(__file__).parent.parent.parent / "requirements"
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def write_requirements(file_name: str, dependencies: List[str]) -> None:
    """Write dependencies to a file.

    Parameters
    ----------
    file_name : str
        The name of the file to write.
    dependencies : List[str]
        The list of dependencies to write.
    """
    REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination_file = REQUIREMENTS_DIR / file_name
    with open(destination_file, "w", encoding="utf-8") as f_out:
        for dependency in dependencies:
            f_out.write(dependency + "\n")


def main() -> None:
    """Parse pyproject.toml and generate requirements."""
    if REQUIREMENTS_DIR.is_dir():
        shutil.rmtree(REQUIREMENTS_DIR)
    with open(PYPROJECT_TOML, "r", encoding="utf-8") as f_in:
        data = toml.load(f_in)
    if "project" not in data:
        print("No project found in toml file")
        return
    project_data = data["project"]
    written_files = []
    if "dependencies" in project_data:
        dependencies: List[str] = project_data["dependencies"]
        write_requirements("main.txt", sorted(dependencies))
        written_files.append("-r main.txt")
    if "optional-dependencies" in project_data:
        optional_dependencies = project_data["optional-dependencies"]
        if not isinstance(optional_dependencies, dict):
            print("Invalid optional dependencies.")
            return
        for key, deps in optional_dependencies.items():
            file_name = f"{key}.txt"
            write_requirements(file_name, sorted(deps))
            written_files.append(f"-r {file_name}")
    if len(written_files) > 1:
        write_requirements("all.txt", written_files)
        print("Generated files:")
        for item in sorted(os.listdir(REQUIREMENTS_DIR)):
            relative = os.path.join(REQUIREMENTS_DIR, item).replace(os.getcwd(), ".")
            print(f" {relative}")
    examples_txt = REQUIREMENTS_DIR / "examples.txt"
    if examples_txt.is_file():
        shutil.copy(examples_txt, EXAMPLES_DIR / "requirements.txt")


if __name__ == "__main__":
    main()
