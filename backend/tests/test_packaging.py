"""The two packaging claims this repository makes about itself in comments.

Both were written as though something checked them and nothing did. `pyproject.toml` says the
trailing star in `include = ["app*"]` is "pinned by tests/test_packaging.py" and the Dockerfile says
this file "asserts the two match"; the file did not exist. So the dependency list was maintained by
hand in two places with no guard, and the star - the character whose loss ships a wheel holding two
modules - was pinned by a comment about a test.

Read off disk rather than through an import. The Dockerfile is not Python and `pyproject.toml` is
read by the build backend, not by the application, so neither is reachable from an import; and an
import-based check on the package layout would pass for the wrong reason, exactly as
`tests/test_api.py` explains for the services layer.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PYPROJECT = BACKEND_DIR / "pyproject.toml"
DOCKERFILE = BACKEND_DIR / "Dockerfile"

# The subpackages a wheel built from `include = ["app"]` would silently leave out. `app/config.py`
# is a module rather than a package, which is why it survives that mistake and these do not - and
# why the resulting wheel looks like a working install right up to the first router import.
APP_SUBPACKAGES = ("app.api", "app.services", "app.repositories", "app.serving", "app.schemas")


def _distribution(requirement: str) -> str:
    """`"uvicorn[standard]>=0.30"` -> `uvicorn`. Extras and version specifiers are not the name."""
    return re.split(r"[\[<>=!~;]", requirement.strip().strip('"').strip("'"), maxsplit=1)[0].strip()


def _pyproject_dependencies() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return {_distribution(entry) for entry in data["project"]["dependencies"]}


def _dockerfile_dependencies() -> set[str]:
    """Package names out of the image's `pip install`, with its line continuations joined."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    joined = re.sub(r"\\\n\s*", " ", text)
    installs = [line for line in joined.splitlines() if re.match(r"\s*RUN\s+pip\s+install", line)]
    assert installs, "the Dockerfile no longer installs anything with pip"

    names: set[str] = set()
    for line in installs:
        # Quoted requirements only. Everything else on the line is a flag or its value, and
        # `--extra-index-url https://...` is exactly the thing a looser parse would swallow.
        names.update(_distribution(match) for match in re.findall(r'"([^"]+)"', line))
    return names


def test_the_image_installs_exactly_what_the_project_declares() -> None:
    """The two lists are maintained by hand in two files, and have drifted before.

    A package in `pyproject.toml` and not in the image is an ImportError at container start, after
    the build succeeded. A package in the image and not in `pyproject.toml` is a dependency the
    tests never exercise, because the test environment is built from `pyproject.toml`.
    """
    declared = _pyproject_dependencies()
    installed = _dockerfile_dependencies()
    assert installed == declared, (
        f"only in the Dockerfile: {sorted(installed - declared)}; "
        f"only in pyproject.toml: {sorted(declared - installed)}"
    )


def test_the_wheel_would_hold_every_subpackage() -> None:
    """The star in `include = ["app*"]` is load-bearing and now it is checked.

    A literal `["app"]` matches the top-level package and nothing below it, so the wheel ships
    `app/__init__.py` and `app/config.py` and the container fails on the first import of a router.
    The failure is at run time, in an image that built cleanly.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["app*"], f"packages.find.include is {include}, expected ['app*']"

    # And the pattern has to be load-bearing rather than decorative: these subpackages exist, so
    # a literal ["app"] really would drop them.
    for module in APP_SUBPACKAGES:
        path = BACKEND_DIR / Path(*module.split(".")) / "__init__.py"
        assert path.exists(), f"{module} is not a package, so this test is checking the wrong list"

    # Every package under app/ has to be in that list, or a subpackage added later goes unguarded.
    on_disk = {
        f"app.{path.parent.name}"
        for path in (BACKEND_DIR / "app").glob("*/__init__.py")
        if not path.parent.name.startswith("_")
    }
    assert on_disk == set(APP_SUBPACKAGES), f"app/ holds {sorted(on_disk)}"
