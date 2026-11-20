"""Access to the reference implementation, UMEP-dev/umep-solweig, without GDAL.

The ``test`` dependency group installs umep-solweig from its repository at a
pinned commit (see ``pyproject.toml``). Its package ``__init__`` imports the
model runner, which needs GDAL, so ``import umep_solweig`` fails in this
environment. The functions the tests compare against do not need GDAL.
``use_umep_solweig`` registers ``umep_solweig`` as a bare package that has the
right path and whose ``__init__`` has not run; after that the submodules
import as usual::

    use_umep_solweig()
    from umep_solweig.functions.SOLWEIGpython.gvf_2018a import gvf_2018a

A local checkout can be used instead of the installed copy, by argument or
through the ``UMEP_SOLWEIG_CHECKOUT`` environment variable.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

CHECKOUT_ENV = "UMEP_SOLWEIG_CHECKOUT"


def _package_dir(checkout: Path | None) -> Path:
    if checkout is None and os.environ.get(CHECKOUT_ENV):
        checkout = Path(os.environ[CHECKOUT_ENV])
    if checkout is not None:
        package_dir = Path(checkout) / "src" / "umep_solweig"
        if not (package_dir / "functions").is_dir():
            raise ImportError(f"{checkout} is not a umep-solweig checkout (no src/umep_solweig/functions)")
        return package_dir
    # find_spec locates a top-level package without running its __init__.
    spec = importlib.util.find_spec("umep_solweig")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "umep-solweig is not installed (uv sync --group test) and no checkout was given "
            f"(argument or {CHECKOUT_ENV})"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def reference_module(name: str) -> types.ModuleType:
    """Import ``umep_solweig.<name>`` for a test module.

    The calling test module is skipped when umep-solweig is not installed (the
    QGIS-compatibility environment does not have it). When it is installed, a
    submodule that fails to import raises, so a renamed or removed reference
    module fails the tests instead of skipping them.

    Args:
        name: Dotted path below the package, e.g. ``functions.SOLWEIGpython.gvf_2018a``.

    Returns:
        The imported module.
    """
    import pytest

    try:
        use_umep_solweig()
    except ImportError as exc:
        pytest.skip(str(exc), allow_module_level=True)
    return importlib.import_module(f"umep_solweig.{name}")


def use_umep_solweig(checkout: Path | None = None) -> Path:
    """Make the submodules of ``umep_solweig`` importable and return the package directory.

    Args:
        checkout: A umep-solweig checkout to use instead of the installed copy.

    Returns:
        The directory of the ``umep_solweig`` package in use.

    Raises:
        ImportError: If neither an installed copy nor a checkout is available.
    """
    registered = sys.modules.get("umep_solweig")
    if registered is not None and checkout is None and getattr(registered, "__file__", None) is None:
        return Path(registered.__path__[0])
    package_dir = _package_dir(checkout)
    package = types.ModuleType("umep_solweig")
    package.__path__ = [str(package_dir)]
    sys.modules["umep_solweig"] = package
    return package_dir
