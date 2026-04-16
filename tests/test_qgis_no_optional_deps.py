"""Confirm solweig loads and its core code paths run inside a QGIS-like
Python environment that has only numpy + osgeo, without any of the other
pip-declared runtime dependencies (rasterio, pyproj, shapely, tqdm) and
without scipy.

This regression guards the ``pip install --no-deps solweig`` install
pattern the QGIS plugin uses. If a future change reaches for one of
those packages outside of a ``try/except`` or ``if RASTERIO_AVAILABLE``
guard, this test breaks and forces the author to either (a) add a guard
or (b) document the new dependency in the plugin installer.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

_POISONED_PACKAGES = (
    "rasterio",
    "rasterio.features",
    "rasterio.mask",
    "rasterio.transform",
    "rasterio.windows",
    "rasterio.io",
    "rasterio.shutil",
    "rasterio.warp",
    "pyproj",
    "shapely",
    "shapely.geometry",
    "tqdm",
    "tqdm.auto",
    "scipy",
    "scipy.ndimage",
)


_POISONED_PREFIXES = ("rasterio", "pyproj", "shapely", "tqdm", "scipy")


@pytest.fixture()
def qgis_like_env(monkeypatch):
    """Simulate a QGIS Python environment: only numpy + osgeo available."""
    monkeypatch.setenv("UMEP_USE_GDAL", "1")

    # Provide a mock osgeo since the dev venv does not bundle real GDAL.
    mock_osgeo = MagicMock()
    monkeypatch.setitem(sys.modules, "osgeo", mock_osgeo)
    monkeypatch.setitem(sys.modules, "osgeo.gdal", mock_osgeo.gdal)
    monkeypatch.setitem(sys.modules, "osgeo.osr", mock_osgeo.osr)

    # Evict any already-imported rasterio/pyproj/shapely/tqdm/scipy submodules
    # (a prior test in the session may have pulled them in via fixtures),
    # then poison every top-level and submodule name so accidental imports
    # raise ImportError instead of silently succeeding against a cached real
    # module object.
    for k in [k for k in list(sys.modules) if any(k == p or k.startswith(p + ".") for p in _POISONED_PREFIXES)]:
        monkeypatch.delitem(sys.modules, k, raising=False)
    for name in _POISONED_PACKAGES:
        monkeypatch.setitem(sys.modules, name, None)

    # Force a fresh solweig import so it re-runs backend detection.
    for k in [k for k in sys.modules if k == "solweig" or k.startswith("solweig.")]:
        monkeypatch.delitem(sys.modules, k)

    yield


def test_import_solweig_in_qgis_like_env_without_optional_deps(qgis_like_env):
    """solweig must import cleanly under QGIS conditions with only numpy+GDAL."""
    import solweig

    assert solweig.__version__ is not None
    from solweig._compat import GDAL_AVAILABLE, GDAL_ENV, RASTERIO_AVAILABLE

    assert GDAL_ENV is True
    assert GDAL_AVAILABLE is True
    assert RASTERIO_AVAILABLE is False


def test_no_optional_deps_pulled_in_on_import(qgis_like_env):
    """Importing solweig must not load rasterio / pyproj / shapely / tqdm /
    scipy — these are either non-existent in QGIS (rasterio, tqdm) or only
    present in pip installs (scipy). Reaching for them at import time would
    break QGIS users."""
    import solweig  # noqa: F401

    for pkg in ("rasterio", "pyproj", "shapely", "tqdm", "scipy"):
        pulled_in = any(
            (k == pkg or k.startswith(pkg + "."))
            and sys.modules.get(k) is not None
            and not isinstance(sys.modules[k], MagicMock)
            for k in sys.modules
        )
        assert not pulled_in, (
            f"{pkg!r} was imported at solweig load time; this breaks the "
            f"QGIS `pip install --no-deps solweig` install pattern."
        )


def test_wall_aspect_runs_without_scipy(qgis_like_env):
    """The Goodwin wall-aspect filter runs on the Rust kernel. It must
    produce usable output in a QGIS environment where scipy is poisoned,
    because scipy is not available in QGIS's Python and solweig must
    not silently fall back to a scipy-based path."""
    from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

    dsm = np.zeros((20, 20), dtype=np.float32)
    dsm[5:15, 5:15] += 20.0

    walls = findwalls(dsm, 1.0)
    aspect = filter1Goodwin_as_aspect_v3(walls, 1.0, dsm)

    # Goodwin corners can legitimately stay at 0 (symmetric filter match) but
    # edges must be assigned. Require ≥90% of wall pixels to carry an aspect.
    assert aspect.shape == dsm.shape
    wall_mask = walls > 0
    nonzero_frac = float((aspect[wall_mask] != 0).mean())
    assert nonzero_frac >= 0.9, f"Rust wall_aspect left {(1 - nonzero_frac) * 100:.0f}% of wall pixels at zero"
