"""Keep the two geospatial backends apart while the tests run.

Inside QGIS the package must work on QGIS's own GDAL alone. ``rasterio``,
``pyproj`` and ``shapely`` are installed there too (pip brings them in as
dependencies of this package), but their wheels carry their own copies of GDAL,
PROJ and GEOS, and loading a second copy into the QGIS process is what the
GDAL backend exists to avoid. A stray ``import rasterio`` on the GDAL path
therefore works in a test environment and breaks in QGIS.

When the test session starts with ``UMEP_USE_GDAL=1`` (the CI job
``test-qgis-compat``), this module makes any import of those packages fail,
naming the module that tried. Tests must then read and write rasters through
``solweig.io`` or ``tests.raster_helpers``, never through rasterio directly.

In the other direction nothing needs blocking: the standard environment has
no ``osgeo`` to import.
"""

from __future__ import annotations

import importlib.abc
import os
import sys

BLOCKED_ON_GDAL_BACKEND = ("rasterio", "pyproj", "shapely", "geopandas", "fiona")


def gdal_backend_requested() -> bool:
    return os.environ.get("UMEP_USE_GDAL", "").strip().lower() in ("1", "true", "yes")


class _Fence(importlib.abc.MetaPathFinder):
    def __init__(self, blocked: tuple[str, ...]) -> None:
        self.blocked = blocked

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            importer = sys._getframe(1).f_globals.get("__name__", "?")
            for depth in range(1, 12):  # skip importlib's own frames
                try:
                    name = sys._getframe(depth).f_globals.get("__name__", "")
                except ValueError:
                    break
                if name and not name.startswith("importlib") and name != __name__:
                    importer = name
                    break
            raise ImportError(
                f"'{fullname}' was imported by '{importer}' while the tests run on the GDAL backend "
                "(UMEP_USE_GDAL=1). Inside QGIS this loads a second copy of GDAL/PROJ/GEOS. "
                "Branch on solweig._compat.GDAL_ENV, or use tests.raster_helpers in a test."
            )
        return None


def install_backend_fence(*, host_packages_ok: bool = False) -> bool:
    """Install the fence if the session is on the GDAL backend. Returns whether it was installed.

    Args:
        host_packages_ok: Inside a running QGIS, packages that QGIS itself has
            already loaded (its own pyproj build, for instance) are safe and
            are left out of the fence. In a plain test session nothing may
            have been loaded yet, and finding one is an error.
    """
    if not gdal_backend_requested():
        return False
    if any(isinstance(finder, _Fence) for finder in sys.meta_path):
        return True
    already = {name.split(".")[0] for name in sys.modules} & set(BLOCKED_ON_GDAL_BACKEND)
    if already and not host_packages_ok:
        raise RuntimeError(f"backend fence installed too late: already imported {sorted(already)}")
    sys.meta_path.insert(0, _Fence(tuple(name for name in BLOCKED_ON_GDAL_BACKEND if name not in already)))
    return True
