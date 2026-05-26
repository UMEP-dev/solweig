"""Backend-detection lazy-load tests for `solweig._compat`.

Pins the PEP 562 `__getattr__` behaviour: backend attrs are computed on first
access, cached afterwards, and re-detected after `importlib.reload()`.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


def _purge_solweig_modules() -> None:
    for k in [k for k in sys.modules if k == "solweig._compat" or k.startswith("solweig._compat.")]:
        del sys.modules[k]


def test_getattr_unknown_attr_raises_attribute_error():
    import solweig._compat as compat

    with pytest.raises(AttributeError):
        compat.NOT_A_REAL_BACKEND_ATTR  # noqa: B018


def test_getattr_returns_bool_and_caches_on_first_access():
    _purge_solweig_modules()
    compat = importlib.import_module("solweig._compat")

    # Before first access the attrs are NOT stamped in the module dict.
    assert "GDAL_ENV" not in compat.__dict__
    assert "RASTERIO_AVAILABLE" not in compat.__dict__
    assert "GDAL_AVAILABLE" not in compat.__dict__

    # Access via attribute triggers _setup_geospatial_backend (PEP 562).
    val = compat.GDAL_ENV
    assert isinstance(val, bool)

    # Once accessed, the value is stamped — subsequent reads must NOT re-fire
    # __getattr__. We check this by reading the value out of __dict__ directly.
    assert "GDAL_ENV" in compat.__dict__
    assert compat.__dict__["GDAL_ENV"] is val

    # The other two attrs are computed in the same backend call.
    assert "RASTERIO_AVAILABLE" in compat.__dict__
    assert "GDAL_AVAILABLE" in compat.__dict__
    # Exactly one of GDAL_AVAILABLE / RASTERIO_AVAILABLE must be True in a
    # standard environment.
    assert compat.GDAL_AVAILABLE or compat.RASTERIO_AVAILABLE


def test_importlib_reload_clears_stamped_attrs():
    """Regression: b64/b65 fix — reload must reset stamped backend attrs.

    Without the module-level clear loop, switching backends across a reload
    (e.g. setting UMEP_USE_GDAL between test cases) would silently retain the
    old detection result.
    """
    _purge_solweig_modules()
    compat = importlib.import_module("solweig._compat")
    _ = compat.GDAL_ENV  # stamp
    assert "GDAL_ENV" in compat.__dict__

    compat2 = importlib.reload(compat)
    # After reload the stamp must be gone — the module's top-level cleanup
    # block deletes any previously stamped attrs so __getattr__ re-fires.
    assert "GDAL_ENV" not in compat2.__dict__
    assert "RASTERIO_AVAILABLE" not in compat2.__dict__
    assert "GDAL_AVAILABLE" not in compat2.__dict__


def test_umep_use_gdal_env_var_forces_gdal_when_available():
    """When UMEP_USE_GDAL=1 is set and GDAL is importable, GDAL_ENV must be True."""
    try:
        from osgeo import gdal  # noqa: F401
    except ImportError:
        pytest.skip("GDAL not installed; cannot verify the forced-GDAL branch")

    _purge_solweig_modules()
    old = os.environ.get("UMEP_USE_GDAL")
    os.environ["UMEP_USE_GDAL"] = "1"
    try:
        compat = importlib.import_module("solweig._compat")
        assert compat.GDAL_ENV is True
        assert compat.GDAL_AVAILABLE is True
        assert compat.RASTERIO_AVAILABLE is False
    finally:
        if old is None:
            os.environ.pop("UMEP_USE_GDAL", None)
        else:
            os.environ["UMEP_USE_GDAL"] = old
        _purge_solweig_modules()


def test_in_osgeo_environment_returns_bool():
    from solweig._compat import in_osgeo_environment

    assert isinstance(in_osgeo_environment(), bool)
