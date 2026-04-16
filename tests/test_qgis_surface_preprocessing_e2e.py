"""End-to-end test for SurfacePreprocessingAlgorithm.processAlgorithm().

Closes the plugin integration gap flagged in the 2026-04-16 audit:
prior tests only covered the base class (``test_qgis_base.py``) and
converters (``test_qgis_converters.py``); the actual
:class:`SurfacePreprocessingAlgorithm` workflow was untested.

This test drives the algorithm against a synthetic GeoTIFF, mocking
only the QGIS parameter-reading layer. It exercises the plugin glue
(parameter dispatching, output directory layout, progress reporting)
and solweig's ``SurfaceData.prepare()``, wall computation, and SVF
computation. Because ``UMEP_USE_GDAL`` is not set in the test
environment, solweig takes the **rasterio backend**, not the GDAL
backend QGIS users actually run under. The backend abstraction itself
is covered by ``test_qgis_base.py`` and ``test_io.py`` (with
``UMEP_USE_GDAL=1`` and mocked osgeo), so this test intentionally
skips that plumbing and focuses on the algorithm-level contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tests.qgis_mocks import install, install_osgeo, preserve_solweig_modules, uninstall_osgeo

install()
install_osgeo()

with preserve_solweig_modules():
    from qgis_plugin.solweig_qgis.algorithms.preprocess.surface_preprocessing import (  # noqa: E402
        SurfacePreprocessingAlgorithm,
    )

uninstall_osgeo()


def _write_synthetic_dsm(path: Path, rows: int = 32, cols: int = 32, pixel_size: float = 2.0) -> None:
    """Small DSM: sloped terrain with two block buildings. Ample for SVF."""
    terrain = np.linspace(0.0, 3.0, rows)[:, None] + np.linspace(0.0, 1.0, cols)[None, :]
    dsm = terrain.astype(np.float32)
    dsm[6:14, 6:14] += 15.0  # block A
    dsm[18:25, 18:26] += 8.0  # block B
    transform = from_origin(100000.0, 200000.0 + rows * pixel_size, pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        transform=transform,
        crs="EPSG:3006",  # SWEREF99 TM (projected metres — required for solweig)
        nodata=-9999.0,
    ) as dst:
        dst.write(dsm, 1)


class _MockExtent:
    """Stand-in for QgsRectangle used by extent parameters."""

    def isNull(self) -> bool:  # noqa: N802 (Qt naming convention)
        return True

    def xMinimum(self):  # noqa: N802
        return 0.0

    def yMinimum(self):  # noqa: N802
        return 0.0

    def xMaximum(self):  # noqa: N802
        return 0.0

    def yMaximum(self):  # noqa: N802
        return 0.0


def _build_mock_raster_layer(path: str) -> MagicMock:
    """Mock QgsRasterLayer whose ``.source()`` returns a real GeoTIFF path."""
    layer = MagicMock()
    layer.source.return_value = path
    layer.isValid.return_value = True
    return layer


@pytest.fixture()
def algo():
    return SurfacePreprocessingAlgorithm()


@pytest.fixture()
def feedback():
    fb = MagicMock()
    fb.isCanceled.return_value = False
    return fb


@pytest.mark.slow
def test_surface_preprocessing_e2e_writes_expected_artifacts(algo, feedback, tmp_path):
    """The full processAlgorithm() flow must write cleaned rasters, walls, SVF,
    and parametersforsolweig.json to the output directory."""
    dsm_path = tmp_path / "dsm.tif"
    _write_synthetic_dsm(dsm_path)
    output_dir = tmp_path / "prepared"

    dsm_layer = _build_mock_raster_layer(str(dsm_path))

    # Parameters dict mimicking what QGIS Processing would pass in.
    # Only DSM is populated; optional layers (CDSM/DEM/TDSM/LAND_COVER) are
    # explicitly None, matching what parameterAsRasterLayer returns when
    # the user leaves them unset.
    params = {
        "DSM": dsm_layer,
        "CDSM": None,
        "DEM": None,
        "TDSM": None,
        "LAND_COVER": None,
        "PIXEL_SIZE": 0.0,  # 0 = use native
        "WALL_LIMIT": 1.0,
        "DSM_HEIGHT_MODE": 1,  # 1 = absolute
        "CDSM_HEIGHT_MODE": 0,  # 0 = relative (default)
        "TDSM_HEIGHT_MODE": 0,
        "MIN_OBJECT_HEIGHT": 1.0,
        "TRANSMISSIVITY": 0.03,
        "TRANSMISSIVITY_LEAFOFF": 0.5,
        "LEAF_START": 97,
        "LEAF_END": 300,
        "EXTENT": _MockExtent(),
        "OUTPUT_DIR": str(output_dir),
    }

    # Wire the parameter-reading shims on the algo instance. The real
    # SolweigAlgorithmBase (subclasses QgsProcessingAlgorithm) resolves
    # each parameterAs* call via QGIS APIs; here we substitute them.
    algo.parameterAsRasterLayer = lambda p, name, ctx: p.get(name)
    algo.parameterAsDouble = lambda p, name, ctx: float(p.get(name, 0.0))
    algo.parameterAsEnum = lambda p, name, ctx: int(p.get(name, 0))
    algo.parameterAsString = lambda p, name, ctx: str(p.get(name, ""))
    algo.parameterAsExtent = lambda p, name, ctx: p.get(name, _MockExtent())
    algo.parameterAsFile = lambda p, name, ctx: p.get(name, "") or ""
    algo.parameterAsMatrix = lambda p, name, ctx: p.get(name, [])

    # Run.
    context = MagicMock()
    result = algo.processAlgorithm(params, context, feedback)

    # Algorithm contract: returns a dict (even if empty on early exit).
    assert isinstance(result, dict)

    # The output directory must exist.
    assert output_dir.is_dir(), "Output directory was not created"

    # SurfaceData.prepare() lays down cleaned rasters + walls + SVF into
    # subdirectories of the working directory. Presence of these is the
    # key regression check.
    assert (output_dir / "cleaned").is_dir(), "cleaned/ directory missing"
    walls_root = output_dir / "walls"
    svf_root = output_dir / "svf"
    assert walls_root.is_dir(), "walls/ directory missing"
    assert svf_root.is_dir(), "svf/ directory missing"

    # Each is a per-pixel-size cache. We don't hard-code the pixel size
    # here (native DSM resolution was used) but we require at least one
    # px<size> sub-directory under each.
    assert any(walls_root.glob("px*")), "walls/ has no pixel-size bucket"
    assert any(svf_root.glob("px*")), "svf/ has no pixel-size bucket"

    # parametersforsolweig.json is the plugin-specific UMEP-compatible
    # materials file — must be written in the output root.
    params_json = output_dir / "parametersforsolweig.json"
    assert params_json.is_file(), "parametersforsolweig.json missing"

    # Final progress should have hit 100.
    setProgress_calls = [c.args[0] for c in feedback.setProgress.call_args_list]
    assert 100 in setProgress_calls, f"Algorithm did not report 100% progress; saw {setProgress_calls}"


@pytest.mark.slow
def test_surface_preprocessing_warm_run_uses_cache(algo, feedback, tmp_path):
    """Second call on the same inputs must hit the warm-run fast-path
    (drastically faster than first call). Verifies the prepare() cache
    fingerprinting is working end-to-end through the plugin."""
    import time

    dsm_path = tmp_path / "dsm.tif"
    _write_synthetic_dsm(dsm_path)
    output_dir = tmp_path / "prepared"

    dsm_layer = _build_mock_raster_layer(str(dsm_path))
    params = {
        "DSM": dsm_layer,
        "CDSM": None,
        "DEM": None,
        "TDSM": None,
        "LAND_COVER": None,
        "PIXEL_SIZE": 0.0,
        "WALL_LIMIT": 1.0,
        "DSM_HEIGHT_MODE": 1,
        "CDSM_HEIGHT_MODE": 0,
        "TDSM_HEIGHT_MODE": 0,
        "MIN_OBJECT_HEIGHT": 1.0,
        "TRANSMISSIVITY": 0.03,
        "TRANSMISSIVITY_LEAFOFF": 0.5,
        "LEAF_START": 97,
        "LEAF_END": 300,
        "EXTENT": _MockExtent(),
        "OUTPUT_DIR": str(output_dir),
    }
    algo.parameterAsRasterLayer = lambda p, name, ctx: p.get(name)
    algo.parameterAsDouble = lambda p, name, ctx: float(p.get(name, 0.0))
    algo.parameterAsEnum = lambda p, name, ctx: int(p.get(name, 0))
    algo.parameterAsString = lambda p, name, ctx: str(p.get(name, ""))
    algo.parameterAsExtent = lambda p, name, ctx: p.get(name, _MockExtent())
    algo.parameterAsFile = lambda p, name, ctx: p.get(name, "") or ""
    algo.parameterAsMatrix = lambda p, name, ctx: p.get(name, [])

    context = MagicMock()

    # Cold run.
    t0 = time.time()
    algo.processAlgorithm(params, context, feedback)
    cold_elapsed = time.time() - t0

    # Warm run on a fresh algo/feedback — same params, same output_dir.
    # Cached walls+SVF should make this substantially faster than the
    # cold run. We do not assert a hard speedup factor (CI variance) but
    # require that the warm run takes less time than the cold run.
    fb2 = MagicMock()
    fb2.isCanceled.return_value = False
    algo2 = SurfacePreprocessingAlgorithm()
    algo2.parameterAsRasterLayer = algo.parameterAsRasterLayer
    algo2.parameterAsDouble = algo.parameterAsDouble
    algo2.parameterAsEnum = algo.parameterAsEnum
    algo2.parameterAsString = algo.parameterAsString
    algo2.parameterAsExtent = algo.parameterAsExtent
    algo2.parameterAsFile = algo.parameterAsFile
    algo2.parameterAsMatrix = algo.parameterAsMatrix

    t0 = time.time()
    algo2.processAlgorithm(params, context, fb2)
    warm_elapsed = time.time() - t0

    assert warm_elapsed < cold_elapsed, (
        f"Warm run ({warm_elapsed:.2f}s) should be faster than cold run ({cold_elapsed:.2f}s)"
    )
