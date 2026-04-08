"""Shared pytest configuration and path setup."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is on sys.path so that both `tests.qgis_mocks`
# and `qgis_plugin.*` imports work regardless of how pytest is invoked.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def pytest_configure(config: pytest.Config) -> None:
    """Fail early if the Rust extension was built in debug mode.

    Debug builds are 5-20x slower and make the test suite impractically slow.
    Rebuild with: ``maturin develop --release``
    """
    import solweig

    if not getattr(solweig, "RELEASE_BUILD", True):
        pytest.exit(
            "Rust extension was built in DEBUG mode — tests will be too slow.\n"
            "Rebuild with:  maturin develop --release",
            returncode=1,
        )


def read_timestep_geotiff(output_dir, field_name: str, index: int = 0):
    """Read the i-th per-timestep GeoTIFF from output_dir/field_name/.

    Returns the numpy array. Raises FileNotFoundError if no files exist
    or the index is out of range.
    """
    from solweig.io import load_raster

    tifs = sorted(Path(output_dir).joinpath(field_name).glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No GeoTIFFs in {Path(output_dir) / field_name}")
    if index >= len(tifs):
        raise FileNotFoundError(f"Only {len(tifs)} files in {field_name}/, requested index {index}")
    arr, _, _, _ = load_raster(str(tifs[index]))
    return arr


def poi_from_geojson(geojson_path, dsm_path) -> tuple[int, int]:
    """Read a POI GeoJSON point and return (row, col) in the DSM grid."""
    import json

    from solweig.io import load_raster

    with open(geojson_path) as f:
        feature = json.load(f)
    x, y = feature["geometry"]["coordinates"]

    _, transform, _, _ = load_raster(str(dsm_path))
    # transform is an affine: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
    col = int((x - transform[0]) / transform[1])
    row = int((y - transform[3]) / transform[5])
    return row, col


@pytest.fixture(scope="module")
def cpu_only():
    """Disable GPU for deterministic CPU-only tests, re-enable on teardown."""
    from solweig.rustalgos import shadowing

    was_enabled = shadowing.is_gpu_enabled()
    shadowing.disable_gpu()
    yield
    if was_enabled:
        shadowing.enable_gpu()


def make_mock_svf(shape: tuple[int, ...]):
    """Create a mock SvfArrays for tests (fully open sky)."""
    from solweig.models.precomputed import SvfArrays

    ones = np.ones(shape, dtype=np.float32)
    return SvfArrays(
        svf=ones.copy(),
        svf_north=ones.copy(),
        svf_east=ones.copy(),
        svf_south=ones.copy(),
        svf_west=ones.copy(),
        svf_veg=ones.copy(),
        svf_veg_north=ones.copy(),
        svf_veg_east=ones.copy(),
        svf_veg_south=ones.copy(),
        svf_veg_west=ones.copy(),
        svf_aveg=ones.copy(),
        svf_aveg_north=ones.copy(),
        svf_aveg_east=ones.copy(),
        svf_aveg_south=ones.copy(),
        svf_aveg_west=ones.copy(),
    )
