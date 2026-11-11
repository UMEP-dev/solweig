"""The fence that keeps rasterio, pyproj and shapely out of GDAL-backend test runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.backend_fence import BLOCKED_ON_GDAL_BACKEND, gdal_backend_requested

REPO = Path(__file__).resolve().parent.parent

PROBE = """
import sys
sys.path.insert(0, {repo!r})
from tests.backend_fence import install_backend_fence
print("installed", install_backend_fence())
try:
    import rasterio
    print("rasterio imported")
except ImportError as exc:
    print("blocked:", exc)
"""


def _probe(env_value: str | None) -> str:
    import os

    env = {k: v for k, v in os.environ.items() if k != "UMEP_USE_GDAL"}
    if env_value is not None:
        env["UMEP_USE_GDAL"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", PROBE.format(repo=str(REPO))], capture_output=True, text=True, env=env, check=True
    )
    return out.stdout


def test_fence_blocks_rasterio_and_names_the_importer_on_the_gdal_backend():
    out = _probe("1")
    assert "installed True" in out
    assert "blocked: 'rasterio' was imported by '__main__'" in out
    assert "rasterio imported" not in out


def test_fence_is_absent_on_the_standard_backend():
    pytest.importorskip("rasterio")
    out = _probe(None)
    assert "installed False" in out
    assert "rasterio imported" in out


def test_this_session_obeys_the_fence():
    """On the GDAL backend none of the fenced packages may have been loaded by the test run."""
    if not gdal_backend_requested():
        pytest.skip("standard backend session")
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED_ON_GDAL_BACKEND)
    assert not loaded, f"fenced packages were imported: {loaded[:5]}"


def test_no_test_module_imports_rasterio_directly():
    """Tests use solweig.io or tests.raster_helpers, so they run on either backend."""
    allowed = {"raster_helpers.py", "test_backend_fence.py", "backend_fence.py"}
    offenders = []
    for path in (REPO / "tests").rglob("*.py"):
        if path.name in allowed:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import rasterio", "from rasterio")):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, f"direct rasterio imports in tests: {offenders}"
