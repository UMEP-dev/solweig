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
    # Runs in a child process without UMEP_USE_GDAL, so it is valid in a GDAL
    # session too. (pytest.importorskip here would trip the fence itself.)
    out = _probe(None)
    assert "installed False" in out
    if "No module named" in out:
        pytest.skip("rasterio is not installed")
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
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            for name in BLOCKED_ON_GDAL_BACKEND:
                direct = stripped.startswith((f"import {name}", f"from {name}"))
                # importorskip does not skip here: the fence raises, it does not report "not installed".
                skipping = f'importorskip("{name}' in stripped
                if direct or skipping:
                    offenders.append(f"{path.relative_to(REPO)}:{number} ({name})")
    assert not offenders, f"tests that use a fenced package directly: {offenders}"
