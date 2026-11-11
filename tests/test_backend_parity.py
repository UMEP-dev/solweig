"""The rasterio backend and the GDAL backend must give the same results.

``tests/backend_parity_probe.py`` does raster reads and writes, a surface
preparation from files, and an untiled and a tiled model run, and prints a
fingerprint of every resulting array. This test runs it once per backend, each
in its own process, and compares the fingerprints.

Separate processes are required: rasterio's wheel carries its own GDAL, and
loading it next to the system GDAL in one process is the conflict the GDAL
backend exists to avoid. On the GDAL side the probe also installs the import
fence, so the run fails if anything touches rasterio, pyproj or shapely.

The comparison needs both backends installed, which is the case in the CI job
``test-qgis-compat``. Elsewhere the GDAL half is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "tests" / "backend_parity_probe.py"


def _env(gdal: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "UMEP_USE_GDAL"}
    env["SOLWEIG_NO_GPU"] = "1"  # one code path on every machine
    if gdal:
        env["UMEP_USE_GDAL"] = "1"
    return env


def _available(module: str, gdal: bool) -> bool:
    done = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, env=_env(gdal))
    return done.returncode == 0


def _run_probe(scratch: Path, *, gdal: bool) -> dict:
    done = subprocess.run(
        [sys.executable, str(PROBE), str(scratch)], capture_output=True, text=True, env=_env(gdal), timeout=600
    )
    backend = "GDAL" if gdal else "rasterio"
    assert done.returncode == 0, f"probe failed on the {backend} backend:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_backends_give_identical_results(tmp_path):
    if not _available("rasterio", gdal=False):
        pytest.skip("rasterio is not installed")
    if not _available("osgeo.gdal", gdal=True):
        pytest.skip("GDAL Python bindings are not installed (runs in the CI job test-qgis-compat)")

    standard = _run_probe(tmp_path / "rasterio", gdal=False)
    gdal = _run_probe(tmp_path / "gdal", gdal=True)

    assert sorted(standard) == sorted(gdal), "the two backends produced different sets of results"
    different = {key: (standard[key], gdal[key]) for key in standard if standard[key] != gdal[key]}
    assert not different, "results differ between backends (rasterio, GDAL):\n" + "\n".join(
        f"  {key}: {a} != {b}" for key, (a, b) in sorted(different.items())
    )


@pytest.mark.slow
def test_probe_is_repeatable_on_one_backend(tmp_path):
    """Guards the comparison above: a probe that varies from run to run would make it flaky."""
    gdal = os.environ.get("UMEP_USE_GDAL", "").strip() in ("1", "true", "yes")
    first = _run_probe(tmp_path / "a", gdal=gdal)
    second = _run_probe(tmp_path / "b", gdal=gdal)
    assert first == second
