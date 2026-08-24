"""SOLWEIG_NO_GPU must reach the calculate() path.

Regression test for a gap where the lazy GPU gate (`_ensure_gpu_initialized`)
only ran via ``is_gpu_available()`` or ``compute_svf()``. A ``calculate()``
run with *precomputed* SVF touched neither, so ``SOLWEIG_NO_GPU=1`` silently
left every GPU path enabled. The gate is now invoked at the top of
``calculate_core_fused``.

On machines without a GPU the dispatch count is trivially zero, so the test
is meaningful only where GPU support is compiled in and available; it still
runs everywhere as a smoke test of the gate.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pytest
import solweig
from solweig.models.precomputed import SvfArrays
from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls


@pytest.fixture()
def _restore_gpu_state():
    yield
    # The test flips the process-wide GPU toggles off via the env gate;
    # restore the default-enabled state for the rest of the suite.
    solweig.enable_gpu()
    setattr(solweig, "_gpu_initialized", True)  # noqa: B010 — ty narrows the literal on direct assignment
    solweig.reset_gpu_metrics()


def test_no_gpu_env_disables_gpu_on_precomputed_svf_path(tmp_path, monkeypatch, _restore_gpu_state):
    if not solweig.GPU_ENABLED:
        pytest.skip("GPU support not compiled in")

    # Simulate process start under SOLWEIG_NO_GPU=1: env set, lazy gate not
    # yet fired, Rust-side toggles at their default (enabled).
    monkeypatch.setenv("SOLWEIG_NO_GPU", "1")
    solweig.enable_gpu()
    monkeypatch.setattr(solweig, "_gpu_initialized", False)

    n = 24
    dsm = np.zeros((n, n), dtype=np.float32)
    dsm[8:14, 8:14] = 10.0  # one building so the shadow kernel has work
    wall_hts = findwalls(dsm, 1.0)
    wall_asp = filter1Goodwin_as_aspect_v3(wall_hts.copy(), 1.0, dsm)
    ones = np.ones((n, n), dtype=np.float32)
    svf = SvfArrays(**{f.name: ones * 0.9 for f in dataclasses.fields(SvfArrays)})

    surface = solweig.SurfaceData(
        dsm=dsm,
        dem=np.zeros_like(dsm),
        pixel_size=1.0,
        dsm_relative=False,
        wall_height=wall_hts,
        wall_aspect=wall_asp,
    )
    surface.preprocess()
    surface.svf = svf

    weather = solweig.Weather(datetime=dt.datetime(2023, 6, 29, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)
    location = solweig.Location(latitude=57.7, longitude=12.0)

    solweig.reset_gpu_metrics()
    solweig.calculate(surface, [weather], location, output_dir=str(tmp_path))

    assert solweig.gpu_dispatch_count() == 0, "SOLWEIG_NO_GPU=1 must prevent all GPU dispatches"
