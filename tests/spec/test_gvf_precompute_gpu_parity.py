"""CPU vs GPU parity for the GVF geometry precompute.

Ports `precompute_gvf_geometry` to a wgpu compute shader (one thread per
pixel, all 18 azimuths marched internally, cross-azimuth reduction in-shader).
This test runs `precompute_gvf_cache` twice on the same synthetic walled scene
— once with the precompute GPU path enabled, once forced onto the CPU — and
asserts the purely-geometric cached outputs agree:

- ``blocking_distance`` (per-azimuth, integer) — exact match. The CPU folds a
  stale edge-clamped buffer, but ``min`` is idempotent on the repeated value,
  so the blocking distance is unaffected and must match exactly.
- ``facesh`` (per-azimuth wall mask) — exact match. Pure per-pixel function of
  aspect/height/azimuth with identical thresholds.
- ``cached_albnosh`` + 4 directional channels — f32 accumulation tolerance on
  the bulk, with a small edge-pixel tail allowance (mirrors
  ``test_gpu_cpu_parity.py``). The edge tail is where the CPU's stale-buffer
  clamp contributes; the shader reproduces the clamp, so the tail stays tiny.

The test is skipped when no GPU is present (``gpu_dispatch_count`` stays 0).
"""

from __future__ import annotations

import numpy as np
import pytest
import solweig
from solweig.rustalgos import pipeline


def _make_walled_scene(rows: int = 48, cols: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic buildings + walls + aspect + albedo.

    Structured (not random-only) so blocking distances, the facesh branches,
    and near-border edge-clamp behaviour are all exercised. Feature positions
    keep clear of the exact border so shifts reach off-grid for many pixels."""
    rng = np.random.default_rng(7)

    buildings = np.zeros((rows, cols), dtype=np.float32)
    # A few rectangular building footprints (1 = building).
    buildings[6:16, 8:20] = 1.0
    buildings[24:34, 28:40] = 1.0
    buildings[30:38, 6:14] = 1.0
    buildings[10:14, 34:44] = 1.0

    # Wall height: building footprints carry their height, ground is 0.
    heights = np.zeros((rows, cols), dtype=np.float32)
    heights[6:16, 8:20] = 9.0
    heights[24:34, 28:40] = 12.0
    heights[30:38, 6:14] = 7.0
    heights[10:14, 34:44] = 5.0
    wall_ht = heights.astype(np.float32)

    # Aspect: full [0, 2π) coverage via a 2-D gradient so every facesh branch
    # sees both true and false cases.
    r = np.arange(rows).reshape(-1, 1)
    c = np.arange(cols).reshape(1, -1)
    wall_asp = (((r * cols + c) % 360) * (np.pi / 180.0)).astype(np.float32)
    wall_asp = np.ascontiguousarray(np.broadcast_to(wall_asp, (rows, cols)), dtype=np.float32)

    # Albedo: spatially varying, differs on/off buildings.
    alb_grid = (0.12 + 0.10 * rng.random((rows, cols))).astype(np.float32)
    alb_grid[buildings > 0] = 0.30

    return (
        np.ascontiguousarray(buildings),
        np.ascontiguousarray(wall_asp),
        np.ascontiguousarray(wall_ht),
        np.ascontiguousarray(alb_grid),
    )


def _precompute(buildings, wall_asp, wall_ht, alb_grid):
    return pipeline.precompute_gvf_cache(
        buildings,
        wall_asp,
        wall_ht,
        alb_grid,
        1.0,  # pixel_size (m)
        1.1,  # human_height (m)  -> first=1, second=22
        0.20,  # wall_albedo
    )


def _extract(cache) -> dict[str, np.ndarray]:
    return {
        "albnosh": np.asarray(cache.cached_albnosh()),
        "albnosh_e": np.asarray(cache.cached_albnosh_e()),
        "albnosh_s": np.asarray(cache.cached_albnosh_s()),
        "albnosh_w": np.asarray(cache.cached_albnosh_w()),
        "albnosh_n": np.asarray(cache.cached_albnosh_n()),
        "blocking_distance": np.asarray(cache.blocking_distance_all()),
        "facesh": np.asarray(cache.facesh_all()),
    }


def _assert_albnosh_parity(name: str, gpu: np.ndarray, cpu: np.ndarray) -> None:
    """Bulk f32 tolerance + small edge-pixel tail (mirrors test_gpu_cpu_parity)."""
    diff = np.abs(gpu - cpu)
    tol = 2e-3 + 1e-3 * np.abs(cpu)
    within_tight = diff <= tol
    frac_ok = float(np.count_nonzero(within_tight)) / gpu.size
    assert frac_ok >= 0.98, (
        f"{name}: only {frac_ok:.4%} of pixels within tight tolerance (rtol=1e-3, atol=2e-3); max diff {diff.max():.6g}"
    )
    # The <=2% tail (near-border edge-clamp pixels) must still be close.
    assert diff.max() <= 0.05, f"{name}: edge-tail max diff {diff.max():.6g} exceeds 0.05"


def test_gvf_precompute_gpu_cpu_parity():
    buildings, wall_asp, wall_ht, alb_grid = _make_walled_scene()

    was_enabled = pipeline.is_gvf_precompute_gpu_enabled()

    # ── GPU precompute ──
    pipeline.enable_gvf_precompute_gpu()
    solweig.reset_gpu_metrics()
    gpu_cache = _precompute(buildings, wall_asp, wall_ht, alb_grid)
    dispatched = solweig.gpu_dispatch_count()

    if dispatched == 0 or not pipeline.is_gvf_precompute_gpu_enabled():
        # No usable GPU (context never initialised) or the precompute fell back
        # to CPU — nothing to compare.
        if was_enabled:
            pipeline.enable_gvf_precompute_gpu()
        pytest.skip("GVF precompute GPU path did not run (no GPU / fell back)")

    gpu = _extract(gpu_cache)

    # ── CPU precompute ──
    pipeline.disable_gvf_precompute_gpu()
    try:
        cpu_cache = _precompute(buildings, wall_asp, wall_ht, alb_grid)
    finally:
        if was_enabled:
            pipeline.enable_gvf_precompute_gpu()
    cpu = _extract(cpu_cache)

    # Sanity: both agree on grid + azimuth counts.
    assert gpu_cache.num_azimuths == cpu_cache.num_azimuths == 18
    assert gpu_cache.first == cpu_cache.first
    assert gpu_cache.second == cpu_cache.second

    # blocking_distance: exact integer match (edge-clamp is idempotent under min).
    assert np.array_equal(gpu["blocking_distance"], cpu["blocking_distance"]), (
        "blocking_distance mismatch: "
        f"{np.count_nonzero(gpu['blocking_distance'] != cpu['blocking_distance'])} cells differ"
    )

    # facesh: exact match (pure per-pixel, identical thresholds).
    assert np.array_equal(gpu["facesh"], cpu["facesh"]), (
        f"facesh mismatch: {np.count_nonzero(gpu['facesh'] != cpu['facesh'])} cells differ"
    )

    # cached_albnosh channels: f32 accumulation tolerance + edge tail.
    for name in ("albnosh", "albnosh_e", "albnosh_s", "albnosh_w", "albnosh_n"):
        _assert_albnosh_parity(name, gpu[name], cpu[name])
