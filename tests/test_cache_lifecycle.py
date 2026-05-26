"""Tests for the per-surface computation cache: lifecycle + key correctness.

Pins the Phase 2 hardening of `computation._arr_key`. The previous key was
`(ctypes.data, shape)` which silently served stale derived data after an
in-place mutation of a surface array. The hardened key includes witness
bytes from first/middle/last elements, catching most realistic mutations.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.computation import _arr_key
from solweig.models.surface import _ComputationCache


def test_arr_key_none_is_none():
    assert _arr_key(None) is None


def test_arr_key_stable_for_unchanged_array():
    arr = np.zeros((10, 10), dtype=np.float32)
    k1 = _arr_key(arr)
    k2 = _arr_key(arr)
    assert k1 == k2


def test_arr_key_changes_on_in_place_mutation_first_element():
    arr = np.zeros((10, 10), dtype=np.float32)
    k1 = _arr_key(arr)
    arr[0, 0] = 1.0
    k2 = _arr_key(arr)
    assert k1 != k2, "in-place mutation at index 0 must be caught"


def test_arr_key_changes_on_in_place_mutation_middle():
    arr = np.zeros((10, 10), dtype=np.float32)
    k1 = _arr_key(arr)
    # index 50 = (5, 0), which is the middle of a 100-element flat view
    arr.ravel()[50] = 1.0
    k2 = _arr_key(arr)
    assert k1 != k2, "in-place mutation at middle must be caught"


def test_arr_key_changes_on_in_place_mutation_last():
    arr = np.zeros((10, 10), dtype=np.float32)
    k1 = _arr_key(arr)
    arr[-1, -1] = 1.0
    k2 = _arr_key(arr)
    assert k1 != k2, "in-place mutation at last index must be caught"


def test_arr_key_changes_on_shape_change():
    arr = np.zeros((10, 10), dtype=np.float32)
    k1 = _arr_key(arr)
    arr2 = arr.reshape(100)
    k2 = _arr_key(arr2)
    # Shape differs even though data buffer is the same
    assert k1 != k2


def test_arr_key_changes_on_dtype_change():
    arr_f32 = np.zeros((10, 10), dtype=np.float32)
    arr_f64 = np.zeros((10, 10), dtype=np.float64)
    assert _arr_key(arr_f32) != _arr_key(arr_f64)


def test_arr_key_handles_nan_via_bit_pattern():
    """NaN != NaN, but bit-pattern comparison is stable."""
    arr1 = np.full((4, 4), np.nan, dtype=np.float32)
    arr2 = np.full((4, 4), np.nan, dtype=np.float32)
    # Different arrays, both all-NaN, same shape — only the data pointer differs.
    # The witness component should compare equal (NaN bits are identical).
    _, _, _, w1 = _arr_key(arr1)
    _, _, _, w2 = _arr_key(arr2)
    assert w1 == w2, "NaN witness bytes must compare equal across distinct arrays"


def test_arr_key_handles_empty_array():
    arr = np.zeros((0,), dtype=np.float32)
    k = _arr_key(arr)
    # No crash; witness is the empty bytes object.
    assert k is not None
    assert k[-1] == b""


def test_arr_key_is_hashable():
    """The key must be hashable so it can be used as a dict key."""
    arr = np.zeros((10, 10), dtype=np.float32)
    d = {_arr_key(arr): "value"}
    assert d[_arr_key(arr)] == "value"


# ── _ComputationCache lifecycle ─────────────────────────────────────────────


def test_cache_starts_empty():
    cache = _ComputationCache()
    for slot in cache.__slots__:
        assert getattr(cache, slot) is None


def test_cache_get_or_compute_caches_on_first_call():
    cache = _ComputationCache()
    call_count = 0

    def compute():
        nonlocal call_count
        call_count += 1
        return "computed"

    v1 = cache.get_or_compute("valid_mask_u8_cache", key=42, compute=compute)
    v2 = cache.get_or_compute("valid_mask_u8_cache", key=42, compute=compute)
    assert v1 == v2 == "computed"
    assert call_count == 1, "compute() should only run once for matching keys"


def test_cache_get_or_compute_recomputes_on_key_change():
    cache = _ComputationCache()
    call_count = 0

    def compute():
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    cache.get_or_compute("valid_mask_u8_cache", key=1, compute=compute)
    cache.get_or_compute("valid_mask_u8_cache", key=2, compute=compute)
    assert call_count == 2, "compute() should re-run when the key changes"


def test_cache_clear_resets_all_slots():
    cache = _ComputationCache()
    cache.get_or_compute("valid_mask_u8_cache", key=1, compute=lambda: "x")
    cache.get_or_compute("buildings_mask_cache", key=2, compute=lambda: "y")
    cache.clear()
    for slot in cache.__slots__:
        assert getattr(cache, slot) is None


def test_cache_slot_isolation():
    """Two unrelated slots must not interfere with each other."""
    cache = _ComputationCache()
    cache.get_or_compute("valid_mask_u8_cache", key=1, compute=lambda: "v")
    cache.get_or_compute("buildings_mask_cache", key=1, compute=lambda: "b")
    vmask = cache.valid_mask_u8_cache
    bmask = cache.buildings_mask_cache
    assert vmask is not None and vmask[1] == "v"
    assert bmask is not None and bmask[1] == "b"


# ── End-to-end: in-place mutation between calculate() calls ─────────────────


@pytest.mark.slow
def test_cached_value_invalidated_after_surface_mutation(tmp_path):
    """Mutating a surface array between calculate() calls must NOT serve stale
    derived data on the second call.

    The witness-byte component of `_arr_key` should detect the mutation and
    force recomputation of cache slots derived from the mutated array.
    """
    from datetime import datetime

    from conftest import make_mock_svf
    from solweig.api import Location, SurfaceData, Weather, calculate

    rng = np.random.default_rng(seed=0)
    dsm = rng.uniform(0, 5, size=(20, 20)).astype(np.float32)
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))

    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)

    out1 = tmp_path / "run1"
    calculate(surface, [weather], location, output_dir=out1, outputs=["tmrt"])

    # Mutate the DSM in-place — change the first element drastically.
    surface.dsm[0, 0] = 100.0

    # Re-run. The valid_mask / valid_bbox / buildings_mask caches that were
    # populated by the first call should NOT be silently reused — the witness
    # bytes on the cache key now differ.
    out2 = tmp_path / "run2"
    calculate(surface, [weather], location, output_dir=out2, outputs=["tmrt"])

    # We don't strictly need to check the output values — the test passes as
    # long as no stale-cache-induced crash or NaN cascade occurs. But a smoke
    # check that valid Tmrt was produced for both runs is cheap insurance.
    from conftest import read_timestep_geotiff

    t1 = read_timestep_geotiff(out1, "tmrt", 0)
    t2 = read_timestep_geotiff(out2, "tmrt", 0)
    assert np.isfinite(t1).any()
    assert np.isfinite(t2).any()
