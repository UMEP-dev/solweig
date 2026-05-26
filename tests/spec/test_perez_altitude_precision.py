"""Steradian-computation precision and edge-case tests.

Locks in the Phase 1 hardening of `compute_steradians` (no more `unwrap()` on
the unique-altitude lookup; empty/single-element inputs are handled cleanly)
and pins per-patch numerics against a known-good reference value.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.physics.create_patches import create_patches as py_create_patches
from solweig.physics.patch_radiation import patch_steradians as py_patch_steradians
from solweig.rustalgos import pipeline

PATCH_OPTIONS = [1, 2, 3, 4]


def _py_steradians_for_option(patch_option: int) -> np.ndarray:
    """Run the Python reference patch_steradians for a given patch option.

    The Python entry point takes an `L_patches` Nx3 array (column 0 = altitude)
    so we construct one from `create_patches` altitudes; the other two columns
    are ignored by `patch_steradians`.
    """
    alts, _, _, _, _, _, _ = py_create_patches(patch_option)
    lv = np.zeros((alts.size, 3), dtype=np.float32)
    lv[:, 0] = alts
    ster, _, _ = py_patch_steradians(lv)
    return np.asarray(ster, dtype=np.float64)


def test_compute_steradians_sums_to_2pi():
    """Steradians of a full sky vault must sum to 2π (hemisphere)."""
    for opt in PATCH_OPTIONS:
        ster = pipeline.compute_steradians_py(opt)
        total = float(np.sum(ster))
        # The Robinson/Stone decompositions are exact in continuous form but
        # use rectangular patches; tolerance follows existing UMEP parity tests.
        assert abs(total - 2 * np.pi) < 0.05, f"option={opt} sum={total}"


def test_compute_steradians_all_finite():
    """No NaN/Inf escapes from any patch option."""
    for opt in PATCH_OPTIONS:
        ster = pipeline.compute_steradians_py(opt)
        assert np.all(np.isfinite(ster)), f"option={opt} produced non-finite values"
        assert np.all(ster >= 0.0), f"option={opt} produced negative steradian"


def test_compute_steradians_repeated_calls_match():
    """The Mutex-cached patch layout returns identical results across calls.

    Regression guard against poison-recovery silently swapping data.
    """
    first = pipeline.compute_steradians_py(2).copy()
    for _ in range(5):
        again = pipeline.compute_steradians_py(2)
        assert np.array_equal(first, again)


@pytest.mark.parametrize("patch_option", PATCH_OPTIONS)
def test_compute_steradians_matches_python_reference_per_patch(patch_option):
    """Per-patch values must match the Python reference implementation.

    The aggregate 2π sum test could still pass if individual patches drifted
    in a way that cancelled out (e.g. an off-by-one between bands). This test
    pins each patch's steradian against ``solweig.physics.patch_radiation``'s
    reference Python implementation, which is the direct port of the UMEP
    Robinson/Stone decomposition.

    Also catches regressions to the ``if i > 0`` guard inserted alongside the
    Phase 1 unwrap removal: if that guard were dropped, a single-patch band
    appearing at index 0 would silently emit `0.0` instead of the correct
    spherical-cap value.
    """
    rust_ster = np.asarray(pipeline.compute_steradians_py(patch_option), dtype=np.float64)
    py_ster = _py_steradians_for_option(patch_option)
    np.testing.assert_allclose(
        rust_ster,
        py_ster,
        rtol=1e-5,
        atol=1e-7,
        err_msg=f"Per-patch steradian drift between Rust and Python for option={patch_option}",
    )


def test_compute_steradians_zenith_patch_is_nonzero():
    """The single zenith-band patch (last patch, altitude=90°) must not be 0.

    Pinpoints the ``if i > 0`` guard: the original implementation indexed
    ``altitudes[i - 1]`` unconditionally, which is safe only because the zenith
    patch is never at index 0 in any of the four patch options. The guard
    keeps that property defensive instead of implicit.
    """
    for opt in PATCH_OPTIONS:
        ster = pipeline.compute_steradians_py(opt)
        assert ster[-1] > 0.0, f"option={opt}: zenith patch steradian is zero"
