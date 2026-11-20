"""Parity tests: Rust Perez_v3 vs upstream UMEP Perez_v3 and steradians.

Verifies that the Rust port produces results matching the upstream UMEP
implementation for a range of atmospheric conditions.
"""

import numpy as np
import pytest
from solweig.physics.patch_radiation import patch_steradians
from solweig.rustalgos import pipeline

from tests.reference_package import reference_module

umep_perez = reference_module("util.SEBESOLWEIGCommonFiles.Perez_v3")
Perez_v3 = umep_perez.Perez_v3

# ── Test parameters: representative atmospheric conditions ──────────────────

# Cases where Rust and UMEP should agree (normal atmospheric conditions).
PEREZ_CASES = [
    # (zen_deg, azimuth_deg, rad_d, rad_i, jday, patch_option, label)
    (30.0, 180.0, 200.0, 400.0, 180, 2, "clear_midday_summer"),
    (60.0, 135.0, 150.0, 300.0, 80, 2, "morning_spring"),
    (75.0, 250.0, 80.0, 100.0, 350, 2, "low_sun_winter"),
    (45.0, 200.0, 300.0, 100.0, 180, 2, "overcast_summer"),
    (20.0, 180.0, 50.0, 800.0, 180, 2, "very_clear_high_sun"),
    (85.0, 90.0, 20.0, 30.0, 1, 2, "near_horizon"),
    # Different patch options
    (40.0, 180.0, 200.0, 400.0, 180, 1, "patch_option_1"),
    (40.0, 180.0, 200.0, 400.0, 180, 3, "patch_option_3"),
]

# Cases where Rust intentionally diverges from UMEP: the Rust port includes
# low-sun and low-diffuse guards (returning uniform distribution) that the
# upstream UMEP implementation lacks. These guards prevent NaN/Inf at edge cases.
PEREZ_EDGE_CASES = [
    (88.0, 180.0, 50.0, 10.0, 180, 2, "below_threshold"),
    (30.0, 180.0, 5.0, 400.0, 180, 2, "low_diffuse"),
]


class TestPerez_v3Parity:
    """Rust perez_v3 must match upstream UMEP Perez_v3 for the same inputs."""

    @pytest.mark.parametrize(
        "zen,azi,rad_d,rad_i,jday,patch_option,label",
        PEREZ_CASES,
        ids=[c[-1] for c in PEREZ_CASES],
    )
    def test_luminance_parity(self, zen, azi, rad_d, rad_i, jday, patch_option, label):
        """Rust luminance column matches UMEP within f32 tolerance."""
        # UMEP reference
        py_lv, _, _ = Perez_v3(zen, azi, rad_d, rad_i, jday, patchchoice=1, patch_option=patch_option)

        # Rust implementation
        rs_lv = np.asarray(pipeline.perez_v3_py(zen, azi, rad_d, rad_i, jday, patch_option))

        # Shape must match
        assert py_lv.shape == rs_lv.shape, f"shape mismatch: umep={py_lv.shape} rs={rs_lv.shape}"

        # Altitudes and azimuths (columns 0,1) come from create_patches — should match exactly
        np.testing.assert_allclose(
            rs_lv[:, 0],
            py_lv[:, 0],
            atol=0.01,
            err_msg=f"[{label}] Patch altitudes differ",
        )
        np.testing.assert_allclose(
            rs_lv[:, 1],
            py_lv[:, 1],
            atol=0.01,
            err_msg=f"[{label}] Patch azimuths differ",
        )

        # Luminances (column 2) — allow f32 precision tolerance
        np.testing.assert_allclose(
            rs_lv[:, 2],
            py_lv[:, 2].astype(np.float32),
            rtol=1e-3,
            atol=1e-6,
            err_msg=f"[{label}] Patch luminances differ",
        )

    @pytest.mark.parametrize(
        "zen,azi,rad_d,rad_i,jday,patch_option,label",
        PEREZ_CASES,
        ids=[c[-1] for c in PEREZ_CASES],
    )
    def test_luminance_normalised(self, zen, azi, rad_d, rad_i, jday, patch_option, label):
        """Rust luminance sums to 1.0 (normalised probability distribution)."""
        rs_lv = np.asarray(pipeline.perez_v3_py(zen, azi, rad_d, rad_i, jday, patch_option))
        lum_sum = rs_lv[:, 2].sum()
        assert abs(lum_sum - 1.0) < 1e-4, f"[{label}] luminance sum = {lum_sum}"


class TestPerez_v3EdgeCases:
    """Rust intentionally diverges from UMEP at edge cases (low sun, low diffuse).

    The Rust port includes guards that return a uniform distribution when
    sun altitude < 3° or diffuse radiation < 10 W/m². UMEP lacks these
    guards and can produce numerical instability at these conditions.
    """

    @pytest.mark.parametrize(
        "zen,azi,rad_d,rad_i,jday,patch_option,label",
        PEREZ_EDGE_CASES,
        ids=[c[-1] for c in PEREZ_EDGE_CASES],
    )
    def test_rust_returns_uniform(self, zen, azi, rad_d, rad_i, jday, patch_option, label):
        """Rust returns uniform distribution at edge cases."""
        rs_lv = np.asarray(pipeline.perez_v3_py(zen, azi, rad_d, rad_i, jday, patch_option))
        std_dev = np.std(rs_lv[:, 2])
        assert std_dev < 1e-6, f"[{label}] Rust should return uniform (std={std_dev:.8f})"

    @pytest.mark.parametrize(
        "zen,azi,rad_d,rad_i,jday,patch_option,label",
        PEREZ_EDGE_CASES,
        ids=[c[-1] for c in PEREZ_EDGE_CASES],
    )
    def test_umep_does_not_return_uniform(self, zen, azi, rad_d, rad_i, jday, patch_option, label):
        """UMEP does NOT return uniform at these edge cases (documenting divergence)."""
        umep_lv, _, _ = Perez_v3(zen, azi, rad_d, rad_i, jday, patchchoice=1, patch_option=patch_option)
        std_dev = np.std(umep_lv[:, 2])
        # UMEP produces non-uniform output here — our Rust port intentionally improves on this
        assert std_dev > 1e-6, f"[{label}] UMEP unexpectedly returns uniform (std={std_dev:.8f})"


class TestSteradiansParity:
    """Rust compute_steradians must match Python patch_steradians."""

    @pytest.mark.parametrize("patch_option", [1, 2, 3])
    def test_steradians_match_python(self, patch_option):
        """Rust steradians match Python for each patch option."""
        # Python reference: patch_steradians needs the lv array (uses column 0 only)
        umep_lv, _, _ = Perez_v3(30.0, 180.0, 200.0, 400.0, 180, patchchoice=1, patch_option=patch_option)
        py_ster, _, _ = patch_steradians(umep_lv)

        # Rust implementation
        rs_ster = np.asarray(pipeline.compute_steradians_py(patch_option))

        assert len(rs_ster) == len(py_ster), f"length mismatch: rs={len(rs_ster)} py={len(py_ster)}"
        np.testing.assert_allclose(
            rs_ster,
            py_ster.astype(np.float32),
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Steradians differ for patch_option={patch_option}",
        )

    @pytest.mark.parametrize("patch_option", [1, 2, 3])
    def test_steradians_positive(self, patch_option):
        """All steradian values should be positive."""
        rs_ster = np.asarray(pipeline.compute_steradians_py(patch_option))
        assert np.all(rs_ster > 0), "Found non-positive steradians"


class TestSteradiansCaching:
    """ShadowArrays.steradians cached property returns correct values."""

    @pytest.mark.parametrize("patch_option", [1, 2])
    def test_cached_steradians_match_direct(self, patch_option):
        """Cached steradians on ShadowArrays match direct computation."""
        from solweig.models.precomputed import ShadowArrays

        patch_map = {1: 145, 2: 153, 3: 305}
        n_patches = patch_map[patch_option]

        # Create a minimal ShadowArrays with the right patch count
        shape = (4, 4, n_patches)
        dummy = np.zeros(shape, dtype=np.uint8)
        sa = ShadowArrays(
            _shmat_u8=dummy,
            _vegshmat_u8=dummy,
            _vbshmat_u8=dummy,
            _n_patches=n_patches,
        )

        # Direct computation via UMEP + local patch_steradians
        umep_lv, _, _ = Perez_v3(30.0, 180.0, 200.0, 400.0, 180, patchchoice=1, patch_option=patch_option)
        py_ster, _, _ = patch_steradians(umep_lv)

        # Cached property
        cached_ster = sa.steradians

        assert len(cached_ster) == n_patches
        np.testing.assert_allclose(
            cached_ster,
            py_ster,
            rtol=1e-5,
            err_msg=f"Cached steradians differ from direct computation (patch_option={patch_option})",
        )

    def test_steradians_property_is_cached(self):
        """Second access returns the same object (no recomputation)."""
        from solweig.models.precomputed import ShadowArrays

        dummy = np.zeros((4, 4, 153), dtype=np.uint8)
        sa = ShadowArrays(_shmat_u8=dummy, _vegshmat_u8=dummy, _vbshmat_u8=dummy, _n_patches=153)
        first = sa.steradians
        second = sa.steradians
        assert first is second, "steradians property not cached — recomputed on second access"


class TestPatchCounts:
    """Rust create_patches returns correct patch counts for each option."""

    @pytest.mark.parametrize("patch_option,expected_count", [(1, 145), (2, 153), (3, 305)])
    def test_patch_count(self, patch_option, expected_count):
        """Rust patch count matches expected for each option."""
        rs_lv = np.asarray(pipeline.perez_v3_py(30.0, 180.0, 200.0, 400.0, 180, patch_option))
        assert rs_lv.shape[0] == expected_count, f"Expected {expected_count}, got {rs_lv.shape[0]}"
