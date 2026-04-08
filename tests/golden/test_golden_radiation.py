"""
Golden Regression Tests for Radiation (Kside/Lside) Calculations

These tests verify the Rust radiation implementations produce physically valid
and consistent outputs for directional shortwave (Kside) and longwave (Lside).

Test strategy:
- Physical property tests: Verify ranges, relationships, direction dependence
- Isotropic mode tests: Test simpler computation path (no shadow matrices needed)
- Regression tests: Compare against pre-computed golden fixtures (when available)

Reference:
- Lindberg et al. (2008, 2016) - SOLWEIG radiation model
- Perez et al. (1993) - Anisotropic sky luminance distribution
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.constants import SBC
from solweig.rustalgos import vegetation

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True, scope="module")
def _cpu_radiation(cpu_only):
    """Use CPU path for deterministic golden radiation tests."""


# Physical constants
KELVIN_OFFSET = 273.15

# Typical weather conditions for testing
DEFAULT_TA = 25.0  # Air temperature (°C)
DEFAULT_RADG = 800.0  # Global radiation (W/m²)
DEFAULT_RADI = 600.0  # Direct radiation (W/m²)
DEFAULT_RADD = 200.0  # Diffuse radiation (W/m²)
DEFAULT_ESKY = 0.75  # Sky emissivity
DEFAULT_CI = 0.85  # Clearness index


@pytest.fixture(scope="module")
def input_data():
    """Load input data from golden fixtures (shared across all tests in module)."""
    return {
        "dsm": np.load(FIXTURES_DIR / "input_dsm.npy"),
        "params": dict(np.load(FIXTURES_DIR / "input_params.npz")),
    }


@pytest.fixture(scope="module")
def svf_data():
    """Load SVF data from golden fixtures (shared across all tests in module)."""
    return {
        "svf": np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float32),
        "svf_north": np.load(FIXTURES_DIR / "svf_north.npy").astype(np.float32),
        "svf_east": np.load(FIXTURES_DIR / "svf_east.npy").astype(np.float32),
        "svf_south": np.load(FIXTURES_DIR / "svf_south.npy").astype(np.float32),
        "svf_west": np.load(FIXTURES_DIR / "svf_west.npy").astype(np.float32),
        "svf_veg": np.load(FIXTURES_DIR / "svf_veg.npy").astype(np.float32),
    }


@pytest.fixture(scope="module")
def shadow_data():
    """Load shadow data from golden fixtures (shared across all tests in module)."""
    return {
        "bldg_sh": np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float32),
        "veg_sh": np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float32),
    }


def create_kside_inputs(svf_data, shadow_data):
    """Create inputs for Kside calculation (isotropic mode)."""
    shape = svf_data["svf"].shape

    # Create synthetic Kup arrays (ground-reflected shortwave)
    kup_base = np.full(shape, 50.0, dtype=np.float32)  # ~50 W/m² reflected

    # Create F_sh array (fraction of shadow on walls)
    f_sh = np.full(shape, 0.5, dtype=np.float32)

    # Combined shadow
    shadow = (shadow_data["bldg_sh"] * shadow_data["veg_sh"]).astype(np.float32)

    # SVF vegetation: Use the actual svf_veg from fixtures
    # SVF_veg represents SVF accounting for vegetation transmissivity
    # It should satisfy: SVF_veg >= SVF (vegetation reduces but doesn't increase sky view)
    # And: svfvegbu = SVF_veg + SVF - 1 should be in [0, 1]
    # Using the actual svf_veg ensures correct relationships
    svf_veg = svf_data["svf_veg"]

    return {
        "shadow": shadow,
        "svf_s": svf_data["svf_south"],
        "svf_w": svf_data["svf_west"],
        "svf_n": svf_data["svf_north"],
        "svf_e": svf_data["svf_east"],
        # Use svf_veg for all directions (simplified - real code uses directional svf_veg)
        "svf_e_veg": svf_veg.copy(),
        "svf_s_veg": svf_veg.copy(),
        "svf_w_veg": svf_veg.copy(),
        "svf_n_veg": svf_veg.copy(),
        "f_sh": f_sh,
        "kup_e": kup_base.copy(),
        "kup_s": kup_base.copy(),
        "kup_w": kup_base.copy(),
        "kup_n": kup_base.copy(),
    }


def create_lside_inputs(svf_data, shadow_data):
    """Create inputs for Lside calculation."""
    shape = svf_data["svf"].shape

    # Compute Ldown (sky longwave)
    ta_k = DEFAULT_TA + KELVIN_OFFSET
    ldown_base = DEFAULT_ESKY * SBC * (ta_k**4)
    ldown = np.full(shape, ldown_base, dtype=np.float32)

    # Create F_sh array
    f_sh = np.full(shape, 0.5, dtype=np.float32)

    # Create Lup arrays (upwelling longwave from ground)
    lup_base = 0.95 * SBC * (ta_k**4)  # Ground emission
    lup = np.full(shape, lup_base, dtype=np.float32)

    # Use svf_veg for both SVF vegetation parameters
    # This ensures correct relationships: svfvegbu = svf_veg + svf - 1 stays in valid range
    svf_veg = svf_data["svf_veg"]

    return {
        "svf_s": svf_data["svf_south"],
        "svf_w": svf_data["svf_west"],
        "svf_n": svf_data["svf_north"],
        "svf_e": svf_data["svf_east"],
        # Use svf_veg for vegetation SVF (proper relationship with base SVF)
        "svf_e_veg": svf_veg.copy(),
        "svf_s_veg": svf_veg.copy(),
        "svf_w_veg": svf_veg.copy(),
        "svf_n_veg": svf_veg.copy(),
        # SVF_aveg (averaged vegetation SVF) - use same as svf_veg for testing
        "svf_e_aveg": svf_veg.copy(),
        "svf_s_aveg": svf_veg.copy(),
        "svf_w_aveg": svf_veg.copy(),
        "svf_n_aveg": svf_veg.copy(),
        "ldown": ldown,
        "f_sh": f_sh,
        "lup_e": lup.copy(),
        "lup_s": lup.copy(),
        "lup_w": lup.copy(),
        "lup_n": lup.copy(),
    }


@pytest.fixture(scope="module")
def kside_inputs(svf_data, shadow_data):
    """Prepare Kside inputs (shared across all tests in module)."""
    return create_kside_inputs(svf_data, shadow_data)


@pytest.fixture(scope="module")
def lside_inputs(svf_data, shadow_data):
    """Prepare Lside inputs (shared across all tests in module)."""
    return create_lside_inputs(svf_data, shadow_data)


@pytest.fixture(scope="module")
def kside_result(kside_inputs):
    """Compute Kside result using Rust implementation (computed once per module)."""
    return vegetation.kside_veg(
        DEFAULT_RADI,  # radI
        DEFAULT_RADD,  # radD
        DEFAULT_RADG,  # radG
        kside_inputs["shadow"],
        kside_inputs["svf_s"],
        kside_inputs["svf_w"],
        kside_inputs["svf_n"],
        kside_inputs["svf_e"],
        kside_inputs["svf_e_veg"],
        kside_inputs["svf_s_veg"],
        kside_inputs["svf_w_veg"],
        kside_inputs["svf_n_veg"],
        180.0,  # azimuth (noon)
        60.0,  # altitude (high sun)
        0.5,  # psi (vegetation transmissivity)
        0.0,  # t (instrument offset)
        0.20,  # albedo
        kside_inputs["f_sh"],
        kside_inputs["kup_e"],
        kside_inputs["kup_s"],
        kside_inputs["kup_w"],
        kside_inputs["kup_n"],
        True,  # cyl (cylindrical body model)
        None,  # lv (None for isotropic)
        False,  # anisotropic_diffuse
        None,  # diffsh
        None,  # asvf
        None,  # shmat
        None,  # vegshmat
        None,  # vbshvegshmat
    )


@pytest.fixture(scope="module")
def lside_result(lside_inputs):
    """Compute Lside result using Rust implementation (computed once per module)."""
    return vegetation.lside_veg(
        lside_inputs["svf_s"],
        lside_inputs["svf_w"],
        lside_inputs["svf_n"],
        lside_inputs["svf_e"],
        lside_inputs["svf_e_veg"],
        lside_inputs["svf_s_veg"],
        lside_inputs["svf_w_veg"],
        lside_inputs["svf_n_veg"],
        lside_inputs["svf_e_aveg"],
        lside_inputs["svf_s_aveg"],
        lside_inputs["svf_w_aveg"],
        lside_inputs["svf_n_aveg"],
        180.0,  # azimuth
        60.0,  # altitude
        DEFAULT_TA,  # Ta
        2.0,  # Tw (wall temperature deviation)
        SBC,  # Stefan-Boltzmann constant
        0.90,  # ewall
        lside_inputs["ldown"],
        DEFAULT_ESKY,  # esky
        0.0,  # t (instrument offset)
        lside_inputs["f_sh"],
        DEFAULT_CI,  # CI
        lside_inputs["lup_e"],
        lside_inputs["lup_s"],
        lside_inputs["lup_w"],
        lside_inputs["lup_n"],
        False,  # anisotropic_longwave
    )


class TestKsidePhysicalProperties:
    """Verify Kside outputs satisfy physical constraints."""

    def test_kside_i_non_negative(self, kside_result):
        """Direct component should be non-negative."""
        kside_i = np.array(kside_result.kside_i)
        valid_mask = ~np.isnan(kside_i)
        assert np.all(kside_i[valid_mask] >= 0), "kside_i has negative values"

    def test_kside_i_upper_bound(self, kside_result):
        """Direct component limited by incident radiation."""
        kside_i = np.array(kside_result.kside_i)
        valid_mask = ~np.isnan(kside_i)
        # Direct on vertical surface can't exceed I × cos(altitude)
        max_direct = DEFAULT_RADI * np.cos(np.radians(60.0))
        assert np.all(kside_i[valid_mask] <= max_direct * 1.1), f"kside_i exceeds physical maximum {max_direct}"

    def test_directional_kside_non_negative(self, kside_result):
        """All directional shortwave should be non-negative."""
        for direction in ["keast", "ksouth", "kwest", "knorth"]:
            arr = np.array(getattr(kside_result, direction))
            valid_mask = ~np.isnan(arr)
            assert np.all(arr[valid_mask] >= 0), f"{direction} has negative values"

    def test_directional_kside_reasonable_range(self, kside_result):
        """Directional shortwave should be in reasonable range."""
        for direction in ["keast", "ksouth", "kwest", "knorth"]:
            arr = np.array(getattr(kside_result, direction))
            valid_mask = ~np.isnan(arr) & (arr > 0)
            if np.any(valid_mask):
                # Shortwave on vertical surfaces can be high in areas with wall reflections
                # Typical range 0-500 W/m², but can exceed 2000 W/m² in complex geometries
                # Check median is reasonable (< 500 W/m²) rather than maximum
                median_val = np.median(arr[valid_mask])
                assert median_val < 500, f"{direction} median {median_val:.1f} exceeds 500 W/m²"


class TestKsideSunPositionDependence:
    """Verify Kside responds correctly to sun position."""

    def test_noon_south_dominates(self, kside_inputs):
        """At solar noon (azimuth=180), south-facing should receive most direct."""
        # Test at noon with high sun
        result = vegetation.kside_veg(
            DEFAULT_RADI,
            DEFAULT_RADD,
            DEFAULT_RADG,
            kside_inputs["shadow"],
            kside_inputs["svf_s"],
            kside_inputs["svf_w"],
            kside_inputs["svf_n"],
            kside_inputs["svf_e"],
            kside_inputs["svf_e_veg"],
            kside_inputs["svf_s_veg"],
            kside_inputs["svf_w_veg"],
            kside_inputs["svf_n_veg"],
            180.0,  # Noon
            60.0,
            0.5,
            0.0,
            0.20,
            kside_inputs["f_sh"],
            kside_inputs["kup_e"],
            kside_inputs["kup_s"],
            kside_inputs["kup_w"],
            kside_inputs["kup_n"],
            False,  # box model to see directional differences
            None,
            False,
            None,
            None,
            None,
            None,
            None,
        )

        ks = np.nanmean(np.array(result.ksouth))
        kn = np.nanmean(np.array(result.knorth))

        # At noon in Northern Hemisphere, south receives more than north
        # (with box model, direct beam goes to south-facing surfaces)
        assert ks >= kn, f"South ({ks:.1f}) should receive >= North ({kn:.1f}) at noon"

    def test_morning_east_receives_direct(self, kside_inputs):
        """In morning (azimuth=90), east-facing should receive direct."""
        result = vegetation.kside_veg(
            DEFAULT_RADI,
            DEFAULT_RADD,
            DEFAULT_RADG,
            kside_inputs["shadow"],
            kside_inputs["svf_s"],
            kside_inputs["svf_w"],
            kside_inputs["svf_n"],
            kside_inputs["svf_e"],
            kside_inputs["svf_e_veg"],
            kside_inputs["svf_s_veg"],
            kside_inputs["svf_w_veg"],
            kside_inputs["svf_n_veg"],
            90.0,  # Morning
            30.0,  # Lower sun
            0.5,
            0.0,
            0.20,
            kside_inputs["f_sh"],
            kside_inputs["kup_e"],
            kside_inputs["kup_s"],
            kside_inputs["kup_w"],
            kside_inputs["kup_n"],
            False,  # box model
            None,
            False,
            None,
            None,
            None,
            None,
            None,
        )

        ke = np.nanmean(np.array(result.keast))
        kw = np.nanmean(np.array(result.kwest))

        # In morning, east receives more than west
        assert ke >= kw, f"East ({ke:.1f}) should receive >= West ({kw:.1f}) in morning"


class TestLsidePhysicalProperties:
    """Verify Lside outputs satisfy physical constraints."""

    def test_lside_mostly_positive(self, lside_result):
        """Longwave radiation should be mostly positive.

        Note: A small percentage of pixels may have negative values due to
        numerical edge cases in the polynomial-based view factor calculation.
        This is a known limitation when SVF values are near extreme bounds.
        """
        for direction in ["least", "lsouth", "lwest", "lnorth"]:
            arr = np.array(getattr(lside_result, direction))
            valid_mask = ~np.isnan(arr)
            valid_vals = arr[valid_mask]
            negative_fraction = (valid_vals < 0).sum() / len(valid_vals)
            # Allow up to 1% negative values (numerical edge cases)
            assert negative_fraction < 0.01, (
                f"{direction} has {negative_fraction * 100:.1f}% negative values (max allowed: 1%)"
            )
            # Mean should definitely be positive
            assert np.mean(valid_vals) > 0, f"{direction} mean is negative"

    def test_lside_reasonable_range(self, lside_result):
        """Longwave should be in physically reasonable range."""
        for direction in ["least", "lsouth", "lwest", "lnorth"]:
            arr = np.array(getattr(lside_result, direction))
            valid_mask = ~np.isnan(arr) & (arr > 0)
            if np.any(valid_mask):
                # Longwave on vertical surfaces typically 100-600 W/m²
                assert np.all(arr[valid_mask] < 1000), f"{direction} exceeds 1000 W/m²"
                # Should be above freezing emission (~200 W/m² at 0°C)
                mean_val = np.mean(arr[valid_mask])
                assert mean_val > 100, f"{direction} mean too low: {mean_val:.1f}"


class TestLsideDirectionalConsistency:
    """Verify Lside directional components are consistent."""

    def test_directional_means_similar(self, lside_result):
        """Directional Lside means should be roughly similar (isotropic sky)."""
        means = []
        for direction in ["least", "lsouth", "lwest", "lnorth"]:
            arr = np.array(getattr(lside_result, direction))
            valid = arr[~np.isnan(arr) & (arr > 0)]
            if len(valid) > 0:
                means.append(np.mean(valid))

        if len(means) >= 2:
            # In isotropic mode, directional Lside should be similar
            max_mean = max(means)
            min_mean = min(means)
            ratio = max_mean / min_mean if min_mean > 0 else 1
            # Allow some variation due to SVF differences
            assert ratio < 2.0, f"Directional Lside ratio {ratio:.2f} too large"


class TestRadiationShapeConsistency:
    """Verify radiation arrays have consistent shapes."""

    def test_kside_shape_matches_input(self, kside_result, kside_inputs):
        """All Kside outputs should match input shape."""
        expected_shape = kside_inputs["shadow"].shape

        for attr in ["keast", "ksouth", "kwest", "knorth", "kside_i", "kside_d", "kside"]:
            arr = np.array(getattr(kside_result, attr))
            assert arr.shape == expected_shape, f"{attr} shape {arr.shape} != {expected_shape}"

    def test_lside_shape_matches_input(self, lside_result, lside_inputs):
        """All Lside outputs should match input shape."""
        expected_shape = lside_inputs["svf_e"].shape

        for attr in ["least", "lsouth", "lwest", "lnorth"]:
            arr = np.array(getattr(lside_result, attr))
            assert arr.shape == expected_shape, f"{attr} shape {arr.shape} != {expected_shape}"


class TestRadiationShadowEffects:
    """Verify radiation responds correctly to shadow conditions."""

    def test_shadow_reduces_direct(self, kside_inputs):
        """Shadows should reduce direct shortwave component."""
        # Fully sunlit
        kside_inputs_sunlit = kside_inputs.copy()
        kside_inputs_sunlit["shadow"] = np.ones_like(kside_inputs["shadow"])

        result_sunlit = vegetation.kside_veg(
            DEFAULT_RADI,
            DEFAULT_RADD,
            DEFAULT_RADG,
            kside_inputs_sunlit["shadow"],
            kside_inputs["svf_s"],
            kside_inputs["svf_w"],
            kside_inputs["svf_n"],
            kside_inputs["svf_e"],
            kside_inputs["svf_e_veg"],
            kside_inputs["svf_s_veg"],
            kside_inputs["svf_w_veg"],
            kside_inputs["svf_n_veg"],
            180.0,
            60.0,
            0.5,
            0.0,
            0.20,
            kside_inputs["f_sh"],
            kside_inputs["kup_e"],
            kside_inputs["kup_s"],
            kside_inputs["kup_w"],
            kside_inputs["kup_n"],
            True,
            None,
            False,
            None,
            None,
            None,
            None,
            None,
        )

        # Fully shaded
        kside_inputs_shaded = kside_inputs.copy()
        kside_inputs_shaded["shadow"] = np.zeros_like(kside_inputs["shadow"])

        result_shaded = vegetation.kside_veg(
            DEFAULT_RADI,
            DEFAULT_RADD,
            DEFAULT_RADG,
            kside_inputs_shaded["shadow"],
            kside_inputs["svf_s"],
            kside_inputs["svf_w"],
            kside_inputs["svf_n"],
            kside_inputs["svf_e"],
            kside_inputs["svf_e_veg"],
            kside_inputs["svf_s_veg"],
            kside_inputs["svf_w_veg"],
            kside_inputs["svf_n_veg"],
            180.0,
            60.0,
            0.5,
            0.0,
            0.20,
            kside_inputs["f_sh"],
            kside_inputs["kup_e"],
            kside_inputs["kup_s"],
            kside_inputs["kup_w"],
            kside_inputs["kup_n"],
            True,
            None,
            False,
            None,
            None,
            None,
            None,
            None,
        )

        kside_i_sunlit = np.nanmean(np.array(result_sunlit.kside_i))
        kside_i_shaded = np.nanmean(np.array(result_shaded.kside_i))

        # Shaded direct should be zero
        assert kside_i_shaded < 1.0, f"Shaded kside_i should be ~0, got {kside_i_shaded:.1f}"
        # Sunlit should be positive
        assert kside_i_sunlit > 100, f"Sunlit kside_i should be significant, got {kside_i_sunlit:.1f}"


# Golden regression tests
class TestRadiationGoldenRegression:
    """
    Golden regression tests comparing current output against stored fixtures.

    These tests are skipped if golden fixtures don't exist yet.
    Run generate_fixtures.py to create them.
    """

    @pytest.fixture
    def radiation_golden(self):
        """Load golden radiation fixtures if they exist."""
        fixtures = {}
        golden_files = {
            "kside_e": FIXTURES_DIR / "radiation_kside_e.npy",
            "kside_s": FIXTURES_DIR / "radiation_kside_s.npy",
            "lside_e": FIXTURES_DIR / "radiation_lside_e.npy",
            "lside_s": FIXTURES_DIR / "radiation_lside_s.npy",
        }
        for name, path in golden_files.items():
            if path.exists():
                fixtures[name] = np.load(path)
        return fixtures if fixtures else None

    def test_kside_matches_golden(self, kside_result, radiation_golden):
        """Kside should match golden fixtures."""
        if radiation_golden is None or "kside_e" not in radiation_golden:
            pytest.skip("Golden radiation fixtures not generated yet")

        np.testing.assert_allclose(
            np.array(kside_result.keast),
            radiation_golden["kside_e"],
            rtol=1e-4,
            atol=1e-4,
            err_msg="Kside east differs from golden fixture",
        )

    def test_lside_matches_golden(self, lside_result, radiation_golden):
        """Lside should match golden fixtures."""
        if radiation_golden is None or "lside_e" not in radiation_golden:
            pytest.skip("Golden radiation fixtures not generated yet")

        # Wider tolerance: golden fixtures use upstream SBC=5.67e-8,
        # our code uses the more accurate SBC=5.67051e-8 (CODATA 2018).
        np.testing.assert_allclose(
            np.array(lside_result.least),
            radiation_golden["lside_e"],
            rtol=2e-4,
            atol=0.1,
            err_msg="Lside east differs from golden fixture",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
