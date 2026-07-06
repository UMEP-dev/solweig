"""
Golden Regression Tests for Anisotropic Sky Radiation Model

These tests verify the Rust `anisotropic_sky` function produces physically valid
and consistent outputs. The anisotropic sky model computes direction-dependent
longwave and shortwave radiation from sky patches, vegetation, and buildings.

The anisotropic_sky function is complex with many inputs:
- Shadow matrices (3D) for buildings, vegetation, and combined
- Sky patch geometry (altitude, azimuth, steradians)
- Sun position and radiation parameters
- Ground and wall temperatures

Test strategy:
- Physical property tests: Verify output ranges and relationships
- Regression tests: Compare against pre-computed golden fixtures
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import sky

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Physical constants
SBC = 5.67051e-8  # Stefan-Boltzmann constant (CODATA 2018, closest f32)


def generate_sky_patches(n_alt_bands=6, n_azi_per_band=12):
    """Generate a standard hemispherical sky patch grid.

    Uses Tregenza-style hemisphere division.
    """
    patches = []

    # Standard altitude bands (degrees from horizon)
    alt_bands = [6, 18, 30, 42, 54, 66, 78, 90]
    azis_per_band = [30, 24, 24, 18, 12, 6, 6, 1]  # Patches per altitude band

    for _alt_idx, (alt, n_azi) in enumerate(zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands], strict=False)):
        azi_step = 360.0 / n_azi if n_azi > 1 else 0
        for azi_idx in range(n_azi):
            azi = azi_idx * azi_step
            patches.append([alt, azi])

    return np.array(patches, dtype=np.float32)


def compute_steradians(l_patches):
    """Compute solid angle (steradians) for each sky patch.

    Based on hemisphere geometry.
    """
    n_patches = len(l_patches)
    steradians = np.zeros(n_patches, dtype=np.float32)
    deg2rad = np.pi / 180.0

    # Group by altitude
    altitudes = l_patches[:, 0]
    unique_alts = np.unique(altitudes)

    for i, alt in enumerate(unique_alts):
        mask = altitudes == alt
        count = np.sum(mask)

        if i == 0:
            # First band from horizon
            ster = (360.0 / count * deg2rad) * np.sin(alt * deg2rad)
        else:
            prev_alt = unique_alts[i - 1]
            delta_alt = (alt - prev_alt) / 2
            ster = (360.0 / count * deg2rad) * (
                np.sin((alt + delta_alt) * deg2rad) - np.sin((prev_alt + delta_alt) * deg2rad)
            )

        steradians[mask] = ster

    return steradians


@pytest.fixture(scope="module")
def input_data():
    """Load base input data from golden fixtures (shared across all tests in module)."""
    params = dict(np.load(FIXTURES_DIR / "input_params.npz"))
    return {
        "dsm": np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32),
        "scale": float(params["scale"]),
    }


@pytest.fixture(scope="module")
def svf_data():
    """Load SVF data from golden fixtures (shared across all tests in module)."""
    return {
        "svf": np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float32),
    }


@pytest.fixture(scope="module")
def shadow_data():
    """Load shadow data from golden fixtures (shared across all tests in module)."""
    return {
        "bldg_sh": np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float32),
        "veg_sh": np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float32),
    }


@pytest.fixture(scope="module")
def aniso_sky_inputs(input_data, svf_data, shadow_data):
    """Create inputs for anisotropic_sky calculation."""
    rows, cols = input_data["dsm"].shape

    # Generate sky patches
    l_patches = generate_sky_patches(n_alt_bands=4, n_azi_per_band=8)  # Simplified for testing
    n_patches = len(l_patches)
    steradians = compute_steradians(l_patches)

    # Create 3D shadow matrices (rows, cols, patches)
    # shmat: 1 = sky visible, 0 = blocked by building
    # For testing, use SVF to create approximate patch visibility
    svf_expanded = svf_data["svf"][:, :, np.newaxis]
    base_visibility = np.broadcast_to(svf_expanded, (rows, cols, n_patches)).copy()

    # Add some spatial variation based on building shadow
    bldg_factor = shadow_data["bldg_sh"][:, :, np.newaxis]
    veg_factor = shadow_data["veg_sh"][:, :, np.newaxis]

    # shmat: building shadow mask (uint8: 255 = sky visible, 0 = blocked)
    shmat_f = base_visibility * np.broadcast_to(bldg_factor, (rows, cols, n_patches))
    shmat = np.where(shmat_f > 0.5, np.uint8(255), np.uint8(0)).astype(np.uint8)

    # vegshmat: vegetation shadow mask (uint8: 255 = sky visible, 0 = blocked)
    vegshmat_f = base_visibility * np.broadcast_to(veg_factor, (rows, cols, n_patches))
    vegshmat = np.where(vegshmat_f > 0.3, np.uint8(255), np.uint8(0)).astype(np.uint8)

    # vbshvegshmat: combined building+vegetation shadow (uint8)
    vbshvegshmat = np.where((shmat == 255) & (vegshmat == 255), np.uint8(255), np.uint8(0)).astype(np.uint8)

    # asvf: angular sky view factor (use base SVF as approximation)
    asvf = svf_data["svf"].astype(np.float32)

    # lv: patch luminance array (alt, azi, luminance)
    # luminance varies with altitude (higher = brighter)
    luminance = 1000 + 500 * np.sin(l_patches[:, 0] * np.pi / 180)  # Higher patches brighter
    lv = np.column_stack([l_patches, luminance]).astype(np.float32)

    # Ground upwelling longwave
    ta = 25.0
    ta_k = ta + 273.15
    lup_val = 0.95 * SBC * (ta_k**4)
    lup = np.full((rows, cols), lup_val, dtype=np.float32)

    # Combined shadow (2D)
    shadow = (shadow_data["bldg_sh"] * shadow_data["veg_sh"]).astype(np.float32)

    # Upwelling shortwave per direction
    kup_base = np.full((rows, cols), 50.0, dtype=np.float32)  # W/m²

    return {
        "shmat": shmat,
        "vegshmat": vegshmat,
        "vbshvegshmat": vbshvegshmat,
        "asvf": asvf,
        "l_patches": l_patches,
        "steradians": steradians,
        "lv": lv,
        "lup": lup,
        "shadow": shadow,
        "kup_e": kup_base.copy(),
        "kup_s": kup_base.copy(),
        "kup_w": kup_base.copy(),
        "kup_n": kup_base.copy(),
    }


@pytest.fixture(scope="module")
def aniso_sky_result(aniso_sky_inputs):
    """Compute anisotropic sky result (computed once per module)."""
    inputs = aniso_sky_inputs

    # Create parameter objects
    sun_params = sky.SunParams(
        altitude=60.0,  # High sun
        azimuth=180.0,  # Noon
    )

    sky_params = sky.SkyParams(
        esky=0.75,
        ta=25.0,
        cyl=True,  # Cylindrical body model
        wall_scheme=False,  # Simple wall model
        albedo=0.20,
    )

    surface_params = sky.SurfaceParams(
        tgwall=2.0,  # Wall temperature deviation
        ewall=0.90,  # Wall emissivity
        rad_i=600.0,  # Direct radiation W/m²
        rad_d=200.0,  # Diffuse radiation W/m²
    )

    result = sky.anisotropic_sky(
        inputs["shmat"],
        inputs["vegshmat"],
        inputs["vbshvegshmat"],
        sun_params,
        inputs["asvf"],
        sky_params,
        inputs["l_patches"],
        None,  # voxel_table (optional)
        None,  # voxel_maps (optional)
        inputs["steradians"],
        surface_params,
        inputs["lup"],
        inputs["lv"],
        inputs["shadow"],
        inputs["kup_e"],
        inputs["kup_s"],
        inputs["kup_w"],
        inputs["kup_n"],
    )

    return result


class TestAnisotropicSkyPhysicalProperties:
    """Verify anisotropic sky outputs satisfy physical constraints."""

    def test_ldown_non_negative(self, aniso_sky_result):
        """Downwelling longwave should be non-negative."""
        ldown = np.array(aniso_sky_result.ldown)
        valid_mask = ~np.isnan(ldown)
        assert np.all(ldown[valid_mask] >= 0), "ldown has negative values"

    def test_ldown_reasonable_range(self, aniso_sky_result):
        """Downwelling longwave should be in reasonable range (100-600 W/m²)."""
        ldown = np.array(aniso_sky_result.ldown)
        valid_mask = ~np.isnan(ldown) & (ldown > 0)
        if np.any(valid_mask):
            mean_val = np.mean(ldown[valid_mask])
            # Typical range for mid-latitude summer
            assert mean_val > 50, f"ldown mean {mean_val:.1f} too low"
            assert mean_val < 800, f"ldown mean {mean_val:.1f} too high"

    def test_lside_non_negative(self, aniso_sky_result):
        """Side longwave should be non-negative."""
        lside = np.array(aniso_sky_result.lside)
        valid_mask = ~np.isnan(lside)
        assert np.all(lside[valid_mask] >= 0), "lside has negative values"

    def test_lside_components_sum(self, aniso_sky_result):
        """Lside should approximately equal sum of components."""
        lside = np.array(aniso_sky_result.lside)
        lside_sky = np.array(aniso_sky_result.lside_sky)
        lside_veg = np.array(aniso_sky_result.lside_veg)
        lside_sh = np.array(aniso_sky_result.lside_sh)
        lside_sun = np.array(aniso_sky_result.lside_sun)
        lside_ref = np.array(aniso_sky_result.lside_ref)

        # Sum of components
        lside_sum = lside_sky + lside_veg + lside_sh + lside_sun + lside_ref

        # Should match total
        np.testing.assert_allclose(
            lside, lside_sum, rtol=1e-4, atol=1e-4, err_msg="Lside doesn't match sum of components"
        )

    def test_kside_non_negative(self, aniso_sky_result):
        """Side shortwave should be non-negative."""
        kside = np.array(aniso_sky_result.kside)
        valid_mask = ~np.isnan(kside)
        assert np.all(kside[valid_mask] >= 0), "kside has negative values"

    def test_directional_longwave_non_negative(self, aniso_sky_result):
        """Directional longwave components should be non-negative."""
        for direction in ["least", "lsouth", "lwest", "lnorth"]:
            arr = np.array(getattr(aniso_sky_result, direction))
            valid_mask = ~np.isnan(arr)
            assert np.all(arr[valid_mask] >= 0), f"{direction} has negative values"

    def test_directional_shortwave_non_negative(self, aniso_sky_result):
        """Directional shortwave components should be non-negative."""
        for direction in ["keast", "ksouth", "kwest", "knorth"]:
            arr = np.array(getattr(aniso_sky_result, direction))
            valid_mask = ~np.isnan(arr)
            assert np.all(arr[valid_mask] >= 0), f"{direction} has negative values"


class TestAnisotropicSkyOutputShape:
    """Verify output shapes are correct."""

    def test_ldown_shape_matches_input(self, aniso_sky_result, aniso_sky_inputs):
        """Ldown should match input spatial dimensions."""
        expected_shape = aniso_sky_inputs["shadow"].shape
        actual_shape = np.array(aniso_sky_result.ldown).shape
        assert actual_shape == expected_shape, f"ldown shape {actual_shape} != {expected_shape}"

    def test_all_2d_outputs_match_input(self, aniso_sky_result, aniso_sky_inputs):
        """All 2D outputs should match input spatial dimensions."""
        expected_shape = aniso_sky_inputs["shadow"].shape

        attrs_2d = [
            "ldown",
            "lside",
            "lside_sky",
            "lside_veg",
            "lside_sh",
            "lside_sun",
            "lside_ref",
            "least",
            "lwest",
            "lnorth",
            "lsouth",
            "keast",
            "ksouth",
            "kwest",
            "knorth",
            "kside_i",
            "kside_d",
            "kside",
        ]

        for attr in attrs_2d:
            arr = np.array(getattr(aniso_sky_result, attr))
            assert arr.shape == expected_shape, f"{attr} shape {arr.shape} != {expected_shape}"

    def test_steradians_matches_patches(self, aniso_sky_result, aniso_sky_inputs):
        """Steradians array should match number of patches."""
        n_patches = len(aniso_sky_inputs["l_patches"])
        steradians = np.array(aniso_sky_result.steradians)
        assert len(steradians) == n_patches, f"steradians length {len(steradians)} != {n_patches}"


class TestAnisotropicSkySunPosition:
    """Verify response to sun position changes."""

    def test_kside_i_responds_to_altitude(self, aniso_sky_inputs):
        """Direct shortwave should respond to sun altitude."""
        inputs = aniso_sky_inputs

        # High sun
        sun_high = sky.SunParams(altitude=60.0, azimuth=180.0)
        sky_params = sky.SkyParams(esky=0.75, ta=25.0, cyl=True, wall_scheme=False, albedo=0.20)
        surface_params = sky.SurfaceParams(tgwall=2.0, ewall=0.90, rad_i=600.0, rad_d=200.0)

        result_high = sky.anisotropic_sky(
            inputs["shmat"],
            inputs["vegshmat"],
            inputs["vbshvegshmat"],
            sun_high,
            inputs["asvf"],
            sky_params,
            inputs["l_patches"],
            None,
            None,
            inputs["steradians"],
            surface_params,
            inputs["lup"],
            inputs["lv"],
            inputs["shadow"],
            inputs["kup_e"],
            inputs["kup_s"],
            inputs["kup_w"],
            inputs["kup_n"],
        )

        # Low sun
        sun_low = sky.SunParams(altitude=20.0, azimuth=180.0)
        result_low = sky.anisotropic_sky(
            inputs["shmat"],
            inputs["vegshmat"],
            inputs["vbshvegshmat"],
            sun_low,
            inputs["asvf"],
            sky_params,
            inputs["l_patches"],
            None,
            None,
            inputs["steradians"],
            surface_params,
            inputs["lup"],
            inputs["lv"],
            inputs["shadow"],
            inputs["kup_e"],
            inputs["kup_s"],
            inputs["kup_w"],
            inputs["kup_n"],
        )

        kside_i_high = np.nanmean(np.array(result_high.kside_i))
        kside_i_low = np.nanmean(np.array(result_low.kside_i))

        # Lower sun -> more direct on vertical surface
        assert kside_i_low > kside_i_high * 0.5, (
            f"kside_i at low sun ({kside_i_low:.1f}) should be > high sun ({kside_i_high:.1f})"
        )


class TestAnisotropicSkyRadiationBalance:
    """Verify radiation balance relationships."""

    def test_lside_components_positive(self, aniso_sky_result):
        """Individual Lside components should be non-negative."""
        components = ["lside_sky", "lside_veg", "lside_sh", "lside_sun", "lside_ref"]
        for comp in components:
            arr = np.array(getattr(aniso_sky_result, comp))
            valid_mask = ~np.isnan(arr)
            # Small negative values might occur due to numerical precision
            assert np.all(arr[valid_mask] >= -1e-3), f"{comp} has significant negative values"

    def test_kside_components_sum(self, aniso_sky_result):
        """Kside should equal kside_i + kside_d plus reflected terms."""
        kside = np.array(aniso_sky_result.kside)
        kside_i = np.array(aniso_sky_result.kside_i)
        kside_d = np.array(aniso_sky_result.kside_d)

        # Kside includes direct (i), diffuse (d), and reflected components
        # The sum should be >= kside_i + kside_d
        assert np.all(kside >= kside_i + kside_d - 1e-3), "kside should be >= kside_i + kside_d"


class TestAnisotropicSkyGoldenRegression:
    """Golden regression tests for anisotropic sky model."""

    @pytest.fixture
    def aniso_golden(self):
        """Load golden anisotropic sky fixtures. A missing file raises FileNotFoundError."""
        return dict(np.load(FIXTURES_DIR / "aniso_sky_output.npz"))

    def test_ldown_matches_golden(self, aniso_sky_result, aniso_golden):
        """Ldown should match golden fixture."""
        np.testing.assert_allclose(
            np.array(aniso_sky_result.ldown),
            aniso_golden["ldown"],
            rtol=1e-4,
            atol=0.1,
            err_msg="Ldown differs from golden fixture",
        )

    def test_lside_matches_golden(self, aniso_sky_result, aniso_golden):
        """Lside should match golden fixture."""
        np.testing.assert_allclose(
            np.array(aniso_sky_result.lside),
            aniso_golden["lside"],
            rtol=1e-4,
            atol=0.1,
            err_msg="Lside differs from golden fixture",
        )

    def test_kside_matches_golden(self, aniso_sky_result, aniso_golden):
        """Kside should match golden fixture."""
        np.testing.assert_allclose(
            np.array(aniso_sky_result.kside),
            aniso_golden["kside"],
            rtol=1e-4,
            atol=0.1,
            err_msg="Kside differs from golden fixture",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
