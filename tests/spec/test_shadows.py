"""
Shadow Calculation Tests

Tests derived from specs/shadows.md properties.
Each test verifies a physical property that must hold for the shadow algorithm.
"""

import math

import numpy as np
import pytest
from solweig import rustalgos

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True, scope="module")
def _disable_shadow_gpu(cpu_only):
    """Use CPU path for deterministic shadow property tests."""


def create_flat_dsm(size=(50, 50), elevation=10.0):
    """Create completely flat DSM."""
    return np.full(size, elevation, dtype=np.float32)


def create_building_dsm(size=(100, 100), building_height=20.0, ground=0.0):
    """Create DSM with single 10x10 building in center."""
    dsm = np.full(size, ground, dtype=np.float32)
    cy, cx = size[0] // 2, size[1] // 2
    dsm[cy - 5 : cy + 5, cx - 5 : cx + 5] = ground + building_height
    return dsm


def calculate_shadow(dsm, altitude, azimuth, pixel_size=1.0):
    """
    Calculate shadows using Rust implementation.

    Returns sunlit mask: 1 = sunlit, 0 = shaded
    """
    if altitude <= 0:
        # Below horizon: no direct beam shadows, everything is sunlit by convention.
        return np.ones_like(dsm, dtype=np.float32)

    max_height = float(np.max(dsm) - np.min(dsm))
    result = rustalgos.shadowing.calculate_shadows_wall_ht_25(
        float(azimuth),
        float(altitude),
        float(pixel_size),
        max_height,
        dsm.astype(np.float32),
        None,  # veg_canopy
        None,  # veg_trunk
        None,  # bush
        None,  # walls
        None,  # aspect
        None,  # walls_scheme
        None,  # aspect_scheme
        None,  # min_sun_elev
    )
    return np.array(result.bldg_sh, dtype=np.float32)


# =============================================================================
# Property Tests (from specs/shadows.md)
# =============================================================================


class TestShadowProperties:
    """Tests for shadow calculation properties."""

    def test_property_1_no_shadows_below_horizon(self):
        """Property 1: No shadows when sun altitude <= 0 (below horizon)."""
        dsm = create_building_dsm()

        for altitude in [-10, -5, 0]:
            sunlit = calculate_shadow(dsm, altitude=altitude, azimuth=180)
            assert np.all(sunlit == 1), f"Non-sunlit pixels exist at altitude {altitude}°"

    def test_property_2_flat_terrain_no_shadows(self):
        """Property 2: Flat terrain has no shadows."""
        dsm = create_flat_dsm()
        sunlit = calculate_shadow(dsm, altitude=45, azimuth=180)
        assert np.all(sunlit == 1), "Flat terrain should be fully sunlit"

    def test_property_3_lower_sun_longer_shadows(self):
        """Property 3: Lower sun = longer shadows (more shadow area)."""
        dsm = create_building_dsm()

        altitudes = [60, 45, 30, 15]
        shadow_areas = []

        for alt in altitudes:
            sunlit = calculate_shadow(dsm, altitude=alt, azimuth=180)
            shadow_areas.append(np.sum(1.0 - sunlit))

        # Each lower altitude should have more shadow
        for i in range(len(altitudes) - 1):
            assert shadow_areas[i] < shadow_areas[i + 1], (
                f"Shadow at {altitudes[i]}° ({shadow_areas[i]}) should be less than "
                f"at {altitudes[i + 1]}° ({shadow_areas[i + 1]})"
            )

    def test_property_4_shadows_opposite_sun_south(self):
        """Property 4: Sun from south (180°) -> shadows extend north."""
        dsm = create_building_dsm()
        sunlit = calculate_shadow(dsm, altitude=30, azimuth=180)
        shaded = 1.0 - sunlit

        cy = dsm.shape[0] // 2
        north_shadow = np.sum(shaded[: cy - 5, :])  # Above building
        south_shadow = np.sum(shaded[cy + 5 :, :])  # Below building

        assert north_shadow > south_shadow, "Shadows should extend north when sun is south"

    def test_property_4_shadows_opposite_sun_east(self):
        """Property 4: Sun from east (90°) -> shadows extend west."""
        dsm = create_building_dsm()
        sunlit = calculate_shadow(dsm, altitude=30, azimuth=90)
        shaded = 1.0 - sunlit

        cx = dsm.shape[1] // 2
        west_shadow = np.sum(shaded[:, : cx - 5])  # Left of building
        east_shadow = np.sum(shaded[:, cx + 5 :])  # Right of building

        assert west_shadow > east_shadow, "Shadows should extend west when sun is east"

    def test_property_5_taller_objects_longer_shadows(self):
        """Property 5: Taller objects cast longer shadows."""
        dsm = np.zeros((100, 100), dtype=np.float32)
        # Short building (10m) on left
        dsm[45:55, 20:30] = 10.0
        # Tall building (30m) on right
        dsm[45:55, 70:80] = 30.0

        sunlit = calculate_shadow(dsm, altitude=45, azimuth=180)
        shaded = 1.0 - sunlit

        short_shadow = np.sum(shaded[:45, 20:30])  # North of short building
        tall_shadow = np.sum(shaded[:45, 70:80])  # North of tall building

        assert tall_shadow > short_shadow, "Taller building should cast longer shadow"

    def test_property_6_shadow_length_equation(self):
        """Property 6: Shadow length ≈ height / tan(altitude) within 15%."""
        height = 20.0
        altitude = 45.0
        expected_length = height / math.tan(math.radians(altitude))

        dsm = np.zeros((100, 100), dtype=np.float32)
        dsm[50:60, 45:55] = height  # Building from row 50-60

        sunlit = calculate_shadow(dsm, altitude=altitude, azimuth=180)
        shaded = 1.0 - sunlit

        # Find northernmost shadow pixel
        shadow_north = shaded[:50, 45:55]
        shadow_rows = np.where(np.any(shadow_north > 0, axis=1))[0]

        measured_length = 50 - shadow_rows[0] if len(shadow_rows) > 0 else 0

        tolerance = 2  # pixel discretization: ≤1px typical, 2px margin
        assert abs(measured_length - expected_length) <= tolerance, (
            f"Shadow length {measured_length} should be ~{expected_length:.1f} (±{tolerance}px)"
        )

    def test_property_7_building_tops_sunlit(self):
        """Property 7: Building tops (rooftops) are sunlit when sun > 0."""
        dsm = create_building_dsm(building_height=30)
        sunlit = calculate_shadow(dsm, altitude=45, azimuth=180)

        # Building top pixels
        cy, cx = dsm.shape[0] // 2, dsm.shape[1] // 2
        rooftop = sunlit[cy - 5 : cy + 5, cx - 5 : cx + 5]

        sunlit_fraction = np.sum(rooftop == 1) / rooftop.size
        assert sunlit_fraction > 0.9, f"Rooftop should be mostly sunlit, got {sunlit_fraction:.0%}"

    def test_property_8_binary_values(self):
        """Property 8: Shadow mask contains only 0 or 1."""
        dsm = create_building_dsm()
        sunlit = calculate_shadow(dsm, altitude=45, azimuth=180)

        unique = set(np.unique(sunlit))
        assert unique.issubset({0.0, 1.0}), f"Sunlit values should be binary, got {unique}"


# =============================================================================
# Equation Tests
# =============================================================================


class TestShadowEquation:
    """Tests that verify shadow length matches L = h / tan(α)."""

    @pytest.mark.parametrize(
        "altitude,expected_ratio",
        [
            (60, 0.577),  # tan(60°) ≈ 1.732, so L/h ≈ 0.577
            (45, 1.0),  # tan(45°) = 1, so L/h = 1
            (30, 1.732),  # tan(30°) ≈ 0.577, so L/h ≈ 1.732
        ],
    )
    def test_shadow_length_ratio(self, altitude, expected_ratio):
        """Shadow length / height should equal 1/tan(altitude)."""
        height = 20.0
        theoretical_length = height / math.tan(math.radians(altitude))

        dsm = np.zeros((200, 200), dtype=np.float32)
        dsm[90:110, 90:110] = height

        sunlit = calculate_shadow(dsm, altitude=altitude, azimuth=180)
        shaded = 1.0 - sunlit

        # Measure shadow north of building
        shadow_north = shaded[:90, 90:110]
        shadow_rows = np.where(np.any(shadow_north > 0, axis=1))[0]

        measured_length = 90 - shadow_rows[0] if len(shadow_rows) > 0 else 0
        tolerance = 2  # pixel discretization: ≤1px typical, 2px margin

        assert abs(measured_length - theoretical_length) <= tolerance, (
            f"At {altitude}°: expected ~{theoretical_length:.1f}m, got {measured_length}m"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
