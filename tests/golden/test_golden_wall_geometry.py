"""
Golden Regression Tests for Wall Geometry Algorithms

These tests verify that the Rust wall-height / wall-aspect kernels
produce results consistent with the UMEP Python reference implementation.
Reference fixtures were generated using UMEP Python.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for wall geometry calculations
# Wall heights are integers (pixel counts), aspects are in degrees
WALL_HT_RTOL = 0.05  # 5% relative tolerance
WALL_HT_ATOL = 0.5  # 0.5m absolute tolerance
WALL_ASP_ATOL = 10.0  # 10 degrees for aspect (rotation quantization)


@pytest.fixture(scope="module")
def dsm():
    """Load DSM input fixture."""
    return np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32)


@pytest.fixture(scope="module")
def expected_wall_ht():
    """Load expected wall height from UMEP Python."""
    return np.load(FIXTURES_DIR / "input_wall_ht.npy").astype(np.float32)


@pytest.fixture(scope="module")
def expected_wall_asp():
    """Load expected wall aspect from UMEP Python."""
    return np.load(FIXTURES_DIR / "input_wall_asp.npy").astype(np.float32)


@pytest.fixture(scope="module")
def params():
    """Load input parameters."""
    return dict(np.load(FIXTURES_DIR / "input_params.npz"))


class TestGoldenWallHeight:
    """Golden tests for wall height detection."""

    def test_findwalls_produces_nonnegative_heights(self, dsm):
        """Wall heights should be non-negative."""
        from solweig.physics.wallalgorithms import findwalls

        walllimit = 2.0  # Minimum wall height to detect
        wall_ht = findwalls(dsm, walllimit)

        assert np.all(wall_ht >= 0), "Wall heights should be non-negative"

    def test_findwalls_shape_matches_dsm(self, dsm):
        """Output shape should match input DSM."""
        from solweig.physics.wallalgorithms import findwalls

        walllimit = 2.0
        wall_ht = findwalls(dsm, walllimit)

        assert wall_ht.shape == dsm.shape, "Wall height shape should match DSM"

    def test_findwalls_detects_walls_at_building_edges(self, dsm, expected_wall_ht):
        """Walls should be detected where expected (building edges)."""
        from solweig.physics.wallalgorithms import findwalls

        walllimit = 2.0
        wall_ht = findwalls(dsm, walllimit)

        # Compare number of wall pixels (within tolerance)
        expected_wall_count = np.sum(expected_wall_ht > 0)
        actual_wall_count = np.sum(wall_ht > 0)

        # Should detect similar number of walls (within 20%)
        ratio = actual_wall_count / max(expected_wall_count, 1)
        assert 0.8 <= ratio <= 1.2, (
            f"Wall pixel count differs too much: expected {expected_wall_count}, "
            f"got {actual_wall_count} (ratio={ratio:.2f})"
        )

    def test_findwalls_heights_match_golden(self, dsm, expected_wall_ht):
        """Wall heights should match golden reference within tolerance."""
        from solweig.physics.wallalgorithms import findwalls

        walllimit = 2.0
        wall_ht = findwalls(dsm, walllimit)

        # Only compare where both have walls
        both_have_walls = (wall_ht > 0) & (expected_wall_ht > 0)
        if np.any(both_have_walls):
            actual_heights = wall_ht[both_have_walls]
            expected_heights = expected_wall_ht[both_have_walls]

            np.testing.assert_allclose(
                actual_heights,
                expected_heights,
                rtol=WALL_HT_RTOL,
                atol=WALL_HT_ATOL,
                err_msg="Wall heights differ from golden reference",
            )


class TestGoldenWallAspect:
    """Golden tests for wall aspect (orientation) calculation."""

    def test_wall_aspect_range(self, dsm, params):
        """Wall aspects should be in valid range [0, 360)."""
        from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

        scale = float(params["scale"])
        walllimit = 2.0
        wall_ht = findwalls(dsm, walllimit)

        # Only run aspect calculation if we have walls
        if np.any(wall_ht > 0):
            wall_asp = filter1Goodwin_as_aspect_v3(wall_ht, scale, dsm)

            # Check range only where walls exist
            wall_mask = wall_ht > 0
            aspects_at_walls = wall_asp[wall_mask]

            # Aspects should be in [0, 360) range
            assert np.all(aspects_at_walls >= 0), "Wall aspects should be >= 0"
            assert np.all(aspects_at_walls < 360), "Wall aspects should be < 360"

    def test_wall_aspect_shape_matches_dsm(self, dsm, params):
        """Output shape should match input DSM."""
        from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

        scale = float(params["scale"])
        walllimit = 2.0
        wall_ht = findwalls(dsm, walllimit)
        wall_asp = filter1Goodwin_as_aspect_v3(wall_ht, scale, dsm)

        assert wall_asp.shape == dsm.shape, "Wall aspect shape should match DSM"

    @pytest.mark.slow
    def test_wall_aspect_matches_golden(self, dsm, expected_wall_ht, expected_wall_asp, params):
        """Wall aspects should match golden reference within tolerance.

        Note: This test is marked slow because aspect calculation iterates
        through 180 angles with filter convolutions.
        """
        from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

        scale = float(params["scale"])
        walllimit = 2.0
        wall_ht = findwalls(dsm, walllimit)
        wall_asp = filter1Goodwin_as_aspect_v3(wall_ht, scale, dsm)

        # Compare aspects where both reference and computed have walls
        both_have_walls = (wall_ht > 0) & (expected_wall_ht > 0)
        if np.any(both_have_walls):
            actual_asp = wall_asp[both_have_walls]
            expected_asp = expected_wall_asp[both_have_walls]

            # For angles, we need circular comparison
            # Difference should be small, accounting for 360 wrap-around
            diff = np.abs(actual_asp - expected_asp)
            diff = np.minimum(diff, 360 - diff)  # Handle wrap-around

            # Most aspects should match within tolerance
            matching = diff <= WALL_ASP_ATOL
            match_rate = np.mean(matching)

            assert match_rate >= 0.80, (
                f"Only {match_rate * 100:.1f}% of wall aspects match within "
                f"{WALL_ASP_ATOL}° tolerance (expected >= 80%)"
            )


class TestBinaryDilationConsistency:
    """Tests for the pure numpy binary_dilation implementation."""

    def test_single_pixel_dilation(self):
        """Single pixel should expand to 3x3 with 8-connectivity."""
        from solweig.physics.morphology import binary_dilation

        arr = np.array([[False, False, False], [False, True, False], [False, False, False]], dtype=bool)

        result = binary_dilation(arr, iterations=1)

        # With 8-connectivity, should expand to all neighbors
        expected = np.array([[True, True, True], [True, True, True], [True, True, True]], dtype=bool)

        np.testing.assert_array_equal(result, expected)

    def test_dilation_iterations(self):
        """Multiple iterations should expand further."""
        from solweig.physics.morphology import binary_dilation

        arr = np.zeros((7, 7), dtype=bool)
        arr[3, 3] = True

        result_1 = binary_dilation(arr, iterations=1)
        result_2 = binary_dilation(arr, iterations=2)

        # More iterations = more expansion
        assert np.sum(result_2) > np.sum(result_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
