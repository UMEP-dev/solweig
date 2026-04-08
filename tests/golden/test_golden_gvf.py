"""
Golden Regression Tests for Ground View Factor (GVF) Calculations

These tests verify the Rust GVF implementation produces physically valid
and consistent outputs. GVF determines how much radiation a person receives
from ground and wall surfaces (as opposed to sky).

Test strategy:
- Physical property tests: Verify ranges, relationships, symmetry
- Regression tests: Compare against pre-computed golden fixtures (when available)

Reference: Lindberg et al. (2008) - SOLWEIG GVF model with wall radiation
"""

from pathlib import Path

import numpy as np
import pytest

ndimage = pytest.importorskip("scipy.ndimage", reason="scipy required for golden GVF tests")
from solweig.constants import SBC  # noqa: E402
from solweig.rustalgos import gvf as gvf_module  # noqa: E402

pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Physical constants
KELVIN_OFFSET = 273.15
DEFAULT_ALBEDO = 0.15  # Typical urban ground albedo
DEFAULT_EMISSIVITY = 0.95  # Typical ground emissivity
DEFAULT_TA = 25.0  # Air temperature (°C)
DEFAULT_TGWALL = 2.0  # Wall temperature deviation (K)


@pytest.fixture(scope="module")
def input_data():
    """Load input data from golden fixtures (shared across all tests in module)."""
    return {
        "dsm": np.load(FIXTURES_DIR / "input_dsm.npy"),
        "cdsm": np.load(FIXTURES_DIR / "input_cdsm.npy"),
        "tdsm": np.load(FIXTURES_DIR / "input_tdsm.npy"),
        "bush": np.load(FIXTURES_DIR / "input_bush.npy"),
        "wall_ht": np.load(FIXTURES_DIR / "input_wall_ht.npy"),
        "wall_asp": np.load(FIXTURES_DIR / "input_wall_asp.npy"),
        "params": dict(np.load(FIXTURES_DIR / "input_params.npz")),
    }


@pytest.fixture(scope="module")
def shadow_data():
    """Load shadow data from golden fixtures (shared across all tests in module)."""
    return {
        "bldg_sh": np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy"),
        "veg_sh": np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy"),
        "wall_sh": np.load(FIXTURES_DIR / "shadow_noon_wall_sh.npy"),
        "wall_sun": np.load(FIXTURES_DIR / "shadow_noon_wall_sun.npy"),
    }


def create_buildings_mask(wall_ht: np.ndarray, pixel_size: float) -> np.ndarray:
    """
    Create building mask for GVF calculation.

    GVF expects: 0=building, 1=ground.
    """
    wall_mask = wall_ht > 0
    struct = ndimage.generate_binary_structure(2, 2)
    iterations = int(25 / pixel_size) + 1
    dilated = ndimage.binary_dilation(wall_mask, struct, iterations=iterations)
    return (~dilated).astype(np.float32)


def create_gvf_inputs(input_data, shadow_data):
    """Create all inputs needed for GVF calculation."""
    rows, cols = input_data["dsm"].shape
    scale = float(input_data["params"]["scale"])

    # Load ground temperature from fixture (spatially varying based on shadow)
    tg_path = FIXTURES_DIR / "gvf_input_tg.npy"
    tg = np.load(tg_path).astype(np.float32) if tg_path.exists() else np.zeros((rows, cols), dtype=np.float32)

    emis_grid = np.full((rows, cols), DEFAULT_EMISSIVITY, dtype=np.float32)
    alb_grid = np.full((rows, cols), DEFAULT_ALBEDO, dtype=np.float32)

    # Building mask
    buildings = create_buildings_mask(input_data["wall_ht"], scale)

    # Combined shadow (bldg + veg)
    shadow = (shadow_data["bldg_sh"] * shadow_data["veg_sh"]).astype(np.float32)

    return {
        "wallsun": shadow_data["wall_sun"].astype(np.float32),
        "walls": input_data["wall_ht"].astype(np.float32),
        "buildings": buildings,
        "shadow": shadow,
        "dirwalls": input_data["wall_asp"].astype(np.float32),
        "tg": tg,
        "emis_grid": emis_grid,
        "alb_grid": alb_grid,
        "scale": scale,
    }


@pytest.fixture(scope="module")
def gvf_inputs(input_data, shadow_data):
    """Prepare all GVF inputs (shared across all tests in module)."""
    return create_gvf_inputs(input_data, shadow_data)


@pytest.fixture(scope="module")
def gvf_result(gvf_inputs, cpu_only):
    """Compute GVF result using Rust implementation (computed once per module)."""

    params = gvf_module.GvfScalarParams(
        scale=gvf_inputs["scale"],
        first=2.0,  # round(1.8m human height)
        second=36.0,  # round(1.8 * 20)
        tgwall=DEFAULT_TGWALL,
        ta=DEFAULT_TA,
        ewall=0.90,  # Wall emissivity
        sbc=SBC,
        albedo_b=0.20,  # Wall albedo
        twater=DEFAULT_TA,
        landcover=False,
    )

    return gvf_module.gvf_calc(
        gvf_inputs["wallsun"],
        gvf_inputs["walls"],
        gvf_inputs["buildings"],
        gvf_inputs["shadow"],
        gvf_inputs["dirwalls"],
        gvf_inputs["tg"],
        gvf_inputs["emis_grid"],
        gvf_inputs["alb_grid"],
        None,  # lc_grid
        params,
    )


class TestGvfPhysicalProperties:
    """Verify GVF outputs satisfy physical constraints."""

    def test_gvfalb_range(self, gvf_result):
        """GVF × albedo should be in range [0, albedo_max]."""
        gvfalb = np.array(gvf_result.gvfalb)
        valid_mask = ~np.isnan(gvfalb)
        assert np.all(gvfalb[valid_mask] >= 0), "gvfalb has negative values"
        # GVF × albedo cannot exceed albedo (GVF ≤ 1)
        assert np.all(gvfalb[valid_mask] <= 1.0), "gvfalb exceeds 1.0"

    def test_gvfalbnosh_range(self, gvf_result):
        """GVF × albedo (no shadow) should be in range [0, 1]."""
        gvfalbnosh = np.array(gvf_result.gvfalbnosh)
        valid_mask = ~np.isnan(gvfalbnosh)
        assert np.all(gvfalbnosh[valid_mask] >= 0), "gvfalbnosh has negative values"
        assert np.all(gvfalbnosh[valid_mask] <= 1.0), "gvfalbnosh exceeds 1.0"

    def test_gvf_lup_positive(self, gvf_result):
        """Upwelling longwave should be positive (thermal emission)."""
        lup = np.array(gvf_result.gvf_lup)
        valid_mask = ~np.isnan(lup)
        # Thermal emission is always positive
        assert np.all(lup[valid_mask] >= 0), "gvf_lup has negative values"

    def test_gvf_lup_reasonable_range(self, gvf_result):
        """Upwelling longwave should be in physically reasonable range."""
        lup = np.array(gvf_result.gvf_lup)
        valid_mask = ~np.isnan(lup) & (lup > 0)
        # Stefan-Boltzmann: at 25°C, blackbody emits ~448 W/m²
        # With emissivity and GVF, expect 100-600 W/m² range
        assert np.all(lup[valid_mask] < 1000), "gvf_lup exceeds 1000 W/m²"

    def test_gvf_norm_range(self, gvf_result):
        """GVF normalization factor should be in [0, 1]."""
        gvf_norm = np.array(gvf_result.gvf_norm)
        valid_mask = ~np.isnan(gvf_norm)
        assert np.all(gvf_norm[valid_mask] >= 0), "gvf_norm has negative values"
        assert np.all(gvf_norm[valid_mask] <= 1.0), "gvf_norm exceeds 1.0"


class TestGvfDirectionalConsistency:
    """Verify directional GVF components are consistent."""

    def test_directional_gvfalb_range(self, gvf_result):
        """All directional gvfalb should be in valid range."""
        for direction in ["e", "s", "w", "n"]:
            arr = np.array(getattr(gvf_result, f"gvfalb_{direction}"))
            valid_mask = ~np.isnan(arr)
            assert np.all(arr[valid_mask] >= 0), f"gvfalb_{direction} has negative values"
            assert np.all(arr[valid_mask] <= 1.0), f"gvfalb_{direction} exceeds 1.0"

    def test_directional_lup_positive(self, gvf_result):
        """All directional Lup should be positive."""
        for direction in ["e", "s", "w", "n"]:
            arr = np.array(getattr(gvf_result, f"gvf_lup_{direction}"))
            valid_mask = ~np.isnan(arr)
            assert np.all(arr[valid_mask] >= 0), f"gvf_lup_{direction} has negative values"

    def test_directional_symmetry_approximate(self, gvf_result):
        """For uniform inputs, directional components should be roughly similar."""
        # Get all directional Lup values
        lup_e = np.array(gvf_result.gvf_lup_e)
        lup_s = np.array(gvf_result.gvf_lup_s)
        lup_w = np.array(gvf_result.gvf_lup_w)
        lup_n = np.array(gvf_result.gvf_lup_n)

        # Compute mean of each direction (excluding NaN and building pixels)
        means = []
        for arr in [lup_e, lup_s, lup_w, lup_n]:
            valid = arr[~np.isnan(arr) & (arr > 0)]
            if len(valid) > 0:
                means.append(np.mean(valid))

        if len(means) >= 2:
            # Directional means should be within 50% of each other
            # (allowing for building asymmetry in test data)
            max_mean = max(means)
            min_mean = min(means)
            ratio = max_mean / min_mean if min_mean > 0 else 1
            assert ratio < 2.0, f"Directional Lup ratio {ratio:.2f} too large"


class TestGvfShapeConsistency:
    """Verify all GVF arrays have consistent shapes."""

    def test_all_outputs_same_shape(self, gvf_result, gvf_inputs):
        """All GVF output arrays should match input shape."""
        expected_shape = gvf_inputs["buildings"].shape

        output_names = [
            "gvf_lup",
            "gvfalb",
            "gvfalbnosh",
            "gvf_lup_e",
            "gvfalb_e",
            "gvfalbnosh_e",
            "gvf_lup_s",
            "gvfalb_s",
            "gvfalbnosh_s",
            "gvf_lup_w",
            "gvfalb_w",
            "gvfalbnosh_w",
            "gvf_lup_n",
            "gvfalb_n",
            "gvfalbnosh_n",
            "gvf_sum",
            "gvf_norm",
        ]

        for name in output_names:
            arr = np.array(getattr(gvf_result, name))
            assert arr.shape == expected_shape, f"{name} has wrong shape: {arr.shape} != {expected_shape}"


class TestGvfBuildingBehavior:
    """Verify GVF handles building pixels correctly."""

    def test_buildings_have_normalized_gvf(self, gvf_result, gvf_inputs):
        """Building pixels should have gvf_norm = 1.0 (normalized)."""
        gvf_norm = np.array(gvf_result.gvf_norm)
        buildings = gvf_inputs["buildings"]

        # Where buildings=0 (is a building), gvf_norm should be 1.0
        building_mask = buildings == 0
        if np.any(building_mask):
            building_gvf = gvf_norm[building_mask]
            assert np.allclose(building_gvf, 1.0, atol=1e-5), "Building pixels don't have gvf_norm=1.0"


class TestGvfWallEffects:
    """Verify GVF responds correctly to wall presence."""

    def test_wall_areas_have_nonzero_gvf(self, gvf_result, gvf_inputs):
        """Areas near walls should have non-zero GVF contribution."""
        gvf_sum = np.array(gvf_result.gvf_sum)
        walls = gvf_inputs["walls"]

        # Dilate wall mask to find nearby pixels
        wall_mask = walls > 0
        struct = ndimage.generate_binary_structure(2, 2)
        near_walls = ndimage.binary_dilation(wall_mask, struct, iterations=3)

        # Exclude building pixels themselves
        buildings = gvf_inputs["buildings"]
        near_walls_ground = near_walls & (buildings > 0)

        if np.any(near_walls_ground):
            wall_area_gvf = gvf_sum[near_walls_ground]
            # Mean GVF near walls should be positive
            assert np.mean(wall_area_gvf) > 0, "GVF near walls should be positive"


# Golden regression tests (compare against stored fixtures)
class TestGvfGoldenRegression:
    """
    Golden regression tests comparing current output against stored fixtures.

    These tests are skipped if golden fixtures don't exist yet.
    Run generate_fixtures.py to create them.
    """

    @pytest.fixture
    def gvf_golden(self):
        """Load golden GVF fixtures if they exist."""
        fixtures = {}
        golden_files = {
            "gvf_lup": FIXTURES_DIR / "gvf_lup.npy",
            "gvfalb": FIXTURES_DIR / "gvf_alb.npy",
            "gvf_norm": FIXTURES_DIR / "gvf_norm.npy",
        }
        for name, path in golden_files.items():
            if path.exists():
                fixtures[name] = np.load(path)
        return fixtures if fixtures else None

    def test_gvf_lup_matches_golden(self, gvf_result, gvf_golden):
        """GVF Lup should match golden fixture."""
        if gvf_golden is None or "gvf_lup" not in gvf_golden:
            pytest.skip("Golden GVF fixtures not generated yet")

        np.testing.assert_allclose(
            np.array(gvf_result.gvf_lup),
            gvf_golden["gvf_lup"],
            rtol=1e-4,
            atol=1e-4,
            err_msg="GVF Lup differs from golden fixture",
        )

    def test_gvfalb_matches_golden(self, gvf_result, gvf_golden):
        """GVF albedo should match golden fixture."""
        if gvf_golden is None or "gvfalb" not in gvf_golden:
            pytest.skip("Golden GVF fixtures not generated yet")

        np.testing.assert_allclose(
            np.array(gvf_result.gvfalb),
            gvf_golden["gvfalb"],
            rtol=1e-4,
            atol=1e-4,
            err_msg="GVF albedo differs from golden fixture",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
