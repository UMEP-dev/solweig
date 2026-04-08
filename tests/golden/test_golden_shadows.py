"""
Golden Regression Tests for Shadow Calculations

These tests compare the Rust shadow algorithm implementation against
pre-computed golden fixtures generated from the UMEP Python module.

The fixtures are the ground truth (UMEP Python outputs), and these tests
verify that the Rust implementation produces equivalent results.

If these tests fail, it means:
- The Rust implementation differs from UMEP Python (investigate the difference)
- Or the fixtures were regenerated (rerun tests to confirm they pass)
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import shadowing

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True, scope="module")
def _cpu_shadows(cpu_only):
    """Use CPU path for deterministic golden shadow tests."""


@pytest.fixture(scope="module")
def input_data():
    """Load input data from golden fixtures (shared across all tests in module)."""
    return {
        "dsm": np.load(FIXTURES_DIR / "input_dsm.npy"),
        "cdsm": np.load(FIXTURES_DIR / "input_cdsm.npy"),
        "tdsm": np.load(FIXTURES_DIR / "input_tdsm.npy"),
        "bush": np.load(FIXTURES_DIR / "input_bush.npy"),
        "wall_ht": np.load(FIXTURES_DIR / "input_wall_ht.npy"),
        "wall_asp": np.load(FIXTURES_DIR / "input_wall_asp.npy") * np.pi / 180.0,
        "params": dict(np.load(FIXTURES_DIR / "input_params.npz")),
    }


def compute_shadows(input_data, azimuth, altitude):
    """Compute shadows with given sun position."""
    return shadowing.calculate_shadows_wall_ht_25(
        azimuth,
        altitude,
        float(input_data["params"]["scale"]),
        float(input_data["params"]["amaxvalue"]),
        input_data["dsm"],
        input_data["cdsm"],
        input_data["tdsm"],
        input_data["bush"],
        input_data["wall_ht"],
        input_data["wall_asp"].astype(np.float32),
        None,
        None,
        None,
    )


class TestGoldenShadowsMorning:
    """Golden tests for morning sun position (azimuth=90, altitude=30)."""

    @pytest.fixture(scope="class")
    def morning_golden(self):
        return {
            "bldg_sh": np.load(FIXTURES_DIR / "shadow_morning_bldg_sh.npy"),
            "veg_sh": np.load(FIXTURES_DIR / "shadow_morning_veg_sh.npy"),
            "wall_sh": np.load(FIXTURES_DIR / "shadow_morning_wall_sh.npy"),
            "wall_sun": np.load(FIXTURES_DIR / "shadow_morning_wall_sun.npy"),
        }

    @pytest.fixture(scope="class")
    def morning_result(self, input_data):
        return compute_shadows(input_data, azimuth=90.0, altitude=30.0)

    def test_bldg_sh_matches_golden(self, morning_result, morning_golden):
        """Building shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(morning_result.bldg_sh),
            morning_golden["bldg_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Building shadows differ from golden fixture",
        )

    def test_veg_sh_matches_golden(self, morning_result, morning_golden):
        """Vegetation shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(morning_result.veg_sh),
            morning_golden["veg_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Vegetation shadows differ from golden fixture",
        )

    def test_wall_sh_matches_golden(self, morning_result, morning_golden):
        """Wall shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(morning_result.wall_sh),
            morning_golden["wall_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Wall shadows differ from golden fixture",
        )

    def test_wall_sun_matches_golden(self, morning_result, morning_golden):
        """Wall sun should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(morning_result.wall_sun),
            morning_golden["wall_sun"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Wall sun differs from golden fixture",
        )


class TestGoldenShadowsNoon:
    """Golden tests for noon sun position (azimuth=180, altitude=60)."""

    @pytest.fixture(scope="class")
    def noon_golden(self):
        return {
            "bldg_sh": np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy"),
            "veg_sh": np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy"),
            "wall_sh": np.load(FIXTURES_DIR / "shadow_noon_wall_sh.npy"),
            "wall_sun": np.load(FIXTURES_DIR / "shadow_noon_wall_sun.npy"),
        }

    @pytest.fixture(scope="class")
    def noon_result(self, input_data):
        return compute_shadows(input_data, azimuth=180.0, altitude=60.0)

    def test_bldg_sh_matches_golden(self, noon_result, noon_golden):
        """Building shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(noon_result.bldg_sh),
            noon_golden["bldg_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Building shadows differ from golden fixture",
        )

    def test_veg_sh_matches_golden(self, noon_result, noon_golden):
        """Vegetation shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(noon_result.veg_sh),
            noon_golden["veg_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Vegetation shadows differ from golden fixture",
        )


class TestGoldenShadowsAfternoon:
    """Golden tests for afternoon sun position (azimuth=270, altitude=45)."""

    @pytest.fixture(scope="class")
    def afternoon_golden(self):
        return {
            "bldg_sh": np.load(FIXTURES_DIR / "shadow_afternoon_bldg_sh.npy"),
            "veg_sh": np.load(FIXTURES_DIR / "shadow_afternoon_veg_sh.npy"),
            "wall_sh": np.load(FIXTURES_DIR / "shadow_afternoon_wall_sh.npy"),
            "wall_sun": np.load(FIXTURES_DIR / "shadow_afternoon_wall_sun.npy"),
        }

    @pytest.fixture(scope="class")
    def afternoon_result(self, input_data):
        return compute_shadows(input_data, azimuth=270.0, altitude=45.0)

    def test_bldg_sh_matches_golden(self, afternoon_result, afternoon_golden):
        """Building shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(afternoon_result.bldg_sh),
            afternoon_golden["bldg_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Building shadows differ from golden fixture",
        )

    def test_veg_sh_matches_golden(self, afternoon_result, afternoon_golden):
        """Vegetation shadows should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(afternoon_result.veg_sh),
            afternoon_golden["veg_sh"],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Vegetation shadows differ from golden fixture",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
