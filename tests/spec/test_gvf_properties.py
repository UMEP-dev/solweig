"""
Ground View Factor (GVF) Physics Property Tests

Tests that GVF outputs satisfy physical invariants using synthetic data.
These complement the golden regression tests in tests/golden/test_golden_gvf.py.

GVF determines upwelling longwave radiation and reflected shortwave
a person receives from ground and wall surfaces.
"""

import numpy as np
import pytest
from solweig.constants import SBC
from solweig.rustalgos import gvf as gvf_module


@pytest.fixture(autouse=True, scope="module")
def _cpu_gvf(cpu_only):
    """Use CPU path for deterministic GVF property tests."""


# Grid must be large enough for GVF's internal neighbourhood kernel
GRID_SIZE = (80, 80)


def _make_gvf_inputs(
    shape=GRID_SIZE,
    wall_height=10.0,
    tg_val=2.0,
    alb=0.15,
    emis=0.95,
    shadow_val=1.0,
    ta=25.0,
    tgwall=2.0,
):
    """Create synthetic GVF inputs with a single building block at centre.

    Returns (arrays_dict, params) ready for gvf_calc.
    """
    walls = np.zeros(shape, dtype=np.float32)
    buildings = np.ones(shape, dtype=np.float32)
    dirwalls = np.zeros(shape, dtype=np.float32)

    # Place a 10x10 building in the centre
    cy, cx = shape[0] // 2, shape[1] // 2
    walls[cy - 5 : cy + 5, cx - 5 : cx + 5] = wall_height
    buildings[cy - 5 : cy + 5, cx - 5 : cx + 5] = 0.0  # 0 = building
    dirwalls[cy - 5 : cy + 5, cx - 5 : cx + 5] = 180.0

    wallsun = np.ones(shape, dtype=np.float32)
    shadow = np.full(shape, shadow_val, dtype=np.float32)
    tg = np.full(shape, tg_val, dtype=np.float32)
    emis_grid = np.full(shape, emis, dtype=np.float32)
    alb_grid = np.full(shape, alb, dtype=np.float32)

    params = gvf_module.GvfScalarParams(
        scale=1.0,
        first=2.0,
        second=36.0,
        tgwall=tgwall,
        ta=ta,
        ewall=0.90,
        sbc=SBC,
        albedo_b=0.20,
        twater=ta,
        landcover=False,
    )

    arrays = dict(
        wallsun=wallsun,
        walls=walls,
        buildings=buildings,
        shadow=shadow,
        dirwalls=dirwalls,
        tg=tg,
        emis_grid=emis_grid,
        alb_grid=alb_grid,
    )
    return arrays, params


def _run_gvf(**kwargs):
    """Run gvf_calc with synthetic inputs. Returns the GvfResult."""
    arrays, params = _make_gvf_inputs(**kwargs)
    return gvf_module.gvf_calc(
        arrays["wallsun"],
        arrays["walls"],
        arrays["buildings"],
        arrays["shadow"],
        arrays["dirwalls"],
        arrays["tg"],
        arrays["emis_grid"],
        arrays["alb_grid"],
        None,  # lc_grid
        params,
    )


class TestGvfTemperatureProperties:
    """GVF upwelling longwave should respond to temperature changes."""

    def test_higher_air_temp_higher_lup(self):
        """Higher air temperature produces higher upwelling longwave.

        Stefan-Boltzmann: Lup ~ epsilon * sigma * T^4, so warmer ground
        emits more thermal radiation.
        """
        result_cold = _run_gvf(ta=10.0)
        result_hot = _run_gvf(ta=35.0)

        lup_cold = np.array(result_cold.gvf_lup)
        lup_hot = np.array(result_hot.gvf_lup)

        # Compare at a ground pixel away from the building
        y, x = 10, 10
        assert lup_hot[y, x] > lup_cold[y, x], (
            f"Higher Ta should give higher Lup: cold={lup_cold[y, x]:.1f}, hot={lup_hot[y, x]:.1f}"
        )

    def test_higher_tg_deviation_higher_lup(self):
        """Higher ground temperature deviation produces higher upwelling longwave."""
        result_low = _run_gvf(tg_val=0.5)
        result_high = _run_gvf(tg_val=5.0)

        lup_low = np.array(result_low.gvf_lup)
        lup_high = np.array(result_high.gvf_lup)

        y, x = 10, 10
        assert lup_high[y, x] > lup_low[y, x], (
            f"Higher Tg deviation should give higher Lup: low={lup_low[y, x]:.1f}, high={lup_high[y, x]:.1f}"
        )


class TestGvfAlbedoProperties:
    """GVF reflected shortwave should respond to albedo changes."""

    def test_higher_albedo_higher_gvfalb(self):
        """Higher ground albedo produces higher reflected shortwave fraction."""
        result_dark = _run_gvf(alb=0.10)
        result_bright = _run_gvf(alb=0.40)

        alb_dark = np.array(result_dark.gvfalb)
        alb_bright = np.array(result_bright.gvfalb)

        y, x = 10, 10
        assert alb_bright[y, x] > alb_dark[y, x], (
            f"Higher albedo should give higher gvfalb: dark={alb_dark[y, x]:.4f}, bright={alb_bright[y, x]:.4f}"
        )

    def test_albedo_range(self):
        """gvfalb values should be in [0, 1]."""
        result = _run_gvf(alb=0.30)
        gvfalb = np.array(result.gvfalb)
        valid = gvfalb[~np.isnan(gvfalb)]
        assert np.all(valid >= 0), "gvfalb has negative values"
        assert np.all(valid <= 1.0), "gvfalb exceeds 1.0"


class TestGvfShadowProperties:
    """GVF reflected shortwave should respond to shadow state."""

    def test_shadow_eliminates_reflected_shortwave(self):
        """Fully shadowed ground should produce zero gvfalb."""
        result_sun = _run_gvf(shadow_val=1.0)
        result_shade = _run_gvf(shadow_val=0.0)

        alb_sun = np.array(result_sun.gvfalb)
        alb_shade = np.array(result_shade.gvfalb)

        y, x = 10, 10
        assert alb_sun[y, x] > 0, "Sunlit gvfalb should be positive"
        assert alb_shade[y, x] == 0.0, f"Shadowed gvfalb should be zero, got {alb_shade[y, x]:.6f}"

    def test_shadow_does_not_affect_longwave(self):
        """Shadow state should not eliminate longwave emission.

        Thermal emission depends on surface temperature, not direct sun.
        Lup may differ slightly due to shadow interaction with wall terms,
        but should remain positive in both cases.
        """
        result_sun = _run_gvf(shadow_val=1.0)
        result_shade = _run_gvf(shadow_val=0.0)

        lup_sun = np.array(result_sun.gvf_lup)
        lup_shade = np.array(result_shade.gvf_lup)

        y, x = 10, 10
        assert lup_sun[y, x] > 0, "Sunlit Lup should be positive"
        assert lup_shade[y, x] > 0, "Shaded Lup should still be positive"


class TestGvfOutputConsistency:
    """All GVF outputs should have consistent shapes and valid ranges."""

    def test_all_outputs_match_input_shape(self):
        """Every output array should match the input grid shape."""
        result = _run_gvf()
        for name in [
            "gvf_lup",
            "gvfalb",
            "gvfalbnosh",
            "gvf_norm",
            "gvf_sum",
            "gvf_lup_e",
            "gvf_lup_s",
            "gvf_lup_w",
            "gvf_lup_n",
            "gvfalb_e",
            "gvfalb_s",
            "gvfalb_w",
            "gvfalb_n",
        ]:
            arr = np.array(getattr(result, name))
            assert arr.shape == GRID_SIZE, f"{name} shape {arr.shape} != {GRID_SIZE}"

    def test_building_pixels_have_unit_norm(self):
        """Building pixels should have gvf_norm = 1.0."""
        arrays, params = _make_gvf_inputs()
        result = gvf_module.gvf_calc(
            arrays["wallsun"],
            arrays["walls"],
            arrays["buildings"],
            arrays["shadow"],
            arrays["dirwalls"],
            arrays["tg"],
            arrays["emis_grid"],
            arrays["alb_grid"],
            None,
            params,
        )

        gvf_norm = np.array(result.gvf_norm)
        building_mask = arrays["buildings"] == 0
        if np.any(building_mask):
            np.testing.assert_allclose(
                gvf_norm[building_mask],
                1.0,
                atol=1e-5,
                err_msg="Building pixels should have gvf_norm=1.0",
            )

    def test_lup_always_positive_on_ground(self):
        """Upwelling longwave should be positive for all ground pixels."""
        result = _run_gvf()
        lup = np.array(result.gvf_lup)
        arrays, _ = _make_gvf_inputs()
        ground_mask = arrays["buildings"] > 0
        ground_lup = lup[ground_mask]
        valid = ground_lup[~np.isnan(ground_lup)]
        assert np.all(valid >= 0), "Ground Lup should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
