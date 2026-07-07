"""Tests for the UMEP 2026a ground-scheme parameters and config flags.

Phase B.1 of the 2026a port: the five per-landcover parameter blocks
(vendored from UMEP-processing's parametersforsolweig.json) must load through
the standard loaders and stay aligned with the land-cover Names table, and
the opt-in flags must default off with the standard override precedence.
"""

from __future__ import annotations

import pytest
from solweig.loaders import load_params
from solweig.models.config import ModelConfig
from solweig.models.settings import Settings

# Land-cover classes the ground scheme parameterizes (wall materials 100-102
# are excluded upstream: the scheme covers ground + roofs, not wall voxels).
GROUND_CLASSES = [
    "Cobble_stone_2014a",
    "Dark_asphalt",
    "Roofs(buildings)",
    "Grass_unmanaged",
    "Bare_soil",
    "Water",
]


@pytest.fixture(scope="module")
def params():
    return load_params()


def test_scalar_blocks_cover_all_named_classes(params):
    """Heat capacity and thermal diffusivity carry every Names entry."""
    names = vars(params.Names.Value).values()
    for block in (params.__dict__["Heat capacity"], params.Thermal_diffusivity):
        values = vars(block.Value)
        for name in names:
            assert name in values, f"{name} missing from parameter block"


def test_ohm_coefficients_shape(params):
    """OHM block has [mean_a1, phase_a1, a2, a3] for each ground class."""
    values = vars(params.OHM_coefficients.Values)
    for name in GROUND_CLASSES:
        coeffs = values[name]
        assert len(coeffs) == 4, f"{name}: expected 4 OHM coefficients"
    # Spot-pin two values against the upstream source so silent edits fail
    assert values["Dark_asphalt"][0] == pytest.approx(0.5)
    assert values["Water"][3] == pytest.approx(-10.0)


@pytest.mark.parametrize("block_name", ["Tg_ini coefficients", "Tm_ini coefficients"])
def test_initial_temperature_blocks_shape(params, block_name):
    """Tg/Tm initialisation blocks carry 3 coefficients per ground class."""
    values = vars(params.__dict__[block_name].Values)
    for name in GROUND_CLASSES:
        assert len(values[name]) == 3, f"{name}: expected 3 coefficients in {block_name}"


def test_heat_capacity_physical_ranges(params):
    """Heat capacities are physically plausible (0.5-5 MJ m-3 K-1); water highest."""
    values = vars(params.__dict__["Heat capacity"].Value)
    ground = {k: v for k, v in values.items() if k in GROUND_CLASSES}
    for name, cap in ground.items():
        assert 5e5 <= cap <= 5e6, f"{name}: implausible heat capacity {cap}"
    assert max(ground, key=ground.get) == "Water"


def test_flags_default_off():
    settings = Settings.resolve()
    assert settings.use_ground_scheme is False
    assert settings.use_outgoing_longwave is False


def test_flags_resolve_precedence():
    """kwargs beat ModelConfig, ModelConfig beats defaults."""
    config = ModelConfig(use_ground_scheme=True, use_outgoing_longwave=True)
    from_config = Settings.resolve(config=config)
    assert from_config.use_ground_scheme is True
    assert from_config.use_outgoing_longwave is True

    overridden = Settings.resolve(config=config, use_ground_scheme=False, use_outgoing_longwave=False)
    assert overridden.use_ground_scheme is False
    assert overridden.use_outgoing_longwave is False
