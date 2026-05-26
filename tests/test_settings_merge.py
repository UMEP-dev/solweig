"""Tests for `solweig.models.settings.Settings` merge semantics.

Locks the precedence (kwarg > config > default) and the lazy-default rules
for physics/materials. Wraps the merge logic that used to be 50 lines of
inline code in `api._calculate_single`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from solweig.models.config import HumanParams, ModelConfig
from solweig.models.settings import Settings

# ── kwarg > config > default precedence ─────────────────────────────────────


def test_resolve_all_none_yields_defaults():
    s = Settings.resolve()
    assert s.use_anisotropic_sky is True
    assert s.conifer is False
    assert s.wall_material is None
    assert s.max_shadow_distance_m == 1000.0
    assert isinstance(s.human, HumanParams)
    assert s.physics is None
    assert s.materials is None


def test_kwarg_overrides_default():
    s = Settings.resolve(use_anisotropic_sky=False, max_shadow_distance_m=500.0, conifer=True)
    assert s.use_anisotropic_sky is False
    assert s.max_shadow_distance_m == 500.0
    assert s.conifer is True


def test_config_overrides_default():
    cfg = ModelConfig(use_anisotropic_sky=False, max_shadow_distance_m=200.0)
    s = Settings.resolve(config=cfg)
    assert s.use_anisotropic_sky is False
    assert s.max_shadow_distance_m == 200.0


def test_kwarg_overrides_config():
    cfg = ModelConfig(use_anisotropic_sky=False, max_shadow_distance_m=200.0)
    s = Settings.resolve(config=cfg, use_anisotropic_sky=True, max_shadow_distance_m=750.0)
    assert s.use_anisotropic_sky is True
    assert s.max_shadow_distance_m == 750.0


def test_kwarg_none_falls_through_to_config():
    """None at the kwarg level must NOT clobber a config value."""
    cfg = ModelConfig(use_anisotropic_sky=False, max_shadow_distance_m=200.0)
    s = Settings.resolve(config=cfg, use_anisotropic_sky=None, max_shadow_distance_m=None)
    assert s.use_anisotropic_sky is False
    assert s.max_shadow_distance_m == 200.0


# ── Field-by-field behaviour ────────────────────────────────────────────────


def test_human_kwarg_used():
    h = HumanParams(abs_k=0.5)
    s = Settings.resolve(human=h)
    assert s.human is h


def test_human_from_config_when_kwarg_none():
    h = HumanParams(abs_k=0.6)
    cfg = ModelConfig(human=h)
    s = Settings.resolve(config=cfg)
    assert s.human is h


def test_human_default_when_both_none():
    s = Settings.resolve()
    assert isinstance(s.human, HumanParams)
    assert s.human.abs_k == 0.7  # dataclass default


def test_wall_material_passes_through_none():
    """wall_material=None means 'use the bundled materials.Walls defaults',
    not 'inherit'. After resolution, None is still a valid value."""
    s = Settings.resolve(wall_material=None)
    assert s.wall_material is None
    s2 = Settings.resolve(wall_material="brick")
    assert s2.wall_material == "brick"


def test_physics_materials_stay_none_after_resolve():
    """resolve() does NOT load bundled defaults — that's with_loaded_defaults()'s job."""
    s = Settings.resolve()
    assert s.physics is None
    assert s.materials is None


def test_physics_materials_kwargs_preserved():
    ns = SimpleNamespace(foo="bar")
    s = Settings.resolve(physics=ns, materials=ns)
    assert s.physics is ns
    assert s.materials is ns


# ── with_loaded_defaults() lazy-loads bundled JSON ──────────────────────────


def test_with_loaded_defaults_loads_when_none():
    s = Settings.resolve().with_loaded_defaults()
    assert s.physics is not None
    assert s.materials is not None


def test_with_loaded_defaults_preserves_explicit():
    explicit_physics = SimpleNamespace(custom=True)
    explicit_materials = SimpleNamespace(custom=True)
    s = Settings.resolve(physics=explicit_physics, materials=explicit_materials).with_loaded_defaults()
    assert s.physics is explicit_physics
    assert s.materials is explicit_materials


# ── Immutability ────────────────────────────────────────────────────────────


def test_settings_is_frozen():
    s = Settings.resolve()
    with pytest.raises(FrozenInstanceError):
        s.use_anisotropic_sky = False  # type: ignore[misc]


# ── Sanity: nothing odd happens with both config and kwargs all set ─────────


def test_full_override_chain():
    cfg = ModelConfig(
        use_anisotropic_sky=False,
        max_shadow_distance_m=100.0,
        human=HumanParams(abs_k=0.6),
    )
    s = Settings.resolve(
        config=cfg,
        use_anisotropic_sky=True,
        max_shadow_distance_m=500.0,
        conifer=True,
        wall_material="concrete",
        human=HumanParams(abs_k=0.9),
    )
    assert s.use_anisotropic_sky is True
    assert s.max_shadow_distance_m == 500.0
    assert s.conifer is True
    assert s.wall_material == "concrete"
    assert s.human.abs_k == 0.9
