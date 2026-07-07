"""Unified runtime settings for a SOLWEIG calculation.

Consolidates the per-call kwargs to :func:`solweig.calculate` and
:class:`solweig.ModelConfig` into a single typed object with explicit
override semantics. Replaces the ad-hoc 50-line merge block that used
to live at the top of :func:`solweig.api._calculate_single`.

Merge precedence (lowest → highest): defaults < ``ModelConfig`` < per-call kwargs.

``None`` in any field at any level means "inherit from the next level"; the
final fall-through is the dataclass default. Two fields stay legitimately
``None`` after resolution:

- ``wall_material``: None means "use the bundled materials.Walls defaults"
- ``physics`` / ``materials``: None signals to the caller to lazy-load the
  bundled JSON later (kept lazy so import-time work stays cheap)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .config import HumanParams, ModelConfig

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for one SOLWEIG calculation.

    Produced by :meth:`Settings.resolve`. After resolution, every field
    holds a concrete value (no more "inherit"-style None for scalar/bool
    fields). See module docstring for the override semantics.
    """

    use_anisotropic_sky: bool
    conifer: bool
    wall_material: str | None
    max_shadow_distance_m: float
    human: HumanParams
    physics: SimpleNamespace | None
    materials: SimpleNamespace | None
    # UMEP 2026a ground-surface scheme (Bridoux): force-restore/OHM surface
    # temperature and solid-angle outgoing longwave. Both default OFF — the
    # validated baseline stays byte-identical until the scheme is validated.
    use_ground_scheme: bool = False
    use_outgoing_longwave: bool = False

    @classmethod
    def resolve(
        cls,
        *,
        config: ModelConfig | None = None,
        use_anisotropic_sky: bool | None = None,
        conifer: bool | None = None,
        wall_material: str | None = None,
        max_shadow_distance_m: float | None = None,
        human: HumanParams | None = None,
        physics: SimpleNamespace | None = None,
        materials: SimpleNamespace | None = None,
        use_ground_scheme: bool | None = None,
        use_outgoing_longwave: bool | None = None,
    ) -> Settings:
        """Merge per-call kwargs with a ModelConfig base and dataclass defaults.

        ``None`` in any kwarg means "use the config's value, else the default."
        """
        cfg_aniso = getattr(config, "use_anisotropic_sky", None) if config else None
        cfg_human = getattr(config, "human", None) if config else None
        cfg_physics = getattr(config, "physics", None) if config else None
        cfg_materials = getattr(config, "materials", None) if config else None
        cfg_max_shadow = getattr(config, "max_shadow_distance_m", None) if config else None
        cfg_ground_scheme = getattr(config, "use_ground_scheme", None) if config else None
        cfg_outgoing_lw = getattr(config, "use_outgoing_longwave", None) if config else None

        return cls(
            use_anisotropic_sky=_pick(use_anisotropic_sky, cfg_aniso, default=True),
            conifer=_pick(conifer, default=False),
            # wall_material is legitimately Optional even after resolution
            wall_material=wall_material,
            max_shadow_distance_m=_pick(max_shadow_distance_m, cfg_max_shadow, default=1000.0),
            human=_pick(human, cfg_human, default_factory=HumanParams),
            # physics / materials stay possibly-None so the caller can lazy-load
            # bundled JSON without paying the cost on import or in tests that
            # don't need them.
            physics=_pick(physics, cfg_physics, default=None),
            materials=_pick(materials, cfg_materials, default=None),
            use_ground_scheme=_pick(use_ground_scheme, cfg_ground_scheme, default=False),
            use_outgoing_longwave=_pick(use_outgoing_longwave, cfg_outgoing_lw, default=False),
        )

    def with_loaded_defaults(self) -> Settings:
        """Return a copy with physics/materials lazy-loaded from bundled JSON.

        Call this after :meth:`resolve` if the downstream computation needs
        non-None values. Kept as a separate step so unit tests can resolve a
        Settings without paying for the JSON load.
        """
        physics = self.physics
        materials = self.materials
        if physics is None:
            from ..loaders import load_physics

            physics = load_physics()
        if materials is None:
            from ..loaders import load_params

            materials = load_params()
        return replace(self, physics=physics, materials=materials)


def _pick(*candidates, default=None, default_factory=None):
    """Return the first non-None candidate, else the default.

    Provide either ``default`` (a constant) or ``default_factory`` (called
    only when needed) — not both. The factory form avoids constructing
    expensive defaults that get thrown away on the common path where a
    candidate is present.
    """
    for c in candidates:
        if c is not None:
            return c
    if default_factory is not None:
        return default_factory()
    return default
