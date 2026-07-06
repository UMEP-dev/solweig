"""Model configuration and human-body parameter classes.

Defines :class:`ModelConfig` (run-time settings such as sky model,
tiling, and shadow distance) and :class:`HumanParams` (posture,
absorption coefficients, and PET body parameters).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class ModelConfig:
    """
    Model configuration for SOLWEIG calculations.

    Groups all computational settings in one typed object.
    Pure configuration - no paths or data.

    Attributes:
        use_anisotropic_sky: Use Perez anisotropic sky model for diffuse
            radiation. Default True. Requires precomputed shadow matrices.
        human: Human body parameters for Tmrt calculations.
        outputs: Which outputs to save in timeseries calculations.
            Default ``["tmrt"]``.
        physics: Physics parameters (vegetation, posture geometry). Optional.
        materials: Material properties (albedo, emissivity). Optional.
        max_shadow_distance_m: Maximum shadow reach in metres. Default 1000.0.
            Caps horizontal shadow ray distance and serves as tile overlap
            buffer for automatic tiled processing. On mountainous terrain,
            increase this to capture terrain shadows across valleys.
        tile_size: Core tile side length in pixels for tiled processing.
            If None (default), auto-calculated from available resources.
            Minimum 256.
        tile_workers: Legacy runtime control for the low-level tiled executor.
            Retained for config compatibility. ``calculate()`` ignores this in
            tile-outer timeseries mode.
        tile_queue_depth: Legacy queue-depth control for the low-level tiled
            executor. Retained for config compatibility. ``calculate()``
            ignores this in tile-outer timeseries mode.
        prefetch_tiles: Legacy prefetch control for the low-level tiled
            executor. Retained for config compatibility. ``calculate()``
            ignores this in tile-outer timeseries mode.

    Note:
        UTCI and PET are computed inline when requested via
        ``outputs=["utci"]``.

    Examples:
        Basic usage with defaults:

        >>> config = ModelConfig.defaults()
        >>> config.save("my_config.json")

        Custom configuration:

        >>> config = ModelConfig(
        ...     use_anisotropic_sky=True,
        ...     human=HumanParams(abs_k=0.7, posture="standing"),
        ... )

        Load from legacy JSON:

        >>> config = ModelConfig.from_json("parametersforsolweig.json")
    """

    use_anisotropic_sky: bool = True
    human: HumanParams | None = None
    outputs: list[str] = field(default_factory=lambda: ["tmrt"])
    physics: SimpleNamespace | None = None
    materials: SimpleNamespace | None = None
    max_shadow_distance_m: float = 1000.0
    tile_size: int | None = None
    tile_workers: int | None = None
    tile_queue_depth: int | None = None
    prefetch_tiles: bool | None = None

    def __post_init__(self):
        """Validate configuration fields.

        Note: ``human`` is intentionally left as None here.
        Default ``HumanParams()`` is instantiated by ``calculate()``
        when no human params are provided.
        """
        if self.max_shadow_distance_m <= 0:
            raise ValueError(f"max_shadow_distance_m must be > 0, got {self.max_shadow_distance_m}")
        if self.tile_size is not None and self.tile_size < 256:
            raise ValueError(f"tile_size must be >= 256, got {self.tile_size}")
        if self.tile_workers is not None and self.tile_workers < 1:
            raise ValueError(f"tile_workers must be >= 1, got {self.tile_workers}")
        if self.tile_queue_depth is not None and self.tile_queue_depth < 0:
            raise ValueError(f"tile_queue_depth must be >= 0, got {self.tile_queue_depth}")

    @classmethod
    def defaults(cls) -> ModelConfig:
        """
        Create a ModelConfig with recommended defaults.

        Returns:
            ModelConfig with ``use_anisotropic_sky=True`` and all other
            fields at their dataclass defaults.
        """
        return cls(
            use_anisotropic_sky=True,
        )

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> ModelConfig:
        """
        Load configuration from legacy JSON parameters file.

        Args:
            path: Path to parametersforsolweig.json. ``None`` loads the
                bundled default parameter file (same resolution as
                :func:`solweig.load_params`).

        Returns:
            ModelConfig with settings extracted from JSON

        Example:
            >>> config = ModelConfig.from_json("parametersforsolweig.json")
            >>> config.human.abs_k  # From Tmrt_params
            0.7
        """
        from ..loaders import load_params

        params = load_params(path)

        def _section_values(section):
            # Legacy UMEP parameter files nest each section's values under
            # a "Value" key (with a sibling "Comment"); tolerate flat files too.
            return getattr(section, "Value", section)

        # Collect human parameters, then construct through HumanParams so
        # __post_init__ validation applies to file-supplied values.
        kwargs: dict = {}
        if hasattr(params, "Tmrt_params"):
            tmrt = _section_values(params.Tmrt_params)
            kwargs["abs_k"] = float(getattr(tmrt, "absK", 0.7))
            kwargs["abs_l"] = float(getattr(tmrt, "absL", 0.97))
            kwargs["posture"] = str(getattr(tmrt, "posture", "Standing")).lower()

        if hasattr(params, "PET_settings"):
            pet = _section_values(params.PET_settings)
            kwargs["age"] = int(getattr(pet, "Age", 35))
            kwargs["weight"] = float(getattr(pet, "Weight", 75.0))
            height = float(getattr(pet, "Height", 1.75))
            # Legacy UMEP files store body height in centimetres (e.g. 180).
            kwargs["height"] = height / 100.0 if height > 3.0 else height
            sex = getattr(pet, "Sex", 1)
            if isinstance(sex, str):
                sex = 1 if sex.lower().startswith("m") else 2
            kwargs["sex"] = int(sex)
            kwargs["activity"] = float(getattr(pet, "Activity", 80.0))
            kwargs["clothing"] = float(getattr(pet, "clo", 0.9))

        return cls(
            human=HumanParams(**kwargs),
            materials=params,
        )

    def save(self, path: str | Path):
        """
        Save configuration to JSON file.

        Args:
            path: Output path for JSON file

        Example:
            >>> config = ModelConfig.defaults()
            >>> config.save("my_settings.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to dict
        data = {
            "use_anisotropic_sky": self.use_anisotropic_sky,
            "max_shadow_distance_m": self.max_shadow_distance_m,
            "tile_size": self.tile_size,
            "tile_workers": self.tile_workers,
            "tile_queue_depth": self.tile_queue_depth,
            "prefetch_tiles": self.prefetch_tiles,
            "outputs": self.outputs,
            "human": {
                "posture": self.human.posture,
                "abs_k": self.human.abs_k,
                "abs_l": self.human.abs_l,
                "age": self.human.age,
                "weight": self.human.weight,
                "height": self.human.height,
                "sex": self.human.sex,
                "activity": self.human.activity,
                "clothing": self.human.clothing,
            }
            if self.human
            else None,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved configuration to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            ModelConfig loaded from file

        Example:
            >>> config = ModelConfig.load("my_settings.json")
            >>> summary = calculate(surface, weather, location, config=config, output_dir="out/")
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        # Deserialize human params
        human = None
        if data.get("human"):
            human = HumanParams(**data["human"])

        return cls(
            use_anisotropic_sky=data.get("use_anisotropic_sky", True),
            max_shadow_distance_m=data.get("max_shadow_distance_m", 1000.0),
            tile_size=data.get("tile_size"),
            tile_workers=data.get("tile_workers"),
            tile_queue_depth=data.get("tile_queue_depth"),
            prefetch_tiles=data.get("prefetch_tiles"),
            human=human,
            outputs=data.get("outputs", ["tmrt"]),
        )


@dataclass
class HumanParams:
    """
    Human body parameters for thermal comfort calculations.

    These parameters affect how radiation is absorbed by a person.
    Default values represent a standard reference person.

    Attributes:
        posture: Body posture ("standing" or "sitting"). Default "standing".
        abs_k: Shortwave absorption coefficient. Default 0.7.
        abs_l: Longwave absorption coefficient. Default 0.97.

    PET-specific parameters (used when ``"pet"`` is in outputs):
        age: Age in years. Default 35.
        weight: Body weight in kg. Default 75.
        height: Body height in meters. Default 1.75.
        sex: Biological sex (1=male, 2=female). Default 1.
        activity: Metabolic activity in W. Default 80.
        clothing: Clothing insulation in clo. Default 0.9.
    """

    posture: str = "standing"
    abs_k: float = 0.7
    abs_l: float = 0.97

    # PET-specific (optional)
    age: int = 35
    weight: float = 75.0
    height: float = 1.75
    sex: int = 1
    activity: float = 80.0
    clothing: float = 0.9

    def __post_init__(self):
        valid_postures = ("standing", "sitting")
        if self.posture not in valid_postures:
            raise ValueError(f"Posture must be one of {valid_postures}, got {self.posture}")
        if not 0 < self.abs_k <= 1:
            raise ValueError(f"abs_k must be in (0, 1], got {self.abs_k}")
        if not 0 < self.abs_l <= 1:
            raise ValueError(f"abs_l must be in (0, 1], got {self.abs_l}")
        if not 0 <= self.age <= 150:
            raise ValueError(f"age must be in [0, 150], got {self.age}")
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0, got {self.weight}")
        if not 0.5 <= self.height <= 2.5:
            raise ValueError(f"height must be in [0.5, 2.5] m, got {self.height}")
        if self.sex not in (1, 2):
            raise ValueError(f"sex must be 1 (male) or 2 (female), got {self.sex}")
        if self.activity < 0:
            raise ValueError(f"activity must be >= 0, got {self.activity}")
        if self.clothing < 0:
            raise ValueError(f"clothing must be >= 0, got {self.clothing}")
