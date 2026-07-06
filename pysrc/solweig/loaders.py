"""Configuration and parameter loading from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .utils import dict_to_namespace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_params(params_json_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load SOLWEIG parameters from a JSON file.

    Returns a mutable SimpleNamespace with all UMEP-standard parameters:
    land cover properties, wall materials, Tmrt settings, PET settings,
    tree settings, and posture geometry.

    Args:
        params_json_path: Path to the parameters JSON file.
            If None (default), loads the bundled default_materials.json
            with all UMEP-standard values.

    Returns:
        SimpleNamespace object with nested parameter values accessible via attributes.
        The namespace is mutable — override individual values as needed.

    Examples:
        Load bundled defaults:

        >>> params = load_params()
        >>> params.Tmrt_params.Value.absK  # 0.7
        >>> params.Albedo.Effective.Value.Dark_asphalt  # 0.18

        Override a specific value:

        >>> params = load_params()
        >>> params.Ts_deg.Value.Walls = 0.50  # Change wall TgK
    """
    if params_json_path is None:
        # Use bundled default parameters (full UMEP-format JSON with all sections)
        params_path = Path(__file__).parent / "data" / "default_materials.json"
    else:
        params_path = Path(params_json_path)

    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with open(params_path) as f:
        params_dict = json.load(f)

    result = dict_to_namespace(params_dict)
    if not isinstance(result, SimpleNamespace):
        raise TypeError(f"Expected SimpleNamespace from JSON, got {type(result).__name__}")
    return result


def load_physics(physics_json_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load physics parameters (site-independent scientific constants).

    Physics parameters include:
    - Tree_settings: Vegetation transmissivity, seasonal dates, trunk ratio
    - Posture: Human posture geometry (Standing/Sitting projected area fractions)

    These are universal constants that rarely need customization.

    Args:
        physics_json_path: Path to a custom physics JSON file.
            If None (default), loads bundled physics_defaults.json with standard values.

    Returns:
        SimpleNamespace object with physics parameters accessible via attributes.

    Examples:
        Load bundled defaults:

        >>> physics = load_physics()  # Uses bundled physics_defaults.json
        >>> physics.Tree_settings.Value.Transmissivity  # 0.03
        >>> physics.Posture.Standing.Value.Fside  # 0.22

        Load custom physics (e.g., different tree transmissivity):

        >>> physics = load_physics("custom_trees.json")
    """
    if physics_json_path is None:
        # Use bundled physics defaults
        physics_path = Path(__file__).parent / "data" / "physics_defaults.json"
    else:
        physics_path = Path(physics_json_path)

    if not physics_path.exists():
        raise FileNotFoundError(f"Physics parameters file not found: {physics_path}")

    with open(physics_path) as f:
        physics_dict = json.load(f)

    result = dict_to_namespace(physics_dict)
    if not isinstance(result, SimpleNamespace):
        raise TypeError(f"Expected SimpleNamespace from JSON, got {type(result).__name__}")
    return result


def load_materials(materials_json_path: str | Path) -> SimpleNamespace:
    """
    Load material properties (site-specific landcover parameters).

    Material properties include per-landcover-class values for:
    - Names: Landcover class names (e.g., "Dark_asphalt", "Grass_unmanaged")
    - Code: Landcover class IDs
    - Albedo: Surface albedo per class
    - Emissivity: Surface emissivity per class
    - TmaxLST, Ts_deg, Tstart: Ground temperature model parameters per class
    - Specific_heat, Thermal_conductivity, Density, Wall_thickness: Wall thermal properties

    These are site-specific and require a landcover grid (land_cover input).

    Args:
        materials_json_path: Path to a materials JSON file.
            This file must contain landcover-specific property definitions.

    Returns:
        SimpleNamespace object with material parameters accessible via attributes.

    Examples:
        Load site-specific materials:

        >>> materials = load_materials("site_materials.json")
        >>> materials.Albedo.Effective.Value.Dark_asphalt  # 0.18
        >>> materials.Emissivity.Value.Grass_unmanaged  # 0.94

    Notes:
        Materials are ONLY used when a landcover grid is provided to SurfaceData.
        If no landcover grid, uniform default properties are used.
    """
    materials_path = Path(materials_json_path)

    if not materials_path.exists():
        raise FileNotFoundError(f"Materials file not found: {materials_path}")

    with open(materials_path) as f:
        materials_dict = json.load(f)

    result = dict_to_namespace(materials_dict)
    if not isinstance(result, SimpleNamespace):
        raise TypeError(f"Expected SimpleNamespace from JSON, got {type(result).__name__}")
    return result


def get_lc_properties_from_params(
    land_cover: NDArray[np.integer],
    params: SimpleNamespace,
    shape: tuple[int, int],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Derive surface properties from land cover grid using loaded params.

    Builds per-pixel lookup grids (albedo, emissivity, TgK, Tstart, TmaxLST)
    from land-cover IDs, mirroring UMEP's TgMaps parameterization.

    Args:
        land_cover: Land cover classification grid (UMEP standard IDs).
        params: Loaded parameters from JSON file.
        shape: Output grid shape (rows, cols).

    Returns:
        Tuple of (albedo_grid, emissivity_grid, tgk_grid, tstart_grid, tmaxlst_grid).
    """
    rows, cols = shape

    # Defaults for pixels with no land-cover match (cobblestone/generic urban)
    _DEFAULT_ALBEDO = 0.15
    _DEFAULT_EMISSIVITY = 0.95
    _DEFAULT_TGK = 0.37  # Ground thermal response coefficient (K/W·m⁻²)
    _DEFAULT_TSTART = -3.41  # Ground temperature offset (°C)
    _DEFAULT_TMAXLST = 15.0  # Max land surface temperature amplitude (°C)

    # Get land cover with wall codes remapped to buildings
    lc = np.copy(land_cover)
    lc[lc >= 100] = 2  # Treat wall codes as buildings

    # Build lookup tables (lc_id → property value) for IDs 0-7
    max_id = 8
    alb_lut = np.full(max_id, _DEFAULT_ALBEDO, dtype=np.float32)
    emis_lut = np.full(max_id, _DEFAULT_EMISSIVITY, dtype=np.float32)
    tgk_lut = np.full(max_id, _DEFAULT_TGK, dtype=np.float32)
    tstart_lut = np.full(max_id, _DEFAULT_TSTART, dtype=np.float32)
    tmaxlst_lut = np.full(max_id, _DEFAULT_TMAXLST, dtype=np.float32)

    for lc_id in range(max_id):
        name = getattr(params.Names.Value, str(lc_id), None)
        if name is None:
            continue
        alb_lut[lc_id] = getattr(params.Albedo.Effective.Value, name, _DEFAULT_ALBEDO)
        emis_lut[lc_id] = getattr(params.Emissivity.Value, name, _DEFAULT_EMISSIVITY)
        tgk_lut[lc_id] = getattr(params.Ts_deg.Value, name, _DEFAULT_TGK)
        tstart_lut[lc_id] = getattr(params.Tstart.Value, name, _DEFAULT_TSTART)
        tmaxlst_lut[lc_id] = getattr(params.TmaxLST.Value, name, _DEFAULT_TMAXLST)

    # Vectorized lookup: clip IDs to valid range and index into LUTs
    lc_safe = np.clip(lc, 0, max_id - 1)
    alb_grid = alb_lut[lc_safe]
    emis_grid = emis_lut[lc_safe]
    tgk_grid = tgk_lut[lc_safe]
    tstart_grid = tstart_lut[lc_safe]
    tmaxlst_grid = tmaxlst_lut[lc_safe]

    return alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid


# Map user-facing wall material names to JSON keys in default_materials.json
WALL_MATERIAL_MAP: dict[str, str] = {
    "brick": "Brick_wall",
    "concrete": "Concrete_wall",
    "wood": "Wood_wall",
    "cobblestone": "Walls",
}


def resolve_wall_params(
    wall_material: str,
    materials: SimpleNamespace | None = None,
) -> tuple[float, float, float]:
    """Resolve wall material name to (tgk_wall, tstart_wall, tmaxlst_wall).

    Args:
        wall_material: Material name (case-insensitive).
            One of: "brick", "concrete", "wood", "cobblestone".
        materials: Loaded materials namespace. If None, loads bundled defaults.

    Returns:
        Tuple of (tgk_wall, tstart_wall, tmaxlst_wall) floats.

    Raises:
        ValueError: If wall_material is not a recognized material name.
    """
    key = wall_material.lower()
    if key not in WALL_MATERIAL_MAP:
        valid = ", ".join(sorted(WALL_MATERIAL_MAP))
        msg = f"Unknown wall material {wall_material!r}. Valid options: {valid}"
        raise ValueError(msg)

    json_name = WALL_MATERIAL_MAP[key]

    if materials is None:
        materials = load_params()

    try:
        tgk = float(getattr(materials.Ts_deg.Value, json_name))
        tstart = float(getattr(materials.Tstart.Value, json_name))
        tmaxlst = float(getattr(materials.TmaxLST.Value, json_name))
    except AttributeError:
        raise ValueError(
            f"Materials JSON missing required properties for wall material {json_name!r}. "
            f"Expected keys: Ts_deg.Value.{json_name}, Tstart.Value.{json_name}, TmaxLST.Value.{json_name}"
        ) from None
    return tgk, tstart, tmaxlst
