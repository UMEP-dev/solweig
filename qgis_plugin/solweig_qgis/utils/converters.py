"""
Converters between QGIS parameters and SOLWEIG dataclasses.

Handles translation of QGIS Processing parameters into the dataclasses
expected by the solweig library API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsRasterLayer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_raster_from_layer(
    layer: QgsRasterLayer,
) -> tuple[NDArray[np.floating], list[float], str]:
    """
    Load QGIS raster layer to numpy array using GDAL.

    Args:
        layer: QGIS raster layer to load.

    Returns:
        tuple of (array, geotransform, crs_wkt):
            - array: 2D numpy float32 array
            - geotransform: GDAL 6-tuple
            - crs_wkt: CRS as WKT string

    Raises:
        QgsProcessingException: If raster cannot be opened.
    """
    source = layer.source()
    ds = gdal.Open(source, gdal.GA_ReadOnly)
    if ds is None:
        raise QgsProcessingException(f"Cannot open raster: {source}")

    try:
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)

        # Match solweig.io.load_raster(): honor any explicit non-NaN nodata
        # sentinel so QGIS and local API runs see the same valid mask.
        nodata = band.GetNoDataValue()
        if nodata is not None and not np.isnan(nodata):
            array = np.where(array == nodata, np.nan, array)

        geotransform = list(ds.GetGeoTransform())
        crs_wkt = ds.GetProjection()

        return array, geotransform, crs_wkt
    finally:
        ds = None


def _read_height_mode(
    parameters: dict[str, Any],
    param_name: str,
    default_absolute: bool = True,
) -> bool:
    """Read a per-layer height mode enum and return True if relative.

    Args:
        parameters: Algorithm parameters dict.
        param_name: Enum parameter name (e.g. "DSM_HEIGHT_MODE").
        default_absolute: If True, default is absolute (enum 1); if False, default is relative (enum 0).

    Returns:
        True if the layer uses relative heights, False if absolute.
    """
    default = 1 if default_absolute else 0
    value = parameters.get(param_name, default)
    return (int(value) if isinstance(value, (int, float)) else default) == 0


def _looks_like_relative_heights(
    layer: NDArray[np.floating] | None,
    reference_surface: NDArray[np.floating] | None,
) -> bool:
    """Heuristically detect height-above-ground rasters passed as absolute.

    Delegates to the canonical implementation in the core API.
    """
    from solweig import looks_like_relative

    return looks_like_relative(layer, reference_surface)


def _load_optional_raster(
    parameters: dict[str, Any],
    param_name: str,
    context: QgsProcessingContext,
    param_handler: Any,
) -> tuple[NDArray[np.floating] | None, list[float] | None]:
    """Load optional raster, returning (array, geotransform) or (None, None)."""
    if param_name not in parameters or not parameters[param_name]:
        return None, None
    layer = param_handler.parameterAsRasterLayer(parameters, param_name, context)
    if layer is None:
        return None, None
    arr, gt, _ = load_raster_from_layer(layer)
    return arr, gt


def _align_layer(
    arr: NDArray[np.floating],
    gt: list[float],
    target_bbox: list[float],
    pixel_size: float,
    method: str,
    crs_wkt: str,
) -> NDArray[np.floating]:
    """Resample a raster to the target grid if extents or shape differ."""
    from solweig import extract_bounds, resample_to_grid

    # Expected target dimensions (same formula as resample_to_grid)
    expected_h = int(np.round((target_bbox[3] - target_bbox[1]) / pixel_size))
    expected_w = int(np.round((target_bbox[2] - target_bbox[0]) / pixel_size))

    bounds = extract_bounds(gt, arr.shape)
    needs_resample = (
        abs(bounds[0] - target_bbox[0]) > 1e-6
        or abs(bounds[1] - target_bbox[1]) > 1e-6
        or abs(bounds[2] - target_bbox[2]) > 1e-6
        or abs(bounds[3] - target_bbox[3]) > 1e-6
        or abs(abs(gt[1]) - pixel_size) > 1e-6
        or arr.shape != (expected_h, expected_w)
    )
    if needs_resample:
        arr, _ = resample_to_grid(arr, gt, target_bbox, pixel_size, method=method, src_crs=crs_wkt)
    return arr


def create_surface_from_parameters(
    parameters: dict[str, Any],
    context: QgsProcessingContext,
    param_handler: Any,  # Algorithm instance with parameterAsRasterLayer
    feedback: QgsProcessingFeedback,
    bbox: list[float] | None = None,
    output_dir: str | None = None,
) -> Any:  # Returns solweig.SurfaceData
    """
    Create SurfaceData from QGIS processing parameters.

    Loads all surface rasters, aligns them to a common grid (intersection
    of all extents or user-specified bbox), computes a unified valid mask,
    and saves cleaned rasters to disk.

    Args:
        parameters: Algorithm parameters dict.
        context: Processing context.
        param_handler: Object with parameterAsRasterLayer method.
        feedback: Processing feedback.
        bbox: Optional explicit bounding box [minx, miny, maxx, maxy].
        output_dir: Optional directory for saving cleaned rasters.

    Returns:
        solweig.SurfaceData instance with aligned, masked arrays.

    Raises:
        QgsProcessingException: If required DSM is missing or invalid.
    """
    try:
        import solweig
        from solweig import extract_bounds, intersect_bounds
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Load required DSM (with geotransform)
    dsm_layer = param_handler.parameterAsRasterLayer(parameters, "DSM", context)
    if dsm_layer is None:
        raise QgsProcessingException("DSM layer is required")

    dsm, dsm_gt, crs_wkt = load_raster_from_layer(dsm_layer)
    lo, hi = float(np.nanmin(dsm)), float(np.nanmax(dsm))
    feedback.pushInfo(f"Loaded DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels, range: {lo:.1f} – {hi:.1f} m")

    pixel_size = abs(dsm_gt[1])
    feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

    # Load optional rasters (keeping geotransforms)
    cdsm, cdsm_gt = _load_optional_raster(parameters, "CDSM", context, param_handler)
    if cdsm is not None:
        feedback.pushInfo(
            f"Loaded CDSM (vegetation), range: {float(np.nanmin(cdsm)):.1f} – {float(np.nanmax(cdsm)):.1f} m"
        )

    dem, dem_gt = _load_optional_raster(parameters, "DEM", context, param_handler)
    if dem is not None:
        feedback.pushInfo(
            f"Loaded DEM (ground elevation), range: {float(np.nanmin(dem)):.1f} – {float(np.nanmax(dem)):.1f} m"
        )

    tdsm, tdsm_gt = _load_optional_raster(parameters, "TDSM", context, param_handler)
    if tdsm is not None:
        feedback.pushInfo(
            f"Loaded TDSM (trunk zone), range: {float(np.nanmin(tdsm)):.1f} – {float(np.nanmax(tdsm)):.1f} m"
        )

    lc_arr, lc_gt = _load_optional_raster(parameters, "LAND_COVER", context, param_handler)
    land_cover = lc_arr.astype(np.uint8) if lc_arr is not None else None
    if land_cover is not None:
        feedback.pushInfo("Loaded land cover classification")

    # Compute extent intersection of all loaded layers
    bounds_list = [extract_bounds(dsm_gt, dsm.shape)]
    for arr, gt in [(cdsm, cdsm_gt), (dem, dem_gt), (tdsm, tdsm_gt), (lc_arr, lc_gt)]:
        if arr is not None and gt is not None:
            bounds_list.append(extract_bounds(gt, arr.shape))

    if bbox is not None:
        target_bbox = bbox
    elif len(bounds_list) > 1:
        target_bbox = intersect_bounds(bounds_list)
        feedback.pushInfo(f"Auto-computed intersection extent: {target_bbox}")
    else:
        target_bbox = bounds_list[0]

    # Align all layers to the target grid
    dsm = _align_layer(dsm, dsm_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if cdsm is not None and cdsm_gt is not None:
        cdsm = _align_layer(cdsm, cdsm_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if dem is not None and dem_gt is not None:
        dem = _align_layer(dem, dem_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if tdsm is not None and tdsm_gt is not None:
        tdsm = _align_layer(tdsm, tdsm_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if land_cover is not None and lc_gt is not None:
        land_cover = _align_layer(
            land_cover.astype(np.float32),
            lc_gt,
            target_bbox,
            pixel_size,
            "nearest",
            crs_wkt,
        ).astype(np.uint8)

    feedback.pushInfo(f"Aligned grid: {dsm.shape[1]}x{dsm.shape[0]} pixels")

    # Build aligned geotransform for the target bbox
    aligned_gt = [target_bbox[0], pixel_size, 0, target_bbox[3], 0, -pixel_size]

    # Get per-layer height convention flags (enum: 0=relative, 1=absolute)
    dsm_relative = _read_height_mode(parameters, "DSM_HEIGHT_MODE", default_absolute=True)
    cdsm_relative = _read_height_mode(parameters, "CDSM_HEIGHT_MODE", default_absolute=False)
    tdsm_relative = _read_height_mode(parameters, "TDSM_HEIGHT_MODE", default_absolute=False)

    # Read min_object_height (advanced parameter, defaults to 1.0)
    min_object_height = 1.0
    if hasattr(param_handler, "parameterAsDouble"):
        import contextlib

        with contextlib.suppress(Exception):
            min_object_height = param_handler.parameterAsDouble(parameters, "MIN_OBJECT_HEIGHT", context)

    # Create SurfaceData
    surface = solweig.SurfaceData(
        dsm=dsm,
        cdsm=cdsm,
        dem=dem,
        tdsm=tdsm,
        land_cover=land_cover,
        pixel_size=pixel_size,
        dsm_relative=dsm_relative,
        cdsm_relative=cdsm_relative,
        tdsm_relative=tdsm_relative,
        min_object_height=min_object_height,
    )

    # Store geospatial metadata for output georeferencing
    surface._geotransform = aligned_gt
    surface._crs_wkt = crs_wkt

    # Convert relative heights to absolute and flatten sub-threshold features
    needs_preprocess = (
        dsm_relative or (cdsm_relative and cdsm is not None) or (tdsm_relative and tdsm is not None) or dem is not None
    )
    if needs_preprocess:
        feedback.pushInfo("Converting relative heights to absolute...")
        surface.preprocess()

    # Compute wall heights and aspects from DSM
    feedback.setProgressText("Computing wall heights...")
    feedback.pushInfo("Computing walls from DSM...")
    from solweig import wallalgorithms as wa

    walls = wa.findwalls(surface.dsm, 1.0)
    feedback.pushInfo("Computing wall aspects...")
    feedback.setProgressText("Computing wall aspects...")
    dsm_scale = 1.0 / pixel_size
    dirwalls = wa.filter1Goodwin_as_aspect_v3(walls, dsm_scale, surface.dsm, feedback=feedback)
    surface.wall_height = walls
    surface.wall_aspect = dirwalls
    feedback.pushInfo("Wall computation complete")

    # Match SurfaceData.prepare(): compute walls on aligned absolute-height
    # rasters first, then derive the unified valid mask and crop everything.
    surface.fill_nan()
    surface.compute_valid_mask()
    surface.apply_valid_mask()
    surface.crop_to_valid_bbox()

    feedback.pushInfo(f"After NaN fill + mask + crop: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")

    # Save cleaned rasters
    if output_dir:
        surface.save_cleaned(output_dir)

    return surface


def load_prepared_surface(
    surface_dir: str,
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns solweig.SurfaceData
    """
    Load a prepared surface directory into SurfaceData.

    Delegates to ``SurfaceData.load()`` which reads the cleaned rasters,
    walls, SVF, and shadow matrices cached by ``SurfaceData.prepare()``.

    Args:
        surface_dir: Path to prepared surface directory.
        feedback: Processing feedback.

    Returns:
        solweig.SurfaceData instance with all arrays loaded.

    Raises:
        QgsProcessingException: If required files are missing.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found.") from e

    feedback.pushInfo(f"Loading prepared surface from {surface_dir}")

    try:
        surface = solweig.SurfaceData.load(surface_dir)
    except FileNotFoundError as e:
        raise QgsProcessingException(str(e)) from e
    except ValueError as e:
        raise QgsProcessingException(str(e)) from e

    feedback.pushInfo(f"  Grid: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} @ {surface.pixel_size:.2f}m")

    return surface


def create_location_from_parameters(
    parameters: dict[str, Any],
    surface: Any,  # solweig.SurfaceData
    feedback: QgsProcessingFeedback,
    epw_path: str | None = None,
) -> Any:  # Returns solweig.Location
    """
    Create Location from QGIS processing parameters.

    Supports core API location derivation from EPW, surface CRS, or manual input.

    Args:
        parameters: Algorithm parameters dict.
        surface: SurfaceData instance (for auto-extraction).
        feedback: Processing feedback.

    Returns:
        solweig.Location instance.

    Raises:
        QgsProcessingException: If location cannot be determined.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    if epw_path:
        try:
            location = solweig.Location.from_epw(epw_path)
        except FileNotFoundError as e:
            raise QgsProcessingException(f"EPW file not found: {epw_path}") from e
        except Exception as e:
            raise QgsProcessingException(f"Cannot derive location from EPW: {e}") from e

        feedback.pushInfo(
            "Using location from EPW header to match the core SOLWEIG API: "
            f"{location.latitude:.4f}N, {location.longitude:.4f}E "
            f"(UTC{location.utc_offset:+g}, {location.altitude:.0f} m)"
        )
        return location

    utc_offset = parameters.get("UTC_OFFSET", 0)

    if parameters.get("AUTO_EXTRACT_LOCATION", False):
        feedback.pushInfo("Auto-extracting location from DSM CRS...")
        try:
            location = solweig.Location.from_surface(surface, utc_offset=utc_offset)
        except Exception as e:
            raise QgsProcessingException(f"Cannot auto-extract location: {e}") from e
        feedback.pushInfo(f"Location: {location.latitude:.4f}N, {location.longitude:.4f}E")
    else:
        # Use manual input
        latitude = parameters.get("LATITUDE")
        longitude = parameters.get("LONGITUDE")

        if latitude is None or longitude is None:
            raise QgsProcessingException("Latitude and longitude are required when auto-extract is disabled")

        location = solweig.Location(
            latitude=latitude,
            longitude=longitude,
            utc_offset=utc_offset,
        )
        feedback.pushInfo(f"Location: {latitude:.4f}N, {longitude:.4f}E")

    return location


def create_weather_from_parameters(
    parameters: dict[str, Any],
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns solweig.Weather
    """
    Create Weather from QGIS processing parameters.

    Args:
        parameters: Algorithm parameters dict.
        feedback: Processing feedback.

    Returns:
        solweig.Weather instance.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Get datetime from QDateTime parameter
    qdt = parameters["DATETIME"]
    dt = qdt.toPyDateTime()

    weather = solweig.Weather(
        datetime=dt,
        ta=parameters.get("TEMPERATURE", 25.0),
        rh=parameters.get("HUMIDITY", 50.0),
        global_rad=parameters.get("GLOBAL_RADIATION", 800.0),
        ws=parameters.get("WIND_SPEED", 1.0),
        pressure=parameters.get("PRESSURE", 1013.25),
    )

    feedback.pushInfo(
        f"Weather: {dt.strftime('%Y-%m-%d %H:%M')}, "
        f"Ta={weather.ta:.1f}C, RH={weather.rh:.0f}%, "
        f"G={weather.global_rad:.0f}W/m2"
    )

    return weather


def create_human_params_from_parameters(
    parameters: dict[str, Any],
) -> Any:  # Returns solweig.HumanParams
    """
    Create HumanParams from QGIS processing parameters.

    Args:
        parameters: Algorithm parameters dict.

    Returns:
        solweig.HumanParams instance.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Map posture enum to string
    posture_map = {0: "standing", 1: "sitting"}
    posture = posture_map.get(parameters.get("POSTURE", 0), "standing")

    # Basic human params
    human = solweig.HumanParams(
        posture=posture,
        abs_k=parameters.get("ABS_K", 0.7),
    )

    # Add detailed body params if present (for PET)
    if "WEIGHT" in parameters:
        human.weight = parameters["WEIGHT"]
    if "HEIGHT" in parameters:
        human.height = parameters["HEIGHT"]
    if "AGE" in parameters:
        human.age = parameters["AGE"]
    if "ACTIVITY" in parameters:
        human.activity = parameters["ACTIVITY"]
    if "CLOTHING" in parameters:
        human.clothing = parameters["CLOTHING"]
    if "SEX" in parameters:
        sex_map = {0: 1, 1: 2}
        human.sex = sex_map.get(parameters["SEX"], 1)

    return human


def create_physics_from_parameters(
    parameters: dict[str, Any],
) -> Any:  # Returns types.SimpleNamespace
    """
    Create a physics namespace from QGIS vegetation parameters.

    Loads default physics and overrides Tree_settings with user-supplied
    transmissivity and seasonal date values.

    Args:
        parameters: Algorithm parameters dict.

    Returns:
        SimpleNamespace with Tree_settings overridden by QGIS parameters.
    """
    try:
        from solweig import load_physics
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    physics = load_physics()
    ts = physics.Tree_settings.Value
    ts.Transmissivity = parameters.get("TRANSMISSIVITY", 0.03)
    ts.Transmissivity_leafoff = parameters.get("TRANSMISSIVITY_LEAFOFF", 0.5)
    ts.First_day_leaf = int(parameters.get("LEAF_START", 97))
    ts.Last_day_leaf = int(parameters.get("LEAF_END", 300))

    return physics


def build_materials_from_lc_mapping(
    parameters: dict[str, Any],
    context: QgsProcessingContext,
    param_handler: Any,
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns types.SimpleNamespace
    """
    Build a materials namespace from QGIS land cover mapping parameters.

    Priority (highest first):
        1. CUSTOM_MATERIALS_FILE — loads a full JSON override.
        2. LC_MATERIALS matrix table — per-code properties
           (Code, Name, Albedo, Emissivity, TgK, Tstart, TmaxLST).
        3. Bundled UMEP defaults.

    Args:
        parameters: Algorithm parameters dict.
        context: Processing context.
        param_handler: Algorithm instance (for parameterAs* methods).
        feedback: Processing feedback for logging.

    Returns:
        SimpleNamespace compatible with ``solweig.calculate(materials=...)``.
    """
    try:
        from solweig import load_params
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Custom JSON takes priority over everything
    custom_path = param_handler.parameterAsFile(parameters, "CUSTOM_MATERIALS_FILE", context)
    if custom_path:
        feedback.pushInfo(f"Using custom materials file: {custom_path}")
        return load_params(custom_path)

    # Load bundled defaults as base
    materials = load_params()

    # Apply material properties from the table
    _apply_lc_materials(parameters, context, param_handler, materials, feedback)

    return materials


def _apply_lc_materials(
    parameters: dict[str, Any],
    context: QgsProcessingContext,
    param_handler: Any,
    materials: Any,
    feedback: QgsProcessingFeedback,
) -> None:
    """Parse LC_MATERIALS matrix and set material properties in-place.

    Each row has 7 values: Code, Name, Albedo, Emissivity, TgK, Tstart, TmaxLST.
    A surface type name is registered in ``materials.Names.Value`` for each code
    so the standard lookup chain resolves correctly.
    """
    raw = parameters.get("LC_MATERIALS")
    if not raw:
        return

    # QgsProcessingParameterMatrix stores values as a flat list
    flat: list[str] = [
        str(v)
        for v in (
            param_handler.parameterAsMatrix(parameters, "LC_MATERIALS", context)
            if hasattr(param_handler, "parameterAsMatrix")
            else raw
        )
    ]
    n_cols = 7  # Code, Name, Albedo, Emissivity, TgK, Tstart, TmaxLST
    if len(flat) < n_cols:
        return

    prop_sections = ["Albedo.Effective", "Emissivity", "Ts_deg", "Tstart", "TmaxLST"]

    for row_start in range(0, len(flat) - n_cols + 1, n_cols):
        row = flat[row_start : row_start + n_cols]
        try:
            code = int(float(row[0]))
        except (ValueError, TypeError):
            continue

        name = row[1].strip() if row[1].strip() else f"LC_{code}"

        # Parse the 5 property columns
        values: list[float | None] = []
        for cell in row[2:]:
            cell = cell.strip() if isinstance(cell, str) else str(cell).strip()
            if not cell:
                values.append(None)
            else:
                try:
                    values.append(float(cell))
                except (ValueError, TypeError):
                    values.append(None)

        if all(v is None for v in values):
            continue

        # Resolve base values from the UMEP default for this code (if any)
        # so that empty cells inherit from the existing land-cover class
        default_name = getattr(materials.Names.Value, str(code), None)
        if default_name is None:
            default_name = "Cobble_stone_2014a"  # fallback if code has no UMEP default

        # Register the type name for this code (after reading the old default)
        type_name = f"LC_{code}_{name.replace(' ', '_')}"
        setattr(materials.Names.Value, str(code), type_name)

        for i, section_path in enumerate(prop_sections):
            parts = section_path.split(".")
            ns = materials
            for part in parts:
                ns = getattr(ns, part, ns)
            ns = getattr(ns, "Value", ns)

            base_val = getattr(ns, default_name, None) if default_name else None
            final_val = values[i] if values[i] is not None else base_val
            if final_val is not None:
                setattr(ns, type_name, final_val)

        feedback.pushInfo(
            f"  LC code {code} ({name}): "
            f"albedo={values[0]}, emis={values[1]}, TgK={values[2]}, "
            f"Tstart={values[3]}, TmaxLST={values[4]}"
        )


def load_weather_from_epw(
    epw_path: str,
    start_dt: Any | None,  # QDateTime, datetime, or None
    end_dt: Any | None,  # QDateTime, datetime, or None
    hours_filter: str | None,
    feedback: QgsProcessingFeedback,
) -> list:  # Returns list[solweig.Weather]
    """
    Load weather data from EPW file via ``solweig.Weather.from_epw()``.

    Args:
        epw_path: Path to EPW file.
        start_dt: Start datetime (inclusive), or None for EPW start.
        end_dt: End datetime (inclusive), or None for EPW end.
        hours_filter: Comma-separated hours to include (e.g., "9,10,11,12").
        feedback: Processing feedback.

    Returns:
        List of solweig.Weather objects.

    Raises:
        QgsProcessingException: If EPW file cannot be read or dates don't overlap.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Convert QDateTime to Python naive datetime
    if start_dt is not None and hasattr(start_dt, "toPyDateTime"):
        start_dt = start_dt.toPyDateTime()
    if end_dt is not None and hasattr(end_dt, "toPyDateTime"):
        end_dt = end_dt.toPyDateTime()
    # Strip timezone info to avoid aware/naive comparison errors
    if start_dt is not None and start_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=None)
    if end_dt is not None and end_dt.tzinfo is not None:
        end_dt = end_dt.replace(tzinfo=None)

    # Parse hours filter
    hours_list = None
    if hours_filter:
        try:
            hours_str = hours_filter.replace(" ", "")
            hours_list = [int(h) for h in hours_str.split(",")]
            feedback.pushInfo(f"Hour filter: {hours_list}")
        except ValueError:
            feedback.reportError(
                f"Invalid hours filter: {hours_filter}. Using all hours.",
                fatalError=False,
            )

    try:
        weather_series = solweig.Weather.from_epw(
            epw_path,
            start=start_dt,
            end=end_dt,
            hours=hours_list,
        )
    except FileNotFoundError as e:
        raise QgsProcessingException(f"EPW file not found: {epw_path}") from e
    except ValueError as e:
        raise QgsProcessingException(str(e)) from e

    feedback.pushInfo(f"Loaded {len(weather_series)} timesteps from EPW")
    feedback.pushInfo(f"Period: {weather_series[0].datetime} to {weather_series[-1].datetime}")

    return weather_series


def load_weather_from_umep_met(
    met_path: str,
    start_dt: Any | None,
    end_dt: Any | None,
    hours_filter: str | None,
    feedback: QgsProcessingFeedback,
) -> list:  # Returns list[solweig.Weather]
    """
    Load weather data from a UMEP/SUEWS meteorological forcing file.

    Args:
        met_path: Path to UMEP met file.
        start_dt: Start datetime (inclusive), or None for full range.
        end_dt: End datetime (inclusive), or None for full range.
        hours_filter: Comma-separated hours to include (e.g., "9,10,11,12").
        feedback: Processing feedback.

    Returns:
        List of solweig.Weather objects.

    Raises:
        QgsProcessingException: If file cannot be read or no data found.
    """
    try:
        from solweig.models.weather import Weather
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    if not met_path:
        raise QgsProcessingException("No UMEP met file specified.")

    feedback.pushInfo(f"Loading UMEP met file: {met_path}")

    # Convert QDateTime to Python naive datetime
    if start_dt is not None and hasattr(start_dt, "toPyDateTime"):
        start_dt = start_dt.toPyDateTime()
    if end_dt is not None and hasattr(end_dt, "toPyDateTime"):
        end_dt = end_dt.toPyDateTime()
    if start_dt is not None and start_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=None)
    if end_dt is not None and end_dt.tzinfo is not None:
        end_dt = end_dt.replace(tzinfo=None)

    # Parse hours filter
    hours_list = None
    if hours_filter:
        try:
            hours_str = hours_filter.replace(" ", "")
            hours_list = [int(h) for h in hours_str.split(",")]
            feedback.pushInfo(f"Hour filter: {hours_list}")
        except ValueError:
            feedback.reportError(
                f"Invalid hours filter: {hours_filter}. Using all hours.",
                fatalError=False,
            )

    # Load via Weather.from_umep_met()
    try:
        weather_series = Weather.from_umep_met(
            paths=[met_path],
            resample_hourly=True,
            start=start_dt,
            end=end_dt,
        )
    except FileNotFoundError as e:
        raise QgsProcessingException(f"UMEP met file not found: {e}") from e
    except ValueError as e:
        raise QgsProcessingException(f"Error reading UMEP met file: {e}") from e

    if not weather_series:
        raise QgsProcessingException("No valid timesteps found in UMEP met file.")

    # Report date range
    met_start = weather_series[0].datetime
    met_end = weather_series[-1].datetime
    feedback.pushInfo(f"UMEP met date range: {met_start} to {met_end}")

    # Apply hours filter if specified
    if hours_list:
        weather_series = [w for w in weather_series if w.datetime.hour in hours_list]
        if not weather_series:
            raise QgsProcessingException(
                f"No timesteps remaining after hour filter {hours_list}.\n"
                f"The data contains hours from {met_start} to {met_end}."
            )

    feedback.pushInfo(f"Loaded {len(weather_series)} timesteps from UMEP met")
    feedback.pushInfo(f"Period: {weather_series[0].datetime} to {weather_series[-1].datetime}")

    return weather_series
