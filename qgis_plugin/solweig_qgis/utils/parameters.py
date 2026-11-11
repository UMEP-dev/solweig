"""
Common parameter definitions for SOLWEIG algorithms.

Provides reusable parameter builders for consistent UI across algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgis.core import (
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDateTime,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterMatrix,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

if TYPE_CHECKING:
    from qgis.core import QgsProcessingAlgorithm


def _canvas_center_latlon() -> tuple[float, float]:
    """Return (lat, lon) of the current map canvas centre in WGS 84.

    Falls back to (0, 0) when the canvas is not available (e.g. headless).
    """
    try:
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
        from qgis.utils import iface

        canvas = iface.mapCanvas()
        center = canvas.center()
        project_crs = canvas.mapSettings().destinationCrs()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")

        if project_crs != wgs84:
            xform = QgsCoordinateTransform(project_crs, wgs84, QgsProject.instance())
            center = xform.transform(center)

        return round(center.y(), 4), round(center.x(), 4)
    except Exception:
        return 0.0, 0.0


def add_surface_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add standard surface data input parameters.

    Parameters added:
        DSM (required): Digital Surface Model
        DSM_HEIGHT_MODE: DSM height convention (0=relative, 1=absolute)
        CDSM (optional): Canopy DSM (vegetation heights)
        CDSM_HEIGHT_MODE: CDSM height convention (0=relative, 1=absolute)
        DEM (optional): Digital Elevation Model (ground)
        TDSM (optional): Trunk zone DSM
        TDSM_HEIGHT_MODE: TDSM height convention (0=relative, 1=absolute)
        LAND_COVER (optional): Land cover classification
    """
    _height_options = [
        "Relative — above ground",
        "Absolute — above sea level",
    ]

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "DSM",
            algorithm.tr("Digital Surface Model (DSM)"),
            optional=False,
        )
    )
    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "DSM_HEIGHT_MODE",
            algorithm.tr("DSM height convention"),
            options=_height_options,
            defaultValue=1,  # Absolute (most common for DSM)
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "CDSM",
            algorithm.tr("Canopy DSM (vegetation heights)"),
            optional=True,
        )
    )
    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "CDSM_HEIGHT_MODE",
            algorithm.tr("CDSM height convention"),
            options=_height_options,
            defaultValue=0,  # Relative (most common for CDSM)
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "DEM",
            algorithm.tr("Digital Elevation Model (ground)"),
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "TDSM",
            algorithm.tr("Trunk zone DSM"),
            optional=True,
        )
    )
    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "TDSM_HEIGHT_MODE",
            algorithm.tr("TDSM height convention"),
            options=_height_options,
            defaultValue=0,  # Relative (most common for TDSM)
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "LAND_COVER",
            algorithm.tr("Land cover classification (UMEP IDs)"),
            optional=True,
        )
    )

    min_obj_ht = QgsProcessingParameterNumber(
        "MIN_OBJECT_HEIGHT",
        algorithm.tr(
            "Minimum object height (m) — DSM features shorter than this "
            "above the DEM are flattened to remove kerbs, street furniture, "
            "and LiDAR noise from shadow casting. Requires DEM. Set to 0 to disable."
        ),
        type=QgsProcessingParameterNumber.Type.Double,
        defaultValue=1.0,
        minValue=0.0,
        maxValue=10.0,
        optional=True,
    )
    min_obj_ht.setFlags(min_obj_ht.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(min_obj_ht)


def add_location_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add location parameters with auto-extraction option.

    Parameters added:
        AUTO_EXTRACT_LOCATION: Extract lat/lon from DSM CRS
        LATITUDE: Manual latitude input
        LONGITUDE: Manual longitude input
        UTC_OFFSET: UTC timezone offset
    """
    algorithm.addParameter(
        QgsProcessingParameterBoolean(
            "AUTO_EXTRACT_LOCATION",
            algorithm.tr("Auto-extract location from DSM CRS (ignored when EPW weather file is used)"),
            defaultValue=False,
        )
    )

    canvas_lat, canvas_lon = _canvas_center_latlon()

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "LATITUDE",
            algorithm.tr("Latitude (degrees, ignored when EPW weather file is used)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=canvas_lat,
            minValue=-90.0,
            maxValue=90.0,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "LONGITUDE",
            algorithm.tr("Longitude (degrees, ignored when EPW weather file is used)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=canvas_lon,
            minValue=-180.0,
            maxValue=180.0,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "UTC_OFFSET",
            algorithm.tr("UTC offset (hours, ignored when EPW weather file is used)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0,
            minValue=-12,
            maxValue=14,
        )
    )


def add_weather_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add weather parameters for single timestep calculation.

    Parameters added:
        DATETIME: Date and time of calculation
        TEMPERATURE: Air temperature (°C)
        HUMIDITY: Relative humidity (%)
        GLOBAL_RADIATION: Global solar radiation (W/m²)
        WIND_SPEED: Wind speed (m/s)
        PRESSURE: Atmospheric pressure (hPa)
    """
    algorithm.addParameter(
        QgsProcessingParameterDateTime(
            "DATETIME",
            algorithm.tr("Date and time"),
            type=QgsProcessingParameterDateTime.Type.DateTime,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "TEMPERATURE",
            algorithm.tr("Air temperature (°C)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=25.0,
            minValue=-50.0,
            maxValue=60.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "HUMIDITY",
            algorithm.tr("Relative humidity (%)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=50.0,
            minValue=0.0,
            maxValue=100.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "GLOBAL_RADIATION",
            algorithm.tr("Global solar radiation (W/m²)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=800.0,
            minValue=0.0,
            maxValue=1400.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "WIND_SPEED",
            algorithm.tr("Wind speed (m/s)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=1.0,
            minValue=0.0,
            maxValue=50.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "PRESSURE",
            algorithm.tr("Atmospheric pressure (hPa)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=1013.25,
            minValue=800.0,
            maxValue=1100.0,
        )
    )


def add_human_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add human body parameters.

    Parameters added:
        POSTURE: Standing or sitting
        ABS_K: Shortwave absorption coefficient
    """
    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "POSTURE",
            algorithm.tr("Body posture"),
            options=["Standing", "Sitting"],
            defaultValue=0,  # Standing
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "ABS_K",
            algorithm.tr("Shortwave absorption coefficient"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.7,
            minValue=0.0,
            maxValue=1.0,
        )
    )


def add_human_body_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add detailed human body parameters for PET calculation.

    Parameters added:
        AGE, WEIGHT, HEIGHT, SEX, ACTIVITY, CLOTHING
    """
    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "AGE",
            algorithm.tr("Age (years)"),
            type=QgsProcessingParameterNumber.Type.Integer,
            defaultValue=35,
            minValue=1,
            maxValue=120,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "WEIGHT",
            algorithm.tr("Body weight (kg)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=75.0,
            minValue=20.0,
            maxValue=200.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "HEIGHT",
            algorithm.tr("Body height (m)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=1.75,
            minValue=1.0,
            maxValue=2.5,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "SEX",
            algorithm.tr("Sex"),
            options=["Male", "Female"],
            defaultValue=0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "ACTIVITY",
            algorithm.tr("Metabolic activity (W)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=80.0,
            minValue=40.0,
            maxValue=500.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "CLOTHING",
            algorithm.tr("Clothing insulation (clo)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.9,
            minValue=0.0,
            maxValue=2.0,
        )
    )


def add_options_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add calculation options parameters.

    Parameters added:
        USE_ANISOTROPIC_SKY: Enable anisotropic sky model
        CONIFER: Treat vegetation as evergreen
        SVF_DIR: Override SVF directory (optional)
        MAX_SHADOW_DISTANCE: Maximum horizontal shadow distance in metres
    """

    algorithm.addParameter(
        QgsProcessingParameterBoolean(
            "USE_ANISOTROPIC_SKY",
            algorithm.tr("Use anisotropic sky model"),
            defaultValue=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterBoolean(
            "CONIFER",
            algorithm.tr("Treat vegetation as evergreen (conifer)"),
            defaultValue=False,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterFile(
            "SVF_DIR",
            algorithm.tr("Override SVF directory (SVF is included in prepared surface by default)"),
            behavior=QgsProcessingParameterFile.Behavior.Folder,
            optional=True,
        )
    )

    max_shadow = QgsProcessingParameterNumber(
        "MAX_SHADOW_DISTANCE",
        algorithm.tr("Maximum shadow distance (m) — caps horizontal shadow ray reach"),
        type=QgsProcessingParameterNumber.Type.Double,
        defaultValue=1000.0,
        minValue=50.0,
        maxValue=5000.0,
    )
    max_shadow.setFlags(max_shadow.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(max_shadow)


def add_vegetation_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """Add vegetation transmissivity parameters (advanced).

    Parameters added:
        TRANSMISSIVITY: Leaf-on canopy transmissivity (0-1)
        TRANSMISSIVITY_LEAFOFF: Leaf-off (winter) transmissivity (0-1)
        LEAF_START: First day of year with leaves (1-366)
        LEAF_END: Last day of year with leaves (1-366)
    """

    trans_on = QgsProcessingParameterNumber(
        "TRANSMISSIVITY",
        algorithm.tr("Vegetation transmissivity — leaf-on season (0 = opaque, 1 = transparent)"),
        type=QgsProcessingParameterNumber.Type.Double,
        defaultValue=0.03,
        minValue=0.0,
        maxValue=1.0,
    )
    trans_on.setFlags(trans_on.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(trans_on)

    trans_off = QgsProcessingParameterNumber(
        "TRANSMISSIVITY_LEAFOFF",
        algorithm.tr("Vegetation transmissivity — leaf-off season (bare branches)"),
        type=QgsProcessingParameterNumber.Type.Double,
        defaultValue=0.5,
        minValue=0.0,
        maxValue=1.0,
    )
    trans_off.setFlags(trans_off.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(trans_off)

    leaf_start = QgsProcessingParameterNumber(
        "LEAF_START",
        algorithm.tr("First day of year with leaves (1–366)"),
        type=QgsProcessingParameterNumber.Type.Integer,
        defaultValue=97,
        minValue=1,
        maxValue=366,
    )
    leaf_start.setFlags(leaf_start.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(leaf_start)

    leaf_end = QgsProcessingParameterNumber(
        "LEAF_END",
        algorithm.tr("Last day of year with leaves (1–366)"),
        type=QgsProcessingParameterNumber.Type.Integer,
        defaultValue=300,
        minValue=1,
        maxValue=366,
    )
    leaf_end.setFlags(leaf_end.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(leaf_end)


def add_land_cover_mapping_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """Add land cover material properties table (advanced).

    Creates a pre-populated matrix table mapping integer land cover codes to
    surface material properties.  Defaults match the UMEP standard.  Users can
    edit values in-place, add rows for additional codes, or provide a full
    custom ``parametersforsolweig.json`` file.

    Parameters added:
        LC_MATERIALS: Matrix table (Code, Name, Albedo, Emissivity, TgK, Tstart, TmaxLST)
        CUSTOM_MATERIALS_FILE: Optional custom materials JSON (overrides table)
    """

    # UMEP standard defaults as flat list (7 columns per row)
    # fmt: off
    umep_defaults = [
        0, "Paved",     0.20, 0.95, 0.37, -3.41, 15.0,
        1, "Asphalt",   0.18, 0.95, 0.58, -9.78, 15.0,
        2, "Buildings", 0.18, 0.95, 0.58, -9.78, 15.0,
        5, "Grass",     0.16, 0.94, 0.21, -3.38, 14.0,
        6, "Bare soil", 0.25, 0.94, 0.33, -3.01, 14.0,
        7, "Water",     0.05, 0.98, 0.00,  0.00, 12.0,
    ]
    # fmt: on

    materials = QgsProcessingParameterMatrix(
        "LC_MATERIALS",
        algorithm.tr("Land cover material properties"),
        headers=[
            "Code",
            "Name",
            "Albedo",
            "Emissivity",
            "TgK (Ts_deg)",
            "Tstart",
            "TmaxLST",
        ],
        hasFixedNumberRows=False,
        numberRows=6,
        defaultValue=umep_defaults,
        optional=True,
    )
    materials.setFlags(materials.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(materials)

    custom_file = QgsProcessingParameterFile(
        "CUSTOM_MATERIALS_FILE",
        algorithm.tr("Custom materials JSON (overrides table)"),
        extension="json",
        optional=True,
    )
    custom_file.setFlags(custom_file.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(custom_file)


def add_heat_threshold_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """Add UTCI heat-stress threshold parameters for timeseries summary.

    Parameters added:
        HEAT_THRESHOLDS_DAY: Comma-separated UTCI thresholds for daytime (°C)
        HEAT_THRESHOLDS_NIGHT: Comma-separated UTCI thresholds for nighttime (°C)
    """
    algorithm.addParameter(
        QgsProcessingParameterString(
            "HEAT_THRESHOLDS_DAY",
            algorithm.tr("Daytime UTCI thresholds (°C, comma-separated)"),
            defaultValue="32, 38",
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterString(
            "HEAT_THRESHOLDS_NIGHT",
            algorithm.tr("Nighttime UTCI thresholds (°C, comma-separated)"),
            defaultValue="26",
            optional=True,
        )
    )


def add_output_tmrt_parameter(algorithm: QgsProcessingAlgorithm) -> None:
    """Add Tmrt output raster parameter."""
    algorithm.addParameter(
        QgsProcessingParameterRasterDestination(
            "OUTPUT_TMRT",
            algorithm.tr("Mean Radiant Temperature (Tmrt)"),
        )
    )


def add_output_dir_parameter(algorithm: QgsProcessingAlgorithm) -> None:
    """Add output directory parameter."""
    algorithm.addParameter(
        QgsProcessingParameterFolderDestination(
            "OUTPUT_DIR",
            algorithm.tr("Output directory"),
        )
    )


def add_epw_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add EPW weather file parameter.

    Parameters added:
        EPW_FILE: Path to EPW file
    """
    algorithm.addParameter(
        QgsProcessingParameterFile(
            "EPW_FILE",
            algorithm.tr("EPW weather file (overrides location coordinates and UTC offset)"),
            extension="epw",
            optional=True,
        )
    )


def add_umep_met_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add UMEP/SUEWS meteorological file parameter.

    Parameters added:
        UMEP_MET_FILE: Path to UMEP/SUEWS met file
        UMEP_HOURLY_ONLY: For a sub-hourly file, run only the on-the-hour records
    """
    algorithm.addParameter(
        QgsProcessingParameterFile(
            "UMEP_MET_FILE",
            algorithm.tr("UMEP/SUEWS meteorological forcing file"),
            extension="txt",
            optional=True,
        )
    )
    # Off by default: every record is a timestep, as in UMEP. Up to
    # 0.1.0b100 the plugin always kept only the on-the-hour records.
    hourly_only = QgsProcessingParameterBoolean(
        "UMEP_HOURLY_ONLY",
        algorithm.tr("Sub-hourly met file: run only the on-the-hour records (faster; values are not hourly means)"),
        defaultValue=False,
    )
    hourly_only.setFlags(hourly_only.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    algorithm.addParameter(hourly_only)


def add_date_filter_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add shared date/time filter parameters (used by EPW and UMEP modes).

    Parameters added:
        START_DATE: Start date for filtering
        END_DATE: End date for filtering
        HOURS_FILTER: Comma-separated hours to include
    """
    algorithm.addParameter(
        QgsProcessingParameterDateTime(
            "START_DATE",
            algorithm.tr("Start date (leave empty to use core API default)"),
            type=QgsProcessingParameterDateTime.Type.DateTime,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterDateTime(
            "END_DATE",
            algorithm.tr("End date (leave empty to use core API default)"),
            type=QgsProcessingParameterDateTime.Type.DateTime,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterString(
            "HOURS_FILTER",
            algorithm.tr("Hours to include (comma-separated, e.g., 9,10,11,12)"),
            optional=True,
        )
    )
