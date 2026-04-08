# %%
"""
Demo: Athens SOLWEIG - Simplified API

This demo shows how to use the solweig package with the new simplified API.
The simplified API automatically handles:
- Wall height and aspect computation from DSM
- Sky View Factor (SVF) preparation and caching via ``SurfaceData.prepare()``
- Extent intersection and resampling
- CRS validation and extraction
- NaN filling in DSM/CDSM/TDSM with ground reference (DEM or DSM)

Legacy config file-driven workflows (parametersforsolweig.json) are
supported via ``ModelConfig.from_json()``.

Data sources
------------
- DSM/DEM: Derived from LiDAR data, Hellenic Cadastre (https://www.ktimatologio.gr/)
- Tree vectors (trees.gpkg): Derived from Athens Urban Atlas
  (https://land.copernicus.eu/local/urban-atlas) and geodata.gov.gr
- EPW weather (athens_2023.epw): Generated using Copernicus Climate Change
  Service information [2025] via PVGIS (https://re.jrc.ec.europa.eu/pvg_tools/en/).
  Contains modified Copernicus Climate Change Service information; neither the
  European Commission nor ECMWF is responsible for any use that may be made of
  the Copernicus information or data it contains.
"""

from pathlib import Path

import geopandas as gpd
import solweig
from pyproj import CRS

# Working folders
input_folder = "demos/data/athens"
input_path = Path(input_folder).absolute()
output_folder = "temp/athens"
output_folder_path = Path(output_folder).absolute()
output_folder_path.mkdir(parents=True, exist_ok=True)
output_dir = output_folder_path / "output_simplified"

# Extents for Athens demo area
EXTENTS_BBOX = [476800, 4205850, 477200, 4206250]
TARGET_CRS = 2100

# %%
# =============================================================================
# SIMPLIFIED API (Recommended)
# =============================================================================

# Generate CDSM from tree vector data
trees_gdf = gpd.read_file(input_folder + "/trees.gpkg")
trees_gdf = trees_gdf.to_crs(TARGET_CRS)
cdsm_rast, cdsm_transf = solweig.io.rasterise_gdf(
    trees_gdf,
    "geometry",
    "height",
    bbox=EXTENTS_BBOX,
    pixel_size=1.0,
)
solweig.io.save_raster(
    str(output_folder_path / "CDSM.tif"),
    cdsm_rast,
    cdsm_transf.to_gdal(),
    CRS.from_epsg(TARGET_CRS).to_wkt(),
)

# %%
# Step 1: Prepare surface data
# - CRS automatically extracted from DSM
# - NaN in DSM/CDSM/TDSM filled with ground reference (DEM or DSM)
# - During prepare(): walls/SVF are computed and cached to working_dir if not already cached
# - During calculate*(): SVF must already be present on the surface/precomputed data
# - Extent and resolution handled automatically
# - Resampled data saved to working_dir for inspection
surface = solweig.SurfaceData.prepare(
    dsm=str(input_path / "DSM.tif"),
    working_dir=str(output_folder_path / "working"),  # Cache preprocessing here
    cdsm=str(output_folder_path / "CDSM.tif"),
    bbox=EXTENTS_BBOX,  # Optional: specify extent
    pixel_size=1.0,  # Optional: specify resolution (default: from DSM)
    # tile_size=200,  # Optional: tile size for SVF computation (default: auto)
)

# Step 2: Load weather data and location from EPW file
epw_path = str(input_path / "athens_2023.epw")
weather_list = solweig.Weather.from_epw(
    epw_path,
    start="2023-07-01",
    end="2023-07-03",
)
location = solweig.Location.from_epw(epw_path)  # lat, lon, UTC offset, elevation

# %%
# Step 3: Calculate Tmrt with all defaults
# All parameters use bundled defaults:
#  - Human: abs_k=0.7, abs_l=0.95, standing, 75kg, 180cm, 35yo, 80W activity
#  - Physics: Tree transmissivity=0.03, seasonal dates, posture geometry
#  - No materials needed (no landcover grid)
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    use_anisotropic_sky=True,  # Uses precomputed SVF from prepare()
    conifer=False,  # Use seasonal leaf on/off (set True for evergreen trees)
    output_dir=str(output_dir),
    outputs=["tmrt", "shadow"],
    # tile_size=200,  # Optional: tile size for timeseries (default: auto)
)
print(summary.report())

# %%
# Plot timeseries (Ta, Tmrt, UTCI, radiation, sun exposure over time)
summary.plot()

# %%
# Visualise summary grids
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

im0 = axes[0, 0].imshow(summary.tmrt_mean, cmap="hot")
axes[0, 0].set_title("Mean Tmrt (°C)")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(summary.utci_mean, cmap="hot")
axes[0, 1].set_title("Mean UTCI (°C)")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(summary.sun_hours, cmap="YlOrRd")
axes[0, 2].set_title("Sun hours")
plt.colorbar(im2, ax=axes[0, 2])

im3 = axes[1, 0].imshow(summary.tmrt_day_mean, cmap="hot")
axes[1, 0].set_title("Mean daytime Tmrt (°C)")
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(summary.tmrt_night_mean, cmap="cool")
axes[1, 1].set_title("Mean nighttime Tmrt (°C)")
plt.colorbar(im4, ax=axes[1, 1])

# Show hours above the first day threshold (32°C by default)
threshold = sorted(summary.utci_hours_above.keys())[0]
im5 = axes[1, 2].imshow(summary.utci_hours_above[threshold], cmap="Reds")
axes[1, 2].set_title(f"UTCI hours > {threshold}°C")
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"SOLWEIG Summary — {len(summary)} timesteps ({summary.n_daytime} day, {summary.n_nighttime} night)")
plt.tight_layout()
plt.show()

# %%
# Optional: Load and inspect run metadata
# This metadata captures all parameters used in the calculation for reproducibility
metadata = solweig.load_run_metadata(output_dir / "run_metadata.json")
print("\nRun metadata loaded:")
print(f"  Timestamp: {metadata['run_timestamp']}")
print(f"  SOLWEIG version: {metadata['solweig_version']}")
print(f"  Location: {metadata['location']['latitude']:.2f}°N, {metadata['location']['longitude']:.2f}°E")
print(f"  Human posture: {metadata.get('human', {}).get('posture', 'default (standing)')}")
print(f"  Anisotropic sky: {metadata['parameters']['use_anisotropic_sky']}")
print(f"  Weather timesteps: {metadata['timeseries']['timesteps']}")
print(f"  Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")

# %%
# Optional parameter customization examples:

# Example 1: Custom human parameters (common use case)
# results = solweig.calculate(
#     surface=surface,
#     weather=weather_list,
#     human=solweig.HumanParams(
#         abs_k=0.65,       # Lower shortwave absorption
#         abs_l=0.97,       # Higher longwave absorption
#         weight=70,        # 70 kg
#         height=1.65,      # 165 cm
#         posture="sitting",
#     ),
#     output_dir=str(output_dir),
# )

# Example 2: Custom physics (e.g., different tree transmissivity)
# Create custom_trees.json with:
# {
#   "Tree_settings": {"Value": {"Transmissivity": 0.05, ...}},
#   "Posture": {"Standing": {...}, "Sitting": {...}}
# }
# physics = solweig.load_physics("custom_trees.json")
# results = solweig.calculate(
#     surface=surface,
#     weather=weather_list,
#     physics=physics,
#     output_dir=str(output_dir),
# )

# Example 3: Custom materials (requires landcover grid)
# surface_with_lc = solweig.SurfaceData.prepare(
#     dsm="dsm.tif",
#     land_cover="landcover.tif",  # Grid with class IDs (0-7, 99-102)
#     working_dir="cache/",
# )
# materials = solweig.load_materials("site_materials.json")  # Albedo, emissivity per class
# results = solweig.calculate(
#     surface=surface_with_lc,
#     weather=weather_list,
#     materials=materials,
#     output_dir=str(output_dir),
# )

# Legacy: Old unified params file (still supported for backwards compatibility)
# params = solweig.load_params("parametersforsolweig.json")
# Contains human + physics + materials in one file
# Note: Prefer the new three-parameter model for clarity!

# %%
# Step 4: Per-timestep UTCI/PET GeoTIFFs
# To save per-timestep UTCI or PET, include them in outputs:
#
# summary = solweig.calculate(
#     surface=surface,
#     weather=weather_list,
#     location=location,
#     output_dir=str(output_dir),
#     outputs=["tmrt", "utci"],  # per-timestep Tmrt + UTCI saved to disk
# )
#
# Note: Summary grids (utci_mean, utci_max, etc.) are always computed regardless.

# %%
# =============================================================================
# NOTE: Legacy API (SolweigRunRust, SolweigRunCore, configs.py) removed in Phase 5.6
# =============================================================================
# The legacy config-file-driven API has been removed. Use the modern simplified API above.
# Tiling is automatic for large rasters — no explicit tiling call needed.

# %%
