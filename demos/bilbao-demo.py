# %%
"""
Demo: Bilbao SOLWEIG - Valley Urban Canyon

Bilbao occupies a tight river valley (the Nervión) flanked by hills rising
200–400 m above the city floor, creating a classic "urban bowl" where terrain
shadows and building shadows interact in ways that flat-city models miss.
This demo covers a 3 km × 3 km extract centred on the Casco Viejo
(old town) and the riverfront, at 2.5 m resolution.

Key features demonstrated:
- Relative building heights (dsm_relative=True) with separate DEM
- Terrain-aware shadows from surrounding hillsides
- max_shadow_distance_m controls horizontal shadow reach independently
  of terrain relief

Data sources
------------
- BDSM.tif: Normalised DSM — relative building heights above ground (nDSM).
  Derived from PNOA-LiDAR point cloud data.
  Source: Instituto Geográfico Nacional (IGN), Spain — https://pnoa.ign.es/pnoa-lidar
  Licence: CC BY 4.0
- CDSM.tif: Relative vegetation canopy heights.
  Derived from PNOA-LiDAR point cloud data (vegetation classification).
  Source: Instituto Geográfico Nacional (IGN), Spain — https://pnoa.ign.es/pnoa-lidar
  Licence: CC BY 4.0
- DEM.tif: Digital Elevation Model (terrain baseline).
  Derived from PNOA-LiDAR point cloud data.
  Source: Instituto Geográfico Nacional (IGN), Spain — https://pnoa.ign.es/pnoa-lidar
  Licence: CC BY 4.0
- bilbao_2021.epw: EPW weather file for Bilbao (43.29°N, −2.97°E).
  Source: EnergyPlus Weather Data, U.S. Department of Energy —
  https://energyplus.net/weather

Original data extent: EPSG:25830 (ETRS89 / UTM zone 30N)
Crop bbox: [499600, 4794000, 502600, 4797000]  (Casco Viejo / Nervión riverfront / hillsides)
"""

from pathlib import Path

import solweig

# Working folders
input_folder = "demos/data/bilbao"
input_path = Path(input_folder).absolute()
output_folder = "temp/bilbao"
output_folder_path = Path(output_folder).absolute()
output_folder_path.mkdir(parents=True, exist_ok=True)
output_dir = output_folder_path / "output_simplified"

# 3 km × 3 km extract: Casco Viejo, Nervión riverfront, and flanking hillsides
# EPSG:25830 (ETRS89 / UTM zone 30N)
EXTENTS_BBOX = [499600, 4794000, 502600, 4797000]

# %%
# Step 1: Prepare surface data
# - BDSM.tif contains relative building heights (height above ground)
#   so we use dsm_relative=True which converts via DSM = DEM + nDSM
# - CDSM.tif contains relative vegetation heights (cdsm_relative=True by default)
# - DEM provides the terrain baseline — essential for valley shadow geometry
# - During prepare(): walls/SVF are computed and cached to working_dir
surface = solweig.SurfaceData.prepare(
    dsm=str(input_path / "BDSM.tif"),
    dem=str(input_path / "DEM.tif"),
    cdsm=str(input_path / "CDSM.tif"),
    working_dir=str(output_folder_path / "working"),
    # bbox=EXTENTS_BBOX,
    pixel_size=2.5,
    dsm_relative=True,  # BDSM is height above ground, not absolute elevation
    cdsm_relative=True,  # CDSM is height above ground (default)
    # tile_size=500,  # Optional
)

# %%
# Step 2: Load weather data and location from EPW file
epw_path = str(input_path / "bilbao_2021.epw")
weather_list = solweig.Weather.from_epw(
    epw_path,
    start="2021-07-01",
    end="2021-07-04",  # 4 days: July 1–4
)
location = solweig.Location.from_epw(epw_path)  # lat, lon, UTC offset, elevation

# %%
# Step 3: Calculate Tmrt timeseries
# max_shadow_distance_m=500 lets terrain shadows reach across the valley floor
# while keeping computation bounded.
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    use_anisotropic_sky=True,
    conifer=False,
    output_dir=str(output_dir),
    outputs=["tmrt", "shadow"],
    max_shadow_distance_m=2000,
    # tile_size=500,  # Optional
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

threshold = sorted(summary.utci_hours_above.keys())[0]
im5 = axes[1, 2].imshow(summary.utci_hours_above[threshold], cmap="Reds")
axes[1, 2].set_title(f"UTCI hours > {threshold}°C")
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"SOLWEIG Bilbao — {len(summary)} timesteps ({summary.n_daytime} day, {summary.n_nighttime} night)")
plt.tight_layout()
plt.show()

# %%
# Optional: Load and inspect run metadata
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

# Example 1: Custom human parameters
# results = solweig.calculate(
#     surface=surface,
#     weather=weather_list,
#     location=location,
#     human=solweig.HumanParams(
#         abs_k=0.65,
#         abs_l=0.97,
#         weight=70,
#         height=1.65,
#         posture="sitting",
#     ),
#     output_dir=str(output_dir),
# )

# Example 2: Wider shadow reach for full valley-to-valley coverage
# summary = solweig.calculate(
#     surface=surface,
#     weather=weather_list,
#     location=location,
#     max_shadow_distance_m=1000,  # Extend to capture distant hillside shadows
#     output_dir=str(output_dir),
#     outputs=["tmrt", "shadow"],
# )

# %%
