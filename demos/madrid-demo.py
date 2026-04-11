# %%
"""
Demo: Madrid SOLWEIG - Automatic Tiling Stress Test (Full Raster)

The goal of this demo is to exercise the **automatic tiling** path on the
entire Madrid raster — about 23028 × 21972 pixels (≈500 M pixels, 55 km × 58 km
at 2.5 m/pixel) — for one full day (24 hourly timesteps), with no manual
``tile_size`` override. Both the SVF preparation and the per-timestep
calculation are forced to split the work into many tiles by the resource-aware
sizer based on real GPU/RAM limits. Watch the logs for ``Resource-aware tile
sizing`` lines emitted from ``solweig.tiling``.

Source rasters are copied once into ``temp/madrid/source/`` so this demo runs
self-contained against local data (the originals live under the
``cities-dataset-spain`` repo).

Data sources
------------
- bdsm.tif: Normalised DSM — relative building heights above ground (nDSM).
  Derived from PNOA-LiDAR point cloud data.
  Source: Instituto Geográfico Nacional (IGN), Spain — https://pnoa.ign.es/pnoa-lidar
  Licence: CC BY 4.0
- cdsm.tif: Relative vegetation canopy heights.
  Derived from PNOA-LiDAR point cloud data (vegetation classification).
  Source: Instituto Geográfico Nacional (IGN), Spain — https://pnoa.ign.es/pnoa-lidar
  Licence: CC BY 4.0
- dem.tif: Digital Elevation Model (terrain baseline, native 5 m — auto-resampled
  by ``SurfaceData.prepare`` to match the 2.5 m DSM grid).
  Derived from PNOA-LiDAR point cloud data.
  Source: Instituto Geográfico Nacional (IGN), Spain — https://pnoa.ign.es/pnoa-lidar
  Licence: CC BY 4.0

Native CRS: EPSG:25830 (ETRS89 / UTM zone 30N)
Raster extent: (410679, 4442245) – (465663, 4499872)
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import solweig
from solweig.tiling import compute_max_tile_side

# Working folders — all under temp/madrid for isolation
output_folder_path = Path("temp/madrid").absolute()
source_path = output_folder_path / "source"
output_dir = output_folder_path / "output_simplified"
output_folder_path.mkdir(parents=True, exist_ok=True)

# %%
# Report the auto-computed tile caps so the test is self-documenting.
# The full raster is intentionally larger than these caps in both dimensions,
# so the resource-aware tiler must subdivide for both SVF and the per-step pass.
solweig._ensure_gpu_initialized()
svf_cap = compute_max_tile_side(context="svf")
solweig_cap = compute_max_tile_side(context="solweig")
print(f"Auto tile cap (svf):     {svf_cap} px")
print(f"Auto tile cap (solweig): {solweig_cap} px")

# %%
# Step 1: Prepare surface data over the FULL raster
# - bdsm.tif contains relative building heights → dsm_relative=True
# - cdsm.tif contains relative vegetation heights (cdsm_relative=True default)
# - dem.tif is at 5 m; prepare() resamples it to the 2.5 m DSM grid automatically
# - bbox is intentionally NOT set — we want the full ~500 M-pixel extent
# - tile_size is intentionally NOT set — let the automatic sizer pick it
surface = solweig.SurfaceData.prepare(
    dsm=str(source_path / "bdsm.tif"),
    dem=str(source_path / "dem.tif"),
    cdsm=str(source_path / "cdsm.tif"),
    working_dir=str(output_folder_path / "working"),
    pixel_size=2.5,
    dsm_relative=True,
    cdsm_relative=True,
    # bbox=[434450.0, 4477350.0, 435950.0, 4478850.0],  # <-- artifact-area test window
    # dem_smooth_sigma=3.0,  # default
)

print(f"\nPrepared surface: {surface.dsm.shape[1]}×{surface.dsm.shape[0]} pixels")

# %%
# Step 2: Build a synthetic 24-hour summer day
# Hourly timesteps from 00:00 → 23:00 local time. Mid-July, clear-sky-ish profile.
# This is enough variation to verify the tile-outer orchestration through the
# diurnal radiation cycle without depending on a bundled EPW file.
location = solweig.Location.from_dsm_crs(
    str(source_path / "bdsm.tif"),
    utc_offset=2,  # CEST (Madrid summer time)
    altitude=650.0,  # Madrid average elevation
)


def _diurnal(hour: int) -> tuple[float, float, float]:
    """Crude clear-sky diurnal profile: (ta °C, rh %, global_rad W/m²)."""
    # Air temperature: ~20°C at 05:00 → ~36°C at 15:00
    ta = 28.0 - 8.0 * math.cos((hour - 15) / 24.0 * 2 * math.pi)
    # Relative humidity: anti-correlated with temperature
    rh = 55.0 - 25.0 * math.cos((hour - 5) / 24.0 * 2 * math.pi)
    # Global radiation: triangular daytime ramp, zero at night
    rad = max(0.0, 950.0 * math.sin((hour - 6) / 14.0 * math.pi)) if 6 <= hour <= 20 else 0.0
    return ta, rh, rad


base = datetime(2024, 7, 15, 0, 0)
weather_list = []
for hour in range(24):
    ta, rh, rad = _diurnal(hour)
    weather_list.append(
        solweig.Weather.from_values(
            ta=ta,
            rh=rh,
            global_rad=rad,
            datetime=base + timedelta(hours=hour),
            ws=2.0,
        )
    )

# %%
# Step 3: Run the full timeseries — automatic tiling end-to-end
# tile_size is intentionally NOT set so the unified tile-outer pipeline picks
# the size from real GPU/RAM limits. Logs will show the chosen tile geometry
# and progress through each tile.
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    use_anisotropic_sky=True,
    conifer=False,
    output_dir=str(output_dir),
    outputs=["tmrt", "shadow"],
    max_shadow_distance_m=500,
)
print(summary.report())

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

im4 = axes[1, 1].imshow(summary.tmrt_max, cmap="hot")
axes[1, 1].set_title("Max Tmrt (°C)")
plt.colorbar(im4, ax=axes[1, 1])

threshold = sorted(summary.utci_hours_above.keys())[0]
im5 = axes[1, 2].imshow(summary.utci_hours_above[threshold], cmap="Reds")
axes[1, 2].set_title(f"UTCI hours > {threshold}°C")
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"SOLWEIG Madrid full-raster auto-tiled — {len(summary)} timesteps")
plt.tight_layout()
plt.show()

# %%
