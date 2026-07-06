# %%
"""
Demo: Gothenburg SOLWEIG preprocessing

This demo shows how to use the solweig package for:
1. Wall height and aspect generation
2. Sky View Factor (SVF) calculation
3. Land-cover-based surface properties (albedo, emissivity)

Uses SurfaceData.prepare() which automatically computes and caches
walls and SVF in the working directory.
"""

from pathlib import Path

import solweig

# %%
# Working folder and input files
working_folder = "temp/goteborg"
working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)

# Input files
dsm_path = "demos/data/Goteborg_SWEREF99_1200/DSM_KRbig.tif"
cdsm_path = "demos/data/Goteborg_SWEREF99_1200/CDSM_KRbig.tif"
dem_path = "demos/data/Goteborg_SWEREF99_1200/DEM_KRbig.tif"
land_cover_path = "demos/data/Goteborg_SWEREF99_1200/landcover.tif"

# Setup parameters
trunk_ratio = 0.25  # Trunk height as fraction of canopy height

# %%
# Prepare surface data with automatic wall and SVF computation
# SurfaceData.prepare() will:
#   - Fill NaN in DSM/CDSM/TDSM with the ground reference (DEM or DSM)
#   - Compute wall heights/aspects and cache in working_dir/walls/
#   - Compute SVF and cache in working_dir/svf/
#   - Reuse cached data on subsequent runs (use force_recompute=True to regenerate)
print("Preparing surface data (walls and SVF will be computed if not cached)...")
print(f"  Working dir: {working_path}")
print(f"GPU acceleration: {'enabled' if solweig.GPU_ENABLED else 'disabled'}")

surface = solweig.SurfaceData.prepare(
    dsm=dsm_path,
    cdsm=cdsm_path,
    dem=dem_path,
    land_cover=land_cover_path,
    working_dir=str(working_path),
    trunk_ratio=trunk_ratio,
    # bbox=None,  # Full extent (default)
    # force_recompute=False,  # Use cached data if available (default)
)

# %%
# The surface object is now ready for SOLWEIG calculations:
#
weather_list = solweig.Weather.from_umep_met(
    "demos/data/Goteborg_SWEREF99_1200/GBG_TMY_1977.txt",
    start="1977-07-01",
    end="1977-07-05",  # 5 days: July 1-5
)
# Location from surface CRS with explicit UTC offset (Gothenburg: CET = UTC+1)
location = solweig.Location.from_surface(surface, utc_offset=1)
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    output_dir=str(working_path / "output"),
    outputs=["tmrt", "shadow"],
)
print(summary.report())
summary.plot()

# %%
