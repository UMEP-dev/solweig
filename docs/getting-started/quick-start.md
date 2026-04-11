# Quick Start

This guide demonstrates a complete SOLWEIG calculation, from input data to a Tmrt map.

## Overview

1. Create a surface with buildings
2. Define the location and time
3. Calculate Mean Radiant Temperature
4. Inspect the results

## Option A: From numpy arrays (no files required)

This approach is suitable for experimentation or when GeoTIFF data is not available.

```python
import numpy as np
import solweig
from datetime import datetime

# --- 1. Create a surface ---
# A 200x200 m flat area at 2 m elevation, with a 15 m tall building in the centre
dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0  # 40x40 m building

surface = solweig.SurfaceData.prepare(dsm=dsm, pixel_size=1.0)  # 1 pixel = 1 metre

# --- 2. Define location and weather ---
location = solweig.Location(
    latitude=48.8,      # Paris
    longitude=2.3,
    utc_offset=1,       # Central European Time (UTC+1)
)

weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),  # 14:00, 15 July
    ta=32.0,            # Air temperature (deg C)
    rh=40.0,            # Relative humidity (%)
    global_rad=850.0,   # Global horizontal irradiance (W/m2)
)

# --- 3. Calculate ---
summary = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")

# --- 4. Inspect results ---
print(f"Mean Tmrt:   {summary.tmrt_mean.mean():.1f} deg C")
print(f"Max Tmrt:    {summary.tmrt_max.max():.1f} deg C")
print(f"Sun hours:   {summary.sun_hours.max():.1f}")
```

## Option B: From GeoTIFF files (real-world data)

This is the standard workflow for applied projects. The following inputs are required:

- A **DSM** GeoTIFF (Digital Surface Model — building and terrain heights)
- An **EPW** weather file (standard format, available from climate databases)

```python
import solweig

# --- 1. Load and prepare surface ---
# Wall heights, sky view factors, and NaN handling are performed during preparation
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",        # Preprocessing cached here for reuse
    cdsm="data/trees.tif",       # Optional: vegetation canopy heights
)

# --- 2. Load weather and location from EPW ---
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",          # Date range to simulate
    end="2025-07-03",
)
location = solweig.Location.from_epw("data/weather.epw")

print(f"Location: {location.latitude:.1f} deg N, {location.longitude:.1f} deg E")
print(f"Loaded {len(weather_list)} hourly timesteps")

# --- 3. Run ---
# Results saved as GeoTIFFs; thermal state carried between timesteps
results = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

print(f"Completed — {results.n_timesteps} timesteps")
```

The returned `TimeseriesSummary` contains aggregated grids (mean/max/min Tmrt,
UTCI, sun hours, etc.) — see [Timeseries](../guide/timeseries.md).

### Surface preparation steps

`SurfaceData.prepare()` performs the following operations:

1. Loads the DSM (and optional CDSM, DEM, land cover)
2. Fills NaN/nodata values using the ground reference
3. Detects integer-quantized DEMs (e.g. 1 m int16 LiDAR products) and
   applies a small Gaussian smooth to break stair-step contours in SVF —
   see `smooth_quantized_dem` / `dem_smooth_sigma` on `prepare()`
4. Computes **wall heights and aspects** from the DSM edges
5. Computes **Sky View Factors** (15 directional grids)
6. Caches results to `working_dir/` with a preparation fingerprint, so
   subsequent runs on unchanged inputs short-circuit via `SurfaceData.load()`
   and skip the entire load → resample → preprocess → walls → SVF pipeline.
   Warm runs complete in ~50 ms; cache mismatches are logged in detail
   (which source file, which kwarg).

## Adding thermal comfort indices

Tmrt quantifies the radiation absorbed by a person, but thermal comfort also depends on air temperature, humidity, and wind speed. UTCI and PET integrate all of these variables.

UTCI and PET summary grids are included in the `TimeseriesSummary` by default.
To save per-timestep GeoTIFFs, include `"utci"` or `"pet"` in `outputs`:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "utci"],  # per-timestep GeoTIFFs
)
print(summary.report())  # Full summary with Tmrt, UTCI, sun hours, thresholds
```

| UTCI range | Classification |
| ---------- | -------------- |
| > 46 deg C | Extreme heat stress |
| 38–46 deg C | Very strong heat stress |
| 32–38 deg C | Strong heat stress |
| 26–32 deg C | Moderate heat stress |
| 9–26 deg C | No thermal stress |
| < 9 deg C | Cold stress categories |

## Common setup patterns

The following patterns address variations in input data and workflow.

### Surface setup patterns

#### Pattern 1: GeoTIFF workflow (recommended)

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    cdsm="data/trees.tif",      # Optional
    dem="data/dem.tif",         # Optional
    working_dir="cache/",
)
```

`prepare()` computes and caches walls and SVF.

#### Pattern 2: In-memory arrays with absolute heights

```python
import numpy as np

dsm_abs = np.array(...)         # Absolute elevation (e.g., m above sea level)
cdsm_abs = np.array(...)        # Optional canopy elevation (absolute)

surface = solweig.SurfaceData.prepare(
    dsm=dsm_abs,
    cdsm=cdsm_abs,              # Optional
    cdsm_relative=False,
    pixel_size=1.0,
)

location = solweig.Location(latitude=48.8, longitude=2.3, utc_offset=1)
```

#### Pattern 3: In-memory arrays with relative heights

```python
import numpy as np

dsm_rel = np.array(...)         # Height above ground
cdsm_rel = np.array(...)        # Optional canopy height above ground
dem = np.array(...)             # Ground elevation

surface = solweig.SurfaceData.prepare(
    dsm=dsm_rel,
    dem=dem,
    cdsm=cdsm_rel,              # Optional
    dsm_relative=True,
    cdsm_relative=True,
    pixel_size=1.0,
)
```

### Weather setup patterns

#### Pattern 1: Existing EPW file

```python
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
```

#### Pattern 2: Download EPW, then load

```python
epw_path = solweig.download_epw(
    latitude=37.98,
    longitude=23.73,
    output_path="athens.epw",
)
weather_list = solweig.Weather.from_epw(epw_path)
```

#### Pattern 3: Single timestep (manual)

```python
from datetime import datetime

weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),
    ta=32.0,
    rh=40.0,
    global_rad=850.0,
    ws=2.0,                     # Optional; used for UTCI/PET
)
```

### Location setup patterns

#### Pattern 1: From EPW metadata (recommended with EPW weather)

```python
location = solweig.Location.from_epw("data/weather.epw")
```

#### Pattern 2: From surface CRS

```python
location = solweig.Location.from_surface(surface, utc_offset=1)
```

#### Pattern 3: Manual coordinates

```python
location = solweig.Location(
    latitude=48.8,
    longitude=2.3,
    utc_offset=1,
)
```

!!! warning "Always set `utc_offset` correctly"
    UTC offset directly affects sun position timing and therefore shadows and Tmrt.

## Input data sources

### DSM (Digital Surface Model)

A raster grid where each pixel contains the height in metres (including buildings and terrain). Common sources:

- **LiDAR point clouds** processed to raster (national mapping agencies often provide these)
- **Photogrammetry** from drone surveys
- **OpenStreetMap building footprints** extruded to heights

### EPW (EnergyPlus Weather)

Hourly weather data in a standard format. Sources:

- [Climate.OneBuilding.Org](https://climate.onebuilding.org/) — global coverage, multi-year TMY files
- [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) — European Commission TMY generator ([methodology](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en))

EPW files can also be downloaded from PVGIS programmatically (no API key required):

```python
epw_path = solweig.download_epw(
    latitude=37.98,
    longitude=23.73,
    output_path="athens.epw",
)

weather_list = solweig.Weather.from_epw(epw_path)
location = solweig.Location.from_epw(epw_path)
```

!!! note "TMY files represent typical climate, not a specific year"
    A Typical Meteorological Year (TMY) is a statistical composite — each calendar month is selected from the most "typical" month across a multi-year reference period (2005–2020 for PVGIS v5.3). The result represents long-term average conditions, not observations from any single year. Data freshness depends on the upstream PVGIS release, not on SOLWEIG. See the [PVGIS TMY documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en) for full methodology and data sources.

!!! note "Data attribution"
    PVGIS weather data is derived from ERA5 reanalysis and contains modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

### CDSM (Canopy Digital Surface Model)

Vegetation canopy heights, either from LiDAR or by rasterising tree survey data:

```python
import geopandas as gpd

trees = gpd.read_file("trees.gpkg")
cdsm, transform = solweig.io.rasterise_gdf(
    trees, "geometry", "height",
    bbox=[minx, miny, maxx, maxy],
    pixel_size=1.0,
)
```

## Key classes

| Class | Description |
| ----- | ----------- |
| `SurfaceData` | DSM, optional vegetation/DEM/land cover, preprocessed walls and SVF |
| `Location` | Latitude, longitude, altitude, UTC offset |
| `Weather` | Air temperature, humidity, radiation for one timestep |
| `HumanParams` | Body parameters for Tmrt/PET (optional — defaults provided) |
| `SolweigResult` | Output grids: Tmrt, shadow, radiation components |

## Working demos

The repository includes end-to-end demos:

- **[demos/athens-demo.py](https://github.com/UMEP-dev/solweig/blob/main/demos/athens-demo.py)** — Full workflow: rasterise tree vectors, load GeoTIFFs, run a multi-day timeseries, post-process UTCI.
- **[demos/bilbao-demo.py](https://github.com/UMEP-dev/solweig/blob/main/demos/bilbao-demo.py)** — Terrain-aware shadows in a mountain valley: relative building heights (`dsm_relative=True`), separate DEM, and `max_shadow_distance_m` for hillside shadow reach.
- **[demos/solweig_gbg_test.py](https://github.com/UMEP-dev/solweig/blob/main/demos/solweig_gbg_test.py)** — Gothenburg test data: surface preparation, SVF caching, and timeseries calculation.

## Further reading

- [Basic Usage](../guide/basic-usage.md) — Vegetation, height conventions, custom parameters, validation
- [Working with GeoTIFFs](../guide/geotiffs.md) — File loading, caching, saving results
- [Timeseries](../guide/timeseries.md) — Multi-day simulations with thermal state
- [Thermal Comfort](../guide/thermal-comfort.md) — UTCI and PET
- [API Reference](../api/index.md) — All classes and functions
