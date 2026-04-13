# Basic Usage

This guide covers the core concepts and common patterns for using SOLWEIG.

## Required Inputs

Every SOLWEIG calculation requires three inputs:

### 1. Surface — the physical environment

A `SurfaceData` object holds building/terrain heights and optional vegetation. The only required field is a DSM (Digital Surface Model).

!!! tip "Arrays or files?"
    Use **numpy arrays** for synthetic grids, parameter sweeps, or when data is already in memory. Use **GeoTIFF files** for real-world data — `prepare()` handles CRS, alignment, caching, and NaN filling.

**From numpy arrays** (in-memory workflows):

```python
import numpy as np
import solweig

# Minimum: a height grid
dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0  # A building

surface = solweig.SurfaceData.prepare(dsm=dsm, pixel_size=1.0)
```

**From GeoTIFF files** (file-based workflows — rasters are aligned to their intersecting extent, walls and SVF are computed and cached to `working_dir/`):

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
)
```

### 2. Location — geographic coordinates

A `Location` provides the geographic position required for sun position calculations.

```python
# Manual specification
location = solweig.Location(latitude=48.8, longitude=2.3, utc_offset=1)

# From the DSM's coordinate system
location = solweig.Location.from_surface(surface, utc_offset=1)

# From an EPW weather file header
location = solweig.Location.from_epw("weather.epw")
```

!!! warning "Always set `utc_offset`"
    The UTC offset determines how clock time maps to sun position. An incorrect offset shifts shadows by hours. When creating a `Location` manually, always provide `utc_offset` explicitly.

### 3. Weather — atmospheric conditions

A `Weather` object holds the meteorological data for one point in time.

```python
from datetime import datetime

weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),
    ta=32.0,          # Air temperature (deg C)
    rh=40.0,          # Relative humidity (%)
    global_rad=850.0, # Global horizontal irradiance (W/m2)
    ws=2.0,           # Wind speed (m/s) — used for UTCI/PET
)
```

`ws` is optional for Tmrt calculations (defaults to 1.0 m/s) but directly affects UTCI and PET results. A difference of 1–3 m/s can shift UTCI by several degrees; measured wind data should be used when computing thermal comfort indices.

For timeseries, load from an EPW file:

```python
weather_list = solweig.Weather.from_epw(
    "weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
```

### Downloading weather data

If no EPW file is available, a Typical Meteorological Year (TMY) file can be downloaded from PVGIS (no API key required):

```python
epw_path = solweig.download_epw(
    latitude=37.98,      # Athens
    longitude=23.73,
    output_path="athens.epw",
)
weather_list = solweig.Weather.from_epw(epw_path)
location = solweig.Location.from_epw(epw_path)
```

!!! note "TMY files represent typical climate, not a specific year"
    A TMY is a statistical composite — each calendar month is selected from the most "typical" month across a multi-year reference period (2005–2023 for PVGIS v5.3). The result represents long-term average conditions, not observations from any single year. Because each month is taken from a real historical year, the row timestamps in the file will legitimately span multiple years within the reference window (e.g. a January from 2022 alongside a February from 2014) — this is the defining property of a TMY, not an artefact. Data freshness depends on the upstream [PVGIS release](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en), not on SOLWEIG.

## Running a Calculation

### Single timestep

```python
summary = solweig.calculate(
    surface=surface,
    weather=[weather],
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
```

### Multiple timesteps (timeseries)

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
```

When given a list of weather timesteps, `calculate()` carries **thermal state** between timesteps — ground and wall temperatures from one hour affect the next. Large rasters are tiled to fit GPU memory. Looping over `calculate()` with single timesteps does not carry thermal state and may produce discontinuities.

Per-timestep GeoTIFFs are saved to `output_dir` when `outputs` is specified. Summary grids (mean/max/min Tmrt, UTCI, sun hours) are saved to `output_dir/summary/`.

See [Timeseries](timeseries.md) for further detail.

## Understanding the Output

Each per-timestep GeoTIFF and summary grid has the same dimensions as the input DSM:

| Field | Unit | Description |
| ----- | ---- | ----------- |
| `tmrt` | deg C | **Mean Radiant Temperature** — radiation absorbed by a person. The primary output. |
| `shadow` | 0–1 | Shadow fraction. 1 = sunlit, 0 = shaded. |
| `kdown` | W/m2 | Incoming shortwave radiation (direct + diffuse). |
| `kup` | W/m2 | Reflected shortwave radiation from the ground. |
| `ldown` | W/m2 | Incoming longwave radiation (thermal, from sky + walls). |
| `lup` | W/m2 | Emitted longwave radiation from the ground. |

**Typical ranges** (mid-latitude summer, midday): Tmrt 20–75 deg C (shaded–sunlit), UTCI 25–45 deg C. Values of Tmrt above 80 deg C or UTCI above 55 deg C may indicate input data issues — common causes are incorrect CRS, missing DEM, or unusually high DSM values.

The `TimeseriesSummary` returned by `calculate()` contains aggregated grids:

```python
print(f"Mean Tmrt: {summary.tmrt_mean.mean():.1f} deg C")
print(f"Max UTCI: {np.nanmax(summary.utci_max):.1f} deg C")
print(f"Sun hours range: {summary.sun_hours.min():.1f} – {summary.sun_hours.max():.1f}")
```

## Adding Vegetation

Trees reduce Tmrt through shading. A Canopy DSM (CDSM) — a grid of vegetation canopy heights above ground — is provided as an additional input.

```python
# From arrays
cdsm = np.zeros_like(dsm)
cdsm[10:40, 50:80] = 8.0  # 8 m tall trees

surface = solweig.SurfaceData.prepare(dsm=dsm, cdsm=cdsm, pixel_size=1.0)

# From GeoTIFF
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    cdsm="data/trees.tif",
    working_dir="cache/",
)
```

### Relative vs. absolute heights

CDSM (and TDSM) heights can be either **relative** (height above ground) or **absolute** (elevation above sea level). By default, SOLWEIG assumes CDSM values are relative.

```python
# Relative CDSM (default): values represent height above ground
# e.g. 8.0 = an 8 m tall tree
surface = solweig.SurfaceData(
    dsm=dsm,
    cdsm=cdsm,
    cdsm_relative=True,   # Default — ground elevation is added during preprocessing
    pixel_size=1.0,
)
surface.preprocess()  # Converts relative to absolute using DEM or DSM as base

# Absolute CDSM: values represent elevation above sea level
# e.g. 135.0 = tree canopy at 135 m elevation
surface = solweig.SurfaceData(
    dsm=dsm,
    cdsm=cdsm_absolute,
    cdsm_relative=False,  # No conversion applied
    pixel_size=1.0,
)
```

When using `SurfaceData.prepare()` with GeoTIFFs, this conversion is handled during preparation.

The same flags exist for DSM and TDSM:

```python
# DSM with relative heights (height above ground) — requires a DEM
surface = solweig.SurfaceData(
    dsm=dsm_relative,
    dem=dem,
    dsm_relative=True,
    pixel_size=1.0,
)
surface.preprocess()  # dsm_absolute = dem + dsm_relative
```

### Deciduous vs. evergreen trees

By default, SOLWEIG applies seasonal leaf-on/leaf-off based on the date. For evergreen trees (conifers):

```python
results = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    conifer=True,  # Year-round canopy
    output_dir="output/",
)
```

## Customising Human Parameters

The default person is standing, 75 kg, 1.75 m tall, 35 years old. These values can be modified:

```python
result = solweig.calculate(
    surface, weather=[weather], location=location,
    human=solweig.HumanParams(
        posture="sitting",   # or "standing" (default)
        abs_k=0.7,           # Shortwave absorption (0–1)
        abs_l=0.97,          # Longwave absorption (0–1)
        weight=65,           # kg (affects PET only)
        height=1.65,         # m (affects PET only)
    ),
    output_dir="output/",
)
```

## Anisotropic Sky Model

The anisotropic sky model is **enabled by default** (`use_anisotropic_sky=True`). It distributes diffuse radiation
across the sky dome according to the Perez model rather than treating it as uniform. `SurfaceData.prepare()` computes the required SVF and shadow matrices.

To disable it (uniform sky distribution):

```python
results = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    use_anisotropic_sky=False,
    output_dir="output/",
)
```

## Input Validation

Validate inputs before running a calculation:

```python
try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")
    result = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")
except solweig.GridShapeMismatch as e:
    print(f"Grid size mismatch: {e.field} expected {e.expected}, got {e.got}")
except solweig.MissingPrecomputedData as e:
    print(f"Missing data: {e}")
```

## Common Issues

### NaN pixels in output

NaN values in DSM/CDSM are filled with the ground reference (DEM, or the DSM itself if no DEM is provided). If output still contains NaN, providing a DEM may increase valid coverage.

### Slow first calculation

SVF must be prepared before `calculate()`. `SurfaceData.prepare()` computes SVF as part of the preparation step. For file workflows, use a persistent `working_dir` so SVF is cached and reused across runs.

### GPU not detected

```python
print(f"GPU: {solweig.is_gpu_available()}")
print(f"Backend: {solweig.get_compute_backend()}")
```

If no GPU is available, the package falls back to CPU. GPU acceleration reduces SVF and shadow computation time.

### CRS must be projected (metres)

SOLWEIG requires pixel distances in metres for shadow and SVF calculations. If the GeoTIFF uses a geographic CRS (lat/lon in degrees), `prepare()` raises an error. Reproject to a projected CRS (e.g. UTM) first:

```bash
gdalwarp -t_srs EPSG:32634 input.tif output_utm.tif
```

### GridShapeMismatch

All input grids must have the same dimensions. When loading from GeoTIFFs, `prepare()` resamples to a common grid. When passing numpy arrays, all arrays must have the same shape before calling `prepare()`.

### MissingPrecomputedData

This error indicates that `calculate()` could not find SVF data. Ensure that `SurfaceData.prepare()` (which computes SVF) was used rather than constructing `SurfaceData` directly.

## Working Demos

For end-to-end examples:

- **[demos/athens-demo.py](https://github.com/UMEP-dev/solweig/blob/main/demos/athens-demo.py)** — Full GeoTIFF workflow with tree vectors, multi-day timeseries, and UTCI post-processing.
- **[demos/bilbao-demo.py](https://github.com/UMEP-dev/solweig/blob/main/demos/bilbao-demo.py)** — Terrain-aware shadows in a mountain valley: relative building heights (`dsm_relative=True`), separate DEM, and `max_shadow_distance_m` for hillside shadow reach.
- **[demos/solweig_gbg_test.py](https://github.com/UMEP-dev/solweig/blob/main/demos/solweig_gbg_test.py)** — Gothenburg test data: surface preparation, SVF caching, and timeseries.
