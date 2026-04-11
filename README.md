# SOLWEIG

**Map how hot it _feels_ across a city — pixel by pixel.**

SOLWEIG computes **Mean Radiant Temperature (Tmrt)** and thermal comfort indices (**UTCI**, **PET**) for urban environments. Give it a building height model and weather data, and it produces high-resolution maps showing where people experience heat stress — and where trees, shade, and cool surfaces make a difference.

Adapted from the [UMEP](https://github.com/UMEP-dev/UMEP-processing) (Urban Multi-scale Environmental Predictor) platform by Fredrik Lindberg, Sue Grimmond, and contributors — see Lindberg et al. ([2008](https://doi.org/10.1007/s00484-008-0162-7), [2018](https://doi.org/10.1016/j.envsoft.2017.09.020)). Re-implemented in Rust for speed, with optional GPU acceleration.

![UTCI thermal comfort map](docs/UTCI.jpg)
_DSM/DEM data: [PNOA-LiDAR](https://pnoa.ign.es/pnoa-lidar), Instituto Geográfico Nacional (IGN), Spain. CC BY 4.0._

> **Experimental:** This package and QGIS plugin are released for testing and discussion purposes. The API is stabilising but may change. Feedback and bug reports welcome — [open an issue](https://github.com/UMEP-dev/solweig/issues).

**[Documentation](https://umep-dev.github.io/solweig/)** · [Installation](https://umep-dev.github.io/solweig/getting-started/installation/) · [Quick Start](https://umep-dev.github.io/solweig/getting-started/quick-start/) · [API Reference](https://umep-dev.github.io/solweig/api/)

---

## What can you do with it?

- **Urban planning** — Compare street canyon designs, tree planting scenarios, or cool-roof strategies by mapping thermal comfort before and after.
- **Heat risk assessment** — Identify the hottest spots in a neighbourhood during a heatwave, hour by hour.
- **Research** — Run controlled microclimate experiments at 1 m resolution with full radiation budgets.
- **Climate services** — Generate thermal comfort maps for public health warnings or outdoor event planning.

## How it works

SOLWEIG models the complete radiation budget experienced by a person standing in an urban environment:

1. **Shadows** — Which pixels are shaded by buildings and trees at a given sun angle?
2. **Sky View Factor (SVF)** — How much sky can a person see from each point? (More sky = more incoming longwave and diffuse radiation.)
3. **Surface temperatures** — How hot are the ground and surrounding walls, accounting for thermal inertia across the diurnal cycle?
4. **Radiation balance** — Sum shortwave (sun) and longwave (heat) radiation from all directions, using either isotropic or Perez anisotropic sky models.
5. **Tmrt** — Convert total absorbed radiation into Mean Radiant Temperature.
6. **Thermal comfort** — Optionally derive UTCI or PET, which combine Tmrt with air temperature, humidity, and wind.

The computation pipeline is implemented in Rust and exposed to Python via PyO3. Shadow casting and anisotropic sky calculations can optionally run on the GPU via WebGPU. Large rasters are automatically tiled to fit GPU memory constraints.

---

## Install

```bash
pip install solweig
```

**Requirements:** Python 3.11–3.13. Pre-built wheels are available for Linux, macOS, and Windows.

### From source

```bash
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
pip install maturin
maturin develop --release
```

This compiles the Rust extension locally. A Rust toolchain is required.

---

## Quick start

### Minimal example (numpy arrays)

```python
import numpy as np
import solweig
from datetime import datetime

# A flat surface with one 15 m building
dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0

surface = solweig.SurfaceData.prepare(dsm=dsm, pixel_size=1.0)

location = solweig.Location(latitude=48.8, longitude=2.3, utc_offset=1)  # Paris
weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),
    ta=32.0,          # Air temperature (°C)
    rh=40.0,          # Relative humidity (%)
    global_rad=850.0, # Solar radiation (W/m²)
)

summary = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")

print(f"Mean Tmrt: {summary.tmrt_mean.mean():.0f}°C")
print(f"Max UTCI:  {np.nanmax(summary.utci_max):.0f}°C")
```

### Real-world workflow (GeoTIFFs + EPW weather)

```python
import solweig

# 1. Load surface — prepare() computes and caches walls/SVF when missing
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    cdsm="data/trees.tif",       # Optional: vegetation canopy heights
    working_dir="cache/",        # Expensive preprocessing cached here
)

# 2. Load weather from an EPW file (standard format from climate databases)
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
location = solweig.Location.from_epw("data/weather.epw")

# 3. Run — outputs saved as GeoTIFFs, thermal state carried between timesteps
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

# 4. Inspect results
print(summary.report())
summary.plot()
```

---

## API overview

### Core classes

| Class               | Purpose                                                                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SurfaceData`       | Holds all spatial inputs (DSM, CDSM, DEM, land cover) and precomputed arrays (walls, SVF). Use `.prepare()` to load GeoTIFFs with automatic caching.  |
| `Location`          | Geographic coordinates (latitude, longitude, UTC offset). Create from coordinates, DSM CRS, or an EPW file.                                           |
| `Weather`           | Per-timestep meteorological data (air temperature, relative humidity, global radiation, optional wind speed). Load from EPW files or create manually. |
| `SolweigResult`     | Output grids from a single timestep: Tmrt, shadow, UTCI, PET, radiation components.                                                                   |
| `TimeseriesSummary` | Aggregated results from a multi-timestep run: mean/max/min grids, sun hours, UTCI threshold exceedance, per-timestep scalars.                         |
| `HumanParams`       | Body parameters: posture (standing/sitting), absorption coefficients, PET body parameters (age, weight, height, etc.).                                |
| `ModelConfig`       | Runtime settings: anisotropic sky, max shadow distance, tiling workers.                                                                               |

### Main functions

```python
# Single timestep
summary = solweig.calculate(surface, weather=[weather], output_dir="output/")

# Multi-timestep with thermal inertia (auto-tiles large rasters)
summary = solweig.calculate(surface, weather=weather_list, output_dir="output/")

# Include UTCI and/or PET in per-timestep GeoTIFFs
summary = solweig.calculate(
    surface, weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "utci", "shadow"],
)

# Input validation
warnings = solweig.validate_inputs(surface, location, weather)
```

### Convenience I/O

```python
# Load/save GeoTIFFs
data, transform, crs, nodata = solweig.io.load_raster("dsm.tif")
solweig.io.save_raster("output.tif", data, transform, crs)

# Rasterise vector data (e.g., tree polygons → height grid)
raster, transform = solweig.io.rasterise_gdf(gdf, "geometry", "height", bbox=bbox, pixel_size=1.0)

# Download EPW weather data (no API key needed)
epw_path = solweig.download_epw(latitude=37.98, longitude=23.73, output_path="athens.epw")
```

---

## Inputs and outputs

### What you need

| Input          | Required? | What it is                                                                                                |
| -------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| **DSM**        | Yes       | Digital Surface Model — a height grid (metres) including buildings. GeoTIFF or numpy array.               |
| **Location**   | Yes       | Latitude, longitude, and UTC offset. Can be extracted from the DSM's CRS or an EPW file.                  |
| **Weather**    | Yes       | Air temperature, relative humidity, and global solar radiation. Load from an EPW file or create manually. |
| **CDSM**       | No        | Canopy heights (trees). Adds vegetation shading.                                                          |
| **DEM**        | No        | Ground elevation. Separates terrain from buildings.                                                       |
| **Land cover** | No        | Surface type grid (paved, grass, water, etc.). Affects surface temperatures.                              |

### What you get

| Output      | Unit | Description                                                                               |
| ----------- | ---- | ----------------------------------------------------------------------------------------- |
| **Tmrt**    | °C   | Mean Radiant Temperature — how much radiation a person absorbs.                           |
| **Shadow**  | 0–1  | Shadow fraction (1 = sunlit, 0 = fully shaded).                                           |
| **UTCI**    | °C   | Universal Thermal Climate Index — "feels like" temperature.                               |
| **PET**     | °C   | Physiological Equivalent Temperature — similar to UTCI with customisable body parameters. |
| Kdown / Kup | W/m² | Shortwave radiation (down and reflected up).                                              |
| Ldown / Lup | W/m² | Longwave radiation (thermal, down and emitted up).                                        |

### Timeseries summary grids

When running `calculate()` with a list of weather timesteps, the returned `TimeseriesSummary` provides aggregated grids across all timesteps:

| Grid                                | Description                                                 |
| ----------------------------------- | ----------------------------------------------------------- |
| `tmrt_mean`, `tmrt_max`, `tmrt_min` | Overall Tmrt statistics                                     |
| `tmrt_day_mean`, `tmrt_night_mean`  | Day/night Tmrt averages                                     |
| `utci_mean`, `utci_max`, `utci_min` | Overall UTCI statistics                                     |
| `utci_day_mean`, `utci_night_mean`  | Day/night UTCI averages                                     |
| `sun_hours`, `shade_hours`          | Hours of direct sun / shade per pixel                       |
| `utci_hours_above`                  | Dict of threshold → grid of hours exceeding that UTCI value |

Plus a `Timeseries` object with per-timestep spatial means (Tmrt, UTCI, sun fraction, air temperature, radiation, etc.) for plotting.

### Don't have an EPW file? Download one

```python
epw_path = solweig.download_epw(latitude=37.98, longitude=23.73, output_path="athens.epw")
weather_list = solweig.Weather.from_epw(epw_path)
```

---

## Configuration

### Human body parameters

```python
human = solweig.HumanParams(
    posture="standing",  # or "sitting"
    abs_k=0.7,           # Shortwave absorption coefficient
    abs_l=0.97,          # Longwave absorption coefficient
    # PET-specific:
    age=35, weight=75, height=1.75, sex=1, activity=80, clothing=0.9,
)
summary = solweig.calculate(surface, weather=[weather], location=location, human=human, output_dir="output/")
```

### Model options

Key parameters accepted by `calculate()`:

| Parameter               | Default      | Description                                                                                                        |
| ----------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------ |
| `use_anisotropic_sky`   | `True`       | Use Perez anisotropic sky model for more accurate diffuse radiation.                                               |
| `conifer`               | `False`      | Treat trees as evergreen (skip seasonal leaf-off).                                                                 |
| `max_shadow_distance_m` | `1000`       | Maximum shadow reach in metres. Increase for mountainous terrain.                                                  |
| `output_dir`            | _(required)_ | Working directory for all output (summary grids, per-timestep GeoTIFFs, metadata).                                 |
| `outputs`               | `None`       | Which per-timestep grids to save: `"tmrt"`, `"utci"`, `"pet"`, `"shadow"`, `"kdown"`, `"kup"`, `"ldown"`, `"lup"`. |

### Physics and materials

```python
# Custom vegetation transmissivity, posture geometry, etc.
physics = solweig.load_physics("custom_physics.json")

# Custom surface materials (albedo, emissivity per land cover class)
materials = solweig.load_materials("site_materials.json")

summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    physics=physics,
    materials=materials,
    output_dir="output/",
)
```

---

## GPU acceleration

SOLWEIG uses WebGPU (via wgpu/Rust) for shadow casting and anisotropic sky computations. GPU is enabled by default when available.

```python
import solweig

# Check GPU status
print(solweig.is_gpu_available())     # True/False
print(solweig.get_compute_backend())  # "gpu" or "cpu"
print(solweig.get_gpu_limits())       # {"max_buffer_size": ..., "backend": "Metal"}

# Disable GPU (fall back to CPU)
solweig.disable_gpu()
```

Large rasters are automatically tiled to fit within GPU buffer limits. Tile size, worker count, and prefetch depth are configurable via `ModelConfig` or keyword arguments.

---

## Run metadata and reproducibility

Every timeseries run records a `run_metadata.json` in the output directory capturing the full parameter set:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")
print(metadata["solweig_version"])
print(metadata["location"])
print(metadata["parameters"]["use_anisotropic_sky"])
print(metadata["timeseries"]["start"], "to", metadata["timeseries"]["end"])
```

---

## QGIS plugin

SOLWEIG is also available as a **QGIS Processing plugin** for point-and-click spatial analysis — no Python scripting required.

### Installation

1. **Plugins** → **Manage and Install Plugins**
2. **Settings** tab → Check **"Show also experimental plugins"**
3. Search for **"SOLWEIG"** → **Install Plugin**

The plugin requires QGIS 4.0+ (Qt6, Python 3.11+). On first use it will offer to install the `solweig` Python library automatically.

### Processing algorithms

Once installed, SOLWEIG algorithms appear in the **Processing Toolbox** under the SOLWEIG group:

| Algorithm                           | Description                                                                                        |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Download / Preview Weather File** | Download a TMY EPW file from PVGIS, or preview an existing EPW file.                               |
| **Prepare Surface Data**            | Align rasters, compute wall heights, wall aspects, and SVF. Results are cached and reused.         |
| **SOLWEIG Calculation**             | Single-timestep or timeseries Tmrt with optional inline UTCI/PET. Supports EPW and UMEP met files. |

### QGIS-specific features

- All inputs and outputs are standard QGIS raster layers (GeoTIFF)
- Automatic tiling for large rasters with GPU support
- QGIS progress bar integration with cancellation support
- Configurable vegetation parameters (transmissivity, seasonal leaf dates, conifer/deciduous)
- Configurable land cover materials table
- UTCI heat stress thresholds for day and night
- Run metadata saved alongside outputs for reproducibility

### Typical QGIS workflow

1. **Surface Preparation** — Load your DSM (and optionally CDSM, DEM, land cover). The algorithm computes walls, SVF, and caches everything to a working directory.
2. **Tmrt Timeseries** — Point to the prepared surface directory and an EPW file. Select your date range, outputs, and run. Results are saved as GeoTIFFs and loaded into the QGIS canvas.
3. **Inspect results** — Use standard QGIS tools to style, compare, and export the output layers.

---

## Demos

Complete working scripts:

- **[demos/athens-demo.py](demos/athens-demo.py)** — Full workflow: rasterise tree vectors, load GeoTIFFs, run a multi-day timeseries, visualise summary grids.
- **[demos/bilbao-demo.py](demos/bilbao-demo.py)** — Valley urban canyon with `dsm_relative=True`, terrain-aware shadows, and `max_shadow_distance_m` for bounded horizontal reach.
- **[demos/madrid-demo.py](demos/madrid-demo.py)** — Automatic tiling stress test on a 500 M-pixel raster (55 km × 58 km at 2.5 m), exercising resource-aware tile sizing and the full timeseries path end-to-end.
- **[demos/solweig_gbg_test.py](demos/solweig_gbg_test.py)** — Gothenburg: surface preparation with SVF caching, timeseries calculation.

---

## Validation

SOLWEIG is validated against field radiation measurements from three sites in Gothenburg, Sweden (Lindberg et al. 2008, 2011). All geodata, measurements, and test scripts are checked into the repository and run as part of the test suite.

| Site                                  | Season          | Days |  Tmrt RMSE |   Tmrt R² |
| ------------------------------------- | --------------- | ---: | ---------: | --------: |
| Kronenhuset (courtyard, 1 m)          | Autumn          |    1 |     6.7 °C |      0.51 |
| Gustav Adolfs torg (open square, 2 m) | Autumn + Summer |    3 | 5.7-7.5 °C | 0.78-0.87 |
| GVC (university campus, 2 m)          | Summer          |    3 | 1.5-6.1 °C | 0.79-0.99 |

Anisotropic sky mode, matched daytime observation hours. Full details, radiation budget comparisons, and version history: **[Validation Report](VALIDATION.md)**.

---

## Citation

Adapted from [UMEP](https://github.com/UMEP-dev/UMEP-processing) by Fredrik Lindberg, Sue Grimmond, and contributors.

If you use SOLWEIG in your research, please cite the original model paper and the UMEP platform:

1. Lindberg F, Holmer B, Thorsson S (2008) SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. _International Journal of Biometeorology_ 52, 697–713 [doi:10.1007/s00484-008-0162-7](https://doi.org/10.1007/s00484-008-0162-7)

2. Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) – An integrated tool for city-based climate services. _Environmental Modelling and Software_ 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## Demo data

The Athens demo dataset (`demos/data/athens/`) uses the following sources:

- **DSM/DEM** — Derived from LiDAR data available via the [Hellenic Cadastre geoportal](https://www.ktimatologio.gr/)
- **Tree vectors** (`trees.gpkg`) — Derived from the [Athens Urban Atlas](https://land.copernicus.eu/local/urban-atlas) and municipal open data at [geodata.gov.gr](https://geodata.gov.gr/)
- **EPW weather** (`athens_2023.epw`) — Generated using Copernicus Climate Change Service information [2025] via [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/). Contains modified Copernicus Climate Change Service information; neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
