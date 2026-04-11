# QGIS Plugin

SOLWEIG is available as a **QGIS Processing plugin** for spatial analysis without Python scripting. All algorithms run within the QGIS Processing Toolbox and produce standard GeoTIFF outputs that load into the map canvas.

!!! warning "Experimental"
    This package and plugin are experimental — released for testing and discussion purposes. The API is stabilising but may change. Feedback and bug reports are welcome via [GitHub Issues](https://github.com/UMEP-dev/solweig/issues).

## Requirements

- **QGIS 4.0+** (Qt6, Python 3.11+)
- The `solweig` Python library (installed on first use)

## Installation

1. Open **Plugins** > **Manage and Install Plugins**
2. Go to the **Settings** tab and check **"Show also experimental plugins"**
3. Switch to the **All** tab and search for **"SOLWEIG"**
4. Click **Install Plugin**

On first use the plugin checks whether the `solweig` library is installed. If it is missing or outdated, a dialog offers to install or upgrade it via pip.

!!! tip "Manual library install"
    If the automatic install fails (e.g. behind a proxy), open the QGIS Python Console and run:

    ```python
    import pip; pip.main(["install", "--upgrade", "solweig"])
    ```

---

## Processing Algorithms

Once installed, three algorithms appear in the **Processing Toolbox** under the SOLWEIG group. They are numbered to indicate the recommended workflow order.

### 1. Download / Preview Weather File

Download a Typical Meteorological Year (TMY) weather file from the EU PVGIS service, or preview the contents of an existing EPW file.

| Parameter | Description |
| --------- | ----------- |
| **Mode** | *Download from PVGIS* or *Preview existing file* |
| **Latitude / Longitude** | Defaults to the map canvas centre |
| **Output EPW** | Save location for the downloaded file |
| **EPW File** | Path to an existing file (preview mode) |

**Download mode** fetches a Typical Meteorological Year (TMY) file from the [PVGIS API](https://re.jrc.ec.europa.eu/pvg_tools/en/) — global coverage, no API key required. A TMY is a statistical composite of typical months selected from the ERA5 reanalysis reference period (2005–2020 for PVGIS v5.3); it represents long-term average climate, not observations from a specific year. See the [PVGIS TMY documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en) for methodology. **Preview mode** generates an HTML report with location, date range, and variable statistics.

### 2. Prepare Surface Data (align, walls, SVF)

Load raster layers, align them to a common grid, compute wall heights, wall aspects, and Sky View Factor. Results are cached to a working directory so subsequent calculations reuse the preprocessing.

| Parameter | Description |
| --------- | ----------- |
| **DSM** | Digital Surface Model (required) |
| **CDSM** | Canopy/vegetation height model (optional) |
| **DEM** | Digital Elevation Model (optional) |
| **TDSM** | Trunk zone heights (optional) |
| **Land cover** | Surface classification grid (optional) |
| **Height modes** | Relative (to ground) or absolute for each input |
| **Extent** | Custom processing area (optional — defaults to DSM extent) |
| **Pixel size** | Output resolution in metres (0 = native DSM resolution) |
| **Vegetation settings** | Transmissivity, leaf-on/off dates, conifer flag |
| **Land cover materials** | Albedo and emissivity per surface class |

**Output directory structure:**

```
prepared_surface/
  metadata.json              # Grid info + preparation fingerprint (warm-run cache key)
  parametersforsolweig.json  # Saved material/vegetation settings (UMEP-compatible)
  cleaned/
    dsm.tif                  # Aligned, absolute DSM
    wall_height.tif          # Computed wall heights
    wall_aspect.tif          # Computed wall aspects
    cdsm.tif                 # Aligned CDSM (if provided)
    dem.tif                  # Aligned DEM, Gaussian-smoothed if quantized
    tdsm.tif                 # Aligned TDSM (if provided)
    land_cover.tif           # Aligned land cover (if provided)
  walls/
    px<size>/                # Per-pixel-size wall cache
  svf/
    px<size>/
      memmap/                # Memmap-backed SVF arrays
      svfs.zip               # SVF arrays (UMEP-compatible archive)
      shadowmats.npz         # Shadow matrices for anisotropic sky (small rasters only)
      shadow_memmaps/        # Shadow memmap fallback for large rasters
```

**Warm-run cache**: rerunning *Prepare Surface Data* with unchanged inputs
short-circuits the full load → resample → preprocess → walls → SVF pipeline
by comparing a fingerprint of source file mtimes/sizes and preparation kwargs
stored in `metadata.json`. A cache hit completes in ~50 ms instead of tens of
seconds. If any source file is edited or any kwarg changes, the log reports
exactly which input invalidated the cache and the pipeline rebuilds from
source.

### 3. SOLWEIG Calculation

Run Tmrt and optional thermal comfort calculations using a prepared surface directory and weather data.

**Weather sources** (select one):

| Source | Description |
| ------ | ----------- |
| Single timestep | Enter date/time and weather values manually |
| EPW file | Load a timeseries from an EPW weather file |
| UMEP met file | Load a timeseries from a UMEP meteorological file |

**Date filtering** (EPW and UMEP modes): optional start/end dates and hour-of-day filter (e.g. `9,10,11,12,13,14,15,16,17` for daytime only).

**Key options:**

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| Anisotropic sky | On | Use Perez sky model (requires shadow matrices from surface preparation) |
| Conifer | Off | Treat trees as evergreen (skip seasonal leaf-off) |
| Max shadow distance | 1 000 m | Increase for mountainous terrain |
| Compute UTCI | Off | Calculate Universal Thermal Climate Index |
| Compute PET | Off | Calculate Physiological Equivalent Temperature (~50x computation time vs UTCI) |
| Heat stress thresholds | — | UTCI thresholds for day/night exceedance grids |

**Output selection:** choose which per-timestep grids to save — Tmrt, shadow, Kdown, Kup, Ldown, Lup.

**Output directory structure:**

```
results/
  tmrt/    tmrt_YYYYMMDD_HHMM.tif
  shadow/  shadow_YYYYMMDD_HHMM.tif
  utci/    utci_YYYYMMDD_HHMM.tif
  ...
```

For timeseries runs, a summary report is logged with mean/max Tmrt, UTCI, sun hours, and heat stress exceedance statistics.

---

## Typical Workflow

```
┌─────────────────────────────────────┐
│  1. Download / Preview Weather File │
│     → Obtain an EPW file            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  2. Prepare Surface Data            │
│     → Load DSM, compute walls & SVF │
│     → Cached for reuse              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  3. SOLWEIG Calculation             │
│     → Point to prepared directory   │
│     → Select EPW + date range       │
│     → Choose outputs (Tmrt, UTCI…)  │
│     → Results load into QGIS canvas │
└─────────────────────────────────────┘
```

1. **Obtain weather data** — run *Download / Preview Weather File* to fetch a TMY EPW, or use an existing EPW file.
2. **Prepare the surface** — run *Prepare Surface Data* with the DSM (and optionally CDSM, DEM, land cover). This computes walls and SVF once; subsequent calculations reuse the cache.
3. **Calculate** — run *SOLWEIG Calculation*, pointing to the prepared surface directory and the EPW file. Select a date range, enable UTCI/PET if required, and run. Results are saved as GeoTIFFs and loaded into the canvas.

!!! tip "Re-running with different parameters"
    Surface preparation is the most computationally intensive step and is cached. Once complete, the calculation step can be re-run with different weather files, date ranges, or human parameters without repeating surface preparation.

---

## QGIS-Specific Features

**Automatic tiling** — Large rasters are divided into overlapping tiles sized to fit GPU memory. Tile buffers account for building heights and shadow distance to ensure continuity at boundaries.

**GPU acceleration** — Shadow casting and anisotropic sky computations run on the GPU when available (Metal on macOS, Vulkan on Linux, DirectX on Windows). The system falls back to multi-threaded CPU if no GPU is detected.

**Progress and cancellation** — All algorithms report progress through the QGIS Task Manager with per-timestep status. Long-running calculations can be cancelled.

**Canvas integration** — Single-timestep results are loaded into the QGIS canvas with thermal comfort colour ramps. Timeseries results are saved as GeoTIFFs for manual loading and styling.

**Dependency management** — The plugin detects whether the `solweig` library is installed and at the correct version. On first use or after a plugin update, it offers to install or upgrade via pip.

**GDAL backend** — The plugin uses QGIS's bundled GDAL for all raster I/O, avoiding additional pip dependencies such as rasterio.
