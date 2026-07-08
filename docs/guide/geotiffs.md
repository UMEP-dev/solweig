# Working with GeoTIFFs

This guide covers the loading, caching, and saving of GeoTIFF raster files in SOLWEIG.

## Loading Surface Data

`SurfaceData.prepare()` is the recommended method for loading GeoTIFFs. It handles loading, NaN filling, wall computation, SVF computation, and caching.

All input rasters must use a **projected CRS** with units in metres (e.g. UTM). Geographic CRS (lat/lon in degrees) is not supported — reproject with `gdalwarp` or an equivalent GIS tool.

```python
import solweig

surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",           # Required: building/terrain heights
    working_dir="cache/",          # Required: cache directory for preprocessing
    cdsm="data/trees.tif",        # Optional: vegetation heights
    dem="data/dem.tif",            # Optional: bare ground elevation
    land_cover="data/lc.tif",     # Optional: surface type classification
)
```

### Extent alignment

When multiple rasters (DSM, CDSM, DEM, etc.) cover different areas or have different resolutions, `prepare()` crops and resamples them to their **intersecting extent** — only the area covered by all inputs is used. No manual alignment is required.

### Cropping to a bounding box

To process a subset of the rasters, pass a `bbox`:

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    bbox=[476800, 4205850, 477200, 4206250],  # [minx, miny, maxx, maxy]
    pixel_size=1.0,  # Resample to 1 m resolution
)
```

Coordinates are in the DSM's native CRS (e.g. UTM metres). The `bbox` is intersected with the available data — pixels outside the raster extents are not extrapolated.

## Caching

The `working_dir` stores preprocessing results so that subsequent runs reuse them:

```text
cache/
├── walls/
│   └── px<size>/                # Keyed by pixel size, e.g. px1.000/
│       ├── wall_hts.tif         # Wall heights derived from DSM
│       └── wall_aspects.tif     # Wall compass directions
└── svf/
    └── px<size>/
        └── memmap/
            ├── svf.npy          # Total Sky View Factor
            ├── svf_north.npy    # Directional SVF (4 cardinal directions)
            └── ...              # 15 SVF grids total
```

### Force recomputation

To regenerate all cached data:

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    force_recompute=True,
)
```

SOLWEIG validates cached data against the current DSM — if the dimensions, extent, or pixel values change, the cache is invalidated and recomputed on the next `prepare()` call. `force_recompute=True` is only needed to bypass this check (e.g. after changing vegetation inputs that do not affect the DSM hash).

## Extracting Location from CRS

When the DSM has a projected CRS (e.g. UTM), latitude and longitude can be extracted:

```python
location = solweig.Location.from_surface(surface, utc_offset=2)
```

Alternatively, from an EPW file (which also includes UTC offset):

```python
location = solweig.Location.from_epw("data/weather.epw")
```

## Saving Results as GeoTIFFs

### During timeseries

Results are saved as they are computed:

```python
results = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
```

This produces files such as:

```text
output/
├── tmrt/
│   ├── tmrt_20250701_0000.tif
│   ├── tmrt_20250701_0100.tif
│   └── ...
└── shadow/
    ├── shadow_20250701_0000.tif
    └── ...
```

### Loading saved results

```python
arr, transform, crs, nodata = solweig.io.load_raster("output/tmrt/tmrt_20250701_1200.tif")
```

## NaN and Nodata Handling

Missing data is handled at each stage of the pipeline:

**At load time:** Negative nodata sentinel values (e.g. -9999) are converted to NaN. Zero-valued pixels are preserved as valid data.

**Before calculation:** NaN pixels in DSM, CDSM, and TDSM are filled with the ground reference (DEM if provided, otherwise the DSM itself). Pixels within 0.1 m of the ground reference are clamped to the ground value to prevent shadow artefacts from resampling noise.

DEM NaN pixels are not filled — they represent genuinely missing ground data.

When using `SurfaceData.prepare()`, this handling is performed during preparation. When constructing from arrays, `fill_nan()` runs inside `calculate()`.

## Rasterising Vector Data

Convert tree polygons (GeoDataFrame) to a raster grid:

```python
import geopandas as gpd

trees = gpd.read_file("trees.gpkg")
trees = trees.to_crs(2154)  # Match DSM CRS

cdsm, transform = solweig.io.rasterise_gdf(
    trees,
    geom_col="geometry",
    ht_col="height",
    bbox=[476800, 4205850, 477200, 4206250],
    pixel_size=1.0,
)

# Optionally save to disk
from pyproj import CRS
solweig.io.save_raster(
    "data/cdsm.tif", cdsm, transform.to_gdal(), CRS.from_epsg(2154).to_wkt()
)
```

## Large Rasters

For rasters exceeding available memory, SOLWEIG tiles the computation internally:

```python
results = solweig.calculate(
    surface=surface,
    weather=weather,
    location=location,
    output_dir="output/",
)
```

No explicit tiling configuration is required.

For large rasters, surface preparation also processes base layers one at a
time. Above roughly 50 megapixels, `SurfaceData.prepare()` loads, resamples,
and spills each input layer to a disk-backed file one layer at a time, rather
than holding the whole layer stack as resident RAM, with no change to the
prepared output.
