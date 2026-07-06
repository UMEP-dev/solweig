# Technical Implementation

Implementation details, performance considerations, and computational requirements.

Normative runtime/API behavior is specified in
[runtime-contract.md](runtime-contract.md). This file focuses on implementation
details rather than public precondition semantics.

## Data Types

### Float32 Precision

All raster calculations use **float32** (single precision) rather than float64:

- **Reason**: GPU compatibility, memory efficiency, sufficient precision
- **Precision**: ~7 significant digits
- **Range**: ±3.4 × 10³⁸

**Properties:**

1. All DSM/CDSM values stored as float32
2. All output rasters are float32
3. Intermediate calculations may use higher precision internally
4. Results should match float64 within 0.1% for typical urban values

### Integer Types

- Shadow masks: f32 (0.0/1.0) during GPU compute; bitpacked uint8 for SVF shadow matrices; quantized uint8 (0-255) for the shadow_to_u8 path
- Indices and counts: int32 or int64

## Tiling

Large rasters are processed in tiles to manage memory:

### Tile Properties

1. **Tile size**: Configurable, typically 256×256 to 2500×2500 pixels (2500 is the fallback when GPU and RAM detection both fail); dynamically sized from GPU/RAM limits via `compute_max_tile_side()`
2. **Overlap**: Tiles overlap by `max_shadow_reach` to avoid edge artifacts
3. **Seamless output**: Stitched results should be identical to full-raster processing

### Shadow Reach Calculation

Overlap must accommodate the longest possible shadow:

```text
max_shadow_reach = max_building_height / tan(min_sun_altitude)
```

At min_sun_altitude = 3° (`MIN_SUN_ELEVATION_DEG = 3.0`):
- 30m building → ~572m shadow → 572 pixels at 1m resolution

### Tile Processing Order

1. Tiles can be processed in parallel (independent)
2. Edge pixels use overlap region for context
3. Only interior pixels written to output

## GPU Acceleration

Optional GPU support for shadow and SVF calculations:

### GPU Properties

1. **Automatic fallback**: If GPU unavailable, uses CPU
2. **Near-equivalent results**: GPU and CPU agree to float32 precision for shadows and building/ground SVF; vegetation SVF (`svf_veg*`) can drift by up to 0.042 at canopy-edge pixels (<1% of pixels, <0.5 °C effect on Tmrt) — see VALIDATION.md
3. **Memory management**: Large rasters automatically tiled for GPU memory limits

### GPU-Accelerated Operations

- Shadow casting (ray marching)
- SVF patch visibility checks
- Anisotropic sky radiation (via `aniso_gpu.rs` + `anisotropic_sky.wgsl`)
- Shadow mask conversion (bitpacking via `shadow_to_bitpack.wgsl`, quantization via `shadow_to_u8.wgsl`)
- Parallel pixel operations

### CPU-Only Operations

- File I/O
- Coordinate transformations
- Final result assembly

## Coordinate Systems

### Raster Coordinates

- Row 0 = North edge of raster
- Column 0 = West edge of raster
- Increasing row index = moving South
- Increasing column index = moving East

### Sun Position

- Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
- Altitude: 0° = horizon, 90° = zenith

### Geographic Coordinates

- Input rasters should have valid CRS (coordinate reference system)
- WGS84 (EPSG:4326) used for sun position calculations
- Local projected CRS used for distance calculations

## Memory Management

### Typical Memory Usage

| Operation | Memory per megapixel |
| --------- | -------------------- |
| Single raster (float32) | ~4 MB |
| SVF calculation (CPU) | ~150 MB (multiple arrays + shadow matrices) |
| SVF calculation (GPU) | ~384 MB (GPU buffer allocations) |
| Full SOLWEIG run | ~400 MB |

### Memory Properties

1. Peak memory scales with tile size, not total raster size
2. Intermediate arrays released after use
3. Output written incrementally for large rasters

## Numerical Stability

### Edge Cases

1. **Sun at low altitude**: Shadow length grows rapidly as altitude decreases
   - Handled by clamping shadow reach to `max_shadow_distance_m` (Rust default 0.0 = no cap; Python tiling layer defaults to 1000m via `MAX_BUFFER_M`)
   - `min_sun_elev_deg` (default 3.0°) caps the maximum shadow reach calculation rather than preventing computation entirely
   - At altitude >= 89.5°, Rust returns all-sunlit (zenith case)

2. **Very tall buildings**: May exceed shadow reach
   - Warning if buildings exceed reasonable height

3. **Flat terrain**: Division by zero avoided
   - max_height = 0 handled gracefully

### NaN Handling

1. Input NaN values indicate no-data (outside study area)
2. NaN propagates through calculations
3. Output NaN indicates invalid/missing result

## Performance Targets

| Operation | Target | Notes |
| --------- | ------ | ----- |
| Shadow (1 timestep) | <1s per megapixel | GPU |
| SVF | <30s per megapixel | GPU |
| Full day (48 timesteps) | <5 min per megapixel | GPU |

## Reproducibility

### Deterministic Results

1. Same inputs → same outputs (bitwise identical)
2. No random number generation in core algorithms
3. Parallel processing order does not affect results

### Version Compatibility

1. Output format stable across minor versions
2. Algorithm changes documented in changelog
3. Regression tests verify consistency
