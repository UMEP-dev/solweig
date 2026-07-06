# Ground View Factor (GVF)

Computes ground- and wall-emitted radiation integrated across all visible surfaces from a point at person height. Despite the name, the primary outputs are thermal emission (Lup in W/m²) and albedo-weighted view factors, not a simple dimensionless view factor.

**Reference:** Lindberg et al. (2008) Section 2.3

## Algorithm

GVF is computed by integrating visible ground and wall surfaces across 18 azimuth directions (5° start, 20° step, covering 360°). For each azimuth, rays march outward from each pixel to detect buildings and walls, accumulating their thermal and reflective contributions.

### Integration Distances

Two distance scales are used, based on person height. They are metric
distances (Smidt et al. source-area model), converted to pixels by dividing
by the pixel size (matching UMEP's `round(metres × scale)` with
`scale = 1/pixel_size`); the far-field march is clamped to the raster extent:

```text
first  = round(height / pixel_size), min 1              (near-field, ~person height)
second = round(height × 20 / pixel_size), ≤ max(rows, cols)   (far-field, ~20× person height)
```

### Near/Far Weighting

The final GVF is a weighted blend of near-field and far-field results:

```text
gvf      = (gvf1 × 0.5 + gvf2 × 0.4) / 0.9
gvf_lup  = (gvfLup1 × 0.5 + gvfLup2 × 0.4) / 0.9 + lup_local × (1 - buildings)
gvf_alb  = (gvfalb1 × 0.5 + gvfalb2 × 0.4) / 0.9 + alb_grid × (1 - buildings) × shadow
```

The near-field captures local wall effects; the far-field captures distant ground/wall contributions. The `(1 - buildings)` term adds the local ground contribution for non-building pixels.

### Longwave Emission (Differential Formula)

The Lup contribution per azimuth uses a differential formula — computing excess emission above ambient:

```text
lup_diff = SBC × emis × (Tg × shadow + Ta + 273.15)⁴ - SBC × emis × (Ta + 273.15)⁴
```

The final `gvf_lup` adds the ambient baseline back:

```text
gvf_lup = weighted_sum(lup_diff) + SBC × emis × (Tg_local × shadow + Ta + 273.15)⁴
          - SBC × emis × (Ta + 273.15)⁴ + that last term for non-building pixels
```

For walls, a similar differential:

```text
lwall = SBC × ewall × (tgwall + Ta + 273.15)⁴ - SBC × ewall × (Ta + 273.15)⁴
```

Where `tgwall` is the wall temperature deviation from air temperature (K), only applied to sunlit walls.

### Water Temperature Override

When land cover data is active (`landcover=true`), water pixels (lc_grid == 3) have their ground temperature overridden:

```text
Tg_water = Twater - Ta    (Twater from weather file)
```

This bypasses the land cover parameter table (TgK/TmaxLST) for water surfaces.

### Sunwall Mask

A sunwall mask identifies building pixels adjacent to sunlit walls:

```text
sunwall = 1.0 if wallsun > tolerance AND walls > 0
```

This determines whether wall-temperature terms (heated walls) or ambient-temperature terms are used for nearby building faces.

### Geometry Caching

The ray-tracing geometry (building blocking distances, wall influence masks) can be precomputed once and reused across timesteps. The cached path (`sun_on_surface_cached`) skips the expensive ray-marching and only recomputes the thermal-dependent outputs (`gvf_lup`, `gvfalb`). The purely geometric output (`gvfalbnosh`) is taken directly from the cache.

## Inputs

| Input | Type | Description |
| --- | --- | --- |
| wallsun | 2D array (f32) | Sunlit wall height from shadow calculation |
| walls | 2D array (f32) | Wall height grid |
| buildings | 2D array (f32) | Building mask (1.0 = building, 0.0 = ground) |
| shadow | 2D array (f32) | Combined shadow mask (0.0-1.0) |
| dirwalls | 2D array (f32) | Wall aspect (orientation) in degrees |
| tg | 2D array (f32) | Ground temperature deviation from air temp (°C) |
| emis_grid | 2D array (f32) | Ground surface emissivity per pixel |
| alb_grid | 2D array (f32) | Ground surface albedo per pixel |
| lc_grid | 2D array (f32) | Land cover classification grid (optional) |
| scale | float | Pixel size in meters |
| first | float | Person height parameter (m), typically 1.1 |
| second | float | Far-field height parameter (m), typically 22.0 |
| tgwall | float | Wall temperature deviation from air temp (K) |
| ta | float | Air temperature (°C) |
| ewall | float | Wall emissivity |
| sbc | float | Stefan-Boltzmann constant (5.67051e-8) |
| albedo_b | float | Building wall albedo |
| twater | float | Water temperature from weather file (°C) |
| landcover | bool | Whether to use land cover classification |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| gvf_lup | 2D array (W/m²) | Longwave upwelling from ground/walls (center) |
| gvfalb | 2D array | Albedo-weighted ground view (shadow-dependent, center) |
| gvfalbnosh | 2D array | Albedo-weighted ground view (no shadow, geometric only, center) |
| gvf_lup_e | 2D array (W/m²) | Longwave upwelling, east-facing contribution |
| gvfalb_e | 2D array | Albedo-weighted view, east |
| gvfalbnosh_e | 2D array | Albedo view (no shadow), east |
| gvf_lup_s | 2D array (W/m²) | Longwave upwelling, south-facing contribution |
| gvfalb_s | 2D array | Albedo-weighted view, south |
| gvfalbnosh_s | 2D array | Albedo view (no shadow), south |
| gvf_lup_w | 2D array (W/m²) | Longwave upwelling, west-facing contribution |
| gvfalb_w | 2D array | Albedo-weighted view, west |
| gvfalbnosh_w | 2D array | Albedo view (no shadow), west |
| gvf_lup_n | 2D array (W/m²) | Longwave upwelling, north-facing contribution |
| gvfalb_n | 2D array | Albedo-weighted view, north |
| gvfalbnosh_n | 2D array | Albedo view (no shadow), north |
| gvf_sum | 2D array | Geometry normalization sum (used internally) |
| gvf_norm | 2D array | Geometry normalization factor (used internally) |

Note: `gvf_lup` has units of W/m² (contains emission calculation), not dimensionless. `gvfalb` already contains albedo weighting and is used directly by the radiation budget — it is not a pure view factor.

## Properties

### Output Properties

1. **gvf_lup contains thermal emission**
   - Not a dimensionless factor — units are W/m²
   - Includes both ground and wall longwave contributions
   - Higher in urban canyons (more wall emission)

2. **gvfalb includes albedo**
   - Already albedo-weighted; no further multiplication by albedo needed
   - Shadow-dependent: higher in sunlit areas (more reflected shortwave)

3. **gvfalbnosh is purely geometric**
   - Independent of shadow state and weather
   - Can be cached across timesteps (used by geometry cache)

### Geometric Properties

1. **Flat open terrain**
   - gvf_lup ≈ local ground emission only
   - No wall contributions

2. **Urban canyon has high wall contribution**
   - Walls on both sides increase gvf_lup
   - More reflected and emitted radiation in canyons

3. **Directional asymmetry**
   - East-facing wall contributes to western GVF (seen from west)
   - Asymmetric building layout → asymmetric directional outputs

## Role in Radiation Budget

GVF outputs feed directly into the radiation computation:

**Reflected Shortwave (Kup)**: Uses `gvfalb` and `gvfalbnosh` directly (albedo already baked in):

```text
kup = gvfalb × rad_i × sin(altitude) + ct × gvfalbnosh
```

**Longwave Upwelling (Lup)**: Uses `gvf_lup` after thermal delay smoothing:

```text
Lup = TsWaveDelay(gvf_lup, ...)    [smoothed across timesteps]
```

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**GVF Model:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
