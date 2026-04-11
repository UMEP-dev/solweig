# Shadow Calculation

Calculates where shadows fall based on sun position, buildings, and vegetation.

**Reference:** Lindberg et al. (2008) Section 2.2 - Shadow casting algorithm

## Equations

### Shadow Length

```text
L = h / tan(α)
```

- L = shadow length (meters)
- h = obstacle height above ground (meters)
- α = sun altitude angle (degrees)

### Ray Marching

The algorithm traces rays from each ground pixel toward the sun:

```text
dx = -sign(cos(θ)) × step / tan(θ)    # When E-W dominant
dy = sign(sin(θ)) × step               # When E-W dominant
dz = ds × step × tan(α) × scale        # Height gain per step (metres)
```

- θ = sun azimuth (radians)
- ds = path length correction for diagonal movement
- scale = pixel size in metres (solweig convention; note: upstream UMEP uses `1/pixel_size`, so the corresponding UMEP formula divides by `scale`)

### Shadow Condition

A pixel is sunlit if no obstacle along the ray to the sun is tall enough:

```text
sunlit[y,x] = 1  if  propagated_height <= DSM[y,x]
            = 0  otherwise
```

## Inputs

| Input | Type | Description |
| --- | --- | --- |
| DSM | 2D array (m) | Digital Surface Model - elevation including buildings |
| sun_altitude | float (0-90°) | Sun elevation above horizon |
| sun_azimuth | float (0-360°) | Sun direction (0=N, 90=E, 180=S, 270=W) |
| pixel_size | float (m) | Resolution of DSM |
| CDSM | 2D array (m) | Canopy DSM for vegetation shadows (optional) |
| TDSM | 2D array (m) | Trunk DSM - height of trunk zone below canopy (optional) |
| bush | 2D array (m) | Bush/low vegetation DSM (optional) |
| walls | 2D array (m) | Wall height grid (optional, enables wall shading outputs) |
| wall_aspect | 2D array (rad) | Wall face orientation in radians (optional) |
| walls_scheme | 2D array | Wall height scheme for shadow propagation (optional) |
| aspect_scheme | 2D array | Aspect scheme paired with walls_scheme (optional) |
| max_local_dsm_ht | float (m) | Maximum local DSM height, used to limit ray march steps |
| min_sun_elev_deg | float (°) | Minimum sun elevation for shadow reach limiting. Default 3.0 |
| max_shadow_distance_m | float (m) | Maximum shadow casting distance. Rust default 0.0 (no cap); Python tiling caps at 1000.0 |

## Outputs

| Output             | Type           | Description                                                                                                          |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| bldg_sh            | 2D array (f32) | Building shadow mask (1.0=sunlit, 0.0=shadow)                                                                        |
| veg_sh             | 2D array (f32) | Vegetation shadow mask (1.0=veg shadow hit, 0.0=no veg shadow). Binary; transmissivity applied externally in Python. |
| veg_blocks_bldg_sh | 2D array (f32) | Vegetation shadow where it overlaps building shadow (for SVF)                                                        |
| wall_sh            | 2D array (f32) | Shadow height on walls from buildings (optional)                                                                     |
| wall_sun           | 2D array (f32) | Sunlit wall height (optional)                                                                                        |
| wall_sh_veg        | 2D array (f32) | Shadow height on walls from vegetation (optional)                                                                    |
| face_sh            | 2D array (f32) | Wall faces in shadow based on orientation vs sun azimuth (optional)                                                  |
| face_sun           | 2D array (f32) | Wall faces in sun based on orientation vs sun azimuth (optional)                                                     |
| sh_on_wall         | 2D array (f32) | Combined shadow-on-wall indicator (optional)                                                                         |

## Properties

### Critical Properties

1. **Sun altitude edge cases**
   - At altitude >= 89.5°: all pixels are sunlit (zenith case, avoids tan(90°)=infinity)
   - Neither the Rust nor upstream UMEP Python have an explicit altitude <= 0° guard. The `min_sun_elev_deg` parameter (default 3.0°) caps maximum shadow reach via `max_shadow_distance_m`, providing implicit protection at low sun angles.

2. **Flat terrain = no shadows**
   - When: DSM is uniform (no elevation differences)
   - Then: sunlit mask is all ones
   - Reason: No obstacles to cast shadows

3. **Binary shadow values**
   - Building shadows are discrete: 0.0 or 1.0
   - No partial shadows (penumbra) in building shadow model
   - Vegetation shadows are also binary in Rust (0.0 or 1.0); transmissivity is applied externally in Python

### Geometric Properties

1. **Shadows opposite sun direction**
   - Sun from south (180°) → shadows extend north (toward row 0)
   - Sun from east (90°) → shadows extend west (toward col 0)

2. **Lower sun = longer shadows**
   - As altitude decreases, shadow area increases
   - At 45°: shadow length = obstacle height
   - At 30°: shadow length ≈ 1.73 × height
   - At 15°: shadow length ≈ 3.73 × height

3. **Taller obstacles = longer shadows**
   - Shadow length proportional to height: L ∝ h

4. **Shadow length follows equation**
   - Measured shadow length ≈ h / tan(α) within ±15%
   - Tolerance accounts for pixel discretization

### Rooftop Properties

1. **Building tops are sunlit**
   - Rooftops (local maxima) receive direct sun when altitude > 0
   - Unless shaded by taller neighboring buildings

## Vegetation Shadows

Vegetation shadows differ from building shadows due to partial light transmission through foliage.

**Primary References:**

- Konarska J, Lindberg F, Larsson A, Thorsson S, Holmer B (2014) "Transmissivity of solar radiation through crowns of single urban trees—application for outdoor thermal comfort modelling." Theoretical and Applied Climatology 117:363-376.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas." Theoretical and Applied Climatology 105:311-323.

### Canopy Transmissivity

**Reference:** Konarska et al. (2014)

Light transmission through tree canopies varies with species, leaf area index (LAI), and season:

| Tree Type                | Transmissivity | LAI | Description               |
| ------------------------ | -------------- | --- | ------------------------- |
| Dense deciduous (summer) | 0.02-0.05      | 5-7 | Oak, maple in full leaf   |
| Medium deciduous         | 0.05-0.15      | 3-5 | Typical urban trees       |
| Open canopy              | 0.15-0.30      | 2-3 | Young trees, sparse crown |
| Conifers                 | 0.10-0.20      | 4-6 | Year-round                |
| Deciduous (winter)       | 0.60-0.80      | 0-1 | Bare branches only        |

**SOLWEIG defaults:**

- Leaf-on transmissivity: 0.03 (3%) - represents dense summer canopy
- Leaf-off transmissivity: 0.5 (50%) - bare branches in winter (deciduous only)
- Default leaf-on period: day 100 (~Apr 10) to day 300 (~Oct 27)
- Conifers: always use leaf-on transmissivity

### Vegetation Shadow Algorithm (Pergola Heuristic)

The Rust implementation uses a "pergola" heuristic rather than a simple trunk/canopy binary model. For each ray marching step, four conditions are evaluated at the current and previous step positions:

```text
For each ray step:
  cond1 = shifted_veg_canopy > target_dsm     (current step canopy above target)
  cond2 = shifted_veg_trunk > target_dsm      (current step trunk above target)
  cond3 = prev_step_veg_canopy > target_dsm   (previous step canopy above target)
  cond4 = prev_step_veg_trunk > target_dsm    (previous step trunk above target)
  sum = cond1 + cond2 + cond3 + cond4

  pergola_shadow = 1.0 if 0 < sum < 4, else 0.0
```

The logic:

- **sum = 0**: Ray misses vegetation entirely → no vegetation shadow
- **sum = 1-3**: Ray hits vegetation edge (canopy boundary) → vegetation shadow
- **sum = 4**: Ray passes entirely within the canopy layer (above trunk, below canopy top at both steps) → pergola effect, no shadow (light passes through the open interior)

The result is accumulated with `max()` across ray steps, then cleared where building shadow already exists (to avoid double-counting).

### Combined Shadow (Python)

Transmissivity is applied externally in Python (`components/shadows.py`), not inside the Rust shadow kernel:

```text
shadow = bldg_sh - (1 - veg_sh) × (1 - transmissivity)
```

Where `bldg_sh` is 1.0 (sunlit) or 0.0 (shadow), `veg_sh` is 1.0 (veg shadow hit) or 0.0, and `transmissivity` is the fraction of light passing through foliage (default 0.03 leaf-on, 0.5 leaf-off).

### Trunk Zone Ratio

**Reference:** Lindberg & Grimmond (2011)

The trunk zone is the lower portion of the tree where only the solid trunk exists (no foliage). The TDSM (Trunk DSM) provides the explicit height of trunk tops. If not provided, a default trunk ratio is applied.

**SOLWEIG default:** trunk_ratio = 0.25 (25%)

Typical values by tree type:

| Tree Form               | Trunk Ratio | Example Species      |
| ----------------------- | ----------- | -------------------- |
| Street tree (pollarded) | 0.30-0.40   | Plane tree, linden   |
| Natural form            | 0.20-0.30   | Oak, beech           |
| Conifer                 | 0.10-0.20   | Pine, spruce         |
| Low-branching           | 0.05-0.15   | Magnolia, ornamental |

## Wall Shading

The `shade_on_walls` function (`rust/src/shadowing.rs`) computes shadow patterns on vertical wall surfaces. It determines which wall faces are sunlit vs shaded based on their orientation relative to the sun azimuth, and how much of the wall height is in shadow from nearby buildings and vegetation.

### Wall Face Orientation

A wall face is considered sun-facing if its aspect (outward normal) is within ±90° of the sun azimuth:

```text
face_sun = 1.0 if wall_aspect is within [azimuth - π/2, azimuth + π/2]
face_sh  = 1.0 - face_sun
```

Wrapping is handled for azimuth ranges that cross 0°/360°.

### Shadow Height on Walls

The propagated building shadow height and vegetation shadow height are compared against wall height:

```text
wall_sh = min(propagated_bldg_sh_height, wall_height)    [building shadow on wall]
wall_sh_veg = min(propagated_veg_sh_height, wall_height)  [vegetation shadow on wall]
wall_sun = wall_height - wall_sh                           [sunlit portion]
```

### Combined Wall Shadow

```text
sh_on_wall = face_sh × wall_mask + wall_sh × face_sun × wall_mask
```

This combines orientation-based shading (self-shading of walls facing away from sun) with cast shadows from nearby buildings.

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**Shadow Algorithm:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.

**Vegetation Shadows:**

- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas: model development and evaluation." Theoretical and Applied Climatology 105, 311-323.
- Konarska J, Lindberg F, Larsson A, Thorsson S, Holmer B (2014) "Transmissivity of solar radiation through crowns of single urban trees—application for outdoor thermal comfort modelling." Theoretical and Applied Climatology 117, 363-376.
