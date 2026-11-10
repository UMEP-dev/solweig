# Sky View Factor (SVF)

Fraction of the sky hemisphere visible from each point. Determines how much diffuse sky radiation and longwave sky emission reaches the surface.

**Reference:** Lindberg et al. (2008) Section 2.1, Lindberg & Grimmond (2011)

## Equation

SVF is the ratio of radiation received from the sky to that from an unobstructed hemisphere:

```text
SVF = Ω_sky / 2π
```

Where Ω_sky is the solid angle of visible sky (steradians).

## Patch-Based Calculation Method

**Reference:** Robinson & Stone (1990) "Solar Radiation Modelling in the Urban Context", Building and Environment 25(3):201-209.

SOLWEIG uses the patch-based method where the sky hemisphere is divided into discrete angular patches (annuli). SVF is computed by testing visibility to each patch and weighting by the patch's solid angle.

### Patch Configuration

The sky is divided into concentric annuli (altitude bands) from 0° to 90° elevation. Each annulus is further subdivided into azimuthal patches. Standard configurations:

- **Option 1** (145 patches): Coarse for fast computation
- **Option 2** (153 patches - default): Balance of accuracy and speed
- **Option 3** (305 patches): Fine for high accuracy
- **Option 4** (609 patches): Research-grade resolution

For option 2 (153 patches):

- 8 altitude bands: 6°, 18°, 30°, 42°, 54°, 66°, 78°, 90°
- Azimuthal divisions per band: 31, 30, 28, 24, 19, 13, 7, 1

### Solid Angle Weight Calculation

Each patch's contribution to SVF is weighted by its solid angle. The discretized computation in `skyview.rs`:

```text
n = 90
common_w_factor = (1 / (2π)) × sin(π / (2n))
steprad_iso = (360 / azimuth_patches) × (π / 180)
steprad_aniso = (360 / azimuth_patches_aniso) × (π / 180)

sin_term_sum = Σ sin(π(2a - 1) / (2n))    for a in annulino_start..=annulino_end

weight_iso = steprad_iso × common_w_factor × sin_term_sum
weight_aniso = steprad_aniso × common_w_factor × sin_term_sum
```

Where:

- `azimuth_patches` = number of azimuthal divisions in this altitude band (isotropic)
- `azimuth_patches_aniso` = `ceil(azimuth_patches / 2)` (directional, covering half-hemisphere)
- `annulino_start..annulino_end` = range of annulus indices within the altitude band

The isotropic weight is used for total SVF. The anisotropic weight is used for directional SVFs (N, E, S, W), with each directional component only accumulating patches whose azimuth falls within its 180° half-hemisphere.

### SVF Accumulation Formula

```text
SVF = Σ_patches (weight_iso × visibility_patch)
```

Where:

- weight_iso = pre-computed isotropic solid angle weight for the patch
- visibility_patch = 1 if patch center is unobstructed, 0 if blocked by DSM

### Last Annulus Correction

A correction factor `LAST_ANNULUS_CORRECTION = 3.0459e-4` is added during finalization to `svf_south` and `svf_west` only (and their vegetation variants, conditioned on `vegdem2 == 0`). It is **not** applied to the isotropic `svf`, `svf_north`, or `svf_east`.

The reason for this asymmetry: the zenith patch has a single azimuthal subdivision at azimuth = 0°. The directional bucketing rules place azimuth 0° in the north (270°–90°) and east (0°–180°) half-hemispheres, but not in south (90°–270°) or west (180°–360°). The correction adds back the missing zenith contribution to the south and west directional SVFs. The isotropic SVF accumulates the zenith patch regardless of azimuth, so no correction is needed there.

### Directional SVF

Directional components split patches by azimuth half-hemisphere:

```text
SVF_north = Σ (weight_aniso × visibility) for azimuth ∈ [270°, 360°) ∪ [0°, 90°)
SVF_east  = Σ (weight_aniso × visibility) for azimuth ∈ [0°, 180°)
SVF_south = Σ (weight_aniso × visibility) for azimuth ∈ [90°, 270°)
SVF_west  = Σ (weight_aniso × visibility) for azimuth ∈ [180°, 360°)
```

## Inputs

| Input | Type | Description |
| --- | --- | --- |
| DSM | 2D array (m) | Digital Surface Model |
| CDSM | 2D array (m) | Canopy DSM for vegetation (optional) |
| TDSM | 2D array (m) | Trunk DSM for vegetation trunk zone (optional) |
| pixel_size | float (m) | Resolution |
| usevegdem | bool | Whether to account for vegetation in SVF |
| max_local_dsm_ht | float (m) | Maximum local DSM height, limits ray march steps |
| patch_option | int (1-4) | Patch resolution option. Default 2 (153 patches) |
| min_sun_elev_deg | float (°) | Minimum elevation for shadow rays. Default 3.0 |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| svf | 2D array (0-1) | Overall sky view factor |
| svf_north | 2D array (0-1) | SVF from northern half-hemisphere |
| svf_east | 2D array (0-1) | SVF from eastern half-hemisphere |
| svf_south | 2D array (0-1) | SVF from southern half-hemisphere |
| svf_west | 2D array (0-1) | SVF from western half-hemisphere |
| svf_veg | 2D array (0-1) | SVF accounting for vegetation canopy |
| svf_veg_north | 2D array (0-1) | Vegetation SVF from northern half-hemisphere |
| svf_veg_east | 2D array (0-1) | Vegetation SVF from eastern half-hemisphere |
| svf_veg_south | 2D array (0-1) | Vegetation SVF from southern half-hemisphere |
| svf_veg_west | 2D array (0-1) | Vegetation SVF from western half-hemisphere |
| svf_veg_blocks_bldg_sh | 2D array (0-1) | SVF where vegetation overlaps building shadow |
| svf_veg_blocks_bldg_sh_north | 2D array (0-1) | Directional variant (north) |
| svf_veg_blocks_bldg_sh_east | 2D array (0-1) | Directional variant (east) |
| svf_veg_blocks_bldg_sh_south | 2D array (0-1) | Directional variant (south) |
| svf_veg_blocks_bldg_sh_west | 2D array (0-1) | Directional variant (west) |
| bldg_sh_matrix | 3D array (uint8) | Bitpacked building shadow per patch |
| veg_sh_matrix | 3D array (uint8) | Bitpacked vegetation shadow per patch |
| veg_blocks_bldg_sh_matrix | 3D array (uint8) | Bitpacked veg-blocks-bldg shadow per patch |

The shadow matrices (3D, uint8 bitpacked) store per-patch shadow results for use by anisotropic sky radiation. The third dimension indexes over patches.

A set bit means the patch is visible through that layer. The anisotropic sky assigns each patch to one emitter from the three bits: sky when the building and vegetation bits are both set, vegetation when either vegetation bit is clear, and buildings when the building bit is clear and the veg-blocks-bldg bit is set.

Without a canopy layer both vegetation matrices have every bit set ("no blocking"), matching the wall-scheme branch of umep-solweig's `svfForProcessing153`. The per-timestep calculation applies the same convention itself whenever the surface has no CDSM, so matrices from older caches or from UMEP exports are not relied on in that case.

## Properties

### Range Properties

1. **SVF in range [0, 1]**
   - SVF = 0: no sky visible (e.g., inside building)
   - SVF = 1: full hemisphere visible (open field)
   - All intermediate values valid

2. **Directional SVF in range [0, 1]**
   - Each directional component (N, E, S, W) also bounded by [0, 1]

### Geometric Properties

3. **Flat open terrain = SVF of 1**
   - No obstructions → full sky visibility
   - Tolerance: SVF > 0.95 for truly flat DSM

4. **Deep canyon has low SVF**
   - Urban canyon with H/W ratio > 2 → SVF < 0.5
   - H = building height, W = street width

5. **Taller obstacles reduce SVF**
   - Higher buildings nearby → lower ground-level SVF
   - SVF decreases monotonically with obstacle height

6. **Rooftops have high SVF**
   - Building tops (local maxima) have SVF close to 1
   - Only reduced if taller buildings nearby

7. **Building density reduces SVF**
   - More buildings → lower ground-level SVF
   - SVF is a measure of urban density/openness

### Symmetry Properties

8. **Symmetric obstacles give symmetric directional SVF**
   - Square courtyard center has equal N/E/S/W SVF
   - Asymmetric buildings create asymmetric directional SVF

## Directional SVF

Directional components split the sky into half-hemispheres:

```text
         N (svf_north)
         |
    W ---+--- E
         |
         S
```

Used for calculating radiation from different sky directions, important for:

- Anisotropic sky radiance (brighter near sun)
- Wall orientation effects
- Asymmetric shading

## Vegetation Effects

Trees reduce SVF but not completely (light passes through canopy):

- **svf_veg**: Sky view through vegetation canopy
- Accounts for the pergola shadow heuristic (see shadows.md)
- svf_veg >= svf (vegetation blocks less than buildings)

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**SVF Algorithm:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas: model development and evaluation." Theoretical and Applied Climatology 105, 311-323.

**Patch-Based Method:**

- Robinson D, Stone A (1990) "Solar Radiation Modelling in the Urban Context." Building and Environment 25(3), 201-209.
