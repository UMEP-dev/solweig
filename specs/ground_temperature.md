# Ground Temperature Model

Surface temperature parameterization for ground longwave emission calculations.

**Primary References:**

- Lindberg F, Onomura S, Grimmond CSB (2016) "Influence of ground surface characteristics on the mean radiant temperature in urban areas." International Journal of Biometeorology 60(9):1439-1452.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas." Theoretical and Applied Climatology 105:311-323.

## Overview

Ground surface temperature directly affects upwelling longwave radiation (Lup), which contributes significantly to mean radiant temperature in urban environments. The model accounts for:

1. **Solar heating** - Direct and diffuse radiation absorption
2. **Thermal inertia** - Delayed response due to material heat capacity
3. **Surface properties** - Albedo, emissivity, thermal conductivity

## TsWaveDelay Model

The thermal delay model simulates ground temperature response to changing radiation conditions using an exponential decay function.

### Equation

```text
T_ground(t) = T_current × (1 - w) + T_previous × w

where:
    w = exp(-33.27 × Δt)
    Δt = time since last update (fraction of day)
```

### Parameters

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| Decay constant | 33.27 | Thermal response rate (day⁻¹) |
| Time threshold | 59/1440 | Minimum time step (~59 minutes) |

### Physical Interpretation

The decay constant (33.27 day⁻¹) corresponds to a thermal time constant of approximately:

```text
τ = 1 / 33.27 ≈ 0.030 days ≈ 43 minutes
```

This represents the characteristic time for surface temperature to respond to changes in radiative forcing. After one time constant:

- 63% of adjustment to new equilibrium
- After 3τ (~2 hours): 95% adjustment

### Algorithm

In Rust: `ts_wave_delay(gvf_lup, firstdaytime, timeadd, timestepdec, tgmap1)` and `ts_wave_delay_batch_pure` for batched 6-in-1 processing (center + 4 directional + ground).

```python
def TsWaveDelay(T_current, firstdaytime, time_accumulated, timestep, T_previous):
    """
    Apply thermal delay to ground temperature.

    Args:
        T_current: Current radiative equilibrium temperature
        firstdaytime: True if first timestep after sunrise
        time_accumulated: Time since last full update (fraction of day)
        timestep: Current timestep duration (fraction of day)
        T_previous: Previous delayed temperature

    Returns:
        T_delayed: Temperature with thermal inertia applied
        time_accumulated: Updated time accumulator
        T_previous: Updated previous temperature for next iteration
    """
    if firstdaytime:
        T_previous = T_current

    if time_accumulated >= 59/1440:  # ~59 minutes threshold
        weight = exp(-33.27 * time_accumulated)
        T_previous = T_current * (1 - weight) + T_previous * weight
        T_delayed = T_previous
        time_accumulated = timestep if timestep > 59/1440 else 0
    else:
        time_accumulated += timestep
        weight = exp(-33.27 * time_accumulated)
        T_delayed = T_current * (1 - weight) + T_previous * weight

    return T_delayed, time_accumulated, T_previous
```

## Surface Temperature Parameterization

For computing the instantaneous radiative equilibrium temperature, SOLWEIG uses a linear parameterization based on solar altitude.

### Sinusoidal Diurnal Model

The ground temperature deviation from air temperature follows a sinusoidal diurnal phase (`rust/src/ground.rs`):

```text
Tgamp = TgK × altmax + Tstart

if dectime > sunrise_frac:
    phase = (dectime - sunrise_frac) / (TmaxLST_frac - sunrise_frac)
    Tg = Tgamp × sin(phase × π/2)
else:
    Tg = 0    (pre-sunrise: no deviation from air temp)
```

Where:

- `Tgamp` = maximum temperature amplitude (°C above air temp)
- `TgK` = temperature increase rate (°C per degree of max solar altitude)
- `altmax` = maximum solar altitude during the day (°)
- `Tstart` = temperature offset at sunrise (°C)
- `dectime` = current time as fraction of day
- `sunrise_frac` = sunrise time as fraction of day
- `TmaxLST_frac` = time of maximum surface temperature as fraction of day

### Clearness Index Correction

After computing the sinusoidal Tg, a clearness index correction is applied to account for non-clear sky conditions:

```text
corr = 0.1473 × ln(90 - zenith_deg) + 0.3454
CI_TgG = (radG / radG0) + (1 - corr)
CI_TgG = min(CI_TgG, 1.0)
Tg = max(Tg × CI_TgG, 0.0)
```

Where `radG` is measured global radiation and `radG0` is theoretical clear-sky radiation. Under clear skies CI_TgG ≈ 1.0; under overcast conditions CI_TgG < 1.0, reducing the ground temperature response.

### Land Cover Parameters

| Surface Type | Tstart (°C) | k (°C/°) | TmaxLST | Source |
| ------------ | ----------- | -------- | ------- | ------ |
| Cobblestone | -3.41 | 0.37 | 15:00 | Lindberg et al. (2016) |
| Dark asphalt | -9.78 | 0.58 | 15:00 | Lindberg et al. (2016) |
| Grass | -3.38 | 0.21 | 14:00 | Lindberg et al. (2016) |
| Bare soil | -3.01 | 0.33 | 14:00 | Lindberg et al. (2008; 2016) |
| Water | 0.0 | 0.00 | 12:00 | Lindberg et al. (2008; 2016) |

Note: Tstart is the temperature offset from air temperature at sunrise. Negative values indicate surfaces cooler than air at dawn.

**Water temperature override (preserved UMEP quirk, inactive):** The Rust override sets ground temperature to `Twater - Ta` (from the weather file) for pixels with `lc_grid == 3`. This reproduces UMEP's `sunonsurface_2018a`, which also checks code 3. In this implementation and the shipped materials table, water is land-cover code **7**, so the override never fires for real water pixels. Water instead follows the parameter table above; with TgK = 0.00 there is no diurnal amplitude, making TmaxLST irrelevant. The code-3 check is knowingly preserved for UMEP parity rather than corrected.

## Properties

### Thermal Inertia Effects

1. **Morning lag** - Surfaces warm slower than instantaneous equilibrium
2. **Afternoon persistence** - Surfaces remain warm after solar maximum
3. **Evening cooling** - Gradual temperature decrease after sunset

### Material Dependence

4. **High thermal mass** (concrete, stone): Slower response, τ > 1 hour
5. **Low thermal mass** (thin asphalt): Faster response, τ < 30 minutes
6. **Vegetation**: Complex due to evapotranspiration

### Diurnal Pattern

```text
Morning:  T_ground < T_equilibrium (heating lag)
Midday:   T_ground ≈ T_equilibrium (near steady state)
Afternoon: T_ground > T_equilibrium (cooling lag)
Night:    T_ground slowly approaches T_air
```

## Implementation Notes

### State Management

The thermal delay model requires state to be carried between timesteps:

- 6 directional `tgmap1` arrays (center, E, S, W, N, ground)
- `tgout1` — ground temperature output history
- `firstdaytime` flag — reset on first timestep after sunrise
- `timeadd` accumulator — tracks time since last full update
- `timestep_dec` — current timestep as fraction of day

For accurate results, use `calculate()` with a timeseries of weather data, which automatically manages thermal state. Single-timestep calculations will not capture thermal inertia effects.

### Directional Components

Ground temperature affects directional Lup components (Lup_E, Lup_S, Lup_W, Lup_N) which are computed using Ground View Factors in each direction. The `ts_wave_delay_batch_pure` function processes all 6 directional channels in a single call.

### Nighttime Behavior

Pre-sunrise (dectime <= sunrise_frac):

- Ground temperature deviation Tg = 0 (no deviation from air temperature)
- The TsWaveDelay model handles smooth transitions via thermal inertia
- Emissivity assumed constant (typically 0.95)

## UMEP 2026a Ground-Surface Scheme (opt-in)

An alternative ground-surface formulation from UMEP-processing's 2026a
generation (Bridoux, University of Gothenburg; merged upstream 2026-05) is
available behind two flags, both default **off**:

- `use_ground_scheme` — replace the sinusoidal Tg parameterization with a
  prognostic force-restore surface temperature model.
- `use_outgoing_longwave` — replace the GVF-based Lup and the TsWaveDelay
  step with a solid-angle view-factor march.

The current port supports the two flags only together. With both off, the
baseline model above runs unchanged (golden output byte-identical).

### Force-restore surface temperature

Each ground pixel carries a prognostic surface temperature `Tg` (°C,
absolute — not a deviation from air temperature), a deep-soil temperature
`Tm`, and the flux history (`Rn`, `Rn_past`, `G`). Per timestep (length
`Δt` in seconds):

```text
dTg/dt = 2·G / (C·d) − ω·(Tg − Tm),     ω = 2π / 86400 s⁻¹
d = sqrt(2·κ / ω)                        (damping depth)
```

where `C` is the volumetric heat capacity (J m⁻³ K⁻¹) and `κ` the thermal
diffusivity (m² s⁻¹) of the land-cover class. The ground heat flux `G`
comes from the Objective Hysteresis Model (Grimmond et al. 1991):

```text
Rn = (1 − α)·Kdown + Ldown − Lup,   Lup = ε·σ·(Tg + 273.15)⁴ + (1 − ε)·Ldown
G  = a1·Rn + a2·(Rn − Rn_past) + a3
```

with per-class OHM coefficients `a1..a3` (`a1` seasonally modulated by a
latitude-signed sinusoid). Integration is a 2nd-order Runge-Kutta step;
at shadow transitions (|shadow − shadow_past| > 0.5) the change in `G` is
clamped to `|a1·ΔRn|` to damp flux spikes. Water pixels (class 7) replace
the OHM step with a 1 m slab energy balance including a latent heat term.
The scheme runs day and night (no night zeroing of Tg).

Initial state comes from `initiate_ground_scheme`: per-class seasonal
sinusoids of the first day's air temperature series set `Tg`/`Tm`, and the
parameter grids (`C`, `κ`, `a1..a3`) are built from the materials JSON
(`Heat capacity`, `Thermal_diffusivity`, `OHM_coefficients`, `Tg_ini`/
`Tm_ini` coefficients). Ground classes 0/1/2/5/6/7 are supported; wall
material codes (≥ 100) are remapped to roofs.

### Solid-angle outgoing longwave

With `use_outgoing_longwave`, Lup, the ground albedo view factors, and the
directional ground/wall side longwave come from a 20-azimuth translated-
raster march out to ~11 m (99 % of the Lambert view factor at a receiver
height of 1.1 m), replacing the GVF step. The TsWaveDelay step is **not**
applied to Lup — thermal inertia lives in the force-restore ODE. Kup is
computed from the march's sunlit/total albedo view factors, and the
directional side longwave (`gvfLside*`) replaces the Lup-derived ground
term in Lside: the pipeline switches to the `Lside_veg_v2026` variant
(reflection term drops Lup; the anisotropic branch contributes zero, with
directional longwave supplied by the march alone). Wall temperature keeps
the classic sinusoidal wall model.

Ordering per timestep (matching upstream `Solweig_2026a_calc`): shadows →
Kdown + isotropic Ldown → force-restore Tg step → outgoing-longwave march
→ Kup → Lside → Tmrt. In anisotropic mode the Tmrt cylinder longwave term
is composed as mean(directional Lside) + anisotropic-sky Lside, following
the 2026a reference.

### Constraints and status

- Requires a land-cover grid; tiled processing is currently rejected.
- Implemented in `rust/src/ground_surface.rs` and wired into the fused
  pipeline (`rust/src/pipeline.rs`) via `GroundSchemeBundle`; state is
  carried by `solweig.components.ground_scheme.GroundSchemeState`.
- Component parity against the vendored upstream reference is gated by
  `tests/spec/test_parity_2026a.py`; the end-to-end path is pinned by
  `tests/golden/test_golden_ground_scheme.py`.
- Not yet validated against field measurements; defaults stay off until a
  VALIDATION.md comparison exists.

**Reference:** Grimmond CSB, Cleugh HA, Oke TR (1991) "An objective urban
heat storage model and its comparison with other schemes." Atmospheric
Environment 25B(3), 311-326.

## Validation Status

The TsWaveDelay model parameters (decay constant 33.27) require validation against:

- [ ] In-situ surface temperature measurements
- [ ] Comparison with force-restore energy balance models
- [ ] Sensitivity analysis for different surface types

The current parameterization is empirical and may need adjustment for specific climates or surface materials.

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**Ground Temperature Model:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
- Lindberg F, Onomura S, Grimmond CSB (2016) "Influence of ground surface characteristics on the mean radiant temperature in urban areas." International Journal of Biometeorology 60(9), 1439-1452.
- Offerle B, Grimmond CSB, Oke TR (2003) "Parameterization of net all-wave radiation for urban areas." Journal of Applied Meteorology 42(8), 1157-1173.
