# Golden Tests Report

This directory contains golden regression tests for SOLWEIG's Rust implementations.
These tests compare the Rust algorithms against pre-computed reference fixtures to
ensure consistency and catch regressions.

## Test Summary

**Total Golden Tests: 146** (see [REPORT.md](REPORT.md) for quantitative comparison results)

| Module | Test File | Tests | Description |
|--------|-----------|------:|-------------|
| Anisotropic Sky | `test_golden_anisotropic_sky.py` | 16 | Direction-dependent sky radiation model |
| Ground Scheme (2026a) | `test_golden_ground_scheme.py` | 20 | Opt-in force-restore/OHM + outgoing-longwave pipeline path |
| Ground Temp | `test_golden_ground_temp.py` | 6 | TsWaveDelay thermal inertia model |
| GVF | `test_golden_gvf.py` | 13 | Ground View Factor calculations |
| PET | `test_golden_pet.py` | 10 | Physiological Equivalent Temperature |
| Radiation | `test_golden_radiation.py` | 14 | Kside/Lside (shortwave/longwave) via vegetation module |
| Shadows | `test_golden_shadows.py` | 8 | Building and vegetation shadow calculations |
| SVF | `test_golden_svf.py` | 8 | Sky View Factor (total, directional, vegetation) |
| Tmrt | `test_golden_tmrt.py` | 6 | Mean Radiant Temperature |
| UTCI | `test_golden_utci.py` | 12 | Universal Thermal Climate Index |
| Wall Geometry | `test_golden_wall_geometry.py` | 30 | Wall height/aspect extraction, rotation, dilation |
| Wall Temp | `test_golden_walls.py` | 7 | Ground and wall temperature deviations |

## Covered Rust Modules

### Exposed via PyO3 (with Golden Tests)

| Rust Module | Python Import | Golden Test |
|-------------|---------------|-------------|
| `shadowing.rs` | `solweig.rustalgos.shadowing` | `test_golden_shadows.py` |
| `skyview.rs` | `solweig.rustalgos.skyview` | `test_golden_svf.py` |
| `vegetation.rs` | `solweig.rustalgos.vegetation` | `test_golden_radiation.py` |
| `gvf.rs` | `solweig.rustalgos.gvf` | `test_golden_gvf.py` |
| `utci.rs` | `solweig.rustalgos.utci` | `test_golden_utci.py` |
| `pet.rs` | `solweig.rustalgos.pet` | `test_golden_pet.py` |
| `tmrt.rs` | `solweig.rustalgos.tmrt` | `test_golden_tmrt.py` |
| `ground.rs` | `solweig.rustalgos.ground` | `test_golden_walls.py`, `test_golden_ground_temp.py` |
| `sky.rs` | `solweig.rustalgos.sky` | `test_golden_anisotropic_sky.py` |

### Internal Modules (Covered by Higher-Level Tests)

These modules are internal implementation details not exposed to Python.
They are tested indirectly through the higher-level functions that use them.

| Internal Module | Used By | Coverage |
|----------------|---------|----------|
| `sun.rs` | `sky.rs`, `vegetation.rs` | Covered by anisotropic_sky, radiation tests |
| `patch_radiation.rs` | `sky.rs` | Covered by anisotropic_sky tests |
| `sunlit_shaded_patches.rs` | `sky.rs`, `vegetation.rs` | Covered by anisotropic_sky, radiation tests |
| `emissivity_models.rs` | `sky.rs` | Covered by anisotropic_sky tests |

## Ground Truth Sources

Golden fixtures are generated from different sources depending on the algorithm:

| Algorithm | Ground Truth | Rationale |
|-----------|--------------|-----------|
| Shadows | UMEP Python | Reference implementation, validated against field measurements |
| SVF | UMEP Python | Reference implementation |
| Radiation | UMEP Python | Reference formulas from Lindberg et al. (2008) |
| GVF | UMEP Python | Ground view factor calculations |
| UTCI | UMEP Python | 6th-order polynomial approximation |
| PET | UMEP Python | Iterative energy balance solver |
| Tmrt | Formula-based | Stefan-Boltzmann formula: `(Sstr / (abs_l * SBC))^0.25 - 273.15` |
| Ground Temp | UMEP Python | TsWaveDelay exponential decay model |
| Wall Temp | Rust (regression) | Numerical consistency checks |
| Anisotropic Sky | Rust (regression) | Complex model with numerical consistency checks |
| Ground Scheme (2026a) | Rust (regression) | Components parity-gated vs vendored upstream in `tests/spec/test_parity_2026a.py`; golden pins the fused composition |

## Running Tests

```bash
# Run all golden tests
uv run pytest tests/golden/ -v

# Run specific module tests
uv run pytest tests/golden/test_golden_shadows.py -v

# Run with coverage
uv run pytest tests/golden/ --cov=solweig.rustalgos
```

## Regenerating Fixtures

Fixtures are generated from the reference implementations. To regenerate:

```bash
uv run python tests/golden/generate_fixtures.py
```

**Warning:** Regenerating fixtures will overwrite existing reference data.
Only do this when intentionally updating the ground truth.

## Test Categories

Each golden test file contains multiple test categories:

1. **Golden Regression Tests**: Compare against pre-computed fixtures
2. **Physical Property Tests**: Verify outputs satisfy physical constraints
3. **Shape Consistency Tests**: Verify output dimensions match inputs
4. **Behavioral Tests**: Verify correct response to input changes

## Tolerance Settings

| Algorithm | RTOL | ATOL | Notes |
|-----------|------|------|-------|
| Shadows | 1e-6 | 1e-6 | Binary masks, high precision |
| SVF | 0.01 | 0.02 | 2% tolerance for complex geometry |
| Radiation | 1e-4 | 1e-4 | Physical radiation values |
| UTCI | 1e-3 | 0.05 | 0.05°C absolute tolerance |
| PET | 0.01 | 0.1 | Iterative solver, 0.1°C tolerance |
| Tmrt | 1e-4 | 0.01 | 0.01°C absolute tolerance |
| Anisotropic Sky | 1e-4 | 0.1 | Complex radiation model |

## Adding New Tests

1. Add fixture generator in `generate_fixtures.py`
2. Create test file `test_golden_<module>.py`
3. Include physical property tests (not just regression)
4. Document ground truth source
5. Update this README

## References

- Lindberg et al. (2008): SOLWEIG 1.0 radiation model
- Lindberg et al. (2016): SOLWEIG 2016a updates
- Perez et al. (1993): Anisotropic sky luminance distribution
- Jendritzky et al. (2012): UTCI formulation
- Höppe (1999): PET energy balance model
